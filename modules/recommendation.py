"""
Motor del recomendador de ejercicios (basado en reglas explícitas).

Qué hace
    1. Valida los datos del usuario (sin calcular nada con datos faltantes o imposibles).
    2. Evalúa precauciones (limitaciones declaradas).
    3. Puntúa cada tipo de entrenamiento con las reglas de `recommendation_rules.py`.
    4. Explica cada recomendación indicando el ORIGEN de cada motivo (criterio de diseño o datos).
    5. Estima las calorías con `calorie_prediction.py`.

Qué NO hace
    * No entrena un clasificador para «predecir el mejor ejercicio»: el conjunto de datos registra lo
      que la gente hizo, no lo que le convenía.
    * No aplica límites de edad o IMC para prohibir actividades.
    * No emite diagnósticos ni valoraciones médicas.

Este módulo no depende de Streamlit.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from . import recommendation_rules as reglas_por_defecto
from .calorie_prediction import estimar_calorias
from .data_processing import calcular_imc, clasificar_imc
from .formato import numero, porcentaje

ORIGEN_DISENO = "Criterio de diseño"
ORIGEN_DATOS = "Datos"


# --------------------------------------------------------------------------- #
# Estructuras de datos
# --------------------------------------------------------------------------- #
@dataclass
class DatosUsuario:
    """Entradas del formulario. None significa «no indicado»."""

    edad: float = None
    genero: str = None               # "Femenino", "Masculino" o None (prefiere no indicarlo)
    peso_kg: float = None
    estatura_m: float = None
    nivel: str = None                # "Principiante" | "Intermedio" | "Avanzado"
    objetivo: str = None             # clave de reglas.OBJETIVOS
    duracion_min: float = None
    limitacion: str = "no"           # clave de reglas.OPCIONES_LIMITACION
    restriccion_profesional: str = "no"  # clave de reglas.OPCIONES_RESTRICCION


@dataclass
class ValidacionUsuario:
    errores: list = field(default_factory=list)
    advertencias: list = field(default_factory=list)
    faltantes_criticos: list = field(default_factory=list)
    faltantes_opcionales: dict = field(default_factory=dict)   # campo -> consecuencia de no indicarlo

    @property
    def puede_recomendar(self) -> bool:
        return not self.errores and not self.faltantes_criticos


@dataclass
class Recomendacion:
    tipo: str
    puntuacion: float
    compatibilidad: str                  # "Alta" | "Media" | "Baja"
    componentes: list                    # detalle de la puntuación por componente
    motivos: list                        # [{"texto": ..., "origen": ...}]
    nivel_inicial: str
    duracion_min: float
    intensidad: str
    calorias: object = None              # ResultadoCalorias
    avisos: list = field(default_factory=list)


@dataclass
class ResultadoRecomendacion:
    estado: str                          # "ok" | "faltan_datos" | "invalido"
    validacion: ValidacionUsuario = None
    recomendaciones: list = field(default_factory=list)
    no_mostradas: list = field(default_factory=list)    # [(tipo, motivo)]
    imc: float = None
    categoria_imc: str = None
    precaucion: dict = field(default_factory=dict)
    perfil_similar: dict = field(default_factory=dict)
    advertencias: list = field(default_factory=list)
    mensaje: str = ""


# --------------------------------------------------------------------------- #
# Entradas: estatura y validación
# --------------------------------------------------------------------------- #
_PATRON_ESTATURA = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(cm|centimetros?|centímetros?|m|metros?)?\s*$", re.IGNORECASE)


def parsear_estatura(texto):
    """
    Interpreta una estatura escrita como «172», «172 cm», «1,72» o «1.72 m».
    Devuelve (metros, mensaje_de_error). Un texto vacío no es un error: la estatura simplemente falta.
    Sin unidad: valores menores que 3 se toman como metros y los demás como centímetros.
    """
    if texto is None or str(texto).strip() == "":
        return None, None
    coincidencia = _PATRON_ESTATURA.match(str(texto))
    if not coincidencia:
        return None, f"No se entiende la estatura «{str(texto).strip()[:20]}». Escríbela como 172 cm o 1,72 m."
    valor = float(coincidencia.group(1).replace(",", "."))
    unidad = (coincidencia.group(2) or "").lower()
    if unidad.startswith("c"):
        metros = valor / 100
    elif unidad.startswith("m"):
        metros = valor
    else:
        metros = valor / 100 if valor >= 3 else valor
    return metros, None


def _es_numero(valor) -> bool:
    return isinstance(valor, (int, float, np.integer, np.floating)) and not isinstance(valor, bool) and math.isfinite(valor)


def validar_datos_usuario(datos: DatosUsuario, reglas=reglas_por_defecto) -> ValidacionUsuario:
    """
    Distingue entre:
      * ERRORES: valores imposibles o inválidos (se corrigen antes de calcular nada);
      * FALTANTES CRÍTICOS: nivel, objetivo y duración (sin ellos no hay recomendación);
      * FALTANTES OPCIONALES: edad, género, peso y estatura (se recomienda igual, pero con menos personalización).
    """
    v = ValidacionUsuario()
    lim = reglas.LIMITES_USUARIO

    # --- Campos numéricos -------------------------------------------------- #
    campos = [
        ("edad", datos.edad, "La edad", "años", False,
         "no se comparan perfiles similares por edad ni se ajusta la estimación de calorías"),
        ("peso_kg", datos.peso_kg, "El peso", "kg", False,
         "no se calcula el IMC ni el ajuste de calorías por peso"),
        ("estatura_m", datos.estatura_m, "La estatura", "m", False, "no se calcula el IMC"),
        ("duracion_min", datos.duracion_min, "La duración", "min", True, ""),
    ]
    for clave, valor, nombre, unidad, critico, consecuencia in campos:
        if valor is None:
            if critico:
                v.faltantes_criticos.append("duración de la sesión")
            else:
                v.faltantes_opcionales[clave] = consecuencia
            continue
        if not _es_numero(valor):
            v.errores.append(f"{nombre} debe ser un número válido.")
            continue
        minimo, maximo = lim[clave]
        if not (minimo <= valor <= maximo):
            if clave == "edad" and valor < minimo:
                v.errores.append("Esta herramienta está pensada para personas adultas (18 años o más); "
                                 "el conjunto de datos tampoco incluye menores de edad.")
            elif clave == "estatura_m":
                v.errores.append(f"La estatura debe estar entre {numero(minimo, 2)} y {numero(maximo, 2)} m "
                                 f"({numero(minimo * 100, 0)} y {numero(maximo * 100, 0)} cm). Revisa el valor y la unidad.")
            else:
                v.errores.append(f"{nombre} debe estar entre {numero(minimo, 0)} y {numero(maximo, 0)} {unidad}.")

    # --- Coherencia peso/estatura (IMC plausible) --------------------------- #
    if not v.errores and datos.peso_kg is not None and datos.estatura_m is not None:
        imc = calcular_imc(datos.peso_kg, datos.estatura_m)
        if imc is None or not (lim["imc"][0] <= imc <= lim["imc"][1]):
            v.errores.append("La combinación de peso y estatura produce un IMC que no es plausible. Revisa ambos valores.")

    # --- Categorías --------------------------------------------------------- #
    if datos.genero is not None and datos.genero not in cfg.GENEROS:
        v.errores.append("El género indicado no es una opción válida.")
    if datos.genero is None:
        v.faltantes_opcionales["genero"] = "no se aplica el ajuste de calorías por género"
    if datos.nivel is None:
        v.faltantes_criticos.append("nivel de experiencia")
    elif datos.nivel not in cfg.ORDEN_NIVELES:
        v.errores.append("El nivel de experiencia indicado no es una opción válida.")
    if datos.objetivo is None:
        v.faltantes_criticos.append("objetivo principal")
    elif datos.objetivo not in reglas.OBJETIVOS:
        v.errores.append("El objetivo indicado no es una opción válida.")
    if datos.limitacion not in reglas.OPCIONES_LIMITACION:
        v.errores.append("La respuesta sobre limitaciones no es válida.")
    if datos.restriccion_profesional not in reglas.OPCIONES_RESTRICCION:
        v.errores.append("La respuesta sobre indicaciones profesionales no es válida.")
    return v


# --------------------------------------------------------------------------- #
# Precauciones
# --------------------------------------------------------------------------- #
def evaluar_precauciones(datos: DatosUsuario) -> dict:
    """
    Nivel de precaución a partir de las respuestas del usuario:
      «alta»  → limitación importante o restricciones indicadas por un profesional;
      «leve»  → limitación leve;
      «ninguna» en otro caso. `sin_respuesta` indica que prefirió no responder alguna pregunta.
    """
    motivos = []
    if datos.limitacion == "importante":
        motivos.append("declaró una limitación importante de movilidad, lesión o dolor")
    if datos.restriccion_profesional == "si":
        motivos.append("indicó que un profesional le señaló precauciones o restricciones")
    if motivos:
        nivel = "alta"
    elif datos.limitacion == "leve":
        nivel, motivos = "leve", ["declaró una limitación leve"]
    else:
        nivel = "ninguna"
    sin_respuesta = "prefiero_no_decir" in (datos.limitacion, datos.restriccion_profesional)
    return {"nivel": nivel, "motivos": motivos, "sin_respuesta": sin_respuesta}


# --------------------------------------------------------------------------- #
# Patrones descriptivos del conjunto de datos
# --------------------------------------------------------------------------- #
def frecuencias_por_nivel(df: pd.DataFrame) -> dict:
    """Rango de frecuencia semanal observado en el CSV para cada nivel (para contrastar con el formulario)."""
    resultado = {}
    for nivel in cfg.ORDEN_NIVELES:
        serie = df.loc[df["nivel_experiencia"].astype(str) == nivel, "frecuencia_semanal"]
        if len(serie):
            resultado[nivel] = (int(serie.min()), int(serie.max()))
    return resultado


def perfil_similar(df: pd.DataFrame, datos: DatosUsuario, reglas=reglas_por_defecto) -> dict:
    """
    Registros «similares» (mismo nivel y, si la edad está dentro del rango observado, edad ± ventana) y
    proporción de cada tipo de entrenamiento entre ellos. Para cada tipo se contrasta (binomial exacta,
    con corrección de Bonferroni por el número de tipos) si su proporción difiere de la del conjunto completo.
    """
    p = reglas.PARAMETROS_DATOS
    mascara = df["nivel_experiencia"].astype(str) == str(datos.nivel)
    criterio = [f"nivel {datos.nivel}"]
    edad_min, edad_max = int(df["edad"].min()), int(df["edad"].max())
    if _es_numero(datos.edad) and edad_min <= datos.edad <= edad_max:
        ventana = p["ventana_edad_anios"]
        inferior, superior = max(edad_min, int(datos.edad - ventana)), min(edad_max, int(datos.edad + ventana))
        mascara &= df["edad"].between(inferior, superior)
        criterio.append(f"edad entre {inferior} y {superior} años")
    similares = df.loc[mascara]
    n = len(similares)
    resultado = {"n": n, "criterio": ", ".join(criterio), "suficiente": n >= p["min_registros_perfil"], "por_tipo": {}}
    if not resultado["suficiente"]:
        return resultado
    total = df["tipo_entrenamiento"].astype(str).value_counts(normalize=True)
    conteo = similares["tipo_entrenamiento"].astype(str).value_counts()
    alfa_ajustado = p["alfa"] / len(reglas.TIPOS)
    for tipo in reglas.TIPOS:
        k = int(conteo.get(tipo, 0))
        proporcion_total = float(total.get(tipo, 0.0))
        p_valor = float(stats.binomtest(k, n, proporcion_total).pvalue) if 0 < proporcion_total < 1 else 1.0
        resultado["por_tipo"][tipo] = {
            "k": k, "proporcion": k / n, "proporcion_total": proporcion_total,
            "p": p_valor, "significativo": p_valor < alfa_ajustado,
        }
    return resultado


def _componente_duracion(df, datos, tipo, reglas):
    """Puntos y texto del componente DURACIÓN (descriptivo, basado en el conjunto de datos)."""
    p, pts = reglas.PARAMETROS_DATOS, reglas.PUNTOS_DURACION
    duracion = float(datos.duracion_min)
    minimo_global, maximo_global = float(df["duracion_min"].min()), float(df["duracion_min"].max())
    if not (minimo_global <= duracion <= maximo_global):
        return None, (f"Tu duración ({numero(duracion, 0)} min) está fuera del rango de sesiones registradas "
                      f"({numero(minimo_global, 0)}–{numero(maximo_global, 0)} min), así que no se compara con los datos.")
    nivel = df["nivel_experiencia"].astype(str) == str(datos.nivel)
    celda = df.loc[nivel & (df["tipo_entrenamiento"].astype(str) == tipo)]
    alcance = f"nivel {datos.nivel} y tipo {tipo}"
    if len(celda) < p["min_registros_celda"]:
        celda, alcance = df.loc[nivel], f"nivel {datos.nivel}"
    if len(celda) < p["min_registros_celda"]:
        return None, "No hay suficientes registros comparables para valorar la duración."
    d = celda["duracion_min"]
    p10, p25, p75, p90 = d.quantile([0.10, 0.25, 0.75, 0.90])
    if p25 <= duracion <= p75:
        return pts["central"], (f"Tu duración ({numero(duracion, 0)} min) está dentro del 50 % central de las sesiones registradas "
                                f"para {alcance} (entre {numero(p25, 0)} y {numero(p75, 0)} min).")
    if p10 <= duracion <= p90:
        return pts["habitual"], (f"Tu duración ({numero(duracion, 0)} min) está dentro del rango habitual (80 % central) de las sesiones "
                                 f"registradas para {alcance} ({numero(p10, 0)}–{numero(p90, 0)} min).")
    if d.min() <= duracion <= d.max():
        return pts["observada"], (f"Tu duración ({numero(duracion, 0)} min) está en los extremos de lo registrado para {alcance} "
                                  f"({numero(d.min(), 0)}–{numero(d.max(), 0)} min).")
    lado = "más corta" if duracion < d.min() else "más larga"
    return pts["inusual"], (f"Tu duración ({numero(duracion, 0)} min) es {lado} que cualquier sesión registrada para {alcance} "
                            f"({numero(d.min(), 0)}–{numero(d.max(), 0)} min). Eso no significa que sea inadecuada, solo que "
                            "no hay datos comparables.")


def _componente_datos(perfil: dict, tipo: str):
    """Puntos (solo si hay diferencia estadísticamente detectable) y texto del componente DATOS."""
    if not perfil.get("suficiente"):
        return None, (f"No hay suficientes registros similares ({perfil.get('n', 0)}) para describir patrones en el conjunto "
                      "de datos; este criterio no se utilizó.")
    info = perfil["por_tipo"][tipo]
    texto = (f"Entre {perfil['n']} registros de perfil similar ({perfil['criterio']}), {porcentaje(info['proporcion'])} "
             f"realizó {tipo} (en todo el conjunto: {porcentaje(info['proporcion_total'])}). ")
    if info["significativo"]:
        puntos = float(np.clip(50 * info["proporcion"] / info["proporcion_total"], 0, 100))
        texto += ("La diferencia es estadísticamente detectable (con corrección por comparaciones múltiples) y se incorporó a la "
                  "puntuación como dato descriptivo: no implica que sea mejor ni más seguro.")
        return puntos, texto
    texto += "La diferencia no es estadísticamente clara, por lo que no influyó en la puntuación."
    return None, texto


def _nivel_compatibilidad(puntuacion: float, reglas) -> str:
    if puntuacion >= reglas.UMBRAL_COMPATIBILIDAD_ALTA:
        return "Alta"
    if puntuacion >= reglas.UMBRAL_COMPATIBILIDAD_MEDIA:
        return "Media"
    return "Baja"


# --------------------------------------------------------------------------- #
# Puntuación y recomendación
# --------------------------------------------------------------------------- #
def puntuar_tipo(tipo, datos, df, perfil, precaucion, reglas=reglas_por_defecto) -> Recomendacion:
    """Calcula la puntuación de un tipo de entrenamiento y construye su explicación."""
    afinidad, motivo_objetivo = reglas.AFINIDAD_OBJETIVO[datos.objetivo][tipo]
    ajuste, motivo_exp, nivel_inicial = reglas.AJUSTE_EXPERIENCIA[datos.nivel][tipo]
    puntos_dur, texto_dur = _componente_duracion(df, datos, tipo, reglas)
    puntos_datos, texto_datos = _componente_datos(perfil, tipo)

    pesos = reglas.PESOS_COMPONENTES
    componentes = [
        {"nombre": "Objetivo", "puntos": afinidad, "peso": pesos["objetivo"], "origen": ORIGEN_DISENO},
        {"nombre": "Experiencia", "puntos": ajuste, "peso": pesos["experiencia"], "origen": ORIGEN_DISENO},
        {"nombre": "Duración", "puntos": puntos_dur, "peso": pesos["duracion"], "origen": ORIGEN_DATOS},
        {"nombre": "Patrones en los datos", "puntos": puntos_datos, "peso": pesos["datos"], "origen": ORIGEN_DATOS},
    ]
    activos = [c for c in componentes if c["puntos"] is not None]
    suma_pesos = sum(c["peso"] for c in activos)
    puntuacion = sum(c["peso"] * c["puntos"] for c in activos) / suma_pesos

    motivos = [
        {"texto": f"Objetivo «{reglas.OBJETIVOS[datos.objetivo]}»: {motivo_objetivo}", "origen": ORIGEN_DISENO},
        {"texto": f"Nivel {datos.nivel.lower()}: {motivo_exp}", "origen": ORIGEN_DISENO},
        {"texto": texto_dur, "origen": ORIGEN_DATOS},
        {"texto": texto_datos, "origen": ORIGEN_DATOS},
    ]
    avisos = []
    if precaucion["nivel"] == "leve":
        resta = reglas.PENALIZACION_LIMITACION_LEVE.get(tipo, 0)
        if resta:
            puntuacion -= resta
            motivos.append({"texto": f"Por la limitación leve declarada se restaron {resta} puntos a esta actividad por su intensidad "
                                     "(no se descarta).", "origen": ORIGEN_DISENO})
    if precaucion["nivel"] == "alta":
        nivel_inicial = reglas.NIVEL_INICIAL_PRECAUCION
        avisos.append("Modo adaptado por precauciones declaradas: consulta con un profesional antes de empezar.")

    puntuacion = float(np.clip(puntuacion, 0, 100))
    return Recomendacion(
        tipo=tipo, puntuacion=puntuacion, compatibilidad=_nivel_compatibilidad(puntuacion, reglas),
        componentes=componentes, motivos=motivos, nivel_inicial=nivel_inicial,
        duracion_min=float(datos.duracion_min), intensidad=reglas.INTENSIDAD_TIPO[tipo], avisos=avisos,
    )


def recomendar(datos: DatosUsuario, df: pd.DataFrame, estimador=None, reglas=reglas_por_defecto) -> ResultadoRecomendacion:
    """
    Punto de entrada del recomendador. Devuelve siempre un ResultadoRecomendacion (nunca lanza una excepción
    por entradas inválidas): `estado` indica si hay recomendaciones, faltan datos o hay entradas inválidas.
    """
    validacion = validar_datos_usuario(datos, reglas)
    resultado = ResultadoRecomendacion(estado="ok", validacion=validacion)

    if validacion.errores:
        resultado.estado = "invalido"
        resultado.mensaje = "Corrige los datos marcados antes de continuar."
        return resultado
    if validacion.faltantes_criticos:
        resultado.estado = "faltan_datos"
        resultado.mensaje = ("Para personalizar la recomendación faltan: " + ", ".join(validacion.faltantes_criticos) + ".")
        return resultado

    # IMC descriptivo (solo con peso y estatura válidos)
    imc = calcular_imc(datos.peso_kg, datos.estatura_m) if datos.peso_kg is not None and datos.estatura_m is not None else None
    resultado.imc, resultado.categoria_imc = imc, clasificar_imc(imc)

    resultado.precaucion = evaluar_precauciones(datos)
    resultado.perfil_similar = perfil_similar(df, datos, reglas)

    # Advertencias sobre el alcance de los datos
    edad_min, edad_max = int(df["edad"].min()), int(df["edad"].max())
    if _es_numero(datos.edad) and not (edad_min <= datos.edad <= edad_max):
        resultado.advertencias.append(
            f"Tu edad ({numero(datos.edad, 0)} años) está fuera del rango de edades del conjunto de datos ({edad_min}–{edad_max}). "
            "Se omite la comparación por edad con perfiles similares y el ajuste de calorías por edad; la recomendación se basa solo "
            "en las reglas de objetivo, experiencia y duración.")

    # Candidatos: se excluyen (no se recomiendan automáticamente) los tipos intensos si hay precaución alta
    candidatos = list(reglas.TIPOS)
    if resultado.precaucion["nivel"] == "alta":
        for tipo in reglas.TIPOS_NO_AUTOMATICOS_CON_PRECAUCION:
            if tipo in candidatos:
                candidatos.remove(tipo)
                resultado.no_mostradas.append((tipo, "No se recomienda de forma automática por su intensidad dadas las precauciones declaradas."))

    puntuadas = [puntuar_tipo(t, datos, df, resultado.perfil_similar, resultado.precaucion, reglas) for t in candidatos]
    orden_tipos = {t: i for i, t in enumerate(reglas.TIPOS)}
    puntuadas.sort(key=lambda r: (-r.puntuacion, orden_tipos[r.tipo]))

    mostrar = []
    for r in puntuadas:
        if r.puntuacion < reglas.PUNTUACION_MINIMA_MOSTRAR:
            resultado.no_mostradas.append((r.tipo, f"Compatibilidad baja con los criterios indicados ({numero(r.puntuacion, 0)}/100)."))
        elif len(mostrar) >= reglas.NUMERO_RECOMENDACIONES:
            resultado.no_mostradas.append((r.tipo, f"Compatible ({numero(r.puntuacion, 0)}/100), pero quedó fuera de las "
                                                   f"{reglas.NUMERO_RECOMENDACIONES} mejor puntuadas."))
        else:
            mostrar.append(r)

    if estimador is not None:
        for r in mostrar:
            r.calorias = estimar_calorias(estimador, r.tipo, datos.duracion_min, datos.edad, datos.genero, datos.peso_kg)

    resultado.recomendaciones = mostrar
    if len(mostrar) < reglas.NUMERO_RECOMENDACIONES:
        resultado.advertencias.append(
            f"Solo se muestran {len(mostrar)} actividad(es) porque no hay suficientes opciones con compatibilidad adecuada "
            "para los criterios indicados.")
    return resultado
