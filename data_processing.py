"""
Carga, validación y limpieza del conjunto de datos de miembros del gimnasio.

Principios que sigue este módulo:
  * No se asume que los datos estén limpios: cada comprobación se ejecuta y se informa.
  * Se distingue entre valores INVÁLIDOS (imposibles; se excluyen o se marcan como
    faltantes) y valores EXTREMOS (posibles; se conservan y solo se informan).
  * Toda transformación queda registrada para mostrarla al usuario.

Este módulo no depende de Streamlit para poder probarse de forma independiente.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .formato import numero


class ErrorDatos(Exception):
    """Error comprensible para el usuario (archivo ausente, columnas faltantes, etc.)."""


@dataclass
class ResultadoProcesamiento:
    """Resultado de cargar y limpiar los datos."""

    datos: pd.DataFrame
    informe: pd.DataFrame                 # tabla de comprobaciones de calidad
    transformaciones: list = field(default_factory=list)
    n_original: int = 0
    n_final: int = 0
    columnas_ignoradas: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Utilidades de IMC (también las usa el recomendador)
# --------------------------------------------------------------------------- #
def calcular_imc(peso_kg, estatura_m):
    """IMC = peso (kg) / estatura (m)². Devuelve None si los datos no son válidos."""
    try:
        peso = float(peso_kg)
        estatura = float(estatura_m)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(peso) and np.isfinite(estatura)) or peso <= 0 or estatura <= 0:
        return None
    return peso / (estatura ** 2)


def clasificar_imc(imc):
    """Categoría convencional (orientativa) para adultos. Devuelve None si no hay IMC."""
    if imc is None or not np.isfinite(imc):
        return None
    indice = int(np.digitize([imc], cfg.LIMITES_IMC, right=False)[0])
    return cfg.CATEGORIAS_IMC[indice]


def categorizar_imc_serie(imc: pd.Series) -> pd.Categorical:
    """Aplica las categorías de IMC a una serie completa (conserva faltantes)."""
    indices = np.digitize(imc.to_numpy(dtype=float), cfg.LIMITES_IMC, right=False)
    etiquetas = np.array(cfg.CATEGORIAS_IMC, dtype=object)[indices]
    etiquetas = np.where(imc.isna().to_numpy(), None, etiquetas)
    return pd.Categorical(etiquetas, categories=cfg.CATEGORIAS_IMC, ordered=True)


# --------------------------------------------------------------------------- #
# Localización y lectura del archivo
# --------------------------------------------------------------------------- #
def localizar_csv(carpeta=None) -> Path:
    """Busca el CSV en la carpeta del proyecto. Lanza ErrorDatos con instrucciones si no está."""
    carpeta = Path(carpeta) if carpeta else cfg.RUTA_PROYECTO
    for nombre in cfg.NOMBRES_CSV_CANDIDATOS:
        ruta = carpeta / nombre
        if ruta.exists():
            return ruta
    otros = sorted(carpeta.glob("*.csv"))
    if len(otros) == 1:  # un único CSV en la carpeta: se usa aunque tenga otro nombre
        return otros[0]
    esperados = " o ".join(f"«{n}»" for n in cfg.NOMBRES_CSV_CANDIDATOS)
    raise ErrorDatos(
        f"No se encontró el archivo de datos en la carpeta {carpeta}. "
        f"Copia el CSV en esa carpeta (junto a app.py) y asegúrate de que se llame {esperados}."
    )


def leer_csv(ruta) -> pd.DataFrame:
    """Lee el CSV probando codificaciones habituales y, si hace falta, el separador «;»."""
    ruta = Path(ruta)
    ultimo_error = None
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(ruta, encoding=codificacion)
            if df.shape[1] == 1:  # posible archivo separado por punto y coma
                df = pd.read_csv(ruta, encoding=codificacion, sep=";")
            return df
        except UnicodeDecodeError as error:
            ultimo_error = error
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise ErrorDatos(f"No se pudo interpretar el CSV ({error}). Revisa que el archivo no esté dañado.")
    raise ErrorDatos(f"No se pudo leer el archivo por un problema de codificación: {ultimo_error}")


# --------------------------------------------------------------------------- #
# Limpieza y validación
# --------------------------------------------------------------------------- #
def _normalizar_categoria(serie: pd.Series, mapa: dict) -> pd.Series:
    """Texto -> minúsculas sin espacios -> etiqueta en español (NaN si no está en el mapa)."""
    texto = serie.astype("string").str.strip().str.lower()
    return texto.map(mapa).astype("object")


def procesar_datos(df_crudo: pd.DataFrame) -> ResultadoProcesamiento:
    """
    Valida y limpia el DataFrame original. Devuelve los datos listos para analizar
    junto con un informe de calidad y la lista de transformaciones aplicadas.
    """
    filas_informe = []
    transformaciones = []

    def anotar(comprobacion, resultado, afectados, accion):
        filas_informe.append({
            "Comprobación": comprobacion,
            "Resultado": resultado,
            "Registros afectados": int(afectados),
            "Acción aplicada": accion,
        })

    df_crudo = df_crudo.copy()
    df_crudo.columns = [str(c).strip() for c in df_crudo.columns]

    # 1) Estructura: columnas esperadas
    faltantes = [c for c in cfg.COLUMNAS_ORIGINALES if c not in df_crudo.columns]
    if faltantes:
        raise ErrorDatos(
            "El CSV no contiene todas las columnas esperadas. Faltan: " + ", ".join(faltantes)
            + ". Columnas encontradas: " + ", ".join(df_crudo.columns) + "."
        )
    ignoradas = [c for c in df_crudo.columns if c not in cfg.COLUMNAS_ORIGINALES]
    df = df_crudo[list(cfg.COLUMNAS_ORIGINALES)].rename(columns=cfg.COLUMNAS_ORIGINALES).copy()
    n_original = len(df)
    if n_original == 0:
        raise ErrorDatos("El CSV no contiene registros.")
    anotar("Estructura del archivo", f"{n_original} registros y {len(cfg.COLUMNAS_ORIGINALES)} columnas esperadas encontradas",
           0, "Sin cambios")
    transformaciones.append("Se renombraron las columnas del CSV (inglés) a nombres internos en español.")

    # 2) Tipos de datos: las columnas numéricas deben ser numéricas
    celdas_no_numericas = 0
    for col in cfg.COLUMNAS_NUMERICAS:
        nulos_antes = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        celdas_no_numericas += int(df[col].isna().sum()) - nulos_antes
    anotar("Tipos de datos (columnas numéricas)",
           "Todas las columnas numéricas contienen números" if celdas_no_numericas == 0
           else f"{celdas_no_numericas} celdas no numéricas",
           celdas_no_numericas,
           "Sin cambios" if celdas_no_numericas == 0 else "Convertidas a valor faltante")

    # 3) Valores faltantes
    nulos_por_columna = df.isna().sum()
    total_nulos = int(nulos_por_columna.sum())
    detalle = ", ".join(f"{c}: {n}" for c, n in nulos_por_columna.items() if n > 0)
    anotar("Valores faltantes", "No hay valores faltantes" if total_nulos == 0 else f"{total_nulos} celdas vacías ({detalle})",
           int((df.isna().any(axis=1)).sum()),
           "Sin cambios" if total_nulos == 0 else "Se excluyen los registros con faltantes en columnas esenciales")

    # 4) Categorías esperadas
    genero_original = df["genero"].copy()
    tipo_original = df["tipo_entrenamiento"].copy()
    df["genero"] = _normalizar_categoria(genero_original, cfg.MAPA_GENERO)
    df["tipo_entrenamiento"] = _normalizar_categoria(tipo_original, cfg.MAPA_TIPO)
    cat_inesperada = (
        (genero_original.notna() & df["genero"].isna())
        | (tipo_original.notna() & df["tipo_entrenamiento"].isna())
        | (df["nivel_experiencia_cod"].notna() & ~df["nivel_experiencia_cod"].isin(list(cfg.NIVELES)))
    )
    n_cat = int(cat_inesperada.sum())
    if n_cat:
        ejemplos = sorted(set(genero_original[cat_inesperada].astype(str)) | set(tipo_original[cat_inesperada].astype(str)))[:5]
        resultado_cat = f"{n_cat} registros con categorías inesperadas (p. ej.: {', '.join(ejemplos)})"
    else:
        resultado_cat = "Género, tipo de entrenamiento y nivel de experiencia solo contienen categorías esperadas"
    anotar("Categorías inesperadas", resultado_cat, n_cat,
           "Sin cambios" if n_cat == 0 else "Registros excluidos")
    transformaciones.append("Se tradujeron las categorías: Male→Masculino, Female→Femenino, Strength→Fuerza "
                            "(Cardio, HIIT y Yoga se mantienen).")

    # 5) Duplicados exactos (con 15 variables, un duplicado exacto es muy improbable como dato real)
    duplicados = df.duplicated(keep="first")
    n_dup = int(duplicados.sum())
    anotar("Registros duplicados (filas idénticas)",
           "No hay duplicados" if n_dup == 0 else f"{n_dup} filas duplicadas",
           n_dup, "Sin cambios" if n_dup == 0 else "Se conserva solo la primera aparición")

    # 6) Valores imposibles según rangos plausibles
    invalido_nucleo = pd.Series(False, index=df.index)
    for col, (minimo, maximo) in cfg.RANGOS_VALIDOS.items():
        invalido = df[col].notna() & ((df[col] < minimo) | (df[col] > maximo))
        n_inv = int(invalido.sum())
        etiqueta = cfg.ETIQUETAS.get(col, col)
        if col in cfg.COLUMNAS_NUCLEO_NUMERICAS:
            invalido_nucleo |= invalido
            accion = "Sin cambios" if n_inv == 0 else "Registro excluido"
        else:
            if n_inv:
                df.loc[invalido, col] = np.nan
            accion = "Sin cambios" if n_inv == 0 else "Valor marcado como faltante"
        anotar(f"Rango válido: {etiqueta}",
               f"Todos los valores están entre {numero(minimo, 1 if minimo % 1 else 0)} y {numero(maximo, 1 if maximo % 1 else 0)}" if n_inv == 0
               else f"{n_inv} valores fuera de [{numero(minimo, 1 if minimo % 1 else 0)}, {numero(maximo, 1 if maximo % 1 else 0)}]",
               n_inv, accion)

    # 7) Exclusión de registros inválidos
    faltante_nucleo = df[cfg.COLUMNAS_NUCLEO].isna().any(axis=1)
    excluir = faltante_nucleo | cat_inesperada | invalido_nucleo | duplicados
    n_excluidos = int(excluir.sum())
    df = df.loc[~excluir].reset_index(drop=True)
    if n_excluidos:
        transformaciones.append(f"Se excluyeron {n_excluidos} registros por datos esenciales faltantes, "
                                f"categorías inesperadas, valores imposibles o duplicados.")
    else:
        transformaciones.append("No fue necesario excluir ningún registro: todos superaron las validaciones.")

    # 8) Coherencia del IMC con peso y estatura (kg y m)
    imc_calculado = df["peso_kg"] / (df["estatura_m"] ** 2)
    diferencia = (df["imc"] - imc_calculado).abs()
    incoherente = df["imc"].isna() | (diferencia > cfg.TOLERANCIA_IMC)
    n_incoh = int(incoherente.sum())
    if n_incoh:
        df.loc[incoherente, "imc"] = imc_calculado[incoherente].round(2)
        transformaciones.append(f"En {n_incoh} registros el IMC no coincidía con peso/estatura² (o faltaba) y se recalculó.")
    anotar("Coherencia del IMC (peso/estatura²)",
           f"El IMC del CSV coincide con peso/estatura² (diferencia máxima {numero(float(diferencia.max()), 3)})" if n_incoh == 0
           else f"{n_incoh} registros con IMC incoherente o ausente",
           n_incoh, "Sin cambios" if n_incoh == 0 else "IMC recalculado")

    # 9) Coherencia de frecuencias cardíacas (promedio no puede superar a la máxima)
    fc_inconsistente = df["fc_prom"].notna() & df["fc_max"].notna() & (df["fc_prom"] > df["fc_max"])
    fc_reposo_incoh = df["fc_reposo"].notna() & df["fc_prom"].notna() & (df["fc_reposo"] >= df["fc_prom"])
    n_fc = int(fc_inconsistente.sum())
    n_fc_rep = int(fc_reposo_incoh.sum())
    df.loc[fc_inconsistente, ["fc_prom", "fc_max"]] = np.nan
    df.loc[fc_reposo_incoh, ["fc_reposo", "fc_prom"]] = np.nan
    anotar("Coherencia de la frecuencia cardíaca",
           "FC promedio ≤ FC máxima y FC en reposo < FC promedio en todos los registros" if n_fc + n_fc_rep == 0
           else f"{n_fc} registros con FC promedio > FC máxima; {n_fc_rep} con FC en reposo ≥ FC promedio",
           n_fc + n_fc_rep,
           "Sin cambios" if n_fc + n_fc_rep == 0
           else "Solo las FC implicadas se marcan como faltantes; el resto del registro se conserva")
    if n_fc + n_fc_rep:
        transformaciones.append(f"Se marcaron como faltantes las frecuencias cardíacas incoherentes de {n_fc + n_fc_rep} registros "
                                f"(el resto de sus datos se conserva).")

    # 10) Valores extremos que se CONSERVAN (podrían ser reales)
    imc_extremo = (df["imc"] < cfg.IMC_EXTREMO_BAJO) | (df["imc"] >= cfg.IMC_EXTREMO_ALTO)
    anotar(f"IMC extremo (< {numero(cfg.IMC_EXTREMO_BAJO, 0)} o ≥ {numero(cfg.IMC_EXTREMO_ALTO, 0)})",
           f"{int(imc_extremo.sum())} registros con IMC extremo; "
           f"{int((df['imc'] < cfg.IMC_EXTREMO_BAJO).sum())} por debajo y {int((df['imc'] >= cfg.IMC_EXTREMO_ALTO).sum())} por encima",
           int(imc_extremo.sum()), "Se conservan (posibles pero poco habituales; interpretar con cautela)")
    atipicos = {}
    for col in ["edad", "peso_kg", "estatura_m", "duracion_h", "calorias", "grasa_pct", "agua_l"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_at = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_at:
            atipicos[cfg.ETIQUETAS[col]] = n_at
    anotar("Valores atípicos (regla 1,5 × rango intercuartílico)",
           "; ".join(f"{k}: {v}" for k, v in atipicos.items()) if atipicos else "No se detectan atípicos en las variables principales",
           sum(atipicos.values()), "Se conservan: no se eliminan observaciones extremas automáticamente")

    # 11) Variables derivadas
    df["edad"] = df["edad"].astype(int)
    df["nivel_experiencia_cod"] = df["nivel_experiencia_cod"].astype(int)
    df["genero"] = pd.Categorical(df["genero"], categories=cfg.GENEROS)
    df["tipo_entrenamiento"] = pd.Categorical(df["tipo_entrenamiento"], categories=cfg.TIPOS_ENTRENAMIENTO)
    df["nivel_experiencia"] = pd.Categorical(df["nivel_experiencia_cod"].map(cfg.NIVELES),
                                             categories=cfg.ORDEN_NIVELES, ordered=True)
    df["categoria_imc"] = categorizar_imc_serie(df["imc"])
    df["duracion_min"] = df["duracion_h"] * 60.0
    df["calorias_por_hora"] = df["calorias"] / df["duracion_h"]
    transformaciones.append("Se asignó el nivel de experiencia 1/2/3 → Principiante/Intermedio/Avanzado "
                            "(SUPUESTO: el CSV no trae diccionario de datos; el orden se respalda en que la frecuencia semanal "
                            "y la duración aumentan con el nivel).")
    transformaciones.append("Se crearon: duración en minutos, calorías por hora (calorías ÷ horas) y categoría de IMC "
                            "(<18,5 / 18,5–24,9 / 25–29,9 / ≥30).")

    informe = pd.DataFrame(filas_informe)
    return ResultadoProcesamiento(
        datos=df, informe=informe, transformaciones=transformaciones,
        n_original=n_original, n_final=len(df), columnas_ignoradas=ignoradas,
    )


def cargar_datos(ruta=None) -> ResultadoProcesamiento:
    """Localiza (si no se indica), lee y procesa el CSV. Punto de entrada para la aplicación."""
    ruta = Path(ruta) if ruta else localizar_csv()
    if not ruta.exists():
        raise ErrorDatos(f"No existe el archivo {ruta}.")
    return procesar_datos(leer_csv(ruta))
