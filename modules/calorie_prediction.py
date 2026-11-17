"""
Estimación de las calorías de una sesión a partir de los registros del conjunto de datos.

MÉTODO PRINCIPAL (sencillo y transparente)
    calorías ≈ mediana de «calorías por hora» del tipo de entrenamiento × horas de sesión
    con un rango que cubre el 80 % central (percentiles 10-90) de las calorías/hora observadas.

MÉTODO COMPLEMENTARIO (opcional)
    Regresión lineal de «calorías por hora» con edad, género y peso. Solo se aplica si el
    usuario aportó esos datos y están dentro del rango observado. Su utilidad se mide con
    validación cruzada (MAE) y se muestra siempre junto al método principal.

SALVAGUARDAS
    * Solo se usan registros válidos (calorías y duración positivas, calorías/hora plausibles).
    * Antes de convertir «por hora» a otra duración se comprueba que calorías/hora no dependa de la
      duración (si no es así, la estimación no se ofrece).
    * No se extrapola fuera del rango de duraciones observado: se informa en lugar de estimar.
    * No se usan variables que un usuario nuevo no conoce (FC promedio, % de grasa, agua), ni la
      propia variable objetivo, para evitar fuga de información.
    * Las estimaciones se redondean a múltiplos de 5 kcal para no aparentar una precisión inexistente.

Este módulo no depende de Streamlit.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from . import config as cfg
from .exploratory_analysis import estilo_figura
from .formato import numero, p_valor

# Si la correlación entre calorías/hora y duración supera este valor, no se escala por duración.
LIMITE_CORRELACION_LINEALIDAD = 0.20


# --------------------------------------------------------------------------- #
# Estructuras de datos
# --------------------------------------------------------------------------- #
@dataclass
class EstimadorCalorias:
    """Estadísticos y modelo aprendidos del conjunto de datos (se entrena una sola vez)."""

    tabla_por_tipo: pd.DataFrame
    duracion_min_obs_h: float
    duracion_max_obs_h: float
    linealidad: dict
    validacion: dict
    p_valor_tipos: float
    modelo: object = None                       # LinearRegression (edad, género masculino, peso)
    rangos_modelo: dict = field(default_factory=dict)
    cuantiles_residuos: tuple = (0.0, 0.0)      # percentiles 10 y 90 de los errores en calorías/hora
    metricas_cv: dict = field(default_factory=dict)


@dataclass
class ResultadoCalorias:
    """Resultado de una estimación, listo para mostrar."""

    estado: str                                  # "ok" | "fuera_de_rango" | "no_disponible"
    tipo: str = ""
    duracion_min: float = np.nan
    base_kcal: float = np.nan
    base_rango: tuple = (np.nan, np.nan)
    refinada_kcal: float = np.nan
    refinada_rango: tuple = (np.nan, np.nan)
    refinada_aplicada: bool = False
    motivo_sin_refinada: str = ""
    n_registros_tipo: int = 0
    mensaje: str = ""
    advertencias: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Análisis previo (sección 6.1 del encargo)
# --------------------------------------------------------------------------- #
def registros_validos_calorias(df: pd.DataFrame):
    """
    Filtra los registros utilizables para estimar calorías y comprueba las unidades:
    calorías > 0, duración > 0 y calorías por hora dentro de un rango fisiológicamente plausible.
    Devuelve (DataFrame válido, dict con el detalle de la validación).
    """
    minimo, maximo = cfg.RANGO_PLAUSIBLE_KCAL_H
    cph = df["calorias"] / df["duracion_h"].replace(0, np.nan)
    valido = (df["calorias"] > 0) & (df["duracion_h"] > 0) & cph.between(minimo, maximo)
    info = {
        "n_total": int(len(df)),
        "n_validos": int(valido.sum()),
        "n_excluidos": int((~valido).sum()),
        "rango_kcal_h_observado": (float(cph[valido].min()), float(cph[valido].max())) if valido.any() else (np.nan, np.nan),
        "rango_plausible": cfg.RANGO_PLAUSIBLE_KCAL_H,
    }
    return df.loc[valido], info


def verificar_linealidad(df: pd.DataFrame) -> dict:
    """
    Comprueba si «calorías por hora» es aproximadamente independiente de la duración, que es lo que
    permite estimar una sesión de otra duración multiplicando por las horas.
    """
    if len(df) < cfg.N_MIN_COMPARACION or df["duracion_h"].nunique() < 3:
        return {"valida": False, "motivo": "No hay registros o variedad de duraciones suficientes para comprobarlo."}
    r_pearson = float(df["calorias_por_hora"].corr(df["duracion_h"]))
    r_spearman = float(df["calorias_por_hora"].corr(df["duracion_h"], method="spearman"))
    pendiente, intercepto = np.polyfit(df["duracion_h"], df["calorias"], 1)
    tramos = pd.qcut(df["duracion_h"], 4, duplicates="drop")
    por_tramo = df.groupby(tramos, observed=True)["calorias_por_hora"].median()
    variacion = float((por_tramo.max() - por_tramo.min()) / por_tramo.median())
    valida = abs(r_pearson) <= LIMITE_CORRELACION_LINEALIDAD and abs(intercepto) <= 0.10 * df["calorias"].mean()
    return {
        "valida": bool(valida),
        "correlacion_pearson": r_pearson,
        "correlacion_spearman": r_spearman,
        "pendiente_kcal_por_hora": float(pendiente),
        "intercepto_kcal": float(intercepto),
        "variacion_relativa_tramos": variacion,
        "por_tramo": por_tramo,
        "motivo": ("" if valida else "Las calorías por hora cambian con la duración: escalar por horas no sería fiable."),
    }


def tabla_calorias_por_tipo(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen robusto de calorías por hora para cada tipo de entrenamiento."""
    grupos = df.groupby("tipo_entrenamiento", observed=True)["calorias_por_hora"]
    tabla = pd.DataFrame({
        "n": grupos.size(),
        "mediana": grupos.median(),
        "p10": grupos.quantile(0.10),
        "p25": grupos.quantile(0.25),
        "p75": grupos.quantile(0.75),
        "p90": grupos.quantile(0.90),
        "media": grupos.mean(),
        "minimo": grupos.min(),
        "maximo": grupos.max(),
    })
    return tabla


def analizar_calorias(df: pd.DataFrame) -> dict:
    """
    Resumen de cómo se distribuyen las calorías según tipo, duración, peso, edad, género y experiencia.
    Alimenta la tabla de «Base de la estimación» de la interfaz.
    """
    validos, validacion = registros_validos_calorias(df)
    resultado = {"validacion": validacion, "n": len(validos)}
    if validos.empty:
        return resultado
    resultado["por_tipo"] = tabla_calorias_por_tipo(validos)
    resultado["linealidad"] = verificar_linealidad(validos)

    def _resumen(col):
        g = validos.groupby(col, observed=True)
        return pd.DataFrame({"n": g.size(), "mediana kcal/h": g["calorias_por_hora"].median(),
                             "media kcal": g["calorias"].mean()})
    resultado["por_genero"] = _resumen("genero")
    resultado["por_experiencia"] = _resumen("nivel_experiencia")

    grupos_tipo = [g["calorias_por_hora"].to_numpy() for _, g in validos.groupby("tipo_entrenamiento", observed=True)
                   if len(g) >= cfg.N_MIN_GRUPO]
    resultado["p_valor_tipos"] = float(stats.kruskal(*grupos_tipo)[1]) if len(grupos_tipo) >= 2 else np.nan
    por_genero = [g["calorias_por_hora"].to_numpy() for _, g in validos.groupby("genero", observed=True)
                  if len(g) >= cfg.N_MIN_GRUPO]
    resultado["p_valor_genero"] = float(stats.mannwhitneyu(*por_genero[:2])[1]) if len(por_genero) >= 2 else np.nan

    asociaciones = []
    for col, nombre in [("duracion_h", "Duración de la sesión"), ("peso_kg", "Peso"), ("edad", "Edad"),
                        ("nivel_experiencia_cod", "Nivel de experiencia")]:
        r, p = stats.spearmanr(validos[col], validos["calorias_por_hora"])
        asociaciones.append({"Variable": nombre, "Spearman con kcal/h": round(float(r), 2), "Valor p": p_valor(p)})
    resultado["asociaciones_kcal_h"] = pd.DataFrame(asociaciones)
    return resultado


# --------------------------------------------------------------------------- #
# Entrenamiento del estimador
# --------------------------------------------------------------------------- #
def _matriz_modelo(df: pd.DataFrame) -> np.ndarray:
    """Variables del modelo complementario: edad, género (1 = masculino) y peso (kg)."""
    return np.column_stack([
        df["edad"].to_numpy(dtype=float),
        (df["genero"].astype(str) == "Masculino").to_numpy(dtype=float),
        df["peso_kg"].to_numpy(dtype=float),
    ])


def entrenar_estimador(df: pd.DataFrame) -> EstimadorCalorias:
    """Calcula las estadísticas por tipo, verifica la linealidad y entrena el modelo complementario."""
    validos, validacion = registros_validos_calorias(df)
    if len(validos) < cfg.N_MIN_COMPARACION:
        raise ValueError("No hay suficientes registros válidos para estimar calorías.")
    tabla = tabla_calorias_por_tipo(validos)
    linealidad = verificar_linealidad(validos)

    grupos = [g["calorias_por_hora"].to_numpy() for _, g in validos.groupby("tipo_entrenamiento", observed=True)
              if len(g) >= cfg.N_MIN_GRUPO]
    p_tipos = float(stats.kruskal(*grupos)[1]) if len(grupos) >= 2 else np.nan

    # Validación cruzada de 5 particiones: método principal (mediana por tipo) frente a regresión lineal.
    validos = validos.reset_index(drop=True)
    x = _matriz_modelo(validos)
    cph = validos["calorias_por_hora"].to_numpy()
    horas = validos["duracion_h"].to_numpy()
    calorias = validos["calorias"].to_numpy(dtype=float)
    pred_base = np.zeros(len(validos))
    pred_ref = np.zeros(len(validos))
    pred_ref_cph = np.zeros(len(validos))   # predicciones de calorías/hora fuera de muestra
    for entrenamiento, prueba in KFold(n_splits=5, shuffle=True, random_state=42).split(x):
        medianas = validos.iloc[entrenamiento].groupby("tipo_entrenamiento", observed=True)["calorias_por_hora"].median()
        medianas = {str(tipo): valor for tipo, valor in medianas.items()}
        cph_base = validos.iloc[prueba]["tipo_entrenamiento"].astype(str).map(medianas).fillna(np.median(cph[entrenamiento]))
        pred_base[prueba] = cph_base.to_numpy() * horas[prueba]
        modelo_fold = LinearRegression().fit(x[entrenamiento], cph[entrenamiento])
        pred_ref_cph[prueba] = modelo_fold.predict(x[prueba])
        pred_ref[prueba] = pred_ref_cph[prueba] * horas[prueba]

    def _r2(real, pred):
        return float(1 - np.sum((real - pred) ** 2) / np.sum((real - real.mean()) ** 2))

    metricas = {
        "n": int(len(validos)),
        "mae_principal": float(np.mean(np.abs(calorias - pred_base))),
        "mae_complementario": float(np.mean(np.abs(calorias - pred_ref))),
        "r2_principal": _r2(calorias, pred_base),
        "r2_complementario": _r2(calorias, pred_ref),
    }
    modelo = LinearRegression().fit(x, cph)
    # Los errores que definen el rango se toman de la validación cruzada (fuera de muestra), no del ajuste.
    residuos = cph - pred_ref_cph
    q10, q90 = np.percentile(residuos, [10, 90])

    return EstimadorCalorias(
        tabla_por_tipo=tabla,
        duracion_min_obs_h=float(validos["duracion_h"].min()),
        duracion_max_obs_h=float(validos["duracion_h"].max()),
        linealidad=linealidad, validacion=validacion, p_valor_tipos=p_tipos,
        modelo=modelo,
        rangos_modelo={"edad": (float(validos["edad"].min()), float(validos["edad"].max())),
                       "peso_kg": (float(validos["peso_kg"].min()), float(validos["peso_kg"].max()))},
        cuantiles_residuos=(float(q10), float(q90)),
        metricas_cv=metricas,
    )


# --------------------------------------------------------------------------- #
# Estimación para una sesión
# --------------------------------------------------------------------------- #
def _redondear_5(valor: float) -> float:
    """Redondea a múltiplos de 5 kcal para no aparentar una precisión que los datos no tienen."""
    return float(5 * round(valor / 5))


def estimar_calorias(est: EstimadorCalorias, tipo: str, duracion_min, edad=None, genero=None, peso_kg=None) -> ResultadoCalorias:
    """
    Estima las calorías de una sesión del tipo y duración indicados.
    Si no hay base suficiente (duración fuera del rango observado, tipo desconocido, etc.), devuelve
    un resultado con estado «fuera_de_rango» o «no_disponible» y un mensaje claro en lugar de un número.
    """
    res = ResultadoCalorias(estado="no_disponible", tipo=str(tipo))
    try:
        duracion_min = float(duracion_min)
    except (TypeError, ValueError):
        res.mensaje = "Indica una duración válida para estimar las calorías."
        return res
    if not np.isfinite(duracion_min) or duracion_min <= 0:
        res.mensaje = "La duración debe ser mayor que cero para estimar las calorías."
        return res
    res.duracion_min = duracion_min

    if not est.linealidad.get("valida", False):
        res.mensaje = ("Con estos datos no es posible convertir calorías por hora a otra duración de forma fiable. "
                       + est.linealidad.get("motivo", ""))
        return res
    if tipo not in est.tabla_por_tipo.index:
        res.mensaje = f"No hay registros suficientes del tipo «{tipo}» para estimar calorías."
        return res

    horas = duracion_min / 60.0
    minimo_min = est.duracion_min_obs_h * 60
    maximo_min = est.duracion_max_obs_h * 60
    if horas < est.duracion_min_obs_h - 1e-9 or horas > est.duracion_max_obs_h + 1e-9:
        res.estado = "fuera_de_rango"
        res.mensaje = (f"La duración ({numero(duracion_min, 0)} min) está fuera del rango observado en los datos "
                       f"({numero(minimo_min, 0)}–{numero(maximo_min, 0)} min). No se extrapola para no ofrecer una cifra sin respaldo; "
                       "elige una duración dentro de ese rango.")
        return res

    fila = est.tabla_por_tipo.loc[tipo]
    res.estado = "ok"
    res.n_registros_tipo = int(fila["n"])
    res.base_kcal = _redondear_5(fila["mediana"] * horas)
    res.base_rango = (_redondear_5(fila["p10"] * horas), _redondear_5(fila["p90"] * horas))

    # Método complementario: solo con datos completos y dentro del rango de entrenamiento
    motivo = ""
    if edad is None or genero not in cfg.GENEROS or peso_kg is None:
        motivo = "faltan edad, género o peso"
    elif not (est.rangos_modelo["edad"][0] <= edad <= est.rangos_modelo["edad"][1]):
        motivo = (f"la edad ({numero(edad, 0)} años) está fuera del rango observado "
                  f"({numero(est.rangos_modelo['edad'][0], 0)}–{numero(est.rangos_modelo['edad'][1], 0)} años)")
    elif not (est.rangos_modelo["peso_kg"][0] <= peso_kg <= est.rangos_modelo["peso_kg"][1]):
        motivo = (f"el peso ({numero(peso_kg, 1)} kg) está fuera del rango observado "
                  f"({numero(est.rangos_modelo['peso_kg'][0], 0)}–{numero(est.rangos_modelo['peso_kg'][1], 0)} kg)")
    if est.modelo is None:
        motivo = "no hay modelo disponible"
    if motivo:
        res.motivo_sin_refinada = "No se aplicó el ajuste por edad, género y peso: " + motivo + "."
    else:
        entrada = np.array([[float(edad), 1.0 if genero == "Masculino" else 0.0, float(peso_kg)]])
        cph_pred = float(est.modelo.predict(entrada)[0])
        res.refinada_kcal = _redondear_5(cph_pred * horas)
        res.refinada_rango = (_redondear_5((cph_pred + est.cuantiles_residuos[0]) * horas),
                              _redondear_5((cph_pred + est.cuantiles_residuos[1]) * horas))
        res.refinada_aplicada = True

    res.advertencias = [
        "Es una estimación basada en registros del conjunto de datos, no una medición. "
        "El gasto energético real puede variar considerablemente entre personas y sesiones.",
        "El rango indicado cubre aproximadamente el 80 % central de los casos observados; "
        "una sesión concreta puede quedar fuera de él.",
    ]
    if np.isfinite(est.p_valor_tipos) and est.p_valor_tipos >= 0.05:
        res.advertencias.append(
            f"En estos datos las calorías por hora no difieren de forma estadísticamente clara entre tipos de entrenamiento "
            f"(Kruskal-Wallis, p {p_valor(est.p_valor_tipos)}): la estimación no sirve para elegir entre actividades.")
    return res


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def figura_estimacion_duracion(est: EstimadorCalorias, df: pd.DataFrame, tipo: str, duracion_actual_min=None,
                               edad=None, genero=None, peso_kg=None) -> go.Figure:
    """Muestra la estimación para todas las duraciones observadas, junto a los registros reales del tipo."""
    validos, _ = registros_validos_calorias(df)
    reales = validos[validos["tipo_entrenamiento"].astype(str) == tipo]
    duraciones = np.arange(np.ceil(est.duracion_min_obs_h * 60), np.floor(est.duracion_max_obs_h * 60) + 1, 5)
    medianas, bajos, altos = [], [], []
    for d in duraciones:
        r = estimar_calorias(est, tipo, d)
        medianas.append(r.base_kcal)
        bajos.append(r.base_rango[0])
        altos.append(r.base_rango[1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reales["duracion_min"], y=reales["calorias"], mode="markers", name=f"Registros de {tipo}",
                             marker=dict(size=6, color=cfg.COLORES_TIPO.get(tipo, cfg.COLOR_MORADO_CLARO), opacity=0.45,
                                         line=dict(width=0.4, color="white")),
                             hovertemplate="Duración: %{x:.0f} min<br>Calorías: %{y:.0f} kcal<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.concatenate([duraciones, duraciones[::-1]]), y=np.concatenate([altos, bajos[::-1]]),
                             fill="toself", fillcolor="rgba(94,60,153,0.15)", line=dict(width=0), name="Rango (≈80 % central)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=duraciones, y=medianas, mode="lines", name="Estimación (método principal)",
                             line=dict(color=cfg.COLOR_MORADO, width=3),
                             hovertemplate="Duración: %{x:.0f} min<br>Estimación: %{y:.0f} kcal<extra></extra>"))
    if duracion_actual_min is not None:
        actual = estimar_calorias(est, tipo, duracion_actual_min, edad, genero, peso_kg)
        if actual.estado == "ok":
            fig.add_trace(go.Scatter(x=[duracion_actual_min], y=[actual.base_kcal], mode="markers", name="Tu duración",
                                     marker=dict(size=14, color=cfg.COLOR_NARANJA, symbol="diamond",
                                                 line=dict(width=2, color="white")),
                                     hovertemplate="Tu duración: %{x:.0f} min<br>Estimación: %{y:.0f} kcal<extra></extra>"))
    fig.update_layout(xaxis_title="Duración de la sesión (min)", yaxis_title="Calorías (kcal)")
    return estilo_figura(fig, f"Calorías según la duración · {tipo}", 420)


def figura_calorias_hora_por_tipo(est: EstimadorCalorias) -> go.Figure:
    """Mediana y rango P10-P90 de calorías por hora por tipo de entrenamiento."""
    tabla = est.tabla_por_tipo
    fig = go.Figure()
    for tipo in tabla.index:
        fila = tabla.loc[tipo]
        fig.add_trace(go.Bar(
            x=[str(tipo)], y=[fila["mediana"]], name=str(tipo), marker_color=cfg.COLORES_TIPO.get(str(tipo), cfg.COLOR_NARANJA),
            marker_line_color=cfg.COLOR_MORADO_OSCURO, marker_line_width=0.6,
            error_y=dict(type="data", symmetric=False, array=[fila["p90"] - fila["mediana"]], arrayminus=[fila["mediana"] - fila["p10"]],
                         color=cfg.COLOR_MORADO_OSCURO),
            customdata=[[fila["n"], fila["p10"], fila["p90"]]],
            hovertemplate=("<b>%{x}</b><br>Mediana: %{y:.0f} kcal/h<br>P10–P90: %{customdata[1]:.0f}–%{customdata[2]:.0f} kcal/h"
                           "<br>Registros: %{customdata[0]}<extra></extra>")))
    fig.update_layout(yaxis_title="Calorías por hora (kcal/h)", xaxis_title="Tipo de entrenamiento")
    return estilo_figura(fig, "Calorías por hora: mediana y rango P10–P90", 360, leyenda=False)
