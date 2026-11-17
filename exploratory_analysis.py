"""
Análisis exploratorio: indicadores, filtros, gráficos interactivos (Plotly),
comparaciones entre variables, correlaciones y análisis de perfiles.

Este módulo NO usa Streamlit: recibe DataFrames y devuelve DataFrames, figuras de
Plotly y textos. Así se puede probar de forma independiente y la interfaz (app.py)
solo se ocupa de mostrar los resultados.

Criterio estadístico general: las «lecturas» automáticas describen los datos
observados con pruebas estándar (Kruskal-Wallis, Mann-Whitney, chi-cuadrado,
Pearson/Spearman). Nunca se interpretan como causalidad ni como prueba de que un
entrenamiento sea mejor que otro.
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from . import config as cfg
from .formato import numero, p_valor, porcentaje

ALFA = 0.05  # nivel de significación usado en las lecturas automáticas


# =========================================================================== #
# Estilo común de los gráficos
# =========================================================================== #
def etiqueta(col: str) -> str:
    """Etiqueta en español (con unidad) de una columna."""
    return cfg.ETIQUETAS.get(col, col)


def etiqueta_corta(col: str) -> str:
    """Etiqueta sin la unidad entre paréntesis (para ejes con poco espacio)."""
    return etiqueta(col).split(" (")[0]


def paleta_para(col: str):
    """Mapa de colores para variables categóricas conocidas (None si no hay)."""
    return {
        "tipo_entrenamiento": cfg.COLORES_TIPO,
        "genero": cfg.COLORES_GENERO,
        "nivel_experiencia": cfg.COLORES_NIVEL,
        "categoria_imc": cfg.COLORES_IMC,
    }.get(col)


def estilo_figura(fig: go.Figure, titulo=None, alto=400, leyenda=True) -> go.Figure:
    """Aplica el estilo visual común (limpio, tipografía legible, separadores españoles)."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=titulo, x=0.0, xanchor="left", y=0.97, yanchor="top",
                   font=dict(size=16, color=cfg.COLOR_MORADO_OSCURO)) if titulo else None,
        font=dict(family="Inter, 'Segoe UI', Arial, sans-serif", size=13, color=cfg.COLOR_TEXTO),
        height=alto,
        # margen superior: deja sitio al título y, si hay leyenda, a la leyenda debajo del título
        margin=dict(l=10, r=10, t=(90 if leyenda else 55) if titulo else (45 if leyenda else 20), b=10),
        separators=",.",  # coma decimal y punto de miles
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor="#EEE8F5", zerolinecolor="#EEE8F5")
    fig.update_yaxes(gridcolor="#EEE8F5", zerolinecolor="#EEE8F5")
    return fig


def figura_vacia(mensaje: str, alto=300) -> go.Figure:
    """Figura con un mensaje (se usa cuando no hay datos suficientes)."""
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=15, color=cfg.COLOR_MORADO))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return estilo_figura(fig, alto=alto, leyenda=False)


# =========================================================================== #
# Grupos de edad (intervalos configurables dentro del rango observado)
# =========================================================================== #
def opciones_cortes_edad(df: pd.DataFrame, paso=5) -> list:
    """Puntos de corte posibles (múltiplos de `paso`) estrictamente dentro del rango de edades observado."""
    minimo, maximo = int(df["edad"].min()), int(df["edad"].max())
    return [c for c in range(minimo + 1, maximo + 1) if c % paso == 0]


def crear_grupos_edad(edad: pd.Series, cortes) -> pd.Series:
    """
    Agrupa la edad según los puntos de corte. Cada corte inicia un grupo nuevo:
    con cortes [30, 40] y edades de 18 a 59 resultan «18–29», «30–39» y «40–59».
    Los cortes fuera del rango observado se ignoran.
    """
    minimo, maximo = int(edad.min()), int(edad.max())
    cortes_validos = sorted({int(c) for c in (cortes or []) if minimo < int(c) <= maximo})
    limites = [minimo] + cortes_validos + [maximo + 1]
    etiquetas = [f"{limites[i]}–{limites[i + 1] - 1}" for i in range(len(limites) - 1)]
    return pd.cut(edad, bins=limites, right=False, labels=etiquetas, ordered=True)


def agregar_grupo_edad(df: pd.DataFrame, cortes=(30, 40, 50)) -> pd.DataFrame:
    """Devuelve una copia del DataFrame con la columna `grupo_edad`."""
    copia = df.copy()
    copia["grupo_edad"] = crear_grupos_edad(copia["edad"], cortes)
    return copia


# =========================================================================== #
# Indicadores (tarjetas KPI)
# =========================================================================== #
def calcular_indicadores(df) -> list:
    """
    Calcula los 5 indicadores principales a partir de los datos cargados.
    Cada indicador es un dict con nombre, valor (texto), unidad y mensaje de error
    (None si todo fue bien). Nunca lanza excepciones por datos ausentes.
    """
    definiciones = [
        ("Registros", None, "registros", 0),
        ("Edad promedio", "edad", "años", 1),
        ("IMC promedio", "imc", "kg/m²", 1),
        ("Duración promedio", "duracion_min", "min", 0),
        ("Calorías promedio", "calorias", "kcal", 0),
    ]
    indicadores = []
    for nombre, col, unidad, decimales in definiciones:
        item = {"nombre": nombre, "valor": "—", "unidad": unidad, "mensaje": None}
        try:
            if df is None or len(df) == 0:
                item["mensaje"] = "No hay registros disponibles para calcular este indicador."
            elif col is None:
                item["valor"] = numero(len(df), 0)
            elif col not in df.columns:
                item["mensaje"] = f"La columna necesaria para «{nombre}» no existe en los datos."
            else:
                serie = pd.to_numeric(df[col], errors="coerce").dropna()
                if serie.empty:
                    item["mensaje"] = f"Todos los valores de «{etiqueta(col)}» faltan o no son numéricos."
                else:
                    item["valor"] = numero(serie.mean(), decimales)
        except Exception as error:  # el dashboard no debe romperse por un indicador
            item["mensaje"] = f"No se pudo calcular este indicador ({error})."
        indicadores.append(item)
    return indicadores


def textos_inicio(df: pd.DataFrame) -> dict:
    """Descripciones breves (calculadas con los datos) para interpretar las gráficas de inicio."""
    if df is None or len(df) == 0:
        mensaje = "No hay datos para describir."
        return {"edad": mensaje, "tipos": mensaje, "niveles": mensaje, "calorias": mensaje}
    q1, med, q3 = df["edad"].quantile([0.25, 0.5, 0.75])
    textos = {
        "edad": (f"La mitad central de los miembros tiene entre {numero(q1, 0)} y {numero(q3, 0)} años "
                 f"(mediana: {numero(med, 0)}). Rango observado: {numero(df['edad'].min(), 0)} a {numero(df['edad'].max(), 0)} años."),
    }
    proporciones = df["tipo_entrenamiento"].value_counts(normalize=True)
    mayor, menor = proporciones.idxmax(), proporciones.idxmin()
    diferencia = proporciones.max() - proporciones.min()
    equilibrio = ("La distribución es bastante equilibrada." if diferencia <= 0.10
                  else "Hay diferencias notables en la popularidad de los tipos.")
    textos["tipos"] = (f"{mayor} es el tipo más registrado ({porcentaje(proporciones.max())}) y {menor} el menos registrado "
                       f"({porcentaje(proporciones.min())}). {equilibrio}")
    prop_niv = df["nivel_experiencia"].value_counts(normalize=True)
    textos["niveles"] = "Participación por nivel: " + "; ".join(
        f"{n} {porcentaje(prop_niv.get(n, 0))}" for n in cfg.ORDEN_NIVELES) + "."
    q1c, medc, q3c = df["calorias"].quantile([0.25, 0.5, 0.75])
    rho = df[["duracion_min", "calorias"]].corr(method="spearman").iloc[0, 1]
    textos["calorias"] = (f"La mitad central de las sesiones registra entre {numero(q1c, 0)} y {numero(q3c, 0)} kcal "
                          f"(mediana: {numero(medc, 0)} kcal). Las calorías están muy asociadas a la duración de la sesión "
                          f"(correlación de Spearman = {numero(rho, 2)}); es una asociación, no una relación causal demostrada.")
    return textos


# =========================================================================== #
# Filtros
# =========================================================================== #
def rangos_filtros(df: pd.DataFrame) -> dict:
    """Límites de los filtros calculados a partir de los datos (redondeados hacia afuera)."""
    return {
        "edad": (int(df["edad"].min()), int(df["edad"].max())),
        "imc": (math.floor(df["imc"].min() * 10) / 10, math.ceil(df["imc"].max() * 10) / 10),
        "duracion": (int(math.floor(df["duracion_min"].min())), int(math.ceil(df["duracion_min"].max()))),
    }


def aplicar_filtros(df, edad=None, generos=None, niveles=None, tipos=None, imc=None, duracion=None):
    """
    Aplica los filtros de la sección de exploración. Un filtro en None no se aplica;
    una lista vacía deja 0 registros (el usuario desmarcó todo).
    """
    mascara = pd.Series(True, index=df.index)
    if edad is not None:
        mascara &= df["edad"].between(edad[0], edad[1])
    if generos is not None:
        mascara &= df["genero"].isin(generos)
    if niveles is not None:
        mascara &= df["nivel_experiencia"].isin(niveles)
    if tipos is not None:
        mascara &= df["tipo_entrenamiento"].isin(tipos)
    if imc is not None:
        mascara &= df["imc"].between(imc[0], imc[1])
    if duracion is not None:
        mascara &= df["duracion_min"].between(duracion[0], duracion[1])
    return df.loc[mascara]


# =========================================================================== #
# Pruebas y «lecturas» automáticas
# =========================================================================== #
def _fuerza_correlacion(r: float) -> str:
    valor = abs(r)
    if valor < 0.1:
        return "prácticamente nula"
    if valor < 0.3:
        return "débil"
    if valor < 0.5:
        return "moderada"
    if valor < 0.7:
        return "fuerte"
    return "muy fuerte"


def lectura_relacion_numerica(df, x: str, y: str) -> str:
    """Correlaciones de Pearson y Spearman entre dos variables numéricas, con cautelas."""
    datos = df[[x, y]].dropna()
    if len(datos) < cfg.N_MIN_COMPARACION:
        return f"Solo hay {len(datos)} registros: es una muestra pequeña para estimar una correlación con fiabilidad."
    if datos[x].nunique() < 2 or datos[y].nunique() < 2:
        return "Una de las variables no varía en la selección actual; no se puede calcular la correlación."
    r_p, p_p = stats.pearsonr(datos[x], datos[y])
    r_s, p_s = stats.spearmanr(datos[x], datos[y])
    direccion = "positiva" if r_s > 0 else "negativa"
    return (f"Relación {_fuerza_correlacion(r_s)} {direccion} (Spearman = {numero(r_s, 2)}, p {p_valor(p_s)}; "
            f"Pearson = {numero(r_p, 2)}, p {p_valor(p_p)}; n = {numero(len(datos), 0)}). "
            "Una correlación describe asociación y no demuestra causalidad.")


def lectura_diferencia_grupos(df, y: str, grupo: str) -> str:
    """Prueba de Kruskal-Wallis de una variable numérica entre grupos, con tamaño del efecto (ε²)."""
    datos = df[[y, grupo]].dropna()
    grupos = {str(k): g[y].to_numpy() for k, g in datos.groupby(grupo, observed=True) if len(g) >= cfg.N_MIN_GRUPO}
    if len(grupos) < 2:
        return (f"Se necesitan al menos 2 grupos con {cfg.N_MIN_GRUPO} registros o más para comparar; "
                "con la selección actual no es posible hacer una prueba estadística.")
    todos = np.concatenate(list(grupos.values()))
    if np.ptp(todos) == 0:
        return "La variable no varía en la selección actual; no hay nada que comparar."
    h, p = stats.kruskal(*grupos.values())
    n = len(todos)
    epsilon2 = h / (n - 1)
    magnitud = ("despreciable" if epsilon2 < 0.01 else "pequeña" if epsilon2 < 0.06
                else "moderada" if epsilon2 < 0.14 else "grande")
    medianas = "; ".join(f"{k}: {numero(np.median(v), 1)}" for k, v in grupos.items())
    if p < ALFA:
        veredicto = ("Se detecta una diferencia estadísticamente significativa entre grupos "
                     f"(Kruskal-Wallis, p {p_valor(p)}; tamaño del efecto {magnitud}, ε² = {numero(epsilon2, 3)}).")
    else:
        veredicto = ("No se detecta una diferencia estadísticamente clara entre grupos "
                     f"(Kruskal-Wallis, p {p_valor(p)}; tamaño del efecto {magnitud}, ε² = {numero(epsilon2, 3)}).")
    return (f"{veredicto} Medianas por grupo → {medianas}. Esto describe los datos observados; "
            "no indica que un grupo sea «mejor» que otro ni una relación causal.")


def lectura_asociacion_categoricas(df, a: str, b: str) -> str:
    """Prueba de chi-cuadrado de independencia entre dos variables categóricas (con V de Cramér)."""
    datos = df[[a, b]].dropna()
    tabla = pd.crosstab(datos[a], datos[b])
    tabla = tabla.loc[tabla.sum(axis=1) > 0, tabla.sum(axis=0) > 0]
    if len(datos) < cfg.N_MIN_COMPARACION or tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return "No hay suficientes registros o categorías para una prueba de asociación fiable."
    chi2, p, _, esperado = stats.chi2_contingency(tabla)
    v = math.sqrt(chi2 / (tabla.to_numpy().sum() * (min(tabla.shape) - 1)))
    magnitud = "despreciable" if v < 0.1 else "pequeña" if v < 0.3 else "moderada" if v < 0.5 else "grande"
    aviso = " (algunas frecuencias esperadas son menores que 5: interpretar con cautela)" if (esperado < 5).any() else ""
    if p < ALFA:
        veredicto = f"Se detecta asociación estadísticamente significativa (chi-cuadrado, p {p_valor(p)}; V de Cramér = {numero(v, 2)}, {magnitud})"
    else:
        veredicto = f"No se detecta una asociación estadísticamente clara (chi-cuadrado, p {p_valor(p)}; V de Cramér = {numero(v, 2)}, {magnitud})"
    return veredicto + aviso + ". Describe los datos observados, no causalidad."


# =========================================================================== #
# Tablas descriptivas
# =========================================================================== #
def tabla_frecuencias(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Frecuencias absolutas y relativas de una variable categórica (conserva el orden de las categorías)."""
    conteo = df[col].value_counts(sort=False, dropna=False)
    tabla = pd.DataFrame({etiqueta(col): conteo.index.astype(str), "Registros": conteo.to_numpy()})
    total = tabla["Registros"].sum()
    tabla["Porcentaje (%)"] = (tabla["Registros"] / total * 100).round(1) if total else 0.0
    return tabla


def estadisticas_descriptivas(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Estadísticos descriptivos de una variable numérica, listos para mostrar."""
    serie = pd.to_numeric(df[col], errors="coerce")
    valida = serie.dropna()
    if valida.empty:
        return pd.DataFrame({"Estadístico": ["Registros válidos"], "Valor": ["0"]})
    filas = [
        ("Registros válidos", numero(len(valida), 0)),
        ("Faltantes", numero(int(serie.isna().sum()), 0)),
        ("Media", numero(valida.mean(), 2)),
        ("Desviación estándar", numero(valida.std(), 2) if len(valida) > 1 else "—"),
        ("Mínimo", numero(valida.min(), 2)),
        ("Percentil 25", numero(valida.quantile(0.25), 2)),
        ("Mediana", numero(valida.median(), 2)),
        ("Percentil 75", numero(valida.quantile(0.75), 2)),
        ("Máximo", numero(valida.max(), 2)),
        ("Asimetría", numero(valida.skew(), 2) if len(valida) > 2 else "—"),
    ]
    return pd.DataFrame(filas, columns=["Estadístico", "Valor"])


# =========================================================================== #
# Gráficos básicos
# =========================================================================== #
def figura_histograma(df, col, titulo=None, color_por=None, nbins=None, color=cfg.COLOR_NARANJA, alto=380):
    """Histograma de una variable numérica (opcionalmente separado por una categoría)."""
    datos = df.dropna(subset=[col])
    if datos.empty:
        return figura_vacia("No hay datos para esta variable con los filtros actuales.")
    kwargs = dict(x=col, nbins=nbins, labels={col: etiqueta(col)}, opacity=0.9)
    if color_por:
        fig = px.histogram(datos, color=color_por, color_discrete_map=paleta_para(color_por),
                           barmode="overlay", labels={color_por: etiqueta(color_por), col: etiqueta(col)},
                           x=col, nbins=nbins, opacity=0.65,
                           category_orders={color_por: list(datos[color_por].cat.categories)} if hasattr(datos[color_por], "cat") else None)
    else:
        fig = px.histogram(datos, color_discrete_sequence=[color], **kwargs)
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(yaxis_title="Número de registros", bargap=0.04)
    return estilo_figura(fig, titulo, alto, leyenda=bool(color_por))


def figura_barras_categoria(df, col, titulo=None, alto=380):
    """Gráfico de barras con el número y porcentaje de registros por categoría."""
    if len(df) == 0:
        return figura_vacia("No hay datos para esta variable con los filtros actuales.")
    tabla = tabla_frecuencias(df, col)
    nombres = tabla[etiqueta(col)].tolist()
    paleta = paleta_para(col) or {}
    colores = [paleta.get(n, cfg.COLOR_NARANJA) for n in nombres]
    fig = go.Figure(go.Bar(
        x=nombres, y=tabla["Registros"], marker_color=colores,
        marker_line_color=cfg.COLOR_MORADO_OSCURO, marker_line_width=0.6,
        text=[f"{numero(p, 1)} %" for p in tabla["Porcentaje (%)"]], textposition="outside",
        customdata=tabla["Porcentaje (%)"],
        hovertemplate="<b>%{x}</b><br>Registros: %{y}<br>Porcentaje: %{customdata:.1f} %<extra></extra>",
    ))
    fig.update_layout(xaxis_title=etiqueta(col), yaxis_title="Número de registros")
    fig.update_yaxes(range=[0, float(tabla["Registros"].max()) * 1.18])
    return estilo_figura(fig, titulo, alto, leyenda=False)


def figura_caja(df, y, x=None, color_por=None, titulo=None, alto=400):
    """Diagrama de caja de una variable numérica, opcionalmente por categoría."""
    datos = df.dropna(subset=[y])
    if datos.empty:
        return figura_vacia("No hay datos para esta variable con los filtros actuales.")
    eje_color = color_por or x
    fig = px.box(datos, x=x, y=y, color=eje_color, points="outliers",
                 color_discrete_map=paleta_para(eje_color) if eje_color else None,
                 color_discrete_sequence=[cfg.COLOR_NARANJA] if not eje_color else None,
                 labels={y: etiqueta(y), **({x: etiqueta(x)} if x else {}), **({color_por: etiqueta(color_por)} if color_por else {})},
                 category_orders={c: list(datos[c].cat.categories) for c in [x, color_por] if c and hasattr(datos[c], "cat")})
    fig.update_traces(boxmean=True, marker=dict(size=4, opacity=0.6), line=dict(width=1.5))
    return estilo_figura(fig, titulo, alto, leyenda=bool(color_por and color_por != x))


# =========================================================================== #
# Comparaciones entre variables (A-H)
# =========================================================================== #
COMPARACIONES = {
    "A": dict(titulo="A. Edad según tipo de entrenamiento", tipo="caja", x="tipo_entrenamiento", y="edad",
              descripcion="Cada caja resume la edad de quienes registraron cada tipo de entrenamiento: la línea central es la mediana, "
                          "la caja contiene el 50 % central y la línea discontinua indica la media."),
    "B": dict(titulo="B. IMC según tipo de entrenamiento", tipo="caja", x="tipo_entrenamiento", y="imc",
              descripcion="Distribución del IMC por tipo de entrenamiento. Las líneas punteadas marcan los límites convencionales "
                          "(18,5 / 25 / 30), que son orientativos."),
    "C": dict(titulo="C. Nivel de experiencia según tipo de entrenamiento", tipo="barras_prop", x="nivel_experiencia", y="tipo_entrenamiento",
              descripcion="Dentro de cada nivel de experiencia, qué porcentaje de los registros corresponde a cada tipo de entrenamiento."),
    "D": dict(titulo="D. Experiencia frente a calorías registradas", tipo="caja", x="nivel_experiencia", y="calorias",
              descripcion="Calorías registradas por nivel de experiencia. Recuerda que el nivel está muy ligado a la duración de la sesión."),
    "E": dict(titulo="E. Duración de la sesión frente a calorías registradas", tipo="dispersion", x="duracion_min", y="calorias",
              descripcion="Cada punto es una sesión. La línea punteada es una tendencia lineal global."),
    "F": dict(titulo="F. Peso corporal frente a calorías registradas", tipo="dispersion", x="peso_kg", y="calorias",
              descripcion="Cada punto es una sesión. Observa si la nube de puntos muestra o no una tendencia."),
    "G": dict(titulo="G. Edad frente a calorías registradas", tipo="dispersion", x="edad", y="calorias",
              descripcion="Cada punto es una sesión. La edad puede estar asociada a otras variables (como la duración), "
                          "por lo que la tendencia no debe leerse de forma aislada."),
    "H": dict(titulo="H. Tipo de entrenamiento frente a duración", tipo="caja", x="tipo_entrenamiento", y="duracion_min",
              descripcion="Duración de las sesiones (en minutos) para cada tipo de entrenamiento."),
}

_COLUMNAS_HOVER = ["tipo_entrenamiento", "nivel_experiencia", "genero", "edad", "peso_kg", "duracion_min", "calorias"]


def _linea_tendencia(x: pd.Series, y: pd.Series):
    """Recta de mínimos cuadrados (devuelve None si no se puede ajustar)."""
    if len(x) < 3 or x.nunique() < 2:
        return None
    pendiente, intercepto = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    return xs, pendiente * xs + intercepto


def _figura_dispersion(datos, x, y, color_por="tipo_entrenamiento", titulo=None, alto=440, con_tendencia=True):
    """Diagrama de dispersión con ventana emergente informativa y tendencia lineal global."""
    hover = {c: (":.1f" if c in ("peso_kg",) else ":.0f" if c in ("edad", "duracion_min", "calorias") else True)
             for c in _COLUMNAS_HOVER if c in datos.columns and c not in (color_por,)}
    hover[x] = ":.1f"
    hover[y] = ":.1f"
    labels = {c: etiqueta(c) for c in set(_COLUMNAS_HOVER + [x, y] + ([color_por] if color_por else []))}
    fig = px.scatter(datos, x=x, y=y, color=color_por, color_discrete_map=paleta_para(color_por) if color_por else None,
                     color_discrete_sequence=[cfg.COLOR_NARANJA] if not color_por else None,
                     hover_data=hover, labels=labels, opacity=0.7,
                     category_orders={color_por: list(datos[color_por].cat.categories)} if color_por and hasattr(datos[color_por], "cat") else None)
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    if con_tendencia:
        tendencia = _linea_tendencia(datos[x], datos[y])
        if tendencia is not None:
            fig.add_trace(go.Scatter(x=tendencia[0], y=tendencia[1], mode="lines", name="Tendencia lineal",
                                     line=dict(color=cfg.COLOR_MORADO_OSCURO, width=2.5, dash="dash"), hoverinfo="skip"))
    return estilo_figura(fig, titulo, alto, leyenda=bool(color_por) or con_tendencia)


def construir_comparacion(df: pd.DataFrame, clave: str):
    """
    Genera la comparación A-H pedida. Devuelve (figura, lectura, mensaje):
    si hay un problema (pocos datos), la figura es None y `mensaje` explica por qué.
    """
    cfg_comp = COMPARACIONES[clave]
    if len(df) < cfg.N_MIN_GRUPO:
        return None, "", ("Con los filtros actuales quedan muy pocos registros "
                          f"({len(df)}). Amplía la selección para ver esta comparación.")
    x, y, tipo = cfg_comp["x"], cfg_comp["y"], cfg_comp["tipo"]
    titulo = cfg_comp["titulo"]

    if tipo == "caja":
        fig = figura_caja(df, y=y, x=x, titulo=titulo)
        if clave == "B":
            for limite in cfg.LIMITES_IMC:
                fig.add_hline(y=limite, line_dash="dot", line_color="#8E8E8E",
                              annotation_text=numero(limite, 1 if limite % 1 else 0), annotation_position="right")
        return fig, lectura_diferencia_grupos(df, y, x), None

    if tipo == "barras_prop":
        tabla = pd.crosstab(df[x], df[y])
        porcentajes = tabla.div(tabla.sum(axis=1).replace(0, np.nan), axis=0) * 100
        fig = go.Figure()
        for categoria in [c for c in cfg.TIPOS_ENTRENAMIENTO if c in tabla.columns]:
            fig.add_trace(go.Bar(
                x=[str(i) for i in tabla.index], y=porcentajes[categoria].fillna(0), name=categoria,
                marker_color=cfg.COLORES_TIPO[categoria], marker_line_color=cfg.COLOR_MORADO_OSCURO, marker_line_width=0.5,
                customdata=tabla[categoria], text=[f"{numero(v, 0)} %" for v in porcentajes[categoria].fillna(0)],
                hovertemplate="<b>%{x}</b><br>" + categoria + ": %{y:.1f} % (%{customdata} registros)<extra></extra>"))
        fig.update_layout(barmode="group", xaxis_title=etiqueta(x), yaxis_title="% de registros dentro de cada nivel")
        estilo_figura(fig, titulo, 420)
        return fig, lectura_asociacion_categoricas(df, x, y), None

    # dispersión
    fig = _figura_dispersion(df, x, y, titulo=titulo)
    return fig, lectura_relacion_numerica(df, x, y), None


# =========================================================================== #
# Correlaciones
# =========================================================================== #
def matriz_correlacion(df: pd.DataFrame, variables, metodo="pearson"):
    """Matriz de correlación (pares con datos completos). Devuelve None si no hay datos suficientes."""
    variables = [v for v in variables if v in df.columns and df[v].nunique(dropna=True) > 1]
    if len(df) < cfg.N_MIN_GRUPO or len(variables) < 2:
        return None
    return df[variables].corr(method=metodo)


def figura_matriz_correlacion(corr: pd.DataFrame, metodo="pearson"):
    """Mapa de calor interactivo de la matriz de correlación."""
    nombres = [etiqueta_corta(c) for c in corr.columns]
    valores = corr.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=valores, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=cfg.ESCALA_CORRELACION,
        text=np.round(valores, 2), texttemplate="%{text:.2f}", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlación: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="r" if metodo == "pearson" else "ρ")),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    nombre_metodo = "Pearson" if metodo == "pearson" else "Spearman"
    return estilo_figura(fig, f"Matriz de correlación ({nombre_metodo})", alto=560, leyenda=False)


# Pares que están relacionados por construcción (no por un hallazgo empírico).
_PARES_DERIVADOS = [{"imc", "peso_kg"}, {"imc", "estatura_m"}]


def pares_mas_correlacionados(corr: pd.DataFrame, n=6) -> pd.DataFrame:
    """Los n pares con mayor correlación absoluta, indicando cuándo la relación es matemática (IMC)."""
    pares = []
    columnas = list(corr.columns)
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            valor = corr.loc[a, b]
            if pd.notna(valor):
                nota = "Por construcción (IMC = peso ÷ estatura²)" if {a, b} in _PARES_DERIVADOS else ""
                pares.append((etiqueta_corta(a), etiqueta_corta(b), round(float(valor), 2), nota))
    tabla = pd.DataFrame(pares, columns=["Variable 1", "Variable 2", "Correlación", "Nota"])
    if tabla.empty:
        return tabla
    return tabla.reindex(tabla["Correlación"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)


# =========================================================================== #
# Perfiles de usuarios
# =========================================================================== #
def filtrar_perfil(df, grupos_edad=None, categorias_imc=None, niveles=None, tipos=None) -> pd.DataFrame:
    """Selecciona un perfil. Una selección vacía (o None) significa «todos»."""
    mascara = pd.Series(True, index=df.index)
    if grupos_edad:
        mascara &= df["grupo_edad"].isin(grupos_edad)
    if categorias_imc:
        mascara &= df["categoria_imc"].isin(categorias_imc)
    if niveles:
        mascara &= df["nivel_experiencia"].isin(niveles)
    if tipos:
        mascara &= df["tipo_entrenamiento"].isin(tipos)
    return df.loc[mascara]


_MEDIDAS_PERFIL = [("Edad (años)", "edad"), ("IMC (kg/m²)", "imc"),
                   ("Duración (min)", "duracion_min"), ("Calorías (kcal)", "calorias")]


def resumir_perfil(perfil: pd.DataFrame, completo: pd.DataFrame, variables_filtradas=()) -> dict:
    """
    Resume un perfil y lo compara con el resto del conjunto (registros que NO pertenecen al perfil)
    solo cuando la comparación es estadísticamente razonable:
      * ambos grupos con al menos N_MIN_COMPARACION registros;
      * no se compara una variable con la que se definió el propio perfil (diferencia «por construcción»).
    """
    resumen = {"n": len(perfil), "pct_total": len(perfil) / len(completo) if len(completo) else np.nan}
    resto = completo.drop(index=perfil.index)
    suficiente = len(perfil) >= cfg.N_MIN_COMPARACION and len(resto) >= cfg.N_MIN_COMPARACION

    # Distribución de tipos de entrenamiento
    conteo = perfil["tipo_entrenamiento"].value_counts(sort=False)
    total_tipos = completo["tipo_entrenamiento"].value_counts(sort=False)
    dist = pd.DataFrame({
        "Tipo de entrenamiento": [str(t) for t in conteo.index],
        "Registros en el perfil": conteo.to_numpy(),
        "% en el perfil": (conteo / max(len(perfil), 1) * 100).round(1).to_numpy(),
        "% en el conjunto completo": (total_tipos / len(completo) * 100).round(1).reindex(conteo.index).to_numpy(),
    })
    resumen["distribucion_tipos"] = dist
    esperado = (total_tipos / len(completo)).to_numpy() * len(perfil)
    if suficiente and (esperado >= 5).all():
        _, p_tipos = stats.chisquare(conteo.to_numpy(), f_exp=esperado)
        resumen["lectura_tipos"] = (
            f"La mezcla de tipos de entrenamiento del perfil {'difiere' if p_tipos < ALFA else 'no difiere de forma estadísticamente clara'} "
            f"de la del conjunto completo (chi-cuadrado de bondad de ajuste, p {p_valor(p_tipos)}). "
            "Es una descripción de lo registrado, no una indicación de qué entrenamiento es mejor.")
    else:
        resumen["lectura_tipos"] = ("Muestra insuficiente para comparar la mezcla de tipos con el conjunto completo "
                                    f"(se requieren al menos {cfg.N_MIN_COMPARACION} registros en el perfil y fuera de él).")

    # Medidas numéricas
    filas = []
    for nombre, col in _MEDIDAS_PERFIL:
        fila = {"Variable": nombre,
                "Perfil · mediana": numero(perfil[col].median(), 1) if len(perfil) else "—",
                "Perfil · media": numero(perfil[col].mean(), 1) if len(perfil) else "—",
                "Conjunto · mediana": numero(completo[col].median(), 1),
                "Conjunto · media": numero(completo[col].mean(), 1),
                "Diferencia de medianas": "—", "Valor p": "—"}
        if col in variables_filtradas:
            fila["Lectura"] = "No comparable: el perfil se definió con esta variable"
        elif not suficiente:
            fila["Lectura"] = f"Muestra insuficiente (n < {cfg.N_MIN_COMPARACION})"
        else:
            diferencia = perfil[col].median() - completo[col].median()
            fila["Diferencia de medianas"] = ("+" if diferencia > 0 else "") + numero(diferencia, 1)
            if perfil[col].nunique() < 2 and resto[col].nunique() < 2:
                fila["Lectura"] = "Sin variación"
            else:
                _, p = stats.mannwhitneyu(perfil[col].dropna(), resto[col].dropna(), alternative="two-sided")
                fila["Valor p"] = p_valor(p)
                if p < ALFA:
                    fila["Lectura"] = "Diferencia estadísticamente detectable (revisar magnitud)"
                else:
                    fila["Lectura"] = "Sin diferencia detectable"
        filas.append(fila)
    resumen["medidas"] = pd.DataFrame(filas)
    return resumen


def figura_distribucion_tipos_perfiles(perfiles: dict) -> go.Figure:
    """Barras agrupadas: % de cada tipo de entrenamiento en cada perfil (y en el conjunto completo)."""
    colores = [cfg.COLOR_NARANJA, cfg.COLOR_MORADO, "#9A9A9A"]
    fig = go.Figure()
    for (nombre, datos), color in zip(perfiles.items(), colores):
        if len(datos) == 0:
            continue
        conteo = datos["tipo_entrenamiento"].value_counts(sort=False)
        pct = conteo / conteo.sum() * 100
        fig.add_trace(go.Bar(
            x=[str(t) for t in conteo.index], y=pct.to_numpy(), name=f"{nombre} (n = {len(datos)})",
            marker_color=color, customdata=conteo.to_numpy(),
            hovertemplate="<b>%{x}</b><br>" + nombre + ": %{y:.1f} % (%{customdata} registros)<extra></extra>"))
    fig.update_layout(barmode="group", xaxis_title="Tipo de entrenamiento", yaxis_title="% de registros del perfil")
    return estilo_figura(fig, "Distribución de tipos de entrenamiento", 400)


def figura_caja_perfiles(perfiles: dict, col: str) -> go.Figure:
    """Diagrama de cajas de una variable para varios perfiles."""
    colores = [cfg.COLOR_NARANJA, cfg.COLOR_MORADO, "#9A9A9A"]
    fig = go.Figure()
    for (nombre, datos), color in zip(perfiles.items(), colores):
        if len(datos) == 0:
            continue
        fig.add_trace(go.Box(y=datos[col], name=f"{nombre} (n = {len(datos)})", marker_color=color,
                             boxmean=True, boxpoints="outliers", line=dict(width=1.5)))
    fig.update_layout(yaxis_title=etiqueta(col))
    return estilo_figura(fig, f"{etiqueta(col)} por perfil", 400, leyenda=False)


# =========================================================================== #
# Exploración personalizada
# =========================================================================== #
def es_numerica(col: str) -> bool:
    """True si la variable se trata como numérica en los análisis."""
    return col in cfg.VARIABLES_NUMERICAS


# Pares que NO tiene sentido graficar (una variable se calcula a partir de la otra).
_COMBINACIONES_NO_VALIDAS = [
    ({"imc", "peso_kg"}, "El IMC se calcula directamente a partir del peso y la estatura; su relación es matemática, no un hallazgo."),
    ({"imc", "estatura_m"}, "El IMC se calcula directamente a partir del peso y la estatura; su relación es matemática, no un hallazgo."),
    ({"categoria_imc", "imc"}, "La categoría de IMC se define a partir del propio IMC; compararlas no aporta información."),
    ({"categoria_imc", "peso_kg"}, "La categoría de IMC depende en parte del peso; compararlas es en gran medida circular."),
    ({"categoria_imc", "estatura_m"}, "La categoría de IMC depende en parte de la estatura; compararlas es en gran medida circular."),
    ({"grupo_edad", "edad"}, "El grupo de edad se define a partir de la propia edad; compararlos no aporta información."),
    ({"calorias", "calorias_por_hora"}, "Las calorías por hora se calculan a partir de las calorías; la relación es en parte matemática."),
]


def validar_combinacion(df, x: str, y: str, color=None):
    """Devuelve un mensaje explicativo si la combinación no tiene sentido estadístico; si es válida, None."""
    if x == y:
        return "Elige dos variables distintas: comparar una variable consigo misma no aporta información."
    if color in (x, y):
        return "La variable de color debe ser distinta de las de los ejes."
    for par, motivo in _COMBINACIONES_NO_VALIDAS:
        if {x, y} == par:
            return motivo
    if not es_numerica(x) and not es_numerica(y) and color:
        return ("Con dos variables categóricas no se usa una tercera para diferenciar grupos. "
                "Elige «Ninguna» en la variable de color.")
    if len(df) < cfg.N_MIN_GRUPO:
        return f"Con los filtros actuales solo quedan {len(df)} registros; amplía la selección."
    return None


def grafico_personalizado(df: pd.DataFrame, x: str, y: str, color=None):
    """
    Elige automáticamente el gráfico más adecuado según los tipos de variable:
      numérica-numérica  -> dispersión con tendencia
      categórica-numérica -> diagrama de caja
      categórica-categórica -> barras apiladas al 100 %
    Devuelve (figura, lectura, mensaje). Si la combinación no es válida, figura=None y mensaje explica por qué.
    """
    mensaje = validar_combinacion(df, x, y, color)
    if mensaje:
        return None, "", mensaje
    titulo = f"{etiqueta(y)} frente a {etiqueta(x)}"
    x_num, y_num = es_numerica(x), es_numerica(y)

    if x_num and y_num:
        datos = df.dropna(subset=[x, y])
        fig = _figura_dispersion(datos, x, y, color_por=color, titulo=titulo)
        return fig, lectura_relacion_numerica(datos, x, y), None

    if (not x_num) and y_num:
        fig = figura_caja(df, y=y, x=x, color_por=color, titulo=titulo)
        grupo = x
        return fig, lectura_diferencia_grupos(df, y, grupo), None

    if x_num and not y_num:
        datos = df.dropna(subset=[x, y])
        fig = px.box(datos, x=x, y=y, color=color or y, orientation="h", points="outliers",
                     color_discrete_map=paleta_para(color or y), labels={x: etiqueta(x), y: etiqueta(y), **({color: etiqueta(color)} if color else {})},
                     category_orders={c: list(datos[c].cat.categories) for c in [y, color] if c and hasattr(datos[c], "cat")})
        fig.update_traces(boxmean=True, marker=dict(size=4, opacity=0.6))
        estilo_figura(fig, titulo, 420, leyenda=bool(color and color != y))
        return fig, lectura_diferencia_grupos(df, x, y), None

    # categórica-categórica
    tabla = pd.crosstab(df[x], df[y])
    porcentajes = tabla.div(tabla.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig = go.Figure()
    paleta = paleta_para(y) or {}
    secuencia = [cfg.COLOR_NARANJA, cfg.COLOR_MORADO, cfg.COLOR_NARANJA_CLARO, cfg.COLOR_MORADO_CLARO, "#9A9A9A"]
    for i, categoria in enumerate(tabla.columns):
        fig.add_trace(go.Bar(
            x=[str(v) for v in tabla.index], y=porcentajes[categoria].fillna(0), name=str(categoria),
            marker_color=paleta.get(str(categoria), secuencia[i % len(secuencia)]),
            marker_line_color=cfg.COLOR_MORADO_OSCURO, marker_line_width=0.4, customdata=tabla[categoria],
            hovertemplate="<b>%{x}</b><br>" + str(categoria) + ": %{y:.1f} % (%{customdata} registros)<extra></extra>"))
    fig.update_layout(barmode="stack", xaxis_title=etiqueta(x), yaxis_title=f"% de registros dentro de cada categoría de {etiqueta_corta(x).lower()}")
    estilo_figura(fig, titulo, 420)
    return fig, lectura_asociacion_categoricas(df, x, y), None
