"""
Gym Analytics · Dashboard de exploración + recomendador de ejercicios.

Ejecutar con:   streamlit run app.py

Estructura:
    app.py                              → interfaz (navegación y las tres secciones)
    modules/data_processing.py          → carga, validación y limpieza del CSV
    modules/exploratory_analysis.py     → filtros, gráficos, comparaciones, correlaciones y perfiles
    modules/recommendation_rules.py     → REGLAS editables del recomendador
    modules/recommendation.py           → motor de puntuación y explicación
    modules/calorie_prediction.py       → estimación de calorías
    modules/ui_components.py            → estilos y tarjetas
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules import calorie_prediction as cp
from modules import config as cfg
from modules import data_processing as dp
from modules import exploratory_analysis as ea
from modules import recommendation as rec
from modules import recommendation_rules as reglas
from modules import ui_components as ui
from modules.formato import numero, p_valor, porcentaje

st.set_page_config(page_title="Gym Analytics · Dashboard y recomendador", page_icon="🏋️",
                   layout="wide", initial_sidebar_state="expanded")

SECCION_INICIO = "🏠 Inicio"
SECCION_EXPLORACION = "📊 Exploración y perfiles"
SECCION_RECOMENDADOR = "🎯 Recomendador de ejercicios"

# Prefijos de las claves de widgets cuyo valor se conserva al cambiar de sección.
NINGUNA = "ninguna"   # valor centinela de los selectores opcionales (evita guardar None en el estado)
PREFIJOS_ESTADO = ("flt_", "cmp_", "cor_", "var_", "perf_", "pers_", "form_", "sim_")


# =========================================================================== #
# Carga de datos (con caché) y utilidades
# =========================================================================== #
@st.cache_data(show_spinner="Cargando y validando los datos…")
def cargar_datos_cacheados(ruta: str, firma: float):
    """Lee y limpia el CSV. `firma` (fecha de modificación) invalida la caché si el archivo cambia."""
    resultado = dp.cargar_datos(ruta)
    return resultado, ea.agregar_grupo_edad(resultado.datos)


@st.cache_resource(show_spinner="Preparando el estimador de calorías…")
def obtener_estimador(_datos: pd.DataFrame, firma: float):
    """Entrena el estimador de calorías una sola vez (el guion bajo evita hashear el DataFrame)."""
    return cp.entrenar_estimador(_datos)


def cargar_todo():
    """Carga datos y estimador mostrando mensajes comprensibles si algo falla."""
    try:
        ruta = dp.localizar_csv()
        firma = ruta.stat().st_mtime
        resultado, datos = cargar_datos_cacheados(str(ruta), firma)
        try:
            estimador = obtener_estimador(datos, firma)
        except ValueError as error:
            estimador = None
            st.warning(f"La estimación de calorías no está disponible: {error}")
        return resultado, datos, estimador
    except dp.ErrorDatos as error:
        st.error(f"**No se pudieron cargar los datos.** {error}")
    except Exception as error:  # último recurso: evitar una pantalla de error técnica
        st.error(f"**Ocurrió un problema inesperado al cargar los datos.** Detalle: {error}")
    st.stop()


def conservar_estado() -> None:
    """Evita que Streamlit olvide los valores de los widgets de una sección cuando se visita otra."""
    for clave in list(st.session_state.keys()):
        if isinstance(clave, str) and clave.startswith(PREFIJOS_ESTADO):
            st.session_state[clave] = st.session_state[clave]


def mostrar_figura(fig, clave: str) -> None:
    """Muestra una figura de Plotly con la configuración común."""
    st.plotly_chart(fig, width="stretch", theme=None, key=clave, config={"displaylogo": False})


def mostrar_tabla(tabla: pd.DataFrame) -> None:
    st.dataframe(tabla, hide_index=True, width="stretch")


# =========================================================================== #
# SECCIÓN 1 · INICIO
# =========================================================================== #
def pagina_inicio(df: pd.DataFrame, resultado: dp.ResultadoProcesamiento) -> None:
    ui.encabezado(
        "Gym Analytics",
        "Explora los datos de los miembros de un gimnasio, compara perfiles y recibe recomendaciones de entrenamiento "
        "explicables con una estimación de las calorías de tu sesión.",
        chips=["📊 Exploración y perfiles", "🎯 Recomendador de ejercicios", "🔥 Estimación de calorías"])

    st.subheader("Indicadores principales")
    columnas = st.columns(5)
    for columna, indicador in zip(columnas, ea.calcular_indicadores(df)):
        with columna:
            ui.tarjeta_kpi(indicador["nombre"], indicador["valor"], indicador["unidad"], indicador["mensaje"])
    st.caption("Todos los indicadores se calculan automáticamente a partir del archivo cargado.")

    st.info("**Cómo usar este dashboard.** En **Exploración y perfiles** puedes filtrar, comparar variables y analizar perfiles de "
            "miembros. En **Recomendador de ejercicios** introduces tus datos y obtienes sugerencias con su explicación. "
            "Todo lo que se muestra describe los registros del conjunto de datos: no son consejos médicos.")

    st.subheader("Visión general de los miembros")
    textos = ea.textos_inicio(df)
    col1, col2 = st.columns(2)
    with col1:
        mostrar_figura(ea.figura_histograma(df, "edad", "Distribución de edades", nbins=14), "ini_edad")
        st.caption(textos["edad"])
    with col2:
        mostrar_figura(ea.figura_barras_categoria(df, "tipo_entrenamiento", "Tipos de entrenamiento registrados"), "ini_tipos")
        st.caption(textos["tipos"])
    col3, col4 = st.columns(2)
    with col3:
        mostrar_figura(ea.figura_barras_categoria(df, "nivel_experiencia", "Niveles de experiencia"), "ini_niveles")
        st.caption(textos["niveles"])
    with col4:
        mostrar_figura(ea.figura_histograma(df, "calorias", "Distribución de calorías registradas", nbins=30,
                                            color=cfg.COLOR_MORADO), "ini_calorias")
        st.caption(textos["calorias"])

    with st.expander("🔍 Calidad de los datos y transformaciones aplicadas"):
        st.markdown(f"Se leyeron **{numero(resultado.n_original, 0)}** registros y tras las validaciones quedan "
                    f"**{numero(resultado.n_final, 0)}**. No se eliminan automáticamente los valores extremos: solo se excluyen "
                    "los imposibles; los extremos que podrían ser reales se conservan y se informan.")
        mostrar_tabla(resultado.informe)
        st.markdown("**Transformaciones realizadas**")
        for transformacion in resultado.transformaciones:
            st.markdown(f"- {transformacion}")
        if resultado.columnas_ignoradas:
            st.markdown("- Columnas del CSV no utilizadas: " + ", ".join(resultado.columnas_ignoradas) + ".")


# =========================================================================== #
# SECCIÓN 2 · EXPLORACIÓN Y PERFILES
# =========================================================================== #
def valores_defecto_filtros(df: pd.DataFrame) -> dict:
    rangos = ea.rangos_filtros(df)
    return {"flt_edad": rangos["edad"], "flt_genero": list(cfg.GENEROS), "flt_nivel": list(cfg.ORDEN_NIVELES),
            "flt_tipo": list(cfg.TIPOS_ENTRENAMIENTO), "flt_imc": rangos["imc"], "flt_duracion": rangos["duracion"]}


def restablecer_valores(valores: dict) -> None:
    """Callback de los botones «Restablecer»: devuelve los widgets a sus valores iniciales."""
    for clave, valor in valores.items():
        st.session_state[clave] = list(valor) if isinstance(valor, list) else valor


def barra_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """Filtros de la sección de exploración (barra lateral). Devuelve el DataFrame filtrado."""
    rangos = ea.rangos_filtros(df)
    defectos = valores_defecto_filtros(df)
    for clave, valor in defectos.items():
        st.session_state.setdefault(clave, list(valor) if isinstance(valor, list) else valor)

    st.markdown("### Filtros")
    st.slider("Edad (años)", rangos["edad"][0], rangos["edad"][1], key="flt_edad")
    st.multiselect("Género", cfg.GENEROS, key="flt_genero")
    st.multiselect("Nivel de experiencia", cfg.ORDEN_NIVELES, key="flt_nivel")
    st.multiselect("Tipo de entrenamiento", cfg.TIPOS_ENTRENAMIENTO, key="flt_tipo")
    st.slider("IMC (kg/m²)", rangos["imc"][0], rangos["imc"][1], step=0.1, key="flt_imc")
    st.slider("Duración de la sesión (min)", rangos["duracion"][0], rangos["duracion"][1], key="flt_duracion")
    st.button("↺ Restablecer filtros", on_click=restablecer_valores, args=(defectos,), width="stretch", key="btn_reset_filtros")

    filtrado = ea.aplicar_filtros(
        df, edad=st.session_state["flt_edad"], generos=st.session_state["flt_genero"],
        niveles=st.session_state["flt_nivel"], tipos=st.session_state["flt_tipo"],
        imc=st.session_state["flt_imc"], duracion=st.session_state["flt_duracion"])
    st.markdown(f"**{numero(len(filtrado), 0)}** de {numero(len(df), 0)} registros cumplen los filtros "
                f"({porcentaje(len(filtrado) / len(df))}).")
    return filtrado


def tab_variables(df: pd.DataFrame) -> None:
    st.caption("Elige una variable para ver su distribución, sus estadísticas y su tabla de frecuencias.")
    tipo = st.radio("Tipo de variable", ["Numérica", "Categórica"], horizontal=True, key="var_tipo")
    if tipo == "Numérica":
        variable = st.selectbox("Variable numérica", cfg.VARIABLES_NUMERICAS, format_func=ea.etiqueta, key="var_numerica")
        separar = st.selectbox("Separar por (opcional)", [NINGUNA, "tipo_entrenamiento", "nivel_experiencia", "genero"],
                               format_func=lambda c: "Sin separar" if c == NINGUNA else ea.etiqueta(c), key="var_separar")
        separar = None if separar == NINGUNA else separar
        col1, col2 = st.columns(2)
        with col1:
            mostrar_figura(ea.figura_histograma(df, variable, f"Histograma · {ea.etiqueta(variable)}", color_por=separar), "var_hist")
            st.caption("El histograma muestra cuántos registros caen en cada intervalo de valores.")
        with col2:
            mostrar_figura(ea.figura_caja(df, variable, x=separar, titulo=f"Diagrama de caja · {ea.etiqueta(variable)}"), "var_caja")
            st.caption("La caja contiene el 50 % central; la línea continua es la mediana y la discontinua la media. Los puntos son valores atípicos.")
        st.markdown("**Estadísticas descriptivas**")
        if separar:
            agrupado = df.groupby(separar, observed=True)[variable].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
            agrupado.columns = ["Registros", "Media", "Mediana", "Desv. estándar", "Mínimo", "Máximo"]
            mostrar_tabla(agrupado.reset_index().rename(columns={separar: ea.etiqueta(separar)}))
        else:
            mostrar_tabla(ea.estadisticas_descriptivas(df, variable))
        faltantes = int(df[variable].isna().sum())
        if faltantes:
            st.caption(f"ℹ️ {faltantes} registros no tienen valor válido en esta variable (se excluyeron al marcarse como incoherentes).")
    else:
        variable = st.selectbox("Variable categórica", cfg.VARIABLES_CATEGORICAS, format_func=ea.etiqueta, key="var_categorica")
        col1, col2 = st.columns([3, 2])
        with col1:
            mostrar_figura(ea.figura_barras_categoria(df, variable, f"Gráfico de barras · {ea.etiqueta(variable)}"), "var_barras")
            st.caption("Cada barra muestra el número de registros y su porcentaje sobre el total de la selección.")
        with col2:
            st.markdown("**Tabla de frecuencias**")
            mostrar_tabla(ea.tabla_frecuencias(df, variable))
        if variable == "categoria_imc":
            st.caption("Categorías convencionales para adultos (orientativas): bajo peso < 18,5; normal 18,5 a < 25; "
                       "sobrepeso 25 a < 30; obesidad ≥ 30.")
        if variable == "grupo_edad":
            st.caption("Grupos con puntos de corte por defecto (30, 40 y 50 años). Puedes configurarlos en la pestaña «Perfiles».")


def tab_comparaciones(df: pd.DataFrame) -> None:
    st.caption("Elige una comparación. Pasa el cursor sobre los gráficos para ver los valores de cada registro o grupo.")
    clave = st.selectbox("Comparación", list(ea.COMPARACIONES), format_func=lambda k: ea.COMPARACIONES[k]["titulo"], key="cmp_clave")
    st.session_state.setdefault("cmp_tipos", list(cfg.TIPOS_ENTRENAMIENTO))
    st.session_state.setdefault("cmp_niveles", list(cfg.ORDEN_NIVELES))
    col1, col2 = st.columns(2)
    with col1:
        tipos = st.multiselect("Filtrar por tipo de entrenamiento", cfg.TIPOS_ENTRENAMIENTO, key="cmp_tipos")
    with col2:
        niveles = st.multiselect("Filtrar por nivel de experiencia", cfg.ORDEN_NIVELES, key="cmp_niveles")
    datos = df[df["tipo_entrenamiento"].isin(tipos) & df["nivel_experiencia"].isin(niveles)]
    st.caption(f"{ea.COMPARACIONES[clave]['descripcion']}  ·  Registros usados: **{numero(len(datos), 0)}**")
    figura, lectura, mensaje = ea.construir_comparacion(datos, clave)
    if mensaje:
        st.warning(mensaje)
        return
    mostrar_figura(figura, f"cmp_{clave}")
    st.info(f"**Lectura de los datos.** {lectura}")


def tab_correlaciones(df: pd.DataFrame) -> None:
    st.caption("La matriz muestra qué variables numéricas tienden a variar juntas. Una correlación NO implica causalidad.")
    col1, col2 = st.columns([1, 3])
    with col1:
        metodo_texto = st.radio("Método de correlación", ["Pearson", "Spearman"], key="cor_metodo",
                                help="Pearson mide relaciones lineales. Spearman usa rangos: es más robusta ante valores extremos y es "
                                     "la opción más adecuada para variables ordinales como el nivel de experiencia o la frecuencia semanal.")
    with col2:
        st.session_state.setdefault("cor_variables", list(cfg.VARIABLES_CORRELACION))
        variables = st.multiselect("Variables incluidas", cfg.VARIABLES_CORRELACION, format_func=ea.etiqueta, key="cor_variables")
    metodo = metodo_texto.lower()
    corr = ea.matriz_correlacion(df, variables, metodo)
    if corr is None:
        st.warning("Selecciona al menos 2 variables que tengan variación y asegúrate de que queden 10 registros o más con los filtros.")
    else:
        mostrar_figura(ea.figura_matriz_correlacion(corr, metodo), "cor_matriz")
        st.markdown("**Pares con mayor correlación (en valor absoluto)**")
        mostrar_tabla(ea.pares_mas_correlacionados(corr, 6))
    with st.expander("ℹ️ Limitaciones y variables excluidas"):
        st.markdown(
            "- **Correlación ≠ causalidad.** Que dos variables varíen juntas no significa que una cause la otra.\n"
            "- **Excluidas de la matriz:** género y tipo de entrenamiento (categorías sin orden numérico), categoría de IMC y "
            "grupo de edad (derivadas de otras variables), calorías por hora (se calcula con calorías y duración) y la duración en "
            "horas (duplica la duración en minutos).\n"
            "- **IMC, peso y estatura:** el IMC se calcula con el peso y la estatura, por lo que su correlación es en parte matemática.\n"
            "- **Nivel de experiencia y frecuencia semanal** son variables ordinales/discretas: Spearman es más apropiado que Pearson.\n"
            "- Los pares se calculan con los registros que tienen ambos datos (algunas frecuencias cardíacas se marcaron como faltantes).\n"
            "- Con muchas variables, algunas correlaciones débiles pueden aparecer por azar; con pocos registros son poco fiables.")


def _selector_perfil(prefijo: str, titulo: str, grupos_edad: list) -> dict:
    """Controles de selección de un perfil. Devuelve las selecciones (lista vacía = todos)."""
    st.markdown(f"**{titulo}**")
    for sufijo, opciones in [("edad", grupos_edad), ("imc", cfg.CATEGORIAS_IMC), ("nivel", cfg.ORDEN_NIVELES), ("tipo", cfg.TIPOS_ENTRENAMIENTO)]:
        clave = f"perf_{prefijo}_{sufijo}"
        st.session_state.setdefault(clave, [])
        st.session_state[clave] = [v for v in st.session_state[clave] if v in opciones]  # descarta opciones que ya no existen
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        grupos = st.multiselect("Grupo de edad", grupos_edad, key=f"perf_{prefijo}_edad", placeholder="Todos")
    with c2:
        imc = st.multiselect("Categoría de IMC", cfg.CATEGORIAS_IMC, key=f"perf_{prefijo}_imc", placeholder="Todas")
    with c3:
        niveles = st.multiselect("Nivel de experiencia", cfg.ORDEN_NIVELES, key=f"perf_{prefijo}_nivel", placeholder="Todos")
    with c4:
        tipos = st.multiselect("Tipo de entrenamiento", cfg.TIPOS_ENTRENAMIENTO, key=f"perf_{prefijo}_tipo", placeholder="Todos")
    return {"grupos_edad": grupos, "categorias_imc": imc, "niveles": niveles, "tipos": tipos}


def _mostrar_resumen_perfil(nombre: str, resumen: dict) -> None:
    st.markdown(f"**{nombre}** · {numero(resumen['n'], 0)} registros ({porcentaje(resumen['pct_total'])} del total)")
    st.markdown("Distribución de los tipos de entrenamiento")
    mostrar_tabla(resumen["distribucion_tipos"])
    st.caption(resumen["lectura_tipos"])
    st.markdown("Edad, IMC, duración y calorías frente al resto del conjunto")
    mostrar_tabla(resumen["medidas"])


def tab_perfiles(df_completo: pd.DataFrame) -> None:
    st.caption("Define uno o dos perfiles de miembros y compáralos entre sí y con el conjunto completo. "
               "Una selección vacía significa «todos».")
    st.info("Los perfiles se calculan sobre el **conjunto completo de datos**, independientemente de los filtros de la barra lateral. "
            "Los patrones observados son descriptivos: no son recomendaciones médicas ni prueban que un entrenamiento sea superior a otro.")

    posibles = ea.opciones_cortes_edad(df_completo)
    st.session_state.setdefault("perf_cortes", [c for c in (30, 40, 50) if c in posibles])
    st.session_state["perf_cortes"] = [c for c in st.session_state["perf_cortes"] if c in posibles]
    cortes = st.multiselect("Puntos de corte de los grupos de edad (cada corte inicia un grupo nuevo)", posibles, key="perf_cortes",
                            help="Solo se ofrecen cortes dentro del rango de edades observado en los datos.")
    datos = df_completo.copy()
    datos["grupo_edad"] = ea.crear_grupos_edad(datos["edad"], cortes)
    grupos_edad = list(datos["grupo_edad"].cat.categories)
    st.caption(f"Grupos de edad resultantes: {', '.join(grupos_edad)} años (rango observado: "
               f"{int(datos['edad'].min())}–{int(datos['edad'].max())}).")

    comparar = st.toggle("Comparar con un segundo perfil (B)", key="perf_comparar")
    defectos_perfiles = {f"perf_{p}_{s}": [] for p in "AB" for s in ("edad", "imc", "nivel", "tipo")}
    seleccion_a = _selector_perfil("A", "Perfil A", grupos_edad)
    seleccion_b = _selector_perfil("B", "Perfil B", grupos_edad) if comparar else None
    st.button("↺ Restablecer perfiles", on_click=restablecer_valores, args=(defectos_perfiles,), key="btn_reset_perfiles")

    perfil_a = ea.filtrar_perfil(datos, **seleccion_a)
    perfil_b = ea.filtrar_perfil(datos, **seleccion_b) if seleccion_b is not None else None
    if len(perfil_a) == 0 or (perfil_b is not None and len(perfil_b) == 0):
        st.warning("Alguno de los perfiles seleccionados no tiene registros. Amplía la selección (por ejemplo, quita algún criterio).")
        return

    def variables_definidas(seleccion):
        return tuple(v for v, usada in (("edad", seleccion["grupos_edad"]), ("imc", seleccion["categorias_imc"])) if usada)

    perfiles = {"Perfil A": perfil_a}
    if perfil_b is not None:
        perfiles["Perfil B"] = perfil_b
    perfiles["Conjunto completo"] = datos

    columnas_kpi = st.columns(len(perfiles))
    for columna, (nombre, datos_perfil) in zip(columnas_kpi, perfiles.items()):
        with columna:
            ui.tarjeta_kpi(f"{nombre} · registros", numero(len(datos_perfil), 0), f"({porcentaje(len(datos_perfil) / len(datos))} del total)")

    col1, col2 = st.columns(2)
    with col1:
        mostrar_figura(ea.figura_distribucion_tipos_perfiles(perfiles), "perf_tipos")
        st.caption("Porcentaje de registros de cada perfil que realizó cada tipo de entrenamiento.")
    with col2:
        variable = st.selectbox("Variable a comparar", ["edad", "imc", "duracion_min", "calorias"], format_func=ea.etiqueta, key="perf_variable")
        mostrar_figura(ea.figura_caja_perfiles(perfiles, variable), "perf_caja")

    resumen_a = ea.resumir_perfil(perfil_a, datos, variables_definidas(seleccion_a))
    st.markdown("#### Detalle de los perfiles")
    if perfil_b is None:
        _mostrar_resumen_perfil("Perfil A", resumen_a)
    else:
        resumen_b = ea.resumir_perfil(perfil_b, datos, variables_definidas(seleccion_b))
        pestana_a, pestana_b = st.tabs(["Perfil A", "Perfil B"])
        with pestana_a:
            _mostrar_resumen_perfil("Perfil A", resumen_a)
        with pestana_b:
            _mostrar_resumen_perfil("Perfil B", resumen_b)
    st.caption("Las diferencias se comparan con los registros que NO pertenecen al perfil (prueba de Mann-Whitney) y solo cuando hay "
               f"al menos {cfg.N_MIN_COMPARACION} registros dentro y fuera del perfil. «Detectable» no significa «importante»: revisa la "
               "magnitud de la diferencia. Al hacer varias comparaciones, alguna puede salir significativa por azar.")
    with st.expander("ℹ️ Categorías de IMC y precauciones de interpretación"):
        st.markdown("Categorías convencionales para adultos: **bajo peso** (IMC < 18,5), **peso normal** (18,5 a < 25), "
                    "**sobrepeso** (25 a < 30) y **obesidad** (≥ 30). Son **orientativas**: el IMC no mide la grasa corporal ni la salud, "
                    "y por sí solo no determina la capacidad física ni la seguridad de una actividad.")


def tab_personalizada(df: pd.DataFrame) -> None:
    st.caption("Elige las variables y la aplicación generará el gráfico más apropiado: dispersión para dos numéricas, "
               "cajas para categórica + numérica y barras apiladas para dos categóricas.")
    opciones = cfg.VARIABLES_NUMERICAS + cfg.VARIABLES_CATEGORICAS
    defectos = {"pers_x": "duracion_min", "pers_y": "calorias", "pers_color": NINGUNA}
    for clave, valor in defectos.items():
        st.session_state.setdefault(clave, valor)
    col1, col2, col3 = st.columns(3)
    with col1:
        x = st.selectbox("Eje X", opciones, format_func=ea.etiqueta, key="pers_x")
    with col2:
        y = st.selectbox("Eje Y", opciones, format_func=ea.etiqueta, key="pers_y")
    with col3:
        color = st.selectbox("Diferenciar grupos (opcional)", [NINGUNA] + cfg.VARIABLES_CATEGORICAS,
                             format_func=lambda c: "Ninguna" if c == NINGUNA else ea.etiqueta(c), key="pers_color")
        color = None if color == NINGUNA else color
    st.button("↺ Restablecer selección", on_click=restablecer_valores, args=(defectos,), key="btn_reset_pers")
    figura, lectura, mensaje = ea.grafico_personalizado(df, x, y, color)
    if mensaje:
        st.warning(mensaje)
        return
    mostrar_figura(figura, "pers_grafico")
    st.info(f"**Lectura de los datos.** {lectura}")


def pagina_exploracion(df: pd.DataFrame, filtrado: pd.DataFrame) -> None:
    ui.encabezado("Exploración y perfiles", "Filtra los datos, analiza variables, compara grupos y estudia perfiles de miembros. "
                  "Los filtros de la barra lateral actualizan automáticamente los indicadores, gráficos y tablas de esta sección.")
    if len(filtrado) == 0:
        st.warning("**Ningún registro cumple los filtros seleccionados.** Amplía los rangos, vuelve a marcar alguna opción "
                   "o usa «Restablecer filtros» en la barra lateral.")
        return
    st.markdown(f"##### Indicadores de la selección · {numero(len(filtrado), 0)} registros")
    columnas = st.columns(5)
    for columna, indicador in zip(columnas, ea.calcular_indicadores(filtrado)):
        with columna:
            ui.tarjeta_kpi(indicador["nombre"], indicador["valor"], indicador["unidad"], indicador["mensaje"])
    st.write("")
    pestanas = st.tabs(["📈 Variables", "🔀 Comparaciones", "🧮 Correlaciones", "👥 Perfiles", "🧪 Exploración personalizada"])
    with pestanas[0]:
        tab_variables(filtrado)
    with pestanas[1]:
        tab_comparaciones(filtrado)
    with pestanas[2]:
        tab_correlaciones(filtrado)
    with pestanas[3]:
        tab_perfiles(df)
    with pestanas[4]:
        tab_personalizada(filtrado)


# =========================================================================== #
# SECCIÓN 3 · RECOMENDADOR DE EJERCICIOS
# =========================================================================== #
def formulario_usuario(df: pd.DataFrame, estimador) -> None:
    """Formulario con los datos del usuario. Al enviarlo guarda el resultado en la sesión."""
    with st.form("form_recomendador"):
        st.markdown("#### 1 · Tus datos")
        col1, col2, col3 = st.columns(3)
        with col1:
            edad = st.number_input("Edad (años)", min_value=0, max_value=120, value=None, step=1, placeholder="Ej.: 32", key="form_edad")
        with col2:
            genero = st.selectbox("Género", ["Femenino", "Masculino", "Prefiero no indicarlo"], index=None, placeholder="Selecciona…",
                                  key="form_genero", help="Solo se usa para ajustar la estimación de calorías; es opcional.")
        with col3:
            peso = st.number_input("Peso (kg)", min_value=0.0, max_value=500.0, value=None, step=0.5, format="%.1f",
                                   placeholder="Ej.: 68,5", key="form_peso")
        col4, col5 = st.columns(2)
        with col4:
            estatura_texto = st.text_input("Estatura", placeholder="Ej.: 172 cm  o  1,72 m", key="form_estatura",
                                           help="Puedes escribirla en centímetros (172) o en metros (1,72).")
        with col5:
            st.session_state.setdefault("form_duracion", 60)   # valor inicial (evita conflicto con el estado conservado)
            duracion = st.number_input("Duración deseada de la sesión (min)", min_value=0, max_value=600, step=5,
                                       key="form_duracion",
                                       help=f"Los registros del conjunto de datos cubren sesiones de "
                                            f"{numero(df['duracion_min'].min(), 0)} a {numero(df['duracion_min'].max(), 0)} minutos; "
                                            "fuera de ese rango no se estiman calorías.")

        st.markdown("#### 2 · Tu experiencia y tu objetivo")
        col6, col7 = st.columns(2)
        with col6:
            nivel = st.selectbox("Nivel de experiencia", cfg.ORDEN_NIVELES, index=None, placeholder="Selecciona…",
                                 format_func=lambda n: reglas.DESCRIPCION_NIVELES[n], key="form_nivel")
            st.caption(reglas.RESUMEN_NIVELES)
        with col7:
            objetivo = st.selectbox("Objetivo principal", list(reglas.OBJETIVOS), index=None, placeholder="Selecciona…",
                                    format_func=lambda o: reglas.OBJETIVOS[o], key="form_objetivo")

        st.markdown("#### 3 · Precauciones (opcional)")
        st.caption("Estas preguntas solo sirven para adaptar la recomendación con prudencia. No es necesario dar detalles médicos.")
        col8, col9 = st.columns(2)
        with col8:
            limitacion = st.selectbox("¿Tienes actualmente alguna lesión, dolor o limitación de movilidad?", list(reglas.OPCIONES_LIMITACION),
                                      format_func=lambda k: reglas.OPCIONES_LIMITACION[k], key="form_limitacion")
        with col9:
            restriccion = st.selectbox("¿Algún profesional de la salud te ha indicado precauciones o restricciones para hacer ejercicio?",
                                       list(reglas.OPCIONES_RESTRICCION), format_func=lambda k: reglas.OPCIONES_RESTRICCION[k],
                                       key="form_restriccion")
        enviado = st.form_submit_button("Obtener recomendaciones", type="primary")

    frecuencias = rec.frecuencias_por_nivel(df)
    with st.expander("ℹ️ Sobre el nivel de experiencia y el conjunto de datos"):
        st.markdown(
            "Las descripciones del formulario (0–1, 2–4 y 5 o más veces por semana) las usa la persona para elegir su nivel. "
            "En el CSV, el nivel viene codificado como 1, 2 y 3 sin diccionario de datos, y se **asume** que equivale a "
            "principiante, intermedio y avanzado. Las frecuencias semanales que se observan en el propio CSV son otras:")
        mostrar_tabla(pd.DataFrame([{"Nivel": n, "Frecuencia semanal observada en el CSV": f"{a} a {b} días"} for n, (a, b) in frecuencias.items()]))
        st.caption("Por esa diferencia, el nivel que eliges se usa para comparar con los registros del mismo nivel, no como una equivalencia exacta de frecuencia.")

    if enviado:
        estatura_m, error_estatura = rec.parsear_estatura(estatura_texto)
        datos = rec.DatosUsuario(
            edad=None if edad is None else float(edad), genero=None if genero in (None, "Prefiero no indicarlo") else genero,
            peso_kg=None if peso is None else float(peso), estatura_m=estatura_m, nivel=nivel, objetivo=objetivo,
            duracion_min=None if duracion is None else float(duracion), limitacion=limitacion, restriccion_profesional=restriccion)
        resultado = rec.recomendar(datos, df, estimador)
        st.session_state["resultado_reco"] = (resultado, datos, error_estatura)
        st.session_state.pop("sim_tipo", None)
        if estimador is not None and datos.duracion_min:
            minimo, maximo = int(np.ceil(estimador.duracion_min_obs_h * 60)), int(np.floor(estimador.duracion_max_obs_h * 60))
            st.session_state["sim_duracion"] = int(min(max(5 * round(datos.duracion_min / 5), minimo), maximo))


def bloque_calorias(r: rec.Recomendacion, estimador) -> None:
    """Estimación de calorías dentro de una tarjeta de recomendación."""
    c = r.calorias
    if c is None:
        st.caption("La estimación de calorías no está disponible.")
        return
    if c.estado != "ok":
        st.caption("⚠️ Sin estimación de calorías (ver el aviso sobre estas tarjetas).")
        return
    st.markdown(f'<div class="kcal-grande">≈ {numero(c.base_kcal, 0)} kcal</div>'
                f'<div class="kcal-detalle">Rango habitual: {numero(c.base_rango[0], 0)}–{numero(c.base_rango[1], 0)} kcal</div>',
                unsafe_allow_html=True)
    if c.refinada_aplicada:
        st.markdown(f"Con ajuste por edad, género y peso: **≈ {numero(c.refinada_kcal, 0)} kcal** "
                    f"({numero(c.refinada_rango[0], 0)}–{numero(c.refinada_rango[1], 0)} kcal)")
    elif c.motivo_sin_refinada:
        st.caption(c.motivo_sin_refinada)
    with st.expander("Método y limitaciones de la estimación"):
        mediana = estimador.tabla_por_tipo.loc[r.tipo, "mediana"]
        st.markdown(f"**Método:** mediana de calorías por hora de {numero(c.n_registros_tipo, 0)} registros de {r.tipo} "
                    f"({numero(mediana, 0)} kcal/h) × {numero(r.duracion_min / 60, 2)} h. "
                    "El ajuste por edad, género y peso es una regresión lineal complementaria.")
        for advertencia in c.advertencias:
            st.markdown(f"- {advertencia}")


def tarjeta_recomendacion(r: rec.Recomendacion, posicion: int, estimador) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="reco-rango">OPCIÓN {posicion}</div><div class="reco-titulo">{r.tipo}</div>', unsafe_allow_html=True)
        st.markdown(ui.barra_puntuacion(r.puntuacion)
                    + f'<div class="score-texto"><b>Compatibilidad {r.compatibilidad.lower()}</b> · {numero(r.puntuacion, 0)}/100</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Nivel inicial sugerido:** {r.nivel_inicial}")
        st.markdown(f"**Duración seleccionada:** {numero(r.duracion_min, 0)} min · **Intensidad típica:** {r.intensidad}")
        for aviso in r.avisos:
            st.warning(aviso)
        st.markdown("**Calorías estimadas**")
        bloque_calorias(r, estimador)
        with st.expander("¿Por qué esta recomendación?", expanded=True):
            for motivo in r.motivos:
                st.markdown(f"- {ui.insignia(motivo['origen'])} {motivo['texto']}", unsafe_allow_html=True)


def tabla_componentes(recomendaciones: list) -> pd.DataFrame:
    filas = []
    for r in recomendaciones:
        for c in r.componentes:
            filas.append({"Actividad": r.tipo, "Componente": c["nombre"], "Origen": c["origen"],
                          "Puntos (0-100)": numero(c["puntos"], 0) if c["puntos"] is not None else "No aplicado",
                          "Peso": numero(c["peso"], 2)})
    return pd.DataFrame(filas)


def seccion_secundaria_recomendador(resultado: rec.ResultadoRecomendacion, datos: rec.DatosUsuario, df: pd.DataFrame, estimador) -> None:
    pestanas = st.tabs(["⏱️ Simular otra duración", "📐 Base de la estimación de calorías", "🧮 Cómo se calculó la puntuación"])
    with pestanas[0]:
        if estimador is None:
            st.info("La estimación de calorías no está disponible.")
        else:
            tipos = [r.tipo for r in resultado.recomendaciones]
            if st.session_state.get("sim_tipo") not in tipos:
                st.session_state["sim_tipo"] = tipos[0]
            minimo, maximo = int(np.ceil(estimador.duracion_min_obs_h * 60)), int(np.floor(estimador.duracion_max_obs_h * 60))
            st.session_state.setdefault("sim_duracion", min(max(60, minimo), maximo))
            col1, col2 = st.columns(2)
            with col1:
                tipo = st.selectbox("Actividad", tipos, key="sim_tipo")
            with col2:
                simulada = st.slider("Duración a simular (min)", minimo, maximo, step=5, key="sim_duracion")
            estimacion = cp.estimar_calorias(estimador, tipo, simulada, datos.edad, datos.genero, datos.peso_kg)
            if estimacion.estado == "ok":
                st.markdown(f"**{tipo} · {simulada} min → ≈ {numero(estimacion.base_kcal, 0)} kcal** "
                            f"(rango habitual {numero(estimacion.base_rango[0], 0)}–{numero(estimacion.base_rango[1], 0)} kcal)")
            else:
                st.warning(estimacion.mensaje)
            mostrar_figura(cp.figura_estimacion_duracion(estimador, df, tipo, simulada, datos.edad, datos.genero, datos.peso_kg), "sim_figura")
            st.caption("Los puntos son sesiones reales del conjunto de datos; la línea es la estimación y la banda cubre aproximadamente "
                       f"el 80 % central. Solo se estima dentro del rango de duraciones observado ({minimo}–{maximo} min).")
    with pestanas[1]:
        pagina_base_calorias(df, estimador)
    with pestanas[2]:
        st.caption("La puntuación es un promedio ponderado de los componentes que se pudieron evaluar; los pesos se reescalan si un "
                   "componente no aplica. Los componentes de «Criterio de diseño» son reglas editables en "
                   "`modules/recommendation_rules.py`; los de «Datos» se calculan con el conjunto de datos de forma descriptiva.")
        mostrar_tabla(tabla_componentes(resultado.recomendaciones))
        perfil = resultado.perfil_similar
        if perfil:
            st.markdown(f"**Perfiles similares en los datos:** {numero(perfil.get('n', 0), 0)} registros ({perfil.get('criterio', '')}).")


def pagina_base_calorias(df: pd.DataFrame, estimador) -> None:
    """Análisis previo de calorías (tipo, duración, peso, edad, género, experiencia) y métricas del método."""
    analisis = cp.analizar_calorias(df)
    if "por_tipo" not in analisis or estimador is None:
        st.info("No hay suficientes registros válidos para describir la base de la estimación.")
        return
    validacion = analisis["validacion"]
    st.markdown(f"**Registros utilizados:** {numero(validacion['n_validos'], 0)} de {numero(validacion['n_total'], 0)} "
                f"(calorías y duración positivas y calorías por hora entre {numero(validacion['rango_plausible'][0], 0)} y "
                f"{numero(validacion['rango_plausible'][1], 0)}; observado: {numero(validacion['rango_kcal_h_observado'][0], 0)}–"
                f"{numero(validacion['rango_kcal_h_observado'][1], 0)} kcal/h, coherente con kcal y horas).")
    col1, col2 = st.columns([3, 2])
    with col1:
        mostrar_figura(cp.figura_calorias_hora_por_tipo(estimador), "cal_tipo")
    with col2:
        tabla = analisis["por_tipo"].reset_index().rename(columns={"tipo_entrenamiento": "Tipo", "n": "Registros", "mediana": "Mediana kcal/h",
                                                                   "p10": "P10", "p90": "P90"})[["Tipo", "Registros", "Mediana kcal/h", "P10", "P90"]]
        tabla["Tipo"] = tabla["Tipo"].astype(str)
        mostrar_tabla(tabla.round(0))
        st.caption(f"Entre tipos, las calorías por hora **no difieren de forma estadísticamente clara** (Kruskal-Wallis, p {p_valor(analisis['p_valor_tipos'])})."
                   if analisis["p_valor_tipos"] >= 0.05 else
                   f"Entre tipos, las calorías por hora difieren (Kruskal-Wallis, p {p_valor(analisis['p_valor_tipos'])}).")
    linealidad = analisis["linealidad"]
    st.markdown("**¿Se puede escalar «calorías por hora» a otra duración?** "
                + ("Sí, dentro del rango observado." if linealidad["valida"] else "No con estos datos.")
                + f" La correlación entre calorías por hora y duración es {numero(linealidad['correlacion_pearson'], 2)} (casi nula); "
                  f"el ajuste calorías~duración tiene un intercepto de {numero(linealidad['intercepto_kcal'], 0)} kcal (prácticamente 0) "
                  f"y una pendiente de {numero(linealidad['pendiente_kcal_por_hora'], 0)} kcal/h.")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Por género**")
        mostrar_tabla(analisis["por_genero"].reset_index().rename(columns={"genero": "Género"}).round(0).astype({"Género": str}))
        st.caption(f"Diferencia entre géneros en kcal/h: Mann-Whitney, p {p_valor(analisis['p_valor_genero'])}.")
    with col4:
        st.markdown("**Por nivel de experiencia**")
        mostrar_tabla(analisis["por_experiencia"].reset_index().rename(columns={"nivel_experiencia": "Nivel"}).round(0).astype({"Nivel": str}))
        st.caption("El nivel se asocia con la duración de la sesión (y por eso con las calorías totales), no con las calorías por hora.")
    st.markdown("**Asociación de las calorías por hora con otras variables (Spearman)**")
    mostrar_tabla(analisis["asociaciones_kcal_h"])
    m = estimador.metricas_cv
    st.markdown(
        "**Precisión de los métodos** (validación cruzada de 5 particiones, calculada al cargar los datos): "
        f"el método principal (mediana por tipo × horas) tiene un error absoluto medio de **{numero(m['mae_principal'], 0)} kcal** "
        f"(R² = {numero(m['r2_principal'], 2)}); el ajuste complementario por edad, género y peso lo reduce a "
        f"**{numero(m['mae_complementario'], 0)} kcal** (R² = {numero(m['r2_complementario'], 2)}). "
        "Aun así, una sesión concreta puede alejarse bastante de la estimación.")


def mostrar_resultado(df: pd.DataFrame, estimador) -> None:
    """Muestra el resultado guardado en la sesión tras enviar el formulario."""
    guardado = st.session_state.get("resultado_reco")
    if guardado is None:
        st.info("Completa el formulario y pulsa **Obtener recomendaciones**.")
        return
    resultado, datos, error_estatura = guardado
    st.divider()
    st.markdown("### Tus recomendaciones")

    errores = list(resultado.validacion.errores) if resultado.validacion else []
    if error_estatura:
        errores.append(error_estatura)
    if resultado.estado == "invalido" or error_estatura:
        st.error("**Revisa estos datos antes de continuar:**\n\n" + "\n".join(f"- {e}" for e in errores))
        return
    if resultado.estado == "faltan_datos":
        st.warning(f"**No se puede personalizar todavía.** {resultado.mensaje}")
        return

    # Advertencias y precauciones
    precaucion = resultado.precaucion
    if precaucion["nivel"] == "alta":
        st.warning("⚠️ " + reglas.ADVERTENCIA_PRECAUCION_ALTA)
    elif precaucion["nivel"] == "leve":
        st.info(reglas.AVISO_PRECAUCION_LEVE)
    if precaucion["sin_respuesta"]:
        st.info(reglas.AVISO_SIN_RESPUESTA)
    for advertencia in resultado.advertencias:
        st.warning(advertencia)

    # IMC descriptivo y datos faltantes
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if resultado.imc is not None:
            ui.tarjeta_kpi("Tu IMC (descriptivo)", numero(resultado.imc, 1), "kg/m²")
            st.caption(f"Categoría convencional: **{resultado.categoria_imc}**. Es orientativa: el IMC no mide la grasa corporal ni la salud "
                       "y no determina por sí solo qué actividad es adecuada para ti.")
        else:
            st.caption("El IMC no se calculó porque falta el peso o la estatura.")
    with col_b:
        opcionales = resultado.validacion.faltantes_opcionales
        if opcionales:
            st.info("**Recomendación parcial:** no indicaste " + ", ".join(
                {"edad": "la edad", "genero": "el género", "peso_kg": "el peso", "estatura_m": "la estatura"}[k] for k in opcionales)
                + ". Consecuencias: " + "; ".join(sorted(set(opcionales.values()))) + ".")
        else:
            st.success("Datos completos: la recomendación usa toda la información disponible.")

    avisos_calorias = []
    for r in resultado.recomendaciones:
        if r.calorias is not None and r.calorias.estado != "ok" and r.calorias.mensaje not in avisos_calorias:
            avisos_calorias.append(r.calorias.mensaje)
    for aviso in avisos_calorias:
        st.warning("🔥 " + aviso)

    if not resultado.recomendaciones:
        st.warning("Con los criterios indicados no hay actividades con compatibilidad suficiente para recomendar. "
                   "Prueba con otro objetivo o consulta con un profesional.")
    else:
        st.caption("La puntuación mide la **coincidencia con los criterios programados** (objetivo, experiencia, duración y datos disponibles). "
                   "No es una probabilidad médica ni una medida validada de seguridad.")
        columnas = st.columns(len(resultado.recomendaciones))
        for posicion, (columna, r) in enumerate(zip(columnas, resultado.recomendaciones), start=1):
            with columna:
                tarjeta_recomendacion(r, posicion, estimador)

    if resultado.no_mostradas:
        with st.expander("Actividades que no se muestran como recomendación"):
            for tipo, motivo in resultado.no_mostradas:
                st.markdown(f"- **{tipo}:** {motivo}")
    st.caption(reglas.ADVERTENCIA_GENERAL)
    if resultado.recomendaciones:
        seccion_secundaria_recomendador(resultado, datos, df, estimador)


def pagina_recomendador(df: pd.DataFrame, estimador) -> None:
    ui.encabezado("Recomendador de ejercicios", "Cuéntanos tu perfil y tu objetivo: obtendrás hasta tres actividades compatibles, el motivo de cada una "
                  "y una estimación de las calorías de tu sesión.")
    with st.expander("¿Cómo funciona?"):
        st.markdown(
            "- Un **sistema de reglas explícitas** (editable en `modules/recommendation_rules.py`) puntúa Cardio, HIIT, Fuerza y Yoga según tu "
            "objetivo, tu experiencia y la duración elegida.\n"
            "- El conjunto de datos aporta información **descriptiva** (qué duraciones son habituales por nivel y qué hacen perfiles "
            "similares). En estos datos el tipo de entrenamiento no se asocia de forma clara con la edad, el género, el IMC ni el "
            "nivel, así que esos patrones **no se usan para puntuar** salvo que una diferencia sea estadísticamente detectable.\n"
            "- **No se usan límites de edad ni de IMC** para descartar actividades.\n"
            "- Si indicas una limitación importante o precauciones profesionales, no se recomiendan actividades intensas de forma "
            "automática y se muestra una advertencia.")
    formulario_usuario(df, estimador)
    mostrar_resultado(df, estimador)


# =========================================================================== #
# Programa principal
# =========================================================================== #
def principal() -> None:
    ui.aplicar_estilos()
    resultado_datos, df, estimador = cargar_todo()
    conservar_estado()

    with st.sidebar:
        ui.marca_lateral()
        seccion = st.radio("Navegación", [SECCION_INICIO, SECCION_EXPLORACION, SECCION_RECOMENDADOR],
                           key="seccion", label_visibility="collapsed")
        st.divider()
        filtrado = None
        if seccion == SECCION_EXPLORACION:
            filtrado = barra_filtros(df)
            st.divider()
        else:
            ui.imagen_lateral()   # en Inicio y Recomendador, la imagen ocupa el espacio de los filtros
        st.caption(f"📁 Datos: {numero(len(df), 0)} registros · Los datos se procesan localmente, sin servicios externos.")

    if seccion == SECCION_INICIO:
        pagina_inicio(df, resultado_datos)
    elif seccion == SECCION_EXPLORACION:
        pagina_exploracion(df, filtrado)
    else:
        pagina_recomendador(df, estimador)


principal()
