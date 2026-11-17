"""
Componentes visuales reutilizables de la interfaz (estilos CSS y tarjetas HTML).

Paleta: naranja y morado (ColorBrewer «PuOr»), apta para daltonismo. Todo el HTML que se
inserta aquí se compone SOLO con textos de la propia aplicación o con números calculados:
nunca con texto libre escrito por el usuario.
"""

import streamlit as st

from . import config as cfg

# Imagen decorativa de la barra lateral (Inicio y Recomendador). Para cambiarla, reemplaza el archivo
# assets/entrenamiento.jpg; si el archivo no existe, simplemente no se muestra.
RUTA_IMAGEN_LATERAL = cfg.RUTA_PROYECTO / "assets" / "entrenamiento.jpg"
# True: la vista se desplaza despacio de un panel a otro (Cardio → HIIT → Yoga → Fuerza y de vuelta).
# False: imagen fija, centrada entre HIIT y Yoga.
IMAGEN_LATERAL_ANIMADA = True

_CSS = f"""
<style>
.block-container {{ padding-top: 3.8rem; padding-bottom: 3rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ background: {cfg.COLOR_FONDO_SUAVE}; border-right: 1px solid #E8E0F2; }}

/* Cabecera principal */
.hero {{ background: linear-gradient(120deg, {cfg.COLOR_MORADO_OSCURO} 0%, {cfg.COLOR_MORADO} 62%, {cfg.COLOR_NARANJA} 140%);
        padding: 1.8rem 2.1rem; border-radius: 18px; margin-bottom: 1.1rem; }}
.hero h1 {{ color: #FFFFFF !important; font-size: 2.05rem; margin: 0 0 .35rem 0; padding: 0; line-height: 1.2; }}
.hero p {{ color: #F1EAFB !important; margin: 0; font-size: 1.03rem; max-width: 820px; }}
.hero .chips {{ margin-top: .9rem; }}
.hero .chips span {{ display: inline-block; background: rgba(255,255,255,.16); color: #FFFFFF; padding: .25rem .8rem;
                    border-radius: 999px; margin: 0 .45rem .35rem 0; font-size: .85rem; }}

/* Marca en la barra lateral */
.marca {{ padding: .2rem 0 .6rem 0; }}
.marca .titulo {{ font-size: 1.25rem; font-weight: 700; color: {cfg.COLOR_MORADO_OSCURO}; }}
.marca .sub {{ font-size: .82rem; color: #6B5B8C; }}

/* Tarjetas de indicadores */
.kpi-card {{ background: #FFFFFF; border: 1px solid #E8E0F2; border-left: 6px solid {cfg.COLOR_NARANJA};
            border-radius: 14px; padding: .85rem 1rem; box-shadow: 0 1px 4px rgba(45,27,78,.06); height: 100%; }}
.kpi-nombre {{ font-size: .78rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: #6B5B8C; }}
.kpi-valor {{ font-size: 1.9rem; font-weight: 700; color: {cfg.COLOR_MORADO_OSCURO}; line-height: 1.25; }}
.kpi-unidad {{ font-size: .95rem; font-weight: 600; color: {cfg.COLOR_NARANJA}; }}

/* Tarjetas de recomendación */
.reco-titulo {{ font-size: 1.35rem; font-weight: 700; color: {cfg.COLOR_MORADO_OSCURO}; margin: 0; }}
.reco-rango {{ font-size: .8rem; font-weight: 700; color: {cfg.COLOR_NARANJA}; letter-spacing: .05em; }}
.score-wrap {{ background: #EEE8F5; border-radius: 999px; height: 12px; width: 100%; overflow: hidden; margin: .35rem 0 .2rem 0; }}
.score-bar {{ height: 12px; border-radius: 999px; background: linear-gradient(90deg, {cfg.COLOR_MORADO} 0%, {cfg.COLOR_NARANJA} 100%); }}
.score-texto {{ font-size: .9rem; color: {cfg.COLOR_TEXTO}; }}
.kcal-grande {{ font-size: 1.75rem; font-weight: 700; color: {cfg.COLOR_MORADO}; line-height: 1.2; }}
.kcal-detalle {{ font-size: .85rem; color: #6B5B8C; }}

/* Insignias de origen de cada motivo */
.badge {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: .08rem .5rem; border-radius: 999px; margin-right: .4rem;
         border: 1px solid transparent; white-space: nowrap; }}
.badge-diseno {{ background: #ECE8F5; color: {cfg.COLOR_MORADO}; border-color: {cfg.COLOR_MORADO_CLARO}; }}
.badge-datos {{ background: #FDEBD9; color: #A84500; border-color: {cfg.COLOR_NARANJA_CLARO}; }}

/* Alertas con la paleta naranja/morado (los mensajes siempre llevan icono y texto, no dependen solo del color) */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ background: #F1ECF9; border: 1px solid #D8CDEB; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) p {{ color: {cfg.COLOR_MORADO_OSCURO}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background: #EAE4F5; border: 1px solid {cfg.COLOR_MORADO_CLARO}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) p {{ color: {cfg.COLOR_MORADO_OSCURO}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background: #FFF1E3; border: 1px solid #F6C493; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) p {{ color: #5A2A00; }}

/* Textos secundarios */
.nota {{ font-size: .86rem; color: #5C4B7D; }}
div[data-testid="stTabs"] button[role="tab"] {{ font-weight: 600; }}
</style>
"""


# Estilos de la imagen lateral. La imagen es horizontal (4 paneles) y el espacio es estrecho y alto, así que se
# recorta con `object-fit: cover` hasta llenar la altura disponible; la animación solo cambia la zona visible.
_CSS_IMAGEN = """
<style>
.st-key-imagen_lateral { margin-top: .2rem; }
.st-key-imagen_lateral img {
    width: 100% !important;
    height: clamp(250px, calc(100vh - 440px), 800px) !important;
    object-fit: cover;
    object-position: 27% 50%;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(45, 27, 78, .20);
    __ANIMACION__
}
@keyframes panoramica {
    0%, 12%   { object-position: 0% 50%; }     /* Cardio */
    30%, 42%  { object-position: 27% 50%; }    /* HIIT */
    58%, 70%  { object-position: 64% 50%; }    /* Yoga */
    88%, 100% { object-position: 100% 50%; }   /* Fuerza */
}
@media (prefers-reduced-motion: reduce) {
    .st-key-imagen_lateral img { animation: none; object-position: 27% 50%; }
}
</style>
"""


def aplicar_estilos() -> None:
    """Inyecta los estilos CSS de la aplicación."""
    st.markdown(_CSS, unsafe_allow_html=True)
    animacion = "animation: panoramica 28s ease-in-out infinite alternate;" if IMAGEN_LATERAL_ANIMADA else ""
    st.markdown(_CSS_IMAGEN.replace("__ANIMACION__", animacion), unsafe_allow_html=True)


def imagen_lateral() -> None:
    """Muestra la imagen decorativa en la barra lateral (no hace nada si el archivo no existe)."""
    if not RUTA_IMAGEN_LATERAL.exists():
        return
    with st.container(key="imagen_lateral"):
        st.image(str(RUTA_IMAGEN_LATERAL), width="stretch")


def encabezado(titulo: str, subtitulo: str, chips=()) -> None:
    """Banner principal con degradado morado-naranja."""
    chips_html = "".join(f"<span>{c}</span>" for c in chips)
    contenedor = f'<div class="chips">{chips_html}</div>' if chips_html else ""
    st.markdown(f'<div class="hero"><h1>{titulo}</h1><p>{subtitulo}</p>{contenedor}</div>', unsafe_allow_html=True)


def marca_lateral() -> None:
    """Nombre de la aplicación en la barra lateral."""
    st.markdown('<div class="marca"><div class="titulo">🏋️ Gym Analytics</div>'
                '<div class="sub">Dashboard de datos y recomendador de ejercicios</div></div>', unsafe_allow_html=True)


def tarjeta_kpi(nombre: str, valor: str, unidad: str, mensaje=None) -> None:
    """Tarjeta de indicador con nombre, valor y unidad (y un aviso si no se pudo calcular)."""
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-nombre">{nombre}</div>'
        f'<div class="kpi-valor">{valor}<span class="kpi-unidad"> {unidad}</span></div></div>',
        unsafe_allow_html=True)
    if mensaje:
        st.caption(f"⚠️ {mensaje}")


def barra_puntuacion(puntos: float) -> str:
    """HTML de una barra de progreso para la puntuación 0-100."""
    ancho = max(0, min(100, float(puntos)))
    return f'<div class="score-wrap"><div class="score-bar" style="width:{ancho:.0f}%"></div></div>'


def insignia(origen: str) -> str:
    """Insignia que indica si un motivo proviene de un criterio de diseño o de los datos."""
    clase = "badge-datos" if origen == "Datos" else "badge-diseno"
    return f'<span class="badge {clase}">{origen}</span>'
