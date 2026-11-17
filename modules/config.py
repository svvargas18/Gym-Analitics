"""
Configuración central del proyecto.

Aquí viven las constantes que comparten todos los módulos: nombres de columnas,
rangos de validez, etiquetas en español, categorías esperadas y la paleta de
colores (naranja y morado, pensada para que sea distinguible con daltonismo).
Si quieres ajustar un umbral de validación o un color, este es el lugar.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Ubicación del archivo de datos
# --------------------------------------------------------------------------- #
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Nombres aceptados para el CSV (se prueban en este orden dentro de la carpeta del proyecto).
NOMBRES_CSV_CANDIDATOS = ["SET_DE_DATOS.csv", "SET DE DATOS.csv"]

# --------------------------------------------------------------------------- #
# Columnas del CSV (nombres REALES verificados) -> nombres internos en español
# --------------------------------------------------------------------------- #
COLUMNAS_ORIGINALES = {
    "Age": "edad",
    "Gender": "genero",
    "Weight (kg)": "peso_kg",
    "Height (m)": "estatura_m",
    "Max_BPM": "fc_max",
    "Avg_BPM": "fc_prom",
    "Resting_BPM": "fc_reposo",
    "Session_Duration (hours)": "duracion_h",
    "Calories_Burned": "calorias",
    "Workout_Type": "tipo_entrenamiento",
    "Fat_Percentage": "grasa_pct",
    "Water_Intake (liters)": "agua_l",
    "Workout_Frequency (days/week)": "frecuencia_semanal",
    "Experience_Level": "nivel_experiencia_cod",
    "BMI": "imc",
}

COLUMNAS_NUMERICAS = [
    "edad", "peso_kg", "estatura_m", "fc_max", "fc_prom", "fc_reposo",
    "duracion_h", "calorias", "grasa_pct", "agua_l", "frecuencia_semanal",
    "nivel_experiencia_cod", "imc",
]

# Columnas imprescindibles: si faltan o son inválidas, el registro se excluye.
COLUMNAS_NUCLEO = [
    "edad", "genero", "peso_kg", "estatura_m", "duracion_h", "calorias",
    "tipo_entrenamiento", "nivel_experiencia_cod",
]

# --------------------------------------------------------------------------- #
# Categorías esperadas (el CSV usa inglés; la interfaz usa español)
# --------------------------------------------------------------------------- #
MAPA_GENERO = {"male": "Masculino", "female": "Femenino"}
MAPA_TIPO = {"cardio": "Cardio", "hiit": "HIIT", "strength": "Fuerza", "yoga": "Yoga"}

GENEROS = ["Femenino", "Masculino"]
TIPOS_ENTRENAMIENTO = ["Cardio", "HIIT", "Fuerza", "Yoga"]

# El CSV codifica la experiencia como 1, 2 y 3 y no incluye un diccionario de datos.
# SUPUESTO DOCUMENTADO: 1 = Principiante, 2 = Intermedio, 3 = Avanzado (orden creciente).
# Los datos respaldan el orden: al subir el nivel suben la frecuencia semanal y la duración.
NIVELES = {1: "Principiante", 2: "Intermedio", 3: "Avanzado"}
ORDEN_NIVELES = list(NIVELES.values())

# Límites convencionales de IMC para adultos (orientativos).
LIMITES_IMC = [18.5, 25.0, 30.0]
CATEGORIAS_IMC = ["Bajo peso", "Peso normal", "Sobrepeso", "Obesidad"]

# --------------------------------------------------------------------------- #
# Validación: rangos FUERA de los cuales un valor se considera IMPOSIBLE/INVÁLIDO.
# Un valor dentro de estos rangos pero extremo se conserva (puede ser real).
# --------------------------------------------------------------------------- #
RANGOS_VALIDOS = {
    "edad": (10, 100),
    "peso_kg": (30, 300),
    "estatura_m": (1.20, 2.30),
    "duracion_h": (0.1, 8.0),        # debe ser > 0; más de 8 h se considera error
    "calorias": (1, 10000),          # no se admiten negativas ni cero
    "fc_max": (80, 230),
    "fc_prom": (30, 230),
    "fc_reposo": (25, 120),
    "grasa_pct": (2, 70),
    "agua_l": (0, 15),
    "frecuencia_semanal": (0, 7),
    "imc": (10, 80),
}
# Columnas cuyo valor inválido provoca la exclusión del registro (las demás pasan a "faltante").
COLUMNAS_NUCLEO_NUMERICAS = ["edad", "peso_kg", "estatura_m", "duracion_h", "calorias"]

# IMC extremo pero posible: se informa, no se elimina.
IMC_EXTREMO_BAJO = 15.0
IMC_EXTREMO_ALTO = 40.0
TOLERANCIA_IMC = 0.1          # diferencia máxima aceptada entre IMC del CSV y peso/estatura²

# Rango plausible de calorías por hora (para validar unidades en la estimación calórica).
RANGO_PLAUSIBLE_KCAL_H = (50.0, 2000.0)

# Tamaños mínimos de muestra para hacer comparaciones estadísticas "razonables".
N_MIN_COMPARACION = 30
N_MIN_GRUPO = 10

# --------------------------------------------------------------------------- #
# Etiquetas (con unidades) para ejes, tablas y selectores
# --------------------------------------------------------------------------- #
ETIQUETAS = {
    "edad": "Edad (años)",
    "genero": "Género",
    "peso_kg": "Peso (kg)",
    "estatura_m": "Estatura (m)",
    "fc_max": "FC máxima (lpm)",
    "fc_prom": "FC promedio (lpm)",
    "fc_reposo": "FC en reposo (lpm)",
    "duracion_h": "Duración (horas)",
    "duracion_min": "Duración de la sesión (min)",
    "calorias": "Calorías quemadas (kcal)",
    "calorias_por_hora": "Calorías por hora (kcal/h)",
    "tipo_entrenamiento": "Tipo de entrenamiento",
    "grasa_pct": "Grasa corporal (%)",
    "agua_l": "Consumo de agua (L)",
    "frecuencia_semanal": "Frecuencia semanal (días/semana)",
    "nivel_experiencia": "Nivel de experiencia",
    "nivel_experiencia_cod": "Nivel de experiencia (código 1-3)",
    "imc": "IMC (kg/m²)",
    "categoria_imc": "Categoría de IMC",
    "grupo_edad": "Grupo de edad",
}

# Variables que se ofrecen en los análisis (se excluyen duplicados como duracion_h
# y códigos internos; el CSV no trae identificadores de persona).
VARIABLES_NUMERICAS = [
    "edad", "peso_kg", "estatura_m", "imc", "fc_max", "fc_prom", "fc_reposo",
    "duracion_min", "calorias", "calorias_por_hora", "grasa_pct", "agua_l",
    "frecuencia_semanal",
]
VARIABLES_CATEGORICAS = [
    "tipo_entrenamiento", "nivel_experiencia", "genero", "categoria_imc", "grupo_edad",
]

# Variables apropiadas para la matriz de correlación.
# Se excluyen: calorias_por_hora (derivada de calorías/duración), duracion_h (duplicada),
# códigos categóricos y variables nominales sin orden (género, tipo).
VARIABLES_CORRELACION = [
    "edad", "peso_kg", "estatura_m", "imc", "fc_max", "fc_prom", "fc_reposo",
    "duracion_min", "calorias", "grasa_pct", "agua_l", "frecuencia_semanal",
    "nivel_experiencia_cod",
]

# --------------------------------------------------------------------------- #
# Paleta (ColorBrewer "PuOr", segura para daltonismo) y estilos
# --------------------------------------------------------------------------- #
COLOR_NARANJA = "#E66101"
COLOR_NARANJA_CLARO = "#FDB863"
COLOR_MORADO = "#5E3C99"
COLOR_MORADO_CLARO = "#B2ABD2"
COLOR_MORADO_OSCURO = "#2D1B4E"
COLOR_FONDO_SUAVE = "#F6F2FB"
COLOR_TEXTO = "#2B2140"

COLORES_TIPO = {
    "Cardio": COLOR_NARANJA,
    "HIIT": COLOR_MORADO,
    "Fuerza": COLOR_NARANJA_CLARO,
    "Yoga": COLOR_MORADO_CLARO,
}
COLORES_GENERO = {"Femenino": COLOR_MORADO, "Masculino": COLOR_NARANJA}
COLORES_NIVEL = {
    "Principiante": COLOR_NARANJA_CLARO,
    "Intermedio": COLOR_NARANJA,
    "Avanzado": COLOR_MORADO,
}
COLORES_IMC = {
    "Bajo peso": COLOR_MORADO_CLARO,
    "Peso normal": COLOR_MORADO,
    "Sobrepeso": COLOR_NARANJA_CLARO,
    "Obesidad": COLOR_NARANJA,
}
# Escala divergente para correlaciones: naranja (negativa) -> blanco (0) -> morado (positiva).
ESCALA_CORRELACION = [
    [0.0, "#B35806"], [0.25, "#F1A340"], [0.5, "#F7F7F7"],
    [0.75, "#998EC3"], [1.0, "#542788"],
]
