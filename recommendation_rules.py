"""
REGLAS DEL RECOMENDADOR (archivo pensado para editarse sin tocar la lógica).

Cómo funciona la puntuación (0-100) de cada tipo de entrenamiento:

    puntuación = Σ (peso_i × puntos_i) / Σ pesos activos     − penalizaciones

con cuatro componentes:
    1. OBJETIVO     → afinidad del tipo con el objetivo elegido (tabla AFINIDAD_OBJETIVO).
    2. EXPERIENCIA  → adaptación del tipo al nivel declarado (tabla AJUSTE_EXPERIENCIA).
    3. DURACIÓN     → qué tan habitual es la duración elegida entre los registros del
                      conjunto de datos del mismo nivel y tipo (patrón DESCRIPTIVO).
    4. DATOS        → diferencia de popularidad del tipo entre perfiles similares del
                      conjunto de datos. SOLO cuenta si la diferencia es estadísticamente
                      detectable; si no, se excluye y los demás pesos se reescalan.

ORIGEN DE CADA REGLA (importante para interpretar el resultado)
    * Las tablas AFINIDAD_OBJETIVO, AJUSTE_EXPERIENCIA, penalizaciones y listas de precaución son
      CRITERIOS DE DISEÑO del autor de la aplicación, basados en conocimiento general del
      entrenamiento (p. ej., que el yoga trabaja la movilidad o que el HIIT exige más adaptación).
      NO se derivan del conjunto de datos y NO han sido validados clínicamente. Deberían ser
      revisados por un profesional del ejercicio o de la salud antes de usarse con personas reales.
    * Los componentes DURACIÓN y DATOS sí se calculan con el conjunto de datos, y solo de forma
      descriptiva (lo que la gente registró), nunca como prueba de que algo sea mejor o más seguro.

No se aplica ningún límite de edad ni de IMC para excluir un tipo de entrenamiento.
"""

# --------------------------------------------------------------------------- #
# Catálogo
# --------------------------------------------------------------------------- #
TIPOS = ["Cardio", "HIIT", "Fuerza", "Yoga"]

OBJETIVOS = {
    "condicion_general": "Mejorar la condición física general",
    "fuerza": "Ganar fuerza",
    "movilidad": "Mejorar la movilidad y la flexibilidad",
    "cardio": "Realizar actividad cardiovascular",
}

# Descripción del nivel de experiencia que se muestra en el formulario (frecuencia de ejercicio declarada por la persona).
DESCRIPCION_NIVELES = {
    "Principiante": "Principiante · hago ejercicio 0 a 1 veces por semana",
    "Intermedio": "Intermedio · hago ejercicio 2 a 4 veces por semana",
    "Avanzado": "Avanzado · hago ejercicio 5 o más veces por semana",
}

RESUMEN_NIVELES = "Principiante: 0 a 1 veces por semana · Intermedio: 2 a 4 veces · Avanzado: 5 o más veces por semana."

# Intensidad típica de cada tipo (solo informativa y para las reglas de precaución).
INTENSIDAD_TIPO = {"Cardio": "moderada", "HIIT": "alta", "Fuerza": "moderada", "Yoga": "baja a moderada"}

# --------------------------------------------------------------------------- #
# 1) Afinidad tipo ↔ objetivo: (puntos 0-100, motivo mostrado al usuario)
# --------------------------------------------------------------------------- #
AFINIDAD_OBJETIVO = {
    "condicion_general": {
        "Cardio": (85, "Trabaja la resistencia cardiovascular, un componente central de la condición física general."),
        "Fuerza": (85, "Trabaja la fuerza muscular, otro componente central de la condición física general."),
        "HIIT": (75, "Combina esfuerzo cardiovascular y muscular en intervalos, por lo que aporta a varios componentes de la condición física."),
        "Yoga": (60, "Aporta movilidad, equilibrio y control corporal, aunque estimula menos la resistencia y la fuerza que otras opciones."),
    },
    "fuerza": {
        "Fuerza": (100, "Es la modalidad enfocada directamente en desarrollar fuerza muscular."),
        "HIIT": (60, "Los circuitos de HIIT suelen incluir ejercicios de fuerza con el propio peso, pero están menos enfocados en ganar fuerza."),
        "Yoga": (40, "Algunas posturas exigen fuerza y control, pero no está pensado para progresar en cargas."),
        "Cardio": (30, "Su foco es la resistencia cardiovascular, no el desarrollo de fuerza."),
    },
    "movilidad": {
        "Yoga": (100, "Trabaja de forma directa la movilidad, la flexibilidad y el control del movimiento."),
        "Fuerza": (50, "Realizado con rango completo de movimiento puede contribuir a la movilidad, aunque no es su foco principal."),
        "Cardio": (35, "Su foco es la resistencia cardiovascular; aporta poco a la movilidad por sí solo."),
        "HIIT": (25, "Su foco es el esfuerzo por intervalos; aporta poco a la movilidad por sí solo."),
    },
    "cardio": {
        "Cardio": (100, "Es la modalidad enfocada directamente en la resistencia cardiovascular."),
        "HIIT": (90, "También exige mucho al sistema cardiovascular, con esfuerzos por intervalos de mayor intensidad."),
        "Fuerza": (35, "Su foco es la fuerza muscular; el componente cardiovascular es secundario."),
        "Yoga": (30, "Su foco es la movilidad y el control corporal; el componente cardiovascular es limitado."),
    },
}

# --------------------------------------------------------------------------- #
# 2) Ajuste por experiencia: (puntos 0-100, motivo, nivel inicial sugerido)
#    El HIIT puntúa más bajo para principiantes porque exige más adaptación,
#    pero NUNCA se excluye: se sugiere una versión introductoria.
# --------------------------------------------------------------------------- #
AJUSTE_EXPERIENCIA = {
    "Principiante": {
        "Cardio": (90, "Se puede empezar con intensidad baja e ir aumentándola de forma gradual.",
                   "Introductorio: ritmo suave que permita conversar y progresión gradual."),
        "Fuerza": (80, "Se puede empezar con pocas series, ejercicios básicos y atención a la técnica.",
                   "Introductorio: ejercicios básicos con poca carga y énfasis en la técnica."),
        "Yoga": (90, "Existen variantes introductorias de ritmo suave.",
                 "Introductorio: posturas básicas a ritmo suave."),
        "HIIT": (55, "Alterna esfuerzos de alta intensidad y requiere más adaptación previa; puede empezarse con una versión introductoria.",
                 "Introductorio adaptado: intervalos cortos de esfuerzo moderado con descansos largos."),
    },
    "Intermedio": {
        "Cardio": (90, "Con algo de base se puede variar la intensidad y la duración.",
                   "Intermedio: combinar ritmo continuo con algunos intervalos."),
        "Fuerza": (90, "Con algo de base se pueden trabajar más grupos musculares y aumentar la carga progresivamente.",
                   "Intermedio: rutinas de cuerpo completo o divididas con progresión de carga."),
        "Yoga": (85, "Con algo de base se pueden practicar secuencias más largas y posturas de mayor exigencia.",
                 "Intermedio: secuencias fluidas con variantes de mayor exigencia."),
        "HIIT": (80, "Con una base previa se tolera mejor el trabajo por intervalos.",
                 "Intermedio: intervalos de esfuerzo alto con recuperación suficiente."),
    },
    "Avanzado": {
        "Cardio": (90, "Con mucha experiencia se pueden trabajar sesiones largas o de ritmo exigente.",
                   "Avanzado: sesiones largas o por ritmos, con control de la carga semanal."),
        "Fuerza": (90, "Con mucha experiencia se pueden planificar progresiones y cargas exigentes.",
                   "Avanzado: programación con progresión de cargas y periodización."),
        "Yoga": (80, "Con mucha experiencia se pueden practicar variantes avanzadas, aunque estimula menos la carga externa.",
                 "Avanzado: secuencias de mayor exigencia técnica."),
        "HIIT": (90, "Con mucha experiencia se tolera el trabajo por intervalos de alta intensidad.",
                 "Avanzado: intervalos de alta intensidad con recuperación planificada."),
    },
}

# --------------------------------------------------------------------------- #
# Pesos de los componentes (se reescalan cuando un componente no está activo)
# --------------------------------------------------------------------------- #
PESOS_COMPONENTES = {"objetivo": 0.60, "experiencia": 0.20, "duracion": 0.10, "datos": 0.10}

# --------------------------------------------------------------------------- #
# 3) Duración: puntos según dónde cae la duración elegida entre los registros
#    del conjunto de datos del mismo nivel de experiencia y tipo (patrón descriptivo).
# --------------------------------------------------------------------------- #
PUNTOS_DURACION = {
    "central": 100,    # dentro del 50 % central (percentiles 25-75)
    "habitual": 90,    # dentro del 80 % central (percentiles 10-90)
    "observada": 75,   # dentro del rango observado (mínimo-máximo) de ese nivel y tipo
    "inusual": 60,     # fuera de lo observado para ese nivel, aunque dentro del rango global
}

# --------------------------------------------------------------------------- #
# 4) Patrones del conjunto de datos (perfiles similares)
# --------------------------------------------------------------------------- #
PARAMETROS_DATOS = {
    "ventana_edad_anios": 5,        # «similar» = mismo nivel y edad ± 5 años
    "min_registros_perfil": 30,     # mínimo de registros similares para describir patrones
    "min_registros_celda": 20,      # mínimo de registros nivel×tipo para valorar la duración
    "alfa": 0.05,                   # nivel de significación (se corrige por el nº de tipos: Bonferroni)
}

# --------------------------------------------------------------------------- #
# Limitaciones y precauciones (preguntas sencillas y respetuosas)
# --------------------------------------------------------------------------- #
OPCIONES_LIMITACION = {
    "no": "No",
    "leve": "Sí, leve: no me impide moverme con normalidad",
    "importante": "Sí, importante: afecta mi forma de moverme o de hacer ejercicio",
    "prefiero_no_decir": "Prefiero no responder",
}
OPCIONES_RESTRICCION = {
    "no": "No",
    "si": "Sí, me han indicado precauciones o restricciones",
    "prefiero_no_decir": "Prefiero no responder",
}

# Con una limitación leve: se resta esta cantidad de puntos a los tipos listados (no se excluyen).
PENALIZACION_LIMITACION_LEVE = {"HIIT": 10}

# Con limitación importante o restricción indicada por un profesional: no se recomiendan de forma
# automática estos tipos (por su intensidad) y los demás se presentan en modo adaptado.
TIPOS_NO_AUTOMATICOS_CON_PRECAUCION = ["HIIT"]
NIVEL_INICIAL_PRECAUCION = ("Adaptado: solo a intensidad suave y después de recibir orientación "
                            "de un profesional de la salud o del ejercicio.")

# --------------------------------------------------------------------------- #
# Presentación de la compatibilidad y recomendaciones mínimas
# --------------------------------------------------------------------------- #
UMBRAL_COMPATIBILIDAD_ALTA = 75
UMBRAL_COMPATIBILIDAD_MEDIA = 55
PUNTUACION_MINIMA_MOSTRAR = UMBRAL_COMPATIBILIDAD_MEDIA   # por debajo, no se muestra como recomendación (se explica aparte)
NUMERO_RECOMENDACIONES = 3

# --------------------------------------------------------------------------- #
# Límites de validación de las entradas del usuario
# --------------------------------------------------------------------------- #
LIMITES_USUARIO = {
    "edad": (18, 100),
    "peso_kg": (30, 300),
    "estatura_m": (1.20, 2.30),
    "duracion_min": (10, 300),
    "imc": (10, 80),
}

# --------------------------------------------------------------------------- #
# Textos de advertencia
# --------------------------------------------------------------------------- #
ADVERTENCIA_GENERAL = (
    "Estas recomendaciones se generan con reglas sencillas y con patrones descriptivos del conjunto de datos. "
    "No son un diagnóstico, una evaluación médica ni una prescripción de ejercicio. Si tienes dudas sobre tu salud, "
    "consulta con un profesional antes de empezar o de aumentar la intensidad."
)
ADVERTENCIA_PRECAUCION_ALTA = (
    "Indicaste una limitación importante o precauciones señaladas por un profesional. Por eso NO se recomiendan "
    "actividades intensas de forma automática. Las opciones mostradas son orientativas, se presentan en modo adaptado "
    "y conviene comentarlas con un profesional de la salud o del ejercicio antes de empezar."
)
AVISO_PRECAUCION_LEVE = (
    "Indicaste una limitación leve. Se redujo ligeramente la compatibilidad de las actividades más intensas; "
    "adapta los ejercicios, evita movimientos que te causen dolor y detente si aparecen molestias."
)
AVISO_SIN_RESPUESTA = (
    "No respondiste alguna de las preguntas sobre limitaciones o precauciones, así que no se pudieron tener en cuenta. "
    "Si tienes alguna lesión o condición que requiera cuidados, consulta con un profesional."
)
