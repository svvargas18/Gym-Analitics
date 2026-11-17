"""
Pruebas automatizadas de Gym Analytics.

Ejecutar desde la carpeta del proyecto (sin instalar nada extra, solo la biblioteca estándar):

    python -m unittest discover -s tests -v

Cubren: carga/limpieza de datos, filtros, gráficos, comparaciones, estimación de calorías,
reglas del recomendador, validación de entradas y la interfaz de Streamlit (AppTest).
"""

import copy
import logging
import math
import os
import sys
import types
import unittest

import numpy as np
import pandas as pd

logging.getLogger("streamlit").setLevel(logging.ERROR)   # silencia avisos internos de Streamlit en modo de prueba
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from modules import calorie_prediction as cp          # noqa: E402
from modules import config as cfg                      # noqa: E402
from modules import data_processing as dp              # noqa: E402
from modules import exploratory_analysis as ea         # noqa: E402
from modules import recommendation as rec              # noqa: E402
from modules import recommendation_rules as reglas     # noqa: E402

# Los datos reales se cargan una sola vez para todas las pruebas.
RESULTADO = dp.cargar_datos()
DF = ea.agregar_grupo_edad(RESULTADO.datos)
ESTIMADOR = cp.entrenar_estimador(DF)


def usuario(**kwargs) -> rec.DatosUsuario:
    """Usuario de ejemplo válido; los argumentos reemplazan campos concretos."""
    base = dict(edad=33, genero="Femenino", peso_kg=62.0, estatura_m=1.65, nivel="Intermedio",
                objetivo="condicion_general", duracion_min=60, limitacion="no", restriccion_profesional="no")
    base.update(kwargs)
    return rec.DatosUsuario(**base)


# =========================================================================== #
class PruebasProcesamiento(unittest.TestCase):
    def test_carga_real(self):
        self.assertEqual(RESULTADO.n_original, 973)
        self.assertEqual(RESULTADO.n_final, 973)
        for columna in cfg.COLUMNAS_NUCLEO:
            self.assertEqual(DF[columna].isna().sum(), 0, columna)
        self.assertEqual(set(DF["tipo_entrenamiento"].astype(str)), set(cfg.TIPOS_ENTRENAMIENTO))
        self.assertEqual(set(DF["nivel_experiencia"].astype(str)), set(cfg.ORDEN_NIVELES))

    def test_imc_coherente_con_peso_y_estatura(self):
        diferencia = (DF["imc"] - DF["peso_kg"] / DF["estatura_m"] ** 2).abs()
        self.assertLessEqual(diferencia.max(), cfg.TOLERANCIA_IMC)

    def test_fc_incoherente_se_marca_como_faltante(self):
        self.assertFalse((DF["fc_prom"] > DF["fc_max"]).any())
        self.assertGreater(DF["fc_prom"].isna().sum(), 0)  # el CSV real tiene registros incoherentes

    def test_extremos_se_conservan(self):
        extremos = ((DF["imc"] < cfg.IMC_EXTREMO_BAJO) | (DF["imc"] >= cfg.IMC_EXTREMO_ALTO)).sum()
        self.assertGreater(extremos, 0)  # no se eliminan observaciones extremas

    def test_datos_corruptos_se_detectan_y_se_excluyen(self):
        crudo = dp.leer_csv(dp.localizar_csv()).head(40).copy()
        crudo["Weight (kg)"] = crudo["Weight (kg)"].astype(object)   # permite introducir texto en una columna numérica
        crudo.loc[1, "Calories_Burned"] = -50          # calorías negativas
        crudo.loc[2, "Height (m)"] = 5.0               # estatura imposible
        crudo.loc[3, "Weight (kg)"] = 900              # peso imposible
        crudo.loc[4, "Session_Duration (hours)"] = 0   # duración no válida
        crudo.loc[5, "Workout_Type"] = "Baile"         # categoría inesperada
        crudo.loc[6, "Age"] = np.nan                   # edad faltante
        crudo.loc[7, "Weight (kg)"] = "abc"            # tipo de dato incorrecto
        crudo = pd.concat([crudo, crudo.iloc[[10]]], ignore_index=True)  # duplicado exacto
        res = dp.procesar_datos(crudo)
        self.assertEqual(res.n_original, 41)
        self.assertEqual(res.n_final, 41 - 7 - 1)       # 7 inválidos + 1 duplicado excluidos
        self.assertTrue((res.datos["calorias"] > 0).all())
        self.assertTrue((res.datos["duracion_h"] > 0).all())
        self.assertGreater(len(res.informe), 8)
        afectados = res.informe.set_index("Comprobación")["Registros afectados"]
        self.assertEqual(afectados["Registros duplicados (filas idénticas)"], 1)

    def test_imc_incoherente_se_recalcula(self):
        crudo = dp.leer_csv(dp.localizar_csv()).head(20).copy()
        crudo.loc[0, "BMI"] = 50.0
        res = dp.procesar_datos(crudo)
        esperado = crudo.loc[0, "Weight (kg)"] / crudo.loc[0, "Height (m)"] ** 2
        self.assertAlmostEqual(res.datos.loc[0, "imc"], esperado, places=1)

    def test_columnas_faltantes_lanzan_error_comprensible(self):
        crudo = dp.leer_csv(dp.localizar_csv()).drop(columns=["Calories_Burned"])
        with self.assertRaises(dp.ErrorDatos) as contexto:
            dp.procesar_datos(crudo)
        self.assertIn("Calories_Burned", str(contexto.exception))

    def test_archivo_inexistente_da_instrucciones(self):
        with self.assertRaises(dp.ErrorDatos) as contexto:
            dp.localizar_csv(os.path.join(RAIZ, "tests"))
        self.assertIn("Copia el CSV", str(contexto.exception))

    def test_categorias_de_imc_en_los_limites(self):
        casos = [(18.49, "Bajo peso"), (18.5, "Peso normal"), (24.99, "Peso normal"), (25.0, "Sobrepeso"),
                 (29.99, "Sobrepeso"), (30.0, "Obesidad"), (47.8, "Obesidad"), (12.0, "Bajo peso")]
        for valor, esperado in casos:
            self.assertEqual(dp.clasificar_imc(valor), esperado, valor)
        self.assertIsNone(dp.clasificar_imc(None))
        self.assertIsNone(dp.clasificar_imc(float("nan")))

    def test_calculo_del_imc(self):
        self.assertAlmostEqual(dp.calcular_imc(70, 1.75), 22.857, places=3)
        self.assertAlmostEqual(dp.calcular_imc(62, 1.65), 22.773, places=3)
        for peso, estatura in [(0, 1.7), (-5, 1.7), (70, 0), (None, 1.7), (70, None), ("a", 1.7), (float("nan"), 1.7)]:
            self.assertIsNone(dp.calcular_imc(peso, estatura), (peso, estatura))


# =========================================================================== #
class PruebasExploratorio(unittest.TestCase):
    def test_indicadores_se_calculan_con_los_datos(self):
        valores = {i["nombre"]: i["valor"] for i in ea.calcular_indicadores(DF)}
        self.assertEqual(valores["Registros"], "973")
        self.assertEqual(valores["Edad promedio"], f"{DF['edad'].mean():.1f}".replace(".", ","))
        self.assertEqual(valores["Calorías promedio"], f"{DF['calorias'].mean():.0f}")

    def test_indicadores_no_fallan_con_datos_vacios_o_incompletos(self):
        for datos in [DF.iloc[0:0], None, DF.drop(columns=["imc"]), DF.assign(edad=np.nan)]:
            indicadores = ea.calcular_indicadores(datos)
            self.assertEqual(len(indicadores), 5)
            self.assertTrue(any(i["mensaje"] for i in indicadores))

    def test_filtros(self):
        f = ea.aplicar_filtros(DF, edad=(30, 40), generos=["Femenino"], niveles=["Intermedio"], tipos=["HIIT", "Yoga"],
                               imc=(18, 30), duracion=(60, 120))
        self.assertGreater(len(f), 0)
        self.assertTrue(f["edad"].between(30, 40).all())
        self.assertTrue((f["genero"] == "Femenino").all())
        self.assertTrue(f["tipo_entrenamiento"].isin(["HIIT", "Yoga"]).all())
        self.assertTrue(f["imc"].between(18, 30).all())
        self.assertEqual(len(ea.aplicar_filtros(DF, generos=[])), 0)
        self.assertEqual(len(ea.aplicar_filtros(DF)), len(DF))
        self.assertEqual(len(ea.aplicar_filtros(DF, edad=(18, 18), tipos=["Yoga"])),
                         int(((DF["edad"] == 18) & (DF["tipo_entrenamiento"] == "Yoga")).sum()))

    def test_grupos_de_edad_configurables_dentro_del_rango(self):
        grupos = ea.crear_grupos_edad(DF["edad"], [30, 40, 50])
        self.assertEqual(list(grupos.cat.categories), ["18–29", "30–39", "40–49", "50–59"])
        self.assertEqual(grupos.isna().sum(), 0)
        fuera = ea.crear_grupos_edad(DF["edad"], [5, 30, 99])      # los cortes fuera del rango se ignoran
        self.assertEqual(list(fuera.cat.categories), ["18–29", "30–59"])
        self.assertEqual(ea.opciones_cortes_edad(DF), [20, 25, 30, 35, 40, 45, 50, 55])

    def test_todas_las_comparaciones_generan_figura_y_lectura(self):
        for clave in ea.COMPARACIONES:
            figura, lectura, mensaje = ea.construir_comparacion(DF, clave)
            self.assertIsNone(mensaje, clave)
            self.assertGreater(len(figura.data), 0, clave)
            self.assertTrue(lectura, clave)

    def test_comparacion_con_pocos_registros_da_mensaje(self):
        figura, _, mensaje = ea.construir_comparacion(DF.head(5), "A")
        self.assertIsNone(figura)
        self.assertIn("pocos registros", mensaje)

    def test_lecturas_no_afirman_causalidad(self):
        _, lectura, _ = ea.construir_comparacion(DF, "E")
        self.assertIn("no demuestra causalidad", lectura)
        _, lectura, _ = ea.construir_comparacion(DF, "H")
        self.assertIn("no indica que un grupo sea", lectura)

    def test_matriz_de_correlacion(self):
        for metodo in ("pearson", "spearman"):
            corr = ea.matriz_correlacion(DF, cfg.VARIABLES_CORRELACION, metodo)
            self.assertEqual(corr.shape, (len(cfg.VARIABLES_CORRELACION),) * 2)
            self.assertTrue(np.allclose(np.diag(corr), 1.0))
            self.assertTrue(np.allclose(corr, corr.T, equal_nan=True))
            self.assertGreater(len(ea.figura_matriz_correlacion(corr, metodo).data), 0)
        self.assertNotIn("calorias_por_hora", cfg.VARIABLES_CORRELACION)
        self.assertIsNone(ea.matriz_correlacion(DF.head(3), cfg.VARIABLES_CORRELACION))
        self.assertIsNone(ea.matriz_correlacion(DF, ["edad"]))

    def test_perfiles_comparan_solo_con_muestra_suficiente(self):
        perfil = ea.filtrar_perfil(DF, grupos_edad=["30–39"], categorias_imc=["Sobrepeso"])
        resumen = ea.resumir_perfil(perfil, DF, variables_filtradas=("edad", "imc"))
        medidas = resumen["medidas"].set_index("Variable")
        self.assertIn("No comparable", medidas.loc["Edad (años)", "Lectura"])
        self.assertIn("No comparable", medidas.loc["IMC (kg/m²)", "Lectura"])
        self.assertEqual(resumen["n"], len(perfil))
        pequeno = DF[(DF["edad"] == 18) & (DF["tipo_entrenamiento"] == "Yoga") & (DF["nivel_experiencia"] == "Avanzado")]
        resumen_pequeno = ea.resumir_perfil(pequeno, DF)
        self.assertTrue(resumen_pequeno["medidas"]["Lectura"].str.contains("Muestra insuficiente").all())
        self.assertIn("insuficiente", resumen_pequeno["lectura_tipos"])

    def test_perfil_vacio_no_lanza_error(self):
        vacio = ea.filtrar_perfil(DF, grupos_edad=["18–29"], tipos=["Yoga"], niveles=["Avanzado"], categorias_imc=["Bajo peso"])
        self.assertIsInstance(ea.resumir_perfil(vacio, DF), dict)

    def test_exploracion_personalizada(self):
        figura, lectura, mensaje = ea.grafico_personalizado(DF, "duracion_min", "calorias")
        self.assertIsNone(mensaje); self.assertEqual(figura.data[0].type, "scatter")
        figura, _, _ = ea.grafico_personalizado(DF, "tipo_entrenamiento", "calorias", "genero")
        self.assertEqual(figura.data[0].type, "box")
        figura, _, _ = ea.grafico_personalizado(DF, "edad", "tipo_entrenamiento")
        self.assertEqual(figura.data[0].type, "box")
        figura, _, _ = ea.grafico_personalizado(DF, "tipo_entrenamiento", "nivel_experiencia")
        self.assertEqual(figura.data[0].type, "bar")
        invalidas = [("edad", "edad", None), ("imc", "peso_kg", None), ("peso_kg", "imc", None), ("categoria_imc", "imc", None),
                     ("grupo_edad", "edad", None), ("tipo_entrenamiento", "nivel_experiencia", "genero"),
                     ("edad", "calorias", "edad")]
        for x, y, color in invalidas:
            figura, _, mensaje = ea.grafico_personalizado(DF, x, y, color)
            self.assertIsNone(figura, (x, y, color)); self.assertTrue(mensaje, (x, y, color))

    def test_tablas_descriptivas(self):
        frec = ea.tabla_frecuencias(DF, "tipo_entrenamiento")
        self.assertEqual(frec["Registros"].sum(), len(DF))
        self.assertAlmostEqual(frec["Porcentaje (%)"].sum(), 100, delta=0.2)
        stats_ = ea.estadisticas_descriptivas(DF, "fc_prom").set_index("Estadístico")
        self.assertEqual(stats_.loc["Faltantes", "Valor"], str(int(DF["fc_prom"].isna().sum())))


# =========================================================================== #
class PruebasCalorias(unittest.TestCase):
    def test_unidades_y_registros_validos(self):
        validos, info = cp.registros_validos_calorias(DF)
        self.assertEqual(info["n_validos"], len(DF))
        minimo, maximo = cfg.RANGO_PLAUSIBLE_KCAL_H
        self.assertTrue(validos["calorias_por_hora"].between(minimo, maximo).all())

    def test_calorias_por_hora_no_depende_de_la_duracion(self):
        self.assertTrue(ESTIMADOR.linealidad["valida"])
        self.assertLess(abs(ESTIMADOR.linealidad["correlacion_pearson"]), cp.LIMITE_CORRELACION_LINEALIDAD)

    def test_estimacion_es_mediana_por_horas(self):
        for tipo in cfg.TIPOS_ENTRENAMIENTO:
            mediana = DF.loc[DF["tipo_entrenamiento"] == tipo, "calorias_por_hora"].median()
            res = cp.estimar_calorias(ESTIMADOR, tipo, 90)
            self.assertEqual(res.estado, "ok")
            self.assertAlmostEqual(res.base_kcal, 5 * round(mediana * 1.5 / 5), delta=0.1)
            self.assertLess(res.base_rango[0], res.base_kcal)
            self.assertGreater(res.base_rango[1], res.base_kcal)
            self.assertTrue(res.advertencias)

    def test_se_duplica_la_duracion_y_casi_se_duplican_las_calorias(self):
        a = cp.estimar_calorias(ESTIMADOR, "Cardio", 45).base_kcal
        b = cp.estimar_calorias(ESTIMADOR, "Cardio", 90).base_kcal
        self.assertAlmostEqual(b / a, 2.0, delta=0.02)

    def test_no_extrapola_fuera_del_rango_observado(self):
        for duracion in (10, 29, 121, 300):
            res = cp.estimar_calorias(ESTIMADOR, "HIIT", duracion)
            self.assertEqual(res.estado, "fuera_de_rango", duracion)
            self.assertTrue(math.isnan(res.base_kcal))
            self.assertIn("No se extrapola", res.mensaje)
        for duracion in (30, 120):   # los extremos observados sí se estiman
            self.assertEqual(cp.estimar_calorias(ESTIMADOR, "HIIT", duracion).estado, "ok")

    def test_entradas_invalidas_no_producen_numeros(self):
        for duracion in (0, -10, None, "abc", float("nan")):
            self.assertEqual(cp.estimar_calorias(ESTIMADOR, "Yoga", duracion).estado, "no_disponible", duracion)
        self.assertEqual(cp.estimar_calorias(ESTIMADOR, "Baile", 60).estado, "no_disponible")

    def test_ajuste_complementario_solo_con_datos_completos_y_en_rango(self):
        completo = cp.estimar_calorias(ESTIMADOR, "Yoga", 60, 33, "Femenino", 62.0)
        self.assertTrue(completo.refinada_aplicada)
        self.assertFalse(cp.estimar_calorias(ESTIMADOR, "Yoga", 60).refinada_aplicada)
        self.assertFalse(cp.estimar_calorias(ESTIMADOR, "Yoga", 60, 33, None, 62.0).refinada_aplicada)
        fuera_edad = cp.estimar_calorias(ESTIMADOR, "Yoga", 60, 75, "Femenino", 62.0)
        self.assertFalse(fuera_edad.refinada_aplicada); self.assertIn("fuera del rango", fuera_edad.motivo_sin_refinada)
        fuera_peso = cp.estimar_calorias(ESTIMADOR, "Yoga", 60, 33, "Femenino", 200.0)
        self.assertFalse(fuera_peso.refinada_aplicada)
        hombre = cp.estimar_calorias(ESTIMADOR, "Yoga", 60, 33, "Masculino", 62.0).refinada_kcal
        self.assertGreater(hombre, completo.refinada_kcal)    # en los datos, los hombres registran más kcal/h

    def test_metricas_de_validacion_cruzada_son_validas(self):
        m = ESTIMADOR.metricas_cv
        self.assertEqual(m["n"], len(DF))
        for clave in ("mae_principal", "mae_complementario"):
            self.assertTrue(0 < m[clave] < 300, clave)
        for clave in ("r2_principal", "r2_complementario"):
            self.assertTrue(0 < m[clave] < 1, clave)

    def test_si_calorias_por_hora_depende_de_la_duracion_no_se_estima(self):
        """Salvaguarda: con calorías constantes (no proporcionales a la duración), no se escala por horas."""
        sintetico = DF.copy()
        sintetico["calorias"] = 700.0
        sintetico["calorias_por_hora"] = sintetico["calorias"] / sintetico["duracion_h"]
        estimador = cp.entrenar_estimador(sintetico)
        self.assertFalse(estimador.linealidad["valida"])
        res = cp.estimar_calorias(estimador, "Cardio", 60)
        self.assertEqual(res.estado, "no_disponible")
        self.assertIn("no es posible convertir", res.mensaje)

    def test_registros_invalidos_se_excluyen_del_entrenamiento(self):
        sucio = DF.copy()
        sucio.loc[0, "calorias"] = -100
        sucio.loc[1, "calorias"] = 90000
        _, info = cp.registros_validos_calorias(sucio)
        self.assertEqual(info["n_excluidos"], 2)


# =========================================================================== #
class PruebasRecomendador(unittest.TestCase):
    def test_parseo_de_estatura(self):
        casos = {"172": 1.72, "172 cm": 1.72, "1,72": 1.72, "1.72 m": 1.72, "1,72m": 1.72, " 165 CM ": 1.65, "1.8 metros": 1.8}
        for texto, esperado in casos.items():
            metros, error = rec.parsear_estatura(texto)
            self.assertIsNone(error, texto); self.assertAlmostEqual(metros, esperado, places=3, msg=texto)
        for texto in ("abc", "1,7,2", "alto", "170 pies"):
            metros, error = rec.parsear_estatura(texto)
            self.assertIsNone(metros); self.assertIn("No se entiende", error)
        self.assertEqual(rec.parsear_estatura(""), (None, None))
        self.assertEqual(rec.parsear_estatura(None), (None, None))

    def test_validacion_de_entradas_imposibles(self):
        casos = {
            "menor de edad": usuario(edad=15), "edad absurda": usuario(edad=150), "peso negativo": usuario(peso_kg=-5),
            "peso bajo": usuario(peso_kg=10), "peso enorme": usuario(peso_kg=900), "estatura imposible": usuario(estatura_m=0.05),
            "estatura enorme": usuario(estatura_m=17.2), "duración mínima": usuario(duracion_min=3), "duración máxima": usuario(duracion_min=900),
            "imc imposible": usuario(peso_kg=300, estatura_m=1.2), "texto como peso": usuario(peso_kg="mucho"),
            "género inválido": usuario(genero="X"), "nivel inválido": usuario(nivel="Experto"), "objetivo inválido": usuario(objetivo="x"),
            "limitación inválida": usuario(limitacion="quizá"), "nan": usuario(edad=float("nan")),
        }
        for nombre, datos in casos.items():
            res = rec.recomendar(datos, DF, ESTIMADOR)
            self.assertEqual(res.estado, "invalido", nombre)
            self.assertTrue(res.validacion.errores, nombre)
            self.assertEqual(res.recomendaciones, [], nombre)
            self.assertIsNone(res.imc, nombre)   # no se calcula nada con entradas inválidas

    def test_faltantes_criticos_y_opcionales(self):
        res = rec.recomendar(usuario(nivel=None, objetivo=None), DF, ESTIMADOR)
        self.assertEqual(res.estado, "faltan_datos")
        self.assertIn("nivel de experiencia", res.mensaje); self.assertIn("objetivo principal", res.mensaje)
        res = rec.recomendar(usuario(duracion_min=None), DF, ESTIMADOR)
        self.assertEqual(res.estado, "faltan_datos")
        res = rec.recomendar(usuario(edad=None, genero=None, peso_kg=None, estatura_m=None), DF, ESTIMADOR)
        self.assertEqual(res.estado, "ok")
        self.assertEqual(set(res.validacion.faltantes_opcionales), {"edad", "genero", "peso_kg", "estatura_m"})
        self.assertIsNone(res.imc); self.assertIsNone(res.categoria_imc)
        self.assertTrue(res.recomendaciones)
        self.assertFalse(res.recomendaciones[0].calorias.refinada_aplicada)

    def test_imc_solo_con_peso_y_estatura_validos(self):
        res = rec.recomendar(usuario(peso_kg=70, estatura_m=1.75), DF, ESTIMADOR)
        self.assertAlmostEqual(res.imc, 22.857, places=2); self.assertEqual(res.categoria_imc, "Peso normal")
        self.assertIsNone(rec.recomendar(usuario(peso_kg=70, estatura_m=None), DF, ESTIMADOR).imc)
        self.assertIsNone(rec.recomendar(usuario(peso_kg=None, estatura_m=1.75), DF, ESTIMADOR).imc)

    def test_puntuaciones_ordenadas_y_acotadas(self):
        for objetivo in reglas.OBJETIVOS:
            for nivel in cfg.ORDEN_NIVELES:
                res = rec.recomendar(usuario(objetivo=objetivo, nivel=nivel), DF, ESTIMADOR)
                puntos = [r.puntuacion for r in res.recomendaciones]
                self.assertEqual(puntos, sorted(puntos, reverse=True))
                self.assertTrue(all(0 <= p <= 100 for p in puntos))
                self.assertLessEqual(len(res.recomendaciones), 3)
                self.assertGreaterEqual(len(res.recomendaciones), 1)
                for r in res.recomendaciones:
                    self.assertGreaterEqual(r.puntuacion, reglas.PUNTUACION_MINIMA_MOSTRAR)

    def test_prioriza_la_actividad_compatible_con_el_objetivo(self):
        esperado = {"fuerza": "Fuerza", "movilidad": "Yoga", "cardio": "Cardio"}
        for objetivo, tipo in esperado.items():
            for nivel in cfg.ORDEN_NIVELES:
                res = rec.recomendar(usuario(objetivo=objetivo, nivel=nivel), DF, ESTIMADOR)
                self.assertEqual(res.recomendaciones[0].tipo, tipo, (objetivo, nivel))

    def test_la_experiencia_ajusta_sin_prohibir(self):
        principiante = rec.recomendar(usuario(objetivo="cardio", nivel="Principiante"), DF, ESTIMADOR)
        avanzado = rec.recomendar(usuario(objetivo="cardio", nivel="Avanzado", duracion_min=105), DF, ESTIMADOR)
        punt_p = {r.tipo: r.puntuacion for r in principiante.recomendaciones}
        punt_a = {r.tipo: r.puntuacion for r in avanzado.recomendaciones}
        self.assertIn("HIIT", punt_p)                     # el HIIT no se prohíbe a principiantes
        self.assertLess(punt_p["HIIT"], punt_a["HIIT"])   # pero puntúa menos que para avanzados
        hiit = [r for r in principiante.recomendaciones if r.tipo == "HIIT"][0]
        self.assertIn("Introductorio", hiit.nivel_inicial)

    def test_el_imc_no_modifica_las_puntuaciones(self):
        a = rec.recomendar(usuario(peso_kg=55, estatura_m=1.75, objetivo="cardio"), DF, ESTIMADOR)
        b = rec.recomendar(usuario(peso_kg=125, estatura_m=1.60, objetivo="cardio"), DF, ESTIMADOR)   # IMC > 48
        self.assertGreater(b.imc, 45)
        self.assertEqual({r.tipo: r.puntuacion for r in a.recomendaciones}, {r.tipo: r.puntuacion for r in b.recomendaciones})
        self.assertIn("Cardio", [r.tipo for r in b.recomendaciones]); self.assertIn("HIIT", [r.tipo for r in b.recomendaciones])

    def test_la_edad_no_prohibe_ni_reduce_a_yoga(self):
        mayor = rec.recomendar(usuario(edad=59, objetivo="cardio", nivel="Avanzado", duracion_min=105), DF, ESTIMADOR)
        joven = rec.recomendar(usuario(edad=18, objetivo="cardio", nivel="Avanzado", duracion_min=105), DF, ESTIMADOR)
        self.assertEqual([r.tipo for r in mayor.recomendaciones][:2], ["Cardio", "HIIT"])
        self.assertEqual({r.tipo: r.puntuacion for r in mayor.recomendaciones}, {r.tipo: r.puntuacion for r in joven.recomendaciones})
        # Edad fuera del rango de los datos: se advierte, pero no se bloquea ni se descarta nada por la edad
        muy_mayor = rec.recomendar(usuario(edad=72, objetivo="cardio", nivel="Avanzado", duracion_min=105), DF, ESTIMADOR)
        self.assertEqual(muy_mayor.estado, "ok")
        self.assertIn("HIIT", [r.tipo for r in muy_mayor.recomendaciones])
        self.assertTrue(any("fuera del rango" in a for a in muy_mayor.advertencias))

    def test_precaucion_alta_no_recomienda_actividades_intensas(self):
        for limitacion, restriccion in [("importante", "no"), ("no", "si"), ("importante", "si")]:
            for objetivo in reglas.OBJETIVOS:
                for nivel in cfg.ORDEN_NIVELES:
                    res = rec.recomendar(usuario(objetivo=objetivo, nivel=nivel, limitacion=limitacion, restriccion_profesional=restriccion),
                                         DF, ESTIMADOR)
                    self.assertEqual(res.precaucion["nivel"], "alta")
                    self.assertNotIn("HIIT", [r.tipo for r in res.recomendaciones], (limitacion, restriccion, objetivo, nivel))
                    self.assertIn("HIIT", [t for t, _ in res.no_mostradas])
                    for r in res.recomendaciones:
                        self.assertIn("Adaptado", r.nivel_inicial)
                        self.assertTrue(r.avisos)

    def test_limitacion_leve_reduce_pero_no_excluye(self):
        base = rec.recomendar(usuario(objetivo="cardio", nivel="Avanzado", duracion_min=105), DF, ESTIMADOR)
        leve = rec.recomendar(usuario(objetivo="cardio", nivel="Avanzado", duracion_min=105, limitacion="leve"), DF, ESTIMADOR)
        pb = {r.tipo: r.puntuacion for r in base.recomendaciones}
        pl = {r.tipo: r.puntuacion for r in leve.recomendaciones}
        self.assertEqual(leve.precaucion["nivel"], "leve")
        self.assertIn("HIIT", pl)
        self.assertAlmostEqual(pb["HIIT"] - pl["HIIT"], reglas.PENALIZACION_LIMITACION_LEVE["HIIT"], places=6)
        self.assertEqual(pb["Cardio"], pl["Cardio"])

    def test_sin_respuesta_a_precauciones_se_informa(self):
        res = rec.recomendar(usuario(limitacion="prefiero_no_decir"), DF, ESTIMADOR)
        self.assertTrue(res.precaucion["sin_respuesta"]); self.assertEqual(res.precaucion["nivel"], "ninguna")

    def test_cada_recomendacion_explica_su_origen(self):
        res = rec.recomendar(usuario(), DF, ESTIMADOR)
        for r in res.recomendaciones:
            self.assertEqual(len(r.motivos), 4)
            self.assertTrue({m["origen"] for m in r.motivos} <= {rec.ORIGEN_DISENO, rec.ORIGEN_DATOS})
            self.assertTrue(all(m["texto"] for m in r.motivos))
            self.assertTrue(r.nivel_inicial); self.assertEqual(r.duracion_min, 60)
            self.assertEqual(r.calorias.estado, "ok")
            self.assertIn(r.compatibilidad, ("Alta", "Media"))

    def test_en_estos_datos_los_patrones_no_se_usan_para_puntuar(self):
        """El tipo de entrenamiento no se asocia con perfil similar en el CSV: el componente 'datos' queda inactivo."""
        for nivel in cfg.ORDEN_NIVELES:
            perfil = rec.perfil_similar(DF, usuario(nivel=nivel))
            self.assertTrue(perfil["suficiente"])
            self.assertFalse(any(info["significativo"] for info in perfil["por_tipo"].values()), nivel)
        res = rec.recomendar(usuario(), DF, ESTIMADOR)
        for r in res.recomendaciones:
            datos = [c for c in r.componentes if c["nombre"] == "Patrones en los datos"][0]
            self.assertIsNone(datos["puntos"])
            self.assertIn("no influyó", r.motivos[3]["texto"])

    def test_si_hubiera_un_patron_real_en_los_datos_se_incorpora_y_se_explica(self):
        """Con datos sintéticos (principiantes de 28-38 años que solo hacen Yoga), el componente se activa."""
        sintetico = DF.copy()
        mascara = (sintetico["nivel_experiencia"] == "Principiante") & sintetico["edad"].between(28, 38)
        sintetico.loc[mascara, "tipo_entrenamiento"] = pd.Categorical(["Yoga"] * int(mascara.sum()), categories=cfg.TIPOS_ENTRENAMIENTO)
        perfil = rec.perfil_similar(sintetico, usuario(nivel="Principiante", edad=33))
        self.assertTrue(perfil["por_tipo"]["Yoga"]["significativo"])
        res = rec.recomendar(usuario(nivel="Principiante", edad=33, objetivo="condicion_general"), sintetico, None)
        yoga = [r for r in res.recomendaciones if r.tipo == "Yoga"][0]
        componente = [c for c in yoga.componentes if c["nombre"] == "Patrones en los datos"][0]
        self.assertIsNotNone(componente["puntos"])
        self.assertIn("estadísticamente detectable", yoga.motivos[3]["texto"])
        self.assertIn("no implica que sea mejor", yoga.motivos[3]["texto"])

    def test_las_reglas_son_modificables_sin_tocar_el_codigo(self):
        reglas_alt = types.SimpleNamespace(**{k: copy.deepcopy(v) for k, v in vars(reglas).items() if not k.startswith("__")})
        reglas_alt.AFINIDAD_OBJETIVO["cardio"]["Yoga"] = (100, "Regla de prueba.")
        reglas_alt.AFINIDAD_OBJETIVO["cardio"]["Cardio"] = (10, "Regla de prueba.")
        res = rec.recomendar(usuario(objetivo="cardio", nivel="Intermedio"), DF, ESTIMADOR, reglas=reglas_alt)
        self.assertEqual(res.recomendaciones[0].tipo, "Yoga")

    def test_duracion_fuera_del_rango_de_datos(self):
        res = rec.recomendar(usuario(duracion_min=20), DF, ESTIMADOR)
        self.assertEqual(res.estado, "ok")
        for r in res.recomendaciones:
            self.assertEqual(r.calorias.estado, "fuera_de_rango")
            self.assertIsNone([c for c in r.componentes if c["nombre"] == "Duración"][0]["puntos"])

    def test_frecuencias_por_nivel_del_csv(self):
        f = rec.frecuencias_por_nivel(DF)
        self.assertEqual(f, {"Principiante": (2, 3), "Intermedio": (3, 4), "Avanzado": (4, 5)})


# =========================================================================== #
class PruebasInterfaz(unittest.TestCase):
    """Pruebas de la aplicación completa con el marco de pruebas de Streamlit (sin navegador)."""

    SECCION_EXPLORACION = "📊 Exploración y perfiles"
    SECCION_RECOMENDADOR = "🎯 Recomendador de ejercicios"

    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        cls.AppTest = AppTest

    def nueva_app(self):
        at = self.AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def ir_a(self, at, seccion):
        at.radio(key="seccion").set_value(seccion).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])

    @staticmethod
    def registros_filtrados(at):
        texto = [m.value for m in at.sidebar.markdown if "registros cumplen" in m.value][0]
        return int(texto.split("**")[1].replace(".", ""))

    def enviar_formulario(self, at, **campos):
        valores = dict(edad=33, genero="Femenino", peso=62.0, estatura="165 cm", duracion=60, nivel="Intermedio",
                       objetivo="condicion_general", limitacion="no", restriccion="no")
        valores.update(campos)
        at.number_input(key="form_edad").set_value(valores["edad"])
        at.selectbox(key="form_genero").set_value(valores["genero"])
        at.number_input(key="form_peso").set_value(valores["peso"])
        at.text_input(key="form_estatura").set_value(valores["estatura"])
        at.number_input(key="form_duracion").set_value(valores["duracion"])
        at.selectbox(key="form_nivel").set_value(valores["nivel"])
        at.selectbox(key="form_objetivo").set_value(valores["objetivo"])
        at.selectbox(key="form_limitacion").set_value(valores["limitacion"])
        at.selectbox(key="form_restriccion").set_value(valores["restriccion"])
        [b for b in at.button if "recomendaciones" in b.label][0].click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])

    @staticmethod
    def tipos_en_tarjetas(at):
        return [m.value.split('reco-titulo">')[1].split("<")[0] for m in at.markdown if "OPCIÓN" in m.value]

    def test_inicio_arranca_y_muestra_indicadores(self):
        at = self.nueva_app()
        self.assertEqual(sum('class="kpi-card"' in m.value for m in at.markdown), 5)
        self.assertFalse(at.error)

    def test_filtros_actualizan_y_se_restablecen(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_EXPLORACION)
        self.assertEqual(self.registros_filtrados(at), 973)
        at.slider(key="flt_edad").set_value((30, 40)).run()
        con_edad = self.registros_filtrados(at)
        self.assertEqual(con_edad, int(DF["edad"].between(30, 40).sum()))
        at.multiselect(key="flt_tipo").set_value(["HIIT"]).run()
        self.assertEqual(self.registros_filtrados(at), int((DF["edad"].between(30, 40) & (DF["tipo_entrenamiento"] == "HIIT")).sum()))
        at.button(key="btn_reset_filtros").click().run()
        self.assertEqual(self.registros_filtrados(at), 973)
        self.assertEqual(at.slider(key="flt_edad").value, (18, 59))

    def test_filtros_sin_resultados_muestran_mensaje(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_EXPLORACION)
        at.multiselect(key="flt_nivel").set_value([]).run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Ningún registro cumple" in w.value for w in at.warning))

    def test_los_filtros_se_conservan_al_cambiar_de_seccion(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_EXPLORACION)
        at.slider(key="flt_edad").set_value((25, 35)).run()
        self.ir_a(at, "🏠 Inicio"); self.ir_a(at, self.SECCION_EXPLORACION)
        self.assertEqual(at.slider(key="flt_edad").value, (25, 35))

    def test_las_pestanas_y_selectores_de_exploracion_no_fallan(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_EXPLORACION)
        for clave in ea.COMPARACIONES:
            at.selectbox(key="cmp_clave").set_value(clave).run(); self.assertFalse(at.exception, clave)
        at.radio(key="cor_metodo").set_value("Spearman").run(); self.assertFalse(at.exception)
        at.radio(key="var_tipo").set_value("Categórica").run()
        for variable in cfg.VARIABLES_CATEGORICAS:
            at.selectbox(key="var_categorica").set_value(variable).run(); self.assertFalse(at.exception, variable)
        at.toggle(key="perf_comparar").set_value(True).run()
        at.multiselect(key="perf_A_nivel").set_value(["Avanzado"]).run()
        at.multiselect(key="perf_B_nivel").set_value(["Principiante"]).run()
        self.assertFalse(at.exception)
        at.selectbox(key="pers_x").set_value("imc"); at.selectbox(key="pers_y").set_value("peso_kg"); at.run()
        self.assertTrue(any("se calcula directamente" in w.value for w in at.warning))

    def test_formulario_valido_genera_tres_recomendaciones_con_calorias(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_RECOMENDADOR)
        self.enviar_formulario(at)
        self.assertEqual(len(self.tipos_en_tarjetas(at)), 3)
        self.assertEqual(sum('class="kcal-grande"' in m.value for m in at.markdown), 3)   # una estimación por tarjeta
        self.assertTrue(any("Tu IMC" in m.value for m in at.markdown))
        self.assertFalse(at.error)

    def test_formulario_con_entradas_invalidas_muestra_errores_claros(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_RECOMENDADOR)
        self.enviar_formulario(at, edad=15)
        self.assertTrue(any("personas adultas" in e.value for e in at.error))
        self.enviar_formulario(at, edad=30, peso=10.0)
        self.assertTrue(any("peso debe estar entre" in e.value for e in at.error))
        self.enviar_formulario(at, peso=62.0, estatura="abc")
        self.assertTrue(any("No se entiende la estatura" in e.value for e in at.error))
        self.enviar_formulario(at, estatura="5")
        self.assertTrue(any("estatura debe estar entre" in e.value for e in at.error))
        self.assertEqual(self.tipos_en_tarjetas(at), [])

    def test_formulario_incompleto_pide_los_datos_que_faltan(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_RECOMENDADOR)
        self.enviar_formulario(at, nivel=None, objetivo=None)
        self.assertTrue(any("faltan: nivel de experiencia, objetivo principal" in w.value for w in at.warning))
        self.assertEqual(self.tipos_en_tarjetas(at), [])

    def test_limitacion_importante_no_muestra_hiit_y_advierte(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_RECOMENDADOR)
        self.enviar_formulario(at, objetivo="cardio", nivel="Avanzado", limitacion="importante")
        self.assertNotIn("HIIT", self.tipos_en_tarjetas(at))
        self.assertTrue(any("NO se recomiendan actividades intensas" in w.value for w in at.warning))

    def test_duracion_fuera_de_rango_no_da_calorias_y_avisa_una_vez(self):
        at = self.nueva_app(); self.ir_a(at, self.SECCION_RECOMENDADOR)
        self.enviar_formulario(at, duracion=20)
        avisos = [w.value for w in at.warning if "fuera del rango observado" in w.value]
        self.assertEqual(len(avisos), 1)
        self.assertEqual(sum('class="kcal-grande"' in m.value for m in at.markdown), 0)   # ninguna cifra de calorías


if __name__ == "__main__":
    unittest.main(verbosity=2)
