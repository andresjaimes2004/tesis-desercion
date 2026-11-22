# Sistema de detección temprana del riesgo de deserción estudiantil

Trabajo de grado – Ingeniería de Sistemas, Universidad de Pamplona.
Autor: Andres Mauricio Jaimes Revolledo · Directora: Fanny Casadiego Chiquillo.

Modelo de aprendizaje automático supervisado para estimar el riesgo de deserción
(formulación binaria: Dropout = 1; Enrolled y Graduate = 0) sobre el conjunto abierto
*Predict Students' Dropout and Academic Success* (UCI, id 697; 4.424 estudiantes), con
explicabilidad SHAP y tablero en Power BI. Metodología: CRISP-DM.

## Objetivos específicos

| Objetivo | Contenido | Cuadernos |
| --- | --- | --- |
| OE1 | Análisis exploratorio y pruebas de asociación con la deserción | 01 |
| OE2 | Preparación de los datos y entrenamiento de modelos de clasificación | 02, 03 |
| OE3 | Evaluación, selección del modelo final y explicabilidad SHAP | 04 |
| OE4 | Tablero de visualización en Power BI | 05 y `powerbi/` |

## Resultados principales

Modelo final: **XGBoost** con ponderación de clases (`scale_pos_weight` = 2,11). Se evalúa en
tres momentos de predicción, cada uno con las variables disponibles hasta ese punto y su propio
umbral de decisión, elegido con el conjunto de entrenamiento.

| Momento | Variables | Umbral | AUC-ROC | Sensibilidad | Precisión |
| --- | --- | --- | --- | --- | --- |
| Matrícula | 19 | 0,352 | 0,839 | 0,827 | 0,503 |
| Cierre del 1.er semestre | 31 | 0,419 | 0,916 | 0,824 | 0,716 |
| Cierre del 2.º semestre | 31 | 0,440 | 0,935 | 0,845 | 0,779 |

Métricas en el conjunto de prueba (885 estudiantes). Al cierre del 2.º semestre se cumplen los
criterios de éxito: sensibilidad ≥ 0,80, precisión ≥ 0,70 y Brier < 0,218 (0,090). Detalle en
`reports/evaluacion_momentos.csv`, `reports/criterios_exito.csv` e `reports/intervalos_confianza.csv`.

## Estructura

| Carpeta | Contenido | Fase CRISP-DM |
| --- | --- | --- |
| `notebooks/` | Cuadernos numerados en orden de ejecución | 2 a 6 |
| `src/` | Funciones que usan los cuadernos (ver tabla siguiente) | 2 a 6 |
| `reports/` | Tablas de resultados (CSV), especificación del modelo final (`modelo_final.json`) y entorno (`entorno.json`) | 2 a 5 |
| `reports/figures/` | Figuras que van al documento | 2 a 5 |
| `powerbi/` | Tablero (.pbix), su exportación en PDF y Excel de entrada | 6 |
| `data/` | Se crea al ejecutar: `raw/` guarda el dataset descargado de UCI y `processed/` las salidas de la preparación. No se versiona | 2 y 3 |

El modelo final no se guarda serializado: `reports/modelo_final.json` registra el algoritmo, los
hiperparámetros, el umbral y las variables de cada momento, y el cuaderno 05 lo reentrena con la
misma semilla.

| Módulo de `src/` | Uso |
| --- | --- |
| `config.py` | Rutas, semilla, variable objetivo, momentos y etiquetas en español |
| `data.py` | Descarga y carga del dataset, binarización del objetivo y variables por momento |
| `eda.py`, `plots.py` | Diccionario de variables, pruebas de asociación y figuras (OE1) |
| `prep.py` | Verificación de consistencia, partición y preprocesamiento sin fuga de información |
| `models.py` | Modelos base y comparación de estrategias frente al desbalance (OE2) |
| `evaluation.py` | Optimización, selección del modelo, umbral y evaluación final (OE3) |
| `explain.py` | Valores SHAP del modelo final (OE3) |
| `export.py` | Tablas del archivo de Power BI (OE4) |
| `utils.py` | Semilla y registro del entorno |

## Cuadernos

| N.º | Cuaderno | Fase CRISP-DM | Qué hace |
| --- | --- | --- | --- |
| 00 | `00_configuracion_entorno.ipynb` | — | Instala dependencias, fija la semilla, descarga el dataset y verifica sus cifras |
| 01 | `01_comprension_datos.ipynb` | 2. Comprensión de los datos | Diccionario de variables, descriptivos y pruebas de asociación (OE1) |
| 02 | `02_preparacion.ipynb` | 3. Preparación de los datos | Consistencia, partición 80/20 y variables por momento (OE2) |
| 03 | `03_modelado.ipynb` | 4. Modelado | Cinco modelos base × tres estrategias frente al desbalance × tres momentos (OE2) |
| 04 | `04_evaluacion_shap.ipynb` | 5. Evaluación | Optimización, modelo final, umbral, evaluación en prueba y SHAP (OE3) |
| 05 | `05_exportacion_powerbi.ipynb` | 6. Despliegue | Genera `powerbi/tablero_desercion.xlsx` y verifica sus cifras (OE4) |

La Fase 1 (comprensión del negocio) se desarrolla en el documento del trabajo de grado.

## Ejecución en Google Colab

1. Abrir `notebooks/00_configuracion_entorno.ipynb` en Colab y ejecutar todas las celdas.
   La primera celda clona este repositorio; si Colab pide reiniciar la sesión, reiniciar y continuar.
2. Ejecutar los demás cuadernos en orden (01 a 05). Cada uno prepara su propio entorno al inicio.
3. Los cuadernos 03, 04 y 05 terminan empaquetando sus resultados en un .zip; su contenido se sube
   a `reports/` (y a `powerbi/` en el caso del 05) para conservarlo en el repositorio.

## Tablero en Power BI

- Archivo: `powerbi/tablero_desercion.pbix`, hecho en Power BI Desktop 2.159.1231.0 (64 bits).
  Quien no tenga Power BI puede consultar `powerbi/tablero_desercion.pdf`.
- Fuente de datos: `powerbi/tablero_desercion.xlsx` (hojas Estudiantes, Predicciones, Calibracion,
  Metricas, ImportanciaSHAP, Asociacion y Control). Si el Excel cambia de ubicación, actualizar la
  ruta en Transformar datos > Configuración de origen de datos > Cambiar origen.
- Páginas: panorama general, factores socioeconómicos y de ingreso, rendimiento académico en el
  primer año y riesgo predicho por estudiante, con un segmentador de programa sincronizado entre ellas.
- El puntaje de riesgo (0–100) ordena a los estudiantes, pero no es una probabilidad: el modelo
  sobrestima el riesgo porque se entrenó con ponderación de clases. La página de riesgo incluye la
  tasa real de deserción por tramo de puntaje.
- Todas las predicciones son fuera de muestra: validación cruzada de 5 pliegues para el
  entrenamiento y el modelo entrenado completo para la prueba.

## Reproducibilidad

- Semilla global `SEED = 42` en `src/config.py`.
- El conjunto de prueba (20 %, estratificado) se separa al inicio y no se usa hasta la evaluación final.
- Las estrategias frente al desbalance (sin balanceo, SMOTE-NC y ponderación de clases) se comparan
  dentro de cada partición de entrenamiento de la validación cruzada; el modelo final usa ponderación.
- Versiones mínimas en `requirements.txt`; las exactas usadas quedan en `requirements-lock.txt` y
  `reports/entorno.json` (Python 3.13.18, scikit-learn 1.6.1, XGBoost 3.4.1, SHAP 0.54.1).

## Licencia de los datos

Conjunto de datos *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository,
https://doi.org/10.24432/C5MC89 — Licencia CC BY 4.0.
