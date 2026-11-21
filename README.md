# Sistema de detección temprana del riesgo de deserción estudiantil

Trabajo de grado – Ingeniería de Sistemas, Universidad de Pamplona.
Autor: Andres Mauricio Jaimes Revolledo · Directora: Fanny Casadiego Chiquillo.

Modelo de aprendizaje automático supervisado para estimar el riesgo de deserción
(formulación binaria: Dropout = 1; Enrolled y Graduate = 0) sobre el conjunto abierto
*Predict Students' Dropout and Academic Success* (UCI, id 697), con explicabilidad SHAP
y tablero en Power BI. Metodología: CRISP-DM.

## Estructura

| Carpeta | Contenido | Fase CRISP-DM |
| --- | --- | --- |
| `data/raw/` | Dataset original, sin modificar | Comprensión de los datos |
| `data/processed/` | Salidas de la preparación (se regeneran) | Preparación |
| `notebooks/` | Cuadernos numerados en orden de ejecución | Todas |
| `src/` | Funciones reutilizables (configuración, carga, utilidades) | Todas |
| `models/` | Modelo final serializado | Modelado / Evaluación |
| `reports/figures/` | Figuras que van al documento | Todas |
| `powerbi/` | Archivo .pbix y CSV de entrada | Despliegue |

## Cuadernos

| N.º | Cuaderno | Objetivo |
| --- | --- | --- |
| 00 | `00_configuracion_entorno.ipynb` | Entorno, semilla, descarga y verificación del dataset |
| 01 | `01_comprension_datos.ipynb` | OE1 |
| 02 | `02_preparacion.ipynb` | OE2 |
| 03 | `03_modelado.ipynb` | OE2 |
| 04 | `04_evaluacion_shap.ipynb` | OE3 |
| 05 | `05_exportacion_powerbi.ipynb` | OE4 |

## Ejecución en Google Colab

1. Abrir `notebooks/00_configuracion_entorno.ipynb` en Colab.
2. Ajustar `REPO_URL` en la primera celda con la URL del repositorio en GitHub.
3. Ejecutar todas las celdas. Las siguientes libretas se ejecutan en orden.

## Reproducibilidad

- Versiones mínimas en `requirements.txt`; las exactas usadas quedan en `requirements-lock.txt`.
- Semilla global `SEED = 42` en `src/config.py`.
- El conjunto de prueba (20 %, estratificado) se separa al inicio y no se usa hasta la evaluación final.
- El balanceo de clases se aplica solo dentro de cada partición de entrenamiento.

## Licencia de los datos

Conjunto de datos *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository,
https://doi.org/10.24432/C5MC89 — Licencia CC BY 4.0.
