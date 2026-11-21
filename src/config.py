"""Configuración global del proyecto: rutas, semilla y definición de la variable objetivo."""
from pathlib import Path

# Rutas
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
POWERBI = ROOT / "powerbi"
RAW_FILE = DATA_RAW / "students_dropout.csv"

# Reproducibilidad
SEED = 42
TEST_SIZE = 0.20   # conjunto de prueba reservado hasta la evaluación final
CV_FOLDS = 5       # validación cruzada estratificada (OE3)

# Variable objetivo (decisión del 13/11/2026: formulación binaria)
TARGET_COL = "Target"
POSITIVE_CLASS = "Dropout"       # 1 = deserción
BINARY_TARGET = "deserta"        # 0 = Enrolled o Graduate

# Momentos de predicción (mejora 5.2): se filtran columnas por prefijo
SEM1_PREFIX = "Curricular units 1st sem"
SEM2_PREFIX = "Curricular units 2nd sem"
MOMENTS = ("matricula", "semestre_1", "semestre_2")

# UCI Machine Learning Repository, id del dataset
UCI_ID = 697

# Tipos de variables (Fase 2). Nombres en minúscula para comparar sin importar
# mayúsculas ni espacios; las demás predictoras se tratan como numéricas.
CATEGORICAL_VARS = {
    "marital status", "application mode", "course", "daytime/evening attendance",
    "previous qualification", "nacionality", "mother's qualification",
    "father's qualification", "mother's occupation", "father's occupation",
    "displaced", "educational special needs", "debtor", "tuition fees up to date",
    "gender", "scholarship holder", "international",
}
RARE_MIN_COUNT = 30   # categorías con menos registros se agrupan en "Otras" para chi cuadrado
ALPHA = 0.05          # nivel de significancia
HIGH_CORR = 0.80      # |rho de Spearman| a partir del cual se reporta redundancia
REPORTS = ROOT / "reports"
