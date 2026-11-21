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

# ---------------------------------------------------------------- Fase 3
# Indicadores de semestre sin unidades inscritas. El nombre empieza con el prefijo
# del semestre para que features_for_moment los asigne al momento correcto.
NO_ENROLL_SUFFIX = " (sin inscripcion)"

# Etiquetas en español para figuras y tablas
LABELS_ES = {
    "marital status": "Estado civil",
    "application mode": "Modalidad de ingreso",
    "application order": "Orden de preferencia",
    "course": "Programa académico",
    "daytime/evening attendance": "Jornada diurna",
    "previous qualification": "Formación previa",
    "previous qualification (grade)": "Nota de la formación previa",
    "nacionality": "Nacionalidad",
    "mother's qualification": "Formación de la madre",
    "father's qualification": "Formación del padre",
    "mother's occupation": "Ocupación de la madre",
    "father's occupation": "Ocupación del padre",
    "admission grade": "Puntaje de admisión",
    "displaced": "Desplazado",
    "educational special needs": "Necesidades educativas especiales",
    "debtor": "Deudor",
    "tuition fees up to date": "Matrícula al día",
    "gender": "Género",
    "scholarship holder": "Becario",
    "age at enrollment": "Edad al ingresar",
    "international": "Estudiante internacional",
    "unemployment rate": "Tasa de desempleo",
    "inflation rate": "Tasa de inflación",
    "gdp": "PIB",
}
UNITS_ES = {
    "credited": "Unidades acreditadas",
    "enrolled": "Unidades inscritas",
    "evaluations": "Evaluaciones",
    "approved": "Unidades aprobadas",
    "grade": "Nota promedio",
    "without evaluations": "Unidades sin evaluación",
    "sin inscripcion": "Sin unidades inscritas",
}
CLASSES_ES = {"Graduate": "Graduado", "Dropout": "Desertó", "Enrolled": "Matriculado"}


def etiqueta(col: str) -> str:
    """Nombre en español de una variable del conjunto."""
    c = col.strip()
    for prefix, sem in ((SEM1_PREFIX, "1.er sem."), (SEM2_PREFIX, "2.º sem.")):
        if c.startswith(prefix):
            unit = c[len(prefix):].strip(" ()").lower()
            return f"{UNITS_ES.get(unit, unit)}, {sem}"
    return LABELS_ES.get(c.lower(), c)
