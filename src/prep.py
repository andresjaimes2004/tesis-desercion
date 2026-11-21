"""Fase 3: verificación de consistencia, partición y preprocesamiento sin fuga de información."""
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, eda
from .data import features_for_moment

SEMESTERS = {"1.er semestre": config.SEM1_PREFIX, "2.º semestre": config.SEM2_PREFIX}


def _cu(prefix: str, unit: str) -> str:
    return f"{prefix} ({unit})"


# ---------------------------------------------------------------- verificación
def consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Reglas lógicas entre variables. Solo cuenta incumplimientos; no borra registros."""
    rows = []
    for sem, p in SEMESTERS.items():
        enr, app, grd = (df[_cu(p, u)] for u in ("enrolled", "approved", "grade"))
        rules = {
            "Unidades aprobadas mayores que las inscritas": app > enr,
            "Nota promedio fuera de la escala 0 a 20": (grd < 0) | (grd > 20),
            "Nota promedio mayor que cero sin unidades aprobadas": (grd > 0) & (app == 0),
            "Unidades aprobadas sin nota promedio": (app > 0) & (grd == 0),
        }
        rows += [{"semestre": sem, "regla": r, "incumplimientos": int(m.sum())} for r, m in rules.items()]
    for col, lo, hi in (("Age at enrollment", 15, 80), ("Admission grade", 0, 200),
                        ("Previous qualification (grade)", 0, 200)):
        if col in df:
            rows.append({"semestre": "matrícula", "regla": f"{config.etiqueta(col)} fuera de [{lo}, {hi}]",
                         "incumplimientos": int(((df[col] < lo) | (df[col] > hi)).sum())})
    return pd.DataFrame(rows)


def no_enrollment_by_course(df: pd.DataFrame) -> pd.DataFrame:
    """Estudiantes sin unidades inscritas en ninguno de los dos semestres, por programa y resultado."""
    mask = (df[_cu(config.SEM1_PREFIX, "enrolled")] == 0) & (df[_cu(config.SEM2_PREFIX, "enrolled")] == 0)
    t = pd.crosstab(df.loc[mask, "Course"], df.loc[mask, config.TARGET_COL])
    t["total_sin_inscripcion"] = t.sum(axis=1)
    t["estudiantes_programa"] = df["Course"].value_counts().reindex(t.index)
    t["pct_del_programa"] = (100 * t["total_sin_inscripcion"] / t["estudiantes_programa"]).round(1)
    return t.sort_values("total_sin_inscripcion", ascending=False)


def add_no_enrollment_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Indicador 1/0 de semestre sin unidades inscritas.

    En esos registros la nota y las unidades aprobadas valen 0 aunque no hubo cursos,
    y ese 0 se confundiría con un semestre perdido. El indicador separa ambos casos.
    """
    out = df.copy()
    for p in SEMESTERS.values():
        out[p + config.NO_ENROLL_SUFFIX] = (out[_cu(p, "enrolled")] == 0).astype(int)
    return out


# ---------------------------------------------------------------- partición
def split(df: pd.DataFrame):
    """Partición estratificada entrenamiento/prueba con la semilla del proyecto."""
    idx_train, idx_test = train_test_split(
        df.index, test_size=config.TEST_SIZE, stratify=df[config.BINARY_TARGET],
        random_state=config.SEED)
    return idx_train, idx_test


def save_split(idx_train, idx_test, path=None) -> pd.DataFrame:
    path = path or config.REPORTS / "particion.csv"
    part = pd.concat([pd.DataFrame({"indice": idx_train, "conjunto": "entrenamiento"}),
                      pd.DataFrame({"indice": idx_test, "conjunto": "prueba"})]).sort_values("indice")
    part.to_csv(path, index=False)
    return part


def load_split(path=None):
    part = pd.read_csv(path or config.REPORTS / "particion.csv")
    return (part.loc[part.conjunto == "entrenamiento", "indice"].to_numpy(),
            part.loc[part.conjunto == "prueba", "indice"].to_numpy())


def split_summary(df, idx_train, idx_test) -> pd.DataFrame:
    y = df[config.BINARY_TARGET]
    rows = []
    for name, idx in (("entrenamiento", idx_train), ("prueba", idx_test), ("total", df.index)):
        rows.append({"conjunto": name, "registros": len(idx), "desercion": int(y.loc[idx].sum()),
                     "pct_desercion": round(100 * y.loc[idx].mean(), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- variables
def non_significant(path=None) -> list[str]:
    """Variables sin asociación significativa en la Fase 2 (candidatas a excluir)."""
    asoc = pd.read_csv(path or config.REPORTS / "asociacion_variables.csv")
    return asoc.loc[~asoc["significativa"], "variable"].tolist()


def column_groups(df: pd.DataFrame, cols: list[str]) -> dict:
    groups = {"categórica": [], "binaria": [], "numérica": []}
    for c in cols:
        groups[eda.variable_type(df, c)].append(c)
    return groups


# ---------------------------------------------------------------- preprocesamiento
def build_preprocessor(df: pd.DataFrame, cols: list[str], scale: bool = True) -> ColumnTransformer:
    """Codificación one-hot de categóricas (agrupando las poco frecuentes) y escalado de numéricas.

    Todo se ajusta con los datos de entrenamiento de cada pliegue, nunca con los de prueba.
    """
    g = column_groups(df, cols)
    return ColumnTransformer([
        ("categoricas", OneHotEncoder(min_frequency=config.RARE_MIN_COUNT,
                                      handle_unknown="infrequent_if_exist", sparse_output=False),
         g["categórica"]),
        ("binarias", "passthrough", g["binaria"]),
        ("numericas", StandardScaler() if scale else "passthrough", g["numérica"]),
    ], verbose_feature_names_out=False)


def build_pipeline(df: pd.DataFrame, cols: list[str], model, balance: str | None = "smote",
                   scale: bool = True) -> Pipeline:
    """Pipeline completo: [SMOTE-NC] -> preprocesamiento -> modelo.

    balance="smote": sobremuestreo SMOTE-NC dentro del pipeline (solo actúa al entrenar).
    balance="pesos" o None: sin sobremuestreo; la ponderación de clases se define en el modelo.
    """
    steps = []
    if balance == "smote":
        g = column_groups(df, cols)
        cat_pos = [cols.index(c) for c in g["categórica"] + g["binaria"]]
        steps.append(("smote", SMOTENC(categorical_features=cat_pos, random_state=config.SEED)))
    steps += [("prep", build_preprocessor(df, cols, scale)), ("modelo", model)]
    return Pipeline(steps)


def preprocessing_summary(df: pd.DataFrame, idx_train, exclude=()) -> pd.DataFrame:
    """Número de variables y de columnas tras la codificación, por momento (ajustado en entrenamiento)."""
    rows = []
    train = df.loc[idx_train]
    for m in config.MOMENTS:
        cols = features_for_moment(df, m, exclude)
        g = column_groups(df, cols)
        pre = build_preprocessor(df, cols).fit(train[cols])
        enc = pre.named_transformers_["categoricas"]
        grouped = sum(len(x) for x in (enc.infrequent_categories_ or []) if x is not None)
        rows.append({"momento": m, "variables": len(cols), "categoricas": len(g["categórica"]),
                     "binarias": len(g["binaria"]), "numericas": len(g["numérica"]),
                     "categorias_agrupadas": grouped,
                     "columnas_tras_codificar": len(pre.get_feature_names_out())})
    return pd.DataFrame(rows)


def smote_demo(df: pd.DataFrame, idx_train, moment: str = "semestre_2") -> pd.DataFrame:
    """Distribución de clases antes y después de SMOTE-NC sobre el entrenamiento (solo ilustrativo)."""
    cols = features_for_moment(df, moment)
    g = column_groups(df, cols)
    sm = SMOTENC(categorical_features=[cols.index(c) for c in g["categórica"] + g["binaria"]],
                 random_state=config.SEED)
    X, y = df.loc[idx_train, cols], df.loc[idx_train, config.BINARY_TARGET]
    _, y_res = sm.fit_resample(X, y)
    return pd.DataFrame({"antes": y.value_counts().sort_index(),
                         "despues": pd.Series(y_res).value_counts().sort_index()}).rename(
        index={0: "no deserta", 1: "deserta"})
