"""Fase 4 (OE2): modelos base y comparación de estrategias de balanceo con validación cruzada."""
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config, prep
from .data import features_for_moment

MODEL_NAMES = ("Regresión logística", "Árbol de decisión", "Random Forest", "XGBoost", "SVM")
STRATEGIES = ("sin balanceo", "SMOTE-NC", "ponderación")
SCORING = {
    "exactitud": "accuracy",
    "sensibilidad": "recall",
    "precision": "precision",
    "f1": "f1",
    "auc_roc": "roc_auc",
    "brier": "neg_brier_score",
}
METRICS = tuple(SCORING)


def make_model(name: str, weighted: bool = False, pos_weight: float = 1.0):
    """Modelo base con hiperparámetros por defecto. weighted=True activa la ponderación de clases."""
    cw = "balanced" if weighted else None
    s = config.SEED
    if name == "Regresión logística":
        return LogisticRegression(max_iter=5000, class_weight=cw)  # penalización L2 (ridge)
    if name == "Árbol de decisión":
        return DecisionTreeClassifier(class_weight=cw, random_state=s)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=s)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.1, eval_metric="logloss",
                             scale_pos_weight=pos_weight if weighted else 1.0,
                             n_jobs=-1, random_state=s)
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, class_weight=cw, random_state=s)
    raise ValueError(f"Modelo no reconocido: {name}")


def make_pipeline(df, cols, name, strategy, pos_weight):
    model = make_model(name, weighted=(strategy == "ponderación"), pos_weight=pos_weight)
    return prep.build_pipeline(df, cols, model, balance="smote" if strategy == "SMOTE-NC" else None)


def cv_base_models(df, idx_train, moments=config.MOMENTS, feature_sets=None,
                   models=MODEL_NAMES, strategies=STRATEGIES, verbose=True) -> pd.DataFrame:
    """Validación cruzada estratificada de 5 pliegues sobre el entrenamiento.

    Devuelve una fila por pliegue con todas las métricas. El conjunto de prueba no se usa.
    """
    feature_sets = feature_sets or {"completo": ()}
    train = df.loc[idx_train]
    y = train[config.BINARY_TARGET]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    rows = []
    for moment in moments:
        for set_name, exclude in feature_sets.items():
            cols = features_for_moment(df, moment, exclude)
            for name in models:
                for strategy in strategies:
                    t0 = time.time()
                    pipe = make_pipeline(df, cols, name, strategy, pos_weight)
                    res = cross_validate(pipe, train[cols], y, cv=cv, scoring=SCORING, n_jobs=1)
                    for k in range(config.CV_FOLDS):
                        row = {"momento": moment, "conjunto": set_name, "modelo": name,
                               "estrategia": strategy, "pliegue": k + 1}
                        for m in METRICS:
                            v = res[f"test_{m}"][k]
                            row[m] = -v if m == "brier" else v
                        rows.append(row)
                    if verbose:
                        print(f"{moment:10s} | {set_name:22s} | {name:19s} | {strategy:12s} | "
                              f"sens={res['test_sensibilidad'].mean():.3f} auc={res['test_auc_roc'].mean():.3f} "
                              f"({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar de cada métrica entre pliegues."""
    keys = ["momento", "conjunto", "modelo", "estrategia"]
    agg = folds.groupby(keys, sort=False)[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{'media' if s == 'mean' else 'de'}" for m, s in agg.columns]
    return agg.reset_index()


def ranking(summary: pd.DataFrame, moment="semestre_2", feature_set="completo") -> pd.DataFrame:
    """Mejor estrategia de cada modelo según el AUC-ROC (métrica independiente del umbral)."""
    t = summary.query("momento == @moment and conjunto == @feature_set")
    best = t.loc[t.groupby("modelo")["auc_roc_media"].idxmax()]
    return best.sort_values("auc_roc_media", ascending=False).reset_index(drop=True)


def feature_set_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """AUC-ROC del conjunto completo frente al reducido, con la mejor estrategia de cada modelo.

    Criterio de parsimonia, fijado antes de ver los resultados: si el AUC del conjunto
    reducido no queda más de un error estándar (de / raíz de los pliegues) por debajo del
    completo, se prefiere el reducido por ser más simple.
    """
    keys = ["momento", "modelo", "conjunto"]
    best = summary.loc[summary.groupby(keys, sort=False)["auc_roc_media"].idxmax()]
    wide = best.pivot_table(index=["momento", "modelo"], columns="conjunto",
                            values=["auc_roc_media", "auc_roc_de"])
    out = pd.DataFrame({
        "auc_completo": wide[("auc_roc_media", "completo")],
        "ee_completo": wide[("auc_roc_de", "completo")] / config.CV_FOLDS ** 0.5,
        "auc_reducido": wide[("auc_roc_media", "sin no significativas")],
    })
    out["diferencia"] = out["auc_reducido"] - out["auc_completo"]
    out["reducido_aceptable"] = out["diferencia"] >= -out["ee_completo"]
    return out.reset_index()
