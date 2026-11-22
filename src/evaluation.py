"""Fase 5 (OE3): optimización, selección, umbral de decisión y evaluación final en el conjunto de prueba."""
import json

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)

from . import config, models, prep
from .data import features_for_moment

CANDIDATES = ("Regresión logística", "Random Forest", "XGBoost", "SVM")  # decisión de la Fase 4
RECALL_TARGET = 0.80     # criterio de éxito: sensibilidad mínima
PRECISION_TARGET = 0.70  # criterio de éxito: precisión mínima
BRIER_BASELINE = 0.218   # 0,321 × 0,679: modelo que asigna a todos la prevalencia
N_ITER = {"Random Forest": 30, "XGBoost": 40, "SVM": 20}


def cv():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def columns(df, name, moment):
    """Variables del modelo según la decisión de la Fase 4 (conjunto reducido salvo excepción)."""
    comp = pd.read_csv(config.REPORTS / "comparacion_conjuntos.csv")
    row = comp[(comp.momento == moment) & (comp.modelo == name)]
    reduced = bool(row["reducido_aceptable"].iloc[0]) if len(row) else True
    return features_for_moment(df, moment, prep.non_significant() if reduced else ())


def search_space(name, pos_weight):
    """Espacio de búsqueda. La ponderación de clases entra como un hiperparámetro más."""
    if name == "Regresión logística":
        return {"modelo__C": list(np.logspace(-3, 2, 11)), "modelo__class_weight": [None, "balanced"]}
    if name == "Random Forest":
        return {"modelo__n_estimators": [200, 300, 500],
                "modelo__max_depth": [None, 5, 10, 15, 20, 30],
                "modelo__min_samples_leaf": [1, 2, 4, 8],
                "modelo__max_features": ["sqrt", 0.3, 0.5],
                "modelo__class_weight": [None, "balanced"]}
    if name == "XGBoost":
        return {"modelo__n_estimators": randint(100, 601),
                "modelo__learning_rate": loguniform(0.01, 0.3),
                "modelo__max_depth": randint(2, 9),
                "modelo__subsample": uniform(0.6, 0.4),
                "modelo__colsample_bytree": uniform(0.5, 0.5),
                "modelo__min_child_weight": randint(1, 11),
                "modelo__reg_lambda": loguniform(0.1, 10),
                "modelo__scale_pos_weight": [1.0, pos_weight]}
    if name == "SVM":
        return {"modelo__C": loguniform(0.1, 100), "modelo__gamma": loguniform(1e-4, 1e-1),
                "modelo__class_weight": [None, "balanced"]}
    raise ValueError(name)


def base_pipeline(df, cols, name, for_search=False):
    model = models.make_model(name)
    if name == "SVM" and for_search:
        model.set_params(probability=False)  # el AUC usa la función de decisión; mucho más rápido
    return prep.build_pipeline(df, cols, model, balance=None)


def tune(df, idx_train, moment="semestre_2", candidates=CANDIDATES, n_iter=None, verbose=True):
    """Búsqueda en malla (regresión logística) o aleatoria (demás) con AUC-ROC en validación cruzada."""
    n_iter = n_iter or N_ITER
    train = df.loc[idx_train]
    y = train[config.BINARY_TARGET]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    rows, best = [], {}
    for name in candidates:
        cols = columns(df, name, moment)
        pipe = base_pipeline(df, cols, name, for_search=True)
        space = search_space(name, pos_weight)
        if name == "Regresión logística":
            search = GridSearchCV(pipe, space, scoring="roc_auc", cv=cv(), n_jobs=-1)
        else:
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter[name], scoring="roc_auc", cv=cv(),
                                        random_state=config.SEED, n_jobs=-1)
        search.fit(train[cols], y)
        i = search.best_index_
        sd = search.cv_results_["std_test_score"][i]
        params = {k.replace("modelo__", ""): _plain(v) for k, v in search.best_params_.items()}
        rows.append({"modelo": name, "auc_cv": search.best_score_, "auc_de": sd,
                     "auc_ee": sd / config.CV_FOLDS ** 0.5, "combinaciones": len(search.cv_results_["params"]),
                     "variables": len(cols), "mejores_parametros": json.dumps(params, ensure_ascii=False)})
        best[name] = params
        if verbose:
            print(f"{name:19s} AUC CV = {search.best_score_:.4f} ± {sd:.4f} | {params}")
    return pd.DataFrame(rows).sort_values("auc_cv", ascending=False).reset_index(drop=True), best


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    return v


def select(tuned: pd.DataFrame) -> str:
    """Modelo final. Criterio de parsimonia fijado de antemano: si la regresión logística queda a
    menos de un error estándar del mejor AUC, se elige por ser el modelo más simple e interpretable."""
    top = tuned.iloc[0]
    lr = tuned[tuned.modelo == "Regresión logística"]
    if len(lr) and top["auc_cv"] - lr["auc_cv"].iloc[0] <= top["auc_ee"]:
        return "Regresión logística"
    return top["modelo"]


def final_pipeline(df, cols, name, params):
    pipe = base_pipeline(df, cols, name)
    pipe.set_params(**{f"modelo__{k}": v for k, v in params.items()})
    return pipe


def oof_probabilities(pipe, X, y):
    """Probabilidades fuera de pliegue sobre el entrenamiento, para elegir el umbral sin tocar la prueba."""
    return cross_val_predict(pipe, X, y, cv=cv(), method="predict_proba")[:, 1]


def choose_threshold(y, p, recall_target=RECALL_TARGET):
    """Umbral más alto que mantiene la sensibilidad objetivo: maximiza la precisión dentro de esa condición."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= recall_target)[0]
    i = ok[-1] if len(ok) else 0
    return float(thr[i]), float(prec[i]), float(rec[i])


def metrics(y, p, threshold):
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"umbral": threshold, "exactitud": accuracy_score(y, yhat), "sensibilidad": recall_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat),
            "auc_roc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn)}


def bootstrap_ci(y, p, threshold, n=1000, seed=config.SEED):
    """Intervalos de confianza del 95 % por remuestreo bootstrap de percentiles sobre la prueba."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    keys = ["sensibilidad", "precision", "f1", "auc_roc", "brier"]
    draws = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m = metrics(y[i], p[i], threshold)
        for k in keys:
            draws[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


def success_table(m):
    rows = [("Sensibilidad ≥ 0,80", m["sensibilidad"], m["sensibilidad"] >= RECALL_TARGET),
            ("Precisión ≥ 0,70", m["precision"], m["precision"] >= PRECISION_TARGET),
            ("Brier < 0,218", m["brier"], m["brier"] < BRIER_BASELINE)]
    return pd.DataFrame(rows, columns=["criterio", "valor", "cumple"])


def evaluate_moments(df, idx_train, idx_test, name, params):
    """El algoritmo final, con sus hiperparámetros, en los tres momentos: umbral elegido en el
    entrenamiento (fuera de pliegue) y evaluación única en la prueba."""
    rows, outputs = [], {}
    tr, te = df.loc[idx_train], df.loc[idx_test]
    for moment in config.MOMENTS:
        cols = columns(df, name, moment)
        pipe = final_pipeline(df, cols, name, params)
        p_oof = oof_probabilities(pipe, tr[cols], tr[config.BINARY_TARGET])
        thr, _, _ = choose_threshold(tr[config.BINARY_TARGET], p_oof)
        pipe.fit(tr[cols], tr[config.BINARY_TARGET])
        p_test = pipe.predict_proba(te[cols])[:, 1]
        m = metrics(te[config.BINARY_TARGET], p_test, thr)
        rows.append({"momento": moment, "variables": len(cols), **m})
        outputs[moment] = {"pipeline": pipe, "cols": cols, "p_test": p_test, "umbral": thr}
    return pd.DataFrame(rows), outputs


def save_final(name, params, outputs, path=None):
    """Especificación del modelo final (se reentrena en segundos en la Fase 6)."""
    spec = {"algoritmo": name, "hiperparametros": params,
            "momentos": {m: {"umbral": o["umbral"], "variables": o["cols"]} for m, o in outputs.items()}}
    (path or config.REPORTS / "modelo_final.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    return spec
