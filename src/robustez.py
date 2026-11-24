"""Fase 5 (OE3), análisis de robustez solicitados por el jurado evaluador.

1. Fuga temporal: desertores que ya no presentaban evaluaciones en el semestre de corte y
   evaluación del modelo excluyéndolos.
2. Comparación estadística del AUC-ROC entre XGBoost y la regresión logística: prueba de DeLong
   (DeLong et al., 1988) con el algoritmo rápido de Sun y Xu (2014) y bootstrap pareado.
3. Sensibilidad del modelo de matrícula sin «Matrícula al día» y «Deudor».
4. Calibración: XGBoost sin ponderación y recalibración (Platt e isotónica) del modelo final.
5. Desempeño por subgrupos (revisión básica de equidad).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score

from . import config, evaluation as ev
from .data import features_for_moment

P1, P2 = config.SEM1_PREFIX, config.SEM2_PREFIX
TUITION, DEBTOR = "Tuition fees up to date", "Debtor"


def _cu(prefix, unit):
    return f"{prefix} ({unit})"


# ---------------------------------------------------------------- 1. fuga temporal
def withdrawn_mask(df, moment):
    """Estudiantes que, en el semestre de corte, inscribieron unidades pero no presentaron ninguna
    evaluación: señal de que ya habían abandonado cuando se emitiría la alerta."""
    if moment == "matricula":
        return pd.Series(False, index=df.index)
    p = P1 if moment == "semestre_1" else P2
    return (df[_cu(p, "enrolled")] > 0) & (df[_cu(p, "evaluations")] == 0)


def withdrawn_table(df):
    """Conteo de desertores y no desertores sin evaluaciones o sin inscripción en cada semestre."""
    y = df[config.BINARY_TARGET] == 1
    rows = []
    for sem, p in (("1.er semestre", P1), ("2.º semestre", P2)):
        enr, evals = df[_cu(p, "enrolled")], df[_cu(p, "evaluations")]
        appr = df[_cu(p, "approved")]
        for cond, m in (("Sin unidades inscritas", enr == 0),
                        ("Inscribió unidades sin presentar evaluaciones", (enr > 0) & (evals == 0)),
                        ("Sin unidades aprobadas", appr == 0)):
            rows.append({"semestre": sem, "condicion": cond,
                         "desertores": int((m & y).sum()), "pct_desertores": 100 * (m & y).sum() / y.sum(),
                         "no_desertores": int((m & ~y).sum()), "pct_no_desertores": 100 * (m & ~y).sum() / (~y).sum()})
    return pd.DataFrame(rows)


def evaluate_without_withdrawn(df, idx_train, idx_test, name, params):
    """Para cada momento: (a) el modelo y el umbral originales evaluados en la prueba sin los
    estudiantes ya retirados; (b) el modelo reentrenado sin ellos, con un umbral nuevo."""
    rows = []
    for moment in config.MOMENTS:
        cols = ev.columns(df, name, moment)
        tr, te = df.loc[idx_train], df.loc[idx_test]
        out_te = withdrawn_mask(te, moment)
        y_tr, y_te = tr[config.BINARY_TARGET], te[config.BINARY_TARGET]
        # (a) modelo original
        pipe = ev.final_pipeline(df, cols, name, params)
        p_oof = ev.oof_probabilities(pipe, tr[cols], y_tr)
        thr, _, _ = ev.choose_threshold(y_tr, p_oof)
        pipe.fit(tr[cols], y_tr)
        p_te = pipe.predict_proba(te[cols])[:, 1]
        keep = ~out_te.to_numpy()
        m = ev.metrics(y_te[keep], p_te[keep], thr)
        rows.append({"momento": moment, "variante": "modelo original, prueba sin retirados",
                     "excluidos_prueba": int(out_te.sum()), "desertores_prueba": int(y_te[keep].sum()), **m})
        # (b) reentrenado sin retirados
        out_tr = withdrawn_mask(tr, moment)
        tr2 = tr[~out_tr]
        pipe2 = ev.final_pipeline(df, cols, name, params)
        p_oof2 = ev.oof_probabilities(pipe2, tr2[cols], tr2[config.BINARY_TARGET])
        thr2, _, _ = ev.choose_threshold(tr2[config.BINARY_TARGET], p_oof2)
        pipe2.fit(tr2[cols], tr2[config.BINARY_TARGET])
        p_te2 = pipe2.predict_proba(te[cols])[:, 1]
        m2 = ev.metrics(y_te[keep], p_te2[keep], thr2)
        rows.append({"momento": moment, "variante": "reentrenado sin retirados",
                     "excluidos_prueba": int(out_te.sum()), "desertores_prueba": int(y_te[keep].sum()), **m2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. DeLong y bootstrap pareado
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y, p1, p2):
    """Prueba de DeLong para dos AUC correlacionados (mismos estudiantes). Devuelve AUC1, AUC2,
    diferencia, error estándar, estadístico z y valor p bilateral."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([np.asarray(p1), np.asarray(p2)])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds[:, np.r_[np.where(y)[0], np.where(~y)[0]]]])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diferencia": float(diff),
            "ee": se, "z": float(z), "p_valor": float(p)}


def paired_bootstrap_auc(y, p1, p2, n=1000, seed=config.SEED):
    """Bootstrap pareado: remuestrea estudiantes y recalcula la diferencia de AUC en la misma muestra."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    return {"diferencia_media": float(d.mean()), "ic95_inferior": float(np.percentile(d, 2.5)),
            "ic95_superior": float(np.percentile(d, 97.5)), "pct_diferencias_menor_o_igual_0": float(100 * (d <= 0).mean())}


def test_probabilities(df, idx_train, idx_test, name, params, moment="semestre_2", cols=None):
    """Probabilidades en la prueba de un modelo optimizado, entrenado con todo el entrenamiento."""
    cols = cols or ev.columns(df, name, moment)
    pipe = ev.final_pipeline(df, cols, name, params)
    pipe.fit(df.loc[idx_train, cols], df.loc[idx_train, config.BINARY_TARGET])
    return pipe.predict_proba(df.loc[idx_test, cols])[:, 1]


def compare_models(df, idx_train, idx_test, best_params, reference="Regresión logística", final="XGBoost"):
    """DeLong y bootstrap pareado del modelo final frente a cada uno de los otros modelos optimizados."""
    y = df.loc[idx_test, config.BINARY_TARGET].to_numpy()
    probs = {k: test_probabilities(df, idx_train, idx_test, k, v) for k, v in best_params.items()}
    rows = []
    for other in [m for m in best_params if m != final]:
        d = delong_test(y, probs[final], probs[other])
        b = paired_bootstrap_auc(y, probs[final], probs[other])
        rows.append({"comparacion": f"{final} vs. {other}", **d, **{f"boot_{k}": v for k, v in b.items()}})
    return pd.DataFrame(rows), probs


# ---------------------------------------------------------------- 3. sin variables financieras
def without_financial(df, idx_train, idx_test, name, params, moment="matricula", drop=(TUITION, DEBTOR)):
    """Modelo del momento indicado con y sin «Matrícula al día» y «Deudor» (mismo protocolo)."""
    rows = []
    base_cols = ev.columns(df, name, moment)
    for label, cols in (("con las variables financieras", base_cols),
                        ("sin Matrícula al día ni Deudor", [c for c in base_cols if c not in drop])):
        tr, te = df.loc[idx_train], df.loc[idx_test]
        pipe = ev.final_pipeline(df, cols, name, params)
        p_oof = ev.oof_probabilities(pipe, tr[cols], tr[config.BINARY_TARGET])
        thr, _, _ = ev.choose_threshold(tr[config.BINARY_TARGET], p_oof)
        pipe.fit(tr[cols], tr[config.BINARY_TARGET])
        p = pipe.predict_proba(te[cols])[:, 1]
        rows.append({"momento": moment, "variante": label, "variables": len(cols),
                     **ev.metrics(te[config.BINARY_TARGET], p, thr)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. calibración
def calibration_summary(y, p, n_bins=10):
    """Brier, calibración global (media predicha frente a prevalencia) e intercepto/pendiente de calibración."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    import statsmodels.api as sm
    slope_model = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
    int_model = sm.GLM(y, np.ones_like(logit), family=sm.families.Binomial(), offset=logit).fit()
    frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    ece = float(np.mean(np.abs(frac - mean)))
    return {"prob_media": float(p.mean()), "prevalencia": float(y.mean()),
            "intercepto_calibracion": float(int_model.params[0]),
            "pendiente_calibracion": float(slope_model.params[1]), "error_medio_deciles": ece}


def calibration_alternatives(df, idx_train, idx_test, name, params, moment="semestre_2"):
    """Compara el modelo final con (a) el mismo XGBoost sin ponderación y (b) el modelo final
    recalibrado con Platt (sigmoide) o regresión isotónica mediante CalibratedClassifierCV.
    En todos los casos el umbral se elige con probabilidades fuera de pliegue del entrenamiento."""
    cols = ev.columns(df, name, moment)
    tr, te = df.loc[idx_train], df.loc[idx_test]
    y_tr, y_te = tr[config.BINARY_TARGET], te[config.BINARY_TARGET]
    variants = {"Modelo final (ponderado)": ev.final_pipeline(df, cols, name, params),
                "Sin ponderación": ev.final_pipeline(df, cols, name, {**params, "scale_pos_weight": 1.0})}
    for method, label in (("sigmoid", "Ponderado + Platt"), ("isotonic", "Ponderado + isotónica")):
        variants[label] = CalibratedClassifierCV(ev.final_pipeline(df, cols, name, params),
                                                 method=method, cv=ev.cv())
    rows, probs = [], {}
    for label, model in variants.items():
        p_oof = ev.oof_probabilities(model, tr[cols], y_tr)
        thr, _, _ = ev.choose_threshold(y_tr, p_oof)
        model.fit(tr[cols], y_tr)
        p = model.predict_proba(te[cols])[:, 1]
        probs[label] = p
        rows.append({"variante": label, **ev.metrics(y_te, p, thr), **calibration_summary(y_te, p)})
    return pd.DataFrame(rows), probs


# ---------------------------------------------------------------- 5. subgrupos
def subgroup_performance(df, idx_test, p, threshold):
    """Sensibilidad y precisión del modelo por género, grupo de edad y jornada (revisión básica de equidad)."""
    te = df.loc[idx_test].copy()
    te["p"] = p
    te["alerta"] = (te["p"] >= threshold).astype(int)
    groups = {
        "Género": te["Gender"].map({1: "Hombre", 0: "Mujer"}),
        "Edad al ingresar": pd.cut(te["Age at enrollment"], [0, 19, 23, 100], labels=["17 a 19", "20 a 23", "24 o más"]),
        "Jornada": te["Daytime/evening attendance"].map({1: "Diurna", 0: "Nocturna"}),
    }
    rows = []
    for var, g in groups.items():
        for level, sub in te.groupby(g, observed=True):
            y, a = sub[config.BINARY_TARGET], sub["alerta"]
            tp = int(((a == 1) & (y == 1)).sum())
            rows.append({"variable": var, "grupo": str(level), "estudiantes": len(sub), "desertores": int(y.sum()),
                         "sensibilidad": tp / y.sum() if y.sum() else np.nan,
                         "precision": tp / a.sum() if a.sum() else np.nan,
                         "tasa_alerta": a.mean()})
    return pd.DataFrame(rows)
