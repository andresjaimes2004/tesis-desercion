"""Fase 5 (OE3): explicabilidad del modelo final con valores SHAP."""
import numpy as np
import pandas as pd
import shap

from . import config, prep


def transformed(pipe, X):
    """Datos tal como los ve el modelo (después de codificar y escalar), con nombres de columna."""
    pre = pipe.named_steps["prep"]
    return pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out(), index=X.index)


def origin_map(df, cols, feature_names):
    """Columna transformada -> variable original (las one-hot vuelven a su variable)."""
    cats = prep.column_groups(df, cols)["categórica"]
    out = {}
    for f in feature_names:
        out[f] = next((c for c in cats if f.startswith(c + "_")), f)
    return out


def readable(name, origin):
    """Nombre en español para figuras: 'Programa académico = 9119' o 'Matrícula al día'."""
    if name == origin:
        return config.etiqueta(name)
    code = name[len(origin) + 1:]
    code = "otras" if "infrequent" in code else code.removesuffix(".0")
    return f"{config.etiqueta(origin)} = {code}"


def shap_values(pipe, X_train, X_explain, seed=config.SEED):
    """Valores SHAP de la clase deserción, con el explicador adecuado a cada algoritmo.

    Árboles: TreeExplainer. Regresión logística: LinearExplainer. SVM: explicador por
    permutaciones sobre la probabilidad (más lento; conviene pasarle una muestra de filas).
    """
    model = pipe.named_steps["modelo"]
    Xt_train = transformed(pipe, X_train)
    Xt = transformed(pipe, X_explain)
    background = shap.sample(Xt_train, 100, random_state=seed)
    kind = type(model).__name__
    if kind in ("RandomForestClassifier", "XGBClassifier"):
        exp = shap.TreeExplainer(model)(Xt)
        if exp.values.ndim == 3:          # Random Forest devuelve una matriz por clase
            exp = exp[:, :, 1]
    elif kind == "LogisticRegression":
        exp = shap.LinearExplainer(model, background)(Xt)
    else:
        f = lambda z: model.predict_proba(pd.DataFrame(z, columns=Xt.columns))[:, 1]
        exp = shap.Explainer(f, background, algorithm="permutation", seed=seed)(Xt)
    exp.feature_names = list(Xt.columns)
    return exp, Xt


def global_importance(exp, df, cols):
    """Importancia global por variable original: media del valor absoluto de la suma de sus columnas."""
    names = list(exp.feature_names)
    omap = origin_map(df, cols, names)
    vals = pd.DataFrame(exp.values, columns=names)
    by_var = vals.T.groupby(pd.Series(omap)).sum().T
    imp = by_var.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"variable": imp.index, "etiqueta": [config.etiqueta(v) for v in imp.index],
                         "importancia_shap": imp.values})


def coherence_with_oe1(importance, path=None):
    """Correlación de Spearman entre la importancia SHAP y el tamaño del efecto de la Fase 2."""
    asoc = pd.read_csv(path or config.REPORTS / "asociacion_variables.csv")[["variable", "efecto"]]
    m = importance.merge(asoc, on="variable", how="inner")
    rho = m["importancia_shap"].corr(m["efecto"], method="spearman")
    return float(rho), len(m)


def pick_cases(y_test: pd.Series, p_test, threshold):
    """Dos casos ilustrativos (índices del conjunto): la deserción detectada con mayor riesgo y la
    deserción no detectada con menor riesgo, que es el error más claro del modelo."""
    p = pd.Series(np.asarray(p_test), index=y_test.index)
    tp = p[(y_test == 1) & (p >= threshold)]
    fn = p[(y_test == 1) & (p < threshold)]
    cases = {}
    if len(tp):
        cases["deserción detectada"] = tp.idxmax()
    if len(fn):
        cases["deserción no detectada"] = fn.idxmin()
    return cases


def raw_row(X_raw_row: pd.Series, feature_names, omap):
    """Valores para mostrar en la figura de un caso: el valor original de cada variable numérica o
    binaria y 1/0 en las columnas one-hot."""
    out = []
    for f in feature_names:
        o = omap[f]
        if f == o:
            v = X_raw_row[o]
            out.append(round(float(v), 2))
        else:
            code = f[len(o) + 1:]
            out.append(int("infrequent" not in code and str(X_raw_row[o]).removesuffix(".0") == code.removesuffix(".0")))
    return np.array(out, dtype=object)
