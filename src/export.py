"""Fase 6 (OE4): tablas para el tablero de Power BI.

Todas las probabilidades son fuera de muestra: las del entrenamiento salen de la validación
cruzada (cada estudiante es puntuado por un modelo que no lo vio) y las de la prueba, del modelo
entrenado con todo el entrenamiento. Los valores SHAP de cada estudiante salen del mismo modelo
que produjo su puntaje.
"""
import json

import numpy as np
import pandas as pd

from . import config, evaluation as ev, explain, prep

MOMENT_ES = {"matricula": "Matrícula", "semestre_1": "Cierre 1.er semestre", "semestre_2": "Cierre 2.º semestre"}
YES_NO = {1: "Sí", 0: "No"}

CODES_ES = {
    "Course": {
        33: "Tecnologías de Producción de Biocombustibles", 171: "Diseño de Animación y Multimedia",
        8014: "Servicio Social (nocturno)", 9003: "Agronomía", 9070: "Diseño de Comunicación",
        9085: "Enfermería Veterinaria", 9119: "Ingeniería Informática", 9130: "Equinicultura",
        9147: "Gestión", 9238: "Servicio Social", 9254: "Turismo", 9500: "Enfermería",
        9556: "Higiene Oral", 9670: "Gestión de Publicidad y Mercadeo",
        9773: "Periodismo y Comunicación", 9853: "Educación Básica", 9991: "Gestión (nocturno)"},
    "Application mode": {
        1: "1.ª fase, contingente general", 2: "Ordenanza 612/93",
        5: "1.ª fase, contingente especial (Azores)", 7: "Titulares de otros estudios superiores",
        10: "Ordenanza 854-B/99", 15: "Estudiante internacional (licenciatura)",
        16: "1.ª fase, contingente especial (Madeira)", 17: "2.ª fase, contingente general",
        18: "3.ª fase, contingente general", 26: "Ordenanza 533-A/99, b2 (plan diferente)",
        27: "Ordenanza 533-A/99, b3 (otra institución)", 39: "Mayores de 23 años", 42: "Transferencia",
        43: "Cambio de programa", 44: "Diploma de especialización tecnológica",
        51: "Cambio de institución o programa", 53: "Diploma de ciclo corto",
        57: "Cambio de institución o programa (internacional)"},
    "Marital Status": {1: "Soltero(a)", 2: "Casado(a)", 3: "Viudo(a)", 4: "Divorciado(a)",
                       5: "Unión de hecho", 6: "Separado(a) legalmente"},
    "Previous qualification": {
        1: "Educación secundaria", 2: "Superior: bachiller universitario", 3: "Superior: licenciatura",
        4: "Superior: maestría", 5: "Superior: doctorado", 6: "Estudios superiores sin terminar",
        9: "12.º año sin terminar", 10: "11.º año sin terminar", 12: "Otro: 11.º año",
        14: "10.º año", 15: "10.º año sin terminar", 19: "Básica, 3.er ciclo (9.º a 11.º año)",
        38: "Básica, 2.º ciclo (6.º a 8.º año)", 39: "Curso de especialización tecnológica",
        40: "Superior: licenciatura (1.er ciclo)", 42: "Curso técnico superior profesional",
        43: "Superior: maestría (2.º ciclo)"},
    "Nacionality": {
        1: "Portuguesa", 2: "Alemana", 6: "Española", 11: "Italiana", 13: "Neerlandesa", 14: "Inglesa",
        17: "Lituana", 21: "Angoleña", 22: "Caboverdiana", 24: "Guineana", 25: "Mozambiqueña",
        26: "Santotomense", 32: "Turca", 41: "Brasileña", 62: "Rumana", 100: "Moldava", 101: "Mexicana",
        103: "Ucraniana", 105: "Rusa", 108: "Cubana", 109: "Colombiana"},
}
BINARY_ES = {"Daytime/evening attendance": {1: "Diurna", 0: "Nocturna"},
             "Gender": {1: "Hombre", 0: "Mujer"}}


def _col(df, name):
    """Nombre real de la columna, sin importar mayúsculas (p. ej. 'Marital Status')."""
    return next(c for c in df.columns if c.lower() == name.lower())


def decode(df, var, value):
    var = _col(df, var)
    key = next((k for k in list(CODES_ES) + list(BINARY_ES) if k.lower() == var.lower()), None)
    if key in CODES_ES:
        return CODES_ES[key].get(int(value), f"Código {int(value)}")
    if key in BINARY_ES:
        return BINARY_ES[key][int(value)]
    if df[var].nunique() == 2:
        return YES_NO.get(int(value), value)
    return value


def students_table(df, idx_train, idx_test):
    """Una fila por estudiante, con etiquetas en español para segmentar el tablero."""
    out = pd.DataFrame({"ID": df.index})
    out["Resultado"] = df[config.TARGET_COL].map(config.CLASSES_ES).values
    out["Desertó"] = df[config.BINARY_TARGET].values
    out["Conjunto"] = np.where(df.index.isin(idx_test), "Prueba", "Entrenamiento")
    for var in ["Course", "Daytime/evening attendance", "Application mode", "Marital Status",
                "Previous qualification", "Nacionality", "Gender", "Displaced", "Debtor",
                "Tuition fees up to date", "Scholarship holder", "International",
                "Educational special needs"]:
        c = _col(df, var)
        out[config.etiqueta(c)] = [decode(df, c, v) for v in df[c]]
    age = df["Age at enrollment"]
    out["Edad al ingresar"] = age.values
    out["Rango de edad"] = pd.cut(age, [0, 19, 22, 29, 200],
                                  labels=["17 a 19", "20 a 22", "23 a 29", "30 o más"]).astype(str).values
    for c in ["Admission grade", "Previous qualification (grade)"]:
        if c in df:
            out[config.etiqueta(c)] = df[c].values
    for p in (config.SEM1_PREFIX, config.SEM2_PREFIX):
        for u in ("enrolled", "approved", "grade"):
            c = f"{p} ({u})"
            out[config.etiqueta(c)] = df[c].round(2).values
    return out


def _top_factors(exp, df, cols, X_raw, k=3):
    """Las k variables que más suben el riesgo de cada estudiante, en texto."""
    omap = explain.origin_map(df, cols, exp.feature_names)
    cats = set(prep.column_groups(df, cols)["categórica"])
    vals = pd.DataFrame(exp.values, columns=exp.feature_names, index=X_raw.index)
    by_var = vals.T.groupby(pd.Series(omap)).sum().T
    rows = []
    for idx, r in by_var.iterrows():
        top = r[r > 0].sort_values(ascending=False).head(k)
        txt = []
        for var in top.index:
            v = X_raw.at[idx, var]
            shown = decode(df, var, v)
            if var in cats and not isinstance(shown, str):
                shown = f"código {int(shown)}"
            elif isinstance(shown, (int, float, np.integer, np.floating)):
                shown = f"{float(shown):g}".replace(".", ",")
            txt.append(f"{config.etiqueta(var)}: {shown}")
        rows.append(txt + [""] * (k - len(txt)))
    return pd.DataFrame(rows, index=X_raw.index, columns=[f"Factor {i + 1}" for i in range(k)])


def predictions_table(df, idx_train, idx_test, spec):
    """Puntaje de riesgo fuera de muestra, alerta y factores principales, por estudiante y momento."""
    name, params = spec["algoritmo"], spec["hiperparametros"]
    y = df[config.BINARY_TARGET]
    tables = []
    for order, (moment, info) in enumerate(spec["momentos"].items(), start=1):
        cols, thr = info["variables"], info["umbral"]
        parts = []
        train_pos = pd.Index(idx_train)
        for fit_idx, pred_idx in ev.cv().split(df.loc[train_pos, cols], y.loc[train_pos]):
            fi, pi = train_pos[fit_idx], train_pos[pred_idx]
            parts.append(_score(df, cols, name, params, fi, pi, "Validación cruzada"))
        parts.append(_score(df, cols, name, params, train_pos, pd.Index(idx_test), "Prueba"))
        t = pd.concat(parts)
        t["Momento"] = MOMENT_ES[moment]
        t["Orden momento"] = order
        t["Umbral"] = round(thr * 100, 1)
        t["Alerta"] = np.where(t["Probabilidad"] >= thr, "Sí", "No")
        t["Desertó"] = y.loc[t.index].values
        tables.append(t)
    out = pd.concat(tables).reset_index(names="ID")
    out["Puntaje de riesgo"] = (out["Probabilidad"].astype(float) * 100).round(1)
    return out[["ID", "Momento", "Orden momento", "Origen", "Puntaje de riesgo", "Umbral", "Alerta",
                "Desertó", "Factor 1", "Factor 2", "Factor 3"]]


def _score(df, cols, name, params, fit_idx, pred_idx, origin):
    pipe = ev.final_pipeline(df, cols, name, params)
    pipe.fit(df.loc[fit_idx, cols], df.loc[fit_idx, config.BINARY_TARGET])
    X = df.loc[pred_idx, cols]
    p = pipe.predict_proba(X)[:, 1]
    exp, _ = explain.shap_values(pipe, df.loc[fit_idx, cols], X)
    f = _top_factors(exp, df, cols, X)
    f["Probabilidad"] = p
    f["Origen"] = origin
    return f


def calibration_table(pred):
    """Proporción observada de deserción por grupo de puntaje, solo en la prueba (lectura honesta
    del puntaje: 'de cada 100 estudiantes con este puntaje, X desertaron')."""
    t = pred[pred["Origen"] == "Prueba"].copy()
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
    labels = ["0–10", "10–20", "20–30", "30–40", "40–50", "50–60", "60–70", "70–80", "80–90", "90–100"]
    t["Rango de puntaje"] = pd.cut(t["Puntaje de riesgo"], bins, labels=labels, right=False).astype(str)
    g = t.groupby(["Momento", "Orden momento", "Rango de puntaje"], observed=True).agg(
        Estudiantes=("ID", "size"), Desertores=("Desertó", "sum"),
        **{"Puntaje promedio": ("Puntaje de riesgo", "mean")}).reset_index()
    g["Deserción observada (%)"] = (100 * g["Desertores"] / g["Estudiantes"]).round(1)
    g["Orden rango"] = g["Rango de puntaje"].map({l: i for i, l in enumerate(labels)})
    return g


def metrics_table():
    m = pd.read_csv(config.REPORTS / "evaluacion_momentos.csv")
    m["Momento"] = m["momento"].map(MOMENT_ES)
    m["Orden momento"] = m["momento"].map({k: i for i, k in enumerate(MOMENT_ES, start=1)})
    keep = {"exactitud": "Exactitud", "sensibilidad": "Sensibilidad", "precision": "Precisión", "f1": "F1",
            "auc_roc": "AUC-ROC", "brier": "Brier", "umbral": "Umbral", "vp": "Verdaderos positivos",
            "fp": "Falsos positivos", "fn": "Falsos negativos", "vn": "Verdaderos negativos"}
    return m[["Momento", "Orden momento", *keep]].rename(columns=keep)


def importance_table():
    s = pd.read_csv(config.REPORTS / "shap_importancia.csv")
    s["Posición"] = range(1, len(s) + 1)
    return s.rename(columns={"etiqueta": "Variable", "importancia_shap": "Importancia SHAP"})[
        ["Posición", "Variable", "Importancia SHAP"]]


def association_table():
    a = pd.read_csv(config.REPORTS / "asociacion_variables.csv")
    a["Variable"] = a["variable"].map(config.etiqueta)
    a["Momento"] = a["momento"].map(MOMENT_ES)
    a["Significativa"] = np.where(a["significativa"], "Sí", "No")
    return a.rename(columns={"tipo": "Tipo", "prueba": "Prueba", "efecto": "Tamaño del efecto",
                             "medida_efecto": "Medida", "p_ajustado_bh": "p ajustado"})[
        ["Variable", "Tipo", "Momento", "Prueba", "Tamaño del efecto", "Medida", "p ajustado", "Significativa"]]


def control_table(students, pred, metrics):
    """Cifras de control para verificar que el tablero muestre lo mismo que Python."""
    rows = [("Estudiantes", len(students)), ("Desertores", int(students["Desertó"].sum())),
            ("Tasa de deserción (%)", round(100 * students["Desertó"].mean(), 1))]
    for m, g in pred[pred["Origen"] == "Prueba"].groupby("Momento", sort=False):
        alert = g["Alerta"] == "Sí"
        rows += [(f"{m} | alertas en prueba", int(alert.sum())),
                 (f"{m} | desertores detectados en prueba", int((alert & (g["Desertó"] == 1)).sum()))]
    for _, r in metrics.iterrows():
        rows += [(f"{r['Momento']} | sensibilidad en prueba", round(r["Sensibilidad"], 3)),
                 (f"{r['Momento']} | precisión en prueba", round(r["Precisión"], 3))]
    return pd.DataFrame(rows, columns=["Indicador", "Valor esperado"])


def export_workbook(df, idx_train, idx_test, path=None):
    spec = json.loads((config.REPORTS / "modelo_final.json").read_text(encoding="utf-8"))
    students = students_table(df, idx_train, idx_test)
    pred = predictions_table(df, idx_train, idx_test, spec)
    metrics = metrics_table()
    sheets = {"Estudiantes": students, "Predicciones": pred, "Calibracion": calibration_table(pred),
              "Metricas": metrics, "ImportanciaSHAP": importance_table(), "Asociacion": association_table(),
              "Control": control_table(students, pred, metrics)}
    path = path or config.POWERBI / "tablero_desercion.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as xw:
        for name, t in sheets.items():
            t.to_excel(xw, sheet_name=name, index=False)
    return path, sheets
