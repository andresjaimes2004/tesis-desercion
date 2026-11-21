"""Fase 2 (OE1): diccionario de variables, análisis exploratorio y pruebas de asociación."""
import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ---------------------------------------------------------------- tipos y diccionario
def predictors(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (config.TARGET_COL, config.BINARY_TARGET)]


def variable_type(df: pd.DataFrame, col: str) -> str:
    if df[col].nunique() == 2:
        return "binaria"
    if col.strip().lower() in config.CATEGORICAL_VARS:
        return "categórica"
    return "numérica"


def moment_of(col: str) -> str:
    if col.startswith(config.SEM2_PREFIX):
        return "semestre_2"
    if col.startswith(config.SEM1_PREFIX):
        return "semestre_1"
    return "matricula"


def fetch_metadata() -> pd.DataFrame | None:
    """Descripciones oficiales de UCI (requiere internet). Devuelve None si no hay acceso."""
    try:
        from ucimlrepo import fetch_ucirepo

        meta = fetch_ucirepo(id=config.UCI_ID).variables
        meta["name"] = meta["name"].astype(str).str.strip()
        return meta
    except Exception as exc:  # sin conexión o cambio en la API
        print("No se pudieron descargar los metadatos de UCI:", exc)
        return None


def build_dictionary(df: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for col in predictors(df):
        s = df[col]
        rows.append({
            "variable": col,
            "tipo": variable_type(df, col),
            "momento": moment_of(col),
            "valores_distintos": s.nunique(),
            "minimo": s.min(),
            "maximo": s.max(),
            "faltantes": int(s.isna().sum()),
        })
    out = pd.DataFrame(rows)
    if metadata is not None:
        cols = [c for c in ("name", "type", "description") if c in metadata.columns]
        m = metadata[cols].rename(columns={"name": "variable", "type": "tipo_uci",
                                           "description": "descripcion_uci"})
        out = out.merge(m, on="variable", how="left")
    return out


# ---------------------------------------------------------------- pruebas de asociación
def group_rare(s: pd.Series, min_count: int = config.RARE_MIN_COUNT) -> pd.Series:
    """Agrupa categorías poco frecuentes para que la prueba chi cuadrado sea válida."""
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.astype(object).where(~s.isin(rare), "Otras")


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan


def assoc_categorical(x: pd.Series, y: pd.Series) -> dict:
    xg = group_rare(x)
    table = pd.crosstab(xg, y)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return {
        "prueba": "chi cuadrado",
        "estadistico": chi2,
        "gl": dof,
        "p_valor": p,
        "efecto": cramers_v(table, chi2),
        "medida_efecto": "V de Cramér",
        "categorias_usadas": table.shape[0],
        "pct_esperadas_menor_5": float((expected < 5).mean() * 100),
    }


def assoc_numeric(x: pd.Series, y: pd.Series) -> dict:
    g1, g0 = x[y == 1], x[y == 0]
    u, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    r_rb = 2 * u / (len(g1) * len(g0)) - 1  # > 0: valores mayores en quienes desertan
    return {
        "prueba": "U de Mann-Whitney",
        "estadistico": u,
        "gl": np.nan,
        "p_valor": p,
        "efecto": abs(r_rb),
        "direccion": "mayor en deserción" if r_rb > 0 else "menor en deserción",
        "r_rango_biserial": r_rb,
        "medida_efecto": "r rango-biserial (valor absoluto)",
        "mediana_deserta": g1.median(),
        "mediana_no_deserta": g0.median(),
    }


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """p-valores ajustados por tasa de falsos descubrimientos (Benjamini-Hochberg)."""
    p = p.astype(float)
    n = len(p)
    order = p.sort_values().index
    ranked = p.loc[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(upper=1)
    return adj.reindex(p.index)


def association_table(df: pd.DataFrame) -> pd.DataFrame:
    y = df[config.BINARY_TARGET]
    rows = []
    for col in predictors(df):
        kind = variable_type(df, col)
        res = assoc_categorical(df[col], y) if kind != "numérica" else assoc_numeric(df[col], y)
        rows.append({"variable": col, "tipo": kind, "momento": moment_of(col), **res})
    out = pd.DataFrame(rows)
    out["p_ajustado_bh"] = benjamini_hochberg(out["p_valor"])
    out["significativa"] = out["p_ajustado_bh"] < config.ALPHA
    return out.sort_values("efecto", ascending=False).reset_index(drop=True)


def dropout_rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tasa de deserción por categoría (tras agrupar las poco frecuentes)."""
    g = group_rare(df[col])
    t = df.groupby(g)[config.BINARY_TARGET].agg(n="size", tasa_desercion="mean")
    return t.sort_values("tasa_desercion", ascending=False)


def redundant_pairs(df: pd.DataFrame, threshold: float = config.HIGH_CORR) -> pd.DataFrame:
    num = [c for c in predictors(df) if variable_type(df, c) == "numérica"]
    rho = df[num].corr(method="spearman")
    pairs = [
        {"variable_1": a, "variable_2": b, "rho_spearman": rho.loc[a, b]}
        for i, a in enumerate(num) for b in num[i + 1:]
        if abs(rho.loc[a, b]) >= threshold
    ]
    cols = ["variable_1", "variable_2", "rho_spearman"]
    return pd.DataFrame(pairs, columns=cols).sort_values("rho_spearman", key=abs, ascending=False)
