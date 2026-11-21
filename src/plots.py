"""Figuras de la Fase 2 con estilo uniforme y etiquetas en español para el documento."""
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

COLORS = {0: "#4C72B0", 1: "#C44E52"}
LABELS = {0: "No deserta", 1: "Deserta"}


def _save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    counts = df[config.TARGET_COL].value_counts()
    axes[0].bar([config.CLASSES_ES.get(k, k) for k in counts.index], counts.values, color="#8C8C8C")
    axes[0].set_title("Variable original (tres clases)")
    b = df[config.BINARY_TARGET].value_counts().sort_index()
    axes[1].bar([LABELS[i] for i in b.index], b.values, color=[COLORS[i] for i in b.index])
    axes[1].set_title("Variable objetivo binaria")
    for ax in axes:
        ax.set_ylabel("Estudiantes")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height()):,}".replace(",", "."),
                        (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "fig_01_distribucion_objetivo")


def effect_sizes(assoc, top=15):
    t = assoc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1))
    colors = ["#C44E52" if k == "numérica" else "#4C72B0" for k in t["tipo"]]
    ax.barh([config.etiqueta(v) for v in t["variable"]], t["efecto"], color=colors)
    ax.set_xlabel("Tamaño del efecto (V de Cramér o |r rango-biserial|)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#C44E52", label="Numérica (Mann-Whitney)"),
                       Patch(color="#4C72B0", label="Categórica o binaria (chi cuadrado)")],
              loc="lower right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "fig_02_tamano_efecto")


def numeric_by_group(df, cols, name="fig_03_numericas_por_grupo"):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    axes = [axes] if n == 1 else axes
    for ax, c in zip(axes, cols):
        sns.boxplot(data=df, x=config.BINARY_TARGET, y=c, hue=config.BINARY_TARGET,
                    palette=COLORS, legend=False, ax=ax, fliersize=1.5)
        ax.set_xticks([0, 1], [LABELS[0], LABELS[1]])
        ax.set_xlabel("")
        ax.set_title(config.etiqueta(c), fontsize=9)
        ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, name)


def categorical_rates(rates: dict, name="fig_04_tasa_por_categoria"):
    n = len(rates)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8))
    axes = [axes] if n == 1 else axes
    for ax, (col, t) in zip(axes, rates.items()):
        ax.bar(t.index.astype(str), t["tasa_desercion"] * 100, color="#C44E52")
        ax.set_title(config.etiqueta(col), fontsize=9)
        ax.set_ylabel("Tasa de deserción (%)")
        ax.set_xlabel("Código de categoría", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)


def spearman_heatmap(df, cols, name="fig_05_correlacion_spearman"):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    corr = df[cols].corr(method="spearman").rename(index=config.etiqueta, columns=config.etiqueta)
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1,
                ax=ax, cbar_kws={"label": "rho de Spearman"})
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- Fase 4
STRAT_COLORS = {"sin balanceo": "#8C8C8C", "SMOTE-NC": "#C44E52", "ponderación": "#4C72B0"}


def strategy_comparison(summary, moment="semestre_2", feature_set="completo",
                        name="fig_06_estrategias_balanceo"):
    """Sensibilidad, precisión y AUC-ROC de cada modelo según la estrategia de balanceo."""
    import numpy as np
    t = summary.query("momento == @moment and conjunto == @feature_set")
    models = list(dict.fromkeys(t["modelo"]))
    strategies = list(dict.fromkeys(t["estrategia"]))
    metrics = [("sensibilidad", "Sensibilidad"), ("precision", "Precisión"), ("auc_roc", "AUC-ROC")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(len(models))
    w = 0.8 / len(strategies)
    for ax, (m, title) in zip(axes, metrics):
        for i, st in enumerate(strategies):
            r = t[t["estrategia"] == st].set_index("modelo").reindex(models)
            ax.bar(x + (i - (len(strategies) - 1) / 2) * w, r[f"{m}_media"], w,
                   yerr=r[f"{m}_de"], capsize=2, color=STRAT_COLORS.get(st), label=st)
        ax.set_xticks(x, models, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Media en validación cruzada")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(strategies), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, name)


def moments_comparison(summary, feature_set="completo", name="fig_07_auc_por_momento"):
    """AUC-ROC de cada modelo en los tres momentos de predicción (mejor estrategia de cada uno)."""
    t = summary.query("conjunto == @feature_set")
    best = t.loc[t.groupby(["momento", "modelo"], sort=False)["auc_roc_media"].idxmax()]
    order = list(config.MOMENTS)
    labels = {"matricula": "Matrícula", "semestre_1": "Cierre 1.er sem.", "semestre_2": "Cierre 2.º sem."}
    fig, ax = plt.subplots(figsize=(7, 4))
    for mod, g in best.groupby("modelo", sort=False):
        g = g.set_index("momento").reindex(order)
        ax.errorbar([labels[m] for m in order], g["auc_roc_media"], yerr=g["auc_roc_de"],
                    marker="o", capsize=3, label=mod)
    ax.set_ylabel("AUC-ROC (media en validación cruzada)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)
