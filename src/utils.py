"""Utilidades de reproducibilidad y registro del entorno."""
import importlib
import platform
import random

import numpy as np

from . import config

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost", "shap",
            "matplotlib", "seaborn", "joblib"]


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def environment_report() -> dict:
    """Versiones de Python y librerías, para anexarlas al documento."""
    report = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            report[name] = importlib.import_module(name).__version__
        except ImportError:
            report[name] = "no instalado"
    return report
