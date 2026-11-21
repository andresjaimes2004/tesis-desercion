"""Carga del dataset, binarización del objetivo y selección de variables por momento."""
import pandas as pd

from . import config


def load_raw(path=config.RAW_FILE) -> pd.DataFrame:
    """Lee el CSV crudo. Si no existe, lo descarga de UCI y lo guarda en data/raw."""
    if path.exists():
        # El archivo original de UCI usa ';'; el guardado por ucimlrepo usa ','
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
    else:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=config.UCI_ID)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df.columns = df.columns.str.strip()  # algunos nombres traen espacios o tabulaciones
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna binaria: 1 = Dropout, 0 = Enrolled o Graduate."""
    out = df.copy()
    out[config.BINARY_TARGET] = (out[config.TARGET_COL] == config.POSITIVE_CLASS).astype(int)
    return out


def features_for_moment(df: pd.DataFrame, moment: str) -> list[str]:
    """Columnas predictoras disponibles en cada momento de predicción."""
    if moment not in config.MOMENTS:
        raise ValueError(f"Momento no válido: {moment}. Opciones: {config.MOMENTS}")
    excluded = {config.TARGET_COL, config.BINARY_TARGET}
    cols = [c for c in df.columns if c not in excluded]
    if moment == "matricula":
        return [c for c in cols if not c.startswith((config.SEM1_PREFIX, config.SEM2_PREFIX))]
    if moment == "semestre_1":
        return [c for c in cols if not c.startswith(config.SEM2_PREFIX)]
    return cols
