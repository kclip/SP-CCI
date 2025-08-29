import numpy as np
import pandas as pd
from typing import Tuple

def load_ihdp_csv(path: str):
    """Load IHDP CSV placed at data/ihdp/ihdp.csv.
    Expected columns: 25 features X_*, treatment T, and potential outcomes Y0,Y1 or factual/cf construction.
    You can edit this loader to match your local file format.
    """
    df = pd.read_csv(path)
    # Try a reasonable guess; user may edit
    X_cols = [c for c in df.columns if c.startswith("x_") or c.startswith("X_")]
    if not X_cols:
        # Fallback: take first 25 numeric columns as X
        X_cols = df.select_dtypes(include=[np.number]).columns[:25].tolist()
    assert len(X_cols) >= 2, "Could not infer feature columns; please edit data_ihdp.py"
    X = df[X_cols].to_numpy()
    if "T" in df.columns:
        T = df["T"].to_numpy().astype(int)
    else:
        # If missing, create a placeholder
        T = np.zeros(len(df), dtype=int)
    # Prefer Y0/Y1 if present; else use observed Y and a placeholder CF (for benchmarking only)
    if {"Y0","Y1"} <= set(df.columns):
        Y0 = df["Y0"].to_numpy()
        Y1 = df["Y1"].to_numpy()
    else:
        Y = df[[c for c in df.columns if c.lower() in ["y","y_obs","yobs"]][0]].to_numpy()
        # Without true counterfactuals, we can only run conformal on observed arms
        # Provide placeholders; user should adapt
        Y0 = np.where(T == 0, Y, np.nan)
        Y1 = np.where(T == 1, Y, np.nan)
    return X, T, Y0, Y1
