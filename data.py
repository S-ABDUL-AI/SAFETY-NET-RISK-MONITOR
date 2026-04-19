"""
Data loading and synthetic dataset generation for PEWS.

The app expects CSV columns:
region, income, employment_rate, food_price_index, inflation, poverty_label
"""

from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np
import pandas as pd

# Column names the rest of the app relies on
REQUIRED_COLUMNS = [
    "region",
    "income",
    "employment_rate",
    "food_price_index",
    "inflation",
    "poverty_label",
]

# Friendly names for display (policymaker-facing)
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# Minimum rows so train or test split is meaningful (matches app messaging).
MIN_ROWS_FOR_TRAINING = 20

NUMERIC_FEATURE_COLUMNS = [
    "income",
    "employment_rate",
    "food_price_index",
    "inflation",
]


def validate_dataframe(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "The file is missing required columns: "
            + ", ".join(missing)
            + ". Expected: "
            + ", ".join(REQUIRED_COLUMNS)
        )


def load_csv_from_upload(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Read an uploaded CSV into a DataFrame and validate columns."""
    raw = uploaded_file.read()
    if raw is None or len(raw) == 0:
        raise ValueError("The uploaded file is empty.")
    if not raw.strip():
        raise ValueError("The uploaded file is empty or contains only blank lines.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV has no readable rows.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "Could not parse this file as CSV. Check commas, quotes, and column headers."
        ) from exc
    validate_dataframe(df)
    extra_pop = "population" in df.columns
    cols = list(REQUIRED_COLUMNS)
    if extra_pop:
        cols.append("population")
    df = df[cols].copy()
    df = _normalize_poverty_label(df)
    df = coerce_numeric_features(df)
    validate_for_training(df)
    # Late import avoids circular import (insights imports this module).
    from insights import attach_population_if_missing

    return attach_population_if_missing(df)


def _normalize_poverty_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure poverty_label is integer 0/1/2.

    Accepts:
    - 0,1,2 (low / medium / high)
    - 0,1 only (binary): kept as 0/1; the model layer will use binary mode
    """
    df = df.copy()
    df["poverty_label"] = pd.to_numeric(df["poverty_label"], errors="coerce")
    if df["poverty_label"].isna().any():
        raise ValueError("poverty_label must be numeric (0, 1, or 0/1/2).")
    uniq = sorted(df["poverty_label"].dropna().unique().tolist())
    if not set(uniq).issubset({0, 1, 2}):
        raise ValueError("poverty_label must use values 0, 1, and optionally 2 only.")
    df["poverty_label"] = df["poverty_label"].astype(int)
    return df


def coerce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cast indicator columns to float; invalid cells become NaN and are caught by validate_for_training."""
    out = df.copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_for_training(df: pd.DataFrame) -> None:
    """
    Raise ValueError if the table cannot support training or honest scoring.

    Expects REQUIRED_COLUMNS present and poverty_label already normalized.
    """
    if df is None or len(df) == 0:
        raise ValueError("The dataset is empty.")
    if len(df) < MIN_ROWS_FOR_TRAINING:
        raise ValueError(
            f"Need at least {MIN_ROWS_FOR_TRAINING} rows to build a train and test split. "
            f"This dataset has {len(df)}."
        )
    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        if series.isna().all():
            raise ValueError(f"Column {col!r} has no valid numeric values.")
        if series.isna().any():
            raise ValueError(
                f"Column {col!r} has missing or non-numeric cells. "
                "Fill or remove those rows, then upload again."
            )
    if (df["income"] < 0).any():
        raise ValueError("Column income contains negative values.")
    emp = df["employment_rate"]
    if (emp < 0).any() or (emp > 100).any():
        raise ValueError(
            "Column employment_rate should be between 0 and 100 (percent of people in work)."
        )
    if df["region"].isna().any():
        raise ValueError("Column region has missing values.")
    if (df["region"].astype(str).str.strip() == "").any():
        raise ValueError("Column region has blank text. Use a real place name in every row.")
    labels = df["poverty_label"].dropna().unique()
    if len(labels) < 2:
        raise ValueError(
            "Column poverty_label uses only one category. "
            "The model needs at least two levels (for example 0 and 1, or 0, 1, and 2)."
        )


def generate_synthetic_dataset(
    n_rows: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a realistic synthetic panel when no CSV is uploaded.

    poverty_label: 0 = low risk, 1 = medium, 2 = high (derived from latent risk).
    """
    rng = np.random.default_rng(random_state)
    regions = [
        "North Valley",
        "Coastal Plain",
        "Central Highlands",
        "Eastern Delta",
        "Western Plateau",
        "Southern Corridor",
        "Lake District",
        "Border Counties",
    ]
    region = rng.choice(regions, size=n_rows)

    # Latent economic stress (not shown to the user) drives labels + observed fields
    latent = rng.normal(0, 1, size=n_rows)
    region_effect = np.array([hash(r) % 5 / 10 - 0.2 for r in region])
    stress = latent + region_effect

    income = np.clip(800 + stress * -180 + rng.normal(0, 120, n_rows), 200, 2500)
    employment_rate = np.clip(55 - stress * 12 + rng.normal(0, 6, n_rows), 25, 92)
    food_price_index = np.clip(100 + stress * 8 + rng.normal(0, 4, n_rows), 85, 145)
    inflation = np.clip(2 + stress * 1.2 + rng.normal(0, 0.8, n_rows), 0.5, 18)

    # Tertiles of stress → three risk classes for a clear learning task
    t1, t2 = np.quantile(stress, [0.33, 0.66])
    poverty_label = np.where(stress <= t1, 0, np.where(stress <= t2, 1, 2)).astype(int)

    # Illustrative population counts for dashboard totals (not used by the risk model).
    population = rng.integers(35_000, 480_000, size=n_rows)

    df = pd.DataFrame(
        {
            "region": region,
            "income": np.round(income, 0),
            "employment_rate": np.round(employment_rate, 1),
            "food_price_index": np.round(food_price_index, 1),
            "inflation": np.round(inflation, 2),
            "poverty_label": poverty_label,
            "population": population.astype(int),
        }
    )
    return df


def risk_display_name(code: int) -> str:
    """Map numeric label to short display tier."""
    return RISK_LABELS.get(int(code), str(code))
