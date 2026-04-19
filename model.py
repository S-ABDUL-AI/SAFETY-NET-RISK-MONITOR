"""
Train a Random Forest classifier and produce risk scores for each row.

Supports:
- Three-class labels (0 low, 1 medium, 2 high)
- Binary labels (0/1): probabilities are split into low / medium / high bands
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import REQUIRED_COLUMNS, risk_display_name, validate_for_training

FEATURE_NUMERIC = [
    "income",
    "employment_rate",
    "food_price_index",
    "inflation",
]
FEATURE_CATEGORICAL = ["region"]
TARGET = "poverty_label"

# When the dataset is binary (0/1), map P(class 1) to three display tiers
BINARY_LOW_MAX = 0.34
BINARY_MEDIUM_MAX = 0.66


@dataclass
class TrainResult:
    """Outcome of training: fitted pipeline, holdout accuracy, and feature names."""

    pipeline: Pipeline
    accuracy: float
    feature_names: list[str]
    is_binary: bool


def _build_pipeline() -> Pipeline:
    """Preprocessor + RandomForest with sensible defaults for tabular data."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FEATURE_CATEGORICAL,
            ),
        ]
    )
    # class_weight helps when tiers are imbalanced
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("model", clf)])


def train_model(df: pd.DataFrame, random_state: int = 42) -> TrainResult:
    """
    Fit the model on df (must include REQUIRED_COLUMNS).

    Returns accuracy on a held-out test split and the fitted pipeline.
    """
    data = df[REQUIRED_COLUMNS].copy()
    validate_for_training(data)

    y = data[TARGET].astype(int).values
    X = data[FEATURE_NUMERIC + FEATURE_CATEGORICAL]

    classes = np.unique(y)
    is_binary = len(classes) == 2

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )
    except ValueError:
        # Too few rows per class for stratified split; fall back to a random split.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=None
        )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    prep: ColumnTransformer = pipeline.named_steps["prep"]
    feature_names = list(prep.get_feature_names_out())

    return TrainResult(
        pipeline=pipeline,
        accuracy=acc,
        feature_names=feature_names,
        is_binary=is_binary,
    )


def _proba_matrix(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return predict_proba array (n_samples, n_classes)."""
    return pipeline.predict_proba(X)


def _tier_from_binary_proba(proba_positive: np.ndarray) -> np.ndarray:
    """Map P(high risk class) to 0=low, 1=medium, 2=high tiers."""
    p = np.asarray(proba_positive, dtype=float)
    tier = np.zeros_like(p, dtype=int)
    tier[(p > BINARY_LOW_MAX) & (p <= BINARY_MEDIUM_MAX)] = 1
    tier[p > BINARY_MEDIUM_MAX] = 2
    return tier


def predict_risk_table(
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_names: list[str],
    is_binary: bool,
) -> pd.DataFrame:
    """
    Build a table with predicted tier, plain-language tier name, class probabilities,
    and policy recommendation text (added by the app from policy.py).
    """
    X = df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    proba = _proba_matrix(pipeline, X)

    if is_binary:
        # sklearn orders classes ascending; use P(label==1) to place each area in a tier
        classes_order = np.asarray(pipeline.classes_)
        if 1 not in classes_order:
            idx_pos = int(classes_order.argmax())
        else:
            idx_pos = int(np.where(classes_order == 1)[0][0])
        p_pos = proba[:, idx_pos]
        predicted_code = _tier_from_binary_proba(p_pos)
        idx_neg = 1 - idx_pos if proba.shape[1] == 2 else 0
        out = pd.DataFrame(
            {
                "region": df["region"].values,
                "predicted_risk": predicted_code,
                "risk_level": [risk_display_name(c) for c in predicted_code],
                "likelihood_lower_concern": np.round(proba[:, idx_neg], 3),
                "likelihood_elevated_concern": np.round(p_pos, 3),
            }
        )
    else:
        pred_class = np.argmax(proba, axis=1)
        # Map argmax index back to original class labels if not 0..K-1
        class_list = list(pipeline.classes_)
        predicted_code = np.array([class_list[i] for i in pred_class], dtype=int)
        out = pd.DataFrame(
            {
                "region": df["region"].values,
                "predicted_risk": predicted_code,
                "risk_level": [risk_display_name(c) for c in predicted_code],
            }
        )
        for j, c in enumerate(class_list):
            col = f"chance_{risk_display_name(int(c)).lower()}"
            out[col] = np.round(proba[:, j], 3)

    # Helpful local context for reviewers (same order as model input rows)
    out["income"] = df["income"].values
    out["employment_rate"] = df["employment_rate"].values
    out["food_price_index"] = df["food_price_index"].values
    out["inflation"] = df["inflation"].values

    # Feature importances (global model view)
    rf: RandomForestClassifier = pipeline.named_steps["model"]
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:12]
    top_features = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    # Attach same global explanation to each row for display in sidebar / expander
    out.attrs["importance_names"] = top_features
    out.attrs["importance_values"] = top_values.tolist()
    return out


def format_top_driver_sentence(
    feature_names: list[str], importances: list[float], max_bullets: int = 4
) -> str:
    """
    One short paragraph naming the strongest predictors in everyday words.
    """
    if not feature_names:
        return "Build scores first to see which signals matter most in this dataset."

    def _plain(name: str) -> str:
        n = name.replace("num__", "").replace("cat__", "")
        if "region_" in n:
            return "which region it is"
        mapping = {
            "income": "typical income in the area",
            "employment_rate": "share of people in work",
            "food_price_index": "food prices",
            "inflation": "inflation",
        }
        for key, label in mapping.items():
            if key in n:
                return label
        return n.replace("_", " ")

    parts = []
    for name, imp in zip(feature_names[:max_bullets], importances[:max_bullets]):
        parts.append(f"{_plain(name)} (about {imp * 100:.0f}% of the weight in the score)")
    return "The score leans most on: " + "; ".join(parts) + "."
