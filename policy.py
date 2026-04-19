"""
Policy recommendations in plain language (one line per risk tier).

These strings are shown directly in the app — avoid technical jargon.
"""

from __future__ import annotations

import pandas as pd

# Numeric codes: 0 low, 1 medium, 2 high (matches data.RISK_LABELS)
RECOMMENDATIONS = {
    2: "Targeted cash transfers, food subsidies",
    1: "Job programs, microfinance support",
    0: "Monitor and maintain policies",
}


def recommendation_for_tier(tier_code: int) -> str:
    """Return the policy line for a predicted risk tier (0/1/2)."""
    code = int(tier_code)
    return RECOMMENDATIONS.get(code, RECOMMENDATIONS[0])


def attach_recommendations(df: pd.DataFrame, tier_column: str = "predicted_risk") -> pd.DataFrame:
    """
    Add a 'recommended_action' column from predicted tier codes.

    Expects tier_column values in {0, 1, 2}.
    """
    out = df.copy()
    out["recommended_action"] = out[tier_column].astype(int).map(RECOMMENDATIONS)
    return out
