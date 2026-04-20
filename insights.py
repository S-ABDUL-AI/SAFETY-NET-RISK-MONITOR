"""
Policy intelligence: plain-language reasons from local indicators, rollups, and briefs.

Uses simple thresholds and comparisons to the rest of the dataset so wording stays intuitive.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import data as data_mod


def attach_population_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simulated population count per row when the file does not include one.

    Stable for the same region name and row order (no flicker on rerun).
    Lives here (not only in data.py) so Streamlit Cloud still works if an older data.py was deployed.
    """
    out = df.copy()
    if "population" in out.columns:
        out["population"] = pd.to_numeric(out["population"], errors="coerce")
        if out["population"].notna().all() and (out["population"] > 0).all():
            out["population"] = out["population"].astype(int)
            return out

    pops: list[int] = []
    for i, r in enumerate(out["region"].astype(str)):
        seed = (abs(hash(r)) + i) % (2**32)
        rng = np.random.default_rng(seed)
        pops.append(int(rng.integers(28_000, 520_000)))
    out["population"] = pops
    return out


# Absolute rule bands (easy to explain without statistics jargon)
FOOD_INDEX_HIGH = 118.0
EMPLOYMENT_LOW = 42.0
INCOME_SOFT_LOW_RATIO = 0.88  # below this share of the dataset median = "on the low side"
INFLATION_HIGH = 6.0


def _benchmarks(df: pd.DataFrame) -> dict[str, float]:
    """Typical levels in the current dataset (for 'higher than usual' wording)."""
    return {
        "food_price_median": float(df["food_price_index"].median()),
        "employment_median": float(df["employment_rate"].median()),
        "income_median": float(df["income"].median()),
        "inflation_median": float(df["inflation"].median()),
    }


def rule_based_reasons(row: pd.Series, bm: dict[str, float]) -> list[str]:
    """
    Return short cause lines from indicator rules (non-technical).

    row: one row with food_price_index, employment_rate, income, inflation.
    """
    reasons: list[str] = []
    food = float(row["food_price_index"])
    emp = float(row["employment_rate"])
    inc = float(row["income"])
    infl = float(row["inflation"])

    food_hot = food >= max(FOOD_INDEX_HIGH, bm["food_price_median"] + 4.0)
    food_mild = food >= bm["food_price_median"] + 1.5 and not food_hot

    if food_hot:
        reasons.append(
            "Food costs are clearly elevated, which often hits lower-income households first."
        )
    elif food_mild:
        reasons.append("Food prices are somewhat above the typical level in this dataset.")

    emp_cold = emp <= min(EMPLOYMENT_LOW, bm["employment_median"] - 4.0)
    emp_mild = emp <= bm["employment_median"] - 2.0 and not emp_cold

    if emp_cold:
        reasons.append(
            "Paid work is scarce relative to what we would like to see, pointing to job scarcity in the area."
        )
    elif emp_mild:
        reasons.append("Employment levels are a bit soft compared with other areas in this view.")

    if inc < bm["income_median"] * INCOME_SOFT_LOW_RATIO:
        reasons.append("Typical incomes here are on the low side, leaving less room to absorb shocks.")

    infl_hot = infl >= max(INFLATION_HIGH, bm["inflation_median"] + 1.5)
    if infl_hot:
        reasons.append("Overall inflation is high enough to add day-to-day budget pressure.")

    if not reasons:
        reasons.append(
            "No single indicator is extreme; the risk score reflects several factors lining up together."
        )
    return reasons


def build_policy_insights_table(preds_with_pop: pd.DataFrame) -> pd.DataFrame:
    """
    One row per region: rolled-up indicators, risk band, and a plain 'why' paragraph.

    Uses mean feature levels within each region vs whole-sample benchmarks.
    """
    out_columns = [
        "region",
        "risk_band",
        "people_represented",
        "avg_food_price_index",
        "avg_employment_rate",
        "why_this_outlook",
    ]
    if preds_with_pop is None or preds_with_pop.empty:
        return pd.DataFrame(columns=out_columns)

    df = preds_with_pop.copy()
    bm = _benchmarks(df)

    agg = (
        df.groupby("region", as_index=False)
        .agg(
            highest_risk=("predicted_risk", "max"),
            food_price_index=("food_price_index", "mean"),
            employment_rate=("employment_rate", "mean"),
            income=("income", "mean"),
            inflation=("inflation", "mean"),
            population=("population", "sum"),
        )
    )

    rows_out = []
    for _, r in agg.iterrows():
        code = int(r["highest_risk"])
        band = data_mod.RISK_LABELS[code]
        reasons = rule_based_reasons(r, bm)
        if code == 2:
            lead = "Higher risk here aligns with local conditions: "
        elif code == 1:
            lead = "Medium-level concern is consistent with: "
        else:
            lead = "Lower concern in this snapshot; still note: "

        why = lead + " ".join(reasons)
        rows_out.append(
            {
                "region": r["region"],
                "risk_band": band,
                "people_represented": int(r["population"]),
                "avg_food_price_index": round(float(r["food_price_index"]), 1),
                "avg_employment_rate": round(float(r["employment_rate"]), 1),
                "why_this_outlook": why,
            }
        )
    return pd.DataFrame(rows_out).sort_values("region")


def build_action_impact_table(
    preds_with_pop: pd.DataFrame,
    top_n: int = 6,
) -> pd.DataFrame:
    """
    Build an executive table linking each region's recommended action to
    a simple expected-impact estimate.

    Impact figures are indicative scenario estimates derived from how far each
    region sits from whole-sample medians on food prices, employment, inflation,
    and income.
    """
    out_cols = [
        "region",
        "risk_band",
        "recommended_action",
        "expected_impact",
        "estimated_people_reached",
    ]
    if preds_with_pop is None or preds_with_pop.empty:
        return pd.DataFrame(columns=out_cols)

    df = preds_with_pop.copy()
    bm = _benchmarks(df)
    agg = (
        df.groupby("region", as_index=False)
        .agg(
            highest_risk=("predicted_risk", "max"),
            avg_food=("food_price_index", "mean"),
            avg_employment=("employment_rate", "mean"),
            avg_inflation=("inflation", "mean"),
            avg_income=("income", "mean"),
            population=("population", "sum"),
        )
        .sort_values(["highest_risk", "population"], ascending=[False, False])
        .head(max(1, int(top_n)))
    )

    def _action_for_code(code: int) -> str:
        if code >= 2:
            return "Targeted cash transfers, food subsidies"
        if code == 1:
            return "Job programs, microfinance support"
        return "Monitor and maintain policies"

    rows: list[dict] = []
    for _, r in agg.iterrows():
        code = int(r["highest_risk"])
        band = data_mod.RISK_LABELS.get(code, "Unknown")
        action = _action_for_code(code)

        food_gap = max(0.0, (float(r["avg_food"]) - bm["food_price_median"]) / max(1.0, bm["food_price_median"]) * 100.0)
        emp_gap = max(0.0, (bm["employment_median"] - float(r["avg_employment"])) / max(1.0, bm["employment_median"]) * 100.0)
        infl_gap = max(0.0, (float(r["avg_inflation"]) - bm["inflation_median"]) / max(1.0, bm["inflation_median"]) * 100.0)
        income_gap = max(0.0, (bm["income_median"] - float(r["avg_income"])) / max(1.0, bm["income_median"]) * 100.0)

        driver = max(
            [("food", food_gap), ("employment", emp_gap), ("inflation", infl_gap), ("income", income_gap)],
            key=lambda x: x[1],
        )[0]

        if driver == "food":
            est = round(min(15.0, max(3.0, food_gap * 0.6)), 1)
            impact = f"Could reduce food-price pressure by about {est}% with effective rollout."
        elif driver == "employment":
            est = round(min(12.0, max(2.0, emp_gap * 0.5)), 1)
            impact = f"Could improve employment outcomes by roughly {est}% over the policy cycle."
        elif driver == "inflation":
            est = round(min(10.0, max(2.0, infl_gap * 0.45)), 1)
            impact = f"Could lower inflation pressure by around {est}% in affected households."
        else:
            est = round(min(12.0, max(2.5, income_gap * 0.5)), 1)
            impact = f"Could improve disposable income resilience by about {est}%."

        rows.append(
            {
                "region": str(r["region"]),
                "risk_band": band,
                "recommended_action": action,
                "expected_impact": impact,
                "estimated_people_reached": int(r["population"]),
            }
        )

    return pd.DataFrame(rows)


def summary_dashboard_stats(preds_with_pop: pd.DataFrame) -> dict:
    """Counts for the headline dashboard and brief."""
    df = preds_with_pop.copy()
    if df.empty:
        return {
            "high_risk_regions": 0,
            "medium_only_regions": 0,
            "medium_or_high_regions": 0,
            "population_high_risk_rows": 0,
            "population_elevated_or_higher": 0,
            "total_regions": 0,
        }
    high_regions = df.groupby("region")["predicted_risk"].max()
    n_high_regions = int((high_regions >= 2).sum())
    n_medium_only_regions = int((high_regions == 1).sum())

    at_risk = df[df["predicted_risk"] >= 2]
    pop_high = int(at_risk["population"].sum()) if len(at_risk) else 0

    elevated = df[df["predicted_risk"] >= 1]
    pop_elevated = int(elevated["population"].sum()) if len(elevated) else 0

    return {
        "high_risk_regions": n_high_regions,
        "medium_only_regions": n_medium_only_regions,
        "medium_or_high_regions": int((high_regions >= 1).sum()),
        "population_high_risk_rows": pop_high,
        "population_elevated_or_higher": pop_elevated,
        "total_regions": int(df["region"].nunique()),
    }


def _theme_counts_for_brief(insights_df: pd.DataFrame) -> list[str]:
    """Collect most common plain themes from high-risk regions for the brief."""
    sub = insights_df[insights_df["risk_band"] == "High"]
    if sub.empty:
        return []
    themes: Counter[str] = Counter()
    for text in sub["why_this_outlook"]:
        t = str(text).lower()
        if "food" in t:
            themes["food costs"] += 1
        if "paid work" in t or "employment" in t or "job" in t:
            themes["jobs and earnings"] += 1
        if "incomes here" in t or "typical incomes" in t:
            themes["income pressure"] += 1
        if "inflation" in t:
            themes["overall inflation"] += 1
    return [k for k, _ in themes.most_common(3)]


def generate_policy_brief(
    stats: dict,
    insights_df: pd.DataFrame,
    model_accuracy: float,
) -> str:
    """One short paragraph decision-makers can lift into notes or meetings."""
    tr = stats["total_regions"]
    if tr == 0:
        return (
            "No rows are in the current view, so there is nothing to summarise. "
            "Widen the region filter in the sidebar or upload a larger dataset."
        )
    hr = stats["high_risk_regions"]
    pop = stats["population_high_risk_rows"]

    themes = _theme_counts_for_brief(insights_df)
    theme_phrase = (
        "Common pressure points include " + ", ".join(themes) + "."
        if themes
        else "Drivers vary by place; use the regional table for specifics."
    )

    acc = int(round(model_accuracy * 100))
    if hr == 0:
        p1 = f"This screen covers {tr} regions. None sit in the highest stress band in the slice you are viewing. "
    elif hr == 1:
        p1 = f"This screen covers {tr} regions. One region sits in the highest stress band in the data shown. "
    else:
        p1 = f"This screen covers {tr} regions. {hr} regions sit in the highest stress band in the data shown. "
    p2 = (
        f"We estimate on the order of {pop:,} people live in local areas placed in that highest band in this run "
        f"(population figures are illustrative where real counts are not uploaded). "
    )
    p3 = (
        f"{theme_phrase} On a random slice of the same data, the score matched the recorded risk level "
        f"about {acc}% of the time. Use that as a sanity check, not a guarantee. "
    )
    p4 = "Suggested next step: confirm food and labour programmes in the high-stress regions, and keep lighter monitoring where scores stay low."

    return p1 + p2 + p3 + p4
