"""
Poverty Early Warning System (PEWS) — Streamlit entry point.

Plain-language interface for exploring regional risk, model results, and policy options.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root (matches GitHub layout: modules live next to app.py, not under a package folder).
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

import data as data_mod
import insights as insights_mod
import model as model_mod
import policy as policy_mod
import viz as viz_mod


def _init_session() -> None:
    """Load default sample data once per session."""
    if "source_df" not in st.session_state:
        st.session_state.source_df = insights_mod.attach_population_if_missing(
            data_mod.generate_synthetic_dataset()
        )
    if "data_note" not in st.session_state:
        st.session_state.data_note = (
            "Sample numbers are loaded for a quick demo. They are not official statistics. "
            "Upload your own CSV for real analysis."
        )
    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = None
    if "_should_auto_train" not in st.session_state:
        st.session_state._should_auto_train = True


def _sidebar_upload() -> None:
    """CSV upload replaces the working dataset when a valid file is provided."""
    st.sidebar.subheader("Data")
    file = st.sidebar.file_uploader(
        "Upload a CSV file (optional)",
        type=["csv"],
        help="Required: region, income, employment_rate, food_price_index, inflation, poverty_label. Optional: population (per row).",
    )
    if file is not None:
        try:
            df = data_mod.load_csv_from_upload(file)
            st.session_state.source_df = insights_mod.attach_population_if_missing(df)
            st.session_state.data_note = (
                "Using your uploaded file. Risk bands and population totals follow your data."
            )
            st.session_state.model_bundle = None
            st.session_state._should_auto_train = True
            st.sidebar.success("File loaded.")
        except Exception as exc:  # noqa: BLE001 — show friendly message to user
            st.sidebar.error(f"Could not read that file: {exc}")


def _sidebar_regions(df) -> list:
    """Return list of regions to include in tables and charts (empty = all)."""
    all_regions = sorted(df["region"].astype(str).unique().tolist())
    picked = st.sidebar.multiselect(
        "Regions to show",
        options=all_regions,
        default=all_regions,
        help="Narrow the tables and charts without changing the uploaded file.",
    )
    return picked if picked else all_regions


def _filter_by_regions(df, regions: list):
    return df[df["region"].isin(regions)].copy()


def _run_training(df_full) -> bool:
    """Fit model, attach recommendations, store bundle. Returns False on error."""
    try:
        result = model_mod.train_model(df_full)
        preds = model_mod.predict_risk_table(
            df_full,
            result.pipeline,
            result.feature_names,
            result.is_binary,
        )
        preds = policy_mod.attach_recommendations(preds)
        st.session_state.model_bundle = {
            "result": result,
            "predictions": preds,
            "importance_names": list(preds.attrs.get("importance_names", [])),
            "importance_values": list(preds.attrs.get("importance_values", [])),
        }
        return True
    except Exception as exc:  # noqa: BLE001
        st.session_state.model_bundle = None
        st.error(f"Scores could not be built: {exc}")
        return False


def main() -> None:
    st.set_page_config(
        page_title="PEWS — Poverty Early Warning",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session()
    df_full = insights_mod.attach_population_if_missing(st.session_state.source_df)
    st.session_state.source_df = df_full

    st.sidebar.title("Settings")
    _sidebar_upload()
    region_pick = _sidebar_regions(df_full)
    df_view = _filter_by_regions(df_full, region_pick)

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    train_clicked = st.sidebar.button(
        "Refresh risk scores",
        type="primary",
        help="Rebuild scores from the full dataset (for example after you upload a new file).",
    )

    should_train = train_clicked or (
        st.session_state.model_bundle is None and st.session_state.get("_should_auto_train", True)
    )
    if should_train:
        with st.spinner("Building scores from your data…"):
            ok = _run_training(df_full)
        if ok:
            st.session_state._should_auto_train = False
            if train_clicked:
                st.sidebar.success("Scores updated.")

    st.sidebar.divider()
    st.sidebar.caption("Designed by Sherriff Abdul-Hamid")

    st.title("Poverty Early Warning System")
    st.caption(
        "A simple decision-support screen: compare places, see where stress looks higher in **your** data, "
        "and read plain-language ideas for next steps. This is not an official poverty count."
    )

    with st.expander("How to use this page", expanded=False):
        st.markdown(
            """
1. **Numbers on screen** — By default you see sample data so the tool works immediately. Upload a CSV if you have real figures.  
2. **Regions** — Use the sidebar list to focus on certain places; leave all selected for the full picture.  
3. **Scores** — The app builds a risk band (low / medium / high) from your columns. It runs automatically the first time; use **Refresh risk scores** after you change the file.  
4. **Read the story** — Scroll to the summary, policy brief, and policy insights for counts, a short paragraph you can reuse in notes, and reasons tied to food prices, jobs, and incomes.  
5. **Population** — Unless you upload a `population` column, people counts are **illustrative** for dashboards only.
            """
        )

    st.info(st.session_state.data_note)

    st.subheader("Dataset preview")
    st.dataframe(df_view.head(50), use_container_width=True, hide_index=True)

    bundle = st.session_state.model_bundle
    if bundle is None:
        st.warning(
            "Scores are not ready yet. Use **Refresh risk scores** in the sidebar, or try the button below."
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Build scores now", type="primary", use_container_width=True):
                st.session_state._should_auto_train = True
                st.rerun()
        with c2:
            st.caption("If this keeps failing, check that your CSV has the required columns listed in the sidebar.")
        return

    result = bundle["result"]
    preds_full = bundle["predictions"]
    preds_view = _filter_by_regions(preds_full, region_pick)

    st.subheader("How well the score matches the data")
    st.metric(
        label="Share of rows the score got right when tested on a random slice of the same data",
        value=f"{result.accuracy * 100:.1f}%",
    )
    st.caption("Higher is better, but real decisions should still use local knowledge—not this number alone.")

    st.subheader("Scores and suggested actions (row level)")
    display_order = [
        "region",
        "income",
        "employment_rate",
        "food_price_index",
        "inflation",
        "risk_level",
        "recommended_action",
    ]
    prob_pref = [
        "chance_low",
        "chance_medium",
        "chance_high",
        "likelihood_lower_concern",
        "likelihood_elevated_concern",
    ]
    extra_prob_cols = [c for c in prob_pref if c in preds_view.columns]
    ordered = [c for c in display_order if c in preds_view.columns] + extra_prob_cols
    st.dataframe(preds_view[ordered], use_container_width=True, hide_index=True)

    st.subheader("Regional summary (uses the highest risk band seen in each region)")
    summary = (
        preds_view.groupby("region", as_index=False)
        .agg({"predicted_risk": "max"})
        .rename(columns={"predicted_risk": "highest_risk_code"})
    )
    summary["risk_band"] = summary["highest_risk_code"].map(data_mod.RISK_LABELS)
    summary["recommended_action"] = summary["highest_risk_code"].map(
        policy_mod.RECOMMENDATIONS
    )
    st.dataframe(
        summary[["region", "risk_band", "recommended_action"]],
        use_container_width=True,
        hide_index=True,
    )

    # --- Policy intelligence (uses same row order as the model output)
    preds_intel = preds_full.copy()
    preds_intel["population"] = df_full["population"].astype(int).values
    all_regions_set = set(sorted(df_full["region"].astype(str).unique().tolist()))
    region_filter_set = set(region_pick)
    intel_scope = preds_intel if region_filter_set == all_regions_set else preds_intel[preds_intel["region"].isin(region_pick)].copy()
    if region_filter_set != all_regions_set:
        st.caption(
            "Summary and policy text below follow your region filter. Switch back to all regions for a whole-country-style view."
        )

    st.subheader("Summary dashboard")
    dash = insights_mod.summary_dashboard_stats(intel_scope)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric(
            "Regions at highest stress band",
            dash["high_risk_regions"],
            help="Count of regions whose worst local score in this view is High.",
        )
    with d2:
        st.metric(
            "Regions with at least medium stress",
            dash["medium_or_high_regions"],
            help="Regions where the worst local score is Medium or High.",
        )
    with d3:
        st.metric(
            "People in highest-stress local areas (illustrative)",
            f"{dash['population_high_risk_rows']:,}",
            help="Sum of illustrative population counts for rows scored in the High band. Upload a population column for real totals.",
        )

    st.subheader("Policy brief")
    insights_table = insights_mod.build_policy_insights_table(intel_scope)
    brief = insights_mod.generate_policy_brief(dash, insights_table, result.accuracy)
    st.info(brief)

    st.subheader("Policy insights")
    st.write(
        "Short, plain-language read on **why** each region looks the way it does, from food costs, jobs, incomes, and inflation in your data."
    )
    st.dataframe(
        insights_table[
            [
                "region",
                "risk_band",
                "people_represented",
                "avg_food_price_index",
                "avg_employment_rate",
                "why_this_outlook",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "“People represented” adds illustrative population counts across rows in each region. "
        "Upload a `population` column on your CSV for real totals."
    )

    st.subheader("Charts")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            viz_mod.chart_risk_by_region(preds_view),
            use_container_width=True,
        )
    names = bundle.get("importance_names", [])
    values = bundle.get("importance_values", [])
    with c2:
        if names and values:
            st.plotly_chart(
                viz_mod.chart_feature_importance(names, values),
                use_container_width=True,
            )
        else:
            st.empty()

    st.subheader("What is driving these scores?")
    sentence = model_mod.format_top_driver_sentence(names, values)
    st.write(sentence)
    with st.expander("How to read this"):
        st.write(
            "The chart shows which inputs the scoring step leans on most. "
            "It is a useful guide for discussion, not proof of what causes poverty. "
            "Always combine these results with local knowledge and other evidence."
        )

    st.divider()
    st.caption("Designed by Sherriff Abdul-Hamid")


if __name__ == "__main__":
    main()
