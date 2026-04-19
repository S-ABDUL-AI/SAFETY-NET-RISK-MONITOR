"""Plotly charts for risk counts by region and model feature focus (importance)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def chart_risk_by_region(pred_df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart: how many areas fall in each risk band, grouped by region.

    pred_df must include columns 'region' and 'risk_level'.
    """
    if pred_df is None or pred_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Risk outlook by region",
            annotations=[
                dict(
                    text="No rows to chart for this filter.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        return fig

    counts = (
        pred_df.groupby(["region", "risk_level"]).size().reset_index(name="count")
    )
    order = ["Low", "Medium", "High"]
    counts["risk_level"] = pd.Categorical(
        counts["risk_level"], categories=order, ordered=True
    )
    counts = counts.sort_values(["region", "risk_level"])
    fig = px.bar(
        counts,
        x="region",
        y="count",
        color="risk_level",
        category_orders={"risk_level": order},
        color_discrete_map={
            "Low": "#2ecc71",
            "Medium": "#f39c12",
            "High": "#e74c3c",
        },
        labels={
            "region": "Region",
            "count": "Number of areas",
            "risk_level": "Risk band",
        },
    )
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=120),
        legend_title_text="",
        title="Risk outlook by region",
        font=dict(size=13),
        xaxis_tickangle=-35,
    )
    return fig


def chart_feature_importance(names: list[str], values: list[float]) -> go.Figure:
    """Horizontal bar chart of the strongest model signals (importance scores)."""
    if not names or not values:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Which signals the model relies on most",
            annotations=[
                dict(
                    text="No importance data yet.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        return fig

    df = pd.DataFrame({"signal": names, "importance": values}).sort_values(
        "importance", ascending=True
    )

    def _short_label(raw: str) -> str:
        s = raw.replace("num__", "").replace("cat__", "")
        if "region_" in s:
            return s.replace("region_", "Region: ")
        return s.replace("_", " ")

    df["label"] = df["signal"].map(_short_label)
    fig = px.bar(
        df,
        x="importance",
        y="label",
        orientation="h",
        labels={"importance": "Strength in model", "label": "Signal"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=160, r=20, t=50, b=40),
        title="Which signals the model relies on most",
        showlegend=False,
        font=dict(size=13),
    )
    return fig
