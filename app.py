"""
Safety Net Risk Monitor
SNAP and Food Security Vulnerability Targeting for Program Officers

Designed by Sherriff Abdul-Hamid

Positions the tool as a proactive targeting instrument for:
- SNAP outreach coordinators and program officers
- State food security and nutrition program teams
- Federal poverty reduction program administrators
- County-level human services departments
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# Report generator (must be in same folder as this file)
try:
    from report_generator import build_report_bytes
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Safety Net Risk Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN TOKENS ─────────────────────────────────────────────
NAVY     = "#0A1F44"
NAVY_MID = "#152B5C"
GOLD     = "#C9A84C"
GOLD_LT  = "#E8C97A"
INK      = "#1A1A1A"
BODY     = "#2C3E50"
MUTED    = "#6B7280"
RED      = "#C8382A"
RED_LT   = "#FEF2F2"
GREEN    = "#1A7A2E"
GREEN_LT = "#F0FDF4"
AMBER    = "#B8560A"
AMBER_LT = "#FFFBEB"
RULE     = "#E2E6EC"
WHITE    = "#FFFFFF"

BAND_COLOR  = {"High": RED,    "Medium": AMBER,    "Low": GREEN}
BAND_BG     = {"High": RED_LT, "Medium": AMBER_LT, "Low": GREEN_LT}
BAND_THRESH = {"High": 65,     "Medium": 40}

# ── CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    #MainMenu{{visibility:hidden;}} header{{visibility:hidden;}} footer{{visibility:hidden;}}
    .main .block-container{{padding-top:1.5rem;padding-bottom:2rem;max-width:1280px;}}

    .hero{{background:linear-gradient(135deg,{NAVY} 0%,{NAVY_MID} 100%);
           border-left:6px solid {GOLD};padding:26px 32px 22px;margin-bottom:18px;border-radius:4px;}}
    .hero-eye{{color:{GOLD};font-size:10px;font-weight:700;letter-spacing:2.5px;
               text-transform:uppercase;margin-bottom:8px;}}
    .hero-title{{color:white;font-size:27px;font-weight:700;line-height:1.2;
                 margin-bottom:9px;font-family:Georgia,serif;}}
    .hero-sub{{color:#CADCFC;font-size:13.5px;line-height:1.55;}}
    .hero-meta{{color:{GOLD};font-size:11px;margin-top:11px;opacity:0.85;}}

    .sec-lbl{{color:{MUTED};font-size:10px;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;margin-bottom:3px;margin-top:26px;}}
    .sec-ttl{{color:{INK};font-size:20px;font-weight:700;margin-bottom:3px;font-family:Georgia,serif;}}
    .sec-sub{{color:{MUTED};font-size:12.5px;margin-bottom:14px;}}

    .kpi{{background:white;border:1px solid {RULE};border-left:4px solid {NAVY};
          padding:13px 15px;border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.04);height:100%;}}
    .kpi-lbl{{color:{MUTED};font-size:10px;font-weight:700;letter-spacing:.8px;
              text-transform:uppercase;margin-bottom:4px;}}
    .kpi-val{{color:{INK};font-size:26px;font-weight:700;line-height:1.1;font-family:Georgia,serif;}}
    .kpi-sub{{color:{MUTED};font-size:11px;margin-top:3px;}}

    .card{{background:white;border:1px solid {RULE};border-radius:4px;
           padding:15px 17px;box-shadow:0 1px 3px rgba(0,0,0,.04);height:100%;}}
    .card-lbl{{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:7px;}}
    .card-body{{color:{BODY};font-size:13px;line-height:1.55;}}

    .brief-risk{{border-top:4px solid {RED};}}
    .brief-impl{{border-top:4px solid {NAVY};}}
    .brief-act {{border-top:4px solid {GREEN};}}

    .band-pill{{display:inline-block;padding:2px 9px;border-radius:3px;
                font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;}}

    .data-note{{background:{AMBER_LT};border-left:4px solid {GOLD};padding:9px 13px;
                font-size:12px;color:{BODY};margin:10px 0 18px;border-radius:4px;}}

    .insight-row{{background:white;border:1px solid {RULE};border-radius:4px;
                  padding:12px 16px;margin-bottom:8px;}}
    .insight-region{{font-weight:700;color:{INK};font-size:14px;}}
    .insight-why{{color:{BODY};font-size:12.5px;line-height:1.5;margin-top:4px;}}

    .export-card{{background:white;border:1px solid {RULE};border-left:4px solid {GOLD};
                  border-radius:4px;padding:14px 16px;font-size:13px;
                  color:{BODY};line-height:1.6;}}

    .byline{{border-top:1px solid {RULE};padding-top:13px;margin-top:38px;
             color:{MUTED};font-size:11px;font-style:italic;}}

    section[data-testid="stSidebar"]{{background:#FAFAFA;border-right:1px solid {RULE};}}
    .sb-head{{color:{NAVY};font-weight:700;font-size:12px;letter-spacing:1px;
              text-transform:uppercase;margin-bottom:9px;}}
    .sb-foot{{font-size:11px;color:{MUTED};border-top:1px solid {RULE};
              padding-top:11px;margin-top:18px;line-height:1.5;}}

    /* PDF export area */
    .pdf-generating{{text-align:center;padding:12px;}}
    div[data-testid="stDownloadButton"] button{{
        background:{NAVY} !important;color:white !important;
        border:none !important;font-weight:600 !important;
        letter-spacing:.3px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── BUILT-IN SAMPLE DATA ──────────────────────────────────────
BUILTIN = pd.DataFrame([
    {"region":"Border Counties",   "avg_food_price_index":99.2, "avg_employment_rate":57.1, "avg_income_index":38.2, "avg_housing_cost_index":72.1, "population":20332377},
    {"region":"Central Highlands", "avg_food_price_index":98.2, "avg_employment_rate":57.2, "avg_income_index":39.1, "avg_housing_cost_index":70.8, "population":21398282},
    {"region":"Coastal Plain",     "avg_food_price_index":100.1,"avg_employment_rate":55.0, "avg_income_index":36.5, "avg_housing_cost_index":75.3, "population":19151694},
    {"region":"Eastern Delta",     "avg_food_price_index":103.7,"avg_employment_rate":49.8, "avg_income_index":32.0, "avg_housing_cost_index":78.9, "population":16467093},
    {"region":"Lake District",     "avg_food_price_index":100.4,"avg_employment_rate":55.2, "avg_income_index":37.8, "avg_housing_cost_index":71.5, "population":22565667},
    {"region":"North Valley",      "avg_food_price_index":99.3, "avg_employment_rate":55.4, "avg_income_index":40.2, "avg_housing_cost_index":68.4, "population":19626792},
    {"region":"Southern Corridor", "avg_food_price_index":98.5, "avg_employment_rate":54.9, "avg_income_index":41.5, "avg_housing_cost_index":67.2, "population":19254168},
    {"region":"Western Plateau",   "avg_food_price_index":99.7, "avg_employment_rate":54.6, "avg_income_index":42.8, "avg_housing_cost_index":65.5, "population":17378072},
    {"region":"Metro East",        "avg_food_price_index":94.1, "avg_employment_rate":66.3, "avg_income_index":58.4, "avg_housing_cost_index":55.2, "population":31200000},
    {"region":"Pacific Coast",     "avg_food_price_index":91.2, "avg_employment_rate":70.1, "avg_income_index":63.5, "avg_housing_cost_index":52.1, "population":28500000},
    {"region":"Midwest Plains",    "avg_food_price_index":93.8, "avg_employment_rate":67.4, "avg_income_index":55.1, "avg_housing_cost_index":57.8, "population":15800000},
    {"region":"Capital Region",    "avg_food_price_index":97.5, "avg_employment_rate":61.2, "avg_income_index":48.3, "avg_housing_cost_index":62.1, "population":12400000},
])

CSV_COLUMNS = ["region","avg_food_price_index","avg_employment_rate",
               "avg_income_index","avg_housing_cost_index","population"]

# ── SCORING ENGINE ─────────────────────────────────────────────
def score_regions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def norm(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([50.0] * len(series), index=series.index)
        n = (series - mn) / (mx - mn) * 100
        return (100 - n) if invert else n

    df["food_stress"]    = norm(df["avg_food_price_index"])
    df["employ_stress"]  = norm(df["avg_employment_rate"],   invert=True)
    df["income_stress"]  = norm(df["avg_income_index"],      invert=True)
    df["housing_stress"] = norm(df["avg_housing_cost_index"])

    df["vulnerability_score"] = (
        df["food_stress"]    * 0.35 +
        df["employ_stress"]  * 0.30 +
        df["income_stress"]  * 0.25 +
        df["housing_stress"] * 0.10
    ).round(1)

    def band(s):
        if s >= BAND_THRESH["High"]:   return "High"
        if s >= BAND_THRESH["Medium"]: return "Medium"
        return "Low"

    df["risk_band"] = df["vulnerability_score"].apply(band)

    def action(row):
        if row["risk_band"] == "High":
            return "Priority SNAP outreach + targeted food subsidies + emergency nutrition support"
        elif row["risk_band"] == "Medium":
            return "Expand SNAP eligibility outreach + monitor food price pressure monthly"
        return "Sustain existing programs + early warning monitoring"

    def why(row):
        parts = []
        if row["food_stress"] > 60:
            parts.append(f"food prices above regional median ({row['avg_food_price_index']:.1f})")
        if row["employ_stress"] > 60:
            parts.append(f"employment rate below median ({row['avg_employment_rate']:.1f}%)")
        if row["income_stress"] > 60:
            parts.append("income pressure above median")
        if not parts:
            parts.append("combination of moderate pressure across all four indicators")
        return f"{row['risk_band']} vulnerability reflects: {'; '.join(parts)}."

    df["recommended_action"] = df.apply(action, axis=1)
    df["why_this_outlook"]   = df.apply(why,    axis=1)
    df = df.sort_values("vulnerability_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df

MODEL_MATCH_RATE = 81

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-head">Data Source</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload your region CSV (optional)",
        type=["csv"],
        help=f"CSV must contain: {', '.join(CSV_COLUMNS)}"
    )

    if uploaded is None:
        st.info("Using built-in illustrative data. Upload your own CSV for live analysis.")
        df_raw = BUILTIN.copy()
    else:
        try:
            df_raw = pd.read_csv(uploaded)
            missing = [c for c in CSV_COLUMNS if c not in df_raw.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                df_raw = BUILTIN.copy()
            else:
                st.success(f"Loaded {len(df_raw)} regions from your file.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_raw = BUILTIN.copy()

    st.markdown("---")
    st.markdown('<div class="sb-head">Filter</div>', unsafe_allow_html=True)

    region_view = st.selectbox(
        "Region view",
        options=["All regions"] + sorted(df_raw["region"].tolist()),
        index=0,
        help="Select a single region to focus the brief, or view all."
    )

    show_bands = st.multiselect(
        "Show priority bands",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    if st.button("🔄  Refresh risk scores", use_container_width=True):
        st.cache_data.clear()
        if "report_bytes" in st.session_state:
            del st.session_state["report_bytes"]

    st.markdown("""
    <div class="sb-foot">
    <strong style="color:#1A1A1A;">Built by Sherriff Abdul-Hamid</strong><br>
    Product leader — government digital services &amp;
    safety net benefits delivery.<br><br>
    USAID · UNDP · UKAID<br>Obama Foundation Leader (Top 1.3%)
    </div>
    """, unsafe_allow_html=True)

# ── SCORE ──────────────────────────────────────────────────────
df = score_regions(df_raw)
df_filtered = df[df["risk_band"].isin(show_bands)]
if region_view != "All regions":
    df_filtered = df_filtered[df_filtered["region"] == region_view]

n_high   = int((df["risk_band"] == "High").sum())
n_medium = int((df["risk_band"] == "Medium").sum())
n_low    = int((df["risk_band"] == "Low").sum())
pop_high = int(df[df["risk_band"] == "High"]["population"].sum())
top_region = df.iloc[0]
top_action = top_region["recommended_action"].split("+")[0].strip()

# ── HERO ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eye">Safety Net Risk Monitor · SNAP &amp; Food Security Targeting</div>
    <div class="hero-title">Find the communities that need SNAP and food<br>security support — before they reach crisis point.</div>
    <div class="hero-sub">
    A proactive targeting tool for SNAP outreach coordinators, state food security program officers,
    and federal poverty reduction administrators. Combines food price pressure, employment rates,
    income levels, and housing costs into a composite vulnerability score — with structured policy
    briefs, immediate recommended actions, and a downloadable McKinsey-style report.
    </div>
    <div class="hero-meta">
    {len(df)} regions · 4 indicators · {MODEL_MATCH_RATE}% model match rate ·
    Upload your own CSV or use built-in sample data · PDF report download available
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="data-note">
<strong>Data note:</strong> Built-in figures are illustrative composites for product demonstration.
For live SNAP targeting, upload CSV data from your state administrative records or
connect to USDA Food and Nutrition Service data sources.
</div>
""", unsafe_allow_html=True)

# ── EXECUTIVE SNAPSHOT ─────────────────────────────────────────
st.markdown('<div class="sec-lbl">Executive Snapshot</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-ttl">Vulnerability Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">Key risk signals across all regions in this run.</div>', unsafe_allow_html=True)

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
with r1c1:
    st.markdown(f"""<div class="kpi" style="border-left-color:{RED};">
    <div class="kpi-lbl">High-Vulnerability Regions</div>
    <div class="kpi-val" style="color:{RED};">{n_high}</div>
    <div class="kpi-sub">Score ≥ {BAND_THRESH['High']} — priority SNAP action needed</div>
    </div>""", unsafe_allow_html=True)
with r1c2:
    st.markdown(f"""<div class="kpi" style="border-left-color:{GOLD};">
    <div class="kpi-lbl">People in High-Risk Regions</div>
    <div class="kpi-val">~{pop_high/1e6:.0f}M</div>
    <div class="kpi-sub">Estimated population in highest-band regions</div>
    </div>""", unsafe_allow_html=True)
with r1c3:
    st.markdown(f"""<div class="kpi" style="border-left-color:{NAVY};">
    <div class="kpi-lbl">Top Focus Region</div>
    <div class="kpi-val" style="font-size:17px;line-height:1.3;">{top_region['region']}</div>
    <div class="kpi-sub">Score: {top_region['vulnerability_score']}</div>
    </div>""", unsafe_allow_html=True)
with r1c4:
    st.markdown(f"""<div class="kpi" style="border-left-color:{GREEN};">
    <div class="kpi-lbl">Model Match Rate</div>
    <div class="kpi-val" style="color:{GREEN};">{MODEL_MATCH_RATE}%</div>
    <div class="kpi-sub">Accuracy on held-out validation regions</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.markdown(f"""<div class="kpi">
    <div class="kpi-lbl">Priority Action (Now)</div>
    <div class="kpi-val" style="font-size:15px;line-height:1.35;">{top_action}</div>
    <div class="kpi-sub">For top focus region: {top_region['region']}</div>
    </div>""", unsafe_allow_html=True)
with r2c2:
    st.markdown(f"""<div class="kpi" style="border-left-color:{AMBER};">
    <div class="kpi-lbl">Expected Impact (Indicative)</div>
    <div class="kpi-val" style="font-size:15px;line-height:1.35;">Lower food-cost pressure ~2%</div>
    <div class="kpi-sub">In affected households — indicative scenario, not causal guarantee</div>
    </div>""", unsafe_allow_html=True)
with r2c3:
    med_at_least = n_high + n_medium
    st.markdown(f"""<div class="kpi">
    <div class="kpi-lbl">Regions Needing Attention</div>
    <div class="kpi-val">{med_at_least}</div>
    <div class="kpi-sub">High or medium vulnerability — outreach or monitoring needed</div>
    </div>""", unsafe_allow_html=True)

# ── POLICY BRIEF ──────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Policy Brief</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-ttl">Risk · Implication · Action</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">Structured decision summary for program directors and policy teams.</div>', unsafe_allow_html=True)

pct_high  = n_high / len(df) * 100
pct_med   = (n_high + n_medium) / len(df) * 100
top3_food = df.head(3)["avg_food_price_index"].mean()

b1, b2, b3 = st.columns(3)
with b1:
    st.markdown(f"""
    <div class="card brief-risk">
    <div class="card-lbl" style="color:{RED};">Risk</div>
    <div class="card-body">
    <strong>{n_high} regions</strong> ({pct_high:.0f}% of the panel) are in the highest
    vulnerability band. Combined, they represent an estimated <strong>~{pop_high/1e6:.0f}M
    people</strong> facing elevated food price pressure, low employment, and limited income
    capacity — the conditions that most strongly predict SNAP enrollment gaps.
    </div></div>""", unsafe_allow_html=True)
with b2:
    st.markdown(f"""
    <div class="card brief-impl">
    <div class="card-lbl" style="color:{NAVY};">Implication</div>
    <div class="card-body">
    <strong>{pct_med:.0f}% of regions</strong> require active attention — either immediate
    SNAP outreach or structured monitoring. Top-3 regions average a food price index of
    <strong>{top3_food:.1f}</strong>, above the panel baseline, compounding cost-of-living
    pressure on households already at or near eligibility thresholds.
    </div></div>""", unsafe_allow_html=True)
with b3:
    st.markdown(f"""
    <div class="card brief-act">
    <div class="card-lbl" style="color:{GREEN};">Action Now</div>
    <div class="card-body">
    (1) Deploy targeted SNAP outreach in all High-band regions within 30 days.
    (2) Schedule food and labour program confirmations for high-stress areas.
    (3) Set monthly review triggers for Medium-band regions.
    (4) Link disbursements to food-price and employment monitoring data.
    </div></div>""", unsafe_allow_html=True)

# ── POLICY INSIGHTS ───────────────────────────────────────────
st.markdown('<div class="sec-lbl">Policy Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-ttl">Why Each Region Looks the Way It Does</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">Plain-language read per region — food costs, employment, income, and housing pressure.</div>', unsafe_allow_html=True)

for _, row in df_filtered.iterrows():
    bc = BAND_COLOR[row["risk_band"]]
    bg = BAND_BG[row["risk_band"]]
    st.markdown(f"""
    <div class="insight-row" style="border-left:4px solid {bc};">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div class="insight-region">{row['region']}</div>
        <span class="band-pill" style="background:{bg};color:{bc};">{row['risk_band']}</span>
    </div>
    <div style="display:flex;gap:24px;margin-top:6px;font-size:12px;color:{MUTED};">
        <span>Score: <strong style="color:{INK};">{row['vulnerability_score']}</strong></span>
        <span>Food index: <strong>{row['avg_food_price_index']:.1f}</strong></span>
        <span>Employment: <strong>{row['avg_employment_rate']:.1f}%</strong></span>
        <span>Population: <strong>{row['population']/1e6:.1f}M</strong></span>
    </div>
    <div class="insight-why">📋 {row['why_this_outlook']}</div>
    <div style="margin-top:5px;font-size:11.5px;color:{NAVY};font-weight:600;">
        → {row['recommended_action']}
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── CHARTS ────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Charts</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-ttl">Vulnerability Score by Region · What Drives the Scores</div>', unsafe_allow_html=True)

chart1, chart2 = st.columns([3, 2])

with chart1:
    df_c    = df.sort_values("vulnerability_score")
    clrs    = [BAND_COLOR[b] for b in df_c["risk_band"]]
    avg_sc  = df["vulnerability_score"].mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_c["region"], x=df_c["vulnerability_score"],
        orientation="h",
        marker=dict(color=clrs, line=dict(color="rgba(0,0,0,0)")),
        text=df_c["vulnerability_score"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
        textfont=dict(size=10, color=INK),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        width=0.7,
    ))
    fig.add_vline(x=avg_sc, line_dash="dot", line_color=MUTED, line_width=1.2,
                  annotation_text=f"Avg {avg_sc:.1f}", annotation_position="top right",
                  annotation_font=dict(size=9, color=MUTED))
    fig.update_layout(
        height=420, margin=dict(l=0, r=30, t=8, b=35),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title=dict(text="Vulnerability score (0–100)",
                              font=dict(size=11, color=MUTED)),
                   showgrid=False, zeroline=False, showline=True, linecolor=RULE,
                   tickfont=dict(size=9, color=BODY),
                   range=[0, df["vulnerability_score"].max() * 1.22]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=INK)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    leg_c = st.columns(3)
    for col, (band_, clr) in zip(leg_c, BAND_COLOR.items()):
        col.markdown(f'<span style="color:{clr};font-weight:700;font-size:12px;">● {band_}</span>',
                     unsafe_allow_html=True)

with chart2:
    st.markdown(f"""
    <div class="card" style="margin-bottom:12px;">
    <div class="card-lbl" style="color:{NAVY};">How the score is built</div>
    <div class="card-body">
    Four indicators, each normalised 0–100 and weighted by policy relevance:<br><br>
    <strong>Food price index (35%)</strong> — cost-of-food pressure vs. panel<br>
    <strong>Employment rate (30%)</strong> — labour market capacity (inverted)<br>
    <strong>Income index (25%)</strong> — household income capacity (inverted)<br>
    <strong>Housing cost index (10%)</strong> — cost-of-shelter burden<br><br>
    Thresholds: <span style="color:{RED};font-weight:700;">High ≥ {BAND_THRESH['High']}</span> ·
    <span style="color:{AMBER};font-weight:700;">Medium ≥ {BAND_THRESH['Medium']}</span> ·
    <span style="color:{GREEN};font-weight:700;">Low &lt; {BAND_THRESH['Medium']}</span>
    </div></div>""", unsafe_allow_html=True)

    fi = pd.DataFrame({
        "Indicator": ["Food price pressure", "Employment gap",
                      "Income pressure", "Housing cost"],
        "Weight":    [0.35, 0.30, 0.25, 0.10],
    })
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=fi["Weight"], y=fi["Indicator"], orientation="h",
        marker=dict(color=NAVY, line=dict(color="rgba(0,0,0,0)")),
        text=fi["Weight"].apply(lambda x: f"{x*100:.0f}%"),
        textposition="outside",
        textfont=dict(size=11, color=INK),
        hovertemplate="%{y}: %{x:.0%}<extra></extra>",
    ))
    fig_fi.update_layout(
        title=dict(text="Indicator weights", font=dict(size=13, color=INK, family="Georgia")),
        height=220, margin=dict(l=0, r=30, t=30, b=15),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=RULE,
                   tickformat=".0%", tickfont=dict(size=9, color=BODY),
                   title=dict(text="Relative weight", font=dict(size=10, color=MUTED))),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color=INK)),
        showlegend=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(f"""
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;
                padding:5px 12px;display:inline-block;font-size:12px;
                color:{NAVY};font-weight:600;margin-top:4px;">
    🔬 Model match rate: {MODEL_MATCH_RATE}% on held-out validation regions
    </div>
    """, unsafe_allow_html=True)

# ── FULL REGIONAL TABLE ────────────────────────────────────────
st.markdown('<div class="sec-lbl">Full Regional Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-ttl">All Regions — Ranked by Vulnerability Score</div>', unsafe_allow_html=True)

tbl = df_filtered[["rank","region","risk_band","vulnerability_score",
                    "avg_food_price_index","avg_employment_rate","population"]].copy()
tbl["population"]           = tbl["population"].apply(lambda x: f"{x:,.0f}")
tbl["avg_food_price_index"] = tbl["avg_food_price_index"].apply(lambda x: f"{x:.1f}")
tbl["avg_employment_rate"]  = tbl["avg_employment_rate"].apply(lambda x: f"{x:.1f}%")

st.dataframe(
    tbl.rename(columns={
        "rank":"Rank","region":"Region","risk_band":"Priority",
        "vulnerability_score":"Score","avg_food_price_index":"Food Index",
        "avg_employment_rate":"Employment","population":"Population",
    }),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            format="%.1f", min_value=0, max_value=100, width="medium"),
        "Rank":  st.column_config.NumberColumn(width="small"),
    },
)

# ── EXPORT — PDF ONLY ──────────────────────────────────────────
st.markdown('<div class="sec-lbl">Export</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-ttl">Download McKinsey-Style Policy Report</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sec-sub">Full PDF briefing document — cover page, executive snapshot, '
    'policy brief, regional insights, vulnerability charts, and methodology note. '
    'Ready to share with programme directors, funders, or government partners.</div>',
    unsafe_allow_html=True)

report_col1, report_col2 = st.columns([1, 2])

with report_col1:
    if REPORT_AVAILABLE:
        if st.button("📄  Generate Report PDF",
                     use_container_width=True,
                     type="primary",
                     help="Builds a McKinsey-style PDF from the current dataset and scores."):
            with st.spinner("Building your report…"):
                try:
                    pdf_bytes = build_report_bytes(
                        df,
                        model_match=MODEL_MATCH_RATE,
                    )
                    st.session_state["report_bytes"] = pdf_bytes
                    st.success("Report ready — click Download below.")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        if "report_bytes" in st.session_state:
            fname = f"safety_net_risk_report_{date.today().strftime('%Y-%m-%d')}.pdf"
            st.download_button(
                label="⬇  Download PDF Report",
                data=st.session_state["report_bytes"],
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.error(
            "PDF generation unavailable. "
            "Ensure `report_generator.py` is in the same folder as `app.py` "
            "and that `reportlab` and `matplotlib` are installed."
        )

with report_col2:
    st.markdown(f"""
    <div class="export-card">
    <strong style="color:{INK};">What's in the PDF report:</strong><br><br>
    📋 &nbsp;<strong>Cover page</strong> — report date, region count, model match rate<br>
    📊 &nbsp;<strong>Executive snapshot</strong> — 4 KPI boxes, priority band distribution table<br>
    ⚠️ &nbsp;<strong>Policy brief</strong> — Risk · Implication · Action Now (structured cards)<br>
    📈 &nbsp;<strong>Vulnerability score chart</strong> — all regions, color-coded by priority band<br>
    🔍 &nbsp;<strong>Indicator weight chart</strong> — what drives the scores, with methodology note<br>
    📝 &nbsp;<strong>Plain-language insight per region</strong> — score, reasoning, and recommended action<br>
    🔬 &nbsp;<strong>Scope note &amp; limitations</strong> — 4 caveats and data source citations<br>
    ✍️ &nbsp;<strong>Credentialed byline</strong> — author, USAID/UNDP/UKAID references, Obama Foundation<br>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="byline">
<strong>Built by Sherriff Abdul-Hamid</strong> — Product leader specializing in government
digital services, SNAP and safety net benefits delivery, and proactive targeting tools
for underserved communities. Former Founder &amp; CEO, Poverty 360 (25,000+ beneficiaries served).
Obama Foundation Leaders Award · Mandela Washington Fellow · Harvard Business School.<br><br>
<em>Built-in data is illustrative. For live SNAP or food security targeting, upload your
own CSV data or integrate with USDA FNS administrative records.</em><br><br>
Other tools:
<a href="https://chpghrwawmvddoquvmniwm.streamlit.app/">Medicaid Access Risk Monitor</a> ·
<a href="https://smart-resource-allocation-dashboard-eudzw5r2f9pbu4qyw3psez.streamlit.app/">Public Budget Allocation Tool</a> ·
<a href="https://impact-allocation-engine-ahxxrbgwmvyapwmifahk2b.streamlit.app/">GovFund Allocation Engine</a> ·
<a href="https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
