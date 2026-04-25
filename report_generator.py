"""
report_generator.py
McKinsey-style PDF report for the Safety Net Risk Monitor.

Generates a downloadable PDF with:
  - Cover page (black + gold, McKinsey sandwich)
  - Executive Snapshot
  - Policy Brief (Risk / Implication / Action)
  - Regional Priority Table
  - Policy Insights per region
  - Vulnerability Score Chart
  - Methodology Note
  - References & Byline

Usage (inside Streamlit app):
    from report_generator import build_report_bytes
    pdf_bytes = build_report_bytes(df_scored, budget_label, model_match)
    st.download_button("Download Report", pdf_bytes, file_name="report.pdf")
"""

from __future__ import annotations
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether,
)
from reportlab.platypus import NextPageTemplate
from reportlab.pdfgen import canvas as rl_canvas
from pypdf import PdfWriter, PdfReader

# ── PALETTE ───────────────────────────────────────────────────
C = {
    "black":     colors.HexColor("#0A0A0A"),
    "charcoal":  colors.HexColor("#1C1C1C"),
    "charcoal2": colors.HexColor("#252525"),
    "gold":      colors.HexColor("#C9A84C"),
    "gold_deep": colors.HexColor("#A67C2E"),
    "gold_light":colors.HexColor("#E8C97A"),
    "gold_wash": colors.HexColor("#2A2310"),
    "white":     colors.white,
    "off_white": colors.HexColor("#F8F5EE"),
    "grey1":     colors.HexColor("#F2EEE5"),
    "grey2":     colors.HexColor("#DDD8CC"),
    "grey3":     colors.HexColor("#9A9080"),
    "grey4":     colors.HexColor("#4A4035"),
    "navy":      colors.HexColor("#0A1F44"),
    "red":       colors.HexColor("#C8382A"),
    "amber":     colors.HexColor("#B8560A"),
    "green":     colors.HexColor("#1A7A2E"),
    "ink":       colors.HexColor("#111111"),
}

BAND_COLOR = {"High": C["red"], "Medium": C["amber"], "Low": C["green"]}
HEX_BAND   = {"High": "#C8382A", "Medium": "#B8560A", "Low": "#1A7A2E"}

# ── FONTS ─────────────────────────────────────────────────────
F_SERIF = "Times-Roman"
F_SERIF_B= "Times-Bold"
F_SANS  = "Helvetica"
F_SANS_B= "Helvetica-Bold"
F_SANS_I= "Helvetica-Oblique"

# ── PAGE DIMS (US Letter) ─────────────────────────────────────
PW, PH = letter          # 612 × 792 pt
ML = MR = 0.65 * inch
MT = MB = 0.65 * inch
CW = PW - ML - MR        # ~7.2"

# ── STYLES ────────────────────────────────────────────────────
def S(name, **kw):
    """Quick ParagraphStyle factory."""
    return ParagraphStyle(name, **kw)

SEC_LBL = S("sec_lbl", fontName=F_SANS_B,  fontSize=7.5,  textColor=C["grey3"],
            spaceAfter=3,  spaceBefore=14, leading=10, characterSpacing=2.2)
SEC_TTL = S("sec_ttl", fontName=F_SERIF_B, fontSize=18,   textColor=C["ink"],
            spaceAfter=4,  spaceBefore=2,  leading=22)
BODY    = S("body",    fontName=F_SANS,    fontSize=9.5,  textColor=C["grey4"],
            spaceAfter=6,  spaceBefore=0,  leading=14,   alignment=TA_JUSTIFY)
BODY_SM = S("body_sm", fontName=F_SANS,   fontSize=8.5,  textColor=C["grey4"],
            spaceAfter=4,  spaceBefore=0,  leading=13,   alignment=TA_JUSTIFY)
BULLET  = S("bullet",  fontName=F_SANS,   fontSize=9.5,  textColor=C["grey4"],
            spaceAfter=4,  spaceBefore=2,  leading=14,
            leftIndent=12, firstLineIndent=-12)
CARD_LBL= S("card_lbl",fontName=F_SANS_B, fontSize=7.5,  textColor=C["gold_deep"],
            spaceAfter=4,  spaceBefore=6,  leading=10,   characterSpacing=1.8)
KPI_N   = S("kpi_n",   fontName=F_SERIF_B,fontSize=28,   textColor=C["gold"],
            spaceAfter=2,  spaceBefore=4,  leading=32,   alignment=TA_CENTER)
KPI_L   = S("kpi_lbl", fontName=F_SANS,   fontSize=8.5,  textColor=C["gold_light"],
            spaceAfter=0,  spaceBefore=0,  leading=11,   alignment=TA_CENTER)
REF     = S("ref",     fontName=F_SANS,   fontSize=8.5,  textColor=C["grey4"],
            spaceAfter=5,  spaceBefore=0,  leading=13,   alignment=TA_JUSTIFY,
            leftIndent=18, firstLineIndent=-18)
EXHIBIT = S("exhibit", fontName=F_SANS_B, fontSize=7,    textColor=C["grey3"],
            spaceAfter=3,  spaceBefore=8,  leading=10,   characterSpacing=2)
CHART_N = S("chart_n", fontName=F_SANS_I, fontSize=7.5, textColor=C["grey3"],
            spaceAfter=3,  spaceBefore=2,  leading=10)
COVER_EYE = S("cov_eye", fontName=F_SANS_B, fontSize=8.5, textColor=C["gold"],
              spaceAfter=6, spaceBefore=0, leading=11, characterSpacing=2)
COVER_TTL = S("cov_ttl", fontName=F_SERIF_B, fontSize=38, textColor=C["white"],
              spaceAfter=8, spaceBefore=4, leading=44)
COVER_SUB = S("cov_sub", fontName=F_SANS_I,  fontSize=13, textColor=C["gold_light"],
              spaceAfter=16, spaceBefore=0, leading=18)
COVER_BY  = S("cov_by",  fontName=F_SANS,    fontSize=10, textColor=C["grey3"],
              spaceAfter=4, spaceBefore=0, leading=14)

# ── HELPERS ───────────────────────────────────────────────────
def gold_rule():
    return HRFlowable(width="100%", thickness=3, color=C["gold"],
                      spaceAfter=4, spaceBefore=0)

def thin_rule():
    return HRFlowable(width="100%", thickness=0.5, color=C["grey2"],
                      spaceAfter=4, spaceBefore=0)

def spacer(h=6):
    return Spacer(1, h)

def bullet_p(txt):
    return Paragraph(f"• {txt}", BULLET)

def kpi_cell(num, lbl, bg=None):
    bg = bg or C["charcoal"]
    t = Table([[Paragraph(num, KPI_N)], [Paragraph(lbl, KPI_L)]],
              colWidths=[1.7*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    return t

def insight_card(label, body_text, bc=None):
    bc = bc or C["gold"]
    lp = Paragraph(label.upper(), CARD_LBL)
    lp.style = ParagraphStyle("cl", parent=CARD_LBL, textColor=bc)
    bp = Paragraph(body_text, BODY_SM)
    t = Table([[lp],[bp]], colWidths=[CW*0.45])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C["white"]),
        ("BOX",        (0,0),(-1,-1), 0.5, C["grey2"]),
        ("LINEBEFORE", (0,0),(0,-1),  3,   bc),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING", (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    return t

# ── CHARTS ────────────────────────────────────────────────────
def _chart_img(fig, w_in, h_in):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=w_in*inch, height=h_in*inch)

def make_vuln_chart(df: pd.DataFrame) -> object:
    df_c = df.sort_values("vulnerability_score")
    fig, ax = plt.subplots(figsize=(5.6, max(2.6, len(df_c)*0.28)))
    fig.patch.set_facecolor("#F8F5EE")
    ax.set_facecolor("#F8F5EE")

    palette = [HEX_BAND[b] for b in df_c["risk_band"]]
    bars = ax.barh(df_c["region"], df_c["vulnerability_score"],
                   color=palette, height=0.6)
    for bar, val in zip(bars, df_c["vulnerability_score"]):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="#111111")

    avg = df_c["vulnerability_score"].mean()
    ax.axvline(avg, color="#9A9080", linewidth=1.1, linestyle="--")
    ax.text(avg + 0.3, -0.5, f"Avg {avg:.1f}", fontsize=7.5,
            color="#9A9080", ha="left", va="top")

    ax.set_xlim(0, df_c["vulnerability_score"].max() * 1.22)
    ax.set_xlabel("Vulnerability score (0–100)", fontsize=8.5, color="#9A9080")
    ax.tick_params(axis="both", labelsize=8.5, colors="#4A4035")
    for spine in ["top","right","left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#DDD8CC")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    # Legend patches
    patches = [mpatches.Patch(color=HEX_BAND[b], label=b)
               for b in ["High","Medium","Low"]]
    ax.legend(handles=patches, fontsize=8, framealpha=0,
              loc="lower right", labelcolor="#4A4035")
    fig.tight_layout(pad=0.4)
    return _chart_img(fig, 5.6, max(2.6, len(df_c)*0.28))

def make_weight_chart():
    labels  = ["Food price pressure","Employment gap",
               "Income pressure","Housing cost"]
    weights = [0.35, 0.30, 0.25, 0.10]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    fig.patch.set_facecolor("#F8F5EE")
    ax.set_facecolor("#F8F5EE")
    bars = ax.barh(labels, weights, color="#0A1F44", height=0.55)
    for bar, val in zip(bars, weights):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val*100:.0f}%", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="#111111")
    ax.set_xlim(0, 0.48)
    ax.set_xlabel("Relative weight", fontsize=8, color="#9A9080")
    ax.tick_params(axis="both", labelsize=8, colors="#4A4035")
    for spine in ["top","right","left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#DDD8CC")
    ax.xaxis.grid(False); ax.yaxis.grid(False)
    fig.tight_layout(pad=0.3)
    return _chart_img(fig, 3.2, 2.0)

# ── PAGE FRAMES ───────────────────────────────────────────────
def cover_frame(canvas, doc):
    canvas.saveState()
    # Full black
    canvas.setFillColorRGB(0.063, 0.063, 0.063)
    canvas.rect(0, 0, PW, PH, fill=1, stroke=0)
    # Gold left stripe
    canvas.setFillColor(C["gold"])
    canvas.rect(0, 0, 0.12*inch, PH, fill=1, stroke=0)
    # Geometric blocks top-right
    canvas.setFillColorRGB(0.145, 0.145, 0.145)
    canvas.rect(PW - 3.0*inch, PH - 3.0*inch, 3.0*inch, 3.0*inch, fill=1, stroke=0)
    canvas.setFillColorRGB(0.165, 0.165, 0.165)
    canvas.rect(PW - 2.5*inch, PH - 2.5*inch, 2.5*inch, 2.5*inch, fill=1, stroke=0)
    canvas.setFillColorRGB(0.165, 0.137, 0.063)
    canvas.rect(PW - 2.0*inch, PH - 2.0*inch, 2.0*inch, 2.0*inch, fill=1, stroke=0)
    canvas.setFillColor(C["gold_deep"])
    canvas.rect(PW - 1.4*inch, PH - 1.4*inch, 1.4*inch, 1.4*inch, fill=1, stroke=0)
    # Footer bar
    canvas.setFillColorRGB(0.04, 0.04, 0.04)
    canvas.rect(0, 0, PW, 0.30*inch, fill=1, stroke=0)
    canvas.restoreState()

def body_frame_fn(canvas, doc):
    canvas.saveState()
    # White background — prevents cover bleed
    canvas.setFillColorRGB(0.969, 0.961, 0.933)
    canvas.rect(0, 0, PW, PH, fill=1, stroke=0)
    # Gold top rule
    canvas.setFillColor(C["gold"])
    canvas.rect(0, PH - 0.055*inch, PW, 0.055*inch, fill=1, stroke=0)
    # Footer rule
    canvas.setStrokeColor(C["grey2"])
    canvas.setLineWidth(0.5)
    canvas.line(ML, MB - 0.12*inch, PW - MR, MB - 0.12*inch)
    # Footer text
    canvas.setFont(F_SANS_I, 7)
    canvas.setFillColor(C["grey3"])
    canvas.drawString(ML, MB - 0.22*inch,
        "Confidential  |  Safety Net Risk Monitor  |  Sherriff Abdul-Hamid")
    canvas.drawRightString(PW - MR, MB - 0.22*inch, f"Page {doc.page}")
    canvas.restoreState()

# ── MAIN BUILD FUNCTION ───────────────────────────────────────
def build_report_bytes(
    df: pd.DataFrame,
    model_match: int = 81,
    report_date: str | None = None,
) -> bytes:
    """
    Generate McKinsey-style PDF from scored DataFrame.
    Returns bytes suitable for st.download_button.
    """
    if report_date is None:
        report_date = date.today().strftime("%B %d, %Y")

    n_high   = (df["risk_band"] == "High").sum()
    n_med    = (df["risk_band"] == "Medium").sum()
    n_low    = (df["risk_band"] == "Low").sum()
    pop_high = int(df[df["risk_band"] == "High"]["population"].sum())
    top_row  = df.sort_values("vulnerability_score", ascending=False).iloc[0]
    avg_sc   = df["vulnerability_score"].mean()
    n_attn   = n_high + n_med
    pct_high = n_high / len(df) * 100
    top3_food= df.nlargest(3, "vulnerability_score")["avg_food_price_index"].mean()

    story = []

    # ═══════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═══════════════════════════════════════════════════════════
    story += [
        spacer(90),
        Paragraph("SAFETY NET RISK MONITOR  ·  COMMUNITY VULNERABILITY REPORT", COVER_EYE),
        spacer(6),
        Paragraph("Safety Net Risk Monitor", COVER_TTL),
        Paragraph("SNAP & Food Security Vulnerability Targeting for Program Officers",
                  COVER_SUB),
        HRFlowable(width=2.2*inch, thickness=2.5, color=C["gold"],
                   spaceAfter=12, spaceBefore=0, hAlign="LEFT"),
        Paragraph(f"Prepared by Sherriff Abdul-Hamid", COVER_BY),
        Paragraph(f"Report date: {report_date}  ·  Regions analysed: {len(df)}  ·  "
                  f"Model match rate: {model_match}%", COVER_BY),
        Paragraph("Illustrative model for demonstration purposes", COVER_BY),
        PageBreak(),
    ]

    # ═══════════════════════════════════════════════════════════
    # PAGE 2 — EXECUTIVE SNAPSHOT
    # ═══════════════════════════════════════════════════════════
    story += [
        Paragraph("EXECUTIVE SNAPSHOT", SEC_LBL),
        gold_rule(),
        Paragraph("High-Vulnerability Regions Identified — Priority SNAP Action Required", SEC_TTL),
        spacer(8),
    ]

    # 4-KPI row
    kpi_row = Table(
        [[kpi_cell(str(n_high),  "High-risk\nregions",         C["charcoal"]),
          kpi_cell(f"~{pop_high//1_000_000}M","People in\nhigh-risk areas",  C["charcoal"]),
          kpi_cell(f"{model_match}%", "Model\nmatch rate",     C["gold_deep"]),
          kpi_cell(str(n_attn),  "Regions needing\nattention", C["charcoal"])]],
        colWidths=[1.7*inch]*4,
        rowHeights=[0.88*inch],
    )
    kpi_row.setStyle(TableStyle([
        ("ALIGN",  (0,0),(-1,-1), "CENTER"),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0),(-1,-1), 4),
        ("RIGHTPADDING",(0,0),(-1,-1), 4),
    ]))
    story += [kpi_row, spacer(14), thin_rule()]

    # Overview narrative
    story += [
        Paragraph("OVERVIEW", SEC_LBL),
        Paragraph(
            f"This report covers {len(df)} regions. "
            f"{n_high} regions ({pct_high:.0f}%) are in the highest vulnerability band, "
            f"representing an estimated {pop_high:,.0f} people facing elevated food price "
            f"pressure, employment gaps, and limited income capacity — the conditions most "
            f"strongly associated with SNAP enrollment shortfalls. "
            f"The top-3 regions average a food price index of {top3_food:.1f}, above the "
            f"panel baseline, compounding cost-of-living pressure on households at or near "
            f"eligibility thresholds. "
            f"A scoring model assigns priority bands and achieves a {model_match}% match rate "
            f"on held-out validation data, indicating reliable signal for initial triage.",
            BODY),
        spacer(6),
    ]

    # Priority table
    story += [
        Paragraph("PRIORITY BAND DISTRIBUTION", SEC_LBL),
    ]
    hdr = dict(fontName=F_SANS_B, textColor=C["white"])
    band_tbl = Table(
        [[Paragraph("<b>Priority Band</b>", ParagraphStyle("th", fontName=F_SANS_B, fontSize=9, textColor=C["white"])),
          Paragraph("<b>Regions</b>",        ParagraphStyle("th", fontName=F_SANS_B, fontSize=9, textColor=C["white"])),
          Paragraph("<b>Definition</b>",     ParagraphStyle("th", fontName=F_SANS_B, fontSize=9, textColor=C["white"])),
          Paragraph("<b>Recommended action</b>", ParagraphStyle("th", fontName=F_SANS_B, fontSize=9, textColor=C["white"]))],
         [Paragraph("High",    ParagraphStyle("r1", fontName=F_SANS_B, fontSize=9, textColor=C["red"])),
          Paragraph(str(n_high), ParagraphStyle("r1v", fontName=F_SANS_B, fontSize=9, textColor=C["red"])),
          Paragraph("Score ≥ 65", ParagraphStyle("r1d", fontSize=9, textColor=C["grey4"])),
          Paragraph("Priority SNAP outreach + targeted food subsidies", ParagraphStyle("r1a", fontSize=9, textColor=C["grey4"]))],
         [Paragraph("Medium",  ParagraphStyle("r2", fontName=F_SANS_B, fontSize=9, textColor=C["amber"])),
          Paragraph(str(n_med),  ParagraphStyle("r2v", fontName=F_SANS_B, fontSize=9, textColor=C["amber"])),
          Paragraph("Score 40–64", ParagraphStyle("r2d", fontSize=9, textColor=C["grey4"])),
          Paragraph("Expand eligibility outreach + monthly monitoring", ParagraphStyle("r2a", fontSize=9, textColor=C["grey4"]))],
         [Paragraph("Low",     ParagraphStyle("r3", fontName=F_SANS_B, fontSize=9, textColor=C["green"])),
          Paragraph(str(n_low),  ParagraphStyle("r3v", fontName=F_SANS_B, fontSize=9, textColor=C["green"])),
          Paragraph("Score < 40", ParagraphStyle("r3d", fontSize=9, textColor=C["grey4"])),
          Paragraph("Sustain programs + early-warning monitoring", ParagraphStyle("r3a", fontSize=9, textColor=C["grey4"]))],
        ],
        colWidths=[0.95*inch, 0.65*inch, 1.3*inch, 4.1*inch],
    )
    band_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C["charcoal"]),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C["white"], C["grey1"], C["white"]]),
        ("GRID",          (0,0),(-1,-1), 0.4, C["grey2"]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("RIGHTPADDING",  (0,0),(-1,-1), 7),
    ]))
    story += [band_tbl, spacer(10), PageBreak()]

    # ═══════════════════════════════════════════════════════════
    # PAGE 3 — POLICY BRIEF (Risk / Implication / Action)
    # ═══════════════════════════════════════════════════════════
    story += [
        Paragraph("POLICY BRIEF", SEC_LBL),
        gold_rule(),
        Paragraph("Risk · Implication · Action — Structured Decision Summary for Program Directors",
                  SEC_TTL),
        spacer(8),
    ]

    brief_data = [
        ("RISK",       C["red"],
         f"{n_high} region(s) ({pct_high:.0f}% of this panel) are in the highest vulnerability band. "
         f"Combined, they represent an estimated {pop_high:,.0f} people facing elevated food price "
         f"pressure and low employment — the conditions most predictive of SNAP coverage gaps. "
         f"Budget concentration risk is elevated if outreach resources are spread evenly across all regions."),
        ("IMPLICATION",C["navy"],
         f"{n_attn} of {len(df)} regions require active attention — either immediate SNAP outreach "
         f"or structured monitoring. Top-3 regions average a food price index of {top3_food:.1f}, "
         f"above the panel baseline, compounding cost pressure on households at or near eligibility "
         f"thresholds. Without proactive intervention, caseload spikes in high-band regions are likely "
         f"to exceed administrative processing capacity."),
        ("ACTION NOW", C["green"],
         f"(1) Deploy targeted SNAP outreach in all High-band regions within 30 days. "
         f"(2) Schedule food and labour programme confirmations for high-stress areas. "
         f"(3) Set monthly review triggers for Medium-band regions. "
         f"(4) Link disbursements to food-price and employment monitoring data. "
         f"(5) Flag {top_row['region']} (highest score: {top_row['vulnerability_score']:.1f}) "
         f"for immediate programme manager review."),
    ]

    for lbl, bc, body_txt in brief_data:
        lbl_p  = Paragraph(lbl, ParagraphStyle("bl", parent=CARD_LBL, textColor=bc))
        body_p = Paragraph(body_txt, BODY_SM)
        card = Table([[lbl_p],[body_p]], colWidths=[CW])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), C["white"]),
            ("BOX",        (0,0),(-1,-1), 0.5, C["grey2"]),
            ("LINEBEFORE", (0,0),(0,-1),  4,   bc),
            ("LINEABOVE",  (0,0),(-1,0),  0.5, C["grey2"]),
            ("LEFTPADDING",(0,0),(-1,-1), 10),
            ("RIGHTPADDING",(0,0),(-1,-1),10),
            ("TOPPADDING", (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ]))
        story += [card, spacer(8)]

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # PAGE 4 — VULNERABILITY CHART + METHODOLOGY
    # ═══════════════════════════════════════════════════════════
    story += [
        Paragraph("EXHIBIT 1  |  VULNERABILITY SCORE BY REGION", EXHIBIT),
        gold_rule(),
        Paragraph("All Regions Ranked by Composite Vulnerability Score — Priority Band Color-Coded",
                  SEC_TTL),
        spacer(6),
    ]

    chart_img = make_vuln_chart(df)
    weight_img= make_weight_chart()

    chart_row = Table(
        [[chart_img, Spacer(0.15*inch, 1), weight_img]],
        colWidths=[5.6*inch, 0.15*inch, 3.2*inch],
    )
    chart_row.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story += [
        chart_row,
        Paragraph("← Lower score = lower vulnerability.  Dashed line = panel average.  "
                  "Color = priority band (red = High, amber = Medium, green = Low).", CHART_N),
        spacer(10), thin_rule(),
    ]

    # Methodology box
    story += [
        Paragraph("METHODOLOGY NOTE", SEC_LBL),
        Paragraph("How the Vulnerability Score Is Constructed", SEC_TTL),
        spacer(4),
        Paragraph(
            "The composite vulnerability score combines four publicly available indicators, "
            "each normalised 0–100 (higher = more stress) and weighted by policy relevance:",
            BODY),
    ]
    for lbl, wt, src in [
        ("Food price index (35%)",    "Primary driver",  "USDA Economic Research Service Food Price Outlook"),
        ("Employment rate (30%)",     "Labour market capacity — inverted (low employment = high stress)",
                                                         "BLS Local Area Unemployment Statistics"),
        ("Income index (25%)",        "Household capacity to afford food and care — inverted",
                                                         "ACS Table B19013 — Median Household Income"),
        ("Housing cost index (10%)",  "Cost-of-shelter burden relative to national baseline",
                                                         "HUD Fair Market Rents"),
    ]:
        story.append(bullet_p(f"<b>{lbl}</b> — {src}"))

    story += [
        spacer(6),
        Paragraph(
            f"Priority bands: High (score ≥ 65), Medium (40–64), Low (< 40). "
            f"A logistic regression classifier assigns bands and achieves a {model_match}% "
            f"match rate on held-out validation regions — providing reliable signal for "
            f"initial triage. Field validation with administrative programme data is "
            f"recommended before use in official targeting cycles.",
            BODY),
        PageBreak(),
    ]

    # ═══════════════════════════════════════════════════════════
    # PAGE 5 — REGIONAL POLICY INSIGHTS
    # ═══════════════════════════════════════════════════════════
    story += [
        Paragraph("EXHIBIT 2  |  REGIONAL POLICY INSIGHTS", EXHIBIT),
        gold_rule(),
        Paragraph("Why Each Region Is Ranked as It Is — Indicators, Score, and Recommended Action",
                  SEC_TTL),
        spacer(6),
    ]

    # Table header
    th = ParagraphStyle("th2", fontName=F_SANS_B, fontSize=8.5, textColor=C["white"])
    td = ParagraphStyle("td2", fontName=F_SANS,    fontSize=8.5, textColor=C["grey4"])
    td_b = ParagraphStyle("tdb",fontName=F_SANS_B, fontSize=8.5, textColor=C["ink"])

    rows = [[
        Paragraph("Region",       th),
        Paragraph("Band",         th),
        Paragraph("Score",        th),
        Paragraph("Food Idx",     th),
        Paragraph("Employment",   th),
        Paragraph("Population",   th),
    ]]
    for _, row in df.sort_values("vulnerability_score", ascending=False).iterrows():
        bc = BAND_COLOR[row["risk_band"]]
        rows.append([
            Paragraph(row["region"],                        td_b),
            Paragraph(f'<font color="{HEX_BAND[row["risk_band"]]}">'
                      f'<b>{row["risk_band"]}</b></font>', td_b),
            Paragraph(f'{row["vulnerability_score"]:.1f}', td_b),
            Paragraph(f'{row["avg_food_price_index"]:.1f}',td),
            Paragraph(f'{row["avg_employment_rate"]:.1f}%',td),
            Paragraph(f'{row["population"]/1e6:.1f}M',     td),
        ])

    ins_tbl = Table(rows, colWidths=[2.0*inch, 0.75*inch, 0.65*inch,
                                      0.75*inch, 0.95*inch, 0.85*inch])
    ins_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C["charcoal"]),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C["white"], C["grey1"]]),
        ("GRID",          (0,0),(-1,-1), 0.4, C["grey2"]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
    ]))
    story += [ins_tbl, spacer(10)]

    # Narrative insights
    story.append(Paragraph("PLAIN-LANGUAGE INSIGHTS BY REGION", SEC_LBL))
    for _, row in df.sort_values("vulnerability_score", ascending=False).iterrows():
        bc  = BAND_COLOR[row["risk_band"]]
        hx  = HEX_BAND[row["risk_band"]]
        lp  = Paragraph(
            f'<font color="{hx}"><b>{row["region"]}</b></font>  ·  '
            f'{row["risk_band"]} (score {row["vulnerability_score"]:.1f})',
            ParagraphStyle("ri_h", fontName=F_SANS_B, fontSize=9.5,
                           textColor=C["ink"], spaceBefore=8, spaceAfter=2, leading=13)
        )
        wp  = Paragraph(f'<i>{row["why_this_outlook"]}</i>', BODY_SM)
        ap  = Paragraph(f'→ <b>{row["recommended_action"]}</b>',
                        ParagraphStyle("ri_a", fontName=F_SANS, fontSize=8.5,
                                       textColor=C["navy"], spaceBefore=2, spaceAfter=2,
                                       leading=12))
        inner = Table([[lp],[wp],[ap]], colWidths=[CW])
        inner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), C["white"]),
            ("BOX",           (0,0),(-1,-1), 0.5, C["grey2"]),
            ("LINEBEFORE",    (0,0),(0,-1),  3,   bc),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ]))
        story += [inner, spacer(5)]

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # PAGE 6 — SCOPE NOTE + BYLINE
    # ═══════════════════════════════════════════════════════════
    story += [
        Paragraph("SCOPE NOTE & LIMITATIONS", SEC_LBL),
        gold_rule(),
        Paragraph("Results Should Be Interpreted in Context — Four Key Caveats",
                  SEC_TTL),
        spacer(8),
    ]

    caveats = [
        ("Illustrative data",
         "Built-in figures are composite sample data for product demonstration. "
         "Production SNAP targeting requires official data from USDA FNS, ACS, "
         "BLS, and state administrative records."),
        ("Model limitations",
         f"The {model_match}% match rate reflects accuracy on a held-out sample. "
         "Rankings should be validated against programme-level enrolment data "
         "and case supervisor review before use in official targeting cycles."),
        ("Impact estimates",
         "Figures such as 'could lower food-cost pressure by ~2%' are indicative "
         "scenario ranges, not causal guarantees. Use as planning estimates only."),
        ("External validity",
         "Indicator weights reflect general policy relevance. Local conditions — "
         "geography, programme capacity, political context — may shift optimal "
         "targeting priorities in ways the model cannot fully capture."),
    ]

    for i, (title, body_txt) in enumerate(caveats):
        n_p  = Paragraph(str(i+1), ParagraphStyle("cn", fontName=F_SANS_B,
                         fontSize=20, textColor=C["gold_deep"], leading=24))
        t_p  = Paragraph(f"<b>{title}</b>", ParagraphStyle("ct", fontName=F_SANS_B,
                         fontSize=9.5, textColor=C["ink"], leading=13))
        b_p  = Paragraph(body_txt, ParagraphStyle("cb", fontSize=8.5,
                         textColor=C["grey4"], leading=12))
        cell = Table([[n_p, t_p],[n_p, b_p]],
                     colWidths=[0.35*inch, CW/2 - 0.5*inch])
        # simpler: one-column card
        card2= Table([[t_p],[b_p]], colWidths=[CW/2 - 0.2*inch])
        card2.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), C["white"]),
            ("BOX",        (0,0),(-1,-1), 0.5, C["grey2"]),
            ("LINEABOVE",  (0,0),(-1,0),  3,   C["gold_deep"]),
            ("LEFTPADDING",(0,0),(-1,-1), 8),
            ("RIGHTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING", (0,0),(-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ]))
        if i % 2 == 0:
            _buf_left = card2
        else:
            pair = Table([[_buf_left, Spacer(0.12*inch,1), card2]],
                         colWidths=[CW/2 - 0.1*inch, 0.12*inch, CW/2 - 0.1*inch])
            pair.setStyle(TableStyle([
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),0),
                ("RIGHTPADDING",(0,0),(-1,-1),0),
            ]))
            story += [pair, spacer(10)]
    if len(caveats) % 2 != 0:
        story += [_buf_left, spacer(10)]

    story += [
        spacer(10), thin_rule(), spacer(6),
        Paragraph(
            "Prepared by <b>Sherriff Abdul-Hamid</b> — Product leader specializing in "
            "government digital services, SNAP and safety net benefits delivery, and "
            "proactive targeting tools for historically underserved communities. "
            "Former Founder & CEO, Poverty 360 (25,000+ beneficiaries served across West Africa). "
            "Obama Foundation Leaders Award (Top 1.3%) · Mandela Washington Fellow (Top 0.3%) · "
            "Harvard Business School.",
            ParagraphStyle("byline", fontName=F_SANS_I, fontSize=8, textColor=C["grey3"],
                           spaceAfter=4, spaceBefore=0, leading=12)),
        Paragraph(
            f"Report generated: {report_date}  |  "
            "All built-in data is illustrative. "
            "For official SNAP programme targeting, pair with USDA FNS and state administrative data.",
            ParagraphStyle("scope2", fontName=F_SANS_I, fontSize=7.5, textColor=C["grey3"],
                           spaceAfter=0, spaceBefore=0, leading=11)),
    ]

    # ═══════════════════════════════════════════════════════════
    # BUILD — Two-PDF merge strategy to prevent canvas bleed
    # ═══════════════════════════════════════════════════════════
    cover_buf = io.BytesIO()
    c = rl_canvas.Canvas(cover_buf, pagesize=letter)
    cover_frame(c, type("D", (), {"page": 1})())
    # Draw cover text directly
    c.setFont(F_SANS_B, 8.5)
    c.setFillColor(C["gold"])
    c.drawString(ML + 0.15*inch, PH - 1.85*inch,
                 "SAFETY NET RISK MONITOR  ·  COMMUNITY VULNERABILITY REPORT")
    c.setFont(F_SERIF_B, 38)
    c.setFillColor(C["white"])
    c.drawString(ML + 0.15*inch, PH - 2.62*inch, "Safety Net")
    c.drawString(ML + 0.15*inch, PH - 3.08*inch, "Risk Monitor")
    c.setFont(F_SANS_I, 13)
    c.setFillColor(C["gold_light"])
    c.drawString(ML + 0.15*inch, PH - 3.52*inch,
                 "SNAP & Food Security Vulnerability Targeting for Program Officers")
    c.setFillColor(C["gold"])
    c.rect(ML + 0.15*inch, PH - 3.78*inch, 2.2*inch, 0.04*inch, fill=1, stroke=0)
    c.setFont(F_SANS, 10)
    c.setFillColor(C["grey3"])
    c.drawString(ML + 0.15*inch, PH - 4.04*inch, "Prepared by Sherriff Abdul-Hamid")
    c.setFont(F_SANS_I, 8.5)
    c.setFillColor(colors.HexColor("#4A4030"))
    c.drawString(ML + 0.15*inch, PH - 4.28*inch,
                 f"Report date: {report_date}  ·  Regions: {len(df)}  ·  "
                 f"Model match rate: {model_match}%")
    c.save()

    body_buf = io.BytesIO()

    class _WBCanvas(rl_canvas.Canvas):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._pg = 2
            self._paint()
        def _paint(self):
            self.saveState()
            self.setFillColorRGB(0.969, 0.961, 0.933)
            self.rect(0, 0, PW, PH, fill=1, stroke=0)
            self.setFillColor(C["gold"])
            self.rect(0, PH - 0.055*inch, PW, 0.055*inch, fill=1, stroke=0)
            self.setStrokeColor(C["grey2"])
            self.setLineWidth(0.5)
            self.line(ML, MB - 0.12*inch, PW - MR, MB - 0.12*inch)
            self.setFont(F_SANS_I, 7)
            self.setFillColor(C["grey3"])
            self.drawString(ML, MB - 0.22*inch,
                "Confidential  |  Safety Net Risk Monitor  |  Sherriff Abdul-Hamid")
            self.drawRightString(PW - MR, MB - 0.22*inch, f"Page {self._pg}")
            self.restoreState()
        def showPage(self):
            super().showPage()
            self._pg += 1
            self._paint()

    from reportlab.platypus import SimpleDocTemplate
    body_doc = SimpleDocTemplate(
        body_buf, pagesize=letter,
        leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title="Safety Net Risk Monitor Report",
        author="Sherriff Abdul-Hamid",
    )
    body_doc.build(story, canvasmaker=_WBCanvas)

    # Merge
    out = io.BytesIO()
    writer = PdfWriter()
    for buf in [cover_buf, body_buf]:
        buf.seek(0)
        for page in PdfReader(buf).pages:
            writer.add_page(page)
    writer.write(out)
    return out.getvalue()
