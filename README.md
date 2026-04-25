# Community Vulnerability & Safety Net Targeting Tool
### SNAP and Food Security Risk Monitor for Program Officers

**Built by Sherriff Abdul-Hamid**  
Product leader specializing in government digital services, SNAP and safety net
benefits delivery, and proactive targeting tools for underserved communities.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://povertyearlywarningsystem-7rrmkktbi7bwha2nna8gk7.streamlit.app/)

---

## The Problem This Solves

> *Find the communities that need SNAP and food security support — before they reach crisis point.*

SNAP outreach coordinators, state food security program officers, and federal poverty
reduction administrators share a common challenge: identifying which communities face
the highest vulnerability to food insecurity *before* caseloads spike, not after.
Reactive responses cost more and reach fewer people.

This tool gives program officers a composite vulnerability score per region — built
from food price pressure, employment rates, income levels, and housing costs —
with structured policy briefs and immediate recommended actions for each area.

---

## What This Tool Produces

| Output | Description |
|---|---|
| **Executive Snapshot** | 6-KPI summary: high-risk regions, population exposed, top focus region, model accuracy, priority action, expected impact |
| **Policy Brief** | Structured three-part summary: Risk · Implication · Action Now |
| **Policy Insights Panel** | Per-region cards with plain-language explanation of why each area is ranked as it is |
| **Vulnerability Score Chart** | All regions ranked by score, color-coded by priority band (High/Medium/Low) |
| **Indicator Weight Chart** | Transparent display of how each indicator contributes to the composite score |
| **Regional Summary Table** | Full panel ranked by vulnerability score with all indicators |
| **CSV Export** | Full results including recommended actions and explanations — briefing-document ready |
| **CSV Upload** | Upload your own regional data for live analysis with the same scoring engine |

---

## Vulnerability Score — Method

```
vulnerability_score =
    food_price_stress    × 0.35   (primary driver: cost-of-food pressure)
  + employment_stress    × 0.30   (labour market capacity — inverted)
  + income_stress        × 0.25   (household income capacity — inverted)
  + housing_cost_stress  × 0.10   (cost-of-shelter burden)
```

Each indicator is normalised 0–100 before weighting. Higher score = higher vulnerability.

**Priority bands:**
- 🔴 **High** — score ≥ 65: Priority SNAP outreach + targeted food subsidies + emergency nutrition support
- 🟡 **Medium** — score 40–64: Expand SNAP eligibility outreach + monthly food price monitoring
- 🟢 **Low** — score < 40: Sustain existing programs + early warning monitoring

**Model validation:**  
A scoring model assigns priority bands and is validated against held-out regional data,
achieving an **81% match rate** — the highest accuracy across all tools in this portfolio.

---

## Input Data Fields

Upload a CSV with these columns for live analysis:

| Field | Type | Description |
|---|---|---|
| `region` | string | Region or county name |
| `avg_food_price_index` | float | Average food price index (national baseline = 100) |
| `avg_employment_rate` | float | Employment rate as a percentage (0–100) |
| `avg_income_index` | float | Composite household income index (0–100) |
| `avg_housing_cost_index` | float | Relative housing cost index (national baseline = 100) |
| `population` | integer | Total population of the region |

**Data sources for production SNAP targeting:**
- **Food price index:** USDA Economic Research Service Food Price Outlook
- **Employment rate:** Bureau of Labor Statistics Local Area Unemployment Statistics (LAUS)
- **Income index:** ACS Table B19013 — Median Household Income
- **Housing cost:** HUD Fair Market Rents or ACS housing cost data

---

## Repository Structure

```
├── app.py                  # Main Streamlit UI — briefs, insights, charts, table
├── scoring.py              # Vulnerability scoring engine and band classification
├── data.py                 # Built-in illustrative regional dataset
├── requirements.txt        # Runtime dependencies
└── README.md               # This file
```

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/S-ABDUL-AI/[REPO-NAME].git
cd [REPO-NAME]

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

**Requirements:** `streamlit` · `pandas` · `numpy` · `plotly`

---

## Deployment

Deployed on Streamlit Community Cloud.  
Live demo: [Community Vulnerability & Safety Net Targeting Tool](https://povertyearlywarningsystem-7rrmkktbi7bwha2nna8gk7.streamlit.app/)

**Rename the URL slug** (recommended):  
In Streamlit Cloud settings, change to: `community-vulnerability-safety-net-targeting`

---

## Why This Matters for SNAP and Safety Net Programs

The hardest problem in benefits delivery is reaching eligible people before their
situation becomes a crisis. Most targeting systems are reactive — built to process
applications, not to find the people who haven't applied yet.

This tool inverts that logic: it surfaces the communities where SNAP enrollment
pressure is likely to be highest *given current economic conditions* — food prices,
employment gaps, income strain, and housing cost burden — so outreach resources
can be deployed proactively.

The **Policy Insights** section is designed specifically to support program officers
who need to explain their targeting rationale to supervisors or grant funders:
each region gets a plain-language explanation of why it scored the way it did,
tied directly to the underlying indicator values.

---

## Scope Note

> All built-in data is **illustrative** for product design demonstration.  
> For live SNAP or food security program targeting, upload your own CSV with
> administrative data from state SNAP agencies or USDA FNS records.  
> Impact estimates ("could lower food-cost pressure by ~2%") are indicative
> scenario figures — not causal guarantees. Use as planning ranges only.

---

## About the Author

**Sherriff Abdul-Hamid** is a product leader and data scientist specializing in
government digital services, SNAP and safety net benefits delivery, and
decision-support tools for historically underserved communities.

- Former Founder & CEO, Poverty 360 — 25,000+ beneficiaries served across West Africa
- Partnered with Ghana's National Health Insurance Authority (NHIA) to enroll
  1,250 vulnerable women and abuse survivors into national health coverage
- Directed $200M+ in resource allocation for USAID, UNDP, and UKAID-funded programs
- **Obama Foundation Leaders Award** — Top 1.3% globally, 2023
- **Mandela Washington Fellow** — Top 0.3%, U.S. Department of State, 2018
- Harvard Business School · Senior Executive Program in General Management
- Healthcare Analytics Essentials — Northeastern University, 2024

**Connect:** [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) · [Portfolio](https://share.streamlit.io/user/s-abdul-ai)

---

## Related Projects

| Project | Description |
|---|---|
| [Medicaid & Healthcare Access Risk Monitor](https://chpghrwawmvddoquvmniwm.streamlit.app/) | State-level healthcare coverage risk scoring for all 50 US states — Medicaid program prioritization |
| [Public Budget Allocation Tool](https://smart-resource-allocation-dashboard-eudzw5r2f9pbu4qyw3psez.streamlit.app/) | Need-based government budget distribution with ministerial brief |
| [GovFund Allocation Engine](https://impact-allocation-engine-ahxxrbgwmvyapwmifahk2b.streamlit.app/) | Cost-effectiveness decision tool for public health funders |
| [Global Vaccination Coverage Explorer](https://worldvaccinationcoverage-etl-ftvwbikifyyx78xyy2j3zv.streamlit.app/) | WHO vaccination data pipeline across 190+ countries |
