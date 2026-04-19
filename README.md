# Poverty Early Warning System (PEWS)

A **Streamlit** decision-support app for exploring **regional poverty risk** in tabular data. It trains a **Random Forest** classifier, shows **low / medium / high** risk bands, charts, and a **policy layer** with plain-language explanations and suggested actions—not official statistics.

**Designed by Sherriff Abdul-Hamid.**

---

## What it does

- Loads **sample data** automatically, or your own **CSV**
- Trains a **Random Forest** model on income, employment, food prices, inflation, and region
- **Scores** each row into risk bands and suggests **policy-style actions**
- **Summary dashboard**: counts of high-stress regions and illustrative population totals
- **Policy brief**: auto-generated paragraph for notes or briefings
- **Policy insights**: rule-based “why” text (e.g. food costs, job scarcity) per region
- **Charts**: risk by region (stacked bars), feature importance (Plotly)

---

## Important disclaimer

Default numbers are **synthetic / illustrative**. Unless you upload real administrative data, outputs are **for demonstration and discussion only**, not for binding policy or official poverty measurement. Always combine with local knowledge and appropriate methods for your jurisdiction.

---

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI and session flow |
| `data.py` | CSV validation, synthetic data, optional `population` column handling |
| `model.py` | sklearn pipeline (scale + one-hot region + Random Forest), predictions |
| `policy.py` | Short recommended actions per risk band |
| `insights.py` | Population fill-in, regional “why” text, brief, dashboard stats |
| `viz.py` | Plotly charts |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Theme (optional) |

---

## Requirements

- Python **3.10+** (3.11 or 3.12 recommended)
- Dependencies listed in `requirements.txt`

---

## Run locally

```bash
cd poverty_early_warning_system
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), **New app** → pick the repo.
3. Set **Main file path** to `app.py`.
4. Python dependencies are read from `requirements.txt`.
5. Deploy; use **Reboot app** after you push code changes.

---

## CSV format

**Required columns** (exact names):

| Column | Description |
|--------|-------------|
| `region` | Region or area label (text) |
| `income` | Typical or aggregate income (numeric) |
| `employment_rate` | Share in work, e.g. percent (numeric) |
| `food_price_index` | Food price level index (numeric) |
| `inflation` | Inflation rate (numeric) |
| `poverty_label` | Risk / poverty label: **0**, **1**, or **0 / 1 / 2** (integer). `0` = low, `1` = medium, `2` = high when three classes are used; binary `0/1` is supported and mapped into three display bands from predicted probabilities. |

**Optional:**

| Column | Description |
|--------|-------------|
| `population` | Population count **per row** (e.g. sub-area). If omitted, the app fills **illustrative** values so dashboard totals have a scale. |

---

## Using the app (quick)

1. Open the app; sample data loads by default.
2. Optionally **upload a CSV** (sidebar).
3. Restrict **Regions to show** if you want a subset.
4. Scores build **automatically** on first load; use **Refresh risk scores** after changing the file.
5. Read the **Summary dashboard**, **Policy brief**, **Policy insights**, and charts.

Expand **How to use this page** at the top of the app for a fuller walkthrough.

---

## Licence

No licence is specified in this repository. Add a `LICENSE` file if you intend to open-source or redistribute the code.
