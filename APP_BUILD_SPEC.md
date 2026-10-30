# App Build Spec: ANAROCK Absorption Forecaster (Prototype)

Owner: Nihal Shetty (Product). Status: ready to build. Target: a working demo deployed to a public URL.

---

## 1. Product context (why this exists)
- **Problem:** before a launch, ANAROCK's research and advisory team estimates "how fast will this project sell?" using experience and spreadsheets. The estimate drives pricing and unit-mix advice to developers, but it is slow, inconsistent across analysts, and hard to explain.
- **ANAROCK already sells this as a consulting service** (Market Feasibility & Demand Assessment). This prototype **productises** it: a consistent, explainable, instant forecast.
- **Primary user:** ANAROCK research/advisory analyst preparing a developer pitch.
- **Secondary user:** developer's strategy head (views output during the pitch).
- **Decision supported:** "At what price and unit mix should this project launch to sell X% in the first 12 months?"
- **Scope:** Pune only, 5 micro-markets (Hinjewadi, Wakad, Baner, Kharadi, Wagholi), residential, 1–4 BHK.

## 2. What already exists (DO NOT rebuild)
The model is already trained and validated. The app **loads it; it does not retrain it.**

| File | What it is |
|---|---|
| `data/synthetic_pune_absorption.csv` | 823 training rows, **synthetic**, calibrated to real Pune data |
| `data/real_projects_filled.csv` | 12 real Pune projects (IndexTap, read 29-Oct-2026); 6 marked USE |
| `data/README_synthetic_data.md` | Column dictionary + what's real vs assumed |
| `generate_synthetic.py` | Generates the synthetic data (seed 42) |
| `train.py` | Trains + evaluates; writes `outputs/` |
| `outputs/model.joblib` | dict: `model` (LightGBM regressor), `q80` (float, ± range in pts), `feats` (feature order), `cats` (category lists) |
| `outputs/metrics.csv`, `reality_check.csv`, `shap_*.png`, `MODEL_RESULTS_EXPLAINED.md` | Evaluation results |

### Model contract
- **Output:** `absorption_12m_pct`, the % of this unit type's inventory booked within 12 months of launch (0–100).
- **Range shown to user:** prediction ± `q80` (≈ ±11 pts), clipped to [0, 100]. Label it "likely range (~75–80% confidence)".
- **Feature order (must match exactly; read it from `feats` in the joblib):** `micro_market, configuration, carpet_area_sqft, price_vs_market_pct, total_units, developer_tier, months_to_possession, dist_metro_km, dist_it_hub_km, market_sales_trend_pct`
- The user enters **price per sq ft**; the app converts it to `price_vs_market_pct` (raw psf is NOT a model input).
- `micro_market` and `configuration` must be pandas `Categorical` with categories exactly = `cats[...]` from the joblib.
- **Derived input:** `price_vs_market_pct = 100 * (price_per_sqft / MARKET_BASE_PSF[micro_market] - 1)`
- **Monotonic guardrails are built in** (higher price premium / weaker tier / farther / later possession / larger inventory never raise the prediction; hotter market never lowers it).

### Reference constants
```python
MARKET_BASE_PSF = {"Hinjewadi": 9800, "Wakad": 9200, "Baner": 11200, "Kharadi": 11300, "Wagholi": 7200}
DEFAULT_DIST = {  # (metro_km, it_hub_km) midpoints
    "Hinjewadi": (2.2, 2.2), "Wakad": (3.0, 5.0), "Baner": (2.0, 6.0), "Kharadi": (4.0, 1.8), "Wagholi": (8.5, 7.5)}
CARPET_DEFAULT = {"1BHK": 480, "2BHK": 750, "3BHK": 1100, "4BHK": 1600}
CARPET_RANGE   = {"1BHK": (400, 560), "2BHK": (640, 860), "3BHK": (950, 1250), "4BHK": (1400, 1800)}
DEFAULT_MARKET_TREND = -9   # trailing-4Q Pune sales growth %, latest quarter in data (cooling market)
TIER_LABELS = {1: "Tier 1 – national listed brand", 2: "Tier 2 – established regional", 3: "Tier 3 – small / new"}
```
Valid input ranges (training range; warn if outside): price_vs_market −20% to +25%, total_units 20–420, months_to_possession 24–60, market trend −17 to +30.

## 3. Screens and features (single-page Streamlit app)

### 3.1 Header
- Title: **"Absorption Forecaster: Pune (Prototype)"**
- Subtitle: "How fast will this launch sell? Forecast, reasons, and price/mix what-ifs."
- Always-visible banner: "⚠️ Prototype trained on **synthetic data** calibrated to real Pune projects. Not for real pricing decisions."

### 3.2 Sidebar: project inputs
- Micro-market (select, 5 options)
- Configuration (select: 1BHK/2BHK/3BHK/4BHK)
- Carpet area sq ft (slider, range depends on configuration, default per CARPET_DEFAULT)
- Price per sq ft (₹) (number/slider; show helper text: "Market average: ₹X → you are +Y% / −Y%")
- Developer tier (select with TIER_LABELS)
- Units of this type (slider 20–420, default 150)
- Months to possession (slider 24–60, default 42)
- Expander "Advanced": distance to metro km, distance to IT hub km (defaults from DEFAULT_DIST, change with the market), market sales trend % (default −9, help text: "Pune sales growth over the last 4 quarters; negative = cooling")

### 3.3 Main area, top: forecast card
- Big number: **"41% sold in 12 months"**
- Sub-line: "Likely range 30%–53% · ≈ 62 of 150 units"
- Colour band: <20% red "slow", 20–40% amber "moderate", >40% green "fast"

### 3.4 "Why this forecast" (explainability)
- SHAP waterfall or horizontal bar chart for **this single prediction** (use `shap.TreeExplainer(model)`), showing top 6 drivers with +/− points.
- Below it, 3 plain-English bullets auto-generated from the top SHAP values, e.g. "Priced 12% above the Hinjewadi average → −4.1 pts".
- Map raw feature names to friendly labels (e.g. `price_vs_market_pct` → "Price vs local market").

### 3.5 Tab: Price what-if
- Line chart: predicted absorption (y) vs price per sq ft (x) from −20% to +25% of market base, all else fixed. Shade the ± range. Mark the current price.
- **Target finder:** input "Target % sold in 12 months" (default 35). Output: "Highest price that still hits 35%: ₹X/sq ft (+Y% vs market)". If not achievable in range, say so.

### 3.6 Tab: Unit-mix comparison
- Table + bar chart: the same project settings run for all 4 configurations (each with its default carpet area; keep the same price_vs_market_pct). Columns: config, predicted %, range, units sold if 150 units.
- One-line takeaway: "2BHK is expected to sell fastest here (+X pts vs 3BHK)."

### 3.7 Tab: Model card (trust and transparency)
- Show `outputs/metrics.csv` as a readable table: baseline vs model error, within ±10 pts, bias, range coverage.
- Show `shap_importance.png`.
- Data: synthetic, 823 rows, calibration anchors (link README).
- Known limitations (bullets): synthetic training data; cannot extrapolate beyond the training market range (drift → quarterly retraining); price baseline is micro-market average (fringe locations look "cheap"); range slightly optimistic (~73% vs 80% target).

### 3.8 Tab: Reality check
- Show `outputs/reality_check.csv` (project, prediction, plausible band, verdict) with a caption explaining the floor/ceiling logic and that PASS = plausible, not proof.

### 3.9 (Optional, nice-to-have) "Explain like I'm the developer" button
- Calls the Gemini API with the inputs, prediction, range and top SHAP drivers. Returns a 4-sentence plain-English summary + one pricing suggestion.
- System instruction for that call: "You explain a real-estate sales forecast to a developer. Use only the numbers given. Do not invent data. Mention the range and that the model is a prototype trained on synthetic data."
- API key from env var `GEMINI_API_KEY`. **If the key is missing, hide the button. The app must fully work without it.**

## 4. Non-functional requirements
- Python 3.11. `requirements.txt` with pinned versions that match the saved model: `lightgbm==4.7.0`, `scikit-learn==1.9.1`, `shap==0.54.1`, `pandas`, `numpy`, `joblib`, `matplotlib`, `streamlit`, `plotly` (optional), `google-genai` (optional).
- Cache model loading with `@st.cache_resource`.
- Each interaction responds in under 2 seconds.
- Mobile-friendly layout (Streamlit default is fine).
- No scraping, no external data calls except the optional Gemini call.
- Clean code: `app.py` + `forecaster.py` (load model, build feature row, predict, explain, what-if) + `tests/test_forecaster.py`.

## 5. Acceptance tests (must pass before deploy)
1. App starts with `streamlit run app.py` with no errors.
2. Default inputs produce a prediction between 0 and 100 with a range.
3. Raising price (all else equal) never increases the prediction (monotonic check over 10 price steps).
4. Changing tier from 1 → 3 never increases the prediction.
5. Unit-mix tab shows 4 rows.
6. Target finder returns a price, or "not achievable."
7. Synthetic-data banner is visible on every view.
8. App works with `GEMINI_API_KEY` unset.
9. Sanity: Hinjewadi, 2BHK, 750 sq ft, ₹10,000/sq ft, Tier 1, 150 units, 42 months, defaults → prediction roughly 35–50% (reference build gives ~41.5%).

## 6. Deployment
- Push to a public GitHub repo `anarock-absorption-forecaster` (include data, scripts, outputs, README).
- Deploy on **Streamlit Community Cloud** (free). Gemini key, if used, goes in Streamlit secrets, never in code.
- README: problem, users, how it works (1 diagram), results table, limitations, "how this plugs into ANAROCK data," screenshots.

## 7. Out of scope
User accounts, databases, other cities, live data feeds, retraining from the UI.
