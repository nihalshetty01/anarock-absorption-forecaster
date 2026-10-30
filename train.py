"""
Train + evaluate the Absorption Forecaster.
Step 1  Time-based split: train on launches 2022Q1-2024Q2, test on 2024Q3-2025Q2 ("predict future launches").
Step 2  Baseline: median absorption of same micro-market x configuration in training data.
Step 3  LightGBM point model + 10th/90th percentile models (prediction range).
Step 4  SHAP: which inputs drive predictions; do they match the effects planted in the data?
Step 5  Reality check against real Pune projects (IndexTap).
"""
import json, numpy as np, pandas as pd, lightgbm as lgb, shap, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "data/"; OUT = "outputs/"
import os; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(D + "synthetic_pune_absorption.csv")

CAT = ["micro_market", "configuration"]
NUM = ["carpet_area_sqft", "price_vs_market_pct", "total_units", "developer_tier",
       "months_to_possession", "dist_metro_km", "dist_it_hub_km", "market_sales_trend_pct"]
FEATS = CAT + NUM
TARGET = "absorption_12m_pct"
for c in CAT: df[c] = df[c].astype("category")

# ---- Step 1: two evaluation splits ----
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
# Business-logic guardrails: higher price premium / worse tier / farther / later possession / bigger inventory
# can never INCREASE predicted absorption; hotter market can never decrease it.
MONO = {"price_vs_market_pct": -1, "developer_tier": -1, "dist_metro_km": -1, "dist_it_hub_km": -1,
        "months_to_possession": -1, "total_units": -1, "market_sales_trend_pct": 1}
params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42,
              monotone_constraints=[MONO.get(f, 0) for f in FEATS], monotone_constraints_method="advanced")

def report(name, split, y, p):
    err = np.abs(y - p)
    r = dict(split=split, model=name, MAE_pts=round(float(err.mean()), 2),
             within_10pts_pct=round(float(100 * (err <= 10).mean()), 1), bias_pts=round(float((p - y).mean()), 2))
    print(r); return r

def baseline(tr, te):
    lookup = tr.groupby(CAT, observed=True)[TARGET].median()
    return te.apply(lambda r: lookup.get((r.micro_market, r.configuration), tr[TARGET].median()), axis=1).values

res = []
# (A) random split by PROJECT (all unit types of a project stay together -> no leakage across siblings)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
i_tr, i_te = next(gss.split(df, groups=df.project_id))
tr, te = df.iloc[i_tr], df.iloc[i_te]
print(f"[A] project split: train={len(tr)} test={len(te)}")
res.append(report("Baseline (market x config median)", "A: random by project", te[TARGET].values, baseline(tr, te)))
model = lgb.LGBMRegressor(**params).fit(tr[FEATS], tr[TARGET])
res.append(report("LightGBM", "A: random by project", te[TARGET].values, model.predict(te[FEATS])))

# (B) time split: train on 2022Q1-2024Q2 launches, test on 2024Q3-2025Q2 (market cooled beyond training range)
test_q = ["2024Q3", "2024Q4", "2025Q1", "2025Q2"]
trB, teB = df[~df.launch_quarter.isin(test_q)], df[df.launch_quarter.isin(test_q)]
print(f"[B] time split: train={len(trB)} test={len(teB)}  trend range train {trB.market_sales_trend_pct.min():.0f}..{trB.market_sales_trend_pct.max():.0f}, test {teB.market_sales_trend_pct.min():.0f}..{teB.market_sales_trend_pct.max():.0f}")
res.append(report("Baseline (market x config median)", "B: future launches (drift)", teB[TARGET].values, baseline(trB, teB)))
mB = lgb.LGBMRegressor(**params).fit(trB[FEATS], trB[TARGET])
res.append(report("LightGBM", "B: future launches (drift)", teB[TARGET].values, mB.predict(teB[FEATS])))

# ---- Step 3: calibrated 80% range (split-conformal on out-of-fold residuals) ----
oof = np.zeros(len(tr))
for a, b in GroupKFold(n_splits=5).split(tr, groups=tr.project_id):
    oof[b] = lgb.LGBMRegressor(**params).fit(tr.iloc[a][FEATS], tr.iloc[a][TARGET]).predict(tr.iloc[b][FEATS])
q80 = float(np.quantile(np.abs(tr[TARGET].values - oof), 0.80))
p = model.predict(te[FEATS])
cov = 100 * (np.abs(te[TARGET].values - p) <= q80).mean()
print(f"80% range = prediction +/- {q80:.1f} pts; actual inside range on test A: {cov:.1f}%")
res.append(dict(split="A: random by project", model="LightGBM 80% range", MAE_pts=None,
                within_10pts_pct=None, bias_pts=None, range_halfwidth_pts=round(q80, 1), coverage_pct=round(cov, 1)))
# ---- Step 4: SHAP ----
expl = shap.TreeExplainer(model)
sv = expl.shap_values(te[FEATS])
imp = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values()
plt.figure(figsize=(7, 4.5)); imp.plot.barh(color="#3b6ea5")
plt.title("What drives 12-month absorption (mean |SHAP|, % points)"); plt.xlabel("Avg impact on prediction (pts)")
plt.tight_layout(); plt.savefig(OUT + "shap_importance.png", dpi=150); plt.close()
shap.summary_plot(sv, te[FEATS], show=False, plot_size=(8, 5))
plt.title("Direction of each driver (red = high value)"); plt.tight_layout()
plt.savefig(OUT + "shap_beeswarm.png", dpi=150); plt.close()
print("\nSHAP importance (pts):\n", imp.sort_values(ascending=False).round(2).to_string())

# direction checks vs planted effects
def corr(f): return np.corrcoef(te[f], sv[:, FEATS.index(f)])[0, 1]
checks = {f: round(corr(f), 2) for f in ["price_vs_market_pct", "developer_tier", "dist_metro_km",
                                         "dist_it_hub_km", "months_to_possession", "total_units", "market_sales_trend_pct"]}
expected = {"price_vs_market_pct": "-", "developer_tier": "-", "dist_metro_km": "-", "dist_it_hub_km": "-",
            "months_to_possession": "-", "total_units": "-", "market_sales_trend_pct": "+"}
print("\nDirection check (corr of feature value with its SHAP; expected sign):")
for f, c in checks.items(): print(f"  {f:25s} {c:+.2f}  expected {expected[f]}  {'OK' if (c < 0) == (expected[f] == '-') else 'MISMATCH'}")
cfg_shap = pd.Series(sv[:, FEATS.index("configuration")], index=te.index).groupby(te.configuration, observed=True).mean().round(1)
print("Avg SHAP by configuration:", cfg_shap.to_dict())

# ---- Step 5: reality check vs real projects ----
real = pd.read_csv(D + "real_projects_filled.csv")
real = real[real.use_for_validation.str.startswith("USE")].copy()
BASE = {"Hinjewadi": 9800, "Wakad": 9200, "Baner": 11200, "Kharadi": 11300, "Wagholi": 7200, "Balewadi": 11200}
MID = {"Hinjewadi": (2.2, 2.2), "Wakad": (3.0, 5.0), "Baner": (2.0, 6.0), "Kharadi": (4.0, 1.8), "Wagholi": (8.5, 7.5), "Balewadi": (2.0, 6.0)}
TIER = {"Godrej Properties": 1, "Kolte-Patil": 1, "Vilas Javdekar": 2, "Kumar Properties": 2, "Ashiana Housing": 2, "Mantra Properties": 3}
CYCLE = dict(zip([f"{y}Q{q}" for y in range(2020, 2026) for q in range(1, 5)],
                 [0]*8 + [18, 22, 20, 15, 10, 8, 6, 4, 2, 0, -3, -6, -8, -9]))
psf_fill = {"Godrej Park World (main, user-entered)": 11.0}  # main GPW page had no psf; use Gale's
rows = []
for _, r in real.iterrows():
    mkt = r.micro_market if r.micro_market in BASE else "Baner"
    psf = (r.price_psf_k if pd.notna(r.price_psf_k) else psf_fill.get(r.project_name, np.nan)) * 1000
    ld, pdt = pd.Timestamp(r.launch_date), (pd.Timestamp(r.possession_date) if pd.notna(r.possession_date) else pd.Timestamp("2029-03-31"))
    q = f"{ld.year}Q{(ld.month - 1)//3 + 1}"
    rows.append(dict(project=r.project_name, micro_market="Baner" if r.micro_market == "Balewadi" else r.micro_market,
                     configuration="2BHK", carpet_area_sqft=750, price_per_sqft=psf,
                     price_vs_market_pct=100 * (psf / BASE[r.micro_market] - 1),
                     total_units=min(r.total_units / 2.5, 420), developer_tier=TIER[r.developer],
                     months_to_possession=min(60, round((pdt - ld).days / 30.44)),
                     dist_metro_km=MID[r.micro_market][0], dist_it_hub_km=MID[r.micro_market][1],
                     market_sales_trend_pct=CYCLE.get(q, 0),
                     real_pct_sold_to_date=r.pct_sold, real_pace_x12=round(12 * r.pct_sold_per_month, 1)))
rv = pd.DataFrame(rows)
X = rv[FEATS].copy()
for c in CAT: X[c] = pd.Categorical(X[c], categories=df[c].cat.categories)
final = lgb.LGBMRegressor(**params).fit(df[FEATS], df[TARGET])
rv["pred_12m"] = final.predict(X).round(1)
rv["pred_lo"], rv["pred_hi"] = (rv.pred_12m - q80).clip(0).round(1), (rv.pred_12m + q80).clip(upper=100).round(1)
# plausible band: 12m absorption should be >= lifetime-average pace x 12 (front-loaded) and <= % sold to date
rv["plausible_band"] = rv.apply(lambda r: f"{r.real_pace_x12:.0f}-{r.real_pct_sold_to_date:.0f}%", axis=1)
rv["verdict"] = rv.apply(lambda r: "PASS" if r.real_pace_x12 <= r.pred_12m <= r.real_pct_sold_to_date
                         else ("LOW" if r.pred_12m < r.real_pace_x12 else "HIGH"), axis=1)
cols = ["project", "micro_market", "developer_tier", "price_vs_market_pct", "pred_12m", "pred_lo", "pred_hi", "plausible_band", "verdict"]
rv["price_vs_market_pct"] = rv.price_vs_market_pct.round(1)
print("\nReality check vs real projects:\n", rv[cols].to_string(index=False))
rv.to_csv(OUT + "reality_check.csv", index=False)

pd.DataFrame(res).to_csv(OUT + "metrics.csv", index=False)
joblib.dump(dict(model=final, q80=q80, feats=FEATS, cats={c: list(df[c].cat.categories) for c in CAT}), OUT + "model.joblib")
print("\nSaved outputs/")
