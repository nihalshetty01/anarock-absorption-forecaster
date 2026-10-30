"""
Synthetic Pune absorption dataset for the ANAROCK Absorption Forecaster prototype.
ALL ROWS ARE SYNTHETIC. One row = one unit type (configuration) within one project.
Target: absorption_12m_pct = % of that unit type's inventory booked within 12 months of launch.
Inputs are only things known on launch day (no leakage).
Calibration anchors are documented in CALIBRATION below and checked in validate().
"""
import numpy as np, pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# --- Micro-market anchors (price psf from 6 real IndexTap projects, read 29-Oct-2026) ---
MARKETS = {
    #            base_psf  demand  metro_km(lo,hi)  it_hub_km(lo,hi)
    "Hinjewadi": (9800,   0.30,   (0.5, 4.0),      (0.5, 4.0)),   # real: 8.8K-11.0K
    "Wakad":     (9200,   0.20,   (1.0, 5.0),      (3.0, 7.0)),   # real: 8.9K
    "Baner":     (11200,  0.10,   (0.5, 3.5),      (4.0, 8.0)),   # real: 10.9K-11.2K
    "Kharadi":   (11300,  0.30,   (2.0, 6.0),      (0.5, 3.0)),   # real: 11.3K (EON IT Park)
    "Wagholi":   (7200,   0.45,   (5.0, 12.0),     (5.0, 10.0)),  # real: 6.5K-9.3K
}
CONFIGS = {  # carpet sqft range, demand effect, share probability
    "1BHK": ((400, 560),   0.10, 0.20),
    "2BHK": ((640, 860),   0.35, 0.45),
    "3BHK": ((950, 1250),  0.00, 0.28),
    "4BHK": ((1400, 1800), -0.45, 0.07),
}
QUARTERS = [f"{y}Q{q}" for y in range(2022, 2026) for q in range(1, 5)][:14]  # 2022Q1..2025Q2
# Pune market cycle at launch: trailing-4Q sales growth (illustrative shape: 2022 boom, 2024-25 cooling)
CYCLE = dict(zip(QUARTERS, [18, 22, 20, 15, 10, 8, 6, 4, 2, 0, -3, -6, -8, -9]))
TIER = {1: 0.35, 2: 0.0, 3: -0.35}

def make_projects(n_projects=370):
    rows = []
    for p in range(n_projects):
        mkt = rng.choice(list(MARKETS))
        base_psf, m_demand, metro_r, hub_r = MARKETS[mkt]
        q = rng.choice(QUARTERS)
        tier = rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30])
        possession = int(rng.integers(24, 61))
        metro = round(rng.uniform(*metro_r), 1)
        hub = round(rng.uniform(*hub_r), 1)
        # project pricing stance vs market (tier-1 brands price higher)
        stance = rng.normal(0.04 * (2 - tier), 0.07)
        n_cfg = rng.choice([1, 2, 3], p=[0.15, 0.45, 0.40])
        cfg_names = list(CONFIGS)
        probs = np.array([CONFIGS[c][2] for c in cfg_names])
        cfgs = rng.choice(cfg_names, size=n_cfg, replace=False, p=probs / probs.sum())
        proj_noise = rng.normal(0, 0.35)  # unobserved project quality (amenities, launch marketing)
        for c in cfgs:
            (alo, ahi), c_demand, _ = CONFIGS[c]
            area = int(rng.uniform(alo, ahi))
            price_vs_mkt = stance + rng.normal(0, 0.03)
            psf = int(base_psf * (1 + price_vs_mkt) * (1 + 0.02 * (c in ("3BHK", "4BHK"))))
            units = int(rng.integers(40, 420) * (0.5 if c == "4BHK" else 1))
            trend = CYCLE[q] + rng.normal(0, 3)
            # ---- latent score -> absorption (the "true" effects the model should recover) ----
            score = (-0.75                       # intercept tuned so mean ~32%
                     + m_demand + c_demand + TIER[tier]
                     - 3.0 * price_vs_mkt        # 10% above market -> -0.30
                     + 0.03 * trend              # hot market at launch helps
                     - 0.06 * metro - 0.05 * hub
                     - 0.012 * (possession - 36) # far possession hurts
                     - 0.0012 * (units - 200)    # big inventory dilutes %
                     + proj_noise + rng.normal(0, 0.25))
            absorption = 100 / (1 + np.exp(-score))
            rows.append(dict(
                project_id=f"SYN-P{p:03d}", micro_market=mkt, configuration=c,
                carpet_area_sqft=area, price_per_sqft=psf,
                ticket_size_lakh=round(area * psf / 1e5, 1),
                price_vs_market_pct=round(100 * price_vs_mkt, 1),
                total_units=units, developer_tier=tier, launch_quarter=q,
                months_to_possession=possession, dist_metro_km=metro, dist_it_hub_km=hub,
                market_sales_trend_pct=round(trend, 1),
                absorption_12m_pct=round(float(np.clip(absorption, 1, 98)), 1),
                is_synthetic=True))
    return pd.DataFrame(rows)

def validate(df, real):
    print(f"rows={len(df)} projects={df.project_id.nunique()}")
    print("\nTarget distribution (absorption_12m_pct):")
    print(df.absorption_12m_pct.describe().round(1).to_string())
    # project-level (unit-weighted) absorption
    df = df.assign(sold12=df.total_units * df.absorption_12m_pct / 100)
    proj = df.groupby(["project_id", "micro_market"]).agg(units=("total_units", "sum"), sold=("sold12", "sum")).reset_index()
    proj["abs12"] = 100 * proj.sold / proj.units
    print("\nProject-level 12m absorption by market (median %):")
    print(proj.groupby("micro_market").abs12.median().round(1).to_string())
    print("\nBy configuration (median %):")
    print(df.groupby("configuration").absorption_12m_pct.median().round(1).to_string())
    print("\nBy developer tier (median %):")
    print(df.groupby("developer_tier").absorption_12m_pct.median().round(1).to_string())
    # Reality check: real projects' average pace over their life; year-1 pace is typically higher (front-loaded)
    r = real[real.use_for_validation.str.startswith("USE")]
    print("\nReal projects (IndexTap) lifetime pace, %/month:", r.pct_sold_per_month.round(2).tolist())
    print("Real lifetime pace mean: %.2f %%/month -> naive 12m equivalent %.0f%%" % (r.pct_sold_per_month.mean(), 12 * r.pct_sold_per_month.mean()))
    print("Synthetic project-level 12m absorption median: %.1f%% (IQR %.1f-%.1f)" %
          (proj.abs12.median(), proj.abs12.quantile(.25), proj.abs12.quantile(.75)))
    return proj

if __name__ == "__main__":
    df = make_projects()
    df.to_csv("data/synthetic_pune_absorption.csv", index=False)
    real = pd.read_csv("data/real_projects_filled.csv")
    validate(df, real)
