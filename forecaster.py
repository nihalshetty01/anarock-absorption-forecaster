"""
Absorption Forecaster core module.

Handles model loading, feature row preparation, prediction with uncertainty bands,
SHAP explainability, price sensitivity curves, target pricing search, and unit-mix analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

# Reference constants per APP_BUILD_SPEC.md
MARKET_BASE_PSF = {
    "Hinjewadi": 9800,
    "Wakad": 9200,
    "Baner": 11200,
    "Kharadi": 11300,
    "Wagholi": 7200,
}

DEFAULT_DIST = {  # (dist_metro_km, dist_it_hub_km) midpoints
    "Hinjewadi": (2.2, 2.2),
    "Wakad": (3.0, 5.0),
    "Baner": (2.0, 6.0),
    "Kharadi": (4.0, 1.8),
    "Wagholi": (8.5, 7.5),
}

CARPET_DEFAULT = {"1BHK": 480, "2BHK": 750, "3BHK": 1100, "4BHK": 1600}
CARPET_RANGE = {"1BHK": (400, 560), "2BHK": (640, 860), "3BHK": (950, 1250), "4BHK": (1400, 1800)}
DEFAULT_MARKET_TREND = -9  # trailing-4Q Pune sales growth %, latest quarter in data
TIER_LABELS = {
    1: "Tier 1 – national listed brand",
    2: "Tier 2 – established regional",
    3: "Tier 3 – small / new",
}

FEATURE_FRIENDLY_NAMES = {
    "micro_market": "Micro-market",
    "configuration": "Configuration (BHK)",
    "carpet_area_sqft": "Carpet area (unit size)",
    "price_vs_market_pct": "Price vs local market",
    "total_units": "Total units (inventory scale)",
    "developer_tier": "Developer brand / tier",
    "months_to_possession": "Months to possession",
    "dist_metro_km": "Distance to metro",
    "dist_it_hub_km": "Distance to IT hub",
    "market_sales_trend_pct": "Pune market sales trend",
}

# Global in-memory cache for the loaded model bundle and SHAP explainer
_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_SHAP_EXPLAINER: Optional[shap.TreeExplainer] = None


def load_model(model_path: str = "outputs/model.joblib") -> Dict[str, Any]:
    """Load model artifact bundle: {'model', 'q80', 'feats', 'cats'}."""
    global _MODEL_BUNDLE, _SHAP_EXPLAINER
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = joblib.load(model_path)
        _SHAP_EXPLAINER = shap.TreeExplainer(_MODEL_BUNDLE["model"])
    return _MODEL_BUNDLE


def get_explainer() -> shap.TreeExplainer:
    """Return the cached SHAP TreeExplainer instance."""
    global _SHAP_EXPLAINER
    if _SHAP_EXPLAINER is None:
        bundle = load_model()
        _SHAP_EXPLAINER = shap.TreeExplainer(bundle["model"])
    return _SHAP_EXPLAINER


def build_row(inputs: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Construct a single-row DataFrame matching the exact feature order and categorical
    types required by the saved model.
    """
    if bundle is None:
        bundle = load_model()

    feats = bundle["feats"]
    cats = bundle["cats"]

    data: Dict[str, Any] = {}

    micro_market = inputs["micro_market"]
    if micro_market not in cats["micro_market"]:
        raise ValueError(f"Unknown micro_market '{micro_market}'. Must be one of {cats['micro_market']}")

    configuration = inputs["configuration"]
    if configuration not in cats["configuration"]:
        raise ValueError(f"Unknown configuration '{configuration}'. Must be one of {cats['configuration']}")

    # Calculate price_vs_market_pct if not explicitly supplied
    if "price_vs_market_pct" in inputs:
        price_vs_market_pct = float(inputs["price_vs_market_pct"])
    elif "price_per_sqft" in inputs:
        base_psf = MARKET_BASE_PSF[micro_market]
        price_vs_market_pct = 100.0 * (float(inputs["price_per_sqft"]) / base_psf - 1.0)
    else:
        raise ValueError("Either 'price_per_sqft' or 'price_vs_market_pct' must be provided in inputs.")

    # Location defaults fallback
    default_metro, default_it = DEFAULT_DIST[micro_market]
    dist_metro_km = float(inputs.get("dist_metro_km", default_metro))
    dist_it_hub_km = float(inputs.get("dist_it_hub_km", default_it))

    data["micro_market"] = pd.Categorical([micro_market], categories=cats["micro_market"])
    data["configuration"] = pd.Categorical([configuration], categories=cats["configuration"])
    data["carpet_area_sqft"] = [float(inputs.get("carpet_area_sqft", CARPET_DEFAULT[configuration]))]
    data["price_vs_market_pct"] = [price_vs_market_pct]
    data["total_units"] = [int(inputs.get("total_units", 150))]
    data["developer_tier"] = [int(inputs.get("developer_tier", 1))]
    data["months_to_possession"] = [int(inputs.get("months_to_possession", 42))]
    data["dist_metro_km"] = [dist_metro_km]
    data["dist_it_hub_km"] = [dist_it_hub_km]
    data["market_sales_trend_pct"] = [float(inputs.get("market_sales_trend_pct", DEFAULT_MARKET_TREND))]

    row = pd.DataFrame(data)[feats]
    return row


def predict(inputs: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float]:
    """
    Predict 12-month absorption % and return (point_estimate, low_bound, high_bound).
    Bounds are calculated using q80 (± range) and clipped to [0, 100].
    """
    if bundle is None:
        bundle = load_model()

    row = build_row(inputs, bundle)
    pred_raw = float(bundle["model"].predict(row)[0])
    q80 = float(bundle["q80"])

    point = float(np.clip(pred_raw, 0.0, 100.0))
    low = float(np.clip(point - q80, 0.0, 100.0))
    high = float(np.clip(point + q80, 0.0, 100.0))

    return round(point, 1), round(low, 1), round(high, 1)


def explain(inputs: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Explain a single prediction using SHAP TreeExplainer.
    Returns:
      - base_value: expected model value across training
      - prediction: point prediction
      - drivers: list of features sorted by magnitude with SHAP impact in points
      - bullets: 3 auto-generated plain-English takeaway bullets
    """
    if bundle is None:
        bundle = load_model()

    explainer = get_explainer()
    row = build_row(inputs, bundle)
    shap_vals = explainer(row)

    # shap_vals[0].values is array of SHAP values for each feature in feats
    values = shap_vals[0].values
    base_value = float(shap_vals[0].base_values)
    pred_val = float(np.clip(bundle["model"].predict(row)[0], 0.0, 100.0))

    drivers = []
    for feat_name, val, raw_val in zip(bundle["feats"], values, row.iloc[0]):
        friendly_label = FEATURE_FRIENDLY_NAMES.get(feat_name, feat_name)
        val_pts = float(val)

        # Format raw display value
        if feat_name == "price_vs_market_pct":
            display_val = f"{raw_val:+.1f}% vs market"
        elif feat_name == "market_sales_trend_pct":
            display_val = f"{raw_val:+.1f}%"
        elif feat_name == "developer_tier":
            display_val = f"Tier {int(raw_val)}"
        elif feat_name == "carpet_area_sqft":
            display_val = f"{int(raw_val)} sq ft"
        elif feat_name in ("dist_metro_km", "dist_it_hub_km"):
            display_val = f"{raw_val:.1f} km"
        elif feat_name == "months_to_possession":
            display_val = f"{int(raw_val)} mos"
        elif feat_name == "total_units":
            display_val = f"{int(raw_val)} units"
        else:
            display_val = str(raw_val)

        drivers.append({
            "feature": feat_name,
            "friendly_label": friendly_label,
            "shap_value": round(val_pts, 2),
            "abs_impact": abs(val_pts),
            "display_val": display_val,
            "impact_direction": "+" if val_pts >= 0 else "−",
        })

    # Sort drivers by absolute impact descending
    drivers.sort(key=lambda d: d["abs_impact"], reverse=True)

    # Generate 3 plain-English bullets from top drivers
    bullets = []
    micro_mkt = str(row.iloc[0]["micro_market"])
    for d in drivers[:3]:
        f = d["feature"]
        pts = d["shap_value"]
        pts_str = f"{pts:+.1f} pts"

        if f == "price_vs_market_pct":
            pct = row.iloc[0]["price_vs_market_pct"]
            if pct > 0:
                bullets.append(f"Priced {pct:.1f}% above the {micro_mkt} benchmark → {pts_str}")
            elif pct < 0:
                bullets.append(f"Priced {abs(pct):.1f}% below the {micro_mkt} benchmark (competitive discount) → {pts_str}")
            else:
                bullets.append(f"Priced exactly at {micro_mkt} market average → {pts_str}")
        elif f == "developer_tier":
            tier = int(row.iloc[0]["developer_tier"])
            label = TIER_LABELS.get(tier, f"Tier {tier}").split("–")[1].strip() if "–" in TIER_LABELS.get(tier, "") else f"Tier {tier}"
            bullets.append(f"{label} developer reputation → {pts_str}")
        elif f == "market_sales_trend_pct":
            trend = row.iloc[0]["market_sales_trend_pct"]
            status = "cooling" if trend < 0 else "hot"
            bullets.append(f"Pune citywide sales trend ({trend:+.1f}%, {status} cycle) → {pts_str}")
        elif f == "dist_it_hub_km":
            dist = row.iloc[0]["dist_it_hub_km"]
            bullets.append(f"{dist:.1f} km distance to major IT hub → {pts_str}")
        elif f == "carpet_area_sqft":
            area = int(row.iloc[0]["carpet_area_sqft"])
            bullets.append(f"Unit size ({area} sq ft) vs buyer segment sweet spot → {pts_str}")
        elif f == "total_units":
            units = int(row.iloc[0]["total_units"])
            bullets.append(f"Launch inventory scale ({units} units) → {pts_str}")
        elif f == "configuration":
            cfg = str(row.iloc[0]["configuration"])
            bullets.append(f"{cfg} configuration buyer appeal → {pts_str}")
        elif f == "months_to_possession":
            mos = int(row.iloc[0]["months_to_possession"])
            bullets.append(f"{mos} months construction timeline to possession → {pts_str}")
        elif f == "dist_metro_km":
            dist = row.iloc[0]["dist_metro_km"]
            bullets.append(f"{dist:.1f} km distance to metro station → {pts_str}")
        else:
            bullets.append(f"{d['friendly_label']} ({d['display_val']}) → {pts_str}")

    return {
        "base_value": round(base_value, 1),
        "prediction": round(pred_val, 1),
        "drivers": drivers,
        "bullets": bullets,
    }


def price_curve(
    inputs: Dict[str, Any],
    bundle: Optional[Dict[str, Any]] = None,
    steps: int = 19,
) -> pd.DataFrame:
    """
    Generate sensitivity curve data: predicted absorption vs price per sq ft
    from -20% to +25% of micro-market base price, holding all other inputs fixed.
    """
    if bundle is None:
        bundle = load_model()

    micro_market = inputs["micro_market"]
    base_psf = MARKET_BASE_PSF[micro_market]

    # -20% to +25%
    pct_range = np.linspace(-20.0, 25.0, steps)
    records = []

    # Copy inputs to avoid mutating caller's dict
    test_inputs = dict(inputs)

    for pct in pct_range:
        psf = round(base_psf * (1.0 + pct / 100.0))
        test_inputs["price_vs_market_pct"] = pct
        test_inputs.pop("price_per_sqft", None)

        pt, low, high = predict(test_inputs, bundle)
        records.append({
            "price_per_sqft": psf,
            "price_vs_market_pct": round(pct, 1),
            "predicted_pct": pt,
            "low_pct": low,
            "high_pct": high,
        })

    return pd.DataFrame(records)


def max_price_for_target(
    inputs: Dict[str, Any],
    target_pct: float,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Find the highest price per sq ft (within -20% to +25% of market base) that still
    achieves at least `target_pct` absorption.
    Because absorption decreases monotonically with price, we search the valid price range.
    """
    if bundle is None:
        bundle = load_model()

    micro_market = inputs["micro_market"]
    base_psf = MARKET_BASE_PSF[micro_market]

    # Test lower bound first (-20% discount)
    test_inputs = dict(inputs)
    test_inputs["price_vs_market_pct"] = -20.0
    test_inputs.pop("price_per_sqft", None)
    best_at_min_price, _, _ = predict(test_inputs, bundle)

    if best_at_min_price < target_pct:
        return {
            "achievable": False,
            "message": f"Not achievable in the standard price range (even at a 20% discount, forecast is {best_at_min_price}%).",
            "max_price_psf": None,
            "price_vs_market_pct": None,
            "predicted_pct": best_at_min_price,
        }

    # Test upper bound (+25% premium)
    test_inputs["price_vs_market_pct"] = 25.0
    at_max_price, _, _ = predict(test_inputs, bundle)
    if at_max_price >= target_pct:
        psf = int(round(base_psf * 1.25))
        return {
            "achievable": True,
            "message": f"Target is achievable across the entire range up to a +25% premium (₹{psf:,}/sq ft).",
            "max_price_psf": psf,
            "price_vs_market_pct": 25.0,
            "predicted_pct": at_max_price,
        }

    # Binary search over price_vs_market_pct in [-20, 25] with 0.1% resolution
    low_pct = -20.0
    high_pct = 25.0
    best_pct = low_pct
    best_pred = best_at_min_price

    for _ in range(25):
        mid_pct = (low_pct + high_pct) / 2.0
        test_inputs["price_vs_market_pct"] = mid_pct
        pred, _, _ = predict(test_inputs, bundle)
        if pred >= target_pct:
            best_pct = mid_pct
            best_pred = pred
            low_pct = mid_pct
        else:
            high_pct = mid_pct

    max_psf = int(round(base_psf * (1.0 + best_pct / 100.0)))
    return {
        "achievable": True,
        "max_price_psf": max_psf,
        "price_vs_market_pct": round(best_pct, 1),
        "predicted_pct": round(best_pred, 1),
        "message": f"Highest price that still hits {target_pct:.0f}%: ₹{max_psf:,}/sq ft ({best_pct:+.1f}% vs market)",
    }


def unit_mix(
    inputs: Dict[str, Any],
    bundle: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Compare absorption for all 4 configurations (1BHK, 2BHK, 3BHK, 4BHK) with their
    respective default carpet areas, keeping the same price_vs_market_pct and inventory scale.
    Returns:
      - DataFrame with columns: [configuration, carpet_area_sqft, price_per_sqft, predicted_pct, low_pct, high_pct, units_sold_est]
      - Takeaway string (e.g. "2BHK is expected to sell fastest here (+X pts vs 3BHK).")
    """
    if bundle is None:
        bundle = load_model()

    micro_market = inputs["micro_market"]
    base_psf = MARKET_BASE_PSF[micro_market]

    # Resolve price_vs_market_pct
    if "price_vs_market_pct" in inputs:
        pct = float(inputs["price_vs_market_pct"])
    elif "price_per_sqft" in inputs:
        pct = 100.0 * (float(inputs["price_per_sqft"]) / base_psf - 1.0)
    else:
        pct = 0.0

    total_units = int(inputs.get("total_units", 150))
    current_psf = int(round(base_psf * (1.0 + pct / 100.0)))

    rows = []
    config_preds = {}

    for cfg in ["1BHK", "2BHK", "3BHK", "4BHK"]:
        test_inputs = dict(inputs)
        test_inputs["configuration"] = cfg
        test_inputs["carpet_area_sqft"] = CARPET_DEFAULT[cfg]
        test_inputs["price_vs_market_pct"] = pct
        test_inputs.pop("price_per_sqft", None)

        pt, low, high = predict(test_inputs, bundle)
        units_sold = int(round(total_units * pt / 100.0))
        config_preds[cfg] = pt

        rows.append({
            "configuration": cfg,
            "carpet_area_sqft": CARPET_DEFAULT[cfg],
            "price_per_sqft": current_psf,
            "predicted_pct": pt,
            "low_pct": low,
            "high_pct": high,
            "units_sold_est": units_sold,
        })

    df = pd.DataFrame(rows)

    # Generate one-line takeaway
    best_cfg = max(config_preds, key=config_preds.get)
    # Compare with 3BHK or second best
    other_cfg = "3BHK" if best_cfg != "3BHK" else "2BHK"
    diff = round(config_preds[best_cfg] - config_preds[other_cfg], 1)
    takeaway = f"{best_cfg} is expected to sell fastest here (+{diff} pts vs {other_cfg})."

    return df, takeaway


def project_mix_forecast(
    inputs: Dict[str, Any],
    mix_dict: Dict[str, int],
    bundle: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulate absorption for a project-level inventory mix (e.g. 1BHK: 0, 2BHK: 150, 3BHK: 100, 4BHK: 0).
    Runs the model for each configuration with units > 0 at its default carpet area,
    and computes the units-weighted whole-project forecast and expected sales.
    """
    if bundle is None:
        bundle = load_model()

    micro_market = inputs["micro_market"]
    base_psf = MARKET_BASE_PSF[micro_market]

    if "price_vs_market_pct" in inputs:
        pct = float(inputs["price_vs_market_pct"])
    elif "price_per_sqft" in inputs:
        pct = 100.0 * (float(inputs["price_per_sqft"]) / base_psf - 1.0)
    else:
        pct = 0.0

    current_psf = int(round(base_psf * (1.0 + pct / 100.0)))

    rows = []
    total_project_units = 0
    total_project_sold = 0
    weighted_sum_pts = 0.0
    weighted_sum_low = 0.0
    weighted_sum_high = 0.0

    for cfg in ["1BHK", "2BHK", "3BHK", "4BHK"]:
        count = int(mix_dict.get(cfg, 0))
        if count <= 0:
            continue

        test_inputs = dict(inputs)
        test_inputs["configuration"] = cfg
        test_inputs["carpet_area_sqft"] = CARPET_DEFAULT[cfg]
        test_inputs["total_units"] = count
        test_inputs["price_vs_market_pct"] = pct
        test_inputs.pop("price_per_sqft", None)

        pt, low, high = predict(test_inputs, bundle)
        sold = int(round(count * pt / 100.0))

        total_project_units += count
        total_project_sold += sold
        weighted_sum_pts += count * pt
        weighted_sum_low += count * low
        weighted_sum_high += count * high

        rows.append({
            "configuration": cfg,
            "units": count,
            "carpet_area_sqft": CARPET_DEFAULT[cfg],
            "price_per_sqft": current_psf,
            "predicted_pct": pt,
            "low_pct": low,
            "high_pct": high,
            "units_sold_est": sold,
        })

    rows_df = pd.DataFrame(rows)

    if total_project_units > 0:
        overall_pct = round(weighted_sum_pts / total_project_units, 1)
        overall_low = round(weighted_sum_low / total_project_units, 1)
        overall_high = round(weighted_sum_high / total_project_units, 1)
    else:
        overall_pct = 0.0
        overall_low = 0.0
        overall_high = 0.0

    summary = {
        "total_units": total_project_units,
        "total_units_sold": total_project_sold,
        "weighted_pct": overall_pct,
        "weighted_low": overall_low,
        "weighted_high": overall_high,
    }

    return rows_df, summary

