"""
ANAROCK Absorption Forecaster: Pune (Prototype)
Streamlit web application for predicting 12-month residential sales absorption.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import forecaster

# Page configuration
st.set_page_config(
    page_title="ANAROCK Absorption Forecaster",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown(
    """
    <style>
    .banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 6px solid #ffeeba;
        font-weight: 500;
        margin-bottom: 22px;
        font-size: 0.95rem;
    }
    .metric-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .metric-headline {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin-bottom: 6px;
        line-height: 1.1;
    }
    .metric-subline {
        font-size: 1.05rem !important;
        color: #334155 !important;
        margin-bottom: 12px;
    }
    .metric-subline strong {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-fast { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-moderate { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-slow { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .bullet-item {
        padding: 8px 14px;
        margin-bottom: 6px;
        background: #f1f5f9;
        color: #0f172a !important;
        border-left: 3px solid #3b82f6;
        border-radius: 0 6px 6px 0;
        font-size: 0.95rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_inr(number: float) -> str:
    """Format number in Indian comma system (e.g. ₹11,400 or ₹1,25,000)."""
    n = int(round(number))
    s = str(abs(n))
    if len(s) <= 3:
        res = s
    else:
        res = s[-3:]
        s = s[:-3]
        while len(s) > 2:
            res = s[-2:] + "," + res
            s = s[:-2]
        if s:
            res = s + "," + res
    return f"-₹{res}" if n < 0 else f"₹{res}"


@st.cache_resource
def get_model():
    """Load and cache the pre-trained LightGBM model bundle."""
    return forecaster.load_model()


bundle = get_model()

# --- ALWAYS-VISIBLE SYNTHETIC DATA BANNER ---
st.markdown(
    """
    <div class="banner">
        ⚠️ <strong>Prototype trained on synthetic data</strong> calibrated to real Pune projects.
        For demonstration & feasibility advisory workflows only; not for unverified pricing decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("Absorption Forecaster: Pune")
st.caption("How fast will this launch sell? 12-month residential absorption forecast, explainable drivers, and pricing what-ifs.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("🏢 Project Configuration")

micro_market = st.sidebar.selectbox(
    "Micro-market",
    options=bundle["cats"]["micro_market"],
    index=bundle["cats"]["micro_market"].index("Hinjewadi"),
    help="Select the Pune micro-market. Sets the baseline benchmark price and local access midpoints.",
)

config = st.sidebar.selectbox(
    "Unit Configuration",
    options=bundle["cats"]["configuration"],
    index=bundle["cats"]["configuration"].index("2BHK"),
    help="Apartment configuration type.",
)

carpet_min, carpet_max = forecaster.CARPET_RANGE[config]
default_carpet = forecaster.CARPET_DEFAULT[config]
carpet_area = st.sidebar.slider(
    "Carpet Area (sq ft)",
    min_value=int(carpet_min),
    max_value=int(carpet_max),
    value=int(default_carpet),
    step=10,
    help=f"RERA carpet area. Recommended range for {config}: {carpet_min}–{carpet_max} sq ft.",
)

# Price per sq ft
base_psf = forecaster.MARKET_BASE_PSF[micro_market]
default_psf = 10000 if micro_market == "Hinjewadi" else int(base_psf)

price_psf = st.sidebar.number_input(
    "Launch Price per sq ft (₹)",
    min_value=int(base_psf * 0.5),
    max_value=int(base_psf * 2.0),
    value=default_psf,
    step=100,
    help=f"Planned launch price per sq ft. Local market benchmark is {format_inr(base_psf)}/sq ft.",
)

price_diff_pct = 100.0 * (price_psf / base_psf - 1.0)
st.sidebar.caption(
    f"📍 Market average: **{format_inr(base_psf)}/sq ft** → You are **{price_diff_pct:+.1f}%** vs benchmark."
)

tier = st.sidebar.selectbox(
    "Developer Brand Tier",
    options=[1, 2, 3],
    index=0,
    format_func=lambda x: forecaster.TIER_LABELS[x],
    help="Developer brand categorization. Tier 1 = national listed; Tier 2 = established regional; Tier 3 = small/new.",
)

total_units = st.sidebar.slider(
    "Inventory of this Unit Type",
    min_value=20,
    max_value=420,
    value=150,
    step=10,
    help="Total number of units of this specific configuration released at launch.",
)

months_possession = st.sidebar.slider(
    "Months to Possession",
    min_value=24,
    max_value=60,
    value=42,
    step=2,
    help="Committed construction timeline from launch date to buyer handover.",
)

# Advanced inputs expander
default_metro, default_it = forecaster.DEFAULT_DIST[micro_market]
with st.sidebar.expander("⚙️ Advanced Location & Market Timing"):
    dist_metro = st.slider(
        "Distance to Metro Station (km)",
        min_value=0.5,
        max_value=15.0,
        value=float(default_metro),
        step=0.1,
        help="Proximity to nearest existing or planned Pune Metro station.",
    )
    dist_it_hub = st.slider(
        "Distance to Major IT Hub (km)",
        min_value=0.5,
        max_value=15.0,
        value=float(default_it),
        step=0.1,
        help="Proximity to major employment corridor (e.g. Hinjewadi Phase 1/2/3, Kharadi EON, etc.).",
    )
    market_trend = st.slider(
        "Pune Market Sales Trend (%)",
        min_value=-20.0,
        max_value=30.0,
        value=float(forecaster.DEFAULT_MARKET_TREND),
        step=1.0,
        help="Trailing 4-quarter citywide Pune residential sales growth at time of launch. Negative indicates cooling cycle.",
    )

# Input parameters dictionary
inputs = {
    "micro_market": micro_market,
    "configuration": config,
    "carpet_area_sqft": carpet_area,
    "price_per_sqft": price_psf,
    "developer_tier": tier,
    "total_units": total_units,
    "months_to_possession": months_possession,
    "dist_metro_km": dist_metro,
    "dist_it_hub_km": dist_it_hub,
    "market_sales_trend_pct": market_trend,
}

# Training distribution guardrail warnings
if price_diff_pct < -20.0 or price_diff_pct > 25.0:
    st.warning(
        f"⚠️ **Price Alert:** Current price is **{price_diff_pct:+.1f}%** vs local market average. "
        "The model is calibrated between −20% and +25%. Extrapolations beyond this range carry increased uncertainty."
    )
if market_trend < -17.0 or market_trend > 30.0:
    st.warning(
        f"⚠️ **Market Trend Alert:** Trailing sales trend of {market_trend:+.1f}% is outside the core training range (−17% to +30%)."
    )

# --- RUN PREDICTION & EXPLANATION ---
point, low, high = forecaster.predict(inputs, bundle)
units_sold_est = int(round(total_units * point / 100.0))
exp_data = forecaster.explain(inputs, bundle)

# Determine pace status badge
if point > 40.0:
    badge_html = '<span class="badge badge-fast">🟢 Fast Pace (>40%)</span>'
elif point >= 20.0:
    badge_html = '<span class="badge badge-moderate">🟡 Moderate Pace (20–40%)</span>'
else:
    badge_html = '<span class="badge badge-slow">🔴 Slow Pace (<20%)</span>'

# --- 3.3 FORECAST CARD ---
st.markdown(
    f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="color: #475569 !important; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
                    12-Month Absorption Forecast
                </div>
                <div class="metric-headline" style="color: #0f172a !important; font-size: 2.6rem !important; font-weight: 800 !important; margin: 4px 0 8px 0;">
                    {point:.1f}% sold in 12 months
                </div>
                <div class="metric-subline" style="color: #334155 !important; font-size: 1.05rem;">
                    Likely range: <strong style="color: #0f172a !important;">{low:.1f}% – {high:.1f}%</strong> (75–80% confidence)
                    &nbsp;·&nbsp;
                    ≈ <strong style="color: #0f172a !important;">{units_sold_est}</strong> of {total_units} {config} units
                </div>
            </div>
            <div>
                {badge_html}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 3.9 OPTIONAL: GEMINI "EXPLAIN LIKE I'M THE DEVELOPER" ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        gemini_api_key = None

if gemini_api_key:
    with st.expander("✨ Executive Pitch: Explain like I'm the developer"):
        if st.button("Generate 4-Sentence Advisory Summary"):
            with st.spinner("Synthesizing narrative with Gemini..."):
                try:
                    from google import genai

                    client = genai.Client(api_key=gemini_api_key)
                    prompt = f"""
                    You are a senior real estate advisory partner at ANAROCK explaining a launch feasibility forecast to a developer's Strategy Head.
                    Use ONLY the provided facts. Do NOT invent numbers or outside data.
                    Mention the predicted range ({low:.1f}% to {high:.1f}%) and clearly note that this forecast is generated by a prototype trained on synthetic data calibrated to Pune benchmarks.
                    Provide a concise 4-sentence summary covering the forecast, primary market drivers, and one actionable pricing suggestion.

                    Project Details:
                    - Micro-market: {micro_market}
                    - Configuration: {config} ({carpet_area} sq ft)
                    - Launch Price: Rs. {price_psf:,}/sq ft ({price_diff_pct:+.1f}% vs {micro_market} benchmark of Rs. {base_psf:,}/sq ft)
                    - Developer Tier: {forecaster.TIER_LABELS[tier]}
                    - Inventory: {total_units} units | Timeline: {months_possession} months
                    - Forecast: {point:.1f}% (Range: {low:.1f}% – {high:.1f}%)
                    - Top SHAP Drivers:
                    {chr(10).join(['  * ' + b for b in exp_data['bullets']])}
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    st.markdown(f"**Advisory Brief:**\n\n{response.text}")
                except Exception as e:
                    st.error(f"Could not connect to Gemini API: {e}")

# --- 3.4 WHY THIS FORECAST (EXPLAINABILITY) ---
st.subheader("🔍 Why this forecast? (SHAP Attribution)")
st.caption("How each project factor added or subtracted points from the base expectation.")

col_chart, col_bullets = st.columns([3, 2])

with col_chart:
    # Top 6 drivers horizontal bar chart
    top6 = exp_data["drivers"][:6][::-1]  # reverse to show largest at top
    labels = [d["friendly_label"] for d in top6]
    vals = [d["shap_value"] for d in top6]
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in vals]
    custom_text = [f"{v:+.1f} pts ({d['display_val']})" for v, d in zip(vals, top6)]

    fig_shap = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=custom_text,
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Impact: %{x:+.2f} pts<extra></extra>",
        )
    )
    fig_shap.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        xaxis=dict(title="Contribution (Percentage Points)", zeroline=True, zerolinecolor="#94a3b8"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

with col_bullets:
    st.markdown("**Top Drivers for This Launch:**")
    for b in exp_data["bullets"]:
        st.markdown(f'<div class="bullet-item">• {b}</div>', unsafe_allow_html=True)
    st.caption(f"Baseline expectation across Pune data: **{exp_data['base_value']:.1f}%**.")

st.divider()

# --- TABS FOR DEEP DIVES ---
tab_price, tab_mix, tab_model, tab_reality = st.tabs([
    "📈 Price What-If",
    "📊 Unit-Mix Analysis",
    "📋 Model Card & Trust",
    "✅ Reality Check vs Real Projects",
])

# ==================== TAB 1: PRICE WHAT-IF ====================
with tab_price:
    st.markdown("### Price Sensitivity & Target Finder")
    st.caption("Evaluate how changes in price per sq ft impact predicted absorption pace, holding all other features fixed.")

    curve_df = forecaster.price_curve(inputs, bundle)

    fig_curve = go.Figure()
    # Confidence band shading
    fig_curve.add_trace(
        go.Scatter(
            x=curve_df["price_per_sqft"],
            y=curve_df["high_pct"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_curve.add_trace(
        go.Scatter(
            x=curve_df["price_per_sqft"],
            y=curve_df["low_pct"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.15)",
            name="Likely Range (±11.4 pts)",
            hoverinfo="skip",
        )
    )
    # Expected forecast line
    fig_curve.add_trace(
        go.Scatter(
            x=curve_df["price_per_sqft"],
            y=curve_df["predicted_pct"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6),
            name="Forecast Absorption %",
            hovertemplate="Price: ₹%{x:,}/sq ft<br>Forecast: %{y:.1f}%<extra></extra>",
        )
    )
    # Current price marker
    fig_curve.add_trace(
        go.Scatter(
            x=[price_psf],
            y=[point],
            mode="markers+text",
            marker=dict(color="#dc2626", size=12, symbol="diamond"),
            name="Current Price",
            text=[f"Current: {point:.1f}%"],
            textposition="top right",
            hovertemplate="Current Price: ₹%{x:,}/sq ft<br>Forecast: %{y:.1f}%<extra></extra>",
        )
    )

    fig_curve.update_layout(
        xaxis=dict(title="Launch Price per sq ft (₹)", tickprefix="₹", tickformat=","),
        yaxis=dict(title="12-Month Absorption %", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=380,
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    # Target Finder
    st.markdown("#### 🎯 Target Finder")
    col_tgt1, col_tgt2 = st.columns([1, 2])
    with col_tgt1:
        target_goal = st.number_input(
            "Target 12-Month Absorption (%)",
            min_value=10.0,
            max_value=90.0,
            value=35.0,
            step=1.0,
            help="What sales pace does the developer need to achieve in year 1?",
        )
    with col_tgt2:
        target_res = forecaster.max_price_for_target(inputs, target_goal, bundle)
        if target_res["achievable"]:
            st.success(
                f"**Achievable Pricing Strategy:**\n\n"
                f"Highest price that still hits **{target_goal:.0f}%**: "
                f"**{format_inr(target_res['max_price_psf'])}/sq ft** ({target_res['price_vs_market_pct']:+.1f}% vs {micro_market} benchmark)."
            )
        else:
            st.error(f"**Target Unreachable:** {target_res['message']}")

# ==================== TAB 2: UNIT-MIX COMPARISON ====================
with tab_mix:
    st.markdown("### 🏗️ Project Mix (Multi-Configuration Forecast)")
    st.caption("Enter the planned inventory across configurations for this project to calculate the blended, whole-project absorption forecast.")

    # Small editable table where user enters units for 1BHK, 2BHK, 3BHK, and 4BHK (default 0, 150, 100, 0)
    default_mix_data = pd.DataFrame([
        {"Configuration": "1BHK", "Units": 0, "Default Carpet (sq ft)": forecaster.CARPET_DEFAULT["1BHK"]},
        {"Configuration": "2BHK", "Units": 150, "Default Carpet (sq ft)": forecaster.CARPET_DEFAULT["2BHK"]},
        {"Configuration": "3BHK", "Units": 100, "Default Carpet (sq ft)": forecaster.CARPET_DEFAULT["3BHK"]},
        {"Configuration": "4BHK", "Units": 0, "Default Carpet (sq ft)": forecaster.CARPET_DEFAULT["4BHK"]},
    ])

    edited_mix = st.data_editor(
        default_mix_data,
        column_config={
            "Configuration": st.column_config.TextColumn("Configuration", disabled=True),
            "Default Carpet (sq ft)": st.column_config.NumberColumn("Default Carpet (sq ft)", disabled=True),
            "Units": st.column_config.NumberColumn("Units", min_value=0, max_value=2000, step=10, required=True),
        },
        hide_index=True,
        use_container_width=True,
        key="project_mix_editor",
    )

    # Extract user entered unit counts
    user_mix_dict = dict(zip(edited_mix["Configuration"], edited_mix["Units"]))

    # Run project mix forecast using current sidebar settings at default carpet area
    proj_df, proj_summary = forecaster.project_mix_forecast(inputs, user_mix_dict, bundle)

    if proj_summary["total_units"] > 0:
        # Determine pace status badge for whole project
        wp_pct = proj_summary["weighted_pct"]
        if wp_pct > 40.0:
            wp_badge = '<span class="badge badge-fast">🟢 Fast Pace (>40%)</span>'
        elif wp_pct >= 20.0:
            wp_badge = '<span class="badge badge-moderate">🟡 Moderate Pace (20–40%)</span>'
        else:
            wp_badge = '<span class="badge badge-slow">🔴 Slow Pace (<20%)</span>'

        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 14px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="color: #475569 !important; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
                            Whole-Project 12-Month Absorption Forecast (Units-Weighted)
                        </div>
                        <div class="metric-headline" style="color: #0f172a !important; font-size: 2.3rem !important; font-weight: 800 !important; margin: 4px 0 8px 0;">
                            {wp_pct:.1f}% sold across project
                        </div>
                        <div class="metric-subline" style="color: #334155 !important; font-size: 1.05rem;">
                            Likely range: <strong style="color: #0f172a !important;">{proj_summary['weighted_low']:.1f}% – {proj_summary['weighted_high']:.1f}%</strong>
                            &nbsp;·&nbsp;
                            Total expected to sell: <strong style="color: #0f172a !important;">{proj_summary['total_units_sold']}</strong> of {proj_summary['total_units']} units in 12 months
                        </div>
                    </div>
                    <div>
                        {wp_badge}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show active configuration breakdown table
        display_proj_df = proj_df.copy()
        display_proj_df["price_per_sqft"] = display_proj_df["price_per_sqft"].apply(format_inr)
        display_proj_df["predicted_pct"] = display_proj_df["predicted_pct"].apply(lambda v: f"{v:.1f}%")
        display_proj_df["likely_range"] = proj_df.apply(lambda r: f"{r['low_pct']:.1f}% – {r['high_pct']:.1f}%", axis=1)
        display_proj_df["units_sold_est"] = proj_df.apply(lambda r: f"≈ {r['units_sold_est']} of {r['units']} units", axis=1)

        display_proj_df = display_proj_df[[
            "configuration", "units", "carpet_area_sqft", "price_per_sqft", "predicted_pct", "likely_range", "units_sold_est"
        ]].rename(columns={
            "configuration": "Config",
            "units": "Planned Units",
            "carpet_area_sqft": "Carpet (sq ft)",
            "price_per_sqft": "Price/sq ft",
            "predicted_pct": "Forecast %",
            "likely_range": "Likely Range",
            "units_sold_est": "Units Sold Est.",
        })
        st.dataframe(display_proj_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Please enter at least one unit configuration with units > 0 to view the whole-project forecast.")

    st.divider()

    st.markdown("#### 📊 Benchmark Comparison (Equal 150-Unit Scale)")
    st.caption("How each BHK would perform under identical inventory sizing (150 units each) at the current price premium/discount.")

    mix_df, takeaway = forecaster.unit_mix(inputs, bundle)
    st.info(f"💡 **Takeaway:** {takeaway}")

    col_mix_table, col_mix_chart = st.columns([3, 2])

    with col_mix_table:
        display_df = mix_df.copy()
        display_df["price_per_sqft"] = display_df["price_per_sqft"].apply(format_inr)
        display_df["predicted_pct"] = display_df["predicted_pct"].apply(lambda v: f"{v:.1f}%")
        display_df["likely_range"] = mix_df.apply(lambda r: f"{r['low_pct']:.1f}% – {r['high_pct']:.1f}%", axis=1)
        display_df["units_sold_est"] = mix_df.apply(lambda r: f"≈ {r['units_sold_est']} of {total_units}", axis=1)

        display_df = display_df[[
            "configuration", "carpet_area_sqft", "price_per_sqft", "predicted_pct", "likely_range", "units_sold_est"
        ]].rename(columns={
            "configuration": "Config",
            "carpet_area_sqft": "Carpet (sq ft)",
            "price_per_sqft": "Price/sq ft",
            "predicted_pct": "Forecast %",
            "likely_range": "Likely Range",
            "units_sold_est": "Units Sold Est.",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_mix_chart:
        fig_mix = go.Figure(
            go.Bar(
                x=mix_df["configuration"],
                y=mix_df["predicted_pct"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=mix_df["high_pct"] - mix_df["predicted_pct"],
                    arrayminus=mix_df["predicted_pct"] - mix_df["low_pct"],
                ),
                marker_color=["#3b82f6", "#10b981", "#f59e0b", "#6366f1"],
                text=[f"{v:.1f}%" for v in mix_df["predicted_pct"]],
                textposition="auto",
            )
        )
        fig_mix.update_layout(
            title="Absorption Comparison by BHK",
            yaxis=dict(title="12-Month Absorption %", range=[0, 100]),
            margin=dict(l=10, r=10, t=40, b=10),
            height=300,
        )
        st.plotly_chart(fig_mix, use_container_width=True)

# ==================== TAB 3: MODEL CARD & TRUST ====================
with tab_model:
    st.markdown("### Model Card & Performance Transparency")
    st.caption("How the model was trained, validated, and how it compares against industry rules of thumb.")

    col_mc1, col_mc2 = st.columns([3, 2])

    with col_mc1:
        st.markdown("#### Evaluation Benchmark")
        if os.path.exists("outputs/metrics.csv"):
            metrics_df = pd.read_csv("outputs/metrics.csv")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("Evaluation metrics file not found.")

        st.markdown(
            """
            **Key Takeaways:**
            - **23% lower error** than standard rule-of-thumb heuristics (8.2 pts vs 10.7 pts MAE).
            - **Unbiased in market cycles:** Stays unbiased (−0.4 pts) during market cooling, whereas simple baselines over-predict by +6.3 pts.
            - **Confidence band:** ±11.4 pts achieves ~73% coverage on held-out projects (slightly optimistic vs 80% target).
            """
        )

        st.markdown("#### Known Limitations")
        st.markdown(
            """
            - **Synthetic training data:** 823 synthetic rows calibrated to real Pune price points and pace metrics.
            - **Tree boundary limits:** Cannot extrapolate beyond observed training conditions (e.g. market trend below −17% requires quarterly retraining).
            - **Micro-market baseline:** Pricing benchmark uses average micro-market levels; fringe sub-locations may appear artificially cheap.
            - **Coverage optimism:** Project-level correlation means actual outcomes cluster, yielding ~73% coverage vs the 80% target.
            """
        )

    with col_mc2:
        st.markdown("#### Global Feature Drivers (SHAP)")
        if os.path.exists("outputs/shap_importance.png"):
            st.image("outputs/shap_importance.png", caption="Mean |SHAP value| impact across the training dataset.")
        elif os.path.exists("outputs/shap_beeswarm.png"):
            st.image("outputs/shap_beeswarm.png", caption="SHAP beeswarm distribution across features.")
        else:
            st.info("SHAP plot image not available.")

# ==================== TAB 4: REALITY CHECK ====================
with tab_reality:
    st.markdown("### Reality Check vs Real Pune Projects")
    st.caption("Validation against 6 real-world Pune residential launches tracked via IndexTap.")

    if os.path.exists("outputs/reality_check.csv"):
        rc_df = pd.read_csv("outputs/reality_check.csv")
        display_rc = rc_df[[
            "project", "micro_market", "configuration", "price_per_sqft",
            "pred_12m", "plausible_band", "verdict"
        ]].copy()
        display_rc["price_per_sqft"] = display_rc["price_per_sqft"].apply(format_inr)
        display_rc["pred_12m"] = display_rc["pred_12m"].apply(lambda v: f"{v:.1f}%")
        display_rc.rename(columns={
            "project": "Project Name",
            "micro_market": "Micro-Market",
            "configuration": "Config",
            "price_per_sqft": "Price/sq ft",
            "pred_12m": "Model 12m Pred",
            "plausible_band": "Plausible Band",
            "verdict": "Verdict",
        }, inplace=True)

        st.dataframe(display_rc, use_container_width=True, hide_index=True)
    else:
        st.info("Reality check file not found.")

    st.markdown(
        """
        > **Understanding the Plausible Band Logic:**
        > * **Floor** = Lifetime sales pace × 12. Because residential launches are front-loaded, Year 1 pace should be at least this high.
        > * **Ceiling** = Real % sold to date. A project cannot sell more units in Year 1 than it has sold across its entire lifetime.
        > * **PASS Verdict** = The model's 12-month forecast falls comfortably within `[Floor, Ceiling]`. Note that PASS confirms *plausibility*, not exact precision.
        """
    )
