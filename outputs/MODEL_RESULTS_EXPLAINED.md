# Absorption Forecaster: Model Results Explained (Day 2)

Data: 823 synthetic rows (calibrated to real Pune projects) + 6 real projects held out as a reality check.
Error is measured in **percentage points (pts)**. If the actual absorption is 30% and the model says 38%, the error is 8 pts.

## 1. Did the model beat a simple rule?
**Baseline rule:** "Predict the typical absorption for this micro-market + configuration."

| Test | Baseline error | LightGBM error | Within ±10 pts | Bias |
|---|---|---|---|---|
| A: random projects held out | 10.7 pts | **8.2 pts** (~23% better) | 57% → 68% | ~0 |
| B: future launches (late 2024–25, cooling market) | 10.3 pts (over-predicts by +6.3) | **6.9 pts** | 56% → 74% | **−0.4** |

**Why test B matters:** the baseline was learnt in the 2022 boom and over-predicts by about 6 pts when the market cools. The model avoids that because "market sales trend at launch" is an input.
**The limit:** test-period trends went down to −17%, below anything seen in training (−8%). Tree models can't extrapolate, so a sharper downturn needs retraining. In production, monitor drift and retrain quarterly.

**Why split by project:** a project's 2BHK and 3BHK rows share the same hidden quality, so they must stay on the same side of the split. Otherwise the test leaks.

## 2. How confident is each prediction?
Every prediction comes with a range of ±11 pts, calibrated on held-out training folds.
- Target: the actual result falls inside the range about 80% of the time. **Achieved: about 73%**, slightly under, because rows from the same project move together.
- Honest framing: "The range is a little optimistic. Next step: a project-level calibration or a wider band."

## Guardrails
The model has business-logic rules (monotonic constraints): a higher price premium, a weaker developer tier, a longer distance, a later possession or a bigger inventory can never *raise* the prediction, and a hotter market can never lower it. The raw price-per-sq-ft input was removed (it duplicated "price vs local market" and made the price curve bumpy). This makes what-if answers sensible, and it slightly improved accuracy.

## 3. What drives absorption? (SHAP)
SHAP splits each prediction into "how many points each input added or removed."

Ranking (average impact, in pts):
1. Market sales trend at launch: 6.0. Timing the launch to the cycle matters most.
2. Price vs micro-market: 4.8. Pricing above the local market slows sales.
3. Developer tier: 3.7. Brand trust sells.
4. Distance to IT hub: 3.1.
5. Carpet area: 2.4. Larger units are a smaller buyer pool.
6. Inventory size: 2.3. A large launch dilutes the % sold.
7. Configuration: 1.7. 2BHK +2.3 pts; 1BHK, 3BHK and 4BHK each about −1 to −1.6.

**Validation:** every direction matches the effect deliberately built into the data (price ↑ → absorption ↓, tier-1 ↑, distance ↓, later possession ↓, hot market ↑). This shows the pipeline recovers the true drivers.

**Why micro-market looks less important (0.8):** its effect is already captured by distance and price. That isn't a bug, but tell the user so they don't conclude "location doesn't matter."

## 4. Reality check vs 6 real Pune projects
Real data shows only "% sold to date," not 12-month absorption. So we check whether the model's 12-month prediction falls in a plausible band:
- **Floor** = lifetime average pace × 12. Sales are front-loaded, so year 1 should be at least this.
- **Ceiling** = % sold to date. You can't sell more in year 1 than in total.

| Project | Prediction | Plausible band | Result |
|---|---|---|---|
| Godrej Park World / Gale | 25% | 20–55% | PASS |
| VJ Palladio Kharadi Central | 39% | 25–90% | PASS |
| Kumar Parth Towers | 38% | 22–84% | PASS |
| Mantra Monarch | 31% | 13–89% | PASS |
| Ashiana Malhar Ph 1 | 65% | 20–86% | PASS (but high) |

**Honest caveats:**
- The bands are wide, so PASS means "plausible," not "accurate." It's a sanity check, not proof.
- Ashiana looks high because its price is about 10% below the *Hinjewadi average*. It's actually on the fringe (Marunji), where prices are lower anyway. Fix: compare price against a finer location baseline (pin code or 2-km radius) rather than the whole micro-market. A good example of a feature-design lesson.
- Real-project inputs had gaps filled with assumptions: 2BHK as the dominant configuration, distances at the market midpoint, developer tiers assigned by me.

## 5. What to say in the interview (30 seconds)
"On held-out projects the model is about 20% more accurate than the rule-of-thumb baseline, about 8.2 vs 10.7 points of error. On future launches in a cooling market it stays unbiased, while the rule over-predicts by 6 points. SHAP recovers the drivers we expect: market timing, price vs local market, developer brand and IT-hub distance. Its main risks are drift beyond the conditions it has seen, and a slightly optimistic confidence range. Both are addressed with quarterly retraining and project-level calibration."
