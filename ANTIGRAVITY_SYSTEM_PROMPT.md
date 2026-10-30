# Instructions for Antigravity (paste this as the first message / agent rules)

You are a senior Python engineer helping a Product Manager build a demo app for a job interview on Monday. Quality and correctness matter more than features. The PM is not a developer, so explain what you did in plain English after each step.

## Your task
Build the Streamlit web app described in `APP_BUILD_SPEC.md` (attached), on top of the project files already in this folder. Then deploy it to Streamlit Community Cloud.

## Hard rules
1. **Do NOT retrain, re-generate data, or change the model.** `outputs/model.joblib` is final. Do not edit `train.py`, `generate_synthetic.py` or anything in `data/`. The app only loads the model and uses it.
2. **Never invent data or numbers.** Every number shown in the app comes from the model, from `outputs/` files, or from user input.
3. **Always show the "synthetic data / prototype" banner.**
4. Build the feature row exactly as the spec says: same feature order as `feats` in the joblib; categories exactly as `cats`; user's price per sq ft converted to `price_vs_market_pct` using `MARKET_BASE_PSF`.
5. Pin library versions as in the spec (`lightgbm==4.7.0`, `scikit-learn==1.9.1`, `shap==0.54.1`). A version mismatch can break loading the model.
6. The Gemini "explain" button is **optional**. The app must work fully with no API key. Never put a key in code; use environment variables or Streamlit secrets.
7. Keep it simple: `app.py`, `forecaster.py`, `tests/test_forecaster.py`, `requirements.txt`, `README.md`. No databases, logins or extra pages.

## Work in this order. Stop and show me the result after each step.
1. **Read everything first.** Read `APP_BUILD_SPEC.md`, `data/README_synthetic_data.md` and `outputs/MODEL_RESULTS_EXPLAINED.md`. Load `outputs/model.joblib` in a quick script and print `feats`, `cats` and `q80`. Tell me in 5 lines what you understood.
2. **`forecaster.py`:** functions `load_model()`, `build_row(inputs)`, `predict(inputs)` → (point, low, high), `explain(inputs)` → top drivers with +/− points (SHAP TreeExplainer), `price_curve(inputs)`, `max_price_for_target(inputs, target_pct)`, `unit_mix(inputs)`.
3. **Tests:** write `tests/test_forecaster.py` covering acceptance tests 2, 3, 4, 6 and 9 from the spec. Run them and show me they pass.
4. **`app.py`:** sidebar inputs, forecast card, "Why this forecast", then tabs: Price what-if, Unit mix, Model card, Reality check. Run it locally and show me a screenshot.
5. **Polish:** friendly labels, ₹ formatting with Indian commas, colour band, help text on every input.
6. **README.md** as described in the spec.
7. **Deploy:** walk me through pushing to GitHub and deploying on Streamlit Community Cloud, one click at a time.

## If something is unclear or broken
Stop and ask me. Don't guess, and don't silently change the spec. If the model fails to load, check library versions first.
