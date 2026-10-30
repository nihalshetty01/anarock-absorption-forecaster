# Start here

1. Unzip this folder and open it in Antigravity (File → Open Folder).
2. Paste the contents of `ANTIGRAVITY_SYSTEM_PROMPT.md` as your first message to the agent.
3. Attach or mention `APP_BUILD_SPEC.md`. It's the PRD for the app.
4. Let the agent work one step at a time and check each result.

Folder contents:
- `data/`: synthetic training data, real Pune projects, data dictionary
- `outputs/`: trained model (`model.joblib`), metrics, SHAP charts, reality check, results explained
- `generate_synthetic.py`, `train.py`: how data and model were made (do not modify)
- `requirements.txt`: pinned library versions
- `DATA_TO_IMPROVE_THE_MODEL.md`: what data would make it more accurate
