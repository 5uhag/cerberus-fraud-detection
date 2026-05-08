# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Cerberus is a credit card fraud detection web app. A Random Forest model (trained offline) is served via a Flask API. Users upload a CSV of transactions and get fraud predictions, a summary, analytics charts, and a downloadable results file.

Live deployment: https://cerberus-fraud-detection.onrender.com (Render free tier — cold starts expected)
GitHub: https://github.com/5uhag/cerberus-fraud-detection

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (requires creditcard.csv in project root)
python model.py

# Run the dev server
python app.py

# Regenerate demo sample CSVs in static/samples/
python scripts/generate_demo_datasets.py
```

Production is served with `gunicorn app:app`.

## Architecture

### Two-phase design

**Phase 1 — offline training (`model.py`):**
Reads `creditcard.csv`, scales `Amount` and `Time` with `StandardScaler`, applies SMOTE to balance the minority fraud class, trains a `RandomForestClassifier`, then writes four artifacts to disk: `fraud_model.pkl`, `data_scaler.pkl`, `features_list.pkl`, `metrics.json`. These artifacts must exist before the Flask app can serve predictions.

**Phase 2 — online serving (`app.py`):**
Loads the four artifacts at startup into module-level globals (`MODEL`, `SCALER`, `FEATURE_ORDER`). All prediction routes call the shared `run_inference()` helper, which validates columns, applies the same scaler transform used during training, runs `predict_proba`, applies a configurable threshold, and returns `(result_df, summary, analytics, error, status_code)`.

### Key invariant — feature alignment

The scaler was fit on `['Amount', 'Time']` together (column order matters). `transform_for_inference()` in `app.py` must replicate the exact same transform. `FEATURE_ORDER` (loaded from `features_list.pkl`) enforces that inference receives columns in the same order the model was trained on.

### Analytics pipeline

After inference, `compute_analytics()` derives chart-ready data from the result: amount distribution buckets, fraud probability histogram (10 buckets × 10%), a time-series sample (up to 300 points), and top-5 highest-risk rows. This is returned as `analytics` in the `/predict` JSON response and rendered client-side with Chart.js.

### Frontend

Single-page app — `templates/index.html` with `static/app.js` and `static/styles.css`. No framework. Chart instances are tracked in `chartInstances{}` and destroyed before re-render to avoid Chart.js duplicate canvas errors. The analytics section, summary grid, and table section are all hidden by default and revealed by JS after a successful `/predict` call.

### Auth

Session-based auth via Flask `session` + `werkzeug.security` (no extra packages). Users stored in `users.json` (gitignored) as `{ username: hashed_password }`. The `login_required` decorator in `app.py` guards protected routes. `SECRET_KEY` env var should be set on Render — falls back to a dev string locally (sessions won't survive restarts in prod without it).

Things still TODO on auth:
- No password reset / forgot password flow
- No email verification
- No account management (change password, delete account)
- `users.json` is a flat file — fine for demo, would need a real DB for multi-user prod

### Routes

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Dashboard — requires login |
| GET/POST | `/login` | Login + sign-up page (tabs) |
| GET | `/logout` | Clears session, redirects to `/login` |
| POST | `/predict` | Returns summary + analytics + 15-row preview |
| POST | `/download` | Returns full result CSV |
| GET | `/metrics` | Reads `metrics.json`, returns model performance |
| POST | `/reload` | Hot-reloads pkl artifacts without restart |
| GET | `/generate_sample` | Synthetic or real-data sample CSV (`?rows=N`, max 1000) |

## Data

`creditcard.csv` is gitignored (100 MB+, not committed). The trained `.pkl` files are committed so Render can serve predictions without retraining. `metrics.json` is gitignored (generated artifact). `users.json` is gitignored (runtime auth data). Sample CSVs in `static/samples/` are committed and safe to use for testing — they include a `Class` column which the app ignores during inference.
