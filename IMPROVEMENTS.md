!# Cerberus Fraud Detection — Improvements & Fixes

A checklist of suggested improvements. Work through these with Copilot.

---

## Bug Fixes

- [ ] **Fix: Remove redundant `class_weight='balanced'` in `model.py`**
  - SMOTE already balances classes by oversampling. Using `class_weight='balanced'` on top of it double-counts the correction and skews the model.
  - **Fix:** Remove `class_weight='balanced'` from `RandomForestClassifier` in `model.py:54`.

- [ ] **Fix: Cap the `rows` query parameter in `/generate_sample`**
  - No upper bound means a request like `?rows=10000000` can crash the server.
  - **Fix:** In `app.py:63`, replace:
    ```python
    rows = int(request.args.get("rows", 10))
    ```
    with:
    ```python
    rows = min(max(int(request.args.get("rows", 10)), 1), 1000)
    ```

---

## Code Quality

- [ ] **Refactor: Deduplicate CSV read + inference logic in `app.py`**
  - `/predict` (`app.py:82`) and `/download` (`app.py:126`) share identical file-reading, validation, transformation, and prediction code.
  - **Fix:** Extract a shared helper function `run_inference(uploaded_file) -> (result_df, error)` and call it from both routes.

- [ ] **Refactor: Extract confusion matrix save in `model.py`**
  - Training metrics are only printed to stdout (`model.py:66`). They are lost after training.
  - **Fix:** After `classification_report`, serialize metrics to `metrics.json` using `json.dump` so the app can read and display them.

---

## Features

- [ ] **Feature: Expose model metrics on the dashboard**
  - Add a `/metrics` GET endpoint in `app.py` that reads `metrics.json` (generated during training) and returns precision, recall, F1, and confusion matrix as JSON.
  - Display these on the frontend (`static/app.js` / `templates/index.html`) so users know the model's reliability before uploading data.

- [ ] **Feature: Add a `/reload` endpoint to hot-reload model artifacts**
  - Currently, if `model.py` is rerun while the Flask server is up, the server continues using stale `.pkl` files until restarted.
  - **Fix:** Add a `POST /reload` endpoint in `app.py` that calls `load_model_assets()` again and updates the global `MODEL`, `SCALER`, `FEATURE_ORDER`.

- [ ] **Feature: Configurable fraud probability threshold**
  - The model uses the default 0.5 classification threshold. For fraud detection, a lower threshold (e.g. 0.3) reduces false negatives at the cost of more false positives — which is usually preferable.
  - **Fix:** Accept an optional `threshold` field in the `/predict` request (form data or query param), defaulting to `0.5`. Apply it when converting `probabilities` to `predictions`.

- [ ] **Feature: Add structured logging to `app.py`**
  - There is no logging anywhere in the Flask app. Errors and prediction stats are invisible in production.
  - **Fix:** Add Python `logging` at the top of `app.py`. Log: request start, fraud summary (total/fraud count), and all caught exceptions with `logging.exception(...)`.

---

## Sample Data

- [ ] **Improve: Make `/generate_sample` use realistic rows**
  - Currently generates random Gaussian noise for V1–V28, which doesn't resemble real PCA-transformed credit card data. Demo predictions will be unreliable.
  - **Fix:** If `creditcard.csv` exists, sample real rows from it (stripping the `Class` label) instead of generating synthetic noise. Fall back to Gaussian only if the CSV is absent.

---

## Project Setup

- [ ] **Add `requirements.txt`**
  - No dependency file exists. The project cannot be reproduced without guessing versions.
  - **Fix:** Run `pip freeze > requirements.txt` inside the venv and commit it. Key packages: `flask`, `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `joblib`.

- [ ] **Add `.gitignore`**
  - `creditcard.csv` (100MB+) and `venv/` should not be committed.
  - Ensure `.gitignore` includes: `venv/`, `*.pkl`, `creditcard.csv`, `__pycache__/`, `*.pyc`.
