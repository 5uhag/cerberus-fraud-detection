import io
import json
import logging
import os
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

USERS_PATH = "users.json"


def load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "data_scaler.pkl"
FEATURES_PATH = "features_list.pkl"
METRICS_PATH = "metrics.json"
DEFAULT_THRESHOLD = 0.5
MAX_SAMPLE_ROWS = 1000
REQUIRED_RAW_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def load_model_assets():
    if not all(os.path.exists(path) for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURES_PATH)
    return model, scaler, feature_order


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except Exception:
        logger.exception("Failed to load model metrics.")
        return None


MODEL, SCALER, FEATURE_ORDER = load_model_assets()


def validate_columns(dataframe):
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in dataframe.columns]
    return missing


def transform_for_inference(dataframe):
    transformed = dataframe.copy()
    scaled_fields = SCALER.transform(transformed[["Amount", "Time"]])
    transformed["scaled_amount"] = scaled_fields[:, 0]
    transformed["scaled_time"] = scaled_fields[:, 1]
    transformed.drop(["Amount", "Time"], axis=1, inplace=True)
    transformed = transformed[FEATURE_ORDER]
    return transformed


def parse_threshold(raw_threshold):
    if raw_threshold in (None, ""):
        return DEFAULT_THRESHOLD, None

    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return None, "Threshold must be a number between 0 and 1."

    if threshold < 0 or threshold > 1:
        return None, "Threshold must be between 0 and 1."

    return threshold, None


def read_uploaded_csv(uploaded_file):
    try:
        uploaded_file.stream.seek(0)
        return pd.read_csv(uploaded_file), None
    except Exception:
        logger.exception("Primary CSV read failed.")

    try:
        uploaded_file.stream.seek(0)
        return pd.read_csv(io.TextIOWrapper(uploaded_file.stream, encoding="utf-8")), None
    except Exception:
        logger.exception("Fallback CSV read failed.")
        return None, "Could not read the CSV file."


def compute_analytics(result_df, probabilities, predictions):
    amounts = result_df["Amount"].values.astype(float)

    bin_edges = [0, 10, 50, 100, 500, 1000, float("inf")]
    bin_labels = ["$0–10", "$10–50", "$50–100", "$100–500", "$500–1K", "$1K+"]
    amount_dist = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (amounts >= lo) & (amounts < hi)
        amount_dist.append({
            "label": bin_labels[i],
            "total": int(mask.sum()),
            "fraud": int((predictions[mask] == 1).sum()),
        })

    prob_dist = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (probabilities >= lo) & (probabilities < hi) if i < 9 else (probabilities >= lo)
        prob_dist.append({"label": f"{i * 10}–{(i + 1) * 10}%", "count": int(mask.sum())})

    n = len(result_df)
    indices = np.sort(np.random.choice(n, min(300, n), replace=False))
    time_col = result_df["Time"].values if "Time" in result_df.columns else np.arange(n)
    time_series = [
        {
            "time": round(float(time_col[idx]), 2),
            "amount": round(float(amounts[idx]), 2),
            "fraud": int(predictions[idx]),
        }
        for idx in indices
    ]

    top_idx = np.argsort(probabilities)[-5:][::-1]
    top_risks = [
        {
            "row": int(idx),
            "amount": round(float(amounts[idx]), 2),
            "probability": round(float(probabilities[idx]) * 100, 1),
        }
        for idx in top_idx
    ]

    return {
        "amount_distribution": amount_dist,
        "probability_distribution": prob_dist,
        "time_series": time_series,
        "top_risks": top_risks,
    }


def run_inference(uploaded_file, threshold=DEFAULT_THRESHOLD):
    if MODEL is None or SCALER is None or FEATURE_ORDER is None:
        return None, None, "Model artifacts not found. Run model.py first.", 500

    raw_df, error = read_uploaded_csv(uploaded_file)
    if error:
        return None, None, error, 400

    missing_columns = validate_columns(raw_df)
    if missing_columns:
        return None, None, f"Missing required columns: {', '.join(missing_columns)}", 400

    try:
        inference_df = transform_for_inference(raw_df[REQUIRED_RAW_COLUMNS])
        probabilities = MODEL.predict_proba(inference_df)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    except Exception:
        logger.exception("Failed during inference.")
        return None, None, "Could not generate predictions.", 500

    result_df = raw_df.copy()
    result_df["fraud_probability"] = (probabilities * 100).round(2)
    result_df["prediction"] = ["Fraud" if pred == 1 else "Legit" for pred in predictions]

    fraud_count = int((predictions == 1).sum())
    total_count = int(len(result_df))
    fraud_rate = round((fraud_count / total_count) * 100, 2) if total_count > 0 else 0.0

    summary = {
        "total_transactions": total_count,
        "fraud_transactions": fraud_count,
        "legit_transactions": total_count - fraud_count,
        "fraud_rate": fraud_rate,
    }
    analytics = compute_analytics(result_df, probabilities, predictions)
    return result_df, summary, analytics, None, 200


@app.get("/")
@login_required
def index():
    assets_ready = MODEL is not None and SCALER is not None and FEATURE_ORDER is not None
    model_info = {"ready": assets_ready}
    if assets_ready:
        try:
            model_info["model_name"] = MODEL.__class__.__name__
            model_info["n_features"] = len(FEATURE_ORDER)
        except Exception:
            model_info["model_name"] = "Unknown"
            model_info["n_features"] = "?"

    return render_template("index.html", assets_ready=assets_ready, model_info=model_info)


@app.get("/generate_sample")
def generate_sample():
    """Generate a synthetic sample CSV for demos. Query param: rows (default 10)."""
    try:
        rows = min(max(int(request.args.get("rows", 10)), 1), MAX_SAMPLE_ROWS)
    except (TypeError, ValueError):
        rows = 10

    logger.info("Generating sample CSV with rows=%s", rows)

    sample_source = "creditcard.csv"
    sample_df = None
    if os.path.exists(sample_source):
        try:
            source_df = pd.read_csv(sample_source)
            sample_df = source_df.drop(columns=["Class"], errors="ignore").sample(
                n=rows,
                replace=len(source_df) < rows,
            )
        except Exception:
            logger.exception("Failed to build sample from creditcard.csv; falling back to synthetic data.")

    if sample_df is None:
        import numpy as np

        cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        data = []
        for i in range(rows):
            time = i
            amount = round(float(abs(np.random.normal(loc=50, scale=120))), 2)
            if np.random.rand() < 0.05:
                amount = round(amount + np.random.rand() * 5000, 2)

            row = [time, amount] + list(np.round(np.random.normal(loc=0.0, scale=1.0, size=28), 5))
            data.append(row)

        sample_df = pd.DataFrame(data, columns=cols)

    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="generated_sample.csv", mimetype="text/csv")


@app.get("/metrics")
def metrics():
    metrics_data = load_metrics()
    if metrics_data is None:
        return jsonify({"available": False, "error": "Metrics file not found. Run model.py first."}), 404

    return jsonify({"available": True, **metrics_data})


@app.post("/reload")
def reload_assets():
    global MODEL, SCALER, FEATURE_ORDER

    MODEL, SCALER, FEATURE_ORDER = load_model_assets()
    assets_ready = MODEL is not None and SCALER is not None and FEATURE_ORDER is not None
    if not assets_ready:
        return jsonify({"ready": False, "error": "Model artifacts not found. Run model.py first."}), 500

    return jsonify(
        {
            "ready": True,
            "model_name": MODEL.__class__.__name__,
            "n_features": len(FEATURE_ORDER),
        }
    )


@app.post("/predict")
def predict_from_csv():
    if "transactions_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded_file = request.files["transactions_file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Please select a CSV file."}), 400

    if not uploaded_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    threshold, error = parse_threshold(request.values.get("threshold", DEFAULT_THRESHOLD))
    if error:
        return jsonify({"error": error}), 400

    logger.info("Prediction request received for %s with threshold=%.2f", uploaded_file.filename, threshold)
    result_df, summary, analytics, error, status_code = run_inference(uploaded_file, threshold)
    if error:
        return jsonify({"error": error}), status_code

    logger.info(
        "Prediction complete: total=%s fraud=%s threshold=%.2f",
        summary["total_transactions"],
        summary["fraud_transactions"],
        threshold,
    )

    response = {
        "summary": summary,
        "analytics": analytics,
        "preview": result_df.head(15).to_dict(orient="records"),
    }
    return jsonify(response)


@app.post("/download")
def download_results():
    if "transactions_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded_file = request.files["transactions_file"]
    if uploaded_file.filename == "" or not uploaded_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Upload a valid CSV file."}), 400

    threshold, error = parse_threshold(request.values.get("threshold", DEFAULT_THRESHOLD))
    if error:
        return jsonify({"error": error}), 400

    result_df, _, _analytics, error, status_code = run_inference(uploaded_file, threshold)
    if error:
        return jsonify({"error": error}), status_code

    output = io.BytesIO()
    result_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="fraud_predictions.csv",
        mimetype="text/csv",
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("index"))

    error = None
    tab = "login"

    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if action == "register":
            tab = "register"
            confirm = request.form.get("confirm_password", "")
            if not username or not password:
                error = "Username and password are required."
            elif len(password) < 6:
                error = "Password must be at least 6 characters."
            elif password != confirm:
                error = "Passwords do not match."
            else:
                users = load_users()
                if username in users:
                    error = "Username already taken."
                else:
                    users[username] = generate_password_hash(password)
                    save_users(users)
                    session["username"] = username
                    return redirect(url_for("index"))
        else:
            users = load_users()
            if username not in users or not check_password_hash(users[username], password):
                error = "Invalid username or password."
            else:
                session["username"] = username
                return redirect(url_for("index"))

    return render_template("login.html", error=error, tab=tab)


@app.get("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)