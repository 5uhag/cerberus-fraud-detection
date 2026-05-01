# Cerberus — Credit Card Fraud Detection

A machine learning web application that scores credit card transactions for fraud risk. Upload a CSV, get instant predictions, analytics charts, and a downloadable results report.

**Live:** https://cerberus-fraud-detection.onrender.com

---

## How It Works

Cerberus uses a **Random Forest classifier** trained on 284,807 real credit card transactions. The training pipeline applies **SMOTE** (Synthetic Minority Oversampling) to handle the extreme class imbalance inherent in fraud data (~0.17% fraud rate), then serializes the trained model for low-latency inference.

**Model performance on held-out test set:**

| Metric | Score |
|--------|-------|
| Precision (fraud) | 0.42 |
| Recall (fraud) | 0.85 |
| F1 Score (fraud) | 0.56 |
| Overall Accuracy | 99.9% |

Recall is prioritized over precision — the model is tuned to catch as many fraud cases as possible, accepting some false positives as the trade-off.

---

## Features

- **CSV upload** with drag-and-drop — accepts any file with `Time`, `Amount`, and `V1`–`V28` columns
- **Configurable threshold** — adjust the fraud probability cutoff (default 0.5) to balance sensitivity vs specificity
- **Analytics dashboard** — four Chart.js charts generated on every prediction:
  - Fraud vs Legit breakdown (donut)
  - Transaction amount distribution (stacked bar)
  - Fraud probability distribution (bar)
  - Amount over time (scatter)
- **Top 5 highest-risk transactions** highlighted per upload
- **Downloadable results** — full CSV with `fraud_probability` and `prediction` columns appended
- **Model metrics panel** — live precision, recall, F1, and confusion matrix from the last training run

---

## Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, imbalanced-learn (SMOTE) |
| Backend | Python, Flask, gunicorn |
| Data | pandas, numpy, joblib |
| Frontend | Vanilla JS, Chart.js, CSS Grid |

---

## Running Locally

**Requirements:** Python 3.8+, and `creditcard.csv` from the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) placed in the project root.

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates .pkl artifacts and metrics.json)
python model.py

# Start the server
python app.py
```

App runs at `http://localhost:5000`.

---

## Project Structure

```
cerberus-fraud-detection/
├── app.py               # Flask app — all routes and inference logic
├── model.py             # Training pipeline — run once to generate artifacts
├── requirements.txt
├── static/
│   ├── app.js           # Frontend logic and Chart.js rendering
│   ├── styles.css
│   └── samples/         # Sample CSVs for testing
├── templates/
│   └── index.html
└── scripts/
    └── generate_demo_datasets.py   # Generates synthetic demo CSVs
```

---

## Dataset

Training data: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB Machine Learning Group on Kaggle. Features V1–V28 are PCA-transformed for confidentiality. `creditcard.csv` is not included in this repository.
