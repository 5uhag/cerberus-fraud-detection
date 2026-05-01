"""
Generate 10 demo CSV datasets (50 rows each) with varying fraud rates.
Outputs to: static/samples/demo_dataset_01.csv ... demo_dataset_10.csv
Columns: Time,Amount,V1..V28,Class
"""
import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'samples')
os.makedirs(OUT_DIR, exist_ok=True)

n_files = 10
rows = 50
# fraud rates from 0% up to 30%
fraud_rates = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]

for idx in range(n_files):
    rate = fraud_rates[idx]
    n_fraud = int(round(rate * rows))
    # ensure deterministic-ish names
    fname = os.path.join(OUT_DIR, f'demo_dataset_{idx+1:02d}.csv')

    data = []
    # pick fraud indices
    fraud_indices = set(np.random.choice(range(rows), size=n_fraud, replace=False)) if n_fraud>0 else set()

    for i in range(rows):
        is_fraud = 1 if i in fraud_indices else 0
        # Time: incremental
        time = i
        # Amount: frauds have larger mean
        if is_fraud:
            amount = abs(np.random.normal(loc=800.0, scale=500.0))
        else:
            amount = abs(np.random.normal(loc=50.0, scale=80.0))
        amount = round(float(amount), 2)

        # Generate V1..V28: random normal; shift some features for frauds slightly
        vs = np.random.normal(loc=0.0, scale=1.0, size=28)
        if is_fraud:
            vs[:3] += np.random.normal(loc=1.8, scale=0.6, size=3)  # small shift in first 3 PCs

        row = [time, amount] + [float(round(x, 5)) for x in vs] + [is_fraud]
        data.append(row)

    cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1,29)] + ['Class']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(fname, index=False)
    print(f'Wrote {fname} (rows={len(df)}, fraud_rate={rate}, fraud_count={n_fraud})')
print('Done')
