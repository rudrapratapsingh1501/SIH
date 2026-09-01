"""
Train the fusion logistic regression model.

This prototype ships with hand-set default weights (see ml/fusion.py) so
the API works out of the box. Run this script to fit real coefficients
instead, once you have labeled data.

Usage:
    python train_fusion.py                # trains on synthetic demo data
    python train_fusion.py my_data.csv    # trains on your own CSV with
                                           # columns: text_score,prosody_score,label

`label` should be 1 for distress/at-risk, 0 for not.
"""

import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.fusion import MODEL_PATH


def make_synthetic_data(n=2000, seed=42):
    """
    Purely illustrative synthetic data so the pipeline is runnable
    end-to-end before real labeled data exists. Replace with a real
    dataset before drawing any conclusions from this model.
    """
    rng = np.random.default_rng(seed)
    text_score = rng.beta(2, 3, n)
    prosody_score = rng.beta(2, 3, n)
    noise = rng.normal(0, 0.08, n)
    combined = 0.55 * text_score + 0.45 * prosody_score + noise
    label = (combined > 0.5).astype(int)
    X = np.column_stack([text_score, prosody_score])
    return X, label


def main():
    if len(sys.argv) > 1:
        import pandas as pd

        df = pd.read_csv(sys.argv[1])
        X = df[["text_score", "prosody_score"]].values
        y = df["label"].values
        print(f"Loaded {len(df)} labeled rows from {sys.argv[1]}")
    else:
        X, y = make_synthetic_data()
        print("No CSV given — training on synthetic demo data (2000 rows).")
        print("Replace with real labeled data before trusting this model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")
    print(f"Coefficients (text, prosody): {model.coef_[0]}")
    print(f"Intercept: {model.intercept_[0]}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved fusion model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
