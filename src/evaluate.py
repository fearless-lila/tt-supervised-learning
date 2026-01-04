from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

DATA_PATH = Path("data/supervised_dataset.csv")
MODEL_PATH = Path("artifacts/model.joblib")

LABEL_COL = "y"
FEATURE_COLS = ["drill_id", "focus", "time_of_day", "difficulty", "session_minutes"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run: python src/train.py")

    df = pd.read_csv(DATA_PATH)

    missing = set(FEATURE_COLS + [LABEL_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL].astype(int)

    model = load(MODEL_PATH)

    # Probabilities for the positive class
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y, pred)

    # ROC-AUC needs both classes present
    roc_auc = roc_auc_score(y, proba) if y.nunique() == 2 else None
    pr_auc = average_precision_score(y, proba) if y.nunique() == 2 else None

    cm = confusion_matrix(y, pred)

    print("=== Evaluation ===")
    print(f"Rows: {len(df)}")
    print(f"Accuracy @0.5: {acc:.3f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.3f}")
        print(f"PR-AUC (Average Precision): {pr_auc:.3f}")
    else:
        print("ROC-AUC / PR-AUC: not available (only one class in y)")

    print("\nConfusion matrix [ [TN FP], [FN TP] ]:")
    print(cm)

    # Optional: show a few example predictions
    print("\nSample predictions:")
    preview = df.copy()
    preview["p_success"] = proba
    preview["pred"] = pred
    cols = FEATURE_COLS + [LABEL_COL, "p_success", "pred"]
    print(preview[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
