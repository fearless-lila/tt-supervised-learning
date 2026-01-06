from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/supervised_dataset.csv")
MODEL_PATH = Path("artifacts/model.joblib")

LABEL_COL = "y_rate"
FEATURE_COLS = ["drill_id", "focus", "time_of_day", "difficulty", "session_minutes"]


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


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
    y_true = df[LABEL_COL].astype(float).to_numpy()

    model = load(MODEL_PATH)

    # Model predicts a real number; we clip to [0,1] because target is a rate
    y_pred = clip01(model.predict(X))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    # Brier score is just MSE for probabilistic targets
    brier = mse

    print("=== Evaluation (Regression on y_rate) ===")
    print(f"Rows: {len(df)}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Brier (MSE): {brier:.4f}")

    print("\nSample predictions:")
    preview = df.copy()
    preview["y_pred"] = y_pred
    cols = FEATURE_COLS + [LABEL_COL, "y_pred"]
    print(preview[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
