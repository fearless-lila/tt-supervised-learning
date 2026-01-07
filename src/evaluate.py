from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/supervised_dataset.csv")
MODEL_PATH = Path("artifacts/model.joblib")

LABEL_COL = "y_rate"
FEATURE_COLS = ["drill_id", "focus", "time_of_day", "skill_level", "fatigue", "difficulty", "session_minutes"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


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
    y = df[LABEL_COL].astype(float)

    # Evaluate only on a held-out split (no leakage).
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = load(MODEL_PATH)

    y_pred = clip01(model.predict(X_val))
    y_true = y_val.to_numpy()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    brier = mse

    print("=== Evaluation (Regression on y_rate) ===")
    print(f"Rows total: {len(df)}")
    print(f"Val rows:   {len(X_val)} (test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Brier (MSE): {brier:.4f}")

    print("\nSample predictions (validation set):")
    preview = X_val.copy()
    preview["y_rate"] = y_val.values
    preview["y_pred"] = y_pred
    print(preview.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
