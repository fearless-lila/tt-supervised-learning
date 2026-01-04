import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/supervised_dataset.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "schema.json"

LABEL_COL = "y"
CATEGORICAL_COLS = ["drill_id", "focus", "time_of_day"]
NUMERIC_COLS = ["difficulty", "session_minutes"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = set(CATEGORICAL_COLS + NUMERIC_COLS + [LABEL_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[LABEL_COL].astype(int)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipeline, MODEL_PATH)

    schema = {
        "label": LABEL_COL,
        "categorical_features": CATEGORICAL_COLS,
        "numeric_features": NUMERIC_COLS,
        "all_features_order": CATEGORICAL_COLS + NUMERIC_COLS,
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Trained and saved model to: {MODEL_PATH}")
    print(f"Saved schema to: {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
