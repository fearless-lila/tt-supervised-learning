import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from joblib import load

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "schema.json"


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class Predictor:
    """
    Loads the trained sklearn Pipeline (preprocess + regressor) and provides
    a stable predict_rate_one(record) interface.
    """

    def __init__(self, model_path: Path = MODEL_PATH, schema_path: Path = SCHEMA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}. Run: python src/train.py")
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}. Run: python src/train.py")

        self.model = load(model_path)
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.feature_order = self.schema["all_features_order"]

    def predict_rate_one(self, record: Dict[str, Any]) -> float:
        missing = [k for k in self.feature_order if k not in record]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        import pandas as pd

        X_one = pd.DataFrame([{k: record[k] for k in self.feature_order}])
        y_pred = self.model.predict(X_one)[0]
        return _clip01(y_pred)


def predict_rate_one(record: Dict[str, Any]) -> float:
    return Predictor().predict_rate_one(record)
