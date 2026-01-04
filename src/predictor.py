import json
from pathlib import Path
from typing import Any, Dict

from joblib import load

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "schema.json"


class Predictor:
    """
    Loads the trained sklearn Pipeline (preprocess + model) and provides
    a stable predict_proba_one(record) interface.
    """

    def __init__(self, model_path: Path = MODEL_PATH, schema_path: Path = SCHEMA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}. Run: python src/train.py")
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}. Run: python src/train.py")

        self.model = load(model_path)
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.feature_order = self.schema["all_features_order"]

    def predict_proba_one(self, record: Dict[str, Any]) -> float:
        """
        record: dict containing keys for all features listed in schema.json.
        returns: probability of y=1 (success)
        """
        missing = [k for k in self.feature_order if k not in record]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Build a single-row DataFrame in the exact feature order expected.
        import pandas as pd

        X_one = pd.DataFrame([{k: record[k] for k in self.feature_order}])

        proba = self.model.predict_proba(X_one)[0][1]
        return float(proba)


def predict_proba_one(record: Dict[str, Any]) -> float:
    """
    Convenience function if you don't want to manage a Predictor instance.
    """
    return Predictor().predict_proba_one(record)