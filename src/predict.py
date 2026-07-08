"""Prediction utilities for saved classification pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "classification_pipeline.joblib"


def load_model(model_path: Path = MODEL_PATH):
    """Load a saved model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict_single_sample(model, sample: dict[str, float]) -> dict[str, float | int]:
    """Predict a single sample and return class plus probability."""
    input_frame = pd.DataFrame([sample])
    prediction = int(model.predict(input_frame)[0])
    probability = float(model.predict_proba(input_frame)[0][prediction])
    return {"prediction": prediction, "probability": probability}
