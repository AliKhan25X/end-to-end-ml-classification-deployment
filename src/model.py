"""Model factory and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    classification_report: str


def build_model(random_state: int = 42) -> Pipeline:
    """Create a reproducible preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str],
) -> EvaluationResult:
    """Evaluate the trained model on held-out test data."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return EvaluationResult(
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions),
        recall=recall_score(y_test, predictions),
        f1=f1_score(y_test, predictions),
        roc_auc=roc_auc_score(y_test, probabilities),
        confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
        classification_report=classification_report(
            y_test,
            predictions,
            target_names=target_names,
        ),
    )
