"""Train and save the classification pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.data import build_train_test_split
from src.model import build_model, evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "classification_pipeline.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_report.md"


def write_report(metrics) -> None:
    """Write model metrics to a Markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Model Evaluation Report

## Metrics

| Metric | Score |
|---|---:|
| Accuracy | {metrics.accuracy:.4f} |
| Precision | {metrics.precision:.4f} |
| Recall | {metrics.recall:.4f} |
| F1-score | {metrics.f1:.4f} |
| ROC-AUC | {metrics.roc_auc:.4f} |

## Confusion Matrix

```text
{metrics.confusion_matrix}
```

## Classification Report

```text
{metrics.classification_report}
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def train() -> None:
    """Train, evaluate, and save the ML pipeline."""
    x_train, x_test, y_train, y_test, target_names = build_train_test_split()
    model = build_model()
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test, target_names)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    write_report(metrics)

    print("Model training complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"F1-score: {metrics.f1:.4f}")
    print(f"ROC-AUC: {metrics.roc_auc:.4f}")


if __name__ == "__main__":
    train()
