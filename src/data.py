"""Data loading utilities for the classification project."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "target"


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the Breast Cancer Wisconsin dataset as features and target."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target
    target_names = list(dataset.target_names)
    return features, target, target_names


def build_train_test_split(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return a stratified train/test split."""
    features, target, target_names = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return x_train, x_test, y_train, y_test, target_names
