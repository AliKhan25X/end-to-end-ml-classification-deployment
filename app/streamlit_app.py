"""Streamlit app for interactive ML classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.predict import MODEL_PATH, load_model, predict_single_sample  # noqa: E402


st.set_page_config(
    page_title="ML Classification Deployment",
    page_icon="ML",
    layout="wide",
)


@st.cache_resource
def get_model():
    """Load the saved model, or train a lightweight model for demo use."""
    if MODEL_PATH.exists():
        return load_model()

    features, target, _ = load_dataset()
    model = build_model()
    model.fit(features, target)
    return model


@st.cache_data
def get_feature_template() -> pd.DataFrame:
    features, _, _ = load_dataset()
    return features


def main() -> None:
    st.title("End-to-End ML Classification Deployment")
    st.write(
        "Interactive machine learning app for binary classification using a trained "
        "Random Forest pipeline. This is an educational portfolio project, not a "
        "medical diagnostic tool."
    )

    model = get_model()
    features = get_feature_template()

    st.sidebar.header("Input Features")
    st.sidebar.write("Adjust feature values and run prediction.")

    sample = {}
    for column in features.columns:
        min_value = float(features[column].min())
        max_value = float(features[column].max())
        mean_value = float(features[column].mean())
        sample[column] = st.sidebar.slider(
            label=column,
            min_value=min_value,
            max_value=max_value,
            value=mean_value,
        )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Prediction")
        result = predict_single_sample(model, sample)
        label = "Benign" if result["prediction"] == 1 else "Malignant"
        st.metric("Predicted Class", label)
        st.metric("Model Confidence", f"{result['probability']:.2%}")

    with right:
        st.subheader("Sample Preview")
        st.dataframe(pd.DataFrame([sample]), use_container_width=True)

    st.subheader("Project Notes")
    st.write(
        "This app demonstrates deployment readiness: preprocessing, model inference, "
        "feature input controls, and interpretable output formatting. The next "
        "research extension would be explainability with SHAP or LIME."
    )


if __name__ == "__main__":
    main()
