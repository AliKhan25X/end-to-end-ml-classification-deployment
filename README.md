# End-to-End ML Classification Deployment

An end-to-end machine learning deployment project that demonstrates a complete classification workflow: data loading, preprocessing, model training, evaluation, artifact saving, and an interactive Streamlit web app for real-time predictions.

This project is designed as a portfolio-quality example for data science, machine learning, and PhD applications. It uses the Breast Cancer Wisconsin benchmark dataset from scikit-learn as a reproducible classification task.

> This project is for educational and research portfolio purposes only. It is not a medical diagnostic tool.

## Project Highlights

- Complete ML pipeline using `scikit-learn`
- Clean preprocessing with `StandardScaler`
- Model training with `RandomForestClassifier`
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
- Saved model artifact using `joblib`
- Interactive Streamlit deployment app
- Reproducible project structure for GitHub and portfolio review

## Problem Statement

The goal is to classify whether a tumor sample is benign or malignant based on numerical diagnostic features. The project focuses on building a reliable classification pipeline and deploying it through a lightweight web interface.

## Repository Structure

```text
end-to-end-ml-classification-deployment/
├── app/
│   └── streamlit_app.py
├── models/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── reports/
│   └── model_report.md
├── src/
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib / Seaborn

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/AliKhan25X/end-to-end-ml-classification-deployment.git
cd end-to-end-ml-classification-deployment
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python -m src.train
```

5. Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Model Workflow

1. Load the Breast Cancer Wisconsin dataset
2. Split data into train/test sets
3. Scale features using `StandardScaler`
4. Train a Random Forest classifier
5. Evaluate classification performance
6. Save the trained pipeline to `models/classification_pipeline.joblib`
7. Load the model in Streamlit for live predictions

## Expected Outputs

After training, the script prints:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

It also writes a Markdown report to `reports/model_report.md`.

## Why This Project Matters

This project demonstrates the full machine learning lifecycle, not only model training. It shows the ability to organize code, evaluate models correctly, save reusable artifacts, and deploy a simple interactive application.

For PhD and research applications, it also provides a foundation for more advanced extensions such as:

- Explainable AI with SHAP or LIME
- Deep learning classifiers
- Data augmentation experiments
- Model comparison studies
- Deployment with Docker or cloud hosting

## Future Improvements

- Add SHAP-based explainability
- Add model comparison between Logistic Regression, SVM, Random Forest, and XGBoost
- Add Docker deployment
- Add CI checks with GitHub Actions
- Extend to deep learning classification experiments

## Author

Ali Khan  
Master's Student, Data Science / Control Science and Computer Engineering  
North China Electric Power University, Beijing, China
