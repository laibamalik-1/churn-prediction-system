# Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn risk and explains *why* each prediction was made, built with a full pipeline from raw data to a deployed, interactive web app.

🔗 **Live App**: [churn-predictionsystem.streamlit.app](https://churn-predictionsystem.streamlit.app/)

## Overview

This isn't just a trained model — it's a complete system:
- Data cleaning and feature engineering pipeline
- A trained, evaluated classification model
- Model explainability using SHAP
- An interactive web dashboard for real-time predictions
- Public deployment

## Problem

Telecom companies lose revenue when customers cancel their service ("churn"). Retaining an existing customer is far cheaper than acquiring a new one, so predicting *which* customers are at risk — and *why* — lets a business intervene before it's too late.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — 7,043 customers, 20 features including demographics, account information, and services subscribed to.

## Approach

1. **Data cleaning**: handled missing values in `TotalCharges`, encoded categorical variables (binary mapping + one-hot encoding)
2. **Handling class imbalance**: the dataset is ~73% non-churn / ~27% churn — addressed using `class_weight='balanced'` rather than relying on raw accuracy
3. **Modeling**: compared Logistic Regression and Random Forest; selected a balanced Logistic Regression model based on churn-class recall, since missing an at-risk customer is costlier than a false alarm
4. **Evaluation**: ROC-AUC of 0.84, with 78% recall on the churn class
5. **Explainability**: SHAP values computed per-prediction, surfacing the top factors driving each individual churn risk score
6. **Deployment**: packaged into a single Streamlit app and deployed on Streamlit Community Cloud

## Tech Stack

- **Python**, **pandas**, **numpy** — data handling
- **scikit-learn** — modeling and evaluation
- **SHAP** — model explainability
- **Streamlit** — interactive dashboard
- **Git/GitHub** — version control
- **Streamlit Community Cloud** — deployment

## Key Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.842 |
| Churn class recall | 0.78 |
| Churn class precision | 0.50 |

## What the App Does

Enter a customer's details (tenure, contract type, services, charges, etc.) and get:
- A churn risk prediction (Yes/No) with probability
- The top 5 factors driving that specific prediction, with direction (increases/decreases risk)

## Project Structure

```
churn-prediction-system/
├── dashboard/
│   ├── app.py              # Streamlit app (model + UI, self-contained)
│   ├── requirements.txt
│   └── model/               # trained model, scaler, column structure
├── api/
│   └── main.py              # FastAPI backend (for local/API-based use)
├── model/                   # original saved model artifacts
└── requirements.txt
```

## Running Locally

```bash
git clone https://github.com/laibamalik-1/churn-prediction-system.git
cd churn-prediction-system/dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Author

Laiba Malik
