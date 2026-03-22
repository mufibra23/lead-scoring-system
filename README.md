# 🎯 AI-Enhanced Lead Scoring System

**ML-powered lead scoring with XGBoost and SHAP explainability**

An end-to-end machine learning pipeline that predicts which leads are most likely to convert into paying customers, with full explainability for every prediction.

## The Problem

Sales teams waste up to 67% of their time on leads that will never convert. Manual lead qualification is inconsistent, unscalable, and based on gut feeling rather than data.

## The Solution

An XGBoost classifier trained on 9,000+ historical leads that predicts conversion probability with **94% accuracy and 0.978 ROC-AUC**, with SHAP explanations showing exactly *why* each lead scores the way it does.

## Key Results

| Metric | Logistic Regression (Baseline) | XGBoost | Improvement |
|--------|-------------------------------|---------|-------------|
| Accuracy | 82.5% | **93.8%** | +11.3pp |
| Precision | 79.9% | **92.2%** | +12.3pp |
| Recall | 73.0% | **91.6%** | +18.6pp |
| ROC-AUC | 0.896 | **0.979** | +8.3pp |

## Top Predictors (SHAP Analysis)

1. **Tags** — Pipeline stage is the strongest signal (e.g., "Will revert after reading email" = 97% conversion vs "Ringing" = 3%)
2. **Time on Website** — Converted leads spend 2.2x more time browsing
3. **Last Activity** — SMS-contacted leads convert at 63% vs 9% for chat
4. **Occupation** — Working professionals convert at 92% vs 44% for unemployed
5. **Lead Quality** — Internal quality rating strongly correlates with outcomes

## Dashboard

Interactive Streamlit dashboard with 4 tabs:

- **Lead Scoreboard** — All leads ranked by conversion probability (Hot/Warm/Cold)
- **Lead Explainer** — Select any lead → see SHAP waterfall showing why it scored that way
- **Model Performance** — Confusion matrix, ROC curves, baseline comparison
- **Feature Insights** — Global SHAP importance + actionable recommendations

**Live Demo:** [lead-scoring-system.streamlit.app](https://lead-scoring-system.streamlit.app)

## Tech Stack

- **ML Model:** XGBoost (gradient boosting classifier)
- **Explainability:** SHAP (TreeExplainer for fast, exact Shapley values)
- **Data Processing:** pandas, NumPy, scikit-learn
- **Dashboard:** Streamlit + Plotly
- **AI Insights:** Google Gemini API
- **Dataset:** [Kaggle Lead Scoring Dataset](https://kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset) (~9,240 leads)

## Project Structure

```
lead-scoring-system/
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── .env.example
├── data/
│   ├── Lead_Scoring.csv        # Kaggle dataset
│   └── Leads_Data_Dictionary.xlsx
├── notebooks/
│   └── 01_eda.py               # Exploratory data analysis
├── src/
│   ├── data_preprocessing.py   # Clean, encode, feature engineering
│   └── model_training.py       # Train models, compute SHAP, save artifacts
├── models/
│   └── xgboost_model.pkl       # Trained model
├── shap_cache/
│   ├── shap_values.pkl         # Pre-computed SHAP values
│   ├── feature_importance.csv  # Global feature rankings
│   ├── test_predictions.csv    # Test set with scores
│   └── metrics.pkl             # Model evaluation metrics
└── docs/
    └── architecture.md
```

## Run Locally

```bash
# Clone the repo
git clone https://github.com/mufibra23/lead-scoring-system.git
cd lead-scoring-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run EDA
python notebooks/01_eda.py

# Train models (generates all artifacts)
python src/model_training.py

# Launch dashboard
streamlit run app.py
```

## How It Works

1. **Data Preprocessing** — Cleans 9,240 leads: handles missing values, encodes categoricals, drops zero-information columns, replaces 'Select' placeholders
2. **Feature Engineering** — Creates 7 new features: engagement score, referral flag, professional flag, activity level, website engagement buckets, lead quality numeric, source conversion rate
3. **Model Training** — Trains XGBoost with 200 estimators, depth 5, 0.8 subsample. Validated with 5-fold stratified cross-validation (CV ROC-AUC: 0.979)
4. **SHAP Explainability** — TreeExplainer computes exact Shapley values for every prediction. Global importance ranks features; local waterfall plots explain individual leads
5. **Dashboard** — Streamlit app loads pre-computed artifacts for instant interactivity

## Dataset

This project uses the [Kaggle Lead Scoring Dataset](https://kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset) for demonstration. The dataset contains ~9,240 leads from an online education company with features like lead source, website engagement, activity history, and conversion outcome.

**Note:** This is a portfolio demonstration project. For production use, train on client-specific CRM data and implement proper model monitoring, retraining pipelines, and A/B testing of lead prioritization strategies.

## Author

**Muhammad Fariz Ibrahim** — AI Marketing Analytics & Data Science

- GitHub: [@mufibra23](https://github.com/mufibra23)
- LinkedIn: [Connect](https://linkedin.com/in/mufibra)
