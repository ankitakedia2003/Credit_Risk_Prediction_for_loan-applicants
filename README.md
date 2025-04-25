
# Credit Risk Prediction for Loan Applicants

This project uses machine learning to predict whether a loan applicant poses a **good or bad credit risk**. Built with a focus on accuracy, explainability, and user accessibility, the model is deployed as an interactive **Streamlit web app** that offers real-time predictions with SHAP-based explanations.

---

## 🔍 Problem Statement

Financial institutions must assess credit risk accurately to reduce defaults and improve lending efficiency. The original German Credit dataset lacked clearly defined risk labels, making model evaluation difficult. To address this, a **labeled version with binary outcomes** (good/bad credit risk) was used, allowing for better classification and clarity.

---

## 📁 Project Structure

```
CreditRiskApp/
├── app.py                    # Streamlit UI & prediction logic
├── final_model.pkl           # Trained LightGBM model (with early stopping)
├── preprocessor.pkl          # Preprocessing pipeline (scaler + encoder)
├── model_development.ipynb   # Full ML pipeline (EDA, modeling, SHAP)
├── requirements.txt          # Required Python packages
└── .streamlit/
    └── config.toml           # Custom theme for the app
```

---

## 📓 Notebook Highlights: `model_development.ipynb`

- Exploratory Data Analysis (EDA)
- Handling missing values and outliers
- Categorical encoding and feature engineering
- Model comparisons: Logistic Regression, Random Forest, XGBoost, LightGBM
- Manual early stopping with LightGBM
- Performance metrics: Accuracy, F1-Score, ROC-AUC
- SHAP-based interpretation for explainability
- Exporting final model and preprocessor

---

## 🧠 Features

- Interactive UI to input applicant details  
- Instant classification: **Good** or **Bad** credit risk  
- Probability confidence score  
- SHAP visualizations:
  - Force plot
  - Waterfall plot  
- Top 5 feature importance chart  
- Transparent, explainable decisions for credit evaluation

---

## 🎨 App Theme

Custom theme set via `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#A78BFA"
backgroundColor = "#FDF6FD"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#4B5563"
font = "sans serif"
```

---

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/ankitakedia2003/Credit_Risk_Prediction_for_loan-applicants.git
cd Credit_Risk_Prediction_for_loan-applicants
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

```bash
streamlit run app.py
```

Make sure `final_model.pkl`, `preprocessor.pkl`, and `.streamlit/config.toml` are present in the directory.

---

## 📈 Final Model & Results

- **Model**: LightGBM with early stopping  
- **Accuracy**: ~78%  
- **F1-Score**: ~0.76  
- **Validation**: Stratified train-test split with cross-validation  
- **Top Features**: Loan duration, credit amount, account balances, job category, loan purpose  
- **Interpretability**: SHAP integration for full model transparency

## 🌐 Live Demo

- **Streamlit App**: [creditriskpredictionforloan-applicant.streamlit.app](https://creditriskpredictionforloan-applicant.streamlit.app)  

## ✨ Built With

- [LightGBM](https://lightgbm.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://shap.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

