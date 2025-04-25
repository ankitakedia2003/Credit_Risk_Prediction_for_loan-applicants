# 📊 Credit Risk Prediction App

Welcome to the **Credit Risk Prediction App** — a Streamlit-based web application that predicts whether a loan applicant is likely to be a **good** or **bad** credit risk, using the German Credit dataset.

🌐 **Live App**: [Launch Here](https://creditriskpredictionforloan-applicant.streamlit.app/)

---

## 🚀 Quick Start

### 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
📁 Project Structure
graphql
Copy
Edit
CreditRiskApp/
├── app.py                    # Streamlit UI & prediction logic
├── final_model.pkl           # Trained LightGBM model (with early stopping)
├── preprocessor.pkl          # Preprocessing pipeline (scaler + encoder)
├── model_development.ipynb   # Jupyter Notebook for full ML pipeline
├── requirements.txt          # List of dependencies
└── .streamlit/
    └── config.toml           # Theme settings for Streamlit UI
📓 Notebook
All steps from data preprocessing to final model evaluation are documented in the Jupyter notebook:

👉 model_development.ipynb

Feature selection & encoding

Manual early stopping with LightGBM

Performance evaluation (accuracy, ROC, F1)

SHAP-based model interpretability

Final model export as .pkl

🧠 Features
Interactive UI to enter applicant details

Instant predictions (Good or Bad risk)

Probability confidence for each prediction

SHAP-based feature explanation (force & waterfall plot)

🎨 App Theme
This app uses a soft pastel theme configured via .streamlit/config.toml:

toml
Copy
Edit
[theme]
primaryColor = "#A78BFA"
backgroundColor = "#FDF6FD"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#4B5563"
font = "sans serif"
🌐 Deployment
You can deploy this app to Streamlit Cloud by connecting your GitHub repo. Make sure to include:

app.py

requirements.txt

final_model.pkl

preprocessor.pkl

.streamlit/config.toml

✨ Credits
Built with using:

LightGBM

scikit-learn

SHAP

Streamlit
