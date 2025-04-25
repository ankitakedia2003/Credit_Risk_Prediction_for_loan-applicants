import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load model and preprocessor
model = pickle.load(open("final_model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

st.set_option('deprecation.showPyplotGlobalUse', False)
shap.initjs()

st.title("📊 Credit Risk Prediction App")
st.markdown("Enter loan applicant details to predict creditworthiness. Results include prediction, confidence, and a SHAP-based explanation.")

# User input
age = st.slider("Age", 18, 75, 30, help="Age of the applicant in years")
job = st.selectbox("Job Type", [0, 1, 2, 3], help="Job category (0 = unemployed/unskilled, 3 = highly skilled)")
housing = st.selectbox("Housing", ['own', 'free', 'rent'], help="Housing status")
saving_acc = st.selectbox("Saving Accounts", ['little', 'moderate', 'rich', 'quite rich', 'no_info'], help="Savings account balance level")
checking_acc = st.selectbox("Checking Account", ['little', 'moderate', 'rich', 'no_info'], help="Checking account balance level")
credit_amount = st.number_input("Credit Amount", 100, 20000, 5000, help="Requested loan amount in DM")
duration = st.slider("Loan Duration (Months)", 4, 72, 24, help="Loan repayment duration in months")
purpose = st.selectbox("Purpose", ['radio/TV', 'education', 'car', 'furniture/equipment', 'business'], help="Purpose for the loan")
sex = st.selectbox("Sex", ['male', 'female'], help="Gender of the applicant")

# Create input dataframe
input_dict = {
    'Age': [age],
    'Job': [job],
    'Sex': [sex],
    'Housing': [housing],
    'Saving accounts': [saving_acc],
    'Checking account': [checking_acc],
    'Credit amount': [credit_amount],
    'Duration': [duration],
    'Purpose': [purpose]
}
user_df = pd.DataFrame(input_dict)

# Preprocess and predict
try:
    user_df_processed = preprocessor.transform(user_df)
    pred = model.predict(user_df_processed)[0]
    prob = model.predict_proba(user_df_processed)[0][1]

    st.subheader("🔍 Prediction Result:")
    st.markdown(f"**Credit Risk:** {'🔴 Bad' if pred == 1 else '🟢 Good'}")
    st.markdown(f"**Probability of Bad Credit Risk:** `{prob:.2%}`")

    # Display the custom bar plot (Top SHAP features)
    st.subheader("📊 Top 5 Risk Indicators for Credit Default")
    risk_factors = {
        "Duration (months)": 0.24,
        "Credit amount": 0.21,
        "No Checking Account Info": 0.18,
        "Low Saving Account": 0.16,
        "Loan Purpose: Radio/TV": 0.14
    }

    plt.figure(figsize=(10, 5))
    plt.barh(list(risk_factors.keys())[::-1], list(risk_factors.values())[::-1], color="crimson")
    plt.xlabel("Relative Risk Contribution (SHAP Importance)")
    plt.title("Top 5 Risk Indicators for Credit Default")
    plt.tight_layout()
    st.pyplot(plt)
    
    # SHAP Explanation
    st.subheader("📌 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_df_processed)

    # Display SHAP plots
    if st.checkbox("Show SHAP Force Plot"):
        fig = shap.force_plot(explainer.expected_value, shap_values[0], user_df_processed, matplotlib=True)
        st.pyplot(fig, bbox_inches='tight')

    if st.checkbox("Show SHAP Waterfall Plot"):
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
        st.pyplot(bbox_inches='tight')

except Exception as e:
    st.error(f"⚠️ Something went wrong during processing: {e}")
    st.stop()
