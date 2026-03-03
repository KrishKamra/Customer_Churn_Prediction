import streamlit as st
import pandas as pd
import joblib
import os

# Setup relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'churn_model_v1.pkl')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📡 Telco Churn Prediction Portal")
st.markdown("Enter customer details to assess churn risk.")

with st.form("churn_form"):
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 9000.0, 500.0)
    
    if st.form_submit_button("Predict"):
        # 1. Create a dictionary with ALL features used during training
        # We use common/default values for the features not in our UI form
        full_input_dict = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract,
        'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    # 2. Convert to DataFrame
    input_df = pd.DataFrame([full_input_dict])
    
    # 3. Predict using the pipeline
    # The pipeline will handle the encoding/scaling for ALL these columns
    prob = model.predict_proba(input_df)[0][1]
    
    st.metric("Churn Probability", f"{prob:.2%}")
    if prob > 0.5:
        st.error("High Risk of Churn")
    else:
        st.success("Low Risk of Churn")