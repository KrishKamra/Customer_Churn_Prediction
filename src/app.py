import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ChurnAI | SaaS Insights",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DARK THEME DEFINITION ---
theme = {
    "bg_primary": "#1a1c23",
    "bg_secondary": "#252a33",
    "text_primary": "#f1f5f9",
    "text_secondary": "#cbd5e1",
    "border_color": "#404856",
    "card_shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3)",
    "card_hover_shadow": "0 0 20px rgba(59, 130, 246, 0.2)",
    "accent": "#3b82f6",
    "success": "#34d399",
    "error": "#f87171",
}

# --- CUSTOM UI STYLING ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {theme['bg_primary']}; }}
    h1, h2, h3, p, span, label {{ color: {theme['text_primary']} !important; }}
    .css-card {{
        background-color: {theme['bg_secondary']};
        padding: 25px;
        border-radius: 15px;
        box-shadow: {theme['card_shadow']};
        margin-bottom: 1.5rem;
        border: 1px solid {theme['border_color']};
        transition: all 0.3s ease;
    }}
    .css-card:hover {{
        transform: translateY(-4px);
        border-color: {theme['accent']};
        box-shadow: {theme['card_hover_shadow']};
    }}
    /* Standard Button Styling */
    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        background-color: {theme['accent']};
        color: white !important;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
    }}
    /* Specific styling for the Download Button to make it stand out */
    .stDownloadButton>button {{
        background-color: transparent !important;
        border: 2px solid {theme['success']} !important;
        color: {theme['success']} !important;
    }}
    .stDownloadButton>button:hover {{
        background-color: {theme['success']} !important;
        color: white !important;
    }}
    .streamlit-expanderHeader {{
        background-color: {theme['bg_secondary']} !important;
        border-radius: 8px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- ASSETS & DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'churn_model_v1.pkl')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("ChurnAI")
    st.markdown("---")
    nav = st.radio("Navigation", ["🔍 Predictor", "📊 Analytics", "📖 Documentation"])
    st.markdown("---")
    st.info("✅ Log-transformed model active.")

# --- MAIN DASHBOARD: PREDICTOR ---
if nav == "🔍 Predictor":
    st.header("Customer Risk Analysis")
    
    col_input, col_output = st.columns([2, 1], gap="medium")

    with col_input:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("📋 Customer Profile Details")
        
        with st.form("churn_form", clear_on_submit=False):
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                tenure = st.slider("Tenure (Months)", 0, 72, 12)
                contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
            with form_col2:
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
                total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)
            
            with st.expander("🛠️ Additional Customer Demographics & Services"):
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    gender = st.radio("Gender", ["Female", "Male"])
                    senior = st.checkbox("Senior Citizen")
                    partner = st.checkbox("Has Partner")
                    dependents = st.checkbox("Has Dependents")
                    phone = st.selectbox("Phone Service", ["Yes", "No"])
                    lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                with adv_col2:
                    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                    billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

            submit = st.form_submit_button("🚀 Run Risk Assessment")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_output:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("📊 Risk Assessment")
        
        if submit:
            full_input_dict = {
                'gender': gender, 'SeniorCitizen': 1 if senior else 0,
                'Partner': 'Yes' if partner else 'No', 'Dependents': 'Yes' if dependents else 'No',
                'tenure': tenure, 'PhoneService': phone, 'MultipleLines': lines,
                'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
                'DeviceProtection': protection, 'TechSupport': support, 'StreamingTV': tv,
                'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': billing,
                'PaymentMethod': payment, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
            }
            
            input_df = pd.DataFrame([full_input_dict])
            prob = model.predict_proba(input_df)[0][1]

            st.metric("Churn Probability", f"{prob:.2%}")
            
            risk_level = "CRITICAL" if prob > 0.5 else "MODERATE" if prob > 0.3 else "LOW"
            if risk_level == "CRITICAL":
                st.error(f"🚨 {risk_level}: HIGH RISK")
            elif risk_level == "MODERATE":
                st.warning(f"⚠️ {risk_level} RISK")
            else:
                st.success(f"✅ {risk_level} RISK")

            # --- GENERATE DOWNLOADABLE REPORT ---
            report_text = f"""
            CHURNAI PREDICTION REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            -------------------------------------------
            CUSTOMER METRICS:
            - Tenure: {tenure} months
            - Contract: {contract}
            - Monthly Charges: ${monthly_charges}
            - Total Charges: ${total_charges}
            
            PREDICTION RESULTS:
            - Churn Probability: {prob:.2%}
            - Risk Category: {risk_level}
            -------------------------------------------
            CONFIDENTIAL - Internal Use Only
            """
            
            st.download_button(
                label="📥 Download Risk Report",
                data=report_text,
                file_name=f"Churn_Report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )
        else:
            st.info("Awaiting analysis...")
        st.markdown('</div>', unsafe_allow_html=True)

# --- ANALYTICS & DOCS ---
elif nav == "📊 Analytics":
    st.header("Business Performance Insights")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.metric("Recall Score", "~80%", "+2%")
        st.markdown('</div>', unsafe_allow_html=True)
    # ... (Rest of analytics code remains same)

elif nav == "📖 Documentation":
    st.header("Model Documentation")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.write("Algorithm: Logistic Regression | Balanced Weights")
    st.write("Validation: ROC-AUC 0.85")
    st.markdown('</div>', unsafe_allow_html=True)