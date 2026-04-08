import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ChurnAI | SaaS Insights",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DARK THEME DEFINITION ---
# Locked to dark mode for a professional SaaS aesthetic
theme = {
    "bg_primary": "#1a1c23",
    "bg_secondary": "#252a33",
    "text_primary": "#f1f5f9",
    "text_secondary": "#cbd5e1",
    "border_color": "#404856",
    "card_shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3)",
    "card_hover_shadow": "0 0 20px rgba(59, 130, 246, 0.2)", # Subtle blue glow
    "accent": "#3b82f6",
    "success": "#34d399",
    "error": "#f87171",
}

# --- CUSTOM UI STYLING ---
st.markdown(f"""
    <style>
    /* Global Styles */
    .stApp {{ background-color: {theme['bg_primary']}; }}
    h1, h2, h3, p, span, label {{ color: {theme['text_primary']} !important; }}
    
    /* CSS Card Container with Hover Color Effect */
    .css-card {{
        background-color: {theme['bg_secondary']};
        padding: 25px;
        border-radius: 15px;
        box-shadow: {theme['card_shadow']};
        margin-bottom: 1.5rem;
        border: 1px solid {theme['border_color']};
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }}
    
    .css-card:hover {{
        transform: translateY(-4px);
        border-color: {theme['accent']};
        box-shadow: {theme['card_hover_shadow']};
    }}

    /* Metric Boxes with Hover Glow */
    .metric-container {{
        background-color: {theme['bg_secondary']};
        padding: 20px;
        border-radius: 12px;
        border: 1px solid {theme['border_color']};
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }}
    
    .metric-container:hover {{
        border-color: {theme['success']};
        background-color: #2d333d;
    }}

    /* Sidebar Customization */
    [data-testid="stSidebar"] {{
        background-color: {theme['bg_secondary']};
        border-right: 1px solid {theme['border_color']};
    }}

    /* Button Styling */
    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        background-color: {theme['accent']};
        color: white !important;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
        transition: all 0.2s ease;
    }}

    .stButton>button:hover {{
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
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
            
            submit = st.form_submit_button("🚀 Run Risk Assessment")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_output:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("📊 Risk Assessment")
        
        if submit:
            full_input_dict = {
                'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
                'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
                'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 
                'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
                'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract,
                'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
                'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
            }
            
            input_df = pd.DataFrame([full_input_dict])
            prob = model.predict_proba(input_df)[0][1]

            st.metric("Churn Probability", f"{prob:.2%}")
            
            if prob > 0.5:
                st.error("🚨 CRITICAL: HIGH RISK")
            else:
                st.success("✅ STABLE: LOW RISK")
        else:
            st.info("Awaiting analysis...")
        st.markdown('</div>', unsafe_allow_html=True)

# --- ANALYTICS SECTION ---
elif nav == "📊 Analytics":
    st.header("Business Performance Insights")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Recall Score", "~80%", "+2%")
        st.markdown('</div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("ROC-AUC", "0.85", "+0.01")
        st.markdown('</div>', unsafe_allow_html=True)
    with m_col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Avg Churn Rate", "26.58%", "-1.2%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("🎯 Feature Importance Analysis")
    importance_data = {'Feature': ['Contract', 'Tenure', 'MonthlyCharges'], 'Impact': [0.32, 0.28, 0.18]}
    st.bar_chart(pd.DataFrame(importance_data).set_index('Feature'))
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "📖 Documentation":
    st.header("Model Documentation")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.write("Algorithm: Logistic Regression with balanced class weights.")
    st.write("Validation: ROC-AUC 0.85.")
    st.markdown('</div>', unsafe_allow_html=True)