# Customer Churn Prediction & Statistical Analysis

> **Project Mission:** This project implements an end-to-end machine learning pipeline to predict telecommunications customer churn. Beyond simple prediction, it employs rigorous **inferential statistics** to derive actionable business insights from the Telco Customer Churn dataset.

---

## 🚀 Project Overview

* **Problem:** Identifying high-risk customers before they leave the service.
* **Target:** `Churn` (Binary: **Yes/No**).
* **Dataset:** 7,043 customer records with 21 features including *tenure*, *contract type*, and *charges*.
* **Tech Stack:** `Python 3.13.12`, `Scikit-Learn`, `SciPy`, `Pandas`, `Seaborn`, and `Streamlit`.

---

## 📊 Key Statistical Findings

We utilized a **four-phase analytical approach** to ensure the model was grounded in statistical truth:

### 1. Descriptive & Distribution Analysis
* **Central Tendency:** Established baseline **means** and **medians** for `tenure` and `MonthlyCharges`.
* **Normalization:** Applied **Log-Transformation** ($log1p$) to `TotalCharges` to resolve heavy right-skewness and satisfy normality assumptions for the model.

### 2. Inferential Hypothesis Testing
* **Numerical (T-Test):** Conducted an **Independent Samples T-test** on `TotalCharges_log`. With a P-value of $\approx 4.52 \times 10^{-15}$, we **rejected the null hypothesis**, confirming that spending patterns between churners and non-churners are statistically distinct.
* **Categorical (Chi-Square):** Performed $\chi^2$ tests across all categorical features. **Contract Type** emerged as the most significant predictor ($p < 0.05$), with Month-to-Month customers showing a dramatically higher churn rate.

---

## 🤖 Machine Learning Pipeline

The project uses a **Full-Stack Scikit-Learn Pipeline** for seamless deployment:

* **Preprocessing:** `SimpleImputer` for missing values, `StandardScaler` for numerical scaling, and `OneHotEncoder` for categorical variables.
* **Model:** `LogisticRegression` with `class_weight='balanced'` to handle the inherent class imbalance (approx. 26% churn rate).
* **Performance Metrics:**
    * **Recall:** **~80%** (Prioritizing the identification of all potential churners).
    * **ROC-AUC:** **~0.85** (High discriminative power between classes).

---

## 💡 Business Insights & Strategy

* **The "Danger Zone":** Threshold analysis reveals churn probability spikes in the **0–6 month** tenure window.
* **Strategic Recommendation:** Implement a **"First 90 Days"** engagement program and offer incentives to migrate "Month-to-Month" users to long-term contracts.

---

## 🛠️ How to Run

### **Environment Setup**
1. Create a virtual environment: `python -m venv churn_env`
2. Install dependencies: `pip install -r requirements.txt`

### **Model Generation**
* Run the Jupyter notebook to execute the pipeline and generate `churn_model_v1.pkl`.

### **Launch Dashboard**
* Execute the following command in your terminal:  
    `streamlit run src/app.py`

## 👨‍🔬 Author

**Krish Kamra**
