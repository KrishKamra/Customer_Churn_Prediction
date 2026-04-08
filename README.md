
---

# 📡 ChurnAI: Customer Prediction & Strategic Analytics

> **Project Mission:** An industry-grade machine learning platform that predicts telecommunications churn. This project bridges the gap between **inferential statistics** and **SaaS-level deployment**, providing actionable business intelligence via a high-performance Streamlit dashboard.

---

## 🚀 Platform Overview

* **Objective:** Proactively identify at-risk customers using a 20-feature predictive vector.
* **Target Metric:** `Churn` (Binary: **Yes/No**).
* **Deployment:** Dark-mode SaaS Dashboard with real-time "What-If" analysis.
* **Tech Stack:** `Python 3.13`, `Scikit-Learn`, `SciPy`, `Streamlit`, and `Pandas`.

---

## 📊 Statistical Rigor & Discovery

We utilized a **four-phase analytical approach** to ensure the model was grounded in statistical truth:

### 1. Inferential Hypothesis Testing
* **Independent Samples T-test:** Conducted on `TotalCharges_log`. With a P-value of $\approx 4.52 \times 10^{-15}$, we confirmed that spending patterns between churners and non-churners are statistically distinct.
* **Chi-Square ($\chi^2$) Analysis:** Established **Contract Type** as the primary churn driver ($p < 0.05$), identifying Month-to-Month users as the highest-risk segment.

### 2. Feature Engineering
* **Log-Transformation:** Applied $log(1+x)$ to `TotalCharges` to normalize right-skewed data, improving model convergence and coefficient interpretability.

---

## 🤖 Machine Learning & Production Pipeline

The project implements a **Full-Stack Scikit-Learn Pipeline** optimized for production:

* **Preprocessing:** Automated handling of `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
* **Class Imbalance:** Utilized `class_weight='balanced'` within the `LogisticRegression` model to maintain sensitivity to the minority churn class (~26%).
* **SaaS Dashboard Features:**
    * **Interactive Predictor:** Core feature sliders for high-correlation variables (Tenure, Contract, Charges).
    * **Advanced Input Hub:** Hidden `st.expander` for secondary demographic and service features, allowing for detailed "What-If" simulations.
    * **Automated Reporting:** Instant generation of timestamped `.txt` risk reports for business stakeholders.

---

## 💡 Strategic Business ROI

* **The "Danger Zone":** Churn probability spikes significantly in the **0–6 month** tenure window.
* **Performance Benchmarks:**
    * **Recall:** **~80%** (Ensuring maximum coverage of at-risk accounts).
    * **ROC-AUC:** **~0.85** (Exceptional class discrimination).

---

## 🛠️ Installation & Usage

### **1. Local Environment**
```bash
# Clone the repository
git clone https://github.com/KrishKamra/Your-Repo-Name.git
cd Your-Repo-Name

# Run the automated setup script
bash setup.sh
```

### **2. Launch the SaaS Portal**
```bash
streamlit run src/app.py
```

---

## 👨‍🔬 Author
**Krish Kamra** 