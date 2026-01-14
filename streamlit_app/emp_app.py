import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = r"E:\python\newbegining\employeeattrition\attrition_model.joblib"
DATA_PATH = r"C:\Users\manjula\Downloads\Employee-Attrition.xlsx"

# Load
model = joblib.load(MODEL_PATH)
df = pd.read_excel(DATA_PATH)

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")
st.title("💼 Employee Attrition Analytics & Prediction")

# ===============================
# DATA PREP
# ===============================
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"No": 0, "Yes": 1})

# Predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:,1]

# ===============================
# METRICS
# ===============================
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc = roc_auc_score(y, y_prob)

st.subheader("📈 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", round(acc,3))
col2.metric("Precision", round(prec,3))
col3.metric("Recall", round(rec,3))
col4.metric("F1 Score", round(f1,3))
col5.metric("ROC-AUC", round(roc,3))

# ===============================
# CONFUSION MATRIX
# ===============================
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ===============================
# ATTRITION DASHBOARD
# ===============================
st.subheader("📊 Attrition Insights")

attrition_rate = y.mean()*100
st.metric("Overall Attrition Rate", f"{round(attrition_rate,2)}%")

col1, col2 = st.columns(2)

with col1:
    dept_attrition = df.groupby("Department")["Attrition"].value_counts(normalize=True).unstack()["Yes"]*100
    st.bar_chart(dept_attrition)

with col2:
    overtime_attrition = df.groupby("OverTime")["Attrition"].value_counts(normalize=True).unstack()["Yes"]*100
    st.bar_chart(overtime_attrition)

# ===============================
# PREDICTION FORM
# ===============================
st.subheader("🔍 Predict Employee Attrition")

with st.form("predict_form"):
    age = st.slider("Age", 18, 60, 30)
    income = st.number_input("Monthly Income", 1000, 200000, 5000)
    total_years = st.slider("Total Working Years", 0, 40, 10)
    company_years = st.slider("Years at Company", 0, 40, 5)
    role_years = st.slider("Years in Current Role", 0, 20, 3)
    gender = st.selectbox("Gender", ["Male", "Female"])
    dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    role = st.selectbox("Job Role", df["JobRole"].unique())
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "TotalWorkingYears": total_years,
        "YearsAtCompany": company_years,
        "YearsInCurrentRole": role_years,
        "Gender": gender,
        "Department": dept,
        "JobRole": role,
        "MaritalStatus": marital,
        "OverTime": overtime
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Risk of Attrition ({round(prob*100,2)}%)")
    else:
        st.success(f"✅ Low Risk of Attrition ({round(prob*100,2)}%)")
