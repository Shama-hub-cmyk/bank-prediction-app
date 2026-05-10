
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import arabic_reshaper
from bidi.algorithm import get_display

# ── إعداد الصفحة ──────────────────────────────────────────────
st.set_page_config(
    page_title="نظام التنبؤ بتبني الخدمات البنكية الرقمية",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS مخصص ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .result-yes {
        background: linear-gradient(135deg, #27AE60, #2ECC71);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .result-no {
        background: linear-gradient(135deg, #E8534A, #E74C3C);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E75B6;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 نظام ذكي للتنبؤ بتبني الخدمات البنكية الرقمية</h1>
    <p>تصميم نظام ذكي للتنبؤ بتبني الخدمات البنكية الرقمية باستخدام تقنيات التعلم الآلي</p>
    <p>شيماء الضاومي | ماستر الهندسة المالية التشاركية والذكاء الاصطناعي | FSJES عين السبع 2025-2026</p>
</div>
""", unsafe_allow_html=True)

# ── تحميل النماذج ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/xgboost.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("models/logistic_regression.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("models/decision_tree.pkl", "rb") as f:
        dt = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return rf, xgb, lr, dt, scaler

rf_model, xgb_model, lr_model, dt_model, scaler = load_models()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ إعدادات النظام")
model_choice = st.sidebar.selectbox(
    "اختر النموذج:",
    ["Random Forest ⭐", "XGBoost", "Logistic Regression", "Decision Tree"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 أداء النماذج")
perf_data = {
    "النموذج": ["Random Forest", "XGBoost", "Logistic Reg.", "Decision Tree"],
    "F1-Score": ["45.17%", "42.72%", "33.31%", "39.95%"],
    "ROC-AUC":  ["77.70%", "77.75%", "71.58%", "71.30%"]
}
st.sidebar.dataframe(pd.DataFrame(perf_data), hide_index=True)

# ── المدخلات ──────────────────────────────────────────────────
st.markdown("## 📋 بيانات العميل")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 المعلومات الشخصية")
    age       = st.slider("العمر", 18, 95, 35)
    job       = st.selectbox("المهنة", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"])
    marital   = st.selectbox("الحالة الزوجية", ["married", "single", "divorced"])
    education = st.selectbox("مستوى التعليم", ["primary", "secondary", "tertiary", "unknown"])

with col2:
    st.markdown("### 💰 المعلومات المالية")
    balance  = st.number_input("الرصيد السنوي (€)", -8000, 102000, 1000, 100)
    default  = st.selectbox("تخلف عن السداد؟", ["no", "yes"])
    housing  = st.selectbox("قرض عقاري؟", ["yes", "no"])
    loan     = st.selectbox("قرض شخصي؟", ["no", "yes"])

with col3:
    st.markdown("### 📞 معلومات الحملة")
    contact   = st.selectbox("طريقة التواصل", ["cellular", "telephone", "unknown"])
    month     = st.selectbox("شهر الاتصال", [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"])
    day       = st.slider("يوم الاتصال", 1, 31, 15)
    campaign  = st.slider("عدد الاتصالات في هذه الحملة", 1, 63, 2)
    pdays     = st.number_input("أيام منذ آخر اتصال (-1 = لا يوجد)", -1, 871, -1)
    previous  = st.slider("اتصالات الحملات السابقة", 0, 275, 0)
    poutcome  = st.selectbox("نتيجة الحملة السابقة", ["unknown", "failure", "other", "success"])

# ── معالجة البيانات ───────────────────────────────────────────
def prepare_input(age, job, marital, education, balance, default,
                  housing, loan, contact, month, day, campaign,
                  pdays, previous, poutcome, scaler):

    raw = pd.DataFrame([{
        "age": age, "job": job, "marital": marital,
        "education": education, "default": default,
        "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month,
        "campaign": campaign, "pdays": pdays,
        "previous": previous, "poutcome": poutcome
    }])

    cat_cols = ["job","marital","education","default",
                "housing","loan","contact","month","poutcome"]
    raw_enc  = pd.get_dummies(raw, columns=cat_cols, drop_first=True)

    # أعمدة التدريب
    train_cols = [
        "age","balance","day","campaign","pdays","previous",
        "job_blue-collar","job_entrepreneur","job_housemaid",
        "job_management","job_retired","job_self-employed",
        "job_services","job_student","job_technician",
        "job_unemployed","job_unknown",
        "marital_married","marital_single",
        "education_secondary","education_tertiary","education_unknown",
        "default_yes","housing_yes","loan_yes",
        "contact_telephone","contact_unknown",
        "month_aug","month_dec","month_feb","month_jan",
        "month_jul","month_jun","month_mar","month_may",
        "month_nov","month_oct","month_sep",
        "poutcome_other","poutcome_success","poutcome_unknown"
    ]

    for col in train_cols:
        if col not in raw_enc.columns:
            raw_enc[col] = 0
    raw_enc = raw_enc[train_cols]

    num_cols = ["age","balance","day","campaign","pdays","previous"]
    raw_enc[num_cols] = scaler.transform(raw_enc[num_cols])

    return raw_enc

# ── زر التنبؤ ─────────────────────────────────────────────────
st.markdown("---")
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predict_btn = st.button("🔍 تنبأ بقرار العميل")

if predict_btn:
    X_input = prepare_input(
        age, job, marital, education, balance, default,
        housing, loan, contact, month, day, campaign,
        pdays, previous, poutcome, scaler)

    # اختيار النموذج
    if "Random Forest" in model_choice:
        model = rf_model
    elif "XGBoost" in model_choice:
        model = xgb_model
    elif "Logistic" in model_choice:
        model = lr_model
    else:
        model = dt_model

    prediction = model.predict(X_input)[0]
    proba      = model.predict_proba(X_input)[0]

    st.markdown("---")
    st.markdown("## 🎯 نتيجة التنبؤ")

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        if prediction == 1:
            st.markdown("""
            <div class="result-yes">
                ✅ سيتبنى الخدمة<br>
                <small>العميل مرشح للاشتراك</small>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-no">
                ❌ لن يتبنى الخدمة<br>
                <small>العميل غير مرشح للاشتراك</small>
            </div>""", unsafe_allow_html=True)

    with col_res2:
        st.metric("احتمال التبني", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

    with col_res3:
        st.metric("احتمال عدم التبني", f"{proba[0]*100:.1f}%")
        st.progress(float(proba[0]))

    # تفاصيل إضافية
    st.markdown("### 📊 تفسير القرار")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if proba[1] > 0.6:
            st.success("🔥 احتمال تبني مرتفع جداً — استهدف هذا العميل فوراً!")
        elif proba[1] > 0.4:
            st.warning("⚠️ احتمال تبني متوسط — يحتاج متابعة إضافية")
        elif proba[1] > 0.2:
            st.info("💡 احتمال تبني منخفض — قد يستجيب مع عرض مخصص")
        else:
            st.error("❌ احتمال تبني ضعيف جداً — وجّه جهودك لعملاء آخرين")

    with col_d2:
        st.markdown("**العوامل المؤثرة في القرار:**")
        factors = []
        if poutcome == "success":
            factors.append("✅ نجاح الحملة السابقة — عامل إيجابي قوي")
        if balance > 1362:
            factors.append("✅ الرصيد فوق المتوسط — إيجابي")
        if campaign > 3:
            factors.append("⚠️ عدد اتصالات مرتفع — سلبي")
        if job in ["student", "retired"]:
            factors.append("✅ مهنة ذات معدل تبني مرتفع — إيجابي")
        if month in ["mar", "sep", "oct", "dec"]:
            factors.append("✅ شهر ذو معدل استجابة مرتفع — إيجابي")
        if not factors:
            factors.append("📊 ملف متوسط بدون عوامل بارزة")
        for f in factors:
            st.write(f)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.9rem;">
    🏦 نظام ذكي للتنبؤ بتبني الخدمات البنكية الرقمية<br>
    شيماء الضاومي | ماستر الهندسة المالية التشاركية والذكاء الاصطناعي<br>
    FSJES عين السبع | 2025-2026
</div>
""", unsafe_allow_html=True)
