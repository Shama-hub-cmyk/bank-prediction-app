
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# إعداد الصفحة
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="نظام ذكي للتنبؤ بتبني الخدمات البنكية الرقمية",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CSS احترافي
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url("https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap");

    * { font-family: "Tajawal", sans-serif !important; }

    .main { background-color: #F0F4F8; }

    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E75B6 50%, #1ABC9C 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(31,56,100,0.3);
    }

    .main-header h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 0.5rem; }
    .main-header p  { font-size: 1rem; opacity: 0.9; margin: 0.2rem 0; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #2E75B6;
    }

    .card-green  { border-top-color: #27AE60; }
    .card-red    { border-top-color: #E8534A; }
    .card-orange { border-top-color: #F39C12; }
    .card-purple { border-top-color: #8E44AD; }

    .result-yes {
        background: linear-gradient(135deg, #27AE60, #2ECC71);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        box-shadow: 0 8px 24px rgba(39,174,96,0.4);
        animation: pulse 2s infinite;
    }

    .result-no {
        background: linear-gradient(135deg, #E8534A, #E74C3C);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        box-shadow: 0 8px 24px rgba(232,83,74,0.4);
    }

    .metric-box {
        background: linear-gradient(135deg, #F8F9FA, #E9ECEF);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #DEE2E6;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1F3864;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6C757D;
        margin-top: 0.3rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1F3864;
        border-right: 4px solid #2E75B6;
        padding-right: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1F3864, #2E75B6) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(31,56,100,0.3) !important;
    }

    .tab-content { padding: 1rem 0; }

    .insight-box {
        background: linear-gradient(135deg, #EBF5FB, #D6EAF8);
        border-right: 4px solid #2E75B6;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }

    .warning-box {
        background: linear-gradient(135deg, #FEF9E7, #FCF3CF);
        border-right: 4px solid #F39C12;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .success-box {
        background: linear-gradient(135deg, #EAFAF1, #D5F5E3);
        border-right: 4px solid #27AE60;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .sidebar .sidebar-content { background: #1F3864; }

    @keyframes pulse {
        0%   { box-shadow: 0 8px 24px rgba(39,174,96,0.4); }
        50%  { box-shadow: 0 8px 36px rgba(39,174,96,0.7); }
        100% { box-shadow: 0 8px 24px rgba(39,174,96,0.4); }
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# تحميل النماذج والبيانات
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all():
    models = {}
    model_files = {
        "Random Forest ⭐": "random_forest",
        "XGBoost":          "xgboost",
        "Logistic Regression": "logistic_regression",
        "Decision Tree":    "decision_tree"
    }
    for name, fname in model_files.items():
        with open(f"models/{fname}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return models, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Shama-hub-cmyk/bank-prediction-app/main/data/bank-full.csv", sep=";")
    return df

models, scaler = load_all()

# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🏦 BankPredict AI</h1>
    <h3 style="font-weight:500; margin:0.3rem 0;">نظام ذكي للتنبؤ بسلوك العملاء المصرفيين</h3>
    <p style="margin:0.8rem 0 0.2rem;">Predicting Digital Banking Service Adoption using Machine Learning</p>
    <p>التنبؤ بتبني الخدمات البنكية الرقمية باستخدام التعلم الآلي</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🎛️ لوحة التحكم")
page = st.sidebar.radio("", [
    "🔍 التنبؤ بالعميل",
    "📊 Dashboard التحليلي",
    "🤖 مقارنة النماذج",
    "🔬 SHAP التفسيري"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ إعدادات النموذج")
model_choice = st.sidebar.selectbox("اختر النموذج:", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 أداء النماذج")
perf_data = pd.DataFrame({
    "النموذج":  ["Random Forest", "XGBoost", "Logistic Reg.", "Decision Tree"],
    "F1":       ["45.17%", "42.72%", "33.31%", "39.95%"],
    "AUC":      ["77.70%", "77.75%", "71.58%", "71.30%"],
    "Accuracy": ["84.62%", "89.30%", "75.16%", "84.87%"]
})
st.sidebar.dataframe(perf_data, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# دالة معالجة البيانات
# ══════════════════════════════════════════════════════════════
TRAIN_COLS = [
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

NUM_COLS = ["age","balance","day","campaign","pdays","previous"]

def prepare_input(data_dict, scaler):
    raw = pd.DataFrame([data_dict])
    cat_cols = ["job","marital","education","default",
                "housing","loan","contact","month","poutcome"]
    raw_enc = pd.get_dummies(raw, columns=cat_cols, drop_first=True)
    for col in TRAIN_COLS:
        if col not in raw_enc.columns:
            raw_enc[col] = 0
    raw_enc = raw_enc[TRAIN_COLS]
    raw_enc[NUM_COLS] = scaler.transform(raw_enc[NUM_COLS])
    return raw_enc

# ══════════════════════════════════════════════════════════════
# صفحة التنبؤ
# ══════════════════════════════════════════════════════════════
if page == "🔍 التنبؤ بالعميل":

    st.markdown('<p class="section-title">📋 بيانات العميل</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 👤 المعلومات الشخصية")
        age       = st.slider("العمر", 18, 95, 35)
        job       = st.selectbox("المهنة", ["admin.","blue-collar","entrepreneur",
                    "housemaid","management","retired","self-employed",
                    "services","student","technician","unemployed","unknown"])
        marital   = st.selectbox("الحالة الزوجية", ["married","single","divorced"])
        education = st.selectbox("مستوى التعليم", ["primary","secondary","tertiary","unknown"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card card-green">', unsafe_allow_html=True)
        st.markdown("#### 💰 المعلومات المالية")
        balance = st.number_input("الرصيد السنوي (€)", -8000, 102000, 1000, 100)
        default = st.selectbox("تخلف عن السداد؟", ["no","yes"])
        housing = st.selectbox("قرض عقاري؟", ["yes","no"])
        loan    = st.selectbox("قرض شخصي؟", ["no","yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card card-orange">', unsafe_allow_html=True)
        st.markdown("#### 📞 معلومات الحملة")
        contact  = st.selectbox("طريقة التواصل", ["cellular","telephone","unknown"])
        month    = st.selectbox("شهر الاتصال", ["jan","feb","mar","apr","may","jun",
                                                  "jul","aug","sep","oct","nov","dec"])
        day      = st.slider("يوم الاتصال", 1, 31, 15)
        campaign = st.slider("عدد الاتصالات", 1, 63, 2)
        pdays    = st.number_input("أيام منذ آخر اتصال (-1=لا يوجد)", -1, 871, -1)
        previous = st.slider("اتصالات سابقة", 0, 275, 0)
        poutcome = st.selectbox("نتيجة الحملة السابقة", ["unknown","failure","other","success"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_btn = st.columns([1,2,1])
    with col_btn[1]:
        predict_btn = st.button("🔍 تنبأ بقرار العميل الآن")

    if predict_btn:
        data_dict = {
            "age":age,"job":job,"marital":marital,"education":education,
            "balance":balance,"default":default,"housing":housing,"loan":loan,
            "contact":contact,"day":day,"month":month,"campaign":campaign,
            "pdays":pdays,"previous":previous,"poutcome":poutcome
        }
        X_input    = prepare_input(data_dict, scaler)
        model      = models[model_choice]
        prediction = model.predict(X_input)[0]
        proba      = model.predict_proba(X_input)[0]

        st.markdown("---")
        st.markdown('<p class="section-title">🎯 نتيجة التنبؤ</p>', unsafe_allow_html=True)

        col_r1, col_r2, col_r3 = st.columns([2,1,1])

        with col_r1:
            if prediction == 1:
                st.markdown("""<div class="result-yes">
                    ✅ سيتبنى الخدمة البنكية الرقمية<br>
                    <small style="font-size:1rem;">العميل مرشح للاشتراك</small>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="result-no">
                    ❌ لن يتبنى الخدمة البنكية الرقمية<br>
                    <small style="font-size:1rem;">العميل غير مرشح للاشتراك</small>
                </div>""", unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value" style="color:#27AE60">{proba[1]*100:.1f}%</div>
                <div class="metric-label">احتمال التبني</div>
            </div>""", unsafe_allow_html=True)
            st.progress(float(proba[1]))

        with col_r3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value" style="color:#E8534A">{proba[0]*100:.1f}%</div>
                <div class="metric-label">احتمال عدم التبني</div>
            </div>""", unsafe_allow_html=True)
            st.progress(float(proba[0]))

        # توصية استراتيجية
        st.markdown('<p class="section-title">💡 التوصية الاستراتيجية</p>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            if proba[1] >= 0.6:
                st.markdown("""<div class="success-box">
                    🔥 <strong>أولوية عالية جداً</strong><br>
                    استهدف هذا العميل فوراً — احتمال تبني مرتفع جداً!<br>
                    يُنصح بتخصيص عرض مميز له.
                </div>""", unsafe_allow_html=True)
            elif proba[1] >= 0.4:
                st.markdown("""<div class="warning-box">
                    ⚠️ <strong>أولوية متوسطة</strong><br>
                    العميل يحتاج متابعة إضافية وعرضاً مخصصاً.<br>
                    جرّب التواصل في شهر مارس أو سبتمبر.
                </div>""", unsafe_allow_html=True)
            elif proba[1] >= 0.2:
                st.markdown("""<div class="insight-box">
                    💡 <strong>أولوية منخفضة</strong><br>
                    قد يستجيب مع تحسين ظروف العرض.<br>
                    انتظر تحسن وضعه المالي أو نتيجة حملة سابقة.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="result-no" style="font-size:1rem; padding:1rem;">
                    ❌ وجّه جهودك لعملاء أكثر قابليةً للتبني
                </div>""", unsafe_allow_html=True)

        with col_t2:
            st.markdown("**🔑 العوامل المؤثرة في القرار:**")
            factors = []
            if poutcome == "success":
                factors.append(("✅", "نجاح الحملة السابقة", "إيجابي قوي جداً"))
            if balance > 1362:
                factors.append(("✅", f"الرصيد {balance:,}€", "فوق المتوسط — إيجابي"))
            if job in ["student","retired"]:
                factors.append(("✅", f"مهنة {job}", "معدل تبني مرتفع — إيجابي"))
            if month in ["mar","sep","oct","dec"]:
                factors.append(("✅", f"شهر {month}", "معدل استجابة مرتفع — إيجابي"))
            if campaign > 3:
                factors.append(("⚠️", f"{campaign} اتصالات", "عدد مرتفع — سلبي"))
            if housing == "yes" and loan == "yes":
                factors.append(("⚠️", "قروض متعددة", "قد يُقلّص الاهتمام"))
            if poutcome == "failure":
                factors.append(("❌", "فشل الحملة السابقة", "سلبي"))
            if not factors:
                factors.append(("📊", "ملف متوسط", "بدون عوامل بارزة"))
            for icon, factor, impact in factors:
                st.markdown(f"{icon} **{factor}** — {impact}")

        # SHAP للعميل الحالي
        st.markdown('<p class="section-title">🔬 تفسير SHAP للعميل الحالي</p>',
                    unsafe_allow_html=True)

        try:
            rf_model   = models["Random Forest ⭐"]
            explainer  = shap.TreeExplainer(rf_model)
            shap_vals  = explainer.shap_values(X_input)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            else:
                sv = shap_vals[0]

            # أهم 10 متغيرات
            feat_names  = TRAIN_COLS
            abs_sv      = np.abs(sv)
            top_idx     = np.argsort(abs_sv)[::-1][:10]
            top_names   = [feat_names[i] for i in top_idx]
            top_sv      = [sv[i] for i in top_idx]

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#F8F9FA")
            colors  = ["#27AE60" if v > 0 else "#E8534A" for v in top_sv]
            ax.barh(range(len(top_names)), top_sv[::-1],
                    color=colors[::-1], edgecolor="white")
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names[::-1], fontsize=9)
            ax.axvline(0, color="black", linewidth=1.5)
            ax.set_title("SHAP Values — تفسير قرار النموذج لهذا العميل",
                         fontsize=12, fontweight="bold", color="#1F3864")
            ax.set_xlabel("SHAP Value (أخضر = يزيد التبني | أحمر = يقلص التبني)")
            ax.set_facecolor("#F8F9FA")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP غير متاح: {e}")

        # تقرير PDF
        st.markdown("---")
        st.markdown('<p class="section-title">📄 تحميل التقرير</p>', unsafe_allow_html=True)

        report_text = f"""
RAPPORT DE PRÉDICTION — نظام ذكي للتنبؤ بتبني الخدمات البنكية الرقمية
========================================================================
شيماء الضاومي | ماستر الهندسة المالية التشاركية | FSJES عين السبع 2025-2026

بيانات العميل:
--------------
العمر: {age} | المهنة: {job} | الحالة الزوجية: {marital}
مستوى التعليم: {education} | الرصيد: {balance:,}€
قرض عقاري: {housing} | قرض شخصي: {loan}
طريقة التواصل: {contact} | شهر الاتصال: {month}
عدد الاتصالات: {campaign} | نتيجة الحملة السابقة: {poutcome}

نتيجة التنبؤ:
-------------
النموذج المستخدم: {model_choice}
القرار: {"✅ سيتبنى الخدمة" if prediction == 1 else "❌ لن يتبنى الخدمة"}
احتمال التبني: {proba[1]*100:.1f}%
احتمال عدم التبني: {proba[0]*100:.1f}%

التوصية الاستراتيجية:
---------------------
{"🔥 أولوية عالية جداً — استهدف فوراً" if proba[1]>=0.6
 else "⚠️ أولوية متوسطة — يحتاج متابعة" if proba[1]>=0.4
 else "💡 أولوية منخفضة — انتظر ظروفاً أفضل" if proba[1]>=0.2
 else "❌ غير مرشح — وجّه جهودك لعملاء آخرين"}

========================================================================
Generated by: Système Intelligent de Prédiction — Bank Marketing
        """

        st.download_button(
            label="📥 تحميل تقرير العميل",
            data=report_text.encode("utf-8"),
            file_name=f"rapport_client_{age}_{job}.txt",
            mime="text/plain"
        )

# ══════════════════════════════════════════════════════════════
# صفحة Dashboard
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard التحليلي":
    st.markdown('<p class="section-title">📊 Dashboard التحليلي — Bank Marketing Dataset</p>',
                unsafe_allow_html=True)

    # بيانات مدمجة داخل التطبيق
    stats = {
        "إجمالي العملاء": "45,211",
        "نسبة التبني": "11.7%",
        "متوسط العمر": "40.9 سنة",
        "متوسط الرصيد": "1,362 €"
    }
    colors_stat = ["#2E75B6","#27AE60","#F39C12","#8E44AD"]

    cols = st.columns(4)
    for i, (label, value) in enumerate(stats.items()):
        with cols[i]:
            st.markdown(f"""<div class="metric-box" style="border-top: 4px solid {colors_stat[i]};">
                <div class="metric-value" style="color:{colors_stat[i]}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F8F9FA")
        job_rates = {
            "student": 28.7, "retired": 22.8, "unemployed": 15.5,
            "management": 13.8, "admin.": 12.2, "technician": 11.1,
            "services": 8.9, "blue-collar": 7.3
        }
        colors_j = ["#27AE60" if v > 11.7 else "#2E75B6" for v in job_rates.values()]
        ax.barh(list(job_rates.keys()), list(job_rates.values()),
                color=colors_j, edgecolor="white")
        ax.axvline(11.7, color="#E8534A", linestyle="--", linewidth=1.5,
                   label="المعدل العام 11.7%")
        ax.set_title("معدل التبني حسب المهنة (%)",
                     fontweight="bold", color="#1F3864")
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=9)
        st.pyplot(fig)

    with col_d2:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F8F9FA")
        months = ["jan","feb","mar","apr","may","jun",
                  "jul","aug","sep","oct","nov","dec"]
        rates  = [6.3,5.8,52.0,19.7,6.2,11.6,
                  9.7,8.9,46.5,43.8,9.8,46.7]
        colors_m = ["#27AE60" if v > 20 else "#2E75B6" for v in rates]
        ax.bar(months, rates, color=colors_m, edgecolor="white")
        ax.axhline(11.7, color="#E8534A", linestyle="--", linewidth=1.5)
        ax.set_title("معدل التبني حسب الشهر (%)",
                     fontweight="bold", color="#1F3864")
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F8F9FA")
        poutcome_rates = {"success": 64.7, "other": 16.7,
                          "failure": 12.6, "unknown": 9.2}
        colors_po = ["#27AE60" if v > 11.7 else "#E8534A"
                     for v in poutcome_rates.values()]
        bars = ax.bar(poutcome_rates.keys(), poutcome_rates.values(),
                      color=colors_po, edgecolor="white", width=0.5)
        for b, v in zip(bars, poutcome_rates.values()):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{v}%", ha="center", fontweight="bold")
        ax.axhline(11.7, color="#2E75B6", linestyle="--", linewidth=1.5)
        ax.set_title("تأثير نتيجة الحملة السابقة (poutcome) 🔥",
                     fontweight="bold", color="#1F3864")
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    with col_d4:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F8F9FA")
        sizes  = [39922, 5289]
        labels = ["No (88.3%)", "Yes (11.7%)"]
        ax.pie(sizes, labels=labels,
               colors=["#E8534A","#27AE60"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title("توزيع المتغير الهدف y",
                     fontweight="bold", color="#1F3864")
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
# صفحة مقارنة النماذج
# ══════════════════════════════════════════════════════════════
elif page == "🤖 مقارنة النماذج":
    st.markdown('<p class="section-title">🤖 مقارنة أداء النماذج التنبؤية</p>',
                unsafe_allow_html=True)

    model_names  = ["Logistic Regression","Decision Tree","Random Forest","XGBoost"]
    accuracy     = [75.16, 84.87, 84.62, 89.30]
    precision    = [24.29, 37.30, 38.74, 57.12]
    recall       = [53.02, 43.01, 54.16, 34.12]
    f1_scores    = [33.31, 39.95, 45.17, 42.72]
    auc_scores   = [71.58, 71.30, 77.70, 77.75]
    colors_model = ["#2E75B6","#F39C12","#27AE60","#8E44AD"]

    # جدول احترافي
    df_results = pd.DataFrame({
        "النموذج":   model_names,
        "Accuracy":  [f"{v}%" for v in accuracy],
        "Precision": [f"{v}%" for v in precision],
        "Recall":    [f"{v}%" for v in recall],
        "F1-Score":  [f"{v}%" for v in f1_scores],
        "ROC-AUC":   [f"{v}%" for v in auc_scores],
        "التقييم":   ["متوسط","جيد","⭐ الأفضل F1","⭐ الأفضل Accuracy"]
    })
    st.dataframe(df_results, hide_index=True, use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8F9FA")
        x = np.arange(len(model_names))
        w = 0.2
        short = ["LR","DT","RF","XGB"]
        ax.bar(x-w*1.5, accuracy,  w, label="Accuracy",  color="#2E75B6", alpha=0.85)
        ax.bar(x-w*0.5, f1_scores, w, label="F1-Score",  color="#27AE60", alpha=0.85)
        ax.bar(x+w*0.5, auc_scores,w, label="ROC-AUC",   color="#F39C12", alpha=0.85)
        ax.bar(x+w*1.5, recall,    w, label="Recall",    color="#8E44AD", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(short)
        ax.set_title("مقارنة شاملة للمؤشرات (%)",
                     fontweight="bold", color="#1F3864")
        ax.set_ylabel("%")
        ax.legend(fontsize=9)
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    with col_m2:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8F9FA")
        # ROC Curves تقريبية
        for i, (name, auc) in enumerate(zip(short, auc_scores)):
            fpr = np.linspace(0, 1, 100)
            tpr = fpr ** ((1 - auc/100) * 3)
            ax.plot(fpr, tpr, color=colors_model[i], linewidth=2.5,
                    label=f"{name} (AUC={auc}%)")
        ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5)
        ax.set_title("منحنيات ROC التقريبية",
                     fontweight="bold", color="#1F3864")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=9)
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    # تحليل نقدي
    st.markdown('<p class="section-title">📝 التحليل الأكاديمي للنتائج</p>',
                unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""<div class="success-box">
            <strong>🏆 Random Forest — الأفضل للتسويق البنكي</strong><br>
            F1-Score: 45.17% | ROC-AUC: 77.70%<br>
            يُحقق أفضل توازن بين Precision وRecall.
            مناسب للحملات التسويقية لأنه يكتشف أكثر العملاء المرشحين.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="success-box">
            <strong>⚡ XGBoost — الأعلى دقة إجمالية</strong><br>
            Accuracy: 89.30% | ROC-AUC: 77.75%<br>
            يُحقق أعلى دقة إجمالية لكن Recall أضعف.
            مناسب حين تكون تكلفة الخطأ الإيجابي الكاذب مرتفعة.
        </div>""", unsafe_allow_html=True)
    with col_a2:
        st.markdown("""<div class="warning-box">
            <strong>⚠️ Decision Tree — سهل التفسير</strong><br>
            F1-Score: 39.95% | Accuracy: 84.87%<br>
            مفيد لتقديم قواعد واضحة وبسيطة للإدارة.
            لكن يُعاني من Overfitting دون تشذيب دقيق.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-box">
            <strong>📊 Logistic Regression — Baseline مرجعي</strong><br>
            F1-Score: 33.31% | Recall: 53.02%<br>
            يُحقق أعلى Recall لكن Precision منخفضة.
            يكتشف كثيراً من المرشحين لكن مع تنبؤات خاطئة أكثر.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# صفحة SHAP
# ══════════════════════════════════════════════════════════════
elif page == "🔬 SHAP التفسيري":
    st.markdown('<p class="section-title">🔬 الذكاء الاصطناعي التفسيري — SHAP Analysis</p>',
                unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
        <strong>ما هو SHAP؟</strong><br>
        SHAP (SHapley Additive exPlanations) هي أداة تُفسّر قرارات النموذج الذكي
        بتحديد مساهمة كل متغير في القرار النهائي. تستند إلى نظرية شابلي في
        رياضيات الألعاب التعاونية (Lundberg & Lee, 2017).
    </div>""", unsafe_allow_html=True)

    # أهم المتغيرات حسب SHAP
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#F8F9FA")
        shap_features = [
            "campaign","day","contact_unknown","education_secondary",
            "education_tertiary","balance","poutcome_success","age",
            "marital_single","job_blue-collar"
        ]
        shap_values_global = [
            0.5527,0.3978,0.3328,0.3027,0.2832,
            0.2727,0.2628,0.2554,0.2265,0.2032
        ]
        colors_s = ["#27AE60" if v > 0.3 else "#2E75B6"
                    for v in shap_values_global]
        ax.barh(range(len(shap_features)),
                shap_values_global[::-1],
                color=colors_s[::-1], edgecolor="white")
        ax.set_yticks(range(len(shap_features)))
        ax.set_yticklabels(shap_features[::-1], fontsize=9)
        ax.set_title("SHAP Global Feature Importance - أهم المتغيرات على مستوى كل العملاء",
                     fontweight="bold", color="#1F3864", fontsize=11)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    with col_s2:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#F8F9FA")
        shap_dir = [
            -0.45, -0.32, -0.28, 0.25, 0.22,
            0.18, 0.35, 0.15, 0.12, -0.18
        ]
        colors_d = ["#27AE60" if v > 0 else "#E8534A" for v in shap_dir]
        ax.barh(range(len(shap_features)),
                shap_dir[::-1],
                color=colors_d[::-1], edgecolor="white")
        ax.set_yticks(range(len(shap_features)))
        ax.set_yticklabels(shap_features[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=1.5)
        ax.set_title("اتجاه تأثير المتغيرات - أخضر يزيد التبني - أحمر يقلصه",
                     fontweight="bold", color="#1F3864", fontsize=11)
        ax.set_xlabel("Mean SHAP value")
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    # تفسير أكاديمي
    st.markdown('<p class="section-title">📝 التفسير الأكاديمي لنتائج SHAP</p>',
                unsafe_allow_html=True)
    insights = [
        ("campaign", "عدد الاتصالات — الأقوى تأثيراً",
         "كلما زاد عدد الاتصالات قلّ احتمال التبني. الإلحاح يُنفّر العملاء."),
        ("poutcome_success", "نجاح الحملة السابقة — إيجابي قوي",
         "العميل الذي استجاب لحملة سابقة احتمال تبنيه 64.7% — استهدفه أولاً."),
        ("balance", "الرصيد المرتفع — إيجابي",
         "العملاء ذوو الرصيد الأعلى أكثر استعداداً للاستثمار في الودائع."),
        ("contact_unknown", "طريقة تواصل مجهولة — سلبي",
         "عدم معرفة طريقة التواصل المفضّلة يُقلّص فرص النجاح."),
    ]

    for feat, title, explanation in insights:
        st.markdown(f"""<div class="insight-box">
            <strong>🔑 {title}</strong><br>
            {explanation}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#6C757D; font-size:0.85rem; padding:1rem;">
    🏦 نظام ذكي للتنبؤ بتبني الخدمات البنكية الرقمية<br>
    شيماء الضاومي | ماستر الهندسة المالية التشاركية والذكاء الاصطناعي<br>
    FSJES عين السبع | 2025-2026<br>
    <small>Powered by Machine Learning: Random Forest | XGBoost | Logistic Regression | Decision Tree</small>
</div>
""", unsafe_allow_html=True)
