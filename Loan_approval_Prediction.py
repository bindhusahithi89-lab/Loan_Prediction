import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS to style the Streamlit interface with a professional gradient background

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right,#1f3c88, #6a1b9a);
    color: white;
    font-family: 'Segoe UI';
}
h1, h2, h3 ,label{ color: white; font-weight: 800; }
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.12);
    text-align: center;
}
.label { font-size: 14px; opacity: 0.85; }
.big-number { font-size: 28px; font-weight: 800; margin-top: 6px; }
.green { color: #22c55e; }
.blue { color: #3b82f6; }
.orange { color: #f59e0b; }
.purple { color: #a855f7; }
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
    border: none;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #00E5FF !important;
    font-weight: 700 !important;
}

/* --- ONLY CHANGE LABELS ("Accuracy", "Precision", etc.) --- */
div[data-testid="stMetricLabel"] p {
    color: #00E5FF !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* --- Metric Labels --- */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {
    color: #00E5FF !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* --- Metric Values --- */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: white !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()

    for col in ["education", "self_employed", "loan_status"]:
        df[col] = df[col].str.strip()

    df.drop("loan_id", axis=1, inplace=True)

    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    df["loan_to_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1)
    df["loan_to_asset_ratio"] = df["loan_amount"] / (df["total_assets"] + 1)
    df["emi"] = df["loan_amount"] / df["loan_term"]
    df["income_to_emi_ratio"] = df["income_annum"] / (df["loan_amount"] / df["loan_term"] + 1)

    le_edu = LabelEncoder()
    le_emp = LabelEncoder()

    df["education"] = le_edu.fit_transform(df["education"])
    df["self_employed"] = le_emp.fit_transform(df["self_employed"])

    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    return df, X, y, le_edu, le_emp


df, X, y, le_edu, le_emp = load_data()

# -----------------------------
# MODEL TRAINING
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        criterion="entropy",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X_train, X_test, y_train, y_test, y_pred, y_prob


@st.cache_data
def compute_metrics(y_test, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc": roc_auc_score(y_test, y_prob)
    }


@st.cache_data
def get_feature_importance(X, _model):
    return pd.DataFrame({
        "Feature": X.columns,
        "Importance": _model.feature_importances_
    }).sort_values("Importance", ascending=False)


# -----------------------------
# TRAIN
# -----------------------------
with st.spinner("Training model..."):
    model, X_train, X_test, y_train, y_test, y_pred, y_prob = train_model(X, y)

metrics = compute_metrics(y_test, y_pred, y_prob)

# -----------------------------
# METRICS UI
# -----------------------------
st.subheader("📊 Model Performance")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
c4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")
c5.metric("ROC-AUC", f"{metrics['roc']*100:.2f}%")

# ====================================================
# 🏦 PREDICTION SECTION
# ====================================================
st.markdown("---")
st.header("🏦 Predict Loan Approval")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Dependents", 0, 10, 0)
    education = st.selectbox("Education", le_edu.classes_)
    self_employed = st.selectbox("Self Employed", le_emp.classes_)
    income_annum = st.number_input("Annual Income", 0, 100000000, 5000000)
    loan_amount = st.number_input("Loan Amount", 0, 100000000, 10000000)

with col2:
    loan_term = st.number_input("Loan Term", 1, 360, 120)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 700)
    residential_assets_value = st.number_input("Residential Assets", 0, 100000000, 3000000)
    commercial_assets_value = st.number_input("Commercial Assets", 0, 100000000, 2000000)
    luxury_assets_value = st.number_input("Luxury Assets", 0, 100000000, 5000000)
    bank_asset_value = st.number_input("Bank Assets", 0, 100000000, 2000000)

if st.button("🔍 Predict Loan Status"):

    total_assets = (
        residential_assets_value +
        commercial_assets_value +
        luxury_assets_value +
        bank_asset_value
    )

    loan_to_income_ratio = loan_amount / (income_annum + 1)
    loan_to_asset_ratio = loan_amount / (total_assets + 1)
    emi = loan_amount / loan_term
    income_to_emi_ratio = income_annum / (emi + 1)

    input_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": le_edu.transform([education])[0],
        "self_employed": le_emp.transform([self_employed])[0],
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "total_assets": total_assets,
        "loan_to_income_ratio": loan_to_income_ratio,
        "loan_to_asset_ratio": loan_to_asset_ratio,
        "emi": emi,
        "income_to_emi_ratio": income_to_emi_ratio
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Loan Approved\nApproval Probability: {probability*100:.2f}%")
    else:
        st.error(f"❌ Loan Rejected\nApproval Probability: {probability*100:.2f}%")

    st.progress(float(probability))

# -----------------------------
# CONFUSION MATRIX (ONLY CHANGED PART)
# -----------------------------
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(3.5, 2.8))  # ✅ reduced size only

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig, use_container_width=False)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📈 Feature Importance")

importance_df = get_feature_importance(X, model)

fig2, ax2 = plt.subplots(figsize=(6, 4))  # 👈 reduced size

sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax2)

ax2.tick_params(labelsize=8)  # smaller labels
ax2.set_xlabel("Importance", fontsize=9)
ax2.set_ylabel("Feature", fontsize=9)

st.pyplot(fig2, use_container_width=False)