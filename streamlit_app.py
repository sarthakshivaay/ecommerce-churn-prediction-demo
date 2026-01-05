import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "models/churn_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["feature_names"]


st.set_page_config(page_title="Churn Prediction Demo", layout="wide")
st.title("🛍️ E‑Commerce Customer Churn Prediction")
st.write("Predict customer churn risk using behavioral features and engagement metrics.")

if not Path(MODEL_PATH).exists():
    st.error("Model file not found. Run `python src/model.py` first.")
    st.stop()

model, scaler, feature_names = load_model()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Predict Churn", "Feature Importance"])

# ========== PAGE 1: PREDICT CHURN ==========
if page == "Predict Churn":
    st.subheader("Input customer features")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        income = st.number_input("Annual Income", min_value=10000, max_value=200000, value=60000, step=1000)
        time_on_site = st.number_input("Time on Site (avg)", min_value=0.0, max_value=1000.0, value=200.0)
        num_purchases = st.number_input("Number of Purchases", min_value=0, max_value=100, value=5)

    with col2:
        avg_rating = st.number_input("Average Purchase Rating", min_value=0.0, max_value=5.0, value=4.0)
        browsing_events = st.number_input("Browsing Events (proxy)", min_value=0.0, max_value=10000.0, value=500.0)
        review_length = st.number_input("Review Text Length", min_value=0.0, max_value=2000.0, value=80.0)
        sentiment_score = st.number_input("Review Sentiment Score", min_value=-10.0, max_value=10.0, value=1.0)

    # Recompute engagement_score exactly as in pipeline
    engagement_score = (
        (time_on_site / (time_on_site + 1e-6)) + (browsing_events / (browsing_events + 1e-6))
    ) / 2.0

    if st.button("Predict churn risk"):
        # Build feature vector in same order as training
        input_array = np.array([[
            age,
            income,
            time_on_site,
            num_purchases,
            avg_rating,
            browsing_events,
            review_length,
            sentiment_score,
            engagement_score,
        ]])

        X_scaled = scaler.transform(input_array)

        proba = model.predict_proba(X_scaled)

        # Handle case: model was trained on a single class
        if proba.shape[1] == 1:
            prob_churn = 0.0
            pred_label = int(model.predict(X_scaled)[0])
        else:
            prob_churn = float(proba[0, 1])
            pred_label = int(model.predict(X_scaled)[0])

        st.markdown("---")
        st.subheader("Prediction result")

        if pred_label == 1:
            st.error(f"⚠️ High churn risk: **{prob_churn*100:.1f}%**")
        else:
            st.success(f"✅ Low churn risk: **{(1 - prob_churn)*100:.1f}%**")

        st.write("Raw probabilities:")
        if proba.shape[1] == 1:
            st.write({"P(only_class)": float(proba[0, 0])})
        else:
            st.write({"P(Churn=0)": float(1 - prob_churn), "P(Churn=1)": float(prob_churn)})

# ========== PAGE 2: FEATURE IMPORTANCE ==========
elif page == "Feature Importance":
    st.subheader("🔍 Model Explainability: Feature Importance")
    st.write("Which features drive churn predictions the most?")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.markdown("### Top Features by Importance")
        st.dataframe(importance_df.style.format({"Importance": "{:.4f}"}), use_container_width=True)

        st.markdown("### Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title("Feature Importance (Random Forest)", fontsize=14)
        ax.invert_yaxis()
        st.pyplot(fig)

        st.markdown("---")
        st.info("""
        **Interpretation:**  
        - Higher importance = that feature contributes more to the model's churn prediction.
        - In Random Forest, importance is measured by how much each feature reduces impurity (Gini) across all trees.
        - Focus retention efforts on customers with poor performance in top‑ranked features.
        """)
    else:
        st.warning("Model does not support feature importance (not a tree‑based model).")
