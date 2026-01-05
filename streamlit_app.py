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


st.set_page_config(
    page_title="Churn Prediction Demo",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛍️ E‑Commerce Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict customer churn risk using behavioral features and engagement metrics.</div>', unsafe_allow_html=True)

if not Path(MODEL_PATH).exists():
    st.error("Model file not found. Run `python src/model.py` first.")
    st.stop()

model, scaler, feature_names = load_model()

# Sidebar navigation with better description
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select a page:",
    ["🔮 Predict Churn", "📊 Feature Importance", "ℹ️ About"],
    label_visibility="collapsed"
)

# ========== PAGE 1: PREDICT CHURN ==========
if page == "🔮 Predict Churn":
    st.markdown("### 📝 Input Customer Features")
    
    # Add preset examples
    st.markdown("**Quick Start:** Try these example profiles")
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    
    if col_preset1.button("👤 Average Customer"):
        st.session_state.preset = "average"
    if col_preset2.button("⚠️ High Risk Profile"):
        st.session_state.preset = "high_risk"
    if col_preset3.button("✅ Loyal Customer"):
        st.session_state.preset = "loyal"
    
    st.markdown("---")
    
    # Set default values based on preset
    if "preset" not in st.session_state:
        st.session_state.preset = "average"
    
    if st.session_state.preset == "high_risk":
        default_age = 28
        default_income = 35000
        default_time = 50.0
        default_purchases = 1
        default_rating = 2.5
        default_browsing = 100.0
        default_review_len = 20.0
        default_sentiment = -2.0
    elif st.session_state.preset == "loyal":
        default_age = 42
        default_income = 85000
        default_time = 450.0
        default_purchases = 15
        default_rating = 4.8
        default_browsing = 1200.0
        default_review_len = 150.0
        default_sentiment = 3.5
    else:  # average
        default_age = 35
        default_income = 60000
        default_time = 200.0
        default_purchases = 5
        default_rating = 4.0
        default_browsing = 500.0
        default_review_len = 80.0
        default_sentiment = 1.0

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=80,
            value=default_age,
            help="Customer's age in years"
        )
        income = st.number_input(
            "Annual Income (€)",
            min_value=10000,
            max_value=200000,
            value=default_income,
            step=1000,
            help="Customer's annual income in euros"
        )
        time_on_site = st.number_input(
            "Time on Site (minutes/month)",
            min_value=0.0,
            max_value=1000.0,
            value=default_time,
            help="Average monthly time spent browsing the site"
        )
        num_purchases = st.number_input(
            "Number of Purchases",
            min_value=0,
            max_value=100,
            value=default_purchases,
            help="Total number of completed purchases"
        )

    with col2:
        avg_rating = st.number_input(
            "Average Purchase Rating",
            min_value=0.0,
            max_value=5.0,
            value=default_rating,
            help="Average rating given by customer (1-5 stars)"
        )
        browsing_events = st.number_input(
            "Browsing Events (count/month)",
            min_value=0.0,
            max_value=10000.0,
            value=default_browsing,
            help="Number of product views, clicks, searches per month"
        )
        review_length = st.number_input(
            "Review Text Length (characters)",
            min_value=0.0,
            max_value=2000.0,
            value=default_review_len,
            help="Average length of product reviews written"
        )
        sentiment_score = st.number_input(
            "Review Sentiment Score",
            min_value=-10.0,
            max_value=10.0,
            value=default_sentiment,
            help="Sentiment analysis score (negative = unhappy, positive = satisfied)"
        )

    # Compute engagement score
    engagement_score = (
        (time_on_site / (time_on_site + 1e-6)) + (browsing_events / (browsing_events + 1e-6))
    ) / 2.0

    st.markdown("---")
    
    if st.button("🔮 Predict Churn Risk", type="primary"):
        # Build feature vector
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

        if proba.shape[1] == 1:
            prob_churn = 0.0
            pred_label = int(model.predict(X_scaled)[0])
        else:
            prob_churn = float(proba[0, 1])
            pred_label = int(model.predict(X_scaled)[0])

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        # Display result in colored box
        col_result1, col_result2 = st.columns([2, 1])
        
        with col_result1:
            if pred_label == 1:
                st.error(f"### ⚠️ High Churn Risk: **{prob_churn*100:.1f}%**")
                st.markdown("""
                **Recommended Actions:**
                - Send personalized retention offer
                - Reach out via customer success team
                - Offer exclusive discount or loyalty reward
                - Schedule feedback call
                """)
            else:
                st.success(f"### ✅ Low Churn Risk: **{(1-prob_churn)*100:.1f}%**")
                st.markdown("""
                **Recommended Actions:**
                - Continue standard engagement
                - Consider upsell opportunities
                - Invite to loyalty program
                - Request product review
                """)
        
        with col_result2:
            st.metric("Churn Probability", f"{prob_churn*100:.1f}%")
            st.metric("Retention Probability", f"{(1-prob_churn)*100:.1f}%")

        # Show raw probabilities
        with st.expander("📊 View Detailed Probabilities"):
            if proba.shape[1] == 1:
                st.json({"P(only_class)": float(proba[0, 0])})
            else:
                prob_df = pd.DataFrame({
                    "Class": ["Will Not Churn (0)", "Will Churn (1)"],
                    "Probability": [1 - prob_churn, prob_churn]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ========== PAGE 2: FEATURE IMPORTANCE ==========
elif page == "📊 Feature Importance":
    st.markdown("### 🔍 Model Explainability: Feature Importance")
    st.markdown("Understanding which customer behaviors most strongly predict churn helps focus retention efforts.")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        # Add human-readable descriptions
        feature_descriptions = {
            "engagement_score": "Combined measure of time on site and browsing activity",
            "Annual_Income": "Customer's annual income",
            "review_length": "Average length of product reviews",
            "Age": "Customer's age",
            "num_purchases": "Total number of purchases made",
            "Time_on_Site": "Average time spent on site per session",
            "browsing_events": "Number of product views and interactions",
            "avg_purchase_rating": "Average rating given to purchased products",
            "review_sentiment_score": "Sentiment analysis of customer reviews"
        }
        
        importance_df["Description"] = importance_df["Feature"].map(feature_descriptions)

        st.markdown("#### 📋 Top Features by Importance")
        st.dataframe(
            importance_df.style.format({"Importance": "{:.4f}"}).background_gradient(subset=["Importance"], cmap="Blues"),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### 📈 Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(importance_df)))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
        ax.set_xlabel("Importance Score", fontsize=12, fontweight='bold')
        ax.set_ylabel("Feature", fontsize=12, fontweight='bold')
        ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.info("""
        **💡 How to Interpret Feature Importance:**
        
        - **Higher importance** = that feature has more influence on churn predictions
        - Random Forest importance measures how much each feature reduces prediction error across all decision trees
        - Focus retention strategies on customers with poor performance in top-ranked features
        
        **Business Insights:**
        - If `engagement_score` is #1: Low engagement is the strongest churn signal → focus on re-engagement campaigns
        - If `review_sentiment_score` ranks high: Negative reviews predict churn → prioritize customer satisfaction improvements
        - If `Annual_Income` is important: Income level affects retention → consider income-targeted pricing strategies
        """)
    else:
        st.warning("Model does not support feature importance (not a tree‑based model).")

# ========== PAGE 3: ABOUT ==========
elif page == "ℹ️ About":
    st.markdown("### 📖 About This Project")
    
    st.markdown("""
    This is a **demonstration churn prediction system** for e-commerce businesses.
    
    #### 🎯 Project Goal
    Help online retailers identify customers at risk of churning before they leave, enabling targeted retention campaigns.
    
    #### 🔧 Technical Stack
    - **Language:** Python 3.12
    - **ML Framework:** scikit-learn (RandomForestClassifier)
    - **Data Processing:** pandas, numpy
    - **Web Framework:** Streamlit
    - **Feature Engineering:** Custom engagement metrics, sentiment analysis
    
    #### 📊 Model Details
    - **Algorithm:** Random Forest Classifier
    - **Features:** 9 behavioral and demographic features
    - **Target:** Binary classification (Churn vs Non-Churn)
    - **Training Data:** Synthetic labels based on engagement and sentiment thresholds
    
    #### ⚠️ Limitations
    - Churn labels are **synthetic** (derived from engagement/sentiment rules) for demonstration purposes
    - Small dataset (13 customers after aggregation)
    - Real production systems would require:
        - Larger datasets with real historical churn outcomes
        - More sophisticated feature engineering
        - Proper validation and calibration
        - A/B testing of retention strategies
    
    #### 🚀 Possible Extensions
    - Replace synthetic labels with real churn data
    - Add SHAP explanations for individual predictions
    - Integrate with CRM systems for automated retention workflows
    - Build customer segmentation and cohort analysis
    - Add time-series features (trend of engagement over time)
    
    #### 📂 Source Code
    - GitHub: https://github.com/sarthakshivaay/ecommerce-churn-prediction-demo/tree/main
    
    ---
    
    **Built as a portfolio project to demonstrate end-to-end ML capabilities:**
    Data pipeline → Feature engineering → Model training → Interactive dashboard
    """)
    
    st.markdown("---")
    st.markdown("*Created by Sarthak Tyagi | January 2026*")
