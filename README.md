# 🛍️ E‑Commerce Customer Churn Prediction (Demo)

End‑to‑end churn prediction prototype for an online retailer.  
From raw behavioural data → engineered features → RandomForest model → Streamlit app.

---

## 1. Project Overview

This project demonstrates a complete churn prediction workflow:

- Reads semi‑structured customer behaviour data (purchases, browsing history, product reviews).
- Aggregates events into a **customer‑level table** with features such as:
  - Engagement: time on site, browsing activity, engagement score.
  - Purchasing: number of purchases, average purchase rating.
  - Reviews: review text length, simple sentiment score.
- Creates a **synthetic churn label** based on low engagement and negative sentiment.
- Trains a **RandomForestClassifier** to predict churn vs non‑churn.
- Provides an interactive **Streamlit dashboard** with:
  - A **Predict Churn** page (input form + risk score).
  - A **Feature Importance** page (which features drive churn).

This is designed as a **portfolio project** to showcase end‑to‑end ML skills: data engineering, modeling, and a user‑friendly web UI.

---

## 2. Project Structure

```text
ecommerce-churn-prediction/
├── data/
│   ├── raw/                    # original CSV from Kaggle
│   └── processed/              # churn_cleaned.csv (engineered dataset)
├── models/
│   └── churn_model.pkl         # trained model + scaler + feature names
├── notebooks/
│   └── 01_exploration.py       # quick data inspection script
├── src/
│   ├── data_pipeline.py        # cleaning, feature engineering, churn label
│   └── model.py                # model training and saving
├── streamlit_app.py            # main Streamlit dashboard
├── test_app.py                 # minimal Streamlit sanity test
├── requirements.txt            # dependencies
└── README.md
```

---

## 3. Setup & Installation

From the project root:

```bash
# 1. Go to project folder
cd ecommerce-churn-prediction

# 2. (Optional) Create and activate a virtual environment
# python -m venv venv
# On Windows:
#   venv\Scripts\activate
# On macOS/Linux:
#   source venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you only want the core packages:

```bash
python -m pip install streamlit pandas scikit-learn numpy matplotlib
```

---

## 4. Data Preparation

1. Download the **“E‑Commerce Customer Behaviour Dataset”** from Kaggle.(https://www.kaggle.com/datasets/paulsamuelwe/e-commerce-customer-behaviour-dataset?resource=download)  
2. Place the CSV file in:

```text
data/raw/E-Commerce.csv
```

3. Build the processed customer‑level dataset:

```bash
python src/data_pipeline.py
```

This script:

- Cleans and normalizes columns.
- Parses text fields (purchase history, browsing history, product reviews) into numeric features.
- Aggregates to one row per customer.
- Creates a synthetic `Churn` label.
- Saves `data/processed/churn_cleaned.csv`.

---

## 5. Train the Model

Train the Random Forest model and save it:

```bash
python src/model.py
```

This script:

- Loads `data/processed/churn_cleaned.csv`.
- Separates features and target (`Churn`).
- Scales numeric features with `StandardScaler`.
- Trains a `RandomForestClassifier`.
- Saves `models/churn_model.pkl` containing:
  - The trained model
  - The fitted scaler
  - The list of feature names

---

## 6. Run the Streamlit App

Start the dashboard:

```bash
python -m streamlit run streamlit_app.py
```

Then open in your browser:

```text
http://localhost:8501
```

### 6.1. Predict Churn Page

- Input fields:
  - Age  
  - Annual Income  
  - Time on Site (average)  
  - Number of Purchases  
  - Average Purchase Rating  
  - Browsing Events (proxy count)  
  - Review Text Length  
  - Review Sentiment Score
- The app:
  - Recomputes an **engagement_score** from time on site and browsing events.
  - Scales features using the saved scaler.
  - Shows churn risk:
    - “High churn risk” or “Low churn risk”
    - Probabilities for `Churn=1` and `Churn=0`.

### 6.2. Feature Importance Page

- Table of all features sorted by importance.
- Horizontal bar chart showing which features contribute most to the model.
- Short explanation text about how to interpret feature importance for business users.

---

## 7. Notes & Limitations

- The `Churn` label is **synthetic**, based on engagement and sentiment thresholds, and is meant only for demonstration.
- The dataset is small and simplified; the goal is to showcase:
  - Data cleaning and feature engineering from semi‑structured text.
  - Training and saving a model for reuse.
  - Building an interactive dashboard for non‑technical users.

For a production system you would:

- Use real churn labels (e.g. “no purchase in 90 days”).
- Train on a much larger dataset.
- Apply proper validation, calibration, and monitoring.

---

## 8. Possible Extensions

- Replace the synthetic labels with real churn outcomes from transaction history.
- Add SHAP explanations for individual customers to understand single predictions.
- Integrate with a CRM or email tool to export high‑risk customers for retention campaigns.
- Containerize the app with Docker and deploy it to a cloud environment.
