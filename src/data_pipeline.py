import pandas as pd
import numpy as np
from datetime import datetime


RAW_PATH = "data/raw/E-commerce.csv"
PROCESSED_PATH = "data/processed/churn_cleaned.csv"


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw e-commerce customer behaviour data from CSV."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw data loaded. Shape: {df.shape}")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle missing values, remove duplicates."""
    print("[INFO] Starting basic cleaning...")

    # Normalize column names
    df = df.rename(columns=lambda c: c.strip().replace(" ", "_"))

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows (exact).")

    # Handle missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"[INFO] Filled NaNs in numeric column '{col}' with median={median_value}.")

    # Handle missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            mode_value = df[col].mode().iloc[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"[INFO] Filled NaNs in categorical column '{col}' with mode='{mode_value}'.")

    print("[INFO] Basic cleaning done.")
    return df


def parse_purchase_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn 'Purchase_History' free text/JSON-ish into simple numeric features.
    - num_purchases: count of products mentioned
    - avg_rating: average rating if patterns like 'Rating: X' or JSON ratings exist
    """
    print("[INFO] Parsing Purchase_History into numeric features...")

    if "Purchase_History" not in df.columns:
        print("[WARN] Column 'Purchase_History' not found. Skipping.")
        df["num_purchases"] = 0
        df["avg_purchase_rating"] = 0.0
        return df

    def count_items(x: str) -> int:
        if pd.isna(x):
            return 0
        # crude heuristic: count occurrences of 'Product'
        return str(x).count("Product")

    def extract_rating_mean(x: str) -> float:
        if pd.isna(x):
            return 0.0
        s = str(x)
        # very simple: look for 'Rating:' patterns and average numbers after them
        ratings = []
        parts = s.split("Rating:")
        for part in parts[1:]:
            # take first number after 'Rating:'
            tokens = part.strip().split()
            for t in tokens:
                try:
                    val = float(t.strip(".,"))
                    ratings.append(val)
                    break
                except ValueError:
                    continue
        if len(ratings) == 0:
            return 0.0
        return float(np.mean(ratings))

    df["num_purchases"] = df["Purchase_History"].apply(count_items)
    df["avg_purchase_rating"] = df["Purchase_History"].apply(extract_rating_mean)

    print("[INFO] Created 'num_purchases' and 'avg_purchase_rating'.")
    return df


def parse_browsing_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn 'Browsing_History' into a simple numeric feature:
    - browsing_events: length of the text (proxy for engagement)
    """
    print("[INFO] Parsing Browsing_History into numeric features...")

    if "Browsing_History" not in df.columns:
        print("[WARN] Column 'Browsing_History' not found. Skipping.")
        df["browsing_events"] = 0
        return df

    df["browsing_events"] = df["Browsing_History"].astype(str).str.len()
    print("[INFO] Created 'browsing_events' from text length.")
    return df


def parse_product_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn 'Product_Reviews' into:
    - review_length: text length
    - review_sentiment_proxy: count of positive vs negative words (very crude)
    """
    print("[INFO] Parsing Product_Reviews into numeric features...")

    if "Product_Reviews" not in df.columns:
        print("[WARN] Column 'Product_Reviews' not found. Skipping.")
        df["review_length"] = 0
        df["review_sentiment_score"] = 0
        return df

    positive_words = ["great", "good", "excellent", "amazing", "love", "comfortable"]
    negative_words = ["bad", "poor", "terrible", "hate", "uncomfortable"]

    def sentiment_score(text: str) -> int:
        if pd.isna(text):
            return 0
        t = str(text).lower()
        score = 0
        for w in positive_words:
            score += t.count(w)
        for w in negative_words:
            score -= t.count(w)
        return score

    df["review_length"] = df["Product_Reviews"].astype(str).str.len()
    df["review_sentiment_score"] = df["Product_Reviews"].apply(sentiment_score)

    print("[INFO] Created 'review_length' and 'review_sentiment_score'.")
    return df


def aggregate_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple rows per Customer_ID into a single row with summary stats.
    """
    print("[INFO] Aggregating to customer level...")

    if "Customer_ID" not in df.columns:
        raise ValueError("Expected column 'Customer_ID' after renaming, but not found.")

    agg_df = df.groupby("Customer_ID").agg(
        Age=("Age", "max"),
        Annual_Income=("Annual_Income", "max"),
        Time_on_Site=("Time_on_Site", "mean"),
        num_purchases=("num_purchases", "sum"),
        avg_purchase_rating=("avg_purchase_rating", "mean"),
        browsing_events=("browsing_events", "mean"),
        review_length=("review_length", "mean"),
        review_sentiment_score=("review_sentiment_score", "mean"),
    ).reset_index()

    print(f"[INFO] Aggregated to customer-level shape: {agg_df.shape}")
    return agg_df


def engineer_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a synthetic churn label.

    - Build engagement_score from Time_on_Site and browsing_events.
    - Mark churn (1) if engagement is in bottom 30% OR sentiment clearly negative.
    """
    print("[INFO] Creating synthetic Churn label...")

    # Build engagement_score if not present
    if "engagement_score" not in df.columns:
        time_norm = df["Time_on_Site"] / (df["Time_on_Site"].max() + 1e-6)
        browse_norm = df["browsing_events"] / (df["browsing_events"].max() + 1e-6)
        df["engagement_score"] = (time_norm + browse_norm) / 2.0

    df["engagement_score"] = df["engagement_score"].fillna(0)

    # Thresholds
    low_eng_threshold = df["engagement_score"].quantile(0.3)
    neg_sent_threshold = 0  # sentiment < 0 => negative

    churn_condition = (df["engagement_score"] <= low_eng_threshold) | (
        df["review_sentiment_score"] < neg_sent_threshold
    )

    df["Churn"] = churn_condition.astype(int)

    print(
        "[INFO] Synthetic 'Churn' column updated. Churn rate:",
        df["Churn"].mean().round(2),
        "(fraction of customers labeled 1)",
    )
    return df




def run_pipeline():
    """Main entry point for the data pipeline."""
    print("[PIPELINE] Loading raw data...")
    df = load_raw_data(RAW_PATH)

    print("[PIPELINE] Running basic cleaning...")
    df = basic_cleaning(df)

    print("[PIPELINE] Parsing text fields into numeric features...")
    df = parse_purchase_history(df)
    df = parse_browsing_history(df)
    df = parse_product_reviews(df)

    print("[PIPELINE] Aggregating to customer level...")
    df = aggregate_by_customer(df)

    print("[PIPELINE] Engineering churn label...")
    df = engineer_churn_label(df)

    print(f"[PIPELINE] Saving cleaned data to: {PROCESSED_PATH}")
    df.to_csv(PROCESSED_PATH, index=False)
    print("[PIPELINE] Done.")


if __name__ == "__main__":
    run_pipeline()
