import pandas as pd

df = pd.read_csv("../data/processed/churn_cleaned.csv")

print("Churn distribution:")
print(df["Churn"].value_counts(normalize=True))

print("\nEngagement stats:")
print(df["engagement_score"].describe())

print("\nSentiment stats:")
print(df["review_sentiment_score"].describe())
