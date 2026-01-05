import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path

DATA_PATH = "data/processed/churn_cleaned.csv"
MODEL_PATH = "models/churn_model.pkl"


def load_data(path: str) -> pd.DataFrame:
    print("[MODEL] Loading processed data...")
    df = pd.read_csv(path)
    print(f"[MODEL] Data shape: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    print("[MODEL] Preparing features and target...")

    X = df.drop(columns=["Churn", "Customer_ID"])
    y = df["Churn"]

    feature_names = X.columns.tolist()
    print("[MODEL] Features:", feature_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[MODEL] Feature matrix shape:", X_scaled.shape)
    return X_scaled, y, scaler, feature_names


def train_model(X, y) -> RandomForestClassifier:
    print("[MODEL] Training RandomForestClassifier on full dataset...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)
    print("[MODEL] Training done.")
    return model


def save_model(model, scaler, feature_names, path: str):
    Path("models").mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_names": feature_names,
            },
            f,
        )
    print(f"[MODEL] Model saved to {path}")


def main():
    df = load_data(DATA_PATH)
    X, y, scaler, feature_names = prepare_features(df)
    model = train_model(X, y)
    save_model(model, scaler, feature_names, MODEL_PATH)


if __name__ == "__main__":
    main()
