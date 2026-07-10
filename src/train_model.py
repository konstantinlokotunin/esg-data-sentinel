import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares numeric model input for IsolationForest.
    """

    model_features = [
        "Amount_Log",
        "Dev_from_Group_Median",
        "Dev_from_Sector_Median",
        "Dev_from_Sector_Group_Median",
        "YoY_Change_Pct",
    ]

    X = df[model_features].copy()

    return X

def train_isolation_forest(df, X: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an IsolationForest model on numeric anomaly-detection features.
    """

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_train)
    
    X_test = scaler.transform(X)

    df["Is_Anomaly"] = model.predict(X_test)
    df["Is_Anomaly"] = df["Is_Anomaly"].map({
        1: 0,
        -1: 1
    })
    df["Anomaly_Score"] = model.decision_function(X_test)

    print("\nAnomaly detection completed.")
    print("Detected anomalies:", df["Is_Anomaly"].sum())
    print("Anomaly rate:", round(df["Is_Anomaly"].mean() * 100, 2), "%")

    return df