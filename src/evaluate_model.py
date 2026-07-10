import pandas as pd
from pathlib import Path

file_path = r"C:\Users\konst\Documents\Python Projects\ESG Data Sentinel\esg-data-sentinel\outputs"

OUTPUT_DIR = Path(file_path)

def create_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a simple summary of anomaly detection results.
    """

    total_samples = len(df)
    anomaly_count = df["Is_Anomaly"].sum()
    anomaly_rate = round(anomaly_count / total_samples * 100, 2)

    summary = pd.DataFrame({
        "Total samples": total_samples,
        "Detected anomalies": anomaly_count,
        "Anomaly Rate %": anomaly_rate
        }, index=[0])

    return summary

def get_top_anomalies(df: pd.DataFrame, top_n=20) -> pd.DataFrame:
    """
    Returns the most unusual records based on Anomaly_Score.
    Lower Anomaly_Score means more unusual.
    """

    top_anomalies = (
        df[df["Is_Anomaly"] == 1]
        .sort_values("Anomaly_Score")
        .head(top_n)
    )

    return top_anomalies

def anomalies_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts anomalies by sector.
    """

    result = (
        df[df["Is_Anomaly"] == 1]
        .groupby("Sector")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Anomaly_Count")
    )

    return result

def anomalies_by_pollutant_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts anomalies by pollutant group.
    """

    result = (
        df[df["Is_Anomaly"] == 1]
        .groupby("Pollutant_Group")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Anomaly_Count")
    )

    return result

def anomalies_by_sector_and_pollutant_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts anomalies by pollutant group.
    """

    result = (
        df[df["Is_Anomaly"] == 1]
        .groupby(["Sector", "Pollutant_Group"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Anomaly_Count")
    )

    return result

def save_anomaly_reports(df: pd.DataFrame):
    """
    Saves anomaly summary reports.
    """

    summary = create_anomaly_summary(df)
    top_anomalies = get_top_anomalies(df)
    sector_report = anomalies_by_sector(df)
    group_report = anomalies_by_pollutant_group(df)
    sector_group_report = anomalies_by_sector_and_pollutant_group(df)

    summary.to_csv(OUTPUT_DIR / "anomaly_summary.csv", sep=";", index=False)
    top_anomalies.to_csv(OUTPUT_DIR / "top_anomalies.csv", sep=";", index=False)
    sector_report.to_csv(OUTPUT_DIR / "anomalies_by_sector.csv", sep=";", index=False)
    group_report.to_csv(OUTPUT_DIR / "anomalies_by_pollutant_group.csv", sep=";", index=False)
    sector_group_report.to_csv(OUTPUT_DIR / "anomalies_by_sector_and_pollutant_group.csv", sep=";", index=False)

    print("Saved anomaly reports:")
    print("-", OUTPUT_DIR / "anomaly_summary.csv")
    print("-", OUTPUT_DIR / "top_anomalies.csv")
    print("-", OUTPUT_DIR / "anomalies_by_sector.csv")
    print("-", OUTPUT_DIR / "anomalies_by_pollutant_group.csv")
    print("-", OUTPUT_DIR / "anomalies_by_sector_and_pollutant_group.csv")