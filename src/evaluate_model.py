"""
evaluate_model.py
Kapselt die Evaluierungsmetriken und aggregiert die Reports.
Ermöglicht den dynamischen Export aller Tabellenblätter ohne harte Pfade.
"""

import logging
from pathlib import Path
import pandas as pd

# Logger für das einheitliche und übersichtliche Protokollieren
logger = logging.getLogger(__name__)

def create_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt eine einfache Zusammenfassung der Ergebnisse der Anomalieerkennung.
    """

    total_samples = len(df)
    anomaly_count = df["Is_Anomaly"].sum()
    anomaly_rate = round(anomaly_count / total_samples * 100, 2) if total_samples > 0 else 0.0

    summary = pd.DataFrame({
        "Total samples": [total_samples],
        "Detected anomalies": [anomaly_count],
        "Anomaly Rate %": [anomaly_rate]
    })
    return summary

def get_top_anomalies(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Returns the most unusual records based on Anomaly_Score (lower is more unusual).
    """

    top_anomalies = (
        df[df["Is_Anomaly"] == 1]
        .sort_values("Anomaly_Score", ascending=True)
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

def generate_and_save_excel_report(
    df: pd.DataFrame, 
    output_path: Path, 
    df_missing_report: pd.DataFrame
) -> None:
    """
    Zentrale I/O-Schnittstelle zur Generierung des finalen Excel-Sammelberichts.
    Akzeptiert einen optionalen Missing-Report, um Datenverlust zu verhindern.

    Args:
        df (pd.DataFrame): Das DataFrame nach erfolgreicher Inferenz.
        output_path (Path): Zielpfad für die Excel-Datei.
        df_missing_report (pd.DataFrame, optional): Der Report aus dem Cleaning Modul.
    """
    
    # Sicherstellen, dass das Zielverzeichnis existiert
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Berechnungen aus reinen Funktionen sammeln
    summary = create_anomaly_summary(df)
    top_anomalies = get_top_anomalies(df)
    sector_report = anomalies_by_sector(df)
    group_report = anomalies_by_pollutant_group(df)
    sector_group_report = anomalies_by_sector_and_pollutant_group(df)

    # I/O-Prozess gebündelt ausführen
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        top_anomalies.to_excel(writer, sheet_name="Top_Anomalies", index=False)
        sector_report.to_excel(writer, sheet_name="By_Sector", index=False)
        group_report.to_excel(writer, sheet_name="By_Group", index=False)
        sector_group_report.to_excel(writer, sheet_name="By_Sector_&_Group", index=False)

    # Den Missing-Report aus der früheren Pipeline-Phase als Tabellenblatt integrieren
        if df_missing_report is not None:
            df_missing_report.to_excel(writer, sheet_name="Missing_Data_Analysis", index=True)

    # Datenschonendes Logging
    logger.info(f"Multi-Sheet Excel-Bericht erfolgreich exportiert nach: {output_path}")