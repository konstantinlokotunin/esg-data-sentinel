"""
cleaning.py
Kapselt die reine Transformations- und Bereinigungslogik
"""

import logging
import pandas as pd
from pathlib import Path
from errors import EmptyDatasetError

# Logger für das einheitliche und übersichtliche Protokollieren kritischer Validierungsfehler
logger = logging.getLogger(__name__)

file_path = r"C:\Users\konst\Documents\Python Projects\ESG Data Sentinel\esg-data-sentinel\outputs"

OUTPUT_DIR = Path(file_path)
output_path = OUTPUT_DIR / "anomaly_reports.xlsx"

def generate_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt ein DataFrame mit einer Übersicht der fehlenden Werte.

    Args:
        df (pd.DataFrame): Das zu prüfende DataFrame.

    Returns:
        pd.DataFrame: Report mit Anzahl und Prozent fehlender Werte pro Spalte.

    """

    report = pd.DataFrame({
        "missing_values": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100
    })

    return report.sort_values("missing_values", ascending=False)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wählt relevante Spalten, formatiert Datentypen und filtert nach Region.

    Args:
        df (pd.DataFrame): Die extrahierten Rohdaten.

    Returns:
        pd.DataFrame: Das bereinigte und gefilterte DataFrame.

    Raises:
        EmptyDatasetError: Wenn nach der Filterung keine Daten übrig bleiben.
    """

    columns_to_keep = {
        "countryName": "Country",
        "reportingYear": "Year",
        "EPRTR_SectorCode": "Code",
        "EPRTR_SectorName": "Sector",
        "facilityName": "Facility",
        "Pollutant": "Pollutant",
        "Releases": "Amount",
    }

    # 1. Spalten filtern (und sicherstellen, dass die gefilterten Spalten existieren)
    existing_cols = [col for col in columns_to_keep.keys() if col in df.columns]
    df = df[existing_cols].copy()

    # 2. Spalten umbenennen
    df = df.rename(columns=columns_to_keep)

    # 3. Numerische Werte sicher konvertieren
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    # Fehlende Werte in der Zielgröße entfernen
    df = df.dropna(subset=["Amount"])

    # 4. Kategoriale Spalten bereinigen und standardisieren
    cat_cols = ["Country", "Year", "Code", "Sector", "Facility", "Pollutant"]
    df[cat_cols] = df[cat_cols].astype(str).apply(lambda x: x.str.strip().replace(
        {"nan": "Unknown", "": "Unknown"}))

    # 5. Geografischen Fokus einschränken
    df = df[df["Country"] == "Austria"].copy()

    # 6. Validierung des Endergebnisses mit eigener Exception
    if df.empty:
        raise EmptyDatasetError("Bereinigung fehlgeschlagen: Keine Datensätze für 'Austria' vorhanden.")

    # Datenschonendes Logging
    logger.info(f"Daten erfolgreich bereinigt. Shape: {df.shape}")
    
    return df
