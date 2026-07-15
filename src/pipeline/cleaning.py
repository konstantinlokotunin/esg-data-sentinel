"""
cleaning.py
Kapselt die Datenbereinung(-slogik) in spezialisierten Komponenten-Klassen.
"""

import pandas as pd
from .errors import EmptyDatasetError

class DataCleaner:
    """Komponente zur Bereinigung und Filterung der extrahierten Rohdaten"""
    def __init__(self, df: pd.DataFrame):
        self.df_raw = df

    def __repr__(self) -> str:
        return f"DataCleaner(RecordsIn={len(self.df_raw)})"

    def generate_missing_report(self) -> pd.DataFrame:
        """Erstellt ein DataFrame mit einer Übersicht der fehlenden Werte."""
        missing_report = pd.DataFrame({
            "missing_values": self.df_raw.isna().sum(),
            "missing_percent": self.df_raw.isna().mean() * 100
        })
        return missing_report.sort_values("missing_values", ascending=False)

    def clean_data(self) -> pd.DataFrame:
        """Wählt relevante Spalten, formatiert Datentypen und filtert nach Region."""

        relevant_features = {
            "countryName": "Country",
            "reportingYear": "Year",
            "EPRTR_SectorCode": "Code",
            "EPRTR_SectorName": "Sector",
            "facilityName": "Facility",
            "Pollutant": "Pollutant",
            "Releases": "Amount",
        }

        # 1. Spalten filtern (und sicherstellen, dass die gefilterten Spalten existieren)
        existing_cols = [col for col in relevant_features.keys() if col in self.df_raw.columns]
        self.df_raw = self.df_raw[existing_cols].copy()

        # 2. Spalten umbenennen
        self.df_raw = self.df_raw.rename(columns=relevant_features)

        # 3. Numerische Werte sicher konvertieren
        self.df_raw["Amount"] = pd.to_numeric(self.df_raw["Amount"], errors="coerce")
        # 4. Fehlende Werte in der Zielgröße entfernen
        self.df_raw = self.df_raw.dropna(subset=["Amount"])

        # 5. Kategoriale Spalten bereinigen und standardisieren
        cat_cols = ["Country", "Year", "Code", "Sector", "Facility", "Pollutant"]
        self.df_raw[cat_cols] = self.df_raw[cat_cols].astype(str).apply(lambda x: x.str.strip().replace({"nan": "Unknown", "": "Unknown"}))

        # 6. Geografischen Fokus einschränken
        self.df_raw = self.df_raw[self.df_raw["Country"] == "Austria"].copy()

        # 7. Validierung des Endergebnisses mit eigener Exception
        if self.df_raw.empty:
            raise EmptyDatasetError("Bereinigung fehlgeschlagen: Keine Datensätze für 'Austria' vorhanden.")
        
        df_cleaned = self.df_raw
        return df_cleaned