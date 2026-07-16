"""
evaluate_model.py
Berechnet rein mathematische Auswertungen und aggregiert Ergebnisse.
"""

import pandas as pd

# ==========================================
# REINE FUNKTIONEN (Pflichtanforderung 4)
# ==========================================

def calculate_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Reine Funktion: Berechnet die globalen Anomalie-Kennzahlen."""
    total_samples = len(df)
    anomaly_count = int(df["Is_Anomaly"].sum())
    anomaly_rate = round(anomaly_count / total_samples * 100, 2) if total_samples > 0 else 0.0

    return pd.DataFrame({
        "Total samples": [total_samples],
        "Detected anomalies": [anomaly_count],
        "Anomaly Rate %": [anomaly_rate]
    })

def aggregate_by_column(df: pd.DataFrame, column_name: str | list) -> pd.DataFrame:
    """Reine Funktion: Aggregiert Anomalien nach einer bestimmten Kategorie (Sektor/Gruppe)."""
    return (
        df[df["Is_Anomaly"] == 1]
        .groupby(column_name)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Anomalien_Anzahl")
    )

# ==========================================
# OOP-KOMPONENTE (Pflichtanforderung 3)
# ==========================================

class ReportGenerator:
    """Komponente zur Strukturierung der Reports (Anforderung 3)."""
    
    def __init__(self, df: pd.DataFrame):
        self.df_results = df

    def build_reports(self) -> dict:
        """Sammelt alle berechneten Reports in einem Dictionary."""
        return {
            "Summary": calculate_anomaly_summary(self.df_results),
            "Top_20_Anomalien": self.df_results.sort_values("Anomaly_Score", ascending=True).head(20),
            "Nach_Sektor": aggregate_by_column(self.df_results, "Sector"),
            "Nach_Schadstoffgruppe": aggregate_by_column(self.df_results, "Pollutant_Group"),
            "Nach_Sektor_und_Schadstoffgruppe": aggregate_by_column(self.df_results, ["Sector", "Pollutant_Group"])
        }

    def __repr__(self) -> str:
        return f"ReportGenerator(Rows={len(self.df_results)})"



    def generate_and_save_excel_report(
        self,
        output_path: Path, 
        df_missing_report: pd.DataFrame
        ) -> None:
        """
        Zentrale I/O-Schnittstelle zur Generierung des finalen Excel-Sammelberichts.
        Akzeptiert einen optionalen Missing-Report, um Datenverlust zu verhindern.
        """
    
        # Sicherstellen, dass das Zielverzeichnis existiert
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Berechnungen aus reinen Funktionen sammeln
        summary_report = self.create_anomaly_summary()
        top_anomalies = self.get_top_anomalies()
        sector_report = self.anomalies_by_sector()
        group_report = self.anomalies_by_pollutant_group()
        sector_group_report = self.anomalies_by_sector_and_pollutant_group()

        # I/O-Prozess gebündelt ausführen
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Den Missing-Report aus der früheren Pipeline-Phase als Tabellenblatt integrieren
            if df_missing_report is not None:
                df_missing_report.to_excel(writer, sheet_name="Missing_Data_Analysis", index=True)
            summary_report.to_excel(writer, sheet_name="Top_Anomalies", index=False)
            top_anomalies.to_excel(writer, sheet_name="Top_Anomalies", index=False)
            sector_report.to_excel(writer, sheet_name="By_Sector", index=False)
            group_report.to_excel(writer, sheet_name="By_Group", index=False)
            sector_group_report.to_excel(writer, sheet_name="By_Sector_&_Group", index=False)