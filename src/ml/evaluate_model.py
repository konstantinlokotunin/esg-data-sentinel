"""
evaluate_model.py
Kapselt die Evaluierungsmetriken und aggregiert die Reports.
Ermöglicht den dynamischen Export aller Tabellenblätter ohne harte Pfade.
"""

from pathlib import Path
import pandas as pd

class ReportGenerator():
    """Komponente zur hochdimensionalen Feature-Generierung"""
    def __init__(self, df: pd.DataFrame):
        self.df_results = df

    def create_anomaly_summary(self) -> pd.DataFrame:
        """
        Erstellt eine übersichtliche Zusammenfassung der Ergebnisse der Anomalieerkennung.
        """

        self.total_samples = len(self.df_results)
        self.anomaly_count = self.df_results["Is_Anomaly"].sum()
        self.anomaly_rate = round(self.anomaly_count / self.total_samples * 100, 2) if self.total_samples > 0 else 0.0

        summary = pd.DataFrame({
            "Total samples": [self.total_samples],
            "Detected anomalies": [self.anomaly_count],
            "Anomaly Rate %": [self.anomaly_rate]
        })
        return summary

    def get_top_anomalies(self, top_n: int = 20) -> pd.DataFrame:
        """
        Returns the most unusual records based on Anomaly_Score (lower is more unusual).
        """

        top_anomalies = (
            self.df_results[self.df_results["Is_Anomaly"] == 1]
            .sort_values("Anomaly_Score", ascending=True)
            .head(top_n)
        )
        return top_anomalies

    def anomalies_by_sector(self) -> pd.DataFrame:
        """
        Counts anomalies by sector.
        """

        result = (
            self.df_results[self.df_results["Is_Anomaly"] == 1]
            .groupby("Sector")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="Anomaly_Count")
        )
        return result

    def anomalies_by_pollutant_group(self) -> pd.DataFrame:
        """
        Counts anomalies by pollutant group.
        """

        result = (
            self.df_results[self.df_results["Is_Anomaly"] == 1]
            .groupby("Pollutant_Group")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="Anomaly_Count")
        )
        return result

    def anomalies_by_sector_and_pollutant_group(self) -> pd.DataFrame:
        """
        Counts anomalies by pollutant group.
        """

        result = (
            self.df_results[self.df_results["Is_Anomaly"] == 1]
            .groupby(["Sector", "Pollutant_Group"])
            .size()
            .sort_values(ascending=False)
            .reset_index(name="Anomaly_Count")
        )
        return result

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

    def __repr__(self) -> str:
        return f"ReportGenerator(ReadyToExport={len(self.df_results)} rows)"