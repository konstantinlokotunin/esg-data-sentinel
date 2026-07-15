"""
main.py
Das zentrale RAG-ähnliche Kontrollzentrum des ESG Data Sentinel.
Verbindet spezialisierte Komponenten-Klassen über tiefe Komposition 
"""

import logging
from pathlib import Path

# Komponenten-Klassen aus den Paketen importieren
from src.pipeline.errors import InvalidFileFormat, DataValidationError, EmptyDatasetError
from src.pipeline.extract import DataLoader
from src.pipeline.cleaning import DataCleaner
from src.ml.transform import DataTransformer
from src.ml.train_model import ModelPipeline
from src.ml.evaluate_model import ReportGenerator
from src.ml.visualization import DashboardRenderer

# Root-Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ESG_Sentinel_Core")

class ESGDataSentinel:
    """
    Zentraler System-Orchestrator.
    Baut über Komposition eine mehrstufige Pipeline aus spezialisierten Unterklassen auf.
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.input_file = self.base_dir / "data" / "raw" / "industrial_releases_of_pollutants_to_air.csv"
        self.output_excel = self.base_dir / "outputs" / "anomaly_reports.xlsx"
        self.output_figures = self.base_dir / "outputs" / "figures"

        # --- KOMPOSITION (RAG-Architektur-Style) ---
        # Die Pipeline besitzt und steuert eigenständige Komponenten-Klassen als Sub-Systeme
        self.loader = DataLoader(file_path=self.input_file)
        self.cleaner = None
        self.transformer = None
        self.model_pipeline = None
        self.report_generator = None
        self.dashboard = None

    def execute_pipeline(self) -> None:
        """Führt die Pipeline geschützt aus und protokolliert jeden Teilschritt."""
        logger.info("=== ESG Data Sentinel Pipeline gestartet ===")

        try:
            # --- PHASE 1: EXTRACTION ---
            logger.info("Schritt 1: Extrahiere Rohdaten über Lazy-Evaluation-Generatoren...")
            df_raw = self.loader.extract_data()
            logger.info(f"Rohdaten erfolgreich geladen. Dimensionen: {df_raw.shape}")

            # --- PHASE 2: CLEANING & MISSING REPORT ---
            logger.info("Schritt 2: Führe eine Filterung und Bereinigung der Datensätze durch und erstelle eine Übersicht der fehlenden Werte...")
            self.cleaner = DataCleaner(df_raw) # Dynamische Kompositions-Injektion
            logger.info(f"Komponente aktiv: {self.cleaner}")
            missing_report = self.cleaner.generate_missing_report()
            df_cleaned = self.cleaner.clean_data()
            logger.info(f"Daten erfolgreich bereinigt. Dimensionen: {df_cleaned.shape}")

            # --- PHASE 3: FEATURE ENGINEERING ---
            logger.info("Schritt 3: Führe Schadstoff-Klassifizierung und Feature Engineering durch...")
            self.transformer = DataTransformer(df_cleaned)
            df_transformed = self.transformer.add_features()
            logger.info(f"Schadstoff-Klassifizierung und Feature Engineering abgeschlossen. Dimensionen: {df_transformed.shape}")

            # --- PHASE 4: MACHINE LEARNING TRAINING ---
            logger.info("Schritt 4: Extrahiere numerische Matrix und trainiere Isolation Forest...")
            self.model_pipeline = ModelPipeline(df_transformed)
            df_results, trained_scaler, trained_model = self.model_pipeline.train_isolation_forest()
            logger.info("Modelltraining abgeschlossen.")
            logger.info(f"Erkannte Anomalien: {df_results["Is_Anomaly"].sum()} von {len(df_results)} Datensätzen ({df_results["Is_Anomaly"].mean() * 100:.1f}%)")
    
            # --- PHASE 5: EVALUATION & EXCEL-REPORTING ---
            logger.info("Schritt 5: Generiere konsolidierten Multi-Sheet Excel-Report...")
            self.report_generator = ReportGenerator(df_results)
            self.report_generator.generate_and_save_excel_report(self.output_excel, missing_report)
            logger.info(f"Multi-Sheet Excel-Bericht erfolgreich exportiert nach: \"{self.output_excel}\".")

            # --- PHASE 6: VISUALIZATIONS ---
            logger.info("Schritt 6: Erzeuge explorative Analyseplots...")
            self.dashboard = DashboardRenderer(df_results)
            self.dashboard.generate_and_save_plots(self.output_figures)
            logger.info(f"Grafiken erfolgreich exportiert nach: \"{self.output_figures}\".")

            logger.info("=== ESG Data Sentinel Pipeline erfolgreich und fehlerfrei beendet ===")

        # --- PHASE 7: ROBUSTE AUSNAHMEBEHANDLUNG ---
        except InvalidFileFormat as e:
            logger.error(f"Pipeline-Abbruch: Ungültiges Dateiformat erkannt -> {e}")
        except EmptyDatasetError as e:
            logger.error(f"Pipeline-Abbruch: Keine Daten nach der Bereinigung übrig -> {e}")
        except DataValidationError as e:
            logger.error(f"Pipeline-Abbruch: Kritischer Validierungsfehler im Datenstrom -> {e}")
        except Exception as e:
            logger.critical(f"Unerwarteter Systemfehler außerhalb der Applikationslogik: {e}", exc_info=True)

if __name__ == "__main__":
    # Instanziierung des zentralen Pipeline-Verwalters
    sentinel = ESGDataSentinel()
    sentinel.execute_pipeline()