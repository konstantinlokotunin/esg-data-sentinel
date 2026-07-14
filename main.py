"""
main.py
Das zentrale Kontrollzentrum des ESG Data Sentinel.
Orchestriert die Pakete 'pipeline' und 'ml' vollständig unter Einhaltung aller Vorgaben.
"""

import logging
from pathlib import Path
from src.pipeline.errors import InvalidFileFormat, DataValidationError, EmptyDatasetError
from src.pipeline.extract import extract_data
from src.pipeline.cleaning import generate_missing_report, clean_data
from src.ml.transform import add_pollutant_groups, add_features_for_isolation_forest
from src.ml.train_model import prepare_model_data, train_isolation_forest
from src.ml.evaluate_model import generate_and_save_excel_report
from src.ml.visualization import generate_and_save_all_plots

def main():
    # Logger für das einheitliche und übersichtliche Protokollieren
    logger = logging.getLogger("ESG_Sentinel_Core")

    # Dynamische Pfadauflösung relativ zur Projektwurzel
    base_dir = Path(__file__).parent
    
    # Eingabedatei im data-Ordner
    input_file = base_dir / "data" / "raw" / "industrial_releases_of_pollutants_to_air.csv"
    
    # Ausgabeordner für Berichte und Grafiken
    output_excel = base_dir / "outputs" / "anomaly_reports.xlsx"
    output_figures = base_dir / "outputs" / "figures"

    logger.info("=== ESG Data Sentinel Pipeline gestartet ===")

    try:
        # --- PHASE 1: EXTRACTION ---
        logger.info("Schritt 1: Extrahiere Rohdaten über Lazy-Evaluation-Generatoren...")
        df_raw = extract_data(input_file)
        logger.info(f"Rohdaten erfolgreich geladen. Dimensionen: {df_raw.shape}")

        # --- PHASE 2: CLEANING & MISSING REPORT ---
        logger.info("""
                Schritt 2: Analysiere Datenqualität, filtriere und bereinige Datensätze,
                erstelle eine Übersicht der fehlenden Werte...
                """)
        missing_report = generate_missing_report(df_raw)
        df_cleaned = clean_data(df_raw)
        logger.info(f"Daten erfolgreich bereinigt. Neue Shape: {df_cleaned.shape}")

        # --- PHASE 3: FEATURE ENGINEERING ---
        logger.info("""
                Schritt 3: Führe Schadstoff-Klassifizierung und Feature Engineering durch...
                """)
        df_grouped = add_pollutant_groups(df_cleaned)
        df_features = add_features_for_isolation_forest(df_grouped)
        logger.info(f"Schadstoff-Klassifizierung und Feature Engineering abgeschlossen. Neue Spaltenanzahl: {df_features.shape[1]}")

        # --- PHASE 4: MACHINE LEARNING TRAINING ---
        logger.info("""
                Schritt 4: Extrahiere numerische Matrix und trainiere Isolation Forest...
                """)
        X_matrix = prepare_model_data(df_features)  
        # Das Modell wird trainiert und gibt die Ergebnisse sowie die Modell-Objekte zurück                       
        df_results, trained_scaler, trained_model = train_isolation_forest(
            df=df_features, 
            X=X_matrix, 
            contamination=0.05
            )
        logger.info("Modelltraining abgeschlossen.")
        logger.info(f"""Erkannte Anomalien: {df_results["Is_Anomaly"].sum()} von
                    {len(df_results)} Datensätzen ({df_results["Is_Anomaly"].mean() * 100:.2f}%)""")
    
        # --- PHASE 5: EVALUATION & EXCEL-REPORTING ---
        logger.info("Schritt 6: Schritt 5: Generiere konsolidierten Multi-Sheet Excel-Report...")
        generate_and_save_excel_report(
            df= df_results,
            output_path=output_excel,
            df_missing_report=missing_report
        )
        logger.info(f"Multi-Sheet Excel-Bericht erfolgreich exportiert nach: {output_excel}")

        # --- PHASE 6: VISUALIZATIONS ---
        logger.info("Schritt 6: Erzeuge explorative Analyseplots...")
        generate_and_save_all_plots(
                df=df_results, 
                output_dir=output_figures
            )
        logger.info(f"Grafiken erfolgreich exportiert nach: {output_figures}")

        logger.info("=== ESG Data Sentinel Pipeline Pipeline erfolgreich beendet ===")

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
    main()