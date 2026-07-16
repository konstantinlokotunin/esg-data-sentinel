"""
train.py
Zentrales Ausführungsskript. Orchestriert das Laden, Bereinigen, Transformieren,
Trainieren sowie das anschließende Speichern der ML-Artefakte und Ergebnisse.
"""

import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.pipeline.extract import DataLoader
from src.pipeline.cleaning import DataCleaner
from src.ml.transform import DataTransformer
from src.ml.train_model import ModelPipeline

# Root-Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ESG_Sentinel_Core")

def main():
    # Pfade dynamisch definieren
    base_dir = Path(__file__).parent
    input_file = base_dir / "data" / "raw" / "industrial_releases_of_pollutants_to_air.csv"

    # Zielpfade für Artefakte und Ergebnisse
    dump_model = base_dir / "models" / "model.joblib"
    dump_scaler = base_dir / "models" / "scaler.joblib"
    dump_predictions = base_dir / "models" / "anomaly_predictions_test.csv"

    logger.info("Starte End-to-End ETL und ML-Pipeline...")
    # 1. ETL & Feature Engineering
    df_raw = DataLoader(file_path=input_file).extract_data()
    df_cleaned = DataCleaner(df_raw).clean_data()
    df_transformed = DataTransformer(df_cleaned).add_features()

    # 2. Modelltraining & Inferenz (Sauber gekapselt über die Komponente)
    pipeline = ModelPipeline(df_transformed)
    df_results, scaler, model = pipeline.train_isolation_forest()

    # 3. Ergebnisse persistieren (Modell-Artefakte)
    joblib.dump(model, dump_model)
    joblib.dump(scaler, dump_scaler)
    logger.info(f"Modell-Artefakte erfolgreich gespeichert in: {dump_model}")

    # 4. Vorhersagen abspeichern
    # Nur die Zeilen speichern, die tatsächlich im Test-Set waren (nicht-NaN)
    df_test_results = df_results.dropna(subset=["Is_Anomaly"])
    df_test_results.to_csv(dump_predictions, index=False)
    logger.info(f"Test-Vorhersagen erfolgreich exportiert nach: {dump_predictions}")
    logger.info("ETL und ML-Pipeline erfolgreich ohne Fehler beendet.")

if __name__ == "__main__":
    main()