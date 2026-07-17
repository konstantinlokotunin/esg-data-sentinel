"""
train.py
Zentrales Skript zum Trainieren, Evaluieren, Visualieren und Speichern des Modells und der Vorhersagen.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

def save_excel_report(output_path: Path, reports: dict) -> None:
    """Zentrale I/O-Schnittstelle: Schreibt die berechneten Reports sauber in Excel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in reports.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def save_model_artefacts(model_path: Path, model: IsolationForest, scaler_path: Path, scaler: StandardScaler) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def main():
    base_dir = Path(__file__).parent
    input_file = base_dir / "data" / "raw" / "industrial_releases_of_pollutants_to_air.csv"
    
    # Zielpfade definieren
    model_path = base_dir / "models" / "model.joblib"
    scaler_path = base_dir / "models" / "scaler.joblib"
    report_path = base_dir / "reports" / "anomaly_report.xlsx"
    plot_path = base_dir / "reports" / "figures"

    logger.info("Starte ETL-Pipeline...")
    df_raw = DataLoader(input_file).extract_data()
    df_cleaned = DataCleaner(df_raw).clean_data()
    df_transformed = DataTransformer(df_cleaned).transform_data()

    logger.info("Trainiere Isolation Forest Modell...")
    pipeline = ModelPipeline(df_transformed)

    # Modell und Scaler aus train_model.py abholen
    model, scaler, X_train, X_test = pipeline.train_pipeline()

    # --- EVALUIERUNG AUF TESTDATEN ---
    logger.info("Evaluiere Modell auf Testdaten...")
    # Isolation Forest: 1 = Normal, -1 = Anomalie
    test_predictions = model.predict(X_test)
    test_scores = model.decision_function(X_test)

    # Gezielte Zuweisung der Vorhersagen und Anomaly Scores über den Index von X_test
    df_results = df_transformed.loc[X_test.index].copy()
    # Ummappen auf Standard-Binärklassifikation: 0 = Normal, 1 = Anomalie
    df_results["Is_Anomaly"] = [1 if x == -1 else 0 for x in test_predictions]
    df_results["Anomaly_Score"] = test_scores

    # Reports generieren
    reporter = ReportGenerator(df_results)
    all_reports = reporter.build_reports()
    
    # Excel-Export ausführen (Zentralisiertes I/O)
    save_excel_report(report_path, all_reports)
    logger.info(f"Excel-Report erfolgreich exportiert nach: \"{report_path}\".")

    # Grafiken generieren & exportieren (Zentralisiertes I/O)
    plotter = DashboardRenderer(df_results)
    plotter.generate_and_save_plots(plot_path)
    logger.info(f"Visualisierungen erfolgreich exportiert nach: \"{plot_path}\".")

    # Modell-Artefakte speichern
    save_model_artefacts(model_path, model, scaler_path, scaler)
    logger.info(f"Modell-Artefakte erfolgreich exportiert nach: \"{model_path}\".")

if __name__ == "__main__":
    main()