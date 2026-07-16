"""
train_model.py
Kapselt das Modell-Training und die Ergebnisse, inkl. der ML-Artefakte für die Wiederverwendbarkeit,
in spezialisierten Komponenten-Klassen.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class ModelPipeline():
    """Komponente zur hochdimensionalen Feature-Generierung"""
    def __init__(self, df: pd.DataFrame):
        self.df_transformed = df
        self.features = [
            "Amount_Log",
            "Dev_from_Group_Median",
            "Dev_from_Sector_Median",
            "Dev_from_Sector_Group_Median",
            "YoY_Change_Pct",
        ]

    def prepare_model_data(self):
        """
        Extrahiert die numerischen Features für den Isolation Forest.
        """

        # Sicherstellen, dass alle geforderten Spalten im DataFrame existieren
        missing_features = [col for col in self.features if col not in self.df_transformed.columns]
        if missing_features:
            raise KeyError(f"Fehlende Features im DataFrame für ML-Training: {missing_features}")

    def train_isolation_forest(self) -> Tuple[pd.DataFrame, StandardScaler, IsolationForest]:
        """
        Trainiert das Anomalieerkennungsmodell und fügt dem transformierten DataFrame die Ergebnisse hinzu.
        Gibt die trainierten Artefakte für die spätere Verwendung/Persistierung zurück.
        """

        # 0. Daten-Integrität prüfen (Führt die obere Validierung aus)
        self.prepare_model_data()

        # Extraktion der reinen numerischen Feature-Matrix
        X = self.df_transformed[self.features].copy()
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        # 1. Feature-Skalierung
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        # 2. Modellinitialisierung
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1 # Nutzt alle CPU-Kerne für schnelleres Training
        )

        # 3. Modell-Fitting
        model.fit(X_train_normalized)
    
        # 4. Vorhersagen treffen (Inferenz NUR auf den Testdaten)
        # Isolation Forest: 1 = Normal, -1 = Anomalie
        predictions = model.predict(X_test_normalized)
        # Ummappen auf Standard-Binärklassifikation: 0 = Normal, 1 = Anomalie
        predictions_mapped = [1 if x == -1 else 0 for x in predictions]
        anomaly_scores = model.decision_function(X_test_normalized)

        # Vorbereitung leerer Spalten im Haupt-DataFrame
        self.df_transformed["Is_Anomaly"] = pd.NA
         # Der Anomaly Score (Je negativer, desto anomaler ist der Datenpunkt)
        self.df_transformed["Anomaly_Score"] = pd.NA

        # Gezielte Zuweisung über den Index von X_test
        self.df_transformed.loc[X_test.index, "Is_Anomaly"] = predictions_mapped
        self.df_transformed.loc[X_test.index, "Anomaly_Score"] = anomaly_scores

        df_results = self.df_transformed
        return df_results, scaler, model
    
    def __repr__(self) -> str:
        return f"ModelPipeline(Features={self.features})"