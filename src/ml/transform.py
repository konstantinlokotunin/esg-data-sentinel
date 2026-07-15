"""
transform.py
Kapselt die Schadstoff-Klassifizierung und Feature-Engineering-Logik für das Isolation-Forest-Modell
in mathematischen Komponenten-Klassen.
"""

import pandas as pd
import numpy as np

class DataTransformer:
    """Komponente zur hochdimensionalen Feature-Generierung"""
    def __init__(self, df: pd.DataFrame):
        self.df_cleaned = df

        # Klassifiziert Schadstoffe in logische, aggregierbare Gruppen für die kontextuelle ML-Analyse."""
        self.risk_groups = {
            "Critical Risk - Immediate Toxicity & Severe Health Damage": [
                "Arsenic and compounds (as As)",
                "Asbestos",
                "Benzene",
                "Cadmium and compounds (as Cd)",
                "Chlordecone",
                "Chromium and compounds (as Cr)",
                "Endrin",
                "Ethylene oxide",
                "Hydrogen cyanide (HCN)",
                "Lead and compounds (as Pb)",
                "Mercury and compounds (as Hg)",
                "PCDD + PCDF (dioxins + furans) (as Teq)",
                "Vinyl chloride",
            ],
            "High Risk - Persistent Toxins, Carcinogens & Long-Term Illness": [
                "1,1,1-trichloroethane (TCE-1,1,1)",
                "1,1,2,2-tetrachloroethane (TETRACHLOROETHANE-1,1,2,2)",
                "1,2,3,4,5,6-hexachlorocyclohexane (HCH)",
                "1,2-dichloroethane (DCE-1,2)",
                "Aldrin",
                "Anthracene",
                "Benzo(g,h,i)perylene",
                "Brominated diphenylethers (PBDE)",
                "Di-(2-ethyl hexyl) phthalate (DEHP)",
                "Dichloromethane (DCM)",
                "Fluoranthene",
                "Halogenated organic compounds (as AOX)",
                "Hexachlorobenzene (HCB)",
                "Lindane",
                "Naphthalene",
                "Nickel and compounds (as Ni)",
                "Nonylphenol and Nonylphenol ethoxylates",
                "Pentachlorobenzene",
                "Pentachlorophenol (PCP)",
                "Polychlorinated biphenyls (PCBs)",
                "Polycyclic aromatic hydrocarbons (PAHs)",
                "Tetrachloroethylene",
                "Tetrachloromethane (TCM)",
                "Trichlorobenzenes (TCB)",
                "Trichloroethylene (TRI)",
                "Trichloromethane",
            ],
            "Moderate Risk - Air Pollution, Corrosives & Community Exposure": [
                "Ammonia (NH3)",
                "Carbon monoxide (CO)",
                "Chlorine and inorganic compounds (as HCl)",
                "Copper and compounds (as Cu)",
                "Fine particulate matter (PM2.5)",
                "Fluorides (as total F)",
                "Fluorine and inorganic compounds (as HF)",
                "Nitrogen oxides (NOX)",
                "Non-methane volatile organic compounds (NMVOC)",
                "Particulate matter (PM10)",
                "Phenols (as total C)",
                "Sulphur oxides (SOX)",
                "Zinc and compounds (as Zn)",
            ],
            "Regulated Climate/Ozone Risk - Global Environmental Impact": [
                "Carbon dioxide (CO2)",
                "Carbon dioxide (CO2) excluding biomass",
                "Chlorofluorocarbons (CFCs)",
                "Halons",
                "Hydro-fluorocarbons (HFCS)",
                "Hydrochlorofluorocarbons (HCFCs)",
                "Methane (CH4)",
                "Nitrous oxide (N2O)",
                "Perfluorocarbons (PFCs)",
                "Sulphur hexafluoride (SF6)",
            ],
            "Low / Context-dependent Risk - Indicators or Lower Direct Toxicity": [
                "Chlorides (as total Cl)",
                "Ethyl benzene",
                "Toluene",
                "Total nitrogen",
                "Total organic carbon(as total C or COD/3) (TOC)",
                "Xylenes",
            ],
        }

        # Dictionary flachklopfen für schnelles Mapping
        self.risk_lookup = {
            substance: risk_level
            for risk_level, substances in self.risk_groups.items()
            for substance in substances
        }
        
    def __repr__(self) -> str:
        return f"DataTransformer(ColumnsIn={len(self.df_cleaned.columns)})"

    def add_features(self) -> pd.DataFrame:
        """
        Berechnet statistische und kontextuelle Features für den Isolation Forest.
        Sichert Ausreißer und mathematische Unendlichkeiten (inf) ab.
        """

        # Vektorisierte Zuordnung mit Rückfallwert für ungeordnete Elemente.
        self.df_cleaned["Pollutant_Group"] = (
            self.df_cleaned["Pollutant"].map(self.risk_lookup).fillna("Unclassified")
        )

        # 1. Flag für unplausible/negative Werte setzen
        self.df_cleaned["Is_Negative"] = self.df_cleaned["Amount"] < 0

        # 2. Log-Transformation mathematisch absichern
        safe_amount = self.df_cleaned["Amount"].clip(lower=0)
        self.df_cleaned["Amount_Log"] = np.log1p(safe_amount)
    
        # 3. Gruppen-Mediane berechnen (Kontextuelle Abweichungen)
        self.df_cleaned["Group_Median"] = self.df_cleaned.groupby("Pollutant_Group")["Amount_Log"].transform("median")
        self.df_cleaned["Dev_from_Group_Median"] = (self.df_cleaned["Amount_Log"] - self.df_cleaned["Group_Median"])

        # 4. Sektoren-Mediane berechnen
        self.df_cleaned["Sector_Median"] = self.df_cleaned.groupby("Sector")["Amount_Log"].transform("median")
        self.df_cleaned["Dev_from_Sector_Median"] = (self.df_cleaned["Amount_Log"] - self.df_cleaned["Sector_Median"])

        # 5. Kombinierte Mediane berechnen
        self.df_cleaned["Sector_Group_Median"] = self.df_cleaned.groupby(
            ["Sector", "Pollutant_Group"])["Amount_Log"].transform("median")
        self.df_cleaned["Dev_from_Sector_Group_Median"] = (self.df_cleaned["Amount_Log"] - self.df_cleaned["Sector_Group_Median"])

        # 6. Year-over-Year (YoY) Änderungen zeitlich korrekt berechnen
        self.df_cleaned = self.df_cleaned.sort_values(["Facility", "Pollutant", "Year"])

        self.df_cleaned["Previous_Year_Amount"] = (
            self.df_cleaned.groupby(["Facility", "Pollutant"])["Amount"].shift(1)
        )

        self.df_cleaned["YoY_Change_Pct"] = (
            (self.df_cleaned["Amount"] - self.df_cleaned["Previous_Year_Amount"])
            / self.df_cleaned["Previous_Year_Amount"]
        ) * 100

        # Mathematischer Guardrail: Bereinigung aller unendlichen Werte (inf) und NaNs, die das ML-Modell zum Absturz bringen
        # Wenn Previous_Year_Amount <= 0 oder NaN ist, setzen wir die prozentuale Änderung auf 0.0
        self.df_cleaned["YoY_Change_Pct"] = (
            (self.df_cleaned["Amount"] - self.df_cleaned["Previous_Year_Amount"])
            / self.df_cleaned["Previous_Year_Amount"]
        ) * 100
        self.df_cleaned["YoY_Change_Pct"] = self.df_cleaned["YoY_Change_Pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df_transformed = self.df_cleaned
        return df_transformed