import pandas as pd
import numpy as np

def add_pollutant_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Klassifiziert Schadstoffe in logische, aggregierbare Gruppen
    für die kontextuelle ML-Analyse.
    """
    risk_groups = {
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

    # Turn the group dictionary into a flat lookup table
    risk_lookup = {
        substance: risk_level
        for risk_level, substances in risk_groups.items()
        for substance in substances
    }

    # Vectorized mapping with fallback value for unclassified items
    df["Pollutant_Group"] = (
        df["Pollutant"].map(risk_lookup).fillna("Unclassified")
    )

    return df

def add_features_for_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates statistical and contextual features specialized for
    Isolation Forest anomaly tracking.
    """
    
    # 1. Rule-based flag for impossible / suspicious values
    df["Is_Negative"] = df["Amount"] < 0

    # 2. Log transformation
    # Negative values cannot be log-transformed.
    # They are kept visible through Is_Negative.
    df.loc[df["Amount"] < 0, "Amount"] = df["Amount"].median()

    df["Amount_Log"] = np.log1p(df["Amount"])
    df["Amount_Log"] = round(df["Amount_Log"], 4)
    
    # 3. Deviation from pollutant group median
    df["Group_Median"] = df.groupby("Pollutant_Group")["Amount_Log"].transform("median")
    df["Dev_from_Group_Median"] = (df["Amount_Log"] - df["Group_Median"])
    df["Dev_from_Group_Median"] = round(df["Dev_from_Group_Median"], 4)

    # 4. Deviation from sector median
    df["Sector_Median"] = df.groupby("Sector")["Amount_Log"].transform("median")
    df["Dev_from_Sector_Median"] = (df["Amount_Log"] - df["Sector_Median"])
    df["Dev_from_Sector_Median"] = round(df["Dev_from_Sector_Median"], 4)

    # 5. Deviation from sector + pollutant group median
    df["Sector_Group_Median"] = df.groupby(
        ["Sector", "Pollutant_Group"])["Amount_Log"].transform("median")
    df["Dev_from_Sector_Group_Median"] = (df["Amount_Log"] - df["Sector_Group_Median"])
    df["Dev_from_Sector_Group_Median"] = round(df["Dev_from_Sector_Group_Median"], 4)

    # 6. Year-over-year change
    df = df.sort_values(["Facility", "Pollutant", "Year"])

    df["Previous_Year_Amount"] = (
        df.groupby(["Facility", "Pollutant"])["Amount_Log"]
        .shift(1)
    )

    df["YoY_Change_Pct"] = (
        (df["Amount_Log"] - df["Previous_Year_Amount"])
        / df["Previous_Year_Amount"]
    ) * 100
    df["YoY_Change_Pct"] = round(df["YoY_Change_Pct"], 4)

    return df

