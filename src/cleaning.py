import pandas as pd
from pathlib import Path

file_path = r"C:\Users\konst\Documents\Python Projects\ESG Data Sentinel\esg-data-sentinel\outputs"

OUTPUT_DIR = Path(file_path)
output_path = OUTPUT_DIR / "anomaly_reports.xlsx"

def missing_report(df: pd.DataFrame):

    missing_report = pd.DataFrame({
        "missing_values": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100
    }, index=[0]).sort_values("missing_values", ascending=False)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        missing_report.to_excel(writer, sheet_name="Missing_Report", index=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select all relevant columns, format datatypes, and filter by target region."""
    columns_to_keep = {
        "countryName": "Country",
        "reportingYear": "Year",
        "EPRTR_SectorCode": "Code",
        "EPRTR_SectorName": "Sector",
        "facilityName": "Facility",
        "Pollutant": "Pollutant",
        "Releases": "Amount",
    }

    # 1. Filter out columns to keep only the keys from the dictionary
    df = df[list(columns_to_keep.keys())].copy()

    # 2. Rename columns using the defined dictionary mappings
    df = df.rename(columns=columns_to_keep)

    # 3. Handle numeric values safely
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # 4. Convert remaining columns safely to string type
    cat_cols = ["Country", "Year", "Code", "Sector", "Facility", "Pollutant"]
    df[cat_cols] = df[cat_cols].astype(str).apply(lambda x: x.str.strip().replace(
        {"nan": "Unknown", "": "Unknown"}))

    # 5. Clean missing records and restrict spatial scope to Austria
    df = df.dropna(subset=["Amount"])
    df = df[df["Country"] == "Austria"].copy()

    print("Data cleaned successfully.")
    print("Shape:", df.shape)
    return df