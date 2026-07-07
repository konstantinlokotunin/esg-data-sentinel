import pandas as pd

def filter_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")


    # 4. Convert remaining columns safely to string type
    string_cols = ["Country", "Code", "Sector", "Facility", "Pollutant"]
    df[string_cols] = (
        df[string_cols].astype(str).apply(lambda x: x.str.strip().fillna("Unknown"))
    )

    # 5. Clean missing records and restrict spatial scope to Austria
    df = df.dropna(subset=["Year", "Amount"])
    df = df[df["Country"] == "Austria"].copy()

    return df