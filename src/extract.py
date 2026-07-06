from pathlib import Path
import pandas as pd

def extract_data(file_path):
    """
    Loads one CSV or Excel file into a pandas DataFrame.
    """

    file_path = Path(file_path)

    print("Loading file:", file_path.name)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, low_memory=False)

    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError("Only CSV and Excel files are supported.")

    print("Data loaded successfully.")
    print("Shape:", df.shape)

    return df