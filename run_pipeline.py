from src.extract import extract_data, InvalidFileFormat
from src.transform import transform_data


def main():

    try:
        df = extract_data(r"""C:\Users\konst\Documents\Python Projects\ESG Data Sentinel\esg-data-sentinel\data\raw\industrial_releases_of_pollutants_to_air.csv""")
    except InvalidFileFormat as e:
        print(e)

    print("\nFirst rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()