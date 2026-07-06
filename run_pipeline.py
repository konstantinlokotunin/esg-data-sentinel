from src.extract import extract_data
from src.transform import transform_data


def main():

    df = extract_data(r"""C:\Users\konst\Documents\Python Projects\ESG Data Sentinel\esg-data-sentinel\data\industrial_releases_of_pollutants_to_air.csv""")
    df = transform_data(df)

    print("\nFirst rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()