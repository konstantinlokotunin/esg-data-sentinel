from src.extract import extract_data, InvalidFileFormat
from src.cleaning import clean_data, missing_report
from src.transform import add_pollutant_groups, add_features_for_isolation_forest
from src.train_model import prepare_model_data, train_isolation_forest
from src.evaluate_model import save_anomaly_reports
from src.visualization import (
    plot_anomaly_score_distribution,
    plot_anomalies_by_sector,
plot_anomalies_by_pollutant_group,
plot_anomalies_by_sector_and_pollutant_group
)


def main():

    try:
        df = extract_data(r"C:\Users\konst\Documents\Python Projects\ESG Data Sentinel\esg-data-sentinel\data\raw\industrial_releases_of_pollutants_to_air.csv")
    except InvalidFileFormat as e:
        print(e)

    print("\nFirst rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    df = clean_data(df)

    print("\nFirst rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    missing_report(df)

    df = add_pollutant_groups(df)
    df = add_features_for_isolation_forest(df)

    print("\nShape:", df.shape)
    print("\nFirst rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    X = prepare_model_data(df)
    df = train_isolation_forest(df, X)

    save_anomaly_reports(df)

    plot_anomaly_score_distribution(df)
    plot_anomalies_by_sector(df)
    plot_anomalies_by_pollutant_group(df)
    plot_anomalies_by_sector_and_pollutant_group(df)

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()