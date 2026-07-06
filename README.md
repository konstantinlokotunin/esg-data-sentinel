# ESG Data Sentinel Austria

A machine-learning project for detecting potential data quality risks in Austrian industrial emissions data.

This project is built as a simple modular data pipeline:

Data Ingestion → Transformation → Synthetic Error Injection → Model Training → Evaluation

---

## Project Idea

The goal of this project is to build a neural network classification model that identifies potential data quality risks in ESG-related emissions data.

The model will classify data rows as:

- `0 = plausible`
- `1 = data quality risk / red flag`

---

## Data Source

The project uses industrial emissions data from the European Industrial Emissions Portal.

The project scope is limited to:

- Country: Austria
- Data type: industrial emissions data
- Data format: tabular data
- ML task: binary classification

---

## Methodological Note

Public ESG and emissions datasets usually do not contain reliable labels for incorrect or faulty data points.

Therefore, this project uses real emissions data as a basis and later creates synthetic data quality issues, such as:

- missing values
- negative values
- scaling errors
- extreme year-over-year changes
- sector-specific outliers

The model does not detect legally confirmed errors. It detects data rows that should be reviewed.

---

## Project Structure

```text
esg-data-sentinel-austria/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── src/
│   ├── extract.py
│   ├── transform.py
│   ├── inject_errors.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── visualization.py
│
├── app.py
├── run_pipeline.py
├── requirements.txt
└── README.md