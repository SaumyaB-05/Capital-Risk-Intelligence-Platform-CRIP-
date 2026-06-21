# Insurance Data Validation Agent

A single-file Streamlit application designed to automatically ingest, validate, clean, and score datasets (with a specific focus on Insurance Fraud detection).

## Features

- **Intelligent Data Cleaning**: Automatically standardizes missing value strings (e.g., `'unknown'`, `'NA'`, `'?'`) to `NaN`. It dynamically drops completely empty rows/columns, columns with zero variance, and columns with excessive missing data (>80%).
- **Outlier Handling**: Applies Interquartile Range (IQR) capping (Winsorization) to numerical columns to neutralize extreme outliers without destroying entire data rows.
- **Missing Value Imputation**: Automatically fills missing numerical values with the median and categorical values with the mode.
- **Anomaly Detection**: Uses `IsolationForest` machine learning to identify and flag statistically anomalous records across all numerical features.
- **Fraud Risk Scoring**: Calculates a heuristic fraud risk score (0-100) based on claim frequency, exception flags, and claim amounts versus policy tenure.
- **Exporting**: Download the fully cleaned dataset as a CSV, or download a comprehensive multi-sheet Excel report containing the clean data, anomalies, and fraud scores.

## Installation

This project is entirely contained within `app.py`. To run it, you will need Python installed along with the following libraries:

```bash
pip install streamlit pandas openpyxl scikit-learn
```

## How to Run

1. Open your terminal or command prompt.
2. Navigate to the folder containing `app.py`.
3. Run the following command:

```bash
python -m streamlit run app.py
```

4. The interactive dashboard will automatically open in your default web browser. Upload your CSV or Excel dataset to get started!
