import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest

# --- Data Ingestion ---
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")
    return df

def get_data_summary(df):
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': df.columns.tolist()
    }

# --- Validation Pipeline ---
def check_missing_values(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def check_duplicates(df, subset=None):
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    return len(duplicates), duplicates

def clean_data(df):
    df_clean = df.copy()
    
    missing_strings = ['unknown', 'NA', 'null', 'Null', 'NULL', '?', '??', '???', ' ', '', 'N/A', 'n/a']
    df_clean.replace(missing_strings, np.nan, inplace=True)
    
    cols_to_drop = set(['Mystery_Risk_Factor', 'Legacy_System_Code'])
    for col in df_clean.columns:
        if df_clean[col].isnull().mean() > 0.80:
            cols_to_drop.add(col)
            continue
        if df_clean[col].nunique(dropna=True) <= 1:
            cols_to_drop.add(col)
            continue
        is_id_col = str(col).lower().endswith('id') or str(col).lower().endswith('_id')
        if not pd.api.types.is_numeric_dtype(df_clean[col]) and not is_id_col:
            if df_clean[col].count() > 0 and (df_clean[col].nunique(dropna=True) / df_clean[col].count()) > 0.90:
                cols_to_drop.add(col)
    
    df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns], inplace=True)
    
    df_clean.dropna(how='all', inplace=True)
    df_clean.dropna(axis=1, how='all', inplace=True)
    
    for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
        is_id_col = str(col).lower().endswith('id') or str(col).lower().endswith('_id')
        if not is_id_col and df_clean[col].count() > 0:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            df_clean[col] = df_clean[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
            
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
    return df_clean

# --- Anomaly Detection ---
def detect_anomalies_isolation_forest(df, numerical_cols):
    df_subset = df[numerical_cols].fillna(df[numerical_cols].median())
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(df_subset)
    anomaly_mask = predictions == -1
    return df[anomaly_mask]

# --- Fraud Scoring ---
def calculate_fraud_score(df):
    df_scored = df.copy()
    for col in ['Claim_Frequency', 'Claim_Amount', 'Policy_Tenure_Months', 'Underwriting_Exception_Flag', 'Claim_Exception_Flag']:
        if col in df_scored.columns:
            df_scored[col] = pd.to_numeric(df_scored[col], errors='coerce').fillna(0)
    df_scored['Calculated_Fraud_Score'] = 0
    if 'Claim_Frequency' in df.columns:
        df_scored.loc[df_scored['Claim_Frequency'] > df_scored['Claim_Frequency'].quantile(0.9), 'Calculated_Fraud_Score'] += 20
    if 'Underwriting_Exception_Flag' in df.columns:
        df_scored.loc[df_scored['Underwriting_Exception_Flag'] == 1, 'Calculated_Fraud_Score'] += 25
    if 'Claim_Exception_Flag' in df.columns:
        df_scored.loc[df_scored['Claim_Exception_Flag'] == 1, 'Calculated_Fraud_Score'] += 25
    if 'Claim_Amount' in df.columns and 'Policy_Tenure_Months' in df.columns:
        high_claim = df_scored['Claim_Amount'] > df_scored['Claim_Amount'].quantile(0.8)
        short_tenure = df_scored['Policy_Tenure_Months'] < 12
        df_scored.loc[high_claim & short_tenure, 'Calculated_Fraud_Score'] += 30
    df_scored['Calculated_Fraud_Score'] = df_scored['Calculated_Fraud_Score'].clip(upper=100)
    return df_scored

# --- Report Generator ---
def generate_excel_report(df_clean, anomalies, fraud_scores, output_dir="reports"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "validation_report.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_clean.to_excel(writer, sheet_name='Cleaned_Data', index=False)
        anomalies.to_excel(writer, sheet_name='Anomalies', index=False)
        if 'Policy_ID' in fraud_scores.columns:
            fraud_scores[['Policy_ID', 'Calculated_Fraud_Score']].to_excel(writer, sheet_name='Fraud_Scores', index=False)
        else:
            fraud_scores[['Calculated_Fraud_Score']].to_excel(writer, sheet_name='Fraud_Scores', index=False)
    return output_path

# --- Streamlit UI ---
st.set_page_config(page_title="Insurance Data Validation Agent", layout="wide")

st.title("🛡️ Insurance Data Validation Agent")

uploaded_file = st.file_uploader("Upload Insurance Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info("Loading Data...")
    df = load_data(temp_path)
    
    st.subheader("Data Summary")
    summary = get_data_summary(df)
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", summary['total_rows'])
    col2.metric("Total Columns", summary['total_columns'])
    
    st.write("### Raw Data Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Validation Pipeline")
    
    missing_df = check_missing_values(df)
    st.write("#### Missing Values")
    if not missing_df.empty:
        st.dataframe(missing_df)
    else:
        st.success("No missing values found!")
        
    num_dupes, duplicates = check_duplicates(df)
    st.write(f"#### Duplicates: {num_dupes} rows")
    if num_dupes > 0:
        st.dataframe(duplicates.head(5))
        
    if st.button("Run Full Validation & Cleaning"):
        with st.spinner("Cleaning Data & Running Validation..."):
            df_clean = clean_data(df)
            
            num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
            num_cols = [col for col in num_cols if not (str(col).lower().endswith('id') or str(col).lower().endswith('_id'))]
                
            anomalies = detect_anomalies_isolation_forest(df_clean, num_cols)
            df_scored = calculate_fraud_score(df_clean)
            
            st.success("Validation & Cleaning Complete!")
            
            st.write("### Anomalies Detected")
            st.write(f"Found {len(anomalies)} anomalies based on Isolation Forest.")
            st.dataframe(anomalies.head(10))
            
            st.write("### Fraud Risk Scores")
            st.write("Top 10 Highest Fraud Scores:")
            if 'Policy_ID' in df_scored.columns:
                st.dataframe(df_scored[['Policy_ID', 'Calculated_Fraud_Score']].sort_values(by='Calculated_Fraud_Score', ascending=False).head(10))
            else:
                st.dataframe(df_scored[['Calculated_Fraud_Score']].sort_values(by='Calculated_Fraud_Score', ascending=False).head(10))
            
            report_path = generate_excel_report(df_clean, anomalies, df_scored)
            
            col1, col2 = st.columns(2)
            with col1:
                with open(report_path, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Validation Report (Excel)",
                        data=file,
                        file_name="validation_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            with col2:
                csv_data = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Data (CSV)",
                    data=csv_data,
                    file_name="cleaned_insurance_data.csv",
                    mime="text/csv"
                )

    if os.path.exists(temp_path):
        os.remove(temp_path)

