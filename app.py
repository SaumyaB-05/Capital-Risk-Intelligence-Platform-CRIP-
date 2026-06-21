import streamlit as st 
from utils.helpers import load_data 
st.set_page_config(page_title="CRIP", layout="wide", page_icon="🏦") 
st.title("🏦 Capital & Risk Intelligence Platform") 
st.caption("AI-Powered Multi-Agent Insurance Risk Analytics") 
st.markdown("---")
# File upload section 
st.subheader("Step 1 — Upload your dataset") 
uploaded = st.file_uploader("Upload CRIP_Dataset.csv", type="csv") 
if uploaded: 
    df = load_data(uploaded) 
    st.session_state["raw_df"] = df 
    st.success(f"✅ Dataset loaded — {len(df):,} records found") 
    st.dataframe(df.head(5)) 
    st.info("👈 Now go to Data Governance Agent in the sidebar")