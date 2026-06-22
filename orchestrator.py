import streamlit as st
import pandas as pd
import time
import os
from agents.data_governance import run_governance_pipeline
from agents.pricing import run_pricing_pipeline
from agents.risk_intelligence import run_risk_pipeline

st.set_page_config(
    page_title="CRIP Unified Orchestrator",
    layout="wide",
    page_icon="🤖"
)

st.title("CRIP: Comprehensive Risk Intelligence Platform")
st.markdown("Upload a raw insurance dataset to automatically orchestrate the AI agents.")

uploaded_file = st.file_uploader("Upload Raw Dataset (.csv or .xlsx)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.success(f"Dataset '{uploaded_file.name}' Loaded Successfully. Found {len(df_raw)} rows.")

    if st.button("🚀 Run All Agents", type="primary"):
        st.divider()
        
        # --- AGENT 1: Data Governance ---
        st.subheader("🛡️ Agent 1: Data Governance Running...")
        progress_bar = st.progress(0)
        
        with st.spinner("Profiling dataset, detecting anomalies, and imputing missing values..."):
            time.sleep(1) # Visual effect
            progress_bar.progress(30)
            
            gov_results = run_governance_pipeline(df_raw)
            df_clean = gov_results["df_clean"]
            
            progress_bar.progress(100)
            time.sleep(0.5)
            
        st.success(f"✅ Data Governance complete! Cleaned dataset has {len(df_clean)} rows.")
        
        # --- AGENT 2: Pricing & Profitability ---
        st.subheader("📈 Agent 2: Pricing & Profitability Running...")
        progress_bar2 = st.progress(0)
        
        with st.spinner("Calculating Loss Ratios, Expense Ratios, and Underwriting Profit..."):
            time.sleep(1)
            progress_bar2.progress(50)
            
            pricing_results = run_pricing_pipeline(df_clean)
            df_pricing = pricing_results["df_pricing"]
            kpis = pricing_results["kpis"]
            
            progress_bar2.progress(100)
            time.sleep(0.5)
            
        st.success("✅ Pricing Analysis complete!")
        
        # --- AGENT 3: Risk Intelligence & ML ---
        st.subheader("🛡️ Agent 3: Risk Intelligence & ML Running...")
        progress_bar3 = st.progress(0)
        
        with st.spinner("Calculating Risk Scores, Training XGBoost, and running Monte Carlo simulations..."):
            time.sleep(1)
            progress_bar3.progress(50)
            
            risk_results = run_risk_pipeline(df_pricing)
            df_risk = risk_results["df_risk"]
            feature_importance = risk_results["feature_importance"]
            portfolio_metrics = risk_results["portfolio_metrics"]
            
            # Save the final dataset with all risk & ML columns to the reports folder
            os.makedirs("reports", exist_ok=True)
            df_risk.to_csv("reports/risk_intelligence_dataset.csv", index=False)
            
            # Generate a Risk Summary Excel Report
            summary_path = "reports/risk_summary_report.xlsx"
            with pd.ExcelWriter(summary_path) as writer:
                # 1. Portfolio Metrics
                metrics_df = pd.DataFrame([portfolio_metrics]).T.reset_index()
                metrics_df.columns = ["Metric", "Value"]
                metrics_df.to_excel(writer, sheet_name="Portfolio Metrics", index=False)
                
                # 2. Average Risks by Product
                if "Product_Type" in df_risk.columns:
                    risk_by_product = df_risk.groupby("Product_Type")[["Insurance_Risk", "Market_Risk", "Credit_Risk", "Operational_Risk", "Hazard_Score"]].mean().reset_index()
                    risk_by_product.to_excel(writer, sheet_name="Risks by Product", index=False)
                    
                # 3. XGBoost Feature Importance
                feature_importance.to_excel(writer, sheet_name="ML Feature Importance", index=False)
                
                # 4. Model Governance
                gov_summary = df_risk["Model Governance & Validation Standards"].value_counts().reset_index()
                gov_summary.columns = ["Status", "Count"]
                gov_summary.to_excel(writer, sheet_name="Model Governance", index=False)
            
            progress_bar3.progress(100)
            time.sleep(0.5)
            
        st.success("✅ Risk Intelligence & ML complete!")
        
        st.divider()
        
        # --- RESULTS TABS ---
        st.header("📊 Final Agent Reports")
        tab1, tab2, tab3 = st.tabs(["🛡️ Data Governance Report", "📈 Pricing & Profitability Report", "🔮 Risk Intelligence & ML"])
        
        with tab1:
            st.subheader("Data Cleaning Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Rows", gov_results["summary"]["total_rows"])
            col2.metric("Cleaned Rows", len(df_clean))
            col3.metric("Anomalies Flagged", len(gov_results["anomalies"]))
            
            st.write("Missing Values Found Before Cleaning:")
            if not gov_results["missing_df"].empty:
                st.dataframe(gov_results["missing_df"])
            else:
                st.info("No missing values.")
                
            st.write("Cleaned Dataset Preview:")
            st.dataframe(df_clean.head(50))
            
        with tab2:
            st.subheader("Financial KPIs")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Total Premium", f"₹{kpis['Total_Premium']:,.0f}")
            pc2.metric("Total Claims", f"₹{kpis['Total_Claims']:,.0f}")
            pc3.metric("Total Expenses", f"₹{kpis['Total_Expenses']:,.0f}")
            
            profit_color = "normal" if kpis['Underwriting_Profit'] > 0 else "inverse"
            pc4.metric("Underwriting Profit", f"₹{kpis['Underwriting_Profit']:,.0f}", delta=f"₹{kpis['Underwriting_Profit']:,.0f}", delta_color=profit_color)
            
            st.write("Profitability Classification Distribution:")
            tier_summary = df_pricing.groupby("Profitability_Tier").size().reset_index(name="Count")
            st.dataframe(tier_summary, use_container_width=True)
            
            if "Product_Type" in df_pricing.columns:
                st.subheader("📊 Loss Ratio by Product")
                loss_ratio_product = df_pricing.groupby("Product_Type")["Loss_Ratio"].mean().sort_values(ascending=False)
                st.bar_chart(loss_ratio_product)
                
                st.subheader("💰 Profit by Product")
                profit_product = df_pricing.groupby("Product_Type")["Underwriting_Profit"].sum().sort_values(ascending=False)
                st.bar_chart(profit_product)

                st.subheader("🤖 AI Pricing Insights")
                ratio_table = df_pricing.groupby("Product_Type")["Combined_Ratio"].mean().reset_index()
                for _, row in ratio_table.iterrows():
                    product = row["Product_Type"]
                    ratio = row["Combined_Ratio"]
                    
                    if ratio > 1:
                        st.error(f"🔴 **{product}**: Combined Ratio {ratio:.2f} → Underpriced / Loss-Making. Immediate rate action required.")
                    elif ratio < 0.80:
                        st.success(f"🟢 **{product}**: Combined Ratio {ratio:.2f} → Highly Profitable. Indicates pricing power.")
                    else:
                        st.warning(f"🟡 **{product}**: Combined Ratio {ratio:.2f} → Monitor Pricing. Margins are tight.")

            st.subheader("Detailed Pricing Dataset Preview")
            st.dataframe(df_pricing.head(50))
            
        with tab3:
            st.subheader("⚠️ Composite Risk Scores (Actuarial)")
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            rc1.metric("Avg Insurance Risk", f"{df_risk['Insurance_Risk'].mean():.1f}/10")
            rc2.metric("Avg Market Risk", f"{df_risk['Market_Risk'].mean():.1f}/10")
            rc3.metric("Avg Credit Risk", f"{df_risk['Credit_Risk'].mean():.1f}/10")
            rc4.metric("Avg Operational Risk", f"{df_risk['Operational_Risk'].mean():.1f}/10")
            rc5.metric("Avg CAT Risk", f"{df_risk['Hazard_Score'].mean():.1f}/10")
            
            st.subheader("🔮 XGBoost Predictive Pricing")
            st.write("Feature Importance for predicting Claim Amount:")
            st.bar_chart(feature_importance.set_index('Feature')['Importance'])
            
            st.subheader("🏛️ Capital Adequacy & Monte Carlo (VaR)")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("VaR (99%)", f"₹{portfolio_metrics['VaR_99']:,.0f}")
            mc2.metric("Expected Shortfall", f"₹{portfolio_metrics['Expected_Shortfall']:,.0f}")
            mc3.metric("Solvency Ratio", f"{portfolio_metrics['Solvency_Ratio']:.1f}%")
            mc4.metric("Model AUC", f"{portfolio_metrics['AUC']:.2f}")
            
            st.write("Model Governance & Validation Table:")
            gov_cols = ['Policy_ID', 'Expected_Claim_Amount_ML', 'AUC', 'KS Statistic', 'Brier Score', 'PSI', 'Model Governance & Validation Standards']
            display_cols = [c for c in gov_cols if c in df_risk.columns]
            st.dataframe(df_risk[display_cols].head(50))
