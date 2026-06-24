import streamlit as st
import pandas as pd
import time
import os

from agents.data_governance  import run_governance_pipeline
from agents.pricing          import run_pricing_pipeline
from agents.risk_intelligence import run_risk_pipeline
from agents.stress_testing   import run_stress_pipeline   # ← Agent 5 added

st.set_page_config(
    page_title="CRIP Unified Orchestrator",
    layout="wide",
    page_icon="🤖"
)

st.title("CRIP: Comprehensive Risk Intelligence Platform")
st.markdown("Upload a raw insurance dataset to automatically orchestrate the AI agents.")

# ── Scenario selector (shown before run) ──────────────────────────────────────
SCENARIO_LABELS = {
    "s1": "S1 — Claims +20% (Moderate)",
    "s2": "S2 — Claims +40% (Severe)",
    "s3": "S3 — Market Risk +25%",
    "s4": "S4 — Combined Shock (Claims +40% & Market +25%)",
    "s5": "S5 — Catastrophic Event (Flood / Cyclone / Earthquake)",
}
with st.sidebar:
    st.header("⚡ Stress Testing Settings")
    selected_scenario = st.selectbox(
        "Stress scenario to run",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda x: SCENARIO_LABELS[x],
        index=3,  # default: s4
    )

uploaded_file = st.file_uploader("Upload Raw Dataset (.csv or .xlsx)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.success(f"Dataset '{uploaded_file.name}' loaded. Found {len(df_raw)} rows.")

    if st.button("🚀 Run All Agents", type="primary"):
        st.divider()

        # ── Agent 1: Data Governance ──────────────────────────────────────────
        st.subheader("🛡️ Agent 1: Data Governance Running...")
        pb1 = st.progress(0)
        with st.spinner("Profiling dataset, detecting anomalies, imputing missing values..."):
            time.sleep(1)
            pb1.progress(30)
            gov_results = run_governance_pipeline(df_raw)
            df_clean    = gov_results["df_clean"]
            pb1.progress(100)
            time.sleep(0.5)
        st.success(f"✅ Data Governance complete! Cleaned dataset has {len(df_clean)} rows.")

        # ── Agent 2: Pricing & Profitability ──────────────────────────────────
        st.subheader("📈 Agent 2: Pricing & Profitability Running...")
        pb2 = st.progress(0)
        with st.spinner("Calculating Loss Ratios, Expense Ratios, and Underwriting Profit..."):
            time.sleep(1)
            pb2.progress(50)
            pricing_results = run_pricing_pipeline(df_clean)
            df_pricing      = pricing_results["df_pricing"]
            kpis            = pricing_results["kpis"]
            pb2.progress(100)
            time.sleep(0.5)
        st.success("✅ Pricing Analysis complete!")

        # ── Agent 3: Risk Intelligence & ML ──────────────────────────────────
        st.subheader("🛡️ Agent 3: Risk Intelligence & ML Running...")
        pb3 = st.progress(0)
        with st.spinner("Calculating Risk Scores, Training XGBoost, running Monte Carlo..."):
            time.sleep(1)
            pb3.progress(50)
            risk_results     = run_risk_pipeline(df_pricing)
            df_risk          = risk_results["df_risk"]
            feature_importance = risk_results["feature_importance"]
            portfolio_metrics  = risk_results["portfolio_metrics"]

            os.makedirs("reports", exist_ok=True)
            df_risk.to_csv("reports/risk_intelligence_dataset.csv", index=False)

            summary_path = "reports/risk_summary_report.xlsx"
            with pd.ExcelWriter(summary_path) as writer:
                metrics_df = pd.DataFrame([portfolio_metrics]).T.reset_index()
                metrics_df.columns = ["Metric", "Value"]
                metrics_df.to_excel(writer, sheet_name="Portfolio Metrics", index=False)
                if "Product_Type" in df_risk.columns:
                    risk_cols = [c for c in ["Insurance_Risk","Market_Risk","Credit_Risk",
                                             "Operational_Risk","Hazard_Score"] if c in df_risk.columns]
                    risk_by_product = df_risk.groupby("Product_Type")[risk_cols].mean().reset_index()
                    risk_by_product.to_excel(writer, sheet_name="Risks by Product", index=False)
                feature_importance.to_excel(writer, sheet_name="ML Feature Importance", index=False)
                if "Model Governance & Validation Standards" in df_risk.columns:
                    gov_summary = df_risk["Model Governance & Validation Standards"].value_counts().reset_index()
                    gov_summary.columns = ["Status", "Count"]
                    gov_summary.to_excel(writer, sheet_name="Model Governance", index=False)

            pb3.progress(100)
            time.sleep(0.5)
        st.success("✅ Risk Intelligence & ML complete!")

        # ── Agent 5: Stress Testing ───────────────────────────────────────────
        st.subheader("⚡ Agent 5: Stress Testing Running...")
        pb5 = st.progress(0)
        with st.spinner(f"Running {SCENARIO_LABELS[selected_scenario]}..."):
            time.sleep(1)
            pb5.progress(40)
            stress_results = run_stress_pipeline(df_risk, scenario_id=selected_scenario)
            pb5.progress(100)
            time.sleep(0.5)

        sol = stress_results["solvency_ratio"]
        if stress_results["is_solvent"]:
            st.success(f"✅ Stress Testing complete! Solvency ratio: {sol:.1f}% — Capital adequate.")
        else:
            st.error(f"⚠️ Stress Testing complete! Solvency ratio: {sol:.1f}% — Capital shortfall detected!")

        st.divider()

        # ── Results tabs ──────────────────────────────────────────────────────
        st.header("📊 Final Agent Reports")
        tab1, tab2, tab3, tab5 = st.tabs([
            "🛡️ Data Governance",
            "📈 Pricing & Profitability",
            "🔮 Risk Intelligence & ML",
            "⚡ Stress Testing",
        ])

        # ── Tab 1: Data Governance ────────────────────────────────────────────
        with tab1:
            st.subheader("Data Cleaning Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Rows",    gov_results["summary"]["total_rows"])
            c2.metric("Cleaned Rows",     len(df_clean))
            c3.metric("Anomalies Flagged", len(gov_results["anomalies"]))
            st.write("Missing Values Found Before Cleaning:")
            if not gov_results["missing_df"].empty:
                st.dataframe(gov_results["missing_df"])
            else:
                st.info("No missing values.")
            st.write("Cleaned Dataset Preview:")
            st.dataframe(df_clean.head(50))

        # ── Tab 2: Pricing ────────────────────────────────────────────────────
        with tab2:
            st.subheader("Financial KPIs")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Total Premium",    f"₹{kpis['Total_Premium']:,.0f}")
            pc2.metric("Total Claims",     f"₹{kpis['Total_Claims']:,.0f}")
            pc3.metric("Total Expenses",   f"₹{kpis['Total_Expenses']:,.0f}")
            profit_color = "normal" if kpis['Underwriting_Profit'] > 0 else "inverse"
            pc4.metric("Underwriting Profit", f"₹{kpis['Underwriting_Profit']:,.0f}",
                       delta=f"₹{kpis['Underwriting_Profit']:,.0f}", delta_color=profit_color)

            st.write("Profitability Classification Distribution:")
            tier_summary = df_pricing.groupby("Profitability_Tier").size().reset_index(name="Count")
            st.dataframe(tier_summary, use_container_width=True)

            if "Product_Type" in df_pricing.columns:
                st.subheader("📊 Loss Ratio by Product")
                st.bar_chart(df_pricing.groupby("Product_Type")["Loss_Ratio"].mean().sort_values(ascending=False))
                st.subheader("💰 Profit by Product")
                st.bar_chart(df_pricing.groupby("Product_Type")["Underwriting_Profit"].sum().sort_values(ascending=False))

            st.subheader("🤖 AI Pricing Insights")
            ratio_table = df_pricing.groupby("Product_Type")["Combined_Ratio"].mean().reset_index()
            for _, row in ratio_table.iterrows():
                product, ratio = row["Product_Type"], row["Combined_Ratio"]
                if ratio > 1:
                    st.error(f"🔴 **{product}**: Combined Ratio {ratio:.2f} → Underpriced. Immediate rate action required.")
                elif ratio < 0.80:
                    st.success(f"🟢 **{product}**: Combined Ratio {ratio:.2f} → Highly Profitable.")
                else:
                    st.warning(f"🟡 **{product}**: Combined Ratio {ratio:.2f} → Monitor Pricing.")

            st.subheader("Detailed Pricing Dataset Preview")
            st.dataframe(df_pricing.head(50))

        # ── Tab 3: Risk Intelligence ──────────────────────────────────────────
        with tab3:
            st.subheader("⚠️ Composite Risk Scores (Actuarial)")
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            rc1.metric("Avg Insurance Risk",   f"{df_risk['Insurance_Risk'].mean():.1f}/10")
            rc2.metric("Avg Market Risk",      f"{df_risk['Market_Risk'].mean():.1f}/10")
            rc3.metric("Avg Credit Risk",      f"{df_risk['Credit_Risk'].mean():.1f}/10")
            rc4.metric("Avg Operational Risk", f"{df_risk['Operational_Risk'].mean():.1f}/10")
            rc5.metric("Avg CAT Risk",         f"{df_risk['Hazard_Score'].mean():.1f}/10")

            st.subheader("🔮 XGBoost Predictive Pricing")
            st.write("Feature Importance for predicting Claim Amount:")
            st.bar_chart(feature_importance.set_index('Feature')['Importance'])

            st.subheader("🏛️ Capital Adequacy & Monte Carlo (VaR)")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("VaR (99%)",          f"₹{portfolio_metrics['VaR_99']:,.0f}")
            mc2.metric("Expected Shortfall",  f"₹{portfolio_metrics['Expected_Shortfall']:,.0f}")
            mc3.metric("Solvency Ratio",      f"{portfolio_metrics['Solvency_Ratio']:.1f}%")
            mc4.metric("Model AUC",           f"{portfolio_metrics['AUC']:.2f}")

            st.write("Model Governance & Validation Table:")
            gov_cols     = ['Policy_ID', 'Expected_Claim_Amount_ML', 'AUC', 'KS Statistic',
                            'Brier Score', 'PSI', 'Model Governance & Validation Standards']
            display_cols = [c for c in gov_cols if c in df_risk.columns]
            st.dataframe(df_risk[display_cols].head(50))

        # ── Tab 5: Stress Testing ─────────────────────────────────────────────
        with tab5:
            sr  = stress_results["scenario_result"]
            skp = stress_results["kpis"]

            st.subheader(f"⚡ {stress_results['scenario_label']}")
            st.caption("Results derived from the cleaned & enriched portfolio produced by Agents 1–3.")

            # Scenario metrics
            st.markdown("#### Stressed Financial Metrics")
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Stressed Claims",
                       f"₹{sr['stressed_claim']:,.0f}",
                       delta=f"+{sr['stressed_claim'] - stress_results['portfolio']['claim_amount']:,.0f}",
                       delta_color="inverse")
            sm2.metric("Stressed Loss Ratio",
                       f"{sr['stressed_lr']:.1f}%",
                       delta=f"+{sr['stressed_lr'] - skp['Base_Loss_Ratio']:.1f}pp",
                       delta_color="inverse")
            sm3.metric("Stressed Combined Ratio",
                       f"{sr['stressed_cr']:.1f}%",
                       delta=f"+{sr['stressed_cr'] - skp['Base_Combined_Ratio']:.1f}pp",
                       delta_color="inverse")
            sm4.metric("Underwriting P&L",
                       f"₹{sr['stressed_uw']:,.0f}",
                       delta="Loss" if sr['stressed_uw'] < 0 else "Profit",
                       delta_color="inverse" if sr['stressed_uw'] < 0 else "normal")

            st.markdown("#### Capital Impact")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Capital Consumed",  f"₹{sr['capital_consumed']:,.0f}")
            cm2.metric("Remaining Capital", f"₹{sr['remaining_capital']:,.0f}")
            cm3.metric("Solvency Ratio",    f"{sr['solvency_ratio']:.1f}%")
            cm4.metric("Capital Shortfall",
                       f"₹{sr['shortfall']:,.0f}" if sr['shortfall'] > 0 else "None")

            # Alert
            if not stress_results["is_solvent"]:
                st.error(f"⚠️ Capital shortfall of ₹{sr['shortfall']:,.0f} detected under "
                         f"**{stress_results['scenario_label']}**. "
                         f"Solvency ratio drops to {sr['solvency_ratio']:.1f}%. "
                         "Immediate capital review required.")
            else:
                st.success(f"✅ Capital adequate under **{stress_results['scenario_label']}**. "
                           f"Solvency ratio: {sr['solvency_ratio']:.1f}%. "
                           f"Buffer: ₹{sr['remaining_capital']:,.0f}.")

            # All-scenario comparison
            st.markdown("#### All Scenarios Comparison")
            all_sc = stress_results["all_scenarios"].copy()

            def color_cr(val):
                if val > 110: return "background-color:#FEF2F2;color:#991B1B;font-weight:600"
                if val > 100: return "background-color:#FFFBEB;color:#92400E;font-weight:600"
                return "background-color:#F0FDF4;color:#166534;font-weight:600"

            def color_sol(val):
                if val < 50: return "background-color:#FEF2F2;color:#991B1B;font-weight:600"
                if val < 75: return "background-color:#FFFBEB;color:#92400E;font-weight:600"
                return "background-color:#F0FDF4;color:#166534;font-weight:600"

            styled = (all_sc.style
                      .applymap(color_cr,  subset=["Stressed CR (%)"])
                      .applymap(color_sol, subset=["Solvency Ratio (%)"])
                      .format({
                          "Base CR (%)":        "{:.1f}%",
                          "Stressed CR (%)":    "{:.1f}%",
                          "Stressed LR (%)":    "{:.1f}%",
                          "Capital Consumed":   "₹{:,.0f}",
                          "Solvency Ratio (%)": "{:.1f}%",
                          "Shortfall":          "₹{:,.0f}",
                      }))
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Product vulnerability
            st.markdown("#### Product Vulnerability Analysis")

            def color_vuln(val):
                if val == "High":   return "background-color:#FEF2F2;color:#991B1B;font-weight:600"
                if val == "Medium": return "background-color:#FFFBEB;color:#92400E;font-weight:600"
                return "background-color:#F0FDF4;color:#166534;font-weight:600"

            vuln_df = stress_results["product_vuln"]
            st.dataframe(
                vuln_df.style
                .applymap(color_vuln, subset=["Vulnerability"])
                .format({
                    "Claim Exposure":      "₹{:,.0f}",
                    "Capital At Risk":     "₹{:,.0f}",
                    "Vulnerability Score": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Charts
            st.markdown("#### Scenario Comparison — Combined Ratio")
            chart_data = all_sc.set_index("Scenario")[["Base CR (%)", "Stressed CR (%)"]].copy()
            st.bar_chart(chart_data)

            st.markdown("#### Solvency Ratio by Scenario")
            st.bar_chart(all_sc.set_index("Scenario")[["Solvency Ratio (%)"]])
