import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Insurance Pricing & Profitability Agent",
    layout="wide"
)

st.title("Insurance Pricing & Profitability Agent")

st.markdown("""
Upload a CSV insurance dataset to analyze:
- Loss Ratio
- Expense Ratio
- Combined Ratio
- Underwriting Profit
- Product Profitability
- Customer Segment Performance
- Distribution Channel Performance
- State-wise Profitability
- Monthly Profit Trends
- AI Pricing Insights
""")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset Loaded Successfully")

        # Convert Date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Numeric conversion
        numeric_cols = [
            "Written_Premium",
            "Claim_Amount",
            "Total_Expense"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Written_Premium"])

        premium = df["Written_Premium"].replace(0, np.nan)

        # Core Metrics
        df["Loss_Ratio"] = df["Claim_Amount"] / premium
        df["Expense_Ratio"] = df["Total_Expense"] / premium
        df["Combined_Ratio"] = (
            df["Loss_Ratio"] +
            df["Expense_Ratio"]
        )

        df["Underwriting_Profit"] = (
            df["Written_Premium"]
            - df["Claim_Amount"]
            - df["Total_Expense"]
        )

        # Profitability Classification
        def classify_ratio(x):

            if pd.isna(x):
                return "Unknown"

            if x < 0.80:
                return "Excellent"

            elif x < 0.95:
                return "Good"

            elif x <= 1.00:
                return "Marginal"

            else:
                return "Loss-Making"

        df["Profitability_Tier"] = (
            df["Combined_Ratio"]
            .apply(classify_ratio)
        )

        # KPI Section
        st.subheader("Portfolio Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Premium",
            f"₹{df['Written_Premium'].sum():,.0f}"
        )

        c2.metric(
            "Claims",
            f"₹{df['Claim_Amount'].sum():,.0f}"
        )

        c3.metric(
            "Expenses",
            f"₹{df['Total_Expense'].sum():,.0f}"
        )

        c4.metric(
            "Profit",
            f"₹{df['Underwriting_Profit'].sum():,.0f}"
        )

        st.divider()

        # Loss Ratio by Product
        if "Product_Type" in df.columns:

            st.subheader("Loss Ratio by Product")

            loss_ratio_product = (
                df.groupby("Product_Type")["Loss_Ratio"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(loss_ratio_product)

            # Product Profit
            st.subheader("Profit by Product")

            product_profit = (
                df.groupby("Product_Type")
                ["Underwriting_Profit"]
                .sum()
                .sort_values(ascending=False)
            )

            st.bar_chart(product_profit)

        # Profitability Tier Summary
        st.subheader("Profitability Tiers")

        tier_summary = (
            df.groupby("Profitability_Tier")
            .size()
            .reset_index(name="Count")
        )

        st.dataframe(
            tier_summary,
            use_container_width=True
        )

        # Customer Segment
        if "Customer_Segment" in df.columns:

            st.subheader("Customer Segment Profitability")

            segment_profit = (
                df.groupby("Customer_Segment")
                ["Underwriting_Profit"]
                .sum()
                .reset_index()
            )

            st.dataframe(
                segment_profit,
                use_container_width=True
            )

        # Distribution Channel
        if "Distribution_Channel" in df.columns:

            st.subheader("Distribution Channel Profitability")

            channel_profit = (
                df.groupby("Distribution_Channel")
                ["Underwriting_Profit"]
                .sum()
                .reset_index()
            )

            st.dataframe(
                channel_profit,
                use_container_width=True
            )

        # State-wise Profitability
        if "State" in df.columns:

            st.subheader("State-wise Profitability")

            state_profit = (
                df.groupby("State")
                ["Underwriting_Profit"]
                .sum()
                .sort_values(ascending=False)
            )

            st.bar_chart(state_profit)

        # Monthly Trend
        if "Date" in df.columns:

            st.subheader("Monthly Profit Trend")

            monthly_profit = (
                df.groupby(
                    pd.Grouper(
                        key="Date",
                        freq="ME"
                    )
                )["Underwriting_Profit"]
                .sum()
                .tail(36)
            )

            st.line_chart(monthly_profit)

        # AI Pricing Insights
        if "Product_Type" in df.columns:

            st.subheader("AI Pricing Insights")

            ratio_table = (
                df.groupby("Product_Type")
                ["Combined_Ratio"]
                .mean()
                .reset_index()
            )

            for _, row in ratio_table.iterrows():

                product = row["Product_Type"]
                ratio = row["Combined_Ratio"]

                if ratio > 1:
                    st.error(
                        f"{product}: Combined Ratio {ratio:.2f} → Underpriced / Loss-Making"
                    )

                elif ratio < 0.80:
                    st.success(
                        f"{product}: Combined Ratio {ratio:.2f} → Highly Profitable"
                    )

                else:
                    st.warning(
                        f"{product}: Combined Ratio {ratio:.2f} → Monitor Pricing"
                    )

        # Detailed Data
        st.subheader("Detailed Dataset")

        st.dataframe(
            df,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
