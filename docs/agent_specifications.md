# Capital & Risk Intelligence Platform (CRIP)

# Agent Specifications

---

# Agent 1: Data Governance Agent

## Purpose

Validate uploaded dataset and ensure data quality before any analysis is performed.

## Inputs

* Policy_ID
* Date
* Product_Type
* Region
* Policy_Status
* Customer_Segment
* Distribution_Channel
* Written_Premium
* Claim_Amount
* Fraud_Flag
* Anomaly_Flag
* Insurance_Risk
* Market_Risk
* Credit_Risk
* Operational_Risk

## Calculations

* Missing Value Check
* Duplicate Policy Check
* Data Type Validation
* Category Validation
* Risk Score Range Validation
* Outlier Detection
* Data Quality Score

## Outputs

* Clean Dataset
* Data Quality Score
* Validation Report
* Anomaly Summary
* Fraud Summary

---

# Agent 2: Pricing & Profitability Agent

## Purpose

Evaluate profitability and premium adequacy across products and segments.

## Inputs

* Written_Premium
* Claim_Amount
* Total_Expense
* Product_Type
* Customer_Segment
* Distribution_Channel
* Region
* Policy_Status
* Claim_Count
* Sum_Insured

## Calculations

### Loss Ratio

Claim_Amount / Written_Premium

### Expense Ratio

Total_Expense / Written_Premium

### Combined Ratio

Loss Ratio + Expense Ratio

### Underwriting Profit

Written_Premium - Claim_Amount - Total_Expense

### Claim Severity

Claim_Amount / Claim_Count

### Premium Adequacy

Written_Premium / Sum_Insured

## Outputs

* Loss Ratio Analysis
* Combined Ratio Analysis
* Underwriting Profit Analysis
* Product Profitability Ranking
* Premium Adequacy Analysis
* Pricing Insights

---

# Agent 3: Risk Intelligence Agent

## Purpose

Assess portfolio risk and identify high-risk products, regions and segments.

## Inputs

* Insurance_Risk
* Market_Risk
* Credit_Risk
* Operational_Risk
* Claim_Count
* Claim_Amount
* Fraud_Flag
* Product_Type
* Policy_Status
* Region
* Customer_Segment
* Sum_Insured
* Renewal_Count

## Calculations

### Composite Risk Score

Weighted average of:

* Insurance Risk
* Market Risk
* Credit Risk
* Operational Risk

### Risk Category

* Low
* Medium
* High
* Critical

### Exposure at Risk

Sum_Insured × Composite Risk Score

### Risk Concentration

Region-wise and Product-wise Risk

### Fraud Rate

Fraud Policies / Total Policies

## Outputs

* Composite Risk Score
* Risk Categories
* Risk Heatmap
* Exposure Analysis
* Fraud Analysis
* Risk Ranking Table

---

# Agent 4: Time Series Intelligence Agent

## Purpose

Forecast future claims and premiums using historical trends.

## Inputs

* Date
* Written_Premium
* Claim_Amount
* Claim_Count
* Product_Type
* Season_Flag
* Anomaly_Flag

## Calculations

### Monthly Aggregation

Claims and Premiums by Month

### Claim Frequency

Claim_Count per Month

### Claim Severity

Claim_Amount per Claim

### Claims Forecast

Prophet Forecast

### Premium Forecast

Prophet Forecast

### Trend Analysis

* Growth Trends
* Seasonal Trends
* Year-over-Year Changes

## Outputs

* Claims Forecast Chart
* Premium Forecast Chart
* Trend Analysis
* Seasonal Analysis
* Forecast Summary

---

# Agent 5: Stress Testing Agent

## Purpose

Evaluate portfolio resilience under adverse scenarios.

## Inputs

* Claim_Amount
* Written_Premium
* Total_Expense
* Capital_Reserve
* Insurance_Risk
* Market_Risk
* Credit_Risk
* Operational_Risk
* Product_Type
* Region

## Scenarios

### Scenario 1

Claims +20%

### Scenario 2

Claims +40%

### Scenario 3

Market Risk +25%

### Scenario 4

Combined Shock

Claims +40% and Market Risk +25%

## Calculations

* Shocked Claims
* Post-Shock Profit
* Capital Impact
* Solvency Ratio
* Capital Shortfall
* Risk Movement

## Outputs

* Scenario Analysis Table
* Capital Impact Analysis
* Solvency Analysis
* Product Vulnerability Analysis
* Stress Testing Dashboard

---

# Agent 6: Executive Reporting Agent

## Purpose

Generate management-level insights and reports.

## Inputs

* Data Governance Outputs
* Pricing Agent Outputs
* Risk Agent Outputs
* Time Series Agent Outputs
* Stress Testing Outputs

## Calculations

* Portfolio Summary
* KPI Consolidation
* Renewal Rate
* Lapse Rate
* Portfolio Growth Rate
* Top Risk Product
* Top Loss Product

## AI Features

Gemini API generates:

* Executive Summary
* Risk Commentary
* Strategic Recommendations

## Outputs

* Executive Summary
* Management Recommendations
* KPI Dashboard
* Risk Report
* Downloadable PDF Report

---

# Workflow

Upload Dataset

↓

Data Governance Agent

↓

Pricing Agent

↓

Risk Agent

↓

Time Series Agent

↓

Stress Testing Agent

↓

Executive Reporting Agent

↓

Dashboard & PDF Report

