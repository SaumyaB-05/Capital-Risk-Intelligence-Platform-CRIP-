"""
CRIP — Agent 5: Stress Testing Agent
agents/stress_testing.py

Drop this file into the agents/ folder.
The orchestrator imports it as:
    from agents.stress_testing import run_stress_pipeline

Input:  df_risk (DataFrame) — output of run_risk_pipeline()
Output: dict matching the pattern of run_governance_pipeline(),
        run_pricing_pipeline(), run_risk_pipeline()
"""

import pandas as pd
import numpy as np
import config

# ── Scenario definitions ───────────────────────────────────────────────────────
SCENARIO_CONFIG = config.STRESS_TESTING['SCENARIO_CONFIG']

PRODUCT_WEIGHTS = config.STRESS_TESTING['PRODUCT_WEIGHTS']

# ── Column name mapping ────────────────────────────────────────────────────────
# The existing repo uses "Hazard_Score" for catastrophe risk.
COL_MAP = {
    "claim_amount":       ["Claim_Amount",    "claim_amount"],
    "written_premium":    ["Written_Premium", "written_premium", "Premium"],
    "total_expense":      ["Total_Expense",   "total_expense",   "Expense"],
    "capital_reserve":    ["Capital_Reserve", "capital_reserve"],
    "insurance_risk":     ["Insurance_Risk",  "insurance_risk"],
    "market_risk":        ["Market_Risk",     "market_risk"],
    "credit_risk":        ["Credit_Risk",     "credit_risk"],
    "operational_risk":   ["Operational_Risk","operational_risk"],
    "catastrophe_risk":   ["Hazard_Score",    "Catastrophe_Risk", "catastrophe_risk", "CAT_Risk"],
    "exposure_at_risk":   ["Exposure_At_Risk","exposure_at_risk", "Sum_Insured"],
    "var_99":             ["VaR (99%)",       "var_99",           "VaR_99"],
    "expected_shortfall": ["Expected Shortfall","expected_shortfall","Expected_Shortfall"],
    "fraud_flag":         ["Fraud_Flag",      "fraud_flag"],
    "reinsurer_rating":   ["Reinsurer_Rating","reinsurer_rating"],
    "psi":                ["PSI", "psi"],
    "auc":                ["AUC", "auc"],
    "ks_stat":            ["KS Statistic", "ks_stat"],
    "brier_score":        ["Brier Score", "brier_score", "Brier_Score"]
}


def _get_col(df, key, default=0.0):
    for col in COL_MAP[key]:
        if col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                return float(series.mean())
            elif key == "reinsurer_rating":
                rating_map = {'AAA': 1.0, 'AA': 0.9, 'A': 0.8, 'BBB': 0.6, 'BB': 0.4, 'B': 0.2}
                return float(series.map(rating_map).fillna(1.0).mean())
    return default


def _extract_portfolio(df):
    def safe_sum(candidates):
        for col in candidates:
            if col in df.columns:
                return float(df[col].sum())
        return 0.0

    claim_amount    = safe_sum(COL_MAP["claim_amount"])
    written_premium = safe_sum(COL_MAP["written_premium"])
    total_expense   = safe_sum(COL_MAP["total_expense"])
    capital_reserve = _get_col(df, "capital_reserve", default=50_000_000)

    def risk_score(key):
        raw = _get_col(df, key, default=5.0)
        return min(100, raw * 10) if raw <= 10 else min(100, raw)

    return {
        "claim_amount":       claim_amount,
        "written_premium":    written_premium,
        "total_expense":      total_expense,
        "capital_reserve":    capital_reserve,
        "exposure_at_risk":   safe_sum(COL_MAP["exposure_at_risk"]),
        "var_99":             _get_col(df, "var_99"),
        "expected_shortfall": _get_col(df, "expected_shortfall"),
        "fraud_flag":         _get_col(df, "fraud_flag"),
        "reinsurer_rating":   _get_col(df, "reinsurer_rating", default=1.0),
        "psi":                _get_col(df, "psi"),
        "auc":                _get_col(df, "auc", default=0.85),
        "ks_stat":            _get_col(df, "ks_stat"),
        "brier_score":        _get_col(df, "brier_score"),
        "insurance_risk":     risk_score("insurance_risk"),
        "market_risk":        risk_score("market_risk"),
        "credit_risk":        risk_score("credit_risk"),
        "operational_risk":   risk_score("operational_risk"),
        "catastrophe_risk":   risk_score("catastrophe_risk"),
    }


def _run_scenario(portfolio, scenario_id):
    cfg = SCENARIO_CONFIG[scenario_id]
    p   = portfolio["written_premium"]
    e   = portfolio["total_expense"]
    c   = portfolio["capital_reserve"]
    exp = portfolio.get("exposure_at_risk", p * 10) if portfolio.get("exposure_at_risk", 0) > 0 else p * 10

    stressed_claim = portfolio["claim_amount"] * cfg["claim_mult"]
    stressed_mkt   = min(100, portfolio["market_risk"]      * cfg["mkt_mult"])
    stressed_cat   = min(100, portfolio["catastrophe_risk"] * cfg["cat_mult"])

    base_er     = (e / p * 100) if p > 0 else 0
    stressed_lr = (stressed_claim / p * 100) if p > 0 else 0
    stressed_cr = stressed_lr + base_er
    stressed_uw = p - stressed_claim - e

    claim_impact   = max(0.0, -stressed_uw)
    market_impact  = (stressed_mkt / 100) * c * 0.40
    
    # Catastrophe impact based on Exposure At Risk rather than Capital Reserve
    cat_impact     = (stressed_cat / 100) * exp * 0.05 * cfg["cat_mult"]
    
    # Incorporate Reinsurer Rating into Credit Shock
    reinsurer_dampener = 1.0 + (1.0 - portfolio.get("reinsurer_rating", 1.0)) * 0.5 
    credit_impact  = (portfolio["credit_risk"] / 100) * c * 0.20 * reinsurer_dampener

    # Incorporate Fraud Flag & Model Governance into Operational Shock
    fraud_penalty  = 1.0 + portfolio.get("fraud_flag", 0.0) * 0.5
    
    # Model Risk Penalty: If AUC is low (< 0.70) or PSI is high (> 0.1, indicating drift), add penalty
    model_penalty  = 1.0
    if portfolio.get("auc", 0.85) < 0.70 or portfolio.get("psi", 0.0) > 0.1:
        model_penalty = 1.25 # 25% increase to operational risk if underlying ML models are drifting
        
    op_impact      = (portfolio["operational_risk"] / 100) * c * 0.15 * fraud_penalty * model_penalty

    total_consumed = claim_impact + market_impact + cat_impact + credit_impact + op_impact

    remaining      = c - total_consumed
    solvency_ratio = (remaining / c * 100) if c > 0 else 0
    shortfall      = max(0.0, -remaining)

    return {
        "scenario_id":       scenario_id,
        "label":             cfg["label"],
        "stressed_claim":    round(stressed_claim,  2),
        "stressed_lr":       round(stressed_lr,     2),
        "stressed_cr":       round(stressed_cr,     2),
        "stressed_uw":       round(stressed_uw,     2),
        "claim_impact":      round(claim_impact,    2),
        "market_impact":     round(market_impact,   2),
        "cat_impact":        round(cat_impact,      2),
        "credit_impact":     round(credit_impact,   2),
        "op_impact":         round(op_impact,       2),
        "capital_consumed":  round(total_consumed,  2),
        "remaining_capital": round(remaining,       2),
        "solvency_ratio":    round(solvency_ratio,  2),
        "shortfall":         round(shortfall,       2),
        "is_solvent":        remaining >= 0,
    }


def _build_comparison(portfolio):
    base_lr = (portfolio["claim_amount"] / portfolio["written_premium"] * 100) \
              if portfolio["written_premium"] > 0 else 0
    base_er = (portfolio["total_expense"] / portfolio["written_premium"] * 100) \
              if portfolio["written_premium"] > 0 else 0
    rows = []
    for sid in SCENARIO_CONFIG:
        sr = _run_scenario(portfolio, sid)
        rows.append({
            "Scenario":           sid.upper(),
            "Label":              SCENARIO_CONFIG[sid]["label"],
            "Base CR (%)":        round(base_lr + base_er, 2),
            "Stressed CR (%)":    sr["stressed_cr"],
            "Stressed LR (%)":    sr["stressed_lr"],
            "Capital Consumed":   sr["capital_consumed"],
            "Solvency Ratio (%)": sr["solvency_ratio"],
            "Shortfall":          sr["shortfall"],
            "Is Solvent":         sr["is_solvent"],
        })
    return pd.DataFrame(rows)


def _product_vulnerability(portfolio, scenario_result):
    p = portfolio["written_premium"]
    c = portfolio["capital_reserve"]
    rows = []
    for product, weight in PRODUCT_WEIGHTS.items():
        ce  = scenario_result["stressed_claim"]   * weight
        car = scenario_result["capital_consumed"] * weight
        vs  = (ce / p * 100 if p > 0 else 0) + (car / c * 50 if c > 0 else 0)
        rows.append({
            "Product":             product,
            "Claim Exposure":      round(ce,  2),
            "Capital At Risk":     round(car, 2),
            "Vulnerability Score": round(vs,  2),
            "Vulnerability":       "High" if vs > 55 else "Medium" if vs > 30 else "Low",
        })
    return pd.DataFrame(rows).sort_values("Vulnerability Score", ascending=False).reset_index(drop=True)


def run_stress_pipeline(df_risk, scenario_id="s4"):
    """
    Main entry point called by orchestrator.py:
        from agents.stress_testing import run_stress_pipeline
        stress_results = run_stress_pipeline(df_risk)

    Args:
        df_risk:     DataFrame from run_risk_pipeline()
        scenario_id: 's1'-'s5', default 's4' (Combined Shock)

    Returns:
        dict with keys: portfolio, scenario_result, all_scenarios,
                        product_vuln, scenario_label, is_solvent,
                        solvency_ratio, shortfall, kpis
    """
    portfolio       = _extract_portfolio(df_risk)
    scenario_result = _run_scenario(portfolio, scenario_id)
    all_scenarios   = _build_comparison(portfolio)
    product_vuln    = _product_vulnerability(portfolio, scenario_result)

    base_lr = (portfolio["claim_amount"] / portfolio["written_premium"] * 100) \
              if portfolio["written_premium"] > 0 else 0
    base_er = (portfolio["total_expense"] / portfolio["written_premium"] * 100) \
              if portfolio["written_premium"] > 0 else 0

    kpis = {
        "Base_Loss_Ratio":         round(base_lr, 2),
        "Base_Expense_Ratio":      round(base_er, 2),
        "Base_Combined_Ratio":     round(base_lr + base_er, 2),
        "Stressed_Loss_Ratio":     scenario_result["stressed_lr"],
        "Stressed_Combined_Ratio": scenario_result["stressed_cr"],
        "Capital_Consumed":        scenario_result["capital_consumed"],
        "Remaining_Capital":       scenario_result["remaining_capital"],
        "Solvency_Ratio":          scenario_result["solvency_ratio"],
        "Shortfall":               scenario_result["shortfall"],
        "VaR_99":                  round(portfolio.get("var_99", 0.0), 2),
        "Expected_Shortfall":      round(portfolio.get("expected_shortfall", 0.0), 2),
        "Model_PSI":               round(portfolio.get("psi", 0.0), 4),
        "Model_AUC":               round(portfolio.get("auc", 0.0), 4),
        "Scenario":                scenario_result["label"],
    }

    return {
        "portfolio":       portfolio,
        "scenario_result": scenario_result,
        "all_scenarios":   all_scenarios,
        "product_vuln":    product_vuln,
        "scenario_label":  scenario_result["label"],
        "is_solvent":      scenario_result["is_solvent"],
        "solvency_ratio":  scenario_result["solvency_ratio"],
        "shortfall":       scenario_result["shortfall"],
        "kpis":            kpis,
    }
