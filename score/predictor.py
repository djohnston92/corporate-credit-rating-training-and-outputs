import os
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv("/home/dad/Documents/Python_Projects/.env")

MODEL_DIR = Path(__file__).parent.parent / "model"

clf = joblib.load(MODEL_DIR / "credit_rating_model_ttm.pkl")
label_encoder = joblib.load(MODEL_DIR / "credit_rating_label_encoder.pkl")

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('PG_SUPERUSER_USER')}:{os.getenv('PG_SUPERUSER_PASSWORD')}"
    f"@{os.getenv('PG_SUPERUSER_HOST')}:{os.getenv('PG_SUPERUSER_PORT')}/{os.getenv('PG_SUPERUSER_DB')}"
)

# Tickers to score — swap for a real query (e.g. against a tickers table) as needed.




source_codes_query = """
select symbol from bronze.yahoo_finance_consolidated_tickers_vw
"""
source_codes_df = pd.read_sql(source_codes_query, engine)
tickers = source_codes_df['symbol'].tolist()


# tickers = ['ORCL']


def engineer_financial_ratios(df):
    df = df.copy()

    # Safe denominators
    df['equity_safe']       = df['equity'].where(df['equity'] > 0, np.nan)
    df['liabilities_safe']  = df['liabilities'].replace(0, np.nan)
    df['totalrev_safe']     = df['totalrev'].replace(0, np.nan)
    df['assets_safe']       = df['assets'].replace(0, np.nan)
    df['longtermdebt_safe'] = df['longtermdebt'].replace(0, np.nan)
    df['payables_safe']     = df['payables'].replace(0, np.nan)
    df['ebitda_safe']       = df['ebitda'].replace(0, np.nan)

    df["Operating_Cash_Flow_Ratio"] = df["operatingcash"] / df["liabilities_safe"]

    # Profitability
    df["ebitda_margin"]    = df["ebitda_safe"]                / df["totalrev_safe"]
    df["net_margin"]       = df["net_income_common_stock"]    / df["totalrev_safe"]
    df["roa"]              = df["net_income_common_stock"]    / df["assets_safe"]
    df["roe"]              = df["net_income_common_stock"]    / df["equity_safe"]
    df["operating_margin"] = df["operating_income"]           / df["totalrev_safe"]

    # Leverage
    df["debt_to_equity"]  = df["longtermdebt"] / df["equity_safe"]
    df["debt_to_assets"]  = df["longtermdebt"] / df["assets_safe"]
    df["leverage_ratio"]  = df["liabilities"]  / df["equity_safe"]

    # Liquidity
    df["current_ratio"]       = (df["receivables"] + df["inventory"] + df["cash"]) / df["payables_safe"]
    df["workingcapital_ratio"] = df["workingcapital"] / df["assets_safe"]
    df["cash_ratio"]           = df["cash"] / df["liabilities_safe"]

    # Cash Flow
    df["ocf_to_debt"] = df["operatingcash"] / df["longtermdebt_safe"]
    df["fcf_margin"]  = df["freecashflow"]  / df["totalrev_safe"]

    # Efficiency
    df["receivables_turnover"] = df["totalrev"] / df["receivables"]
    df["inventory_turnover"]   = df["cogs"]     / df["inventory"]
    df["asset_turnover"]       = df["totalrev"] / df["assets_safe"]

    # Interaction terms
    df["cash_x_wcap"]  = df["cash_ratio"]                * df["workingcapital_ratio"]
    df["ocf_x_debt"]   = df["Operating_Cash_Flow_Ratio"] * df["debt_to_assets"]
    df["margin_x_roa"] = df["operating_margin"]           * df["roa"]

    # Altman Z-Score
    A = df["workingcapital"]   / df["assets_safe"]
    B = df["retainedearnings"] / df["assets_safe"]
    C = df["ebit"]             / df["assets_safe"]
    D = df["equity_safe"]      / df["liabilities_safe"]
    E = df["totalrev"]         / df["assets_safe"]
    df["altman_z"] = (1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E).fillna(0)

    df = df.drop(columns=[
        "equity_safe", "liabilities_safe", "totalrev_safe", "assets_safe",
        "longtermdebt_safe", "payables_safe", "ebitda_safe",
    ])

    return df


feature_cols = [
    "ebitda_margin", "net_margin", "roa", "roe", "operating_margin",
    "debt_to_equity", "debt_to_assets", "leverage_ratio", "current_ratio",
    "workingcapital_ratio", "cash_ratio", "ocf_to_debt", "fcf_margin",
    "receivables_turnover", "inventory_turnover", "asset_turnover",
    "cash_x_wcap", "ocf_x_debt", "margin_x_roa", "altman_z",
]

# class index -> probability column name; matches label_encoder.classes_ order
PROB_COL_MAP = {
    0: 'prob_high_risk',
    1: 'prob_speculative',
    2: 'prob_low_invest',
    3: 'prob_ig',
}

CONFIDENCE_THRESHOLD = 0.51  # below this, flag the prediction as low-confidence

for ticker in tickers:

    date_query = f"""
        SELECT "date" AS date
        FROM python.ml_corporate_credit_ratings_financials_mv
        WHERE "TICKER" = '{ticker}'
          AND "date" >= '2000-01-01'
        GROUP BY "date"
    """
    df_report_date = pd.read_sql(date_query, engine)
    date_list = df_report_date['date'].tolist()

    for report_date in date_list:

        df_query = f"""
            SELECT *
            FROM python.ml_corporate_credit_ratings_financials_mv
            WHERE "TICKER" = '{ticker}'
              AND "date" = '{report_date}'
        """
        df = pd.read_sql(df_query, engine)

        df = engineer_financial_ratios(df)

        row_df = df[feature_cols].fillna(0)
        row_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        row_df.fillna(0, inplace=True)

        pred_encoded = clf.predict(row_df.iloc[[0]])[0]
        predicted_label = label_encoder.inverse_transform([pred_encoded])[0]

        pred_proba = clf.predict_proba(row_df.iloc[[0]])[0]

        high_risk_prob   = pred_proba[0] if len(pred_proba) > 0 else 0.0
        prob_speculative = pred_proba[1] if len(pred_proba) > 1 else 0.0
        prob_low_invest  = pred_proba[2] if len(pred_proba) > 2 else 0.0
        prob_ig          = pred_proba[3] if len(pred_proba) > 3 else 0.0

        flag = pred_proba.max() < CONFIDENCE_THRESHOLD

        print(f"{ticker} | {report_date} | Predicted: {predicted_label} | "
              f"Max prob: {pred_proba.max():.2f} | Gut-check flag: {int(flag)}")

        result_df = pd.DataFrame({
            'TICKER':           [ticker],
            'date':             [report_date],
            'predicted_rating': [predicted_label],
            'prob_high_risk':   [high_risk_prob],
            'prob_speculative': [prob_speculative],
            'prob_low_invest':  [prob_low_invest],
            'prob_ig':          [prob_ig],
            'maxprob':          [pred_proba.max()],
            'gut_check_flag':   [1 if flag else 0],
            'report_type':      ['TTM'],
            'LOAD_DATE':        [datetime.now()],
        })

        result_df.to_sql(
            'ML_CORPORATE_CREDIT_RATINGS_OUTPUT_ENCHANCED',
            con=engine,
            schema='python',
            if_exists='append',
            index=False
        )
