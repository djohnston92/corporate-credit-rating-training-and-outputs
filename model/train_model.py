import os
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
from datetime import date

OUT_DIR = Path(__file__).parent
load_dotenv("/home/dad/Documents/Python_Projects/.env")

# ── DB ────────────────────────────────────────────────────────────────────────
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('PG_SUPERUSER_USER')}:{os.getenv('PG_SUPERUSER_PASSWORD')}"
    f"@{os.getenv('PG_SUPERUSER_HOST')}:{os.getenv('PG_SUPERUSER_PORT')}/{os.getenv('PG_SUPERUSER_DB')}"
)

df = pd.read_sql(
    "SELECT * FROM python.ml_corporate_credit_ratings_answer_key_w_financials_mv",
    engine
)
n_before = len(df)
df = df[df["category"].notna() & (df["category"].astype(str).str.strip() != "")]
n_after = len(df)
if n_before != n_after:
    print(f"Dropped {n_before - n_after} rows with missing/blank category")


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def engineer_financial_ratios(df):
    df = df.copy()
    df = df.sort_values(['TICKER', 'date']).reset_index(drop=True)

    df['equity_safe']       = df['equity'].where(df['equity'] > 0, np.nan)
    df['liabilities_safe']  = df['liabilities'].replace(0, np.nan)
    df['totalrev_safe']     = df['totalrev'].replace(0, np.nan)
    df['assets_safe']       = df['assets'].replace(0, np.nan)
    df['longtermdebt_safe'] = df['longtermdebt'].replace(0, np.nan)
    df['payables_safe']     = df['payables'].replace(0, np.nan)
    df['ebitda_safe']       = df['ebitda'].replace(0, np.nan)
    df["Operating_Cash_Flow_Ratio"] = df["operatingcash"] / df["liabilities_safe"]

    # Profitability
    df["ebitda_margin"]    = df["ebitda_safe"]       / df["totalrev_safe"]
    df["net_margin"]       = df["net_income_common_stock"] / df["totalrev_safe"]
    df["roa"]              = df["net_income_common_stock"] / df["assets_safe"]
    df["roe"]              = df["net_income_common_stock"] / df["equity_safe"]
    df["operating_margin"] = df["operating_income"]   / df["totalrev_safe"]

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

    # Altman Z
    A = df["workingcapital"]   / df["assets_safe"]
    B = df["retainedearnings"] / df["assets_safe"]
    C = df["ebit"]             / df["assets_safe"]
    D = df["equity_safe"]      / df["liabilities_safe"]
    E = df["totalrev"]         / df["assets_safe"]

    # Interaction terms
    df["cash_x_wcap"]  = df["cash_ratio"]             * df["workingcapital_ratio"]
    df["ocf_x_debt"]   = df["Operating_Cash_Flow_Ratio"] * df["debt_to_assets"]
    df["margin_x_roa"] = df["operating_margin"]        * df["roa"]

    df["altman_z"] = (1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E).fillna(0)

    return df

df = engineer_financial_ratios(df)

# ── FEATURE LIST ──────────────────────────────────────────────────────────────
feature_cols = [
    "ebitda_margin", "net_margin", "roa", "roe", "operating_margin",
    "debt_to_equity", "debt_to_assets", "leverage_ratio", "current_ratio",
    "workingcapital_ratio", "cash_ratio", "ocf_to_debt", "fcf_margin",
    "receivables_turnover", "inventory_turnover", "asset_turnover",
    "cash_x_wcap", "ocf_x_debt", "margin_x_roa", "altman_z",
]

# ── CLEAN DATA ────────────────────────────────────────────────────────────────
print("Records per bucket BEFORE dropna:")
print(df.groupby("category")["TICKER"].agg(['count', 'nunique']))

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

print("\nRecords per bucket AFTER dropna:")
print(df.groupby("category")["TICKER"].agg(['count', 'nunique']))

# ── LABEL ENCODING ────────────────────────────────────────────────────────────
label_encoder = LabelEncoder()
df["encoded_rating"] = label_encoder.fit_transform(df["category"])

# ── CROSS-VALIDATION (grouped by ticker, so no company leaks across folds) ────
ticker_labels = df.groupby("TICKER")["encoded_rating"].last().reset_index()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []
fold_scores = []

for fold, (train_idx, test_idx) in enumerate(
        skf.split(ticker_labels["TICKER"], ticker_labels["encoded_rating"])):

    train_tickers = ticker_labels.iloc[train_idx]["TICKER"]
    test_tickers  = ticker_labels.iloc[test_idx]["TICKER"]

    train_df = df[df["TICKER"].isin(train_tickers)]
    test_df  = df[df["TICKER"].isin(test_tickers)]

    X_train = train_df[feature_cols]
    y_train = train_df["encoded_rating"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["encoded_rating"]

    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=10000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    fold_f1 = f1_score(y_test, y_pred, average='macro')
    fold_scores.append(fold_f1)
    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    print(f"Fold {fold+1}: macro F1 = {fold_f1:.3f}")

print(f"\nCV Macro F1: {np.mean(fold_scores):.3f} +/- {np.std(fold_scores):.3f}")

# ── FULL CV REPORT ────────────────────────────────────────────────────────────
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

print("\n── Classification Report (all folds combined) ──")
print(classification_report(
    all_y_true,
    all_y_pred,
    labels=np.unique(all_y_true),
    target_names=label_encoder.inverse_transform(np.unique(all_y_true))
))

# ── FINAL FIT — retrain on full data ──────────────────────────────────────────
clf_final = LogisticRegression(
    class_weight='balanced',
    max_iter=10000,
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42
)
clf_final.fit(df[feature_cols], df["encoded_rating"])

# Feature importance via coefficients
coef_df = pd.DataFrame(
    clf_final.coef_,
    columns=feature_cols,
    index=label_encoder.classes_
)
feature_importance_df = (
    coef_df.abs()
    .mean(axis=0)
    .sort_values(ascending=False)
    .reset_index()
)
feature_importance_df.columns = ["feature", "importance"]

# Confusion matrix, saved next to this script
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig(OUT_DIR / 'confusion_matrix.png')
print(f"\nSaved {OUT_DIR / 'confusion_matrix.png'}")

#── SAVE ARTIFACTS ─────────────────────────────────────────────────────────────
meta = {
    "version": "1.01",
    "trained_on": str(date.today()),
    "training_n": len(all_y_true),
    "threshold": 0.5,
    "features": feature_cols,
    "notes": "initial model, TTM features only",
}

joblib.dump(clf_final, OUT_DIR / "credit_rating_model_ttm.pkl")
joblib.dump(label_encoder, OUT_DIR / "credit_rating_label_encoder.pkl")
joblib.dump({"feature_cols": feature_cols}, OUT_DIR / "credit_rating_mappings.pkl")

with open(OUT_DIR / "credit_rating_model_ttm.meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nModel, encoder, mappings, and meta.json saved.")
