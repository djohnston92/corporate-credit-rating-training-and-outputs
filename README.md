# Credit Rating Model

Training and scoring pipeline for a multinomial logistic regression model that predicts a corporate credit rating bucket (High Risk / Speculative / Low Investment Grade / Investment Grade) from a company's own reported financial statements — no ticker lookup, no vendor credit score, just the same ratios a human credit analyst would compute by hand.

This is the model behind the live app at [danielscreditrating.streamlit.app](https://danielscreditrating.streamlit.app) ([app repo](https://github.com/djohnston92/credit_ratings)) and the original write-up, [Move Over Moody's, My Python Script Has Opinions](https://medium.com/@djohnston92). That repo ships the trained artifacts for serving; this one is the actual training/scoring pipeline that produced them.

## How it works

- **Features (20 total)**: standard profitability, leverage, liquidity, cash flow, and efficiency ratios (margins, ROA/ROE, debt/equity, current ratio, cash ratio, turnover ratios, etc.), plus an Altman Z-Score and three hand-picked interaction terms (cash × working capital, operating cash flow × debt, margin × ROA).
- **Model**: `LogisticRegression(multi_class='multinomial', class_weight='balanced')` from scikit-learn. Balanced class weights matter here — investment-grade companies dominate any real financial dataset, so an unweighted model would just learn to always predict "safe."
- **Validation**: 5-fold `StratifiedKFold`, split by ticker (not by row) so no company's data leaks across the train/test boundary.
- **Training data**: SEC XBRL financials (trailing-twelve-month), labeled against real published credit ratings as the answer key.

## Structure

```
model/
  train_model.py                    # pulls training data, engineers features, trains + saves the model
  credit_rating_model_ttm.pkl       # trained LogisticRegression
  credit_rating_label_encoder.pkl   # maps rating category <-> encoded int
  credit_rating_mappings.pkl        # feature column list used at train time
  credit_rating_model_ttm.meta.json # version, training date, sample size, feature list
score/
  predictor.py                      # loads the trained model, scores a ticker's TTM financials
```

## Running it

Both scripts expect a Postgres connection via environment variables — copy `.env.example` to `.env` and fill in your own database:

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env
python model/train_model.py     # retrain from scratch, overwrites the pkl/json artifacts in model/
python score/predictor.py       # score a ticker (edit the `tickers` list at the top)
```

Both scripts assume a source table shaped like `python.ml_corporate_credit_ratings_answer_key_w_financials_mv` (training) / `python.ml_corporate_credit_ratings_financials_mv` (scoring) — a wide table of TTM financial statement line items keyed by ticker and reporting date. Swap the SQL for your own data source if you're adapting this.

## Caveats

- Training data is XBRL-sourced and not every company reports every field the same way — the pipeline drops rows with missing/inf values in any of the 20 features rather than imputing, which trims the usable sample.
- This predicts a rating **bucket**, not a notch-level rating (e.g. not "BBB+" vs "BBB") — see the app's disclaimer for the full caveat on what this is and isn't.
