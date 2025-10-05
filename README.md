# Earnings Call Sentiment & Market Impact Analyzer

**Overview**  
This project analyzes corporate earnings call transcripts to quantify spoken sentiment and measure its short-term market impact. It combines NLP (FinBERT) sentiment scoring of speaker turns with equities price data (ticker vs. S&P 500) to produce event-level sentiment features and predict post-event excess returns (alpha). The notebook demonstrates preprocessing, feature engineering, visualization, model training, and evaluation.

---

## Key Objectives
- Parse and clean earnings call transcripts into speaker-level turns.  
- Compute fine-grained sentiment (FinBERT) for each speaker turn.  
- Aggregate sentiment into presentation vs. Q&A and compute speaker-weighted sentiment.  
- Merge sentiment features with price data (ticker and S&P 500) and compute N-day returns and alpha.  
- Train regression models (Random Forest, XGBoost) to predict future alpha and evaluate predictive and directional performance using Leave-One-Out CV.

---

## What’s in the Notebook
- **Data acquisition**: download historical price series via `yfinance`.  
- **Transcript parsing**: load raw transcript files, parse filenames for date/ticker and split transcripts into speaker turns.  
- **Text cleaning**: remove boilerplate, disclaimers, and noisy tokens; handle Q&A vs. presentation sections.  
- **Sentiment scoring**: inference with `ProsusAI/finbert` (transformers) to score positive/negative/neutral and construct a composite sentiment metric.  
- **Aggregation & weighting**:
  - Compute `presentation_sentiment` and `qa_sentiment` (section means).
  - Compute speaker-weighted sentiment and an `avg_sentiment` feature.  
- **Price features**: compute future returns for horizons (1d, 5d, 21d) for the ticker and S&P 500, then compute `alpha_n` = ticker_return − sp500_return.  
- **Modeling & evaluation**:
  - Train `RandomForestRegressor` (and XGBoost optionally) to predict alpha.
  - Evaluate with Leave-One-Out Cross Validation (LOOCV) reporting MSE / RMSE.
  - Compute **directional accuracy** (percentage of times model correctly predicted sign of alpha).
  - Plot correlation heatmaps, regression plots, and feature importances.

---

## Requirements

Install required packages (conda/pip). FinBERT requires `transformers` and `torch` and benefits from GPU acceleration.

```bash
pip install pandas numpy yfinance transformers torch scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn


