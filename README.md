# TEAM7 — AI Stock Market Analyst

TEAM7 is a Streamlit stock-market analysis and forecasting app with worldwide company-name search, interactive charts, technical indicators, AI recommendations, sector-related stocks, daily returns, market news, comparison tools, and an optional deep-learning forecast.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

- Repository: `team7aiproject/AI-Stock-Market`
- Branch: `main`
- Entrypoint: `app.py`
- Recommended Python: `3.11`

The app uses Yahoo Finance market data. The Deep AI/LSTM section loads TensorFlow only when requested to keep normal dashboard usage lighter.

> Educational analysis only. This project does not provide financial advice.
