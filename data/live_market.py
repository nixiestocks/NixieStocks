from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from data.yahoo import get_stock_history, get_stock_info


MARKETS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC",
}

STOCK_UNIVERSE = [
    ("NVIDIA Corporation", "NVDA", "NASDAQ", "USD"),
    ("Apple Inc.", "AAPL", "NASDAQ", "USD"),
    ("Microsoft Corporation", "MSFT", "NASDAQ", "USD"),
    ("Tesla Inc.", "TSLA", "NASDAQ", "USD"),
    ("Amazon.com Inc.", "AMZN", "NASDAQ", "USD"),
    ("Alphabet Inc.", "GOOGL", "NASDAQ", "USD"),
    ("Meta Platforms Inc.", "META", "NASDAQ", "USD"),
    ("Reliance Industries", "RELIANCE.NS", "NSE", "INR"),
    ("Tata Consultancy Services", "TCS.NS", "NSE", "INR"),
    ("Infosys Limited", "INFY.NS", "NSE", "INR"),
    ("HDFC Bank Ltd.", "HDFCBANK.NS", "NSE", "INR"),
    ("State Bank of India", "SBIN.NS", "NSE", "INR"),
]

USDINR = "USDINR=X"


def _series_from_download(data, ticker, field="Close"):
    if data is None or data.empty:
        return pd.Series(dtype=float)

    try:
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:
                frame = data[ticker]
                if field in frame.columns:
                    return pd.to_numeric(
                        frame[field],
                        errors="coerce",
                    ).dropna()

            if field in level0 and ticker in level1:
                return pd.to_numeric(
                    data[field][ticker],
                    errors="coerce",
                ).dropna()

        if field in data.columns:
            return pd.to_numeric(
                data[field],
                errors="coerce",
            ).dropna()

    except Exception:
        pass

    return pd.Series(dtype=float)


def _date_only(value):
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _latest_and_previous(daily_close, intraday_close):
    if intraday_close is not None and not intraday_close.empty:
        latest = float(intraday_close.iloc[-1])

        if daily_close is None or daily_close.empty:
            return latest, None

        intraday_date = _date_only(intraday_close.index[-1])
        daily_date = _date_only(daily_close.index[-1])

        if intraday_date == daily_date and len(daily_close) >= 2:
            previous = float(daily_close.iloc[-2])
        else:
            previous = float(daily_close.iloc[-1])

        return latest, previous

    if daily_close is not None and len(daily_close) >= 2:
        return float(daily_close.iloc[-1]), float(daily_close.iloc[-2])

    if daily_close is not None and len(daily_close) == 1:
        return float(daily_close.iloc[-1]), None

    return None, None


def _percent_change(latest, previous):
    if latest is None or previous in (None, 0):
        return 0.0
    return ((latest - previous) / previous) * 100


@st.cache_data(ttl=60, show_spinner=False)
def _download_home_batches():
    stock_tickers = [item[1] for item in STOCK_UNIVERSE]
    all_tickers = list(MARKETS.values()) + stock_tickers + [USDINR]

    try:
        daily = yf.download(
            tickers=all_tickers,
            period="5d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        daily = pd.DataFrame()

    try:
        intraday = yf.download(
            tickers=all_tickers,
            period="1d",
            interval="5m",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception:
        intraday = pd.DataFrame()

    return daily, intraday


@st.cache_data(ttl=60, show_spinner=False)
def load_live_home_market_data():
    daily, intraday = _download_home_batches()

    if daily.empty and intraday.empty:
        return [], [], [], []

    usd_daily = _series_from_download(daily, USDINR)
    usd_intraday = _series_from_download(intraday, USDINR)
    usd_inr, _ = _latest_and_previous(usd_daily, usd_intraday)
    usd_inr = usd_inr or 1.0

    markets = []

    for name, ticker in MARKETS.items():
        daily_close = _series_from_download(daily, ticker)
        intraday_close = _series_from_download(intraday, ticker)
        latest, previous = _latest_and_previous(daily_close, intraday_close)

        sparkline = daily_close.tolist()
        if intraday_close is not None and not intraday_close.empty:
            live_value = float(intraday_close.iloc[-1])
            if not sparkline or abs(sparkline[-1] - live_value) > 1e-12:
                sparkline.append(live_value)

        markets.append(
            {
                "name": name,
                "price": latest,
                "change": _percent_change(latest, previous) if latest is not None else None,
                "sparkline": sparkline,
            }
        )

    movers = []

    for company, ticker, exchange, currency in STOCK_UNIVERSE:
        daily_close = _series_from_download(daily, ticker)
        intraday_close = _series_from_download(intraday, ticker)
        latest, previous = _latest_and_previous(daily_close, intraday_close)

        if latest is None:
            continue

        change = _percent_change(latest, previous)
        inr_price = latest if currency == "INR" else latest * usd_inr

        movers.append(
            {
                "company": company,
                "symbol": ticker,
                "exchange": exchange,
                "price": inr_price,
                "change": change,
            }
        )

    movers.sort(key=lambda item: item["change"], reverse=True)
    gainers = movers[:4]
    losers = sorted(movers, key=lambda item: item["change"])[:4]

    return markets, gainers, losers, movers


@st.cache_data(ttl=60, show_spinner=False)
def load_live_market_status():
    tickers = ["^NSEI", "^BSESN"]

    try:
        daily = yf.download(
            tickers=tickers,
            period="5d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        daily = pd.DataFrame()

    try:
        intraday = yf.download(
            tickers=tickers,
            period="1d",
            interval="5m",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception:
        intraday = pd.DataFrame()

    names = {
        "^NSEI": "NIFTY 50",
        "^BSESN": "SENSEX",
    }

    rows = []

    for ticker in tickers:
        daily_close = _series_from_download(daily, ticker)
        intraday_close = _series_from_download(intraday, ticker)
        latest, previous = _latest_and_previous(daily_close, intraday_close)

        if latest is None:
            continue

        rows.append(
            {
                "name": names[ticker],
                "price": latest,
                "change": _percent_change(latest, previous),
            }
        )

    return rows


@st.cache_data(ttl=120, show_spinner=False)
def load_live_stock_info(symbol):
    try:
        return get_stock_info(symbol)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _load_base_history(symbol, period):
    try:
        return get_stock_history(symbol, period)
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _load_intraday_history(symbol):
    try:
        return get_stock_history(symbol, "1d")
    except Exception:
        return None


def augment_history_with_live_session(history, symbol):
    if history is None or history.empty:
        return history

    intraday = _load_intraday_history(symbol)

    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        return history

    work = intraday.copy()

    if "Date" not in work.columns:
        work = work.reset_index()

    if "Datetime" in work.columns:
        work = work.rename(columns={"Datetime": "Date"})

    if "Date" not in work.columns:
        return history

    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date", "Close"])

    if work.empty:
        return history

    session_date = work["Date"].iloc[-1].date()
    session = work[work["Date"].dt.date == session_date]

    if session.empty:
        return history

    close = pd.to_numeric(session["Close"], errors="coerce").dropna()
    if close.empty:
        return history

    result = history.copy()

    if "Date" not in result.columns:
        result = result.reset_index()

    if "Datetime" in result.columns:
        result = result.rename(columns={"Datetime": "Date"})

    if "Date" not in result.columns:
        return history

    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")

    def first_numeric(column, fallback):
        if column not in session.columns:
            return fallback
        values = pd.to_numeric(session[column], errors="coerce").dropna()
        return float(values.iloc[0]) if not values.empty else fallback

    def max_numeric(column, fallback):
        if column not in session.columns:
            return fallback
        values = pd.to_numeric(session[column], errors="coerce").dropna()
        return float(values.max()) if not values.empty else fallback

    def min_numeric(column, fallback):
        if column not in session.columns:
            return fallback
        values = pd.to_numeric(session[column], errors="coerce").dropna()
        return float(values.min()) if not values.empty else fallback

    open_value = first_numeric("Open", float(close.iloc[0]))
    close_value = float(close.iloc[-1])
    high_value = max_numeric("High", max(open_value, close_value))
    low_value = min_numeric("Low", min(open_value, close_value))

    volume_value = 0.0
    if "Volume" in session.columns:
        volume = pd.to_numeric(session["Volume"], errors="coerce").dropna()
        if not volume.empty:
            volume_value = float(volume.sum())

    live_row = {}

    for column in result.columns:
        if column == "Date":
            live_row[column] = session["Date"].iloc[-1]
        elif column == "Open":
            live_row[column] = open_value
        elif column == "High":
            live_row[column] = high_value
        elif column == "Low":
            live_row[column] = low_value
        elif column in ("Close", "Adj Close"):
            live_row[column] = close_value
        elif column == "Volume":
            live_row[column] = volume_value
        elif column in ("Dividends", "Stock Splits", "Capital Gains"):
            live_row[column] = 0.0
        else:
            live_row[column] = None

    existing_dates = result["Date"].dt.date
    result = result[existing_dates != session_date]

    result = pd.concat(
        [result, pd.DataFrame([live_row])],
        ignore_index=True,
    )

    result = result.sort_values("Date").reset_index(drop=True)
    return result


def load_dashboard_history(symbol, period):
    history = _load_base_history(symbol, period)

    # 1y is also used by the AI and Daily Returns. Keep it strictly
    # completed daily history here. The chart receives a live overlay
    # separately in app.py so model inputs are not changed.
    if period == "1y":
        return history

    # These chart periods are daily/weekly historical views. Add the
    # current intraday session as the final visual candle/point.
    if period in {"3mo", "6mo", "2y", "5y"}:
        return augment_history_with_live_session(history, symbol)

    # 1d / 5d / 1mo are already intraday histories.
    return history
