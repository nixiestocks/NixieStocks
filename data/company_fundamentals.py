from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


COUNTRY_BY_SUFFIX = {
    ".NS": "India",
    ".BO": "India",
    ".L": "United Kingdom",
    ".TO": "Canada",
    ".V": "Canada",
    ".AX": "Australia",
    ".NZ": "New Zealand",
    ".HK": "Hong Kong",
    ".SI": "Singapore",
    ".T": "Japan",
    ".KS": "South Korea",
    ".KQ": "South Korea",
    ".SS": "China",
    ".SZ": "China",
    ".DE": "Germany",
    ".F": "Germany",
    ".PA": "France",
    ".AS": "Netherlands",
    ".BR": "Belgium",
    ".MI": "Italy",
    ".MC": "Spain",
    ".SW": "Switzerland",
    ".ST": "Sweden",
    ".OL": "Norway",
    ".CO": "Denmark",
    ".JO": "South Africa",
    ".SA": "Brazil",
    ".MX": "Mexico",
}


def _safe_number(value):
    try:
        if value is None:
            return None
        number = float(value)
        if pd.isna(number):
            return None
        return number
    except Exception:
        return None


def _usable_text(value):
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    return text


def _country_from_symbol(symbol):
    symbol = str(symbol or "").upper()
    for suffix, country in COUNTRY_BY_SUFFIX.items():
        if symbol.endswith(suffix):
            return country

    # Yahoo symbols without an exchange suffix are most commonly US-listed
    # equities (NASDAQ/NYSE/AMEX) in this app's equity-only search flow.
    if symbol and "." not in symbol:
        return "United States"

    return "Global"


def _fast_value(fast, *names):
    if fast is None:
        return None

    for name in names:
        try:
            value = getattr(fast, name)
        except Exception:
            value = None

        if value is not None:
            return value

        try:
            value = fast.get(name)
        except Exception:
            value = None

        if value is not None:
            return value

    return None


def _statement_value(statement, labels):
    if statement is None or getattr(statement, "empty", True):
        return None

    try:
        normalized = {
            str(index).replace(" ", "").replace("_", "").lower(): index
            for index in statement.index
        }

        for label in labels:
            key = str(label).replace(" ", "").replace("_", "").lower()
            index = normalized.get(key)
            if index is None:
                continue

            values = pd.to_numeric(
                statement.loc[index],
                errors="coerce",
            ).dropna()

            if not values.empty:
                return _safe_number(values.iloc[0])
    except Exception:
        pass

    return None


def _load_statement(stock):
    candidates = []

    try:
        candidates.append(stock.ttm_income_stmt)
    except Exception:
        pass

    try:
        candidates.append(stock.get_income_stmt(freq="yearly"))
    except Exception:
        pass

    try:
        candidates.append(stock.income_stmt)
    except Exception:
        pass

    for statement in candidates:
        if statement is not None and not statement.empty:
            return statement

    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamental_fallbacks(symbol):
    symbol = str(symbol or "").strip().upper()

    result = {
        "country": _country_from_symbol(symbol),
        "pe_ratio": None,
        "eps_original": None,
        "dividend_yield": None,
        "summary": None,
        "website": None,
    }

    if not symbol:
        return result

    stock = yf.Ticker(symbol)

    info = {}
    try:
        info = stock.get_info() or {}
    except Exception:
        try:
            info = stock.info or {}
        except Exception:
            info = {}

    result["country"] = (
        _usable_text(info.get("country"))
        or result["country"]
    )
    result["pe_ratio"] = _safe_number(info.get("trailingPE"))
    result["eps_original"] = _safe_number(info.get("trailingEps"))
    result["dividend_yield"] = _safe_number(info.get("dividendYield"))
    result["summary"] = _usable_text(info.get("longBusinessSummary"))
    result["website"] = _usable_text(info.get("website"))

    # Financial-statement fallback for EPS when quoteSummary/info is blocked.
    if result["eps_original"] is None:
        try:
            statement = _load_statement(stock)
        except Exception:
            statement = None

        eps = _statement_value(
            statement,
            ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"],
        )

        if eps is None:
            net_income = _statement_value(
                statement,
                ["Net Income", "NetIncome", "Net Income Common Stockholders"],
            )

            shares = None
            try:
                fast = stock.fast_info
            except Exception:
                fast = None

            shares = _safe_number(
                _fast_value(
                    fast,
                    "shares",
                    "shares_outstanding",
                    "sharesOutstanding",
                )
            )

            if shares is None:
                market_cap = _safe_number(
                    _fast_value(fast, "market_cap", "marketCap")
                )
                last_price = _safe_number(
                    _fast_value(fast, "last_price", "lastPrice")
                )
                if market_cap is not None and last_price not in (None, 0):
                    shares = market_cap / last_price

            if net_income is not None and shares not in (None, 0):
                eps = net_income / shares

        result["eps_original"] = _safe_number(eps)

    # Dividend-yield fallback from the last 12 months of cash dividends.
    if result["dividend_yield"] is None:
        try:
            history = stock.history(
                period="1y",
                interval="1d",
                auto_adjust=False,
                actions=True,
            )

            if history is not None and not history.empty and "Dividends" in history.columns:
                dividends = pd.to_numeric(
                    history["Dividends"],
                    errors="coerce",
                ).fillna(0).sum()

                close = pd.to_numeric(
                    history["Close"],
                    errors="coerce",
                ).dropna()

                if not close.empty and float(close.iloc[-1]) != 0:
                    result["dividend_yield"] = float(dividends) / float(close.iloc[-1])
        except Exception:
            pass

    return result


def enrich_stock_info(symbol, base_info):
    if base_info is None:
        return None

    info = dict(base_info)
    fallback = load_fundamental_fallbacks(symbol)

    if not _usable_text(info.get("country")):
        info["country"] = fallback.get("country") or "Global"

    eps_original = _safe_number(info.get("original_eps"))
    if eps_original is None:
        eps_original = _safe_number(fallback.get("eps_original"))

    inr_rate = _safe_number(info.get("inr_rate"))
    eps_inr = _safe_number(info.get("eps"))

    if eps_inr is None and eps_original is not None:
        eps_inr = eps_original * inr_rate if inr_rate is not None else eps_original
        info["eps"] = eps_inr
        info["original_eps"] = eps_original

    pe = _safe_number(info.get("pe_ratio"))
    if pe is None:
        pe = _safe_number(fallback.get("pe_ratio"))

    if pe is None:
        original_price = _safe_number(info.get("original_price"))
        if original_price is not None and eps_original is not None and eps_original > 0:
            pe = original_price / eps_original

    if pe is not None and pe > 0:
        info["pe_ratio"] = pe
        info["trailingPE"] = pe

    dividend = _safe_number(info.get("dividend_yield"))
    if dividend is None:
        dividend = _safe_number(fallback.get("dividend_yield"))

    if dividend is not None:
        info["dividend_yield"] = dividend
        info["dividendYield"] = dividend
        info["dividend"] = dividend

    if not _usable_text(info.get("website")):
        website = _usable_text(fallback.get("website"))
        if website:
            info["website"] = website

    if not _usable_text(info.get("summary")):
        summary = _usable_text(fallback.get("summary"))

        if not summary:
            name = _usable_text(info.get("name")) or str(symbol).upper()
            sector = _usable_text(info.get("sector")) or "the listed-equities market"
            industry = _usable_text(info.get("industry"))
            country = _usable_text(info.get("country")) or "its home market"

            if industry:
                summary = (
                    f"{name} is a publicly listed company in the {industry} industry "
                    f"within the {sector} sector, based in {country}. "
                    "A detailed business description is temporarily unavailable from the market-data provider."
                )
            else:
                summary = (
                    f"{name} is a publicly listed company in the {sector} sector, "
                    f"based in {country}. A detailed business description is temporarily "
                    "unavailable from the market-data provider."
                )

        info["summary"] = summary

    return info
