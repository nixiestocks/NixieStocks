import yfinance as yf
import pandas as pd
import time


# =========================================================
# FX CACHE
# =========================================================

_fx_cache = {}

FX_CACHE_SECONDS = 60 * 60


# =========================================================
# SAFE NUMBER
# =========================================================

def _safe_number(value):

    try:

        if value is None:
            return None

        value = float(value)

        if pd.isna(value):
            return None

        return value

    except Exception:
        return None


# =========================================================
# NORMALIZE CURRENCY
# =========================================================

def _normalize_currency(currency):

    if not currency:
        return "USD", 1.0

    raw = str(currency).strip()

    # -----------------------------------------
    # Yahoo sometimes reports minor currency
    # units instead of the main currency
    # -----------------------------------------

    if raw in ["GBp", "GBX"]:
        return "GBP", 0.01

    if raw == "ZAc":
        return "ZAR", 0.01

    if raw == "ILA":
        return "ILS", 0.01

    return raw.upper(), 1.0


# =========================================================
# GET LATEST FX PRICE
# =========================================================

def _get_latest_fx_price(ticker):

    try:

        fx = yf.Ticker(ticker)

        history = fx.history(
            period="5d",
            interval="1d",
            auto_adjust=False
        )

        if history is None or history.empty:
            return None

        close = history["Close"].dropna()

        if close.empty:
            return None

        value = float(
            close.iloc[-1]
        )

        if value <= 0:
            return None

        return value

    except Exception:
        return None


# =========================================================
# CURRENCY -> INR RATE
# =========================================================

def get_inr_rate(currency):

    base_currency, unit_multiplier = (
        _normalize_currency(currency)
    )

    # -----------------------------------------
    # Already INR
    # -----------------------------------------

    if base_currency == "INR":

        return unit_multiplier

    cache_key = (
        base_currency,
        unit_multiplier
    )

    now = time.time()

    cached = _fx_cache.get(
        cache_key
    )

    if cached:

        rate, timestamp = cached

        if (
            now - timestamp
            < FX_CACHE_SECONDS
        ):

            return rate

    # =========================================
    # DIRECT PAIR
    #
    # USD -> USDINR=X
    # EUR -> EURINR=X
    # GBP -> GBPINR=X
    # JPY -> JPYINR=X
    # =========================================

    direct_pair = (
        f"{base_currency}INR=X"
    )

    rate = _get_latest_fx_price(
        direct_pair
    )

    if rate is not None:

        rate = (
            rate
            * unit_multiplier
        )

        _fx_cache[
            cache_key
        ] = (
            rate,
            now
        )

        return rate

    # =========================================
    # INVERSE FALLBACK
    #
    # If XXXINR is unavailable,
    # try INRXXX and invert it.
    # =========================================

    inverse_pair = (
        f"INR{base_currency}=X"
    )

    inverse_rate = (
        _get_latest_fx_price(
            inverse_pair
        )
    )

    if (
        inverse_rate is not None
        and
        inverse_rate != 0
    ):

        rate = (
            1 / inverse_rate
        )

        rate = (
            rate
            * unit_multiplier
        )

        _fx_cache[
            cache_key
        ] = (
            rate,
            now
        )

        return rate

    return None


# =========================================================
# CONVERT VALUE TO INR
# =========================================================

def convert_to_inr(
    value,
    currency
):

    number = _safe_number(
        value
    )

    if number is None:
        return None

    rate = get_inr_rate(
        currency
    )

    if rate is None:
        return number

    return number * rate


# =========================================================
# FORMAT INR
# =========================================================

def format_inr(
    value,
    decimals=2
):

    number = _safe_number(
        value
    )

    if number is None:
        return "N/A"

    return (
        f"₹{number:,.{decimals}f}"
    )


# =========================================================
# STOCK INFORMATION
# =========================================================

def get_stock_info(symbol):

    try:

        symbol = (
            symbol.strip().upper()
        )

        stock = yf.Ticker(
            symbol
        )

        info = stock.info or {}

        # =====================================
        # FALLBACK PRICE
        # =====================================

        history = stock.history(
            period="5d",
            auto_adjust=False
        )

        latest_price = None

        if (
            history is not None
            and
            not history.empty
        ):

            close = (
                history["Close"]
                .dropna()
            )

            if not close.empty:

                latest_price = float(
                    close.iloc[-1]
                )

        # =====================================
        # ORIGINAL CURRENCY
        # =====================================

        original_currency = (
            info.get("currency")
            or "USD"
        )

        inr_rate = get_inr_rate(
            original_currency
        )

        # =====================================
        # ORIGINAL VALUES
        # =====================================

        original_price = (
            info.get("currentPrice")
            or
            info.get("regularMarketPrice")
            or
            latest_price
        )

        original_previous_close = (
            info.get("previousClose")
        )

        original_open = (
            info.get("open")
        )

        original_day_high = (
            info.get("dayHigh")
        )

        original_day_low = (
            info.get("dayLow")
        )

        original_high_52 = (
            info.get(
                "fiftyTwoWeekHigh"
            )
        )

        original_low_52 = (
            info.get(
                "fiftyTwoWeekLow"
            )
        )

        original_market_cap = (
            info.get("marketCap")
        )

        original_eps = (
            info.get("trailingEps")
        )

        # =====================================
        # CONVERT FUNCTION
        # =====================================

        def to_inr(value):

            number = _safe_number(
                value
            )

            if number is None:
                return None

            if inr_rate is None:
                return number

            return (
                number
                * inr_rate
            )

        conversion_available = (
            inr_rate is not None
        )

        display_currency = (
            "INR"
            if conversion_available
            else original_currency
        )

        # =====================================
        # RESULT
        # =====================================

        return {

            # ---------------------------------
            # BASIC INFORMATION
            # ---------------------------------

            "symbol":
                symbol,

            "name":
                info.get("longName")
                or info.get("shortName")
                or symbol,

            # ---------------------------------
            # INR DISPLAY VALUES
            # ---------------------------------

            "price":
                to_inr(
                    original_price
                ),

            "current_price":
                to_inr(
                    original_price
                ),

            "previous_close":
                to_inr(
                    original_previous_close
                ),

            "open":
                to_inr(
                    original_open
                ),

            "day_high":
                to_inr(
                    original_day_high
                ),

            "day_low":
                to_inr(
                    original_day_low
                ),

            "high_52":
                to_inr(
                    original_high_52
                ),

            "low_52":
                to_inr(
                    original_low_52
                ),

            "fifty_two_week_high":
                to_inr(
                    original_high_52
                ),

            "fifty_two_week_low":
                to_inr(
                    original_low_52
                ),

            "market_cap":
                to_inr(
                    original_market_cap
                ),

            "eps":
                to_inr(
                    original_eps
                ),

            # ---------------------------------
            # CURRENCY
            # ---------------------------------

            "currency":
                display_currency,

            "currency_symbol":
                (
                    "₹"
                    if conversion_available
                    else original_currency
                ),

            "inr_rate":
                inr_rate,

            "conversion_available":
                conversion_available,

            # ---------------------------------
            # ORIGINAL VALUES
            # Keep these for reference
            # ---------------------------------

            "original_currency":
                original_currency,

            "original_price":
                _safe_number(
                    original_price
                ),

            "original_previous_close":
                _safe_number(
                    original_previous_close
                ),

            "original_open":
                _safe_number(
                    original_open
                ),

            "original_day_high":
                _safe_number(
                    original_day_high
                ),

            "original_day_low":
                _safe_number(
                    original_day_low
                ),

            "original_high_52":
                _safe_number(
                    original_high_52
                ),

            "original_low_52":
                _safe_number(
                    original_low_52
                ),

            "original_market_cap":
                _safe_number(
                    original_market_cap
                ),

            "original_eps":
                _safe_number(
                    original_eps
                ),

            # ---------------------------------
            # FUNDAMENTALS
            # ---------------------------------

            "pe_ratio":
                info.get(
                    "trailingPE"
                ),

            "trailingPE":
                info.get(
                    "trailingPE"
                ),

            "dividend_yield":
                info.get(
                    "dividendYield"
                ),

            "dividendYield":
                info.get(
                    "dividendYield"
                ),

            "dividend":
                info.get(
                    "dividendYield"
                ),

            "volume":
                info.get(
                    "volume"
                ),

            # ---------------------------------
            # COMPANY INFORMATION
            # ---------------------------------

            "sector":
                info.get(
                    "sector",
                    "N/A"
                ),

            "industry":
                info.get(
                    "industry",
                    "N/A"
                ),

            "country":
                info.get(
                    "country",
                    "N/A"
                ),

            "website":
                info.get(
                    "website",
                    "N/A"
                ),

            "summary":
                info.get(
                    "longBusinessSummary",
                    "N/A"
                ),
        }

    except Exception as e:

        print(
            "Stock info error:",
            e
        )

        return None


# =========================================================
# STOCK HISTORY
# =========================================================

def get_stock_history(
    symbol,
    period="1y"
):

    interval_map = {

        "1d": "5m",

        "5d": "15m",

        "1mo": "1h",

        "3mo": "1d",

        "6mo": "1d",

        "1y": "1d",

        "2y": "1wk",

        "5y": "1wk",

        "max": "1mo",
    }

    interval = interval_map.get(
        period,
        "1d"
    )

    try:

        symbol = (
            symbol.strip().upper()
        )

        stock = yf.Ticker(
            symbol
        )

        # =====================================
        # FIND ORIGINAL CURRENCY
        # =====================================

        try:

            info = (
                stock.info or {}
            )

            original_currency = (
                info.get("currency")
                or "USD"
            )

        except Exception:

            original_currency = "USD"

        inr_rate = get_inr_rate(
            original_currency
        )

        # =====================================
        # DOWNLOAD HISTORY
        # =====================================

        df = stock.history(
            period=period,
            interval=interval,
            auto_adjust=False
        )

        if (
            df is None
            or df.empty
        ):

            return None

        df = df.copy()

        # =====================================
        # FLATTEN MULTIINDEX
        # =====================================

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        # =====================================
        # CONVERT HISTORICAL PRICES TO INR
        # =====================================

        if inr_rate is not None:

            price_columns = [

                "Open",

                "High",

                "Low",

                "Close",

                "Adj Close",

                "Dividends",

                "Capital Gains",
            ]

            for column in price_columns:

                if column in df.columns:

                    df[column] = (
                        pd.to_numeric(
                            df[column],
                            errors="coerce"
                        )
                        * inr_rate
                    )

        # =====================================
        # RESET DATE INDEX
        # =====================================

        df = df.reset_index()

        if "Datetime" in df.columns:

            df = df.rename(
                columns={
                    "Datetime":
                        "Date"
                }
            )

        if (
            "Date" not in df.columns
            and
            "index" in df.columns
        ):

            df = df.rename(
                columns={
                    "index":
                        "Date"
                }
            )

        # =====================================
        # SAVE CURRENCY METADATA
        # =====================================

        df.attrs[
            "currency"
        ] = (
            "INR"
            if inr_rate is not None
            else original_currency
        )

        df.attrs[
            "currency_symbol"
        ] = (
            "₹"
            if inr_rate is not None
            else original_currency
        )

        df.attrs[
            "original_currency"
        ] = original_currency

        df.attrs[
            "inr_rate"
        ] = inr_rate

        return df

    except Exception as e:

        print(
            "Stock history error:",
            e
        )

        return None