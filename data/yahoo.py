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
# YAHOO FALLBACK HELPERS
# =========================================================

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


def _infer_currency_from_symbol(symbol):

    symbol = str(symbol or "").upper()

    suffix_map = {
        ".NS": "INR",
        ".BO": "INR",
        ".L": "GBP",
        ".TO": "CAD",
        ".V": "CAD",
        ".AX": "AUD",
        ".NZ": "NZD",
        ".HK": "HKD",
        ".SI": "SGD",
        ".T": "JPY",
        ".KS": "KRW",
        ".KQ": "KRW",
        ".SS": "CNY",
        ".SZ": "CNY",
        ".DE": "EUR",
        ".F": "EUR",
        ".PA": "EUR",
        ".AS": "EUR",
        ".BR": "EUR",
        ".MI": "EUR",
        ".MC": "EUR",
        ".SW": "CHF",
        ".ST": "SEK",
        ".OL": "NOK",
        ".CO": "DKK",
        ".JO": "ZAR",
        ".SA": "BRL",
        ".MX": "MXN",
    }

    for suffix, currency in suffix_map.items():

        if symbol.endswith(suffix):
            return currency

    return "USD"


def _safe_company_metadata(symbol):

    metadata = {
        "name": symbol,
        "sector": "N/A",
        "industry": "N/A",
        "country": "N/A",
        "website": "N/A",
        "summary": "N/A",
    }

    try:
        search = yf.Search(
            symbol,
            max_results=6,
            news_count=0,
        )

        quotes = getattr(
            search,
            "quotes",
            []
        ) or []

        for quote in quotes:

            if not isinstance(quote, dict):
                continue

            if str(
                quote.get("symbol", "")
            ).upper() != symbol:
                continue

            metadata["name"] = (
                quote.get("longname")
                or quote.get("shortname")
                or quote.get("displayName")
                or symbol
            )

            metadata["sector"] = (
                quote.get("sector")
                or quote.get("sectorDisp")
                or "N/A"
            )

            metadata["industry"] = (
                quote.get("industry")
                or quote.get("industryDisp")
                or "N/A"
            )

            break

    except Exception:
        pass

    return metadata


# =========================================================
# STOCK INFORMATION
# =========================================================

def get_stock_info(symbol):

    try:

        symbol = (
            str(symbol)
            .strip()
            .upper()
        )

        if not symbol:
            return None

        stock = yf.Ticker(
            symbol
        )

        # Price data is read from fast_info/history first.
        # Ticker.info can be rate-limited on cloud hosts and
        # must never make the whole dashboard fail.

        try:
            fast = stock.fast_info
        except Exception:
            fast = None

        history = None

        try:
            history = stock.history(
                period="5d",
                interval="1d",
                auto_adjust=False,
            )
        except Exception:
            history = None

        if (
            history is None
            or history.empty
        ):

            try:
                history = yf.download(
                    symbol,
                    period="5d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception:
                history = None

        latest_price = None
        history_previous_close = None
        history_open = None
        history_high = None
        history_low = None
        history_volume = None

        if (
            history is not None
            and not history.empty
        ):

            history = history.copy()

            if isinstance(
                history.columns,
                pd.MultiIndex
            ):
                history.columns = (
                    history.columns
                    .get_level_values(0)
                )

            try:
                close = pd.to_numeric(
                    history["Close"],
                    errors="coerce"
                ).dropna()

                if not close.empty:
                    latest_price = float(
                        close.iloc[-1]
                    )

                    if len(close) >= 2:
                        history_previous_close = float(
                            close.iloc[-2]
                        )
            except Exception:
                pass

            try:
                history_open = _safe_number(
                    history["Open"].iloc[-1]
                )
            except Exception:
                pass

            try:
                history_high = _safe_number(
                    history["High"].iloc[-1]
                )
            except Exception:
                pass

            try:
                history_low = _safe_number(
                    history["Low"].iloc[-1]
                )
            except Exception:
                pass

            try:
                history_volume = _safe_number(
                    history["Volume"].iloc[-1]
                )
            except Exception:
                pass

        # Full info is optional. Fundamentals/company metadata
        # may be missing if Yahoo rate-limits this endpoint.

        try:
            info = stock.get_info() or {}
        except Exception:
            try:
                info = stock.info or {}
            except Exception:
                info = {}

        search_metadata = None

        if not info:
            search_metadata = _safe_company_metadata(
                symbol
            )

        original_currency = (
            _fast_value(
                fast,
                "currency",
            )
            or info.get("currency")
            or _infer_currency_from_symbol(
                symbol
            )
        )

        inr_rate = get_inr_rate(
            original_currency
        )

        original_price = (
            _fast_value(
                fast,
                "last_price",
                "lastPrice",
            )
            or info.get("currentPrice")
            or info.get("regularMarketPrice")
            or latest_price
        )

        original_previous_close = (
            _fast_value(
                fast,
                "previous_close",
                "previousClose",
                "regular_market_previous_close",
                "regularMarketPreviousClose",
            )
            or info.get("previousClose")
            or history_previous_close
        )

        original_open = (
            _fast_value(
                fast,
                "open",
            )
            or info.get("open")
            or history_open
        )

        original_day_high = (
            _fast_value(
                fast,
                "day_high",
                "dayHigh",
            )
            or info.get("dayHigh")
            or history_high
        )

        original_day_low = (
            _fast_value(
                fast,
                "day_low",
                "dayLow",
            )
            or info.get("dayLow")
            or history_low
        )

        original_high_52 = (
            _fast_value(
                fast,
                "year_high",
                "yearHigh",
            )
            or info.get(
                "fiftyTwoWeekHigh"
            )
        )

        original_low_52 = (
            _fast_value(
                fast,
                "year_low",
                "yearLow",
            )
            or info.get(
                "fiftyTwoWeekLow"
            )
        )

        original_market_cap = (
            _fast_value(
                fast,
                "market_cap",
                "marketCap",
            )
            or info.get("marketCap")
        )

        original_eps = (
            info.get("trailingEps")
        )

        volume = (
            _fast_value(
                fast,
                "last_volume",
                "lastVolume",
            )
            or info.get("volume")
            or history_volume
        )

        if _safe_number(
            original_price
        ) is None:
            return None

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

        metadata = (
            search_metadata
            or {}
        )

        company_name = (
            info.get("longName")
            or info.get("shortName")
            or metadata.get("name")
            or symbol
        )

        sector = (
            info.get("sector")
            or metadata.get("sector")
            or "N/A"
        )

        industry = (
            info.get("industry")
            or metadata.get("industry")
            or "N/A"
        )

        country = (
            info.get("country")
            or metadata.get("country")
            or "N/A"
        )

        website = (
            info.get("website")
            or metadata.get("website")
            or "N/A"
        )

        summary = (
            info.get("longBusinessSummary")
            or metadata.get("summary")
            or "N/A"
        )

        return {

            "symbol": symbol,
            "name": company_name,

            "price": to_inr(original_price),
            "current_price": to_inr(original_price),
            "previous_close": to_inr(original_previous_close),
            "open": to_inr(original_open),
            "day_high": to_inr(original_day_high),
            "day_low": to_inr(original_day_low),
            "high_52": to_inr(original_high_52),
            "low_52": to_inr(original_low_52),
            "fifty_two_week_high": to_inr(original_high_52),
            "fifty_two_week_low": to_inr(original_low_52),
            "market_cap": to_inr(original_market_cap),
            "eps": to_inr(original_eps),

            "currency": display_currency,
            "currency_symbol": (
                "₹"
                if conversion_available
                else original_currency
            ),
            "inr_rate": inr_rate,
            "conversion_available": conversion_available,

            "original_currency": original_currency,
            "original_price": _safe_number(original_price),
            "original_previous_close": _safe_number(original_previous_close),
            "original_open": _safe_number(original_open),
            "original_day_high": _safe_number(original_day_high),
            "original_day_low": _safe_number(original_day_low),
            "original_high_52": _safe_number(original_high_52),
            "original_low_52": _safe_number(original_low_52),
            "original_market_cap": _safe_number(original_market_cap),
            "original_eps": _safe_number(original_eps),

            "pe_ratio": info.get("trailingPE"),
            "trailingPE": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "dividendYield": info.get("dividendYield"),
            "dividend": info.get("dividendYield"),
            "volume": _safe_number(volume),

            "sector": sector,
            "industry": industry,
            "country": country,
            "website": website,
            "summary": summary,
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
            str(symbol)
            .strip()
            .upper()
        )

        if not symbol:
            return None

        stock = yf.Ticker(
            symbol
        )

        original_currency = None

        try:
            fast = stock.fast_info

            original_currency = (
                _fast_value(
                    fast,
                    "currency",
                )
            )
        except Exception:
            pass

        if not original_currency:

            try:
                info = (
                    stock.get_info()
                    or {}
                )

                original_currency = (
                    info.get("currency")
                )
            except Exception:
                pass

        if not original_currency:
            original_currency = (
                _infer_currency_from_symbol(
                    symbol
                )
            )

        inr_rate = get_inr_rate(
            original_currency
        )

        df = None

        try:
            df = stock.history(
                period=period,
                interval=interval,
                auto_adjust=False,
            )
        except Exception:
            df = None

        if (
            df is None
            or df.empty
        ):

            try:
                df = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception:
                df = None

        if (
            df is None
            or df.empty
        ):
            return None

        df = df.copy()

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

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
