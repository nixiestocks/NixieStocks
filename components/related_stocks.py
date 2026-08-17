from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import yfinance as yf

from data.yahoo import convert_to_inr


# =========================================================
# FALLBACK PEER UNIVERSE
# Used only when Yahoo search does not return enough peers.
# =========================================================

INDIA_SECTOR_PEERS = {
    "Technology": [
        ("Tata Consultancy Services", "TCS.NS"),
        ("Infosys", "INFY.NS"),
        ("HCL Technologies", "HCLTECH.NS"),
        ("Wipro", "WIPRO.NS"),
        ("Tech Mahindra", "TECHM.NS"),
        ("LTIMindtree", "LTIM.NS"),
    ],
    "Financial Services": [
        ("HDFC Bank", "HDFCBANK.NS"),
        ("ICICI Bank", "ICICIBANK.NS"),
        ("State Bank of India", "SBIN.NS"),
        ("Axis Bank", "AXISBANK.NS"),
        ("Kotak Mahindra Bank", "KOTAKBANK.NS"),
        ("Bajaj Finance", "BAJFINANCE.NS"),
    ],
    "Consumer Cyclical": [
        ("Maruti Suzuki", "MARUTI.NS"),
        ("Mahindra & Mahindra", "M&M.NS"),
        ("Tata Motors", "TATAMOTORS.NS"),
        ("Titan Company", "TITAN.NS"),
        ("Eicher Motors", "EICHERMOT.NS"),
        ("Trent", "TRENT.NS"),
    ],
    "Consumer Defensive": [
        ("Hindustan Unilever", "HINDUNILVR.NS"),
        ("ITC", "ITC.NS"),
        ("Nestle India", "NESTLEIND.NS"),
        ("Britannia Industries", "BRITANNIA.NS"),
        ("Tata Consumer Products", "TATACONSUM.NS"),
        ("Dabur India", "DABUR.NS"),
    ],
    "Energy": [
        ("Reliance Industries", "RELIANCE.NS"),
        ("ONGC", "ONGC.NS"),
        ("Bharat Petroleum", "BPCL.NS"),
        ("Indian Oil", "IOC.NS"),
        ("GAIL India", "GAIL.NS"),
        ("Oil India", "OIL.NS"),
    ],
    "Healthcare": [
        ("Sun Pharmaceutical", "SUNPHARMA.NS"),
        ("Dr. Reddy's Laboratories", "DRREDDY.NS"),
        ("Cipla", "CIPLA.NS"),
        ("Divi's Laboratories", "DIVISLAB.NS"),
        ("Apollo Hospitals", "APOLLOHOSP.NS"),
        ("Lupin", "LUPIN.NS"),
    ],
    "Industrials": [
        ("Larsen & Toubro", "LT.NS"),
        ("Siemens India", "SIEMENS.NS"),
        ("ABB India", "ABB.NS"),
        ("Bharat Electronics", "BEL.NS"),
        ("Hindustan Aeronautics", "HAL.NS"),
        ("Cummins India", "CUMMINSIND.NS"),
    ],
    "Basic Materials": [
        ("Tata Steel", "TATASTEEL.NS"),
        ("Hindalco Industries", "HINDALCO.NS"),
        ("JSW Steel", "JSWSTEEL.NS"),
        ("UltraTech Cement", "ULTRACEMCO.NS"),
        ("Grasim Industries", "GRASIM.NS"),
        ("Vedanta", "VEDL.NS"),
    ],
    "Communication Services": [
        ("Bharti Airtel", "BHARTIARTL.NS"),
        ("Vodafone Idea", "IDEA.NS"),
        ("Zee Entertainment", "ZEEL.NS"),
        ("Sun TV Network", "SUNTV.NS"),
    ],
    "Utilities": [
        ("NTPC", "NTPC.NS"),
        ("Power Grid Corporation", "POWERGRID.NS"),
        ("Tata Power", "TATAPOWER.NS"),
        ("Adani Power", "ADANIPOWER.NS"),
        ("NHPC", "NHPC.NS"),
    ],
    "Real Estate": [
        ("DLF", "DLF.NS"),
        ("Godrej Properties", "GODREJPROP.NS"),
        ("Prestige Estates", "PRESTIGE.NS"),
        ("Oberoi Realty", "OBEROIRLTY.NS"),
        ("Phoenix Mills", "PHOENIXLTD.NS"),
    ],
}


GLOBAL_SECTOR_PEERS = {
    "Technology": [
        ("Apple", "AAPL"),
        ("Microsoft", "MSFT"),
        ("NVIDIA", "NVDA"),
        ("Oracle", "ORCL"),
        ("Adobe", "ADBE"),
        ("Salesforce", "CRM"),
    ],
    "Financial Services": [
        ("JPMorgan Chase", "JPM"),
        ("Bank of America", "BAC"),
        ("Goldman Sachs", "GS"),
        ("Morgan Stanley", "MS"),
        ("Wells Fargo", "WFC"),
        ("Citigroup", "C"),
    ],
    "Consumer Cyclical": [
        ("Amazon", "AMZN"),
        ("Tesla", "TSLA"),
        ("Home Depot", "HD"),
        ("Nike", "NKE"),
        ("McDonald's", "MCD"),
        ("Booking Holdings", "BKNG"),
    ],
    "Consumer Defensive": [
        ("Walmart", "WMT"),
        ("Costco", "COST"),
        ("Procter & Gamble", "PG"),
        ("Coca-Cola", "KO"),
        ("PepsiCo", "PEP"),
        ("Philip Morris", "PM"),
    ],
    "Energy": [
        ("Exxon Mobil", "XOM"),
        ("Chevron", "CVX"),
        ("ConocoPhillips", "COP"),
        ("Schlumberger", "SLB"),
        ("EOG Resources", "EOG"),
        ("Marathon Petroleum", "MPC"),
    ],
    "Healthcare": [
        ("Eli Lilly", "LLY"),
        ("Johnson & Johnson", "JNJ"),
        ("UnitedHealth Group", "UNH"),
        ("AbbVie", "ABBV"),
        ("Merck", "MRK"),
        ("Pfizer", "PFE"),
    ],
    "Industrials": [
        ("GE Aerospace", "GE"),
        ("Caterpillar", "CAT"),
        ("RTX", "RTX"),
        ("Honeywell", "HON"),
        ("Union Pacific", "UNP"),
        ("Deere", "DE"),
    ],
    "Basic Materials": [
        ("Linde", "LIN"),
        ("Freeport-McMoRan", "FCX"),
        ("Newmont", "NEM"),
        ("Dow", "DOW"),
        ("Nucor", "NUE"),
        ("Sherwin-Williams", "SHW"),
    ],
    "Communication Services": [
        ("Alphabet", "GOOGL"),
        ("Meta Platforms", "META"),
        ("Netflix", "NFLX"),
        ("Walt Disney", "DIS"),
        ("T-Mobile US", "TMUS"),
        ("Verizon", "VZ"),
    ],
    "Utilities": [
        ("NextEra Energy", "NEE"),
        ("Southern Company", "SO"),
        ("Duke Energy", "DUK"),
        ("Constellation Energy", "CEG"),
        ("American Electric Power", "AEP"),
    ],
    "Real Estate": [
        ("Prologis", "PLD"),
        ("American Tower", "AMT"),
        ("Equinix", "EQIX"),
        ("Realty Income", "O"),
        ("Welltower", "WELL"),
    ],
}


# =========================================================
# HELPERS
# =========================================================

def _clean(value):
    if value is None:
        return ""
    return str(value).strip()


def _normalize(value):
    return _clean(value).lower()


def _fast_get(fast_info, key):
    try:
        value = fast_info.get(key)
        if value is not None:
            return value
    except Exception:
        pass

    try:
        return getattr(fast_info, key)
    except Exception:
        return None


def _format_market_cap(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except Exception:
        return "N/A"

    crore = value / 10_000_000

    if crore >= 100_000:
        return f"₹{crore / 100_000:.2f}L Cr"

    if crore >= 1_000:
        return f"₹{crore / 1_000:.2f}K Cr"

    return f"₹{crore:,.0f} Cr"


# =========================================================
# DISCOVER PEERS
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _discover_peer_candidates(
    symbol,
    sector,
    industry,
    country,
):
    symbol = _clean(symbol).upper()
    sector = _clean(sector)
    industry = _clean(industry)
    country = _clean(country)

    results = []
    seen = {symbol}

    # =====================================================
    # FAST PATH
    #
    # For known sectors, use our sector-matched peer universe
    # first. This avoids a slow Yahoo Search request on most
    # dashboard loads while still keeping peers sector-based.
    # =====================================================

    fallback_map = (
        INDIA_SECTOR_PEERS
        if country.lower() == "india"
        or symbol.endswith(".NS")
        or symbol.endswith(".BO")
        else GLOBAL_SECTOR_PEERS
    )

    fallback = fallback_map.get(
        sector,
        []
    )

    for peer_name, peer_symbol in fallback:
        peer_symbol = peer_symbol.upper()

        if peer_symbol in seen:
            continue

        seen.add(
            peer_symbol
        )

        results.append(
            (
                peer_name,
                peer_symbol,
            )
        )

        # Related Stocks displays five rows.
        # Once we have enough valid sector peers there is no
        # reason to make an additional Yahoo Search request.
        if len(results) >= 6:
            return results

    queries = []

    if industry and industry.upper() != "N/A":
        queries.append(industry)

    if sector and sector.upper() != "N/A":
        queries.append(sector)

    for query in queries[:2]:
        try:
            try:
                search = yf.Search(
                    query=query,
                    max_results=25,
                    news_count=0,
                    lists_count=0,
                    include_cb=True,
                    include_nav_links=False,
                    include_research=False,
                    enable_fuzzy_query=True,
                    recommended=20,
                    timeout=7,
                    raise_errors=False,
                )
            except TypeError:
                search = yf.Search(
                    query,
                    max_results=25,
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

                quote_type = _clean(
                    quote.get("quoteType")
                    or quote.get("typeDisp")
                ).upper()

                if quote_type not in {
                    "EQUITY",
                    "STOCK",
                }:
                    continue

                peer_symbol = _clean(
                    quote.get("symbol")
                ).upper()

                if (
                    not peer_symbol
                    or peer_symbol in seen
                ):
                    continue

                quote_sector = _clean(
                    quote.get("sector")
                    or quote.get("sectorDisp")
                )

                quote_industry = _clean(
                    quote.get("industry")
                    or quote.get("industryDisp")
                )

                # Prefer explicit sector / industry matches.
                metadata_matches = (
                    (
                        sector
                        and quote_sector
                        and _normalize(sector) == _normalize(quote_sector)
                    )
                    or
                    (
                        industry
                        and quote_industry
                        and _normalize(industry) == _normalize(quote_industry)
                    )
                )

                # Yahoo Search does not always include sector metadata.
                # Query relevance is used as a secondary fallback.
                if not metadata_matches and (
                    quote_sector
                    or quote_industry
                ):
                    continue

                peer_name = _clean(
                    quote.get("longname")
                    or quote.get("shortname")
                    or quote.get("displayName")
                    or peer_symbol
                )

                # Prefer the same market when the selected stock is Indian.
                if country.lower() == "india":
                    if not (
                        peer_symbol.endswith(".NS")
                        or peer_symbol.endswith(".BO")
                    ):
                        continue

                seen.add(
                    peer_symbol
                )

                results.append(
                    (
                        peer_name,
                        peer_symbol,
                    )
                )

                if len(results) >= 10:
                    break

        except Exception:
            continue

        if len(results) >= 10:
            break

    # =====================================================
    # FALLBACK PEERS
    # =====================================================

    fallback_map = (
        INDIA_SECTOR_PEERS
        if country.lower() == "india"
        or symbol.endswith(".NS")
        or symbol.endswith(".BO")
        else GLOBAL_SECTOR_PEERS
    )

    fallback = fallback_map.get(
        sector,
        []
    )

    for peer_name, peer_symbol in fallback:
        if peer_symbol.upper() in seen:
            continue

        seen.add(
            peer_symbol.upper()
        )

        results.append(
            (
                peer_name,
                peer_symbol.upper(),
            )
        )

        if len(results) >= 10:
            break

    return results


# =========================================================
# PEER MARKET DATA
# =========================================================

def _load_one_peer(
    name,
    symbol,
):
    try:
        ticker = yf.Ticker(
            symbol
        )

        fast = ticker.fast_info

        price = _fast_get(
            fast,
            "last_price"
        )

        if price is None:
            price = _fast_get(
                fast,
                "lastPrice"
            )

        previous = _fast_get(
            fast,
            "previous_close"
        )

        if previous is None:
            previous = _fast_get(
                fast,
                "previousClose"
            )

        market_cap = _fast_get(
            fast,
            "market_cap"
        )

        if market_cap is None:
            market_cap = _fast_get(
                fast,
                "marketCap"
            )

        currency = _fast_get(
            fast,
            "currency"
        )

        if not currency:
            if (
                symbol.endswith(".NS")
                or symbol.endswith(".BO")
            ):
                currency = "INR"
            else:
                currency = "USD"

        price_inr = convert_to_inr(
            price,
            currency
        )

        market_cap_inr = convert_to_inr(
            market_cap,
            currency
        )

        change = None

        try:
            price_float = float(
                price
            )

            previous_float = float(
                previous
            )

            if previous_float != 0:
                change = (
                    (
                        price_float
                        - previous_float
                    )
                    / previous_float
                ) * 100

        except Exception:
            change = None

        return {
            "name":
                name,

            "symbol":
                symbol,

            "price":
                price_inr,

            "change":
                change,

            "market_cap":
                market_cap_inr,
        }

    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_related_stocks(
    symbol,
    sector,
    industry,
    country,
    limit=5,
):
    candidates = _discover_peer_candidates(
        symbol,
        sector,
        industry,
        country,
    )

    if not candidates:
        return []

    rows = []

    with ThreadPoolExecutor(
        max_workers=5
    ) as executor:

        futures = [
            executor.submit(
                _load_one_peer,
                name,
                peer_symbol,
            )
            for name, peer_symbol
            in candidates[:7]
        ]

        for future in as_completed(
            futures
        ):
            try:
                row = future.result()
            except Exception:
                row = None

            if (
                row
                and row.get("price") is not None
            ):
                rows.append(
                    row
                )

            if len(rows) >= limit:
                break

    rows = rows[:limit]

    return rows


# =========================================================
# UI
# =========================================================

def related_stocks(
    info,
):
    symbol = _clean(
        info.get("symbol")
    ).upper()

    sector = _clean(
        info.get("sector")
    )

    industry = _clean(
        info.get("industry")
    )

    country = _clean(
        info.get("country")
    )

    st.markdown(
        f"""
        <div class="t7-related-head">
            <div>
                <div class="t7-panel-title">Related Stocks</div>
                <div class="t7-panel-sub">{sector or "Same sector"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = get_related_stocks(
        symbol,
        sector,
        industry,
        country,
        limit=5,
    )

    if not rows:
        st.caption(
            "Related stocks are temporarily unavailable."
        )
        return

    header = st.columns(
        [2.4, 1.1, 0.9, 1.2, 0.55]
    )

    labels = [
        "Stock",
        "Price",
        "Change",
        "Market Cap",
        "",
    ]

    for column, label in zip(
        header,
        labels
    ):
        with column:
            st.markdown(
                f'<div class="t7-related-col">{label}</div>',
                unsafe_allow_html=True,
            )

    for index, row in enumerate(
        rows
    ):
        cols = st.columns(
            [2.4, 1.1, 0.9, 1.2, 0.55]
        )

        with cols[0]:
            st.markdown(
                f"""
                <div class="t7-related-company">
                    <b>{row["symbol"]}</b>
                    <span>{row["name"]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with cols[1]:
            price = row.get(
                "price"
            )

            price_text = (
                f"₹{price:,.2f}"
                if price is not None
                else "N/A"
            )

            st.markdown(
                f'<div class="t7-related-value">{price_text}</div>',
                unsafe_allow_html=True,
            )

        with cols[2]:
            change = row.get(
                "change"
            )

            if change is None:
                change_text = "N/A"
                change_class = ""
            else:
                change_text = (
                    f"{change:+.2f}%"
                )

                change_class = (
                    "positive"
                    if change >= 0
                    else "negative"
                )

            st.markdown(
                f'<div class="t7-related-change {change_class}">{change_text}</div>',
                unsafe_allow_html=True,
            )

        with cols[3]:
            st.markdown(
                f'<div class="t7-related-value">{_format_market_cap(row.get("market_cap"))}</div>',
                unsafe_allow_html=True,
            )

        with cols[4]:
            if st.button(
                "↗",
                key=(
                    f"related_open_"
                    f"{index}_"
                    f"{row['symbol']}"
                ),
                use_container_width=True,
            ):
                st.session_state.stock = (
                    row["symbol"]
                )

                st.session_state.page = (
                    "dashboard"
                )

                st.session_state.pop(
                    "_dashboard_last_search_selection",
                    None
                )

                st.rerun()
