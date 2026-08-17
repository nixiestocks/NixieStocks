from __future__ import annotations

import html
from datetime import datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from components.hero import hero


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


# =========================================================
# SELF-CONTAINED HOME STYLES
# No dependency on assets/style.css for this page.
# =========================================================

def _inject_home_styles():

    st.markdown(
        """
        <style>
        :root {
            --t7-bg: #050914;
            --t7-panel: #0b1322;
            --t7-panel2: #0e1829;
            --t7-border: rgba(125, 151, 198, .20);
            --t7-border2: rgba(105, 139, 207, .32);
            --t7-text: #f7f9fc;
            --t7-muted: #8290aa;
            --t7-green: #32e58f;
            --t7-red: #ff5364;
            --t7-blue: #678fff;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background:
                radial-gradient(circle at 50% -8%, rgba(55,88,155,.16), transparent 30%),
                #050914 !important;
            color: var(--t7-text);
        }

        .block-container {
            max-width: 1460px !important;
            padding-top: 1.1rem !important;
            padding-bottom: 2rem !important;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .t7-hero {
            text-align: center;
            padding: 12px 0 14px;
        }

        .t7-brand {
            color: #ffffff;
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-size: 3.2rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -0.035em;
            text-rendering: geometricPrecision;
            -webkit-font-smoothing: antialiased;
            text-shadow: 0 4px 18px rgba(0,0,0,.20);
        }

        .t7-subtitle {
            color: #a3aec2;
            font-size: .82rem;
            letter-spacing: .22em;
            font-weight: 650;
            margin-top: 11px;
        }

        .t7-tagline {
            color: #687791;
            font-size: .82rem;
            margin-top: 7px;
        }

        .t7-search-note {
            text-align: center;
            color: #687791;
            font-size: .76rem;
            margin-top: 8px;
            margin-bottom: 8px;
        }

        .t7-section-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 25px 0 10px;
        }

        .t7-section {
            color: #f5f7fb;
            font-size: .82rem;
            font-weight: 750;
            letter-spacing: .07em;
        }

        .t7-live {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: #7f8da5;
            font-size: .72rem;
        }

        .t7-live-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
            background: var(--t7-green);
            box-shadow: 0 0 10px rgba(50,229,143,.7);
        }

        .t7-market-card {
            position: relative;
            overflow: hidden;
            min-height: 142px;
            padding: 17px 18px 12px;
            border-radius: 13px;
            border: 1px solid var(--t7-border);
            background:
                radial-gradient(circle at 85% 70%, rgba(103,143,255,.06), transparent 35%),
                linear-gradient(145deg, rgba(15,25,43,.98), rgba(7,13,24,.98));
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,.025),
                0 12px 32px rgba(0,0,0,.17);
        }

        .t7-market-card.pos {
            background:
                radial-gradient(circle at 82% 75%, rgba(50,229,143,.09), transparent 38%),
                linear-gradient(145deg, rgba(15,25,43,.98), rgba(7,13,24,.98));
        }

        .t7-market-card.neg {
            background:
                radial-gradient(circle at 82% 75%, rgba(255,83,100,.09), transparent 38%),
                linear-gradient(145deg, rgba(15,25,43,.98), rgba(7,13,24,.98));
        }

        .t7-market-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            position: relative;
            z-index: 2;
        }

        .t7-market-name {
            color: #dce4f2;
            font-size: .82rem;
            font-weight: 700;
        }

        .t7-market-price {
            color: #fff;
            font-size: 1.63rem;
            line-height: 1.15;
            font-weight: 700;
            letter-spacing: -.025em;
            margin-top: 5px;
        }

        .t7-change {
            margin-top: 5px;
            font-size: .78rem;
            font-weight: 700;
        }

        .t7-change.pos { color: var(--t7-green); }
        .t7-change.neg { color: var(--t7-red); }

        .t7-index {
            min-width: 42px;
            height: 42px;
            padding: 0 7px;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 999px;
            border: 1px solid rgba(130,150,190,.16);
            background: rgba(255,255,255,.025);
            color: #8491a8;
            font-size: .58rem;
            font-weight: 800;
            letter-spacing: .05em;
        }

        .t7-spark {
            position: absolute;
            left: 0;
            bottom: 5px;
            width: 100%;
            height: 37px;
            opacity: .92;
        }

        .t7-mover-title {
            color: #f2f5fb;
            font-size: .82rem;
            font-weight: 750;
            letter-spacing: .07em;
            margin: 21px 0 9px;
        }

        .t7-mover-link {
            display: block;
            color: inherit !important;
            text-decoration: none !important;
            border-radius: 11px;
        }

        .t7-mover-link:visited {
            color: inherit !important;
        }

        .t7-mover-link:hover {
            color: inherit !important;
            text-decoration: none !important;
        }

        .t7-mover {
            min-height: 68px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: 11px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: linear-gradient(145deg, rgba(12,21,37,.96), rgba(8,14,25,.96));
            border: 1px solid var(--t7-border);
        }

        .t7-mover.gain {
            border-color: rgba(50,229,143,.24);
            box-shadow: inset 3px 0 0 rgba(50,229,143,.60);
        }

        .t7-mover.loss {
            border-color: rgba(255,83,100,.24);
            box-shadow: inset 3px 0 0 rgba(255,83,100,.60);
        }

        .t7-mover-link .t7-mover {
            cursor: pointer;
            transition: transform .16s ease, border-color .16s ease, background .16s ease, box-shadow .16s ease;
        }

        .t7-mover-link:hover .t7-mover {
            transform: translateY(-2px);
            background: linear-gradient(145deg, rgba(17,29,50,.99), rgba(9,16,29,.99));
            border-color: rgba(117,151,224,.46);
            box-shadow: 0 12px 28px rgba(0,0,0,.22);
        }

        .t7-mover-link:hover .t7-mover.gain {
            box-shadow: inset 3px 0 0 rgba(50,229,143,.85), 0 12px 28px rgba(0,0,0,.22);
        }

        .t7-mover-link:hover .t7-mover.loss {
            box-shadow: inset 3px 0 0 rgba(255,83,100,.85), 0 12px 28px rgba(0,0,0,.22);
        }

        .t7-mover-left {
            display: flex;
            align-items: center;
            gap: 11px;
            min-width: 0;
        }

        .t7-avatar {
            width: 38px;
            height: 38px;
            flex: 0 0 38px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #eef3fb;
            font-size: .79rem;
            font-weight: 800;
            border: 1px solid rgba(130,150,190,.18);
            background: #111d31;
        }

        .t7-company {
            color: #f5f7fb;
            font-size: .85rem;
            font-weight: 650;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .t7-meta {
            color: #65738c;
            font-size: .69rem;
            margin-top: 2px;
        }

        .t7-mover-right {
            text-align: right;
            flex: 0 0 auto;
        }

        .t7-mover-price {
            color: #f5f7fb;
            font-size: .82rem;
            font-weight: 700;
        }

        .t7-mover-change {
            font-size: .71rem;
            font-weight: 750;
            margin-top: 3px;
        }

        .t7-mover-change.gain { color: var(--t7-green); }
        .t7-mover-change.loss { color: var(--t7-red); }

        .t7-news-title {
            color: #f2f5fb;
            font-size: .82rem;
            font-weight: 750;
            letter-spacing: .07em;
            margin: 24px 0 9px;
        }

        .t7-news-card {
            min-height: 142px;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid var(--t7-border);
            background: linear-gradient(145deg, rgba(12,21,37,.96), rgba(8,14,25,.96));
        }

        .t7-news-source {
            color: #6e7f9b;
            font-size: .68rem;
            text-transform: uppercase;
            letter-spacing: .05em;
            margin-bottom: 8px;
        }

        .t7-news-headline {
            color: #f5f7fb;
            font-size: .92rem;
            line-height: 1.45;
            font-weight: 650;
        }

        .t7-news-time {
            color: #697892;
            font-size: .69rem;
            margin-top: 11px;
        }

        .t7-footer {
            border-top: 1px solid rgba(125,151,198,.12);
            margin-top: 30px;
            padding: 18px 0 3px;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            color: #5f6d84;
            font-size: .69rem;
        }

        .t7-dot-sep {
            width: 3px;
            height: 3px;
            border-radius: 50%;
            background: #49566c;
        }

        div[data-testid="stLinkButton"] a {
            border-radius: 9px !important;
            border: 1px solid rgba(125,151,198,.18) !important;
            background: rgba(13,22,38,.86) !important;
            min-height: 37px !important;
        }


        /* =====================================================
           LIVE STOCK TICKER
           Pure CSS animation — no extra reruns or API calls.
        ===================================================== */

        .t7-ticker-shell {
            width: 100%;
            overflow: hidden;
            border-top: 1px solid rgba(125,151,198,.14);
            border-bottom: 1px solid rgba(125,151,198,.14);
            background:
                linear-gradient(90deg, rgba(8,14,25,.98), rgba(12,21,37,.98), rgba(8,14,25,.98));
            box-shadow: 0 8px 28px rgba(0,0,0,.12);
            margin: 0 0 10px;
            position: relative;
        }

        .t7-ticker-shell::before,
        .t7-ticker-shell::after {
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            width: 72px;
            z-index: 3;
            pointer-events: none;
        }

        .t7-ticker-shell::before {
            left: 0;
            background: linear-gradient(90deg, #050914 12%, rgba(5,9,20,0));
        }

        .t7-ticker-shell::after {
            right: 0;
            background: linear-gradient(270deg, #050914 12%, rgba(5,9,20,0));
        }

        .t7-ticker-track {
            display: flex;
            align-items: center;
            width: max-content;
            min-width: 200%;
            animation: t7TickerScroll 48s linear infinite;
            will-change: transform;
        }

        .t7-ticker-shell:hover .t7-ticker-track {
            animation-play-state: paused;
        }

        .t7-ticker-group {
            display: flex;
            align-items: center;
            flex-shrink: 0;
        }

        .t7-ticker-label {
            display: flex;
            align-items: center;
            gap: 7px;
            padding: 0 18px;
            height: 42px;
            color: #8391aa;
            font-size: .67rem;
            font-weight: 750;
            letter-spacing: .08em;
            white-space: nowrap;
            border-right: 1px solid rgba(125,151,198,.12);
        }

        .t7-ticker-live-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #32e58f;
            box-shadow: 0 0 9px rgba(50,229,143,.75);
        }

        .t7-ticker-item {
            height: 42px;
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 0 18px;
            white-space: nowrap;
            text-decoration: none !important;
            border-right: 1px solid rgba(125,151,198,.10);
            transition: background .15s ease;
        }

        .t7-ticker-item:hover {
            background: rgba(103,143,255,.08);
            text-decoration: none !important;
        }

        .t7-ticker-symbol {
            color: #e9eef8;
            font-size: .72rem;
            font-weight: 800;
        }

        .t7-ticker-price {
            color: #aeb9cc;
            font-size: .71rem;
            font-weight: 650;
        }

        .t7-ticker-change {
            font-size: .69rem;
            font-weight: 800;
        }

        .t7-ticker-change.pos { color: #32e58f; }
        .t7-ticker-change.neg { color: #ff5364; }

        @keyframes t7TickerScroll {
            from { transform: translateX(0); }
            to { transform: translateX(-50%); }
        }

        @media (prefers-reduced-motion: reduce) {
            .t7-ticker-track {
                animation: none;
                transform: none;
            }
        }

        @media (max-width: 900px) {
            .t7-brand { font-size: 2.6rem; }
            .t7-tagline { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS
# =========================================================

def _close_series(data, ticker):

    if data is None or data.empty:
        return pd.Series(dtype=float)

    try:
        if isinstance(data.columns, pd.MultiIndex):

            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:
                frame = data[ticker]
                if "Close" in frame.columns:
                    return pd.to_numeric(frame["Close"], errors="coerce").dropna()

            if "Close" in level0 and ticker in level1:
                series = data["Close"][ticker]
                return pd.to_numeric(series, errors="coerce").dropna()

        if "Close" in data.columns:
            return pd.to_numeric(data["Close"], errors="coerce").dropna()

    except Exception:
        pass

    return pd.Series(dtype=float)


def _sparkline_svg(values, positive):

    values = [float(v) for v in values if pd.notna(v)]

    if len(values) < 2:
        return ""

    low = min(values)
    high = max(values)
    spread = max(high - low, 1e-9)

    width = 280
    height = 40
    points = []

    for i, value in enumerate(values):
        x = i / max(len(values) - 1, 1) * width
        y = height - (((value - low) / spread) * 31) - 4
        points.append(f"{x:.1f},{y:.1f}")

    stroke = "#32e58f" if positive else "#ff5364"

    return (
        f'<svg class="t7-spark" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" aria-hidden="true">'
        f'<polyline fill="none" stroke="{stroke}" stroke-width="2" '
        f'points="{" ".join(points)}" /></svg>'
    )


# =========================================================
# ONE BATCH DOWNLOAD FOR HOME PAGE
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def _load_home_market_data():

    stock_tickers = [item[1] for item in STOCK_UNIVERSE]
    all_tickers = list(MARKETS.values()) + stock_tickers + [USDINR]

    try:
        data = yf.download(
            tickers=all_tickers,
            period="5d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        return [], [], [], []

    usd_inr_close = _close_series(data, USDINR)
    usd_inr = float(usd_inr_close.iloc[-1]) if not usd_inr_close.empty else 1.0

    markets = []

    for name, ticker in MARKETS.items():

        close = _close_series(data, ticker)

        if len(close) < 2:
            markets.append(
                {
                    "name": name,
                    "price": None,
                    "change": None,
                    "sparkline": [],
                }
            )
            continue

        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2])

        change = ((latest - previous) / previous) * 100 if previous else 0.0

        markets.append(
            {
                "name": name,
                "price": latest,
                "change": change,
                "sparkline": close.tolist(),
            }
        )

    movers = []

    for company, ticker, exchange, currency in STOCK_UNIVERSE:

        close = _close_series(data, ticker)

        if len(close) < 2:
            continue

        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2])

        change = ((latest - previous) / previous) * 100 if previous else 0.0

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

    movers.sort(key=lambda x: x["change"], reverse=True)

    gainers = movers[:4]
    losers = sorted(movers, key=lambda x: x["change"])[:4]

    return markets, gainers, losers, movers


# =========================================================
# CACHED NEWS
# =========================================================

def _clean_news_item(item):

    if not isinstance(item, dict):
        return None

    content = item.get("content", item)

    if not isinstance(content, dict):
        return None

    title = content.get("title") or item.get("title") or ""

    provider = content.get("provider") or {}
    source = ""

    if isinstance(provider, dict):
        source = provider.get("displayName") or provider.get("name") or ""

    if not source:
        source = content.get("publisher") or item.get("publisher") or "Yahoo Finance"

    published = (
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
    )

    published_text = ""

    if isinstance(published, (int, float)):
        try:
            published_text = datetime.fromtimestamp(published).strftime("%d %b %Y")
        except Exception:
            pass
    elif published:
        published_text = str(published).replace("T", " ").replace("Z", "")[:16]

    url = ""

    for key in ("clickThroughUrl", "canonicalUrl"):
        value = content.get(key)
        if isinstance(value, dict) and value.get("url"):
            url = value["url"]
            break

    if not url:
        url = content.get("link") or item.get("link") or ""

    if not title:
        return None

    return {
        "title": str(title),
        "source": str(source),
        "published": published_text,
        "url": str(url),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def _load_news():

    try:
        try:
            search = yf.Search(
                "global stock market",
                max_results=0,
                news_count=6,
                lists_count=0,
                include_cb=True,
                include_nav_links=False,
                include_research=False,
                timeout=6,
                raise_errors=False,
            )
        except TypeError:
            search = yf.Search(
                "global stock market",
                max_results=0,
                news_count=6,
            )

        raw = getattr(search, "news", []) or []

        cleaned = []

        for item in raw:
            article = _clean_news_item(item)

            if article:
                cleaned.append(article)

            if len(cleaned) == 3:
                break

        return cleaned

    except Exception:
        return []


# =========================================================
# RENDERING
# =========================================================

def _ticker_tape(movers):

    if not movers:
        return

    item_html = []

    for row in movers:

        symbol = html.escape(
            str(row["symbol"])
        )

        price = float(
            row["price"]
        )

        change = float(
            row["change"]
        )

        positive = change >= 0

        change_class = (
            "pos"
            if positive
            else "neg"
        )

        arrow = (
            "▲"
            if positive
            else "▼"
        )

        stock_link = quote(
            str(row["symbol"]),
            safe=""
        )

        # Keep raw HTML at column 0.
        # Indented multiline HTML can be interpreted by Markdown
        # as a code block.
        item_html.append(
            f'<a class="t7-ticker-item" href="?stock={stock_link}" target="_self">'
            f'<span class="t7-ticker-symbol">{symbol}</span>'
            f'<span class="t7-ticker-price">₹{price:,.2f}</span>'
            f'<span class="t7-ticker-change {change_class}">'
            f'{arrow} {change:+.2f}%'
            f'</span>'
            f'</a>'
        )

    ticker_items = "".join(
        item_html
    )

    group = (
        '<div class="t7-ticker-group">'
        '<div class="t7-ticker-label">'
        '<span class="t7-ticker-live-dot"></span>'
        'LIVE STOCKS'
        '</div>'
        f'{ticker_items}'
        '</div>'
    )

    ticker_html = (
        '<div class="t7-ticker-shell">'
        '<div class="t7-ticker-track">'
        f'{group}'
        f'{group}'
        '</div>'
        '</div>'
    )

    st.markdown(
        ticker_html,
        unsafe_allow_html=True,
    )
def _market_section(markets):

    st.markdown(
        """
        <div class="t7-section-row">
            <div class="t7-section">MARKET OVERVIEW</div>
            <div class="t7-live">
                <span class="t7-live-dot"></span>
                Market data
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4, gap="medium")

    for col, market in zip(cols, markets):

        with col:

            price = market["price"]
            change = market["change"]

            if price is None or change is None:
                price_text = "N/A"
                change_text = "Unavailable"
                positive = False
            else:
                price_text = f"{price:,.2f}"
                positive = change >= 0
                arrow = "▲" if positive else "▼"
                change_text = f"{arrow} {change:+.2f}%"

            cls = "pos" if positive else "neg"
            sparkline = _sparkline_svg(market["sparkline"], positive)

            st.markdown(
                f"""
                <div class="t7-market-card {cls}">
                    <div class="t7-market-top">
                        <div>
                            <div class="t7-market-name">{html.escape(market["name"])}</div>
                            <div class="t7-market-price">{price_text}</div>
                            <div class="t7-change {cls}">{change_text}</div>
                        </div>
                        <div class="t7-index">IDX</div>
                    </div>
                    {sparkline}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _mover_panel(title, rows, kind):

    st.markdown(
        f'<div class="t7-mover-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

    if not rows:
        st.info("Mover data unavailable.")
        return

    for row in rows:

        initial = html.escape(row["company"][:1].upper())
        company = html.escape(row["company"])
        symbol = html.escape(row["symbol"])
        exchange = html.escape(row["exchange"])

        arrow = "▲" if row["change"] >= 0 else "▼"

        stock_link = quote(str(row["symbol"]), safe="")

        st.markdown(
            f"""
            <a class="t7-mover-link" href="?stock={stock_link}" target="_self">
                <div class="t7-mover {kind}">
                    <div class="t7-mover-left">
                        <div class="t7-avatar">{initial}</div>
                        <div>
                            <div class="t7-company">{company}</div>
                            <div class="t7-meta">{symbol} • {exchange}</div>
                        </div>
                    </div>
                    <div class="t7-mover-right">
                        <div class="t7-mover-price">₹{row["price"]:,.2f}</div>
                        <div class="t7-mover-change {kind}">
                            {arrow} {row["change"]:+.2f}%
                        </div>
                    </div>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )


def _movers_section(gainers, losers):

    left, right = st.columns(2, gap="large")

    with left:
        _mover_panel("TOP GAINERS", gainers, "gain")

    with right:
        _mover_panel("TOP LOSERS", losers, "loss")


def _news_section():

    st.markdown(
        '<div class="t7-news-title">LATEST MARKET NEWS</div>',
        unsafe_allow_html=True,
    )

    articles = _load_news()

    if not articles:
        st.info("Latest market news is temporarily unavailable.")
        return

    cols = st.columns(3, gap="medium")

    for col, article in zip(cols, articles):

        with col:

            st.markdown(
                f"""
                <div class="t7-news-card">
                    <div class="t7-news-source">
                        {html.escape(article["source"])}
                    </div>
                    <div class="t7-news-headline">
                        {html.escape(article["title"])}
                    </div>
                    <div class="t7-news-time">
                        {html.escape(article["published"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if article["url"]:
                st.link_button(
                    "Read article",
                    article["url"],
                    use_container_width=True,
                )


def _footer():

    st.markdown(
        """
        <div class="t7-footer">
            <span>Data: Yahoo Finance</span>
            <span class="t7-dot-sep"></span>
            <span>Stock values displayed in INR</span>
            <span class="t7-dot-sep"></span>
            <span>Educational analysis only</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _handle_mover_navigation():

    try:
        clicked_stock = st.query_params.get("stock")
    except Exception:
        clicked_stock = None

    if isinstance(clicked_stock, list):
        clicked_stock = clicked_stock[0] if clicked_stock else None

    if not clicked_stock:
        return

    clicked_stock = str(clicked_stock).strip().upper()

    if not clicked_stock:
        return

    st.session_state["stock"] = clicked_stock
    st.session_state["page"] = "dashboard"

    try:
        st.query_params.clear()
    except Exception:
        pass

    st.rerun()


def home_page():

    _handle_mover_navigation()

    _inject_home_styles()

    markets, gainers, losers, movers = _load_home_market_data()

    _ticker_tape(movers)

    hero()

    _market_section(markets)

    _movers_section(gainers, losers)

    _news_section()

    _footer()
