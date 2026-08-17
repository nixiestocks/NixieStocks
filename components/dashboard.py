from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from plotly.subplots import make_subplots
from streamlit_searchbox import st_searchbox

from data.yahoo import (
    get_stock_info,
    get_stock_history,
)
from data.company_search import search_company_options

from components.dashboard_ai import (
    analyze_stock_ai,
    analyze_stock_ai_from_history,
    build_recommendation,
)
from components.dashboard_company import dashboard_company
from components.stock_comparison import stock_comparison
from components.news import market_news
from components.related_stocks import related_stocks


# =========================================================
# SEARCH STYLE
# =========================================================

DASHBOARD_SEARCH_STYLE = {
    "dropdown": {
        "rotate": True,
        "width": 22,
        "height": 22,
        "fill": "#7f8da8",
    },
    "clear": {
        "width": 17,
        "height": 17,
        "icon": "cross",
        "clearable": "always",
    },
    "searchbox": {
        "control": {
            "backgroundColor": "#090f1a",
            "border": "1px solid #26344b",
            "borderRadius": "11px",
            "minHeight": "48px",
            "boxShadow": "none",
        },
        "menu": {
            "backgroundColor": "#080e18",
            "border": "1px solid #26344b",
            "borderRadius": "11px",
            "overflow": "hidden",
        },
        "menuList": {
            "backgroundColor": "#080e18",
        },
        "option": {
            "backgroundColor": "#080e18",
            "color": "#e8eef9",
            "highlightColor": "#3b82f6",
        },
        "placeholder": {
            "color": "#6f7f98",
        },
        "input": {
            "color": "#f8fafc",
        },
        "singleValue": {
            "color": "#f8fafc",
        },
    },
}


# =========================================================
# CSS
# =========================================================

def _inject_dashboard_styles():
    st.markdown(
        """
<style>
:root {
    --t7-bg: #05080d;
    --t7-panel: #090f18;
    --t7-panel-2: #0b121d;
    --t7-border: rgba(132, 153, 188, .24);
    --t7-text: #f4f7fb;
    --t7-muted: #78869d;
    --t7-green: #00df78;
    --t7-red: #ff4658;
    --t7-blue: #2f7fff;
    --t7-purple: #a855f7;
    --t7-gold: #f0b84b;
}

html,
body,
[data-testid="stAppViewContainer"],
.stApp {
    background:
        radial-gradient(circle at 57% -15%, rgba(32, 77, 142, .10), transparent 30%),
        #05080d !important;
    color: var(--t7-text);
}

.block-container {
    max-width: 1700px !important;
    padding-top: .65rem !important;
    padding-left: .8rem !important;
    padding-right: .8rem !important;
    padding-bottom: 2rem !important;
}

#MainMenu,
footer {
    visibility: hidden;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ================= HEADER ================= */

.t7-page-title {
    color: #fff;
    font-size: 1.65rem;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -.025em;
}

.t7-page-subtitle {
    color: #8a97aa;
    font-size: .76rem;
    margin-top: 5px;
}

.t7-market-mini {
    height: 56px;
    border-left: 1px solid rgba(132,153,188,.16);
    padding-left: 17px;
}

.t7-market-mini-label {
    color: #758399;
    font-size: .62rem;
}

.t7-market-mini-value {
    color: #f5f7fb;
    font-size: .82rem;
    font-weight: 750;
    margin-top: 2px;
}

.t7-market-mini-change {
    font-size: .68rem;
    font-weight: 750;
    margin-top: 2px;
}

.positive {
    color: var(--t7-green) !important;
}

.negative {
    color: var(--t7-red) !important;
}

.neutral {
    color: var(--t7-gold) !important;
}

/* ================= KPI CARDS ================= */

.t7-kpi {
    position: relative;
    min-height: 120px;
    border-radius: 13px;
    padding: 16px 17px;
    overflow: hidden;
    background:
        radial-gradient(circle at 90% 18%, rgba(47,127,255,.08), transparent 30%),
        linear-gradient(145deg, rgba(12,19,31,.98), rgba(7,12,20,.98));
    border: 1px solid var(--t7-border);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.02),
        0 12px 32px rgba(0,0,0,.15);
}

.t7-kpi.green {
    background:
        radial-gradient(circle at 88% 20%, rgba(0,223,120,.10), transparent 33%),
        linear-gradient(145deg, rgba(12,19,31,.98), rgba(7,12,20,.98));
}

.t7-kpi.purple {
    background:
        radial-gradient(circle at 88% 20%, rgba(168,85,247,.11), transparent 33%),
        linear-gradient(145deg, rgba(12,19,31,.98), rgba(7,12,20,.98));
}

.t7-kpi-label {
    color: #d8e0ec;
    font-size: .75rem;
    font-weight: 650;
}

.t7-kpi-value {
    color: #fff;
    font-size: 1.55rem;
    font-weight: 820;
    margin-top: 9px;
    line-height: 1;
}

.t7-kpi-sub {
    color: #7d8ba1;
    font-size: .68rem;
    margin-top: 9px;
}

.t7-kpi-sub strong {
    color: #a7b5ca;
}

/* ================= PANELS ================= */

.t7-panel-header {
    margin-bottom: 8px;
}

.t7-panel-title {
    color: #f6f8fb;
    font-size: .91rem;
    font-weight: 760;
}

.t7-panel-sub {
    color: #77869c;
    font-size: .67rem;
    margin-top: 2px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 13px !important;
    border-color: rgba(132,153,188,.22) !important;
    background:
        linear-gradient(145deg, rgba(10,16,26,.98), rgba(6,11,18,.98)) !important;
    box-shadow: 0 12px 30px rgba(0,0,0,.13);
}

/* ================= STOCK HEADER ================= */

.t7-stock-name {
    color: #f8fafc;
    font-size: 1.05rem;
    font-weight: 780;
}

.t7-stock-symbol {
    display: inline-block;
    margin-left: 7px;
    padding: 3px 7px;
    border-radius: 5px;
    background: rgba(255,255,255,.05);
    color: #8796ad;
    font-size: .58rem;
    vertical-align: middle;
}

.t7-stock-price {
    color: #fff;
    font-size: 1.65rem;
    font-weight: 820;
    margin-top: 5px;
}

.t7-stock-change {
    display: inline-block;
    margin-left: 8px;
    font-size: .72rem;
    font-weight: 780;
}

/* ================= AI SIDE PANEL ================= */

.t7-ai-callout {
    padding: 7px 0 2px;
}

.t7-ai-price {
    color: var(--t7-green);
    font-size: 2rem;
    font-weight: 850;
    line-height: 1.05;
    margin-top: 16px;
}

.t7-ai-price.sell {
    color: var(--t7-red);
}

.t7-ai-price.hold {
    color: var(--t7-gold);
}

.t7-ai-return {
    font-size: .85rem;
    font-weight: 800;
    margin-top: 8px;
}

.t7-ai-pill {
    display: inline-block;
    border-radius: 7px;
    padding: 5px 9px;
    font-size: .69rem;
    font-weight: 800;
    background: rgba(0,223,120,.10);
    color: var(--t7-green);
    border: 1px solid rgba(0,223,120,.16);
}

.t7-ai-pill.sell {
    background: rgba(255,70,88,.10);
    color: var(--t7-red);
    border-color: rgba(255,70,88,.16);
}

.t7-ai-pill.hold {
    background: rgba(240,184,75,.10);
    color: var(--t7-gold);
    border-color: rgba(240,184,75,.16);
}

.t7-ai-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    margin-top: 7px;
    border-top: 1px solid rgba(132,153,188,.14);
}

.t7-ai-stat {
    padding: 11px 8px 3px 0;
}

.t7-ai-stat + .t7-ai-stat {
    border-left: 1px solid rgba(132,153,188,.14);
    padding-left: 12px;
}

.t7-ai-stat-label {
    color: #69788f;
    font-size: .60rem;
}

.t7-ai-stat-value {
    color: #f4f7fb;
    font-size: .76rem;
    font-weight: 760;
    margin-top: 4px;
}

/* ================= TECHNICAL ================= */

.t7-tech-card {
    min-height: 100px;
    padding: 12px 13px;
    border-radius: 10px;
    background: rgba(255,255,255,.018);
    border: 1px solid rgba(132,153,188,.12);
}

.t7-tech-label {
    color: #7e8ba0;
    font-size: .62rem;
}

.t7-tech-value {
    color: #f3f6fa;
    font-size: .93rem;
    font-weight: 760;
    margin-top: 6px;
}

.t7-tech-status {
    font-size: .64rem;
    font-weight: 760;
    margin-top: 4px;
}

.t7-spark {
    margin-top: 8px;
    width: 100%;
    height: 21px;
}

/* ================= RELATED STOCKS ================= */

.t7-related-head {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
}

.t7-related-col {
    color: #68788f;
    font-size: .60rem;
    padding-bottom: 5px;
}

.t7-related-company {
    display: flex;
    flex-direction: column;
    min-height: 38px;
    justify-content: center;
}

.t7-related-company b {
    color: #eef3f9;
    font-size: .70rem;
}

.t7-related-company span {
    color: #6f7e94;
    font-size: .57rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.t7-related-value,
.t7-related-change {
    color: #dfe6f1;
    font-size: .67rem;
    padding-top: 10px;
    font-weight: 650;
}

/* ================= FOOTER SELECTOR ================= */

.t7-more-title {
    color: #f5f7fb;
    font-size: .82rem;
    font-weight: 760;
    letter-spacing: .05em;
    margin: 16px 0 7px;
}



/* ================= DAILY RETURNS ================= */

.t7-returns-shell {
    margin-top: 12px;
    border-radius: 13px;
    border: 1px solid rgba(132,153,188,.20);
    background:
        radial-gradient(circle at 88% 12%, rgba(47,127,255,.055), transparent 28%),
        linear-gradient(145deg, rgba(10,16,26,.98), rgba(6,11,18,.98));
    padding: 15px;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.018),
        0 12px 30px rgba(0,0,0,.13);
}

.t7-returns-title {
    color: #f5f7fb;
    font-size: .83rem;
    font-weight: 780;
}

.t7-returns-sub {
    color: #697891;
    font-size: .62rem;
    margin-top: 3px;
    margin-bottom: 11px;
}

/* ================= LOWER DASHBOARD ================= */

.t7-lower-stack {
    margin-top: 0;
}

.t7-snapshot-shell,
.t7-signal-shell,
.t7-analysis-shell {
    border-radius: 13px;
    border: 1px solid rgba(132,153,188,.20);
    background:
        radial-gradient(circle at 88% 12%, rgba(47,127,255,.055), transparent 28%),
        linear-gradient(145deg, rgba(10,16,26,.98), rgba(6,11,18,.98));
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.018),
        0 12px 30px rgba(0,0,0,.13);
}

.t7-snapshot-shell,
.t7-signal-shell {
    margin-top: 12px;
    padding: 15px;
}

.t7-lower-title {
    color: #f5f7fb;
    font-size: .83rem;
    font-weight: 780;
    letter-spacing: .02em;
}

.t7-lower-subtitle {
    color: #697891;
    font-size: .62rem;
    margin-top: 3px;
    margin-bottom: 11px;
}

.t7-snapshot-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
}

.t7-snapshot-item {
    min-height: 78px;
    padding: 11px 12px;
    border-radius: 10px;
    background: rgba(255,255,255,.018);
    border: 1px solid rgba(132,153,188,.12);
}

.t7-snapshot-label {
    color: #697991;
    font-size: .58rem;
    text-transform: uppercase;
    letter-spacing: .055em;
}

.t7-snapshot-value {
    color: #f6f8fb;
    font-size: .92rem;
    font-weight: 770;
    margin-top: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.t7-snapshot-note {
    color: #53627a;
    font-size: .56rem;
    margin-top: 3px;
}

.t7-signal-hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    padding: 12px 13px;
    border-radius: 11px;
    background: rgba(255,255,255,.018);
    border: 1px solid rgba(132,153,188,.12);
}

.t7-signal-hero.buy {
    border-color: rgba(0,223,120,.24);
    box-shadow: inset 3px 0 0 rgba(0,223,120,.72);
}

.t7-signal-hero.sell {
    border-color: rgba(255,70,88,.24);
    box-shadow: inset 3px 0 0 rgba(255,70,88,.72);
}

.t7-signal-hero.hold {
    border-color: rgba(240,184,75,.24);
    box-shadow: inset 3px 0 0 rgba(240,184,75,.72);
}

.t7-signal-kicker {
    color: #697991;
    font-size: .57rem;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.t7-signal-action {
    font-size: 1.45rem;
    font-weight: 850;
    line-height: 1;
    margin-top: 5px;
}

.t7-signal-confidence {
    text-align: right;
}

.t7-signal-confidence-label {
    color: #697991;
    font-size: .57rem;
}

.t7-signal-confidence-value {
    color: #f6f8fb;
    font-size: 1.12rem;
    font-weight: 800;
    margin-top: 4px;
}

.t7-signal-bar {
    width: 125px;
    height: 5px;
    background: rgba(255,255,255,.07);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 7px;
}

.t7-signal-fill {
    height: 100%;
    border-radius: inherit;
}

.t7-signal-fill.buy {
    background: #00df78;
}

.t7-signal-fill.sell {
    background: #ff4658;
}

.t7-signal-fill.hold {
    background: #f0b84b;
}

.t7-signal-counts {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 9px;
}

.t7-signal-count {
    min-height: 58px;
    border-radius: 9px;
    padding: 9px 10px;
    background: rgba(255,255,255,.015);
    border: 1px solid rgba(132,153,188,.10);
}

.t7-signal-count-label {
    color: #64738b;
    font-size: .56rem;
}

.t7-signal-count-value {
    color: #f3f6fa;
    font-size: .88rem;
    font-weight: 780;
    margin-top: 5px;
}

.t7-signal-note {
    margin-top: 9px;
    padding: 10px 11px;
    border-radius: 9px;
    color: #9fb0c7;
    font-size: .64rem;
    line-height: 1.45;
    background: rgba(47,127,255,.05);
    border: 1px solid rgba(47,127,255,.11);
}

.t7-analysis-shell {
    margin-top: 18px;
    padding: 17px;
}

.t7-analysis-head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 8px;
}

.t7-analysis-title {
    color: #f6f8fb;
    font-size: .94rem;
    font-weight: 790;
}

.t7-analysis-sub {
    color: #697991;
    font-size: .64rem;
    margin-top: 3px;
}

.t7-company-hero {
    padding: 15px 16px;
    border-radius: 11px;
    margin-top: 12px;
    background:
        radial-gradient(circle at 90% 20%, rgba(47,127,255,.09), transparent 31%),
        rgba(255,255,255,.018);
    border: 1px solid rgba(132,153,188,.13);
}

.t7-company-name {
    color: #f7f9fc;
    font-size: 1.08rem;
    font-weight: 790;
}

.t7-company-meta {
    color: #76869e;
    font-size: .65rem;
    margin-top: 5px;
}

.t7-company-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 9px;
    margin-top: 10px;
}

.t7-company-item {
    padding: 11px 12px;
    min-height: 72px;
    border-radius: 10px;
    background: rgba(255,255,255,.016);
    border: 1px solid rgba(132,153,188,.11);
}

.t7-company-label {
    color: #68778f;
    font-size: .56rem;
    text-transform: uppercase;
    letter-spacing: .055em;
}

.t7-company-value {
    color: #edf2f8;
    font-size: .78rem;
    font-weight: 710;
    margin-top: 7px;
    overflow-wrap: anywhere;
}

.t7-company-summary {
    color: #98a7bb;
    font-size: .70rem;
    line-height: 1.65;
    margin-top: 11px;
    padding: 13px 14px;
    border-radius: 10px;
    background: rgba(255,255,255,.014);
    border: 1px solid rgba(132,153,188,.10);
}

div[data-testid="stRadio"] > div {
    gap: .45rem !important;
}

div[data-testid="stRadio"] label {
    border: 1px solid rgba(132,153,188,.16) !important;
    border-radius: 9px !important;
    padding: 7px 12px !important;
    background: rgba(255,255,255,.018) !important;
}

div[data-testid="stRadio"] label:hover {
    border-color: rgba(47,127,255,.42) !important;
    background: rgba(47,127,255,.07) !important;
}

@media (max-width: 900px) {
    .t7-snapshot-grid,
    .t7-company-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 900px) {
    .t7-market-mini {
        display: none;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS
# =========================================================

def _safe_float(value):
    try:
        if value is None:
            return None

        value = float(value)

        if (
            np.isnan(value)
            or np.isinf(value)
        ):
            return None

        return value

    except Exception:
        return None


def _format_inr(value):
    value = _safe_float(
        value
    )

    if value is None:
        return "N/A"

    return f"₹{value:,.2f}"


def _format_market_cap(value):
    value = _safe_float(
        value
    )

    if value is None:
        return "N/A"

    crore = value / 10_000_000

    if crore >= 100_000:
        return (
            f"₹{crore / 100_000:.2f}L Cr"
        )

    if crore >= 1_000:
        return (
            f"₹{crore / 1_000:.2f}K Cr"
        )

    return (
        f"₹{crore:,.0f} Cr"
    )


def _calculate_change(
    current_price,
    previous_close,
):
    current_price = _safe_float(
        current_price
    )

    previous_close = _safe_float(
        previous_close
    )

    if (
        current_price is None
        or previous_close is None
        or previous_close == 0
    ):
        return 0.0, 0.0

    change = (
        current_price
        - previous_close
    )

    percent = (
        change
        / previous_close
    ) * 100

    return change, percent


def _close_series(
    data,
    ticker,
):
    if data is None or data.empty:
        return pd.Series(
            dtype=float
        )

    try:
        if isinstance(
            data.columns,
            pd.MultiIndex
        ):
            level0 = (
                data.columns
                .get_level_values(0)
            )

            level1 = (
                data.columns
                .get_level_values(1)
            )

            if ticker in level0:
                frame = data[
                    ticker
                ]

                if "Close" in frame.columns:
                    return pd.to_numeric(
                        frame["Close"],
                        errors="coerce"
                    ).dropna()

            if (
                "Close" in level0
                and ticker in level1
            ):
                return pd.to_numeric(
                    data["Close"][
                        ticker
                    ],
                    errors="coerce"
                ).dropna()

        if "Close" in data.columns:
            return pd.to_numeric(
                data["Close"],
                errors="coerce"
            ).dropna()

    except Exception:
        pass

    return pd.Series(
        dtype=float
    )


def _spark_svg(
    values,
    positive=True,
):
    values = [
        float(value)
        for value in values
        if pd.notna(value)
    ]

    if len(values) < 2:
        return ""

    low = min(
        values
    )

    high = max(
        values
    )

    spread = max(
        high - low,
        1e-9
    )

    width = 120
    height = 22

    points = []

    for index, value in enumerate(
        values
    ):
        x = (
            index
            / max(
                len(values) - 1,
                1
            )
        ) * width

        y = (
            height
            - (
                (
                    value - low
                )
                / spread
            )
            * 16
            - 3
        )

        points.append(
            f"{x:.1f},{y:.1f}"
        )

    color = (
        "#00df78"
        if positive
        else "#ff4658"
    )

    return (
        f'<svg class="t7-spark" '
        f'viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none">'
        f'<polyline '
        f'fill="none" '
        f'stroke="{color}" '
        f'stroke-width="2" '
        f'points="{" ".join(points)}" />'
        f'</svg>'
    )


# =========================================================
# CACHED MARKET DATA
# =========================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def _load_stock_info(
    symbol,
):
    try:
        return get_stock_info(
            symbol
        )
    except Exception:
        return None


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def _load_history(
    symbol,
    period,
):
    try:
        return get_stock_history(
            symbol,
            period
        )
    except Exception:
        return None


@st.cache_data(
    ttl=300,
    show_spinner=False
)
def _load_market_status():
    tickers = [
        "^NSEI",
        "^BSESN",
    ]

    try:
        data = yf.download(
            tickers=tickers,
            period="5d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        return []

    rows = []

    names = {
        "^NSEI":
            "NIFTY 50",

        "^BSESN":
            "SENSEX",
    }

    for ticker in tickers:
        close = _close_series(
            data,
            ticker
        )

        if len(close) < 2:
            continue

        latest = float(
            close.iloc[-1]
        )

        previous = float(
            close.iloc[-2]
        )

        change = (
            (
                latest
                - previous
            )
            / previous
        ) * 100

        rows.append(
            {
                "name":
                    names[ticker],

                "price":
                    latest,

                "change":
                    change,
            }
        )

    return rows


# =========================================================
# LIVE DASHBOARD SEARCH
# =========================================================

@st.fragment
def _dashboard_search():
    selected_symbol = st_searchbox(
        search_company_options,
        key="dashboard_company_search",
        placeholder="Search stocks, companies...",
        default=None,
        clear_on_submit=False,
        rerun_on_update=True,
        rerun_scope="fragment",
        debounce=450,
        edit_after_submit="option",
        style_overrides=DASHBOARD_SEARCH_STYLE,
    )

    if selected_symbol is None:
        return

    selected_symbol = str(
        selected_symbol
    ).strip().upper()

    if not selected_symbol:
        return

    if (
        st.session_state.get(
            "_dashboard_last_search_selection"
        )
        == selected_symbol
    ):
        return

    st.session_state[
        "_dashboard_last_search_selection"
    ] = selected_symbol

    st.session_state.stock = (
        selected_symbol
    )

    st.session_state.page = (
        "dashboard"
    )

    st.rerun()


# =========================================================
# PAGE HEADER
# =========================================================

def _render_header(
    info,
):
    title_col, search_col, market_col, home_col = st.columns(
        [2.1, 3.5, 2.8, 0.8],
        gap="medium"
    )

    with title_col:
        st.markdown(
            """
            <div class="t7-page-title">
                Dashboard
            </div>
            <div class="t7-page-subtitle">
                AI-powered stock analysis and forecasting
            </div>
            """,
            unsafe_allow_html=True,
        )

    with search_col:
        _dashboard_search()

    with market_col:
        market_rows = (
            _load_market_status()
        )

        if not market_rows:
            st.markdown(
                """
                <div class="t7-market-mini">
                    <div class="t7-market-mini-label">Market Status</div>
                    <div class="t7-market-mini-value positive">Open</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            columns = st.columns(
                len(market_rows)
            )

            for column, row in zip(
                columns,
                market_rows
            ):
                with column:
                    change_class = (
                        "positive"
                        if row["change"] >= 0
                        else "negative"
                    )

                    st.markdown(
                        f"""
                        <div class="t7-market-mini">
                            <div class="t7-market-mini-label">{row["name"]}</div>
                            <div class="t7-market-mini-value">{row["price"]:,.2f}</div>
                            <div class="t7-market-mini-change {change_class}">
                                {row["change"]:+.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with home_col:
        if st.button(
            "Home",
            key="dashboard_home_button",
            use_container_width=True,
        ):
            st.session_state.page = "home"

            st.session_state.pop(
                "dashboard_company_search",
                None
            )

            st.session_state.pop(
                "_dashboard_last_search_selection",
                None
            )

            st.rerun()


# =========================================================
# KPI CARDS
# =========================================================

def _render_kpis(
    info,
    analysis,
    recommendation,
):
    price = _safe_float(
        info.get("price")
    )

    ma20 = _safe_float(
        analysis.get("ma20")
    )

    ma50 = _safe_float(
        analysis.get("ma50")
    )

    forecast_change = _safe_float(
        analysis.get(
            "forecast_change"
        )
    )

    high_52 = _safe_float(
        info.get("high_52")
    )

    low_52 = _safe_float(
        info.get("low_52")
    )

    if (
        price is not None
        and ma20 is not None
        and ma50 is not None
    ):
        if (
            price > ma20
            and ma20 > ma50
        ):
            trend = "Bullish"
            trend_class = "positive"
        elif (
            price < ma20
            and ma20 < ma50
        ):
            trend = "Bearish"
            trend_class = "negative"
        else:
            trend = "Neutral"
            trend_class = "neutral"
    else:
        trend = "Neutral"
        trend_class = "neutral"

    confidence = (
        recommendation[
            "confidence"
        ]
    )

    if forecast_change is None:
        forecast_text = "N/A"
        forecast_class = ""
    else:
        forecast_text = (
            f"{forecast_change:+.2f}%"
        )

        forecast_class = (
            "positive"
            if forecast_change >= 0
            else "negative"
        )

    position = None

    if (
        price is not None
        and high_52 is not None
        and low_52 is not None
        and high_52 > low_52
    ):
        position = (
            (
                price - low_52
            )
            / (
                high_52 - low_52
            )
        ) * 100

    position_text = (
        f"{position:.0f}%"
        if position is not None
        else "N/A"
    )

    cards = [
        (
            "Market Trend",
            trend,
            trend_class,
            "Price trend vs MA20 / MA50",
            "green",
        ),
        (
            "AI Confidence",
            f"{confidence}%",
            "positive"
            if confidence >= 65
            else "neutral",
            f'{recommendation["recommendation"]} signal',
            "",
        ),
        (
            "30-Day Outlook",
            forecast_text,
            forecast_class,
            analysis.get(
                "best_model",
                "TEAM7 AI"
            ),
            "purple",
        ),
        (
            "52W Position",
            position_text,
            "positive"
            if position is not None
            and position >= 50
            else "neutral",
            "Position inside 52-week range",
            "green",
        ),
    ]

    cols = st.columns(
        4,
        gap="medium"
    )

    for (
        column,
        (
            label,
            value,
            value_class,
            subtitle,
            card_class,
        )
    ) in zip(
        cols,
        cards
    ):
        with column:
            st.markdown(
                f"""
                <div class="t7-kpi {card_class}">
                    <div class="t7-kpi-label">{label}</div>
                    <div class="t7-kpi-value {value_class}">{value}</div>
                    <div class="t7-kpi-sub">{html.escape(str(subtitle))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# CHART
# =========================================================

def _build_price_chart(
    history,
    chart_type,
    company_name,
):
    history = history.copy()

    date_column = (
        "Date"
        if "Date" in history.columns
        else history.columns[0]
    )

    rows = 2

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[
            0.78,
            0.22,
        ],
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=history[
                    date_column
                ],
                open=history[
                    "Open"
                ],
                high=history[
                    "High"
                ],
                low=history[
                    "Low"
                ],
                close=history[
                    "Close"
                ],
                name=company_name,
                increasing_line_color="#00df78",
                decreasing_line_color="#ff4658",
                increasing_fillcolor="#00df78",
                decreasing_fillcolor="#ff4658",
            ),
            row=1,
            col=1,
        )

    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=history[
                    date_column
                ],
                open=history[
                    "Open"
                ],
                high=history[
                    "High"
                ],
                low=history[
                    "Low"
                ],
                close=history[
                    "Close"
                ],
                name=company_name,
                increasing_line_color="#00df78",
                decreasing_line_color="#ff4658",
            ),
            row=1,
            col=1,
        )

    else:
        fill = (
            "tozeroy"
            if chart_type == "Area"
            else None
        )

        fig.add_trace(
            go.Scatter(
                x=history[
                    date_column
                ],
                y=history[
                    "Close"
                ],
                mode="lines",
                name=company_name,
                fill=fill,
                line={
                    "width": 2,
                    "color": "#2f7fff",
                },
                fillcolor="rgba(47,127,255,.12)",
            ),
            row=1,
            col=1,
        )

    if "Volume" in history.columns:
        volume_colors = []

        for _, row in history.iterrows():
            try:
                volume_colors.append(
                    "#00b96b"
                    if row["Close"] >= row["Open"]
                    else "#d63c4d"
                )
            except Exception:
                volume_colors.append(
                    "#506078"
                )

        fig.add_trace(
            go.Bar(
                x=history[
                    date_column
                ],
                y=history[
                    "Volume"
                ],
                marker_color=volume_colors,
                opacity=.72,
                name="Volume",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=500,
        margin={
            "l": 8,
            "r": 12,
            "t": 14,
            "b": 5,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#8b99ad",
            "size": 11,
        },
        hovermode="x unified",
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        color="#78869b",
    )

    fig.update_yaxes(
        gridcolor="rgba(120,140,170,.12)",
        zeroline=False,
        color="#7d899a",
        side="right",
        tickprefix="₹",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        color="#66758b",
        row=2,
        col=1,
    )

    return fig


def _render_chart_panel(
    info,
    preloaded_1y=None,
):
    current_price = _safe_float(
        info.get("price")
    )

    previous_close = _safe_float(
        info.get("previous_close")
    )

    _, percent = _calculate_change(
        current_price,
        previous_close,
    )

    change_class = (
        "positive"
        if percent >= 0
        else "negative"
    )

    period_col, chart_col = st.columns(
        [3.6, 1.2]
    )

    with period_col:
        period = st.radio(
            "Period",
            [
                "1d",
                "5d",
                "1mo",
                "3mo",
                "6mo",
                "1y",
                "2y",
                "5y",
            ],
            index=5,
            horizontal=True,
            key="dashboard_v2_period",
            label_visibility="collapsed",
        )

    with chart_col:
        chart_type = st.selectbox(
            "Chart",
            [
                "Candlestick",
                "Line",
                "Area",
                "OHLC",
            ],
            key="dashboard_v2_chart_type",
            label_visibility="collapsed",
        )

    if (
        period == "1y"
        and preloaded_1y is not None
        and not preloaded_1y.empty
    ):
        history = preloaded_1y
    else:
        history = _load_history(
            info["symbol"],
            period,
        )

    if (
        history is None
        or history.empty
    ):
        st.error(
            "Unable to load historical data."
        )

        return None

    st.markdown(
        f"""
        <div class="t7-stock-name">
            {html.escape(str(info.get("name", info["symbol"])))}
            <span class="t7-stock-symbol">{html.escape(str(info["symbol"]))}</span>
        </div>
        <div class="t7-stock-price">
            {_format_inr(current_price)}
            <span class="t7-stock-change {change_class}">
                {percent:+.2f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = _build_price_chart(
        history,
        chart_type,
        str(
            info.get(
                "name",
                info["symbol"]
            )
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )

    return history


# =========================================================
# AI PREDICTION PANEL
# =========================================================

def _render_ai_panel(
    info,
    analysis,
    recommendation,
    history,
):
    forecast_price = _safe_float(
        analysis.get(
            "forecast_price"
        )
    )

    forecast_change = _safe_float(
        analysis.get(
            "forecast_change"
        )
    )

    current_price = _safe_float(
        info.get("price")
    )

    action = recommendation[
        "recommendation"
    ]

    theme = recommendation[
        "theme"
    ]

    pill_class = (
        "sell"
        if theme == "sell"
        else "hold"
        if theme == "hold"
        else ""
    )

    price_class = pill_class

    forecast_price_text = (
        _format_inr(
            forecast_price
        )
        if forecast_price is not None
        else "N/A"
    )

    forecast_change_text = (
        f"{forecast_change:+.2f}%"
        if forecast_change is not None
        else "N/A"
    )

    change_class = (
        "positive"
        if forecast_change is not None
        and forecast_change >= 0
        else "negative"
        if forecast_change is not None
        else ""
    )

    st.markdown(
        f"""
        <div class="t7-panel-header">
            <div class="t7-panel-title">AI Prediction</div>
            <div class="t7-panel-sub">{html.escape(str(info.get("name", info["symbol"])))} · 30 Trading Days</div>
        </div>
        <div class="t7-ai-callout">
            <span class="t7-ai-pill {pill_class}">
                {action}
            </span>
            <div class="t7-ai-price {price_class}">
                {forecast_price_text}
            </div>
            <div class="t7-ai-return {change_class}">
                {forecast_change_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        current_price is not None
        and forecast_price is not None
    ):
        recent = history

        if (
            recent is None
            or recent.empty
            or "Close" not in recent.columns
            or len(recent) < 10
        ):
            recent = _load_history(
                info["symbol"],
                "1mo",
            )

        if (
            recent is not None
            and not recent.empty
            and "Close" in recent.columns
        ):
            close = (
                recent["Close"]
                .dropna()
                .astype(float)
                .tail(30)
            )

            historical_x = list(
                range(
                    len(close)
                )
            )

            future_x = list(
                range(
                    len(close) - 1,
                    len(close) + 7
                )
            )

            future_y = np.linspace(
                close.iloc[-1],
                forecast_price,
                len(future_x)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=historical_x,
                    y=close,
                    mode="lines",
                    line={
                        "color": "#00df78",
                        "width": 2,
                    },
                    fill="tozeroy",
                    fillcolor="rgba(0,223,120,.08)",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=future_x,
                    y=future_y,
                    mode="lines",
                    line={
                        "color": "#00df78",
                        "width": 2,
                        "dash": "dot",
                    },
                )
            )

            fig.update_layout(
                height=250,
                margin={
                    "l": 2,
                    "r": 4,
                    "t": 8,
                    "b": 4,
                },
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )

            fig.update_xaxes(
                visible=False
            )

            fig.update_yaxes(
                gridcolor="rgba(120,140,170,.10)",
                color="#74839a",
                side="right",
                tickprefix="₹",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False,

                    "displaylogo":
                        False,
                },
            )

    st.markdown(
        f"""
        <div class="t7-ai-stats">
            <div class="t7-ai-stat">
                <div class="t7-ai-stat-label">Confidence</div>
                <div class="t7-ai-stat-value">{recommendation["confidence"]}%</div>
            </div>
            <div class="t7-ai-stat">
                <div class="t7-ai-stat-label">Time Horizon</div>
                <div class="t7-ai-stat-value">30 Days</div>
            </div>
            <div class="t7-ai-stat">
                <div class="t7-ai-stat-label">Algorithm</div>
                <div class="t7-ai-stat-value">{html.escape(str(analysis.get("best_model", "TEAM7 AI")))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def _render_technicals(
    analysis,
    history,
):
    rsi = _safe_float(
        analysis.get("rsi")
    )

    macd = _safe_float(
        analysis.get("macd")
    )

    macd_signal = _safe_float(
        analysis.get(
            "macd_signal"
        )
    )

    ma20 = _safe_float(
        analysis.get("ma20")
    )

    ma50 = _safe_float(
        analysis.get("ma50")
    )

    volatility = None

    try:
        returns = (
            history["Close"]
            .pct_change()
            .dropna()
        )

        if not returns.empty:
            volatility = (
                returns.std()
                * np.sqrt(252)
                * 100
            )
    except Exception:
        volatility = None

    if rsi is None:
        rsi_status = "N/A"
        rsi_class = ""
    elif rsi < 30:
        rsi_status = "Oversold"
        rsi_class = "positive"
    elif rsi > 70:
        rsi_status = "Overbought"
        rsi_class = "negative"
    elif rsi >= 50:
        rsi_status = "Bullish"
        rsi_class = "positive"
    else:
        rsi_status = "Neutral"
        rsi_class = "neutral"

    if (
        macd is not None
        and macd_signal is not None
    ):
        macd_status = (
            "Bullish"
            if macd > macd_signal
            else "Bearish"
        )

        macd_class = (
            "positive"
            if macd > macd_signal
            else "negative"
        )
    else:
        macd_status = "N/A"
        macd_class = ""

    ma20_status = (
        "Trend"
        if ma20 is not None
        else "N/A"
    )

    ma50_status = (
        "Trend"
        if ma50 is not None
        else "N/A"
    )

    volatility_status = (
        "Low"
        if volatility is not None
        and volatility < 25
        else "Moderate"
        if volatility is not None
        and volatility < 45
        else "High"
        if volatility is not None
        else "N/A"
    )

    volatility_class = (
        "positive"
        if volatility_status == "Low"
        else "neutral"
        if volatility_status == "Moderate"
        else "negative"
        if volatility_status == "High"
        else ""
    )

    close_values = []

    try:
        close_values = (
            history["Close"]
            .dropna()
            .tail(18)
            .tolist()
        )
    except Exception:
        close_values = []

    spark = _spark_svg(
        close_values,
        True,
    )

    cards = [
        (
            "RSI (14)",
            f"{rsi:.2f}"
            if rsi is not None
            else "N/A",
            rsi_status,
            rsi_class,
        ),
        (
            "MACD (12,26,9)",
            f"{macd:.2f}"
            if macd is not None
            else "N/A",
            macd_status,
            macd_class,
        ),
        (
            "Moving Avg (20)",
            _format_inr(
                ma20
            ),
            ma20_status,
            "positive",
        ),
        (
            "Moving Avg (50)",
            _format_inr(
                ma50
            ),
            ma50_status,
            "positive",
        ),
        (
            "Volatility",
            f"{volatility:.2f}%"
            if volatility is not None
            else "N/A",
            volatility_status,
            volatility_class,
        ),
    ]

    st.markdown(
        '<div class="t7-panel-title">Technical Indicators</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(
        5,
        gap="small"
    )

    for (
        column,
        (
            label,
            value,
            status,
            status_class,
        )
    ) in zip(
        cols,
        cards
    ):
        with column:
            st.markdown(
                f"""
                <div class="t7-tech-card">
                    <div class="t7-tech-label">{label}</div>
                    <div class="t7-tech-value">{value}</div>
                    <div class="t7-tech-status {status_class}">{status}</div>
                    {spark}
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# DAILY RETURNS
# =========================================================

def _render_daily_returns(
    history,
):
    if (
        history is None
        or history.empty
        or "Close" not in history.columns
    ):
        return

    data = history.copy()

    if "Date" not in data.columns:
        data = data.reset_index()

    if "Datetime" in data.columns:
        data = data.rename(
            columns={
                "Datetime": "Date"
            }
        )

    date_column = (
        "Date"
        if "Date" in data.columns
        else data.columns[0]
    )

    data[
        "Close"
    ] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data[
        "Daily Return"
    ] = (
        data["Close"]
        .pct_change()
        * 100
    )

    data = data.dropna(
        subset=[
            date_column,
            "Daily Return",
        ]
    )

    if data.empty:
        return

    daily_returns = data[
        "Daily Return"
    ]

    latest_return = float(
        daily_returns.iloc[-1]
    )

    average_return = float(
        daily_returns.mean()
    )

    best_return = float(
        daily_returns.max()
    )

    worst_return = float(
        daily_returns.min()
    )

    positive_days = float(
        (
            daily_returns > 0
        ).mean()
        * 100
    )

    st.markdown(
        '<div class="t7-returns-shell">'
        '<div class="t7-returns-title">Daily Returns</div>'
        '<div class="t7-returns-sub">'
        'Day-to-day percentage movement based on closing prices'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(
        5,
        gap="small"
    )

    c1.metric(
        "Latest",
        f"{latest_return:+.2f}%"
    )

    c2.metric(
        "Average",
        f"{average_return:+.2f}%"
    )

    c3.metric(
        "Best Day",
        f"{best_return:+.2f}%"
    )

    c4.metric(
        "Worst Day",
        f"{worst_return:+.2f}%"
    )

    c5.metric(
        "Positive Days",
        f"{positive_days:.0f}%"
    )

    recent = data.tail(
        30
    )

    colors = [
        "#00df78"
        if value >= 0
        else "#ff4658"
        for value in recent[
            "Daily Return"
        ]
    ]

    fig = go.Figure(
        go.Bar(
            x=recent[
                date_column
            ],
            y=recent[
                "Daily Return"
            ],
            marker_color=colors,
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Return: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="rgba(150,165,190,.30)",
    )

    fig.update_layout(
        height=230,
        margin={
            "l": 4,
            "r": 4,
            "t": 8,
            "b": 6,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=.28,
        font={
            "color": "#7d899d",
            "size": 10,
        },
    )

    fig.update_xaxes(
        showgrid=False,
        color="#65758d",
    )

    fig.update_yaxes(
        title="Return %",
        gridcolor="rgba(120,140,170,.10)",
        zeroline=False,
        ticksuffix="%",
        color="#65758d",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar":
                False,

            "displaylogo":
                False,
        },
    )


# =========================================================
# POLISHED LOWER CARDS
# =========================================================

def _render_stock_snapshot(
    info,
):
    pe = _safe_float(
        info.get("pe_ratio")
    )

    eps = _safe_float(
        info.get("eps")
    )

    cards = [
        (
            "Current Price",
            _format_inr(
                info.get("price")
            ),
            "Latest market value",
        ),
        (
            "Market Cap",
            _format_market_cap(
                info.get("market_cap")
            ),
            "Company valuation",
        ),
        (
            "P/E Ratio",
            f"{pe:.2f}"
            if pe is not None
            else "N/A",
            "Price to earnings",
        ),
        (
            "EPS",
            f"{eps:.2f}"
            if eps is not None
            else "N/A",
            "Earnings per share",
        ),
        (
            "52W High",
            _format_inr(
                info.get("high_52")
            ),
            "Highest in 52 weeks",
        ),
        (
            "52W Low",
            _format_inr(
                info.get("low_52")
            ),
            "Lowest in 52 weeks",
        ),
    ]

    items = []

    for label, value, note in cards:
        items.append(
            '<div class="t7-snapshot-item">'
            f'<div class="t7-snapshot-label">{label}</div>'
            f'<div class="t7-snapshot-value">{value}</div>'
            f'<div class="t7-snapshot-note">{note}</div>'
            '</div>'
        )

    snapshot_html = (
        '<div class="t7-snapshot-shell">'
        '<div class="t7-lower-title">Stock Snapshot</div>'
        '<div class="t7-lower-subtitle">Key financial and price statistics</div>'
        '<div class="t7-snapshot-grid">'
        + ''.join(items)
        + '</div>'
        '</div>'
    )

    st.markdown(
        snapshot_html,
        unsafe_allow_html=True,
    )

def _render_ai_signal_summary(
    recommendation,
):
    action = recommendation[
        "recommendation"
    ]

    confidence = recommendation[
        "confidence"
    ]

    theme = recommendation[
        "theme"
    ]

    bullish_count = len(
        recommendation[
            "bullish"
        ]
    )

    risk_count = len(
        recommendation[
            "bearish"
        ]
    )

    neutral_count = len(
        recommendation.get(
            "neutral",
            []
        )
    )

    action_class = (
        "positive"
        if theme == "buy"
        else "negative"
        if theme == "sell"
        else "neutral"
    )

    if recommendation[
        "bullish"
    ]:
        note = recommendation[
            "bullish"
        ][0]

    elif recommendation[
        "bearish"
    ]:
        note = recommendation[
            "bearish"
        ][0]

    elif recommendation.get(
        "neutral"
    ):
        note = recommendation[
            "neutral"
        ][0]

    else:
        note = (
            "No dominant technical signal "
            "was detected."
        )

    signal_html = (
        '<div class="t7-signal-shell">'
        '<div class="t7-lower-title">AI Signal Summary</div>'
        '<div class="t7-lower-subtitle">Condensed view of TEAM7 AI recommendation engine</div>'
        f'<div class="t7-signal-hero {theme}">'
        '<div>'
        '<div class="t7-signal-kicker">TEAM7 AI SIGNAL</div>'
        f'<div class="t7-signal-action {action_class}">{action}</div>'
        '</div>'
        '<div class="t7-signal-confidence">'
        '<div class="t7-signal-confidence-label">Confidence</div>'
        f'<div class="t7-signal-confidence-value">{confidence}%</div>'
        '<div class="t7-signal-bar">'
        f'<div class="t7-signal-fill {theme}" style="width:{confidence}%;"></div>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="t7-signal-counts">'
        '<div class="t7-signal-count">'
        '<div class="t7-signal-count-label">Bullish Signals</div>'
        f'<div class="t7-signal-count-value positive">{bullish_count}</div>'
        '</div>'
        '<div class="t7-signal-count">'
        '<div class="t7-signal-count-label">Risk Signals</div>'
        f'<div class="t7-signal-count-value negative">{risk_count}</div>'
        '</div>'
        '<div class="t7-signal-count">'
        '<div class="t7-signal-count-label">Neutral</div>'
        f'<div class="t7-signal-count-value neutral">{neutral_count}</div>'
        '</div>'
        '</div>'
        f'<div class="t7-signal-note">{html.escape(str(note))}</div>'
        '</div>'
    )

    st.markdown(
        signal_html,
        unsafe_allow_html=True,
    )

# =========================================================
# MORE ANALYSIS
# =========================================================

def _render_company_profile(
    info,
):
    pe = _safe_float(
        info.get("pe_ratio")
    )

    eps = _safe_float(
        info.get("eps")
    )

    dividend = _safe_float(
        info.get("dividend_yield")
    )

    if dividend is not None:
        if abs(dividend) <= 1:
            dividend_text = (
                f"{dividend * 100:.2f}%"
            )
        else:
            dividend_text = (
                f"{dividend:.2f}%"
            )
    else:
        dividend_text = "N/A"

    summary = str(
        info.get(
            "summary",
            ""
        )
        or
        "Company description is not available."
    )

    company_name = html.escape(
        str(
            info.get(
                "name",
                "N/A"
            )
        )
    )

    symbol = html.escape(
        str(
            info.get(
                "symbol",
                "N/A"
            )
        )
    )

    sector = html.escape(
        str(
            info.get(
                "sector",
                "N/A"
            )
        )
    )

    industry = html.escape(
        str(
            info.get(
                "industry",
                "N/A"
            )
        )
    )

    country = html.escape(
        str(
            info.get(
                "country",
                "N/A"
            )
        )
    )

    company_html = (
        '<div class="t7-company-hero">'
        f'<div class="t7-company-name">{company_name}</div>'
        f'<div class="t7-company-meta">{symbol} · {sector} · {industry}</div>'
        '</div>'
        '<div class="t7-company-grid">'
        '<div class="t7-company-item">'
        '<div class="t7-company-label">Country</div>'
        f'<div class="t7-company-value">{country}</div>'
        '</div>'
        '<div class="t7-company-item">'
        '<div class="t7-company-label">P/E Ratio</div>'
        f'<div class="t7-company-value">{f"{pe:.2f}" if pe is not None else "N/A"}</div>'
        '</div>'
        '<div class="t7-company-item">'
        '<div class="t7-company-label">EPS</div>'
        f'<div class="t7-company-value">{f"{eps:.2f}" if eps is not None else "N/A"}</div>'
        '</div>'
        '<div class="t7-company-item">'
        '<div class="t7-company-label">Dividend Yield</div>'
        f'<div class="t7-company-value">{dividend_text}</div>'
        '</div>'
        '</div>'
        f'<div class="t7-company-summary">{html.escape(summary)}</div>'
    )

    st.markdown(
        company_html,
        unsafe_allow_html=True,
    )

    website = str(
        info.get(
            "website",
            ""
        )
        or
        ""
    ).strip()

    if website.startswith(
        (
            "http://",
            "https://",
        )
    ):
        st.link_button(
            "Open Company Website ↗",
            website,
            use_container_width=True,
        )

def _render_more_analysis(
    info,
    base_history,
):
    analysis_header_html = (
        '<div class="t7-analysis-shell">'
        '<div class="t7-analysis-head">'
        '<div>'
        '<div class="t7-analysis-title">More Analysis</div>'
        '<div class="t7-analysis-sub">'
        'Explore company fundamentals, peer comparison, market news or run the full LSTM forecast.'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )

    st.markdown(
        analysis_header_html,
        unsafe_allow_html=True,
    )

    section = st.radio(
        "More Analysis",
        [
            "Company",
            "Compare",
            "News",
            "Deep AI",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_v2_more_section",
    )

    with st.container(
        border=True
    ):
        if section == "Company":
            _render_company_profile(
                info
            )

        elif section == "Compare":
            stock_comparison(
                info["symbol"]
            )

        elif section == "News":
            market_news(
                info["symbol"]
            )

        elif section == "Deep AI":
            st.markdown(
                """
                <div class="t7-panel-header">
                    <div class="t7-panel-title">Deep AI Forecast</div>
                    <div class="t7-panel-sub">
                        LSTM runs only when requested to keep the dashboard fast.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "Run Full LSTM Forecast",
                key="dashboard_run_deep_ai",
                use_container_width=True,
            ):
                st.session_state[
                    "dashboard_show_deep_ai"
                ] = True

            if st.session_state.get(
                "dashboard_show_deep_ai",
                False
            ):
                from components.prediction import prediction

                prediction(
                    base_history
                )
# =========================================================
# MAIN DASHBOARD
# =========================================================

def dashboard_page():
    _inject_dashboard_styles()

    symbol = str(
        st.session_state.get(
            "stock",
            ""
        )
    ).strip().upper()

    if not symbol:
        st.warning(
            "Please select a company from the Home page."
        )

        if st.button(
            "Go to Home",
            use_container_width=True,
        ):
            st.session_state.page = (
                "home"
            )

            st.rerun()

        return

    with st.spinner(
        "Loading dashboard..."
    ):
        info = _load_stock_info(
            symbol
        )

        # One shared one-year history powers:
        # AI recommendation, default price chart and Daily Returns.
        # This avoids downloading the same stock history multiple times.
        analysis_history = _load_history(
            symbol,
            "1y"
        )

    if info is None:
        st.error(
            "Unable to load this stock."
        )
        return

    # =====================================================
    # AI DATA
    # =====================================================

    if (
        analysis_history is not None
        and not analysis_history.empty
    ):
        analysis = analyze_stock_ai_from_history(
            analysis_history
        )
    else:
        analysis = analyze_stock_ai(
            symbol
        )

    if analysis is None:
        analysis = {
            "price":
                _safe_float(
                    info.get("price")
                ),

            "ma20":
                None,

            "ma50":
                None,

            "rsi":
                None,

            "macd":
                None,

            "macd_signal":
                None,

            "best_model":
                "TEAM7 AI",

            "forecast_price":
                None,

            "forecast_change":
                None,
        }

    recommendation = (
        build_recommendation(
            info,
            analysis,
        )
    )

    # =====================================================
    # FULL-WIDTH DASHBOARD
    # =====================================================

    # TOP HEADER
    _render_header(
        info
    )

    st.write("")

    # ================================================
    # KPI CARDS
    # ================================================

    _render_kpis(
        info,
        analysis,
        recommendation,
    )

    st.write("")

    # ================================================
    # CHART + AI PREDICTION
    # ================================================

    chart_col, ai_col = st.columns(
        [2.15, 1.0],
        gap="medium"
    )

    with chart_col:
        with st.container(
            border=True
        ):
            history = _render_chart_panel(
                info,
                preloaded_1y=analysis_history,
            )

    with ai_col:
        with st.container(
            border=True
        ):
            _render_ai_panel(
                info,
                analysis,
                recommendation,
                history,
            )

    if (
        history is None
        or history.empty
    ):
        return

    st.write("")

    # ================================================
    # BALANCED LOWER GRID
    #
    # Left column:
    #   Technical Indicators
    #   Stock Snapshot
    #
    # Right column:
    #   Related Stocks
    #   AI Signal Summary
    #
    # This removes the large blank area caused by putting
    # unequal-height panels in separate Streamlit rows.
    # ================================================

    left_stack, right_stack = st.columns(
        [2.15, 1.0],
        gap="medium"
    )

    with left_stack:
        with st.container(
            border=True
        ):
            _render_technicals(
                analysis,
                history,
            )

        _render_daily_returns(
            analysis_history
            if analysis_history is not None
            else history
        )

        _render_stock_snapshot(
            info
        )

    with right_stack:
        with st.container(
            border=True
        ):
            related_stocks(
                info
            )

        _render_ai_signal_summary(
            recommendation
        )

    # ================================================
    # MORE ANALYSIS
    # ================================================

    _render_more_analysis(
        info,
        history,
    )
