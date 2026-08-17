import streamlit as st


def _format_market_cap(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)

        if value >= 1_000_000_000_000:
            return f"${value/1_000_000_000_000:.2f}T"

        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"

        if value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"

        return f"${value:,.0f}"

    except Exception:
        return "N/A"


def _format_volume(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)

        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"

        if value >= 1_000_000:
            return f"{value/1_000_000:.2f}M"

        if value >= 1_000:
            return f"{value/1_000:.2f}K"

        return f"{int(value):,}"

    except Exception:
        return "N/A"


def _money(currency, value):
    if value is None:
        return "N/A"

    try:
        return f"{currency} {float(value):,.2f}"
    except Exception:
        return "N/A"


def dashboard_metrics(info):

    currency = info.get("currency", "$")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        st.metric(
            "Market Cap",
            _format_market_cap(info.get("market_cap"))
        )

    with c2:
        pe = info.get("pe_ratio")
        st.metric(
            "P/E Ratio",
            "N/A" if pe is None else f"{float(pe):.2f}"
        )

    with c3:
        eps = info.get("eps")
        st.metric(
            "EPS",
            "N/A" if eps is None else f"{float(eps):.2f}"
        )

    with c4:
        st.metric(
            "52 Week High",
            _money(currency, info.get("high_52"))
        )

    with c5:
        st.metric(
            "52 Week Low",
            _money(currency, info.get("low_52"))
        )

    with c6:
        st.metric(
            "Volume",
            _format_volume(info.get("volume"))
        )