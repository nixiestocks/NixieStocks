import streamlit as st

import components.home as home_module
import components.dashboard as dashboard_module

from data.company_fundamentals import enrich_stock_info
from data.live_market import (
    load_live_home_market_data,
    load_live_market_status,
    load_live_stock_info,
    load_dashboard_history,
    augment_history_with_live_session,
)


st.set_page_config(
    page_title="NixieStocks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# HIDE STREAMLIT CLOUD TOP TOOLBAR
# Keeps the NixieStocks animated stock ticker fully visible.
# =========================================================

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    [data-testid="stAppViewContainer"] > section.main {
        top: 0 !important;
    }

    /* Rebrand the two legacy AI signal labels without touching the
       dashboard layout/CSS architecture. */
    .t7-signal-shell .t7-lower-subtitle,
    .t7-signal-kicker {
        font-size: 0 !important;
    }

    .t7-signal-shell .t7-lower-subtitle::after {
        content: "Condensed view of NixieStocks AI recommendation engine";
        font-size: .72rem;
    }

    .t7-signal-kicker::after {
        content: "NixieStocks AI SIGNAL";
        font-size: .68rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LIVE MARKET OVERLAYS
#
# Home ticker + market cards: 5-minute intraday data.
# Dashboard NIFTY/SENSEX: 5-minute intraday data.
# Dashboard current quote: short cache.
# Historical chart: today's intraday session is added visually.
# AI + Daily Returns: keep completed 1-day history unchanged.
# =========================================================

home_module._load_home_market_data = (
    load_live_home_market_data
)

dashboard_module._load_market_status = (
    load_live_market_status
)


def _load_enriched_stock_info(symbol):
    base_info = load_live_stock_info(symbol)
    return enrich_stock_info(symbol, base_info)


dashboard_module._load_stock_info = (
    _load_enriched_stock_info
)

dashboard_module._load_history = (
    load_dashboard_history
)


# Normalize the legacy fallback model label to the new brand.
def _brand_analysis(result):
    if isinstance(result, dict):
        result = result.copy()
        if result.get("best_model") == ("TEAM" + "7 AI"):
            result["best_model"] = "NixieStocks AI"
    return result


if not hasattr(dashboard_module, "_nixie_original_analyze_stock_ai"):
    dashboard_module._nixie_original_analyze_stock_ai = (
        dashboard_module.analyze_stock_ai
    )
    dashboard_module._nixie_original_analyze_stock_ai_from_history = (
        dashboard_module.analyze_stock_ai_from_history
    )


def _nixie_analyze_stock_ai(symbol):
    return _brand_analysis(
        dashboard_module._nixie_original_analyze_stock_ai(symbol)
    )


def _nixie_analyze_stock_ai_from_history(history):
    return _brand_analysis(
        dashboard_module._nixie_original_analyze_stock_ai_from_history(history)
    )


dashboard_module.analyze_stock_ai = _nixie_analyze_stock_ai
dashboard_module.analyze_stock_ai_from_history = (
    _nixie_analyze_stock_ai_from_history
)


# Keep the original dashboard chart function only once across
# Streamlit reruns, then feed it an augmented 1Y history for display.
if not hasattr(
    dashboard_module,
    "_t7_original_render_chart_panel",
):
    dashboard_module._t7_original_render_chart_panel = (
        dashboard_module._render_chart_panel
    )


def _render_chart_panel_live(
    info,
    preloaded_1y=None,
):
    chart_history = preloaded_1y

    if (
        chart_history is not None
        and not chart_history.empty
    ):
        chart_history = augment_history_with_live_session(
            chart_history,
            info["symbol"],
        )

    return (
        dashboard_module
        ._t7_original_render_chart_panel(
            info,
            preloaded_1y=chart_history,
        )
    )


dashboard_module._render_chart_panel = (
    _render_chart_panel_live
)


home_page = home_module.home_page
dashboard_page = dashboard_module.dashboard_page


if "page" not in st.session_state:
    st.session_state.page = "home"

if "stock" not in st.session_state:
    st.session_state.stock = ""


# =========================================================
# ROUTER
# =========================================================

if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "dashboard":
    dashboard_page()

else:
    st.session_state.page = "home"
    st.rerun()
