import streamlit as st
from streamlit_searchbox import st_searchbox

from data.company_search import search_company_options


SEARCH_STYLE = {
    "dropdown": {
        "rotate": True,
        "width": 24,
        "height": 24,
        "fill": "#7f8da8",
    },
    "clear": {
        "width": 18,
        "height": 18,
        "icon": "cross",
        "clearable": "always",
    },
    "searchbox": {
        "control": {
            "backgroundColor": "#0d1526",
            "border": "1px solid #263a61",
            "borderRadius": "12px",
            "minHeight": "54px",
            "boxShadow": "0 10px 30px rgba(0,0,0,0.22)",
        },
        "menu": {
            "backgroundColor": "#08111f",
            "border": "1px solid #223454",
            "borderRadius": "12px",
            "overflow": "hidden",
        },
        "menuList": {
            "backgroundColor": "#08111f",
        },
        "option": {
            "backgroundColor": "#08111f",
            "color": "#e8eef9",
            "highlightColor": "#7aa2ff",
        },
        "placeholder": {
            "color": "#75829b",
        },
        "input": {
            "color": "#f8fafc",
        },
        "singleValue": {
            "color": "#f8fafc",
        },
    },
}


@st.fragment
def _company_search_fragment():

    selected_symbol = st_searchbox(
        search_company_options,
        key="team7_company_search",
        placeholder="Search any listed company worldwide...",
        default=None,
        clear_on_submit=False,
        rerun_on_update=True,
        rerun_scope="fragment",
        debounce=450,
        edit_after_submit="option",
        style_overrides=SEARCH_STYLE,
    )

    if selected_symbol is None:
        return

    selected_symbol = str(selected_symbol).strip().upper()

    if not selected_symbol:
        return

    if st.session_state.get("_last_search_selection") == selected_symbol:
        return

    st.session_state["_last_search_selection"] = selected_symbol
    st.session_state["stock"] = selected_symbol
    st.session_state["page"] = "dashboard"

    st.rerun()


def hero():

    st.markdown(
        """
        <div class="t7-hero">
            <div class="t7-brand">TEAM7</div>
            <div class="t7-subtitle">AI STOCK MARKET ANALYST</div>
            <div class="t7-tagline">
                Global market intelligence, technical analysis and AI forecasting
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns([1.25, 7.5, 1.25])

    with middle:
        _company_search_fragment()

        st.markdown(
            """
            <div class="t7-search-note">
                Search by company name — ticker symbols are handled automatically
            </div>
            """,
            unsafe_allow_html=True,
        )
