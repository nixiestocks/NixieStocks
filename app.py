import streamlit as st

from components.home import home_page
from components.dashboard import dashboard_page


st.set_page_config(
    page_title="TEAM7",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# HIDE STREAMLIT CLOUD TOP TOOLBAR
# Keeps the TEAM7 animated stock ticker fully visible.
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
    </style>
    """,
    unsafe_allow_html=True,
)


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
