import streamlit as st

from components.home import home_page
from components.dashboard import dashboard_page


st.set_page_config(
    page_title="TEAM7",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
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
