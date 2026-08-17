import streamlit as st


def navbar():

    page = st.session_state.get("page", "home")

    # The prototype home page has no extra navigation bar.
    # Keep navigation compact only when a dashboard is open.
    if page == "home":
        return

    left, center, right = st.columns([1.1, 5.8, 1.1])

    with left:
        if st.button(
            "Home",
            key="nav_home",
            use_container_width=True,
        ):
            st.session_state.page = "home"
            st.session_state.pop("team7_company_search", None)
            st.session_state.pop("_last_search_selection", None)
            st.rerun()

    with center:
        st.markdown(
            """
            <div class="dashboard-nav-brand">
                <span class="dashboard-nav-logo">TEAM7</span>
                <span class="dashboard-nav-divider"></span>
                <span class="dashboard-nav-label">Stock Intelligence Dashboard</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="dashboard-nav-status">LIVE MARKET DATA</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="dashboard-nav-line"></div>', unsafe_allow_html=True)
