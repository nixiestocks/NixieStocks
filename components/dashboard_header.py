import streamlit as st


def dashboard_header(info):

    left, right = st.columns([4, 1])

    with left:
        st.title(info.get("name", "Unknown Company"))
        st.caption(info.get("symbol", ""))

    with right:
        currency = info.get("currency", "")
        price = info.get("price")

        if price is None:
            value = "N/A"
        else:
            value = f"{currency} {price:,.2f}"

        st.metric(
            "Current Price",
            value,
        )

    st.divider()