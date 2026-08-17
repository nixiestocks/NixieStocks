import streamlit as st
from data.market import get_market_overview


def market_overview():

    st.subheader(" Market Overview")

    data = get_market_overview()

    cols = st.columns(4)

    indexes = [
        "NIFTY 50",
        "SENSEX",
        "BANK NIFTY",
        "NASDAQ"
    ]

    for col, name in zip(cols, indexes):

        value = data.get(name)

        if value:

            col.metric(
                label=name,
                value=value["price"],
                delta=f'{value["change"]:+.2f}%'
            )

        else:

            col.metric(
                label=name,
                value="N/A",
                delta="0.00%"
            )