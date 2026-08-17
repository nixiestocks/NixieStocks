import streamlit as st

from data.yahoo import get_stock_history
from charts.plotly_chart import create_price_chart


def dashboard_chart(symbol, company_name):

    left, right = st.columns([2, 1])

    with left:

        period = st.selectbox(

            "Time Period",

            [

                "1d",

                "5d",

                "1mo",

                "3mo",

                "6mo",

                "1y",

                "2y",

                "5y",

                "max",

            ],

            index=5,

            key="period"

        )

    with right:

        chart_type = st.selectbox(

            "Chart Type",

            [

                "Candlestick",

                "Line",

                "Area",

                "OHLC",

            ],

            key="chart_type"

        )

    history = get_stock_history(

        symbol,

        period,
        

    )

    if history is None or history.empty:

        st.error("Unable to load stock history.")

        return

    fig = create_price_chart(

        history,

        company_name,

        chart_type,

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

        config={

            "displaylogo": False,

            "scrollZoom": True,

            "responsive": True,

        },

    )

    return history