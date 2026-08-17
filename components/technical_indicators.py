import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def technical_indicators(history):

    if history is None or history.empty:
        return

    st.header("Technical Indicators")

    df = history.copy()

    # Moving Averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    # ------------------------
    # Price + Moving Average
    # ------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Close"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA20"],
            name="MA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA50"],
            name="MA50"
        )
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        title="Moving Averages"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------
    # RSI
    # ------------------------

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            name="RSI"
        )
    )

    fig2.add_hline(y=70)

    fig2.add_hline(y=30)

    fig2.update_layout(
        height=300,
        template="plotly_dark",
        title="Relative Strength Index"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------
    # MACD
    # ------------------------

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MACD"],
            name="MACD"
        )
    )

    fig3.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Signal"],
            name="Signal"
        )
    )

    fig3.update_layout(
        height=300,
        template="plotly_dark",
        title="MACD"
    )

    st.plotly_chart(fig3, use_container_width=True)