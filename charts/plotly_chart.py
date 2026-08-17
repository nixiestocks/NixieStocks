import plotly.graph_objects as go


def create_price_chart(df, company_name, chart_type="Candlestick"):

    if df is None or df.empty:
        return go.Figure()

    # Ensure Date column exists
    if "Date" not in df.columns:
        df = df.reset_index()

    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )

    elif chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Close Price",
            )
        )

    elif chart_type == "Area":
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                fill="tozeroy",
                name="Close Price",
            )
        )

    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=company_name,
        height=600,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        dragmode="zoom",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig