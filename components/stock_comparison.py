import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data.yahoo import get_stock_history


# =========================================================
# PERIOD SETTINGS
# =========================================================

PERIODS = {
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}


# =========================================================
# LOAD STOCK HISTORY
# =========================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_comparison_history(
    symbol,
    period
):

    try:

        df = get_stock_history(
            symbol,
            period
        )

        if df is None or df.empty:
            return None

        df = df.copy()

        # -----------------------------------------
        # Make sure Date exists
        # -----------------------------------------

        if "Date" not in df.columns:
            df = df.reset_index()

        if "Datetime" in df.columns:

            df = df.rename(
                columns={
                    "Datetime": "Date"
                }
            )

        if (
            "Date" not in df.columns
            or
            "Close" not in df.columns
        ):
            return None

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "Date",
                "Close"
            ]
        )

        df = df.sort_values(
            "Date"
        )

        df = df.drop_duplicates(
            subset=["Date"]
        )

        df = df.reset_index(
            drop=True
        )

        return df

    except Exception:
        return None


# =========================================================
# PARSE SYMBOLS
# =========================================================

def parse_symbols(
    text,
    current_symbol
):

    symbols = []

    if current_symbol:

        current_symbol = (
            current_symbol
            .strip()
            .upper()
        )

        if current_symbol:
            symbols.append(
                current_symbol
            )

    if text:

        entered = text.split(",")

        for symbol in entered:

            symbol = (
                symbol
                .strip()
                .upper()
            )

            if (
                symbol
                and
                symbol not in symbols
            ):

                symbols.append(
                    symbol
                )

    return symbols[:4]


# =========================================================
# VOLATILITY FACTOR
# =========================================================

def volatility_factor(period):

    if period in [
        "2y",
        "5y"
    ]:

        return np.sqrt(52)

    return np.sqrt(252)


# =========================================================
# BUILD COMPARISON DATA
# =========================================================

def build_comparison_data(
    symbols,
    period
):

    prices = {}

    stock_metrics = []

    failed = []

    factor = volatility_factor(
        period
    )

    for symbol in symbols:

        history = load_comparison_history(
            symbol,
            period
        )

        if (
            history is None
            or
            history.empty
        ):

            failed.append(
                symbol
            )

            continue

        close = history[
            [
                "Date",
                "Close"
            ]
        ].copy()

        close = close.rename(
            columns={
                "Close": symbol
            }
        )

        prices[
            symbol
        ] = close

        # =========================================
        # STOCK METRICS
        # =========================================

        first_price = float(
            history[
                "Close"
            ].iloc[0]
        )

        last_price = float(
            history[
                "Close"
            ].iloc[-1]
        )

        if first_price != 0:

            total_return = (
                (
                    last_price
                    - first_price
                )
                / first_price
            ) * 100

        else:

            total_return = 0

        returns = (
            history[
                "Close"
            ]
            .pct_change()
            .dropna()
        )

        if not returns.empty:

            volatility = (
                returns.std()
                * factor
                * 100
            )

            average_return = (
                returns.mean()
                * 100
            )

        else:

            volatility = 0

            average_return = 0

        highest = float(
            history[
                "Close"
            ].max()
        )

        lowest = float(
            history[
                "Close"
            ].min()
        )

        stock_metrics.append(
            {
                "Symbol": symbol,

                "Start Price (INR)":
                    first_price,

                "Current Price (INR)":
                    last_price,

                "Period Return (%)":
                    total_return,

                "Volatility (%)":
                    volatility,

                "Average Period Return (%)":
                    average_return,

                "Period High (INR)":
                    highest,

                "Period Low (INR)":
                    lowest,
            }
        )

    return (
        prices,
        stock_metrics,
        failed
    )


# =========================================================
# MERGE PRICE DATA
# =========================================================

def merge_prices(
    price_data
):

    merged = None

    for symbol, df in (
        price_data.items()
    ):

        if merged is None:

            merged = df.copy()

        else:

            merged = pd.merge(
                merged,
                df,
                on="Date",
                how="inner"
            )

    return merged


# =========================================================
# NORMALIZED PERFORMANCE
# =========================================================

def normalized_data(
    merged,
    symbols
):

    normalized = (
        merged[
            ["Date"] + symbols
        ]
        .copy()
    )

    for symbol in symbols:

        first_valid = (
            normalized[
                symbol
            ]
            .dropna()
        )

        if first_valid.empty:
            continue

        first_price = float(
            first_valid.iloc[0]
        )

        if first_price == 0:
            continue

        normalized[
            symbol
        ] = (
            normalized[
                symbol
            ]
            / first_price
        ) * 100

    return normalized


# =========================================================
# NORMALIZED PERFORMANCE GRAPH
# =========================================================

def show_performance_chart(
    normalized,
    symbols
):

    st.markdown(
        "### Normalized Performance"
    )

    st.caption(
        "Every stock starts at 100 so performance can be compared fairly."
    )

    fig = go.Figure()

    for symbol in symbols:

        fig.add_trace(
            go.Scatter(
                x=normalized[
                    "Date"
                ],

                y=normalized[
                    symbol
                ],

                mode="lines",

                name=symbol
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Performance Index",
        hovermode="x unified",
        height=520,
        legend_title="Stocks"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# INR PRICE GRAPH
# =========================================================

def show_price_chart(
    merged,
    symbols
):

    st.markdown(
        "### Stock Prices in INR"
    )

    fig = go.Figure()

    for symbol in symbols:

        fig.add_trace(
            go.Scatter(
                x=merged[
                    "Date"
                ],

                y=merged[
                    symbol
                ],

                mode="lines",

                name=symbol
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        height=520,
        legend_title="Stocks"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# CORRELATION
# =========================================================

def show_correlation(
    merged,
    symbols
):

    st.markdown(
        "### Return Correlation"
    )

    returns = (
        merged[
            symbols
        ]
        .pct_change()
        .dropna()
    )

    if (
        returns.empty
        or
        len(returns) < 2
    ):

        st.info(
            "Not enough matching trading data "
            "to calculate correlation."
        )

        return

    correlation = (
        returns.corr()
    )

    # =========================================
    # CORRELATION TABLE
    # =========================================

    st.dataframe(
        correlation.style.format(
            "{:.3f}"
        ),
        use_container_width=True
    )

    # =========================================
    # CORRELATION HEATMAP
    # =========================================

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,

            x=correlation.columns,

            y=correlation.index,

            zmin=-1,

            zmax=1,

            colorscale="RdBu",

            reversescale=True,

            text=np.round(
                correlation.values,
                2
            ),

            texttemplate="%{text}",

            hovertemplate=(
                "%{x} vs %{y}"
                "<br>Correlation: %{z:.3f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        height=450,
        xaxis_title="Stock",
        yaxis_title="Stock"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Correlation near +1 means stocks often move together. "
        "Near -1 means they often move in opposite directions. "
        "Near 0 means the relationship is weak."
    )


# =========================================================
# STOCK METRIC CARDS
# =========================================================

def show_stock_cards(
    metrics_df
):

    st.markdown(
        "### Comparison Snapshot"
    )

    columns = st.columns(
        len(metrics_df)
    )

    for index, row in (
        metrics_df
        .reset_index(drop=True)
        .iterrows()
    ):

        with columns[index]:

            st.markdown(
                f"#### {row['Symbol']}"
            )

            st.metric(
                "Current Price",
                (
                    f"₹"
                    f"{row['Current Price (INR)']:,.2f}"
                )
            )

            st.metric(
                "Period Return",
                (
                    f"{row['Period Return (%)']:.2f}%"
                )
            )

            st.metric(
                "Volatility",
                (
                    f"{row['Volatility (%)']:.2f}%"
                )
            )


# =========================================================
# BEST PERFORMER
# =========================================================

def show_best_performer(
    metrics_df
):

    if metrics_df.empty:
        return

    best_row = (
        metrics_df
        .sort_values(
            "Period Return (%)",
            ascending=False
        )
        .iloc[0]
    )

    lowest_volatility_row = (
        metrics_df
        .sort_values(
            "Volatility (%)",
            ascending=True
        )
        .iloc[0]
    )

    st.markdown(
        "### Comparison Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Best Performer",
            best_row[
                "Symbol"
            ],
            (
                f"{best_row['Period Return (%)']:.2f}%"
            )
        )

    with col2:

        st.metric(
            "Lowest Volatility",
            lowest_volatility_row[
                "Symbol"
            ],
            (
                f"{lowest_volatility_row['Volatility (%)']:.2f}%"
            )
        )

    if (
        best_row[
            "Period Return (%)"
        ] > 0
    ):

        st.success(
            f"{best_row['Symbol']} had the strongest "
            f"performance during the selected period "
            f"with a return of "
            f"{best_row['Period Return (%)']:.2f}%."
        )

    else:

        st.info(
            f"{best_row['Symbol']} performed best relative "
            f"to the other selected stocks, although its "
            f"period return was "
            f"{best_row['Period Return (%)']:.2f}%."
        )


# =========================================================
# FULL METRIC TABLE
# =========================================================

def show_metrics_table(
    metrics_df
):

    st.markdown(
        "### Detailed Comparison"
    )

    display_df = (
        metrics_df.copy()
    )

    st.dataframe(
        display_df.style.format(
            {
                "Start Price (INR)":
                    "₹{:,.2f}",

                "Current Price (INR)":
                    "₹{:,.2f}",

                "Period Return (%)":
                    "{:.2f}%",

                "Volatility (%)":
                    "{:.2f}%",

                "Average Period Return (%)":
                    "{:.3f}%",

                "Period High (INR)":
                    "₹{:,.2f}",

                "Period Low (INR)":
                    "₹{:,.2f}",
            }
        ),

        use_container_width=True,

        hide_index=True
    )


# =========================================================
# MAIN STOCK COMPARISON
# =========================================================

def stock_comparison(
    current_symbol=None
):

    st.subheader(
        "Stock Comparison and Correlation"
    )

    st.write(
        "Compare up to four Indian or global stocks."
    )

    # =====================================================
    # INPUTS
    # =====================================================

    default_examples = ""

    comparison_text = st.text_input(
        "Stocks to compare",

        value=default_examples,

        placeholder=(
            "Example: AAPL, MSFT, RELIANCE.NS"
        ),

        help=(
            "Your current stock is automatically included. "
            "Enter up to three additional ticker symbols "
            "separated by commas."
        ),

        key="comparison_stock_input"
    )

    period_name = st.selectbox(
        "Comparison Period",

        list(
            PERIODS.keys()
        ),

        index=2,

        key="comparison_period"
    )

    period = PERIODS[
        period_name
    ]

    # =====================================================
    # BUTTON
    # =====================================================

    if st.button(
        "Compare Stocks",
        use_container_width=True,
        key="run_stock_comparison"
    ):

        symbols = parse_symbols(
            comparison_text,
            current_symbol
        )

        st.session_state[
            "comparison_symbols"
        ] = symbols

        st.session_state[
            "comparison_selected_period"
        ] = period

        st.session_state[
            "comparison_selected_period_name"
        ] = period_name

        st.session_state[
            "comparison_ready"
        ] = True

    # =====================================================
    # WAIT FOR USER
    # =====================================================

    if not st.session_state.get(
        "comparison_ready",
        False
    ):

        st.info(
            "Enter at least one additional ticker and press Compare Stocks."
        )

        return

    symbols = st.session_state.get(
        "comparison_symbols",
        []
    )

    period = st.session_state.get(
        "comparison_selected_period",
        "1y"
    )

    period_name = st.session_state.get(
        "comparison_selected_period_name",
        "1 Year"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    if len(symbols) < 2:

        st.warning(
            "Please compare at least two stocks."
        )

        return

    if len(symbols) > 4:

        symbols = symbols[:4]

    # =====================================================
    # LOAD DATA
    # =====================================================

    with st.spinner(
        "Loading comparison data..."
    ):

        (
            price_data,
            stock_metrics,
            failed
        ) = build_comparison_data(
            symbols,
            period
        )

    if failed:

        st.warning(
            "Unable to load: "
            + ", ".join(
                failed
            )
        )

    valid_symbols = list(
        price_data.keys()
    )

    if len(valid_symbols) < 2:

        st.error(
            "At least two valid stocks are required."
        )

        return

    # =====================================================
    # MERGE
    # =====================================================

    merged = merge_prices(
        price_data
    )

    if (
        merged is None
        or
        merged.empty
    ):

        st.error(
            "The selected stocks do not have enough "
            "matching historical data."
        )

        return

    # =====================================================
    # METRICS
    # =====================================================

    metrics_df = pd.DataFrame(
        stock_metrics
    )

    metrics_df = metrics_df[
        metrics_df[
            "Symbol"
        ].isin(
            valid_symbols
        )
    ]

    st.caption(
        f"Comparison period: {period_name}. "
        f"Price values are displayed in Indian rupees."
    )

    # =====================================================
    # CARDS
    # =====================================================

    show_stock_cards(
        metrics_df
    )

    st.divider()

    # =====================================================
    # NORMALIZED PERFORMANCE
    # =====================================================

    normalized = normalized_data(
        merged,
        valid_symbols
    )

    show_performance_chart(
        normalized,
        valid_symbols
    )

    st.divider()

    # =====================================================
    # INR PRICE CHART
    # =====================================================

    show_price_chart(
        merged,
        valid_symbols
    )

    st.divider()

    # =====================================================
    # CORRELATION
    # =====================================================

    show_correlation(
        merged,
        valid_symbols
    )

    st.divider()

    # =====================================================
    # BEST STOCK
    # =====================================================

    show_best_performer(
        metrics_df
    )

    st.divider()

    # =====================================================
    # TABLE
    # =====================================================

    show_metrics_table(
        metrics_df
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv = (
        metrics_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "Download Comparison Report",
        data=csv,
        file_name=(
            "TEAM7_stock_comparison.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Comparison statistics are based on historical market "
        "data and are for educational purposes only."
    )