import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_searchbox import st_searchbox

from data.company_search import search_company_options
from data.yahoo import get_stock_history, get_stock_info


PERIODS = {
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}


COMPARE_SEARCH_STYLE = {
    "dropdown": {
        "rotate": True,
        "width": 22,
        "height": 22,
        "fill": "#7f8da8",
    },
    "clear": {
        "width": 17,
        "height": 17,
        "icon": "cross",
        "clearable": "always",
    },
    "searchbox": {
        "control": {
            "backgroundColor": "#111521",
            "border": "1px solid #2b3445",
            "borderRadius": "9px",
            "minHeight": "42px",
        },
        "menu": {
            "backgroundColor": "#08111f",
            "border": "1px solid #223454",
            "borderRadius": "10px",
            "overflow": "hidden",
        },
        "menuList": {"backgroundColor": "#08111f"},
        "option": {
            "backgroundColor": "#08111f",
            "color": "#e8eef9",
            "highlightColor": "#7aa2ff",
        },
        "placeholder": {"color": "#75829b"},
        "input": {"color": "#f8fafc"},
        "singleValue": {"color": "#f8fafc"},
    },
}


@st.cache_data(ttl=900, show_spinner=False)
def load_comparison_history(symbol, period):
    try:
        df = get_stock_history(symbol, period)
        if df is None or df.empty:
            return None

        df = df.copy()
        if "Date" not in df.columns:
            df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        if "Date" not in df.columns or "Close" not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        df = df.sort_values("Date").drop_duplicates(subset=["Date"])
        return df.reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def company_name_for_symbol(symbol):
    try:
        info = get_stock_info(symbol)
        if info:
            name = str(info.get("name") or "").strip()
            if name:
                return name
    except Exception:
        pass
    return str(symbol)


def unique_symbols(current_symbol, selected_symbols):
    symbols = []

    for value in [current_symbol] + list(selected_symbols):
        symbol = str(value or "").strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)

    return symbols[:4]


def volatility_factor(period):
    return np.sqrt(52) if period in {"2y", "5y"} else np.sqrt(252)


def build_comparison_data(symbols, period):
    prices = {}
    stock_metrics = []
    failed = []
    factor = volatility_factor(period)

    for symbol in symbols:
        history = load_comparison_history(symbol, period)

        if history is None or history.empty:
            failed.append(symbol)
            continue

        company = company_name_for_symbol(symbol)

        close = history[["Date", "Close"]].copy()
        close = close.rename(columns={"Close": symbol})
        prices[symbol] = close

        first_price = float(history["Close"].iloc[0])
        last_price = float(history["Close"].iloc[-1])
        total_return = (
            ((last_price - first_price) / first_price) * 100
            if first_price != 0
            else 0.0
        )

        returns = history["Close"].pct_change().dropna()
        if not returns.empty:
            volatility = float(returns.std() * factor * 100)
            average_return = float(returns.mean() * 100)
        else:
            volatility = 0.0
            average_return = 0.0

        stock_metrics.append(
            {
                "Company": company,
                "Symbol": symbol,
                "Start Price (INR)": first_price,
                "Current Price (INR)": last_price,
                "Period Return (%)": total_return,
                "Volatility (%)": volatility,
                "Average Period Return (%)": average_return,
                "Period High (INR)": float(history["Close"].max()),
                "Period Low (INR)": float(history["Close"].min()),
            }
        )

    return prices, stock_metrics, failed


def merge_prices(price_data):
    merged = None

    for _, df in price_data.items():
        if merged is None:
            merged = df.copy()
        else:
            merged = pd.merge(merged, df, on="Date", how="inner")

    return merged


def normalized_data(merged, symbols):
    normalized = merged[["Date"] + symbols].copy()

    for symbol in symbols:
        valid = normalized[symbol].dropna()
        if valid.empty:
            continue
        first_price = float(valid.iloc[0])
        if first_price != 0:
            normalized[symbol] = normalized[symbol] / first_price * 100

    return normalized


def display_name_map(metrics_df):
    return {
        str(row["Symbol"]): str(row["Company"])
        for _, row in metrics_df.iterrows()
    }


def show_stock_cards(metrics_df):
    st.markdown("### Comparison Snapshot")
    columns = st.columns(len(metrics_df))

    for column, (_, row) in zip(columns, metrics_df.reset_index(drop=True).iterrows()):
        with column:
            st.markdown(f"#### {row['Company']}")
            st.caption(str(row["Symbol"]))
            st.metric("Current Price", f"₹{row['Current Price (INR)']:,.2f}")
            st.metric("Period Return", f"{row['Period Return (%)']:.2f}%")
            st.metric("Volatility", f"{row['Volatility (%)']:.2f}%")


def show_performance_chart(normalized, symbols, names):
    st.markdown("### Normalized Performance")
    st.caption("Every company starts at 100 so performance can be compared fairly.")

    fig = go.Figure()
    for symbol in symbols:
        fig.add_trace(
            go.Scatter(
                x=normalized["Date"],
                y=normalized[symbol],
                mode="lines",
                name=names.get(symbol, symbol),
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Performance Index",
        hovermode="x unified",
        height=520,
        legend_title="Companies",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_price_chart(merged, symbols, names):
    st.markdown("### Stock Prices in INR")

    fig = go.Figure()
    for symbol in symbols:
        fig.add_trace(
            go.Scatter(
                x=merged["Date"],
                y=merged[symbol],
                mode="lines",
                name=names.get(symbol, symbol),
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        height=520,
        legend_title="Companies",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_correlation(merged, symbols, names):
    st.markdown("### Return Correlation")

    returns = merged[symbols].pct_change().dropna()
    if returns.empty or len(returns) < 2:
        st.info("Not enough matching trading data to calculate correlation.")
        return

    correlation = returns.corr()
    label_map = {symbol: names.get(symbol, symbol) for symbol in symbols}
    display_correlation = correlation.rename(index=label_map, columns=label_map)

    st.dataframe(
        display_correlation.style.format("{:.3f}"),
        use_container_width=True,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=display_correlation.values,
            x=display_correlation.columns,
            y=display_correlation.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(display_correlation.values, 2),
            texttemplate="%{text}",
            hovertemplate=(
                "%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(height=450, xaxis_title="Company", yaxis_title="Company")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Correlation near +1 means the stocks often move together; near -1 means "
        "they often move in opposite directions; near 0 means the relationship is weak."
    )


def show_best_performer(metrics_df):
    if metrics_df.empty:
        return

    best = metrics_df.sort_values("Period Return (%)", ascending=False).iloc[0]
    low_vol = metrics_df.sort_values("Volatility (%)", ascending=True).iloc[0]

    st.markdown("### Comparison Result")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Best Performer",
            best["Company"],
            f"{best['Period Return (%)']:.2f}%",
        )

    with col2:
        st.metric(
            "Lowest Volatility",
            low_vol["Company"],
            f"{low_vol['Volatility (%)']:.2f}%",
        )

    if best["Period Return (%)"] > 0:
        st.success(
            f"{best['Company']} had the strongest performance during the selected "
            f"period with a return of {best['Period Return (%)']:.2f}%."
        )
    else:
        st.info(
            f"{best['Company']} performed best relative to the other selected companies, "
            f"although its period return was {best['Period Return (%)']:.2f}%."
        )


def show_metrics_table(metrics_df):
    st.markdown("### Detailed Comparison")

    st.dataframe(
        metrics_df.style.format(
            {
                "Start Price (INR)": "₹{:,.2f}",
                "Current Price (INR)": "₹{:,.2f}",
                "Period Return (%)": "{:.2f}%",
                "Volatility (%)": "{:.2f}%",
                "Average Period Return (%)": "{:.3f}%",
                "Period High (INR)": "₹{:,.2f}",
                "Period Low (INR)": "₹{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def _company_selector(key, placeholder):
    return st_searchbox(
        search_company_options,
        key=key,
        placeholder=placeholder,
        default=None,
        clear_on_submit=False,
        rerun_on_update=True,
        debounce=450,
        edit_after_submit="option",
        style_overrides=COMPARE_SEARCH_STYLE,
    )


def stock_comparison(current_symbol=None):
    st.subheader("Stock Comparison and Correlation")

    st.write(
        "Compare up to four publicly listed company stocks from India or global markets."
    )

    st.info(
        "Supported: listed company equities/shares available through Yahoo Finance — "
        "for example NSE/BSE, NASDAQ/NYSE, LSE, TSX, ASX, HKEX and other major exchanges. "
        "This comparison is designed for company stocks, not mutual funds, ETFs, market "
        "indices, forex or cryptocurrencies."
    )

    current_symbol = str(current_symbol or "").strip().upper()
    if current_symbol:
        current_name = company_name_for_symbol(current_symbol)
        st.caption(f"Current company included automatically: {current_name}")

    st.markdown("**Choose up to three additional companies by name**")

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        st.caption("Additional company 1")
        stock_1 = _company_selector(
            "comparison_company_1",
            "Search company name...",
        )

    with col2:
        st.caption("Additional company 2")
        stock_2 = _company_selector(
            "comparison_company_2",
            "Search company name...",
        )

    with col3:
        st.caption("Additional company 3")
        stock_3 = _company_selector(
            "comparison_company_3",
            "Search company name...",
        )

    period_name = st.selectbox(
        "Comparison Period",
        list(PERIODS.keys()),
        index=2,
        key="comparison_period",
    )
    period = PERIODS[period_name]

    if st.button(
        "Compare Stocks",
        use_container_width=True,
        key="run_stock_comparison",
    ):
        symbols = unique_symbols(
            current_symbol,
            [stock_1, stock_2, stock_3],
        )

        st.session_state["comparison_symbols"] = symbols
        st.session_state["comparison_selected_period"] = period
        st.session_state["comparison_selected_period_name"] = period_name
        st.session_state["comparison_ready"] = True

    if not st.session_state.get("comparison_ready", False):
        st.info("Choose at least one additional company and press Compare Stocks.")
        return

    symbols = st.session_state.get("comparison_symbols", [])
    period = st.session_state.get("comparison_selected_period", "1y")
    period_name = st.session_state.get("comparison_selected_period_name", "1 Year")

    if len(symbols) < 2:
        st.warning("Please choose at least one additional company to compare.")
        return

    with st.spinner("Loading comparison data..."):
        price_data, stock_metrics, failed = build_comparison_data(symbols, period)

    if failed:
        failed_names = [company_name_for_symbol(symbol) for symbol in failed]
        st.warning("Unable to load: " + ", ".join(failed_names))

    valid_symbols = list(price_data.keys())
    if len(valid_symbols) < 2:
        st.error("At least two valid company stocks are required for comparison.")
        return

    merged = merge_prices(price_data)
    if merged is None or merged.empty:
        st.error(
            "The selected companies do not have enough matching historical data."
        )
        return

    metrics_df = pd.DataFrame(stock_metrics)
    metrics_df = metrics_df[metrics_df["Symbol"].isin(valid_symbols)]
    names = display_name_map(metrics_df)

    st.caption(
        f"Comparison period: {period_name}. Price values are displayed in Indian rupees."
    )

    show_stock_cards(metrics_df)
    st.divider()

    normalized = normalized_data(merged, valid_symbols)
    show_performance_chart(normalized, valid_symbols, names)
    st.divider()

    show_price_chart(merged, valid_symbols, names)
    st.divider()

    show_correlation(merged, valid_symbols, names)
    st.divider()

    show_best_performer(metrics_df)
    st.divider()

    show_metrics_table(metrics_df)

    csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Comparison Report",
        data=csv,
        file_name="NixieStocks_stock_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Comparison statistics are based on historical market data and are for educational purposes only."
    )
