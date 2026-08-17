import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data.yahoo import get_stock_history


def safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def format_number(value, decimals=2):
    value = safe_float(value)
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_inr(value):
    value = safe_float(value)
    if value is None:
        return "N/A"
    return f"₹{value:,.2f}"


def clean_ai_history(history):
    if history is None or history.empty:
        return None

    df = history.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Date" not in df.columns:
        df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    if "Date" not in df.columns or "Close" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "Date",
            "Close",
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


@st.cache_data(ttl=1800, show_spinner=False)
def load_ai_history(symbol):
    try:
        history = get_stock_history(
            symbol,
            "1y"
        )

        return clean_ai_history(
            history
        )

    except Exception:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_indicators(history):
    df = history.copy()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"])

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def prepare_ml_data(history):
    data = history.copy()

    data["Lag1"] = data["Close"].shift(1)
    data["Lag2"] = data["Close"].shift(2)
    data["Lag3"] = data["Close"].shift(3)
    data["Lag5"] = data["Close"].shift(5)
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["Return"] = data["Close"].pct_change()
    data["Target"] = data["Close"].shift(-1)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    return data


def create_future_features(values):
    if len(values) < 10:
        return None

    previous = values[-2]
    daily_return = ((values[-1] - previous) / previous) if previous != 0 else 0

    return pd.DataFrame(
        [
            {
                "Lag1": values[-1],
                "Lag2": values[-2],
                "Lag3": values[-3],
                "Lag5": values[-5],
                "MA5": np.mean(values[-5:]),
                "MA10": np.mean(values[-10:]),
                "Return": daily_return,
            }
        ]
    )


def recursive_forecast(model, history, days=30):
    values = history["Close"].dropna().astype(float).tolist()
    predictions = []

    for _ in range(days):
        features = create_future_features(values)

        if features is None:
            break

        prediction = float(model.predict(features)[0])
        prediction = max(prediction, 0)

        predictions.append(prediction)
        values.append(prediction)

    return predictions


@st.cache_data(ttl=1800, show_spinner=False)
def analyze_stock_ai_from_history(history):
    history = clean_ai_history(
        history
    )

    if history is None or history.empty:
        return None

    technical = calculate_indicators(
        history
    )

    latest = technical.iloc[-1]

    result = {
        "price":
            safe_float(
                latest.get("Close")
            ),

        "ma20":
            safe_float(
                latest.get("MA20")
            ),

        "ma50":
            safe_float(
                latest.get("MA50")
            ),

        "rsi":
            safe_float(
                latest.get("RSI")
            ),

        "macd":
            safe_float(
                latest.get("MACD")
            ),

        "macd_signal":
            safe_float(
                latest.get("MACD_Signal")
            ),

        "best_model":
            "N/A",

        "forecast_price":
            None,

        "forecast_change":
            None,
    }

    ml_data = prepare_ml_data(
        history
    )

    if len(ml_data) < 35:
        return result

    features = [
        "Lag1",
        "Lag2",
        "Lag3",
        "Lag5",
        "MA5",
        "MA10",
        "Return",
    ]

    X = ml_data[
        features
    ]

    y = ml_data[
        "Target"
    ]

    split = int(
        len(ml_data)
        * 0.80
    )

    if split < 20:
        return result

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_train = y.iloc[
        :split
    ]

    y_test = y.iloc[
        split:
    ]

    if len(X_test) < 3:
        return result

    # Linear Regression
    linear = LinearRegression()

    linear.fit(
        X_train,
        y_train
    )

    linear_predictions = (
        linear.predict(
            X_test
        )
    )

    linear_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                linear_predictions
            )
        )
    )

    # Random Forest
    #
    # 100 trees is plenty for this small one-year dataset and is
    # noticeably faster than rebuilding 160 trees.
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    forest.fit(
        X_train,
        y_train
    )

    forest_predictions = (
        forest.predict(
            X_test
        )
    )

    forest_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                forest_predictions
            )
        )
    )

    if linear_rmse <= forest_rmse:
        best_model_name = (
            "Linear Regression"
        )

        final_model = (
            LinearRegression()
        )

    else:
        best_model_name = (
            "Random Forest"
        )

        final_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    final_model.fit(
        X,
        y
    )

    forecast = recursive_forecast(
        final_model,
        history,
        days=30
    )

    result[
        "best_model"
    ] = best_model_name

    if forecast:
        forecast_price = float(
            forecast[-1]
        )

        result[
            "forecast_price"
        ] = forecast_price

        current_price = result[
            "price"
        ]

        if (
            current_price is not None
            and current_price != 0
        ):
            result[
                "forecast_change"
            ] = (
                (
                    forecast_price
                    - current_price
                )
                / current_price
            ) * 100

    return result


@st.cache_data(ttl=1800, show_spinner=False)
def analyze_stock_ai(symbol):
    history = load_ai_history(
        symbol
    )

    if history is None or history.empty:
        return None

    return analyze_stock_ai_from_history(
        history
    )

def build_recommendation(info, analysis):
    price = safe_float(analysis.get("price"))
    ma20 = safe_float(analysis.get("ma20"))
    ma50 = safe_float(analysis.get("ma50"))
    rsi = safe_float(analysis.get("rsi"))
    macd = safe_float(analysis.get("macd"))
    macd_signal = safe_float(analysis.get("macd_signal"))
    forecast_change = safe_float(analysis.get("forecast_change"))

    pe = safe_float(info.get("pe_ratio"))
    eps = safe_float(info.get("eps"))

    score = 0.0
    bullish = []
    bearish = []
    neutral = []

    if price is not None and ma20 is not None:
        if price > ma20:
            score += 1
            bullish.append("Price is trading above the 20-day moving average.")
        else:
            score -= 1
            bearish.append("Price is trading below the 20-day moving average.")

    if ma20 is not None and ma50 is not None:
        if ma20 > ma50:
            score += 1.5
            bullish.append("Short-term trend is above the 50-day trend.")
        else:
            score -= 1.5
            bearish.append("Short-term trend is below the 50-day trend.")

    if rsi is not None:
        if rsi < 30:
            score += 1
            bullish.append("RSI indicates the stock may be oversold.")
        elif rsi > 70:
            score -= 1
            bearish.append("RSI indicates the stock may be overbought.")
        elif 45 <= rsi <= 65:
            score += 0.5
            bullish.append("RSI is in a healthy momentum range.")
        else:
            neutral.append("RSI is currently neutral.")

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            score += 1
            bullish.append("MACD is above its signal line.")
        else:
            score -= 1
            bearish.append("MACD is below its signal line.")

    if eps is not None:
        if eps > 0:
            score += 1
            bullish.append("The company currently has positive EPS.")
        elif eps < 0:
            score -= 1.5
            bearish.append("The company currently has negative EPS.")

    if pe is not None and pe > 0:
        if pe <= 20:
            score += 1
            bullish.append("P/E is in a relatively conservative valuation range.")
        elif pe <= 35:
            score += 0.5
            neutral.append("P/E is within a moderate valuation range.")
        elif pe > 50:
            score -= 1
            bearish.append("P/E is relatively high.")

    if forecast_change is not None:
        if forecast_change >= 8:
            score += 2.5
            bullish.append("ML model projects strong upside over the next 30 trading days.")
        elif forecast_change >= 3:
            score += 1.5
            bullish.append("ML model projects positive upside over the next 30 trading days.")
        elif forecast_change <= -8:
            score -= 2.5
            bearish.append("ML model projects notable downside over the next 30 trading days.")
        elif forecast_change <= -3:
            score -= 1.5
            bearish.append("ML model projects negative movement over the next 30 trading days.")
        else:
            neutral.append("ML forecast is close to the current price.")

    if score >= 3:
        recommendation = "BUY"
        status = "Bullish"
        theme = "buy"
    elif score <= -3:
        recommendation = "SELL"
        status = "Bearish"
        theme = "sell"
    else:
        recommendation = "HOLD"
        status = "Neutral"
        theme = "hold"

    confidence = 50 + (min(abs(score), 10) / 10) * 45
    confidence = min(round(confidence), 95)

    return {
        "recommendation": recommendation,
        "status": status,
        "theme": theme,
        "score": score,
        "confidence": confidence,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
    }


def inject_styles():
    st.markdown(
        """
<style>
.t7-ai-heading{color:#f7f9fc;font-size:.88rem;font-weight:800;letter-spacing:.075em;margin:6px 0 12px}
.t7-ai-main{position:relative;overflow:hidden;border-radius:15px;padding:22px 24px;background:radial-gradient(circle at 92% 20%,rgba(84,122,220,.12),transparent 32%),linear-gradient(145deg,rgba(13,23,40,.98),rgba(7,13,24,.98));border:1px solid rgba(125,151,198,.22);box-shadow:0 16px 38px rgba(0,0,0,.18)}
.t7-ai-main.buy{border-color:rgba(50,229,143,.34);box-shadow:inset 4px 0 0 rgba(50,229,143,.82),0 16px 38px rgba(0,0,0,.18)}
.t7-ai-main.sell{border-color:rgba(255,83,100,.34);box-shadow:inset 4px 0 0 rgba(255,83,100,.82),0 16px 38px rgba(0,0,0,.18)}
.t7-ai-main.hold{border-color:rgba(245,184,72,.34);box-shadow:inset 4px 0 0 rgba(245,184,72,.82),0 16px 38px rgba(0,0,0,.18)}
.t7-ai-top{display:flex;align-items:center;justify-content:space-between;gap:20px}
.t7-ai-left{display:flex;align-items:center;gap:16px}
.t7-ai-badge{width:58px;height:58px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#101d31;border:1px solid rgba(125,151,198,.22);color:#eef3fb;font-size:1.25rem;font-weight:850}
.t7-ai-kicker{color:#73839e;font-size:.68rem;font-weight:750;letter-spacing:.12em}
.t7-ai-action{color:#fff;font-size:2rem;line-height:1.05;font-weight:850;margin-top:4px}
.t7-ai-action.buy{color:#32e58f}.t7-ai-action.sell{color:#ff5364}.t7-ai-action.hold{color:#f5b848}
.t7-ai-status{color:#7888a3;font-size:.76rem;margin-top:4px}
.t7-ai-confidence{min-width:170px;text-align:right}
.t7-ai-confidence-label{color:#71809a;font-size:.68rem;text-transform:uppercase;letter-spacing:.08em}
.t7-ai-confidence-value{color:#fff;font-size:1.65rem;font-weight:800;margin-top:2px}
.t7-ai-bar{height:6px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.08);margin-top:8px}
.t7-ai-bar-fill{height:100%;border-radius:999px}.t7-ai-bar-fill.buy{background:#32e58f}.t7-ai-bar-fill.sell{background:#ff5364}.t7-ai-bar-fill.hold{background:#f5b848}
.t7-ai-summary{color:#7e8da7;font-size:.76rem;margin-top:17px}
.t7-ai-mini{min-height:92px;border-radius:12px;padding:14px 15px;margin-top:12px;background:linear-gradient(145deg,rgba(13,23,40,.96),rgba(8,14,25,.96));border:1px solid rgba(125,151,198,.18)}
.t7-ai-mini-label{color:#697992;font-size:.66rem;text-transform:uppercase;letter-spacing:.07em}
.t7-ai-mini-value{color:#f4f7fb;font-size:1.08rem;font-weight:750;margin-top:7px}
.t7-ai-mini-sub{color:#62718a;font-size:.68rem;margin-top:3px}
.t7-ai-signal-title{color:#eef3fb;font-size:.78rem;font-weight:800;letter-spacing:.06em;margin:17px 0 8px}
.t7-ai-signal{border-radius:10px;padding:10px 12px;margin-bottom:7px;color:#cdd7e8;font-size:.76rem;line-height:1.45;background:linear-gradient(145deg,rgba(12,21,37,.96),rgba(8,14,25,.96));border:1px solid rgba(125,151,198,.15)}
.t7-ai-signal.good{border-color:rgba(50,229,143,.24);box-shadow:inset 3px 0 0 rgba(50,229,143,.70)}
.t7-ai-signal.bad{border-color:rgba(255,83,100,.24);box-shadow:inset 3px 0 0 rgba(255,83,100,.70)}
.t7-ai-signal.neutral{border-color:rgba(245,184,72,.18);box-shadow:inset 3px 0 0 rgba(245,184,72,.50)}
.t7-ai-disclaimer{color:#5f6f88;font-size:.67rem;margin-top:13px}
@media(max-width:700px){.t7-ai-top{align-items:flex-start;flex-direction:column}.t7-ai-confidence{text-align:left;width:100%}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_signal_list(title, signals, css_class):
    st.markdown(
        f'<div class="t7-ai-signal-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if not signals:
        st.markdown(
            '<div class="t7-ai-signal neutral">No strong signal detected in this category.</div>',
            unsafe_allow_html=True,
        )
        return

    for signal in signals[:5]:
        st.markdown(
            f'<div class="t7-ai-signal {css_class}">{signal}</div>',
            unsafe_allow_html=True,
        )


def dashboard_ai(info):
    inject_styles()

    symbol = str(info.get("symbol", "")).strip().upper()

    if not symbol:
        st.warning("Unable to generate AI recommendation.")
        return

    st.markdown(
        '<div class="t7-ai-heading">AI INVESTMENT RECOMMENDATION</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Analyzing trend, momentum and AI outlook..."):
        analysis = analyze_stock_ai(symbol)

    if analysis is None:
        st.info("AI recommendation is temporarily unavailable for this stock.")
        return

    result = build_recommendation(info, analysis)

    recommendation = result["recommendation"]
    status = result["status"]
    theme = result["theme"]
    score = result["score"]
    confidence = result["confidence"]
    first_letter = recommendation[0] if recommendation else "AI"

    st.markdown(
        f'''<div class="t7-ai-main {theme}">
<div class="t7-ai-top">
<div class="t7-ai-left">
<div class="t7-ai-badge">{first_letter}</div>
<div>
<div class="t7-ai-kicker">TEAM7 AI SIGNAL</div>
<div class="t7-ai-action {theme}">{recommendation}</div>
<div class="t7-ai-status">{status} market setup</div>
</div>
</div>
<div class="t7-ai-confidence">
<div class="t7-ai-confidence-label">Confidence</div>
<div class="t7-ai-confidence-value">{confidence}%</div>
<div class="t7-ai-bar"><div class="t7-ai-bar-fill {theme}" style="width:{confidence}%;"></div></div>
</div>
</div>
<div class="t7-ai-summary">Combined signal from price trend, momentum indicators, company fundamentals and a 30-day machine-learning outlook.</div>
</div>''',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    forecast_change = safe_float(analysis.get("forecast_change"))
    forecast_price = safe_float(analysis.get("forecast_price"))
    outlook_text = "N/A" if forecast_change is None else f"{forecast_change:+.2f}%"

    cards = [
        (c1, "AI SCORE", f"{score:+.1f}", "Composite signal"),
        (c2, "30-DAY OUTLOOK", outlook_text, "Expected change"),
        (c3, "BEST MODEL", analysis.get("best_model", "N/A"), "Lowest test RMSE"),
        (c4, "FORECAST PRICE", format_inr(forecast_price), "30 trading days"),
    ]

    for column, label, value, subtitle in cards:
        with column:
            st.markdown(
                f'''<div class="t7-ai-mini">
<div class="t7-ai-mini-label">{label}</div>
<div class="t7-ai-mini-value">{value}</div>
<div class="t7-ai-mini-sub">{subtitle}</div>
</div>''',
                unsafe_allow_html=True,
            )

    t1, t2, t3, t4 = st.columns(4, gap="medium")

    technical_cards = [
        (t1, "RSI", format_number(analysis.get("rsi")), "Momentum"),
        (t2, "MA20", format_inr(analysis.get("ma20")), "Short-term trend"),
        (t3, "MA50", format_inr(analysis.get("ma50")), "Medium-term trend"),
        (t4, "MACD", format_number(analysis.get("macd")), "Trend momentum"),
    ]

    for column, label, value, subtitle in technical_cards:
        with column:
            st.markdown(
                f'''<div class="t7-ai-mini">
<div class="t7-ai-mini-label">{label}</div>
<div class="t7-ai-mini-value">{value}</div>
<div class="t7-ai-mini-sub">{subtitle}</div>
</div>''',
                unsafe_allow_html=True,
            )

    left, right = st.columns(2, gap="large")

    with left:
        render_signal_list("BULLISH SIGNALS", result["bullish"], "good")

    with right:
        render_signal_list("RISK SIGNALS", result["bearish"], "bad")

    if result["neutral"]:
        render_signal_list("NEUTRAL OBSERVATIONS", result["neutral"], "neutral")

    st.markdown(
        '<div class="t7-ai-disclaimer">TEAM7 AI signals are generated from historical market data and financial metrics for educational analysis only. They are not financial advice.</div>',
        unsafe_allow_html=True,
    )
