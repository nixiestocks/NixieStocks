import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =========================================================
# OPTIONAL TENSORFLOW / LSTM
# =========================================================

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True

except ImportError:
    TENSORFLOW_AVAILABLE = False


# =========================================================
# SAFE VALUES
# =========================================================

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


def display_number(value, decimals=2):

    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:,.{decimals}f}"


# =========================================================
# CLEAN HISTORY
# =========================================================

def clean_history(history):

    if history is None or history.empty:
        return None

    df = history.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Date" not in df.columns:
        df = df.reset_index()

    if "Date" not in df.columns:
        return None

    if "Close" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
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
                errors="coerce",
            )

    df = df.dropna(
        subset=["Date", "Close"],
    )

    df = df.sort_values(
        "Date",
    )

    df = df.drop_duplicates(
        subset=["Date"],
    )

    df = df.reset_index(
        drop=True,
    )

    return df


# =========================================================
# RSI
# =========================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0,
    )

    loss = -delta.clip(
        upper=0,
    )

    avg_gain = gain.rolling(
        period,
    ).mean()

    avg_loss = loss.rolling(
        period,
    ).mean()

    avg_loss = avg_loss.replace(
        0,
        np.nan,
    )

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def calculate_indicators(df):

    data = df.copy()

    data["MA20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["MA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    data["MA100"] = (
        data["Close"]
        .rolling(100)
        .mean()
    )

    data["RSI"] = calculate_rsi(
        data["Close"],
    )

    ema12 = (
        data["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        data["Close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    data["MACD"] = (
        ema12 - ema26
    )

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    data["Daily_Return"] = (
        data["Close"]
        .pct_change()
    )

    data["Volatility"] = (
        data["Daily_Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    return data


# =========================================================
# MODEL METRICS
# =========================================================

def calculate_metrics(actual, predicted):

    actual = np.array(
        actual,
    ).reshape(-1)

    predicted = np.array(
        predicted,
    ).reshape(-1)

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    mse = mean_squared_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mse,
    )

    try:

        r2 = r2_score(
            actual,
            predicted,
        )

    except Exception:

        r2 = 0

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


# =========================================================
# CLASSICAL ML DATA
# =========================================================

def prepare_ml_data(df):

    data = df.copy()

    data["Lag1"] = (
        data["Close"].shift(1)
    )

    data["Lag2"] = (
        data["Close"].shift(2)
    )

    data["Lag3"] = (
        data["Close"].shift(3)
    )

    data["Lag5"] = (
        data["Close"].shift(5)
    )

    data["MA5"] = (
        data["Close"]
        .rolling(5)
        .mean()
    )

    data["MA10"] = (
        data["Close"]
        .rolling(10)
        .mean()
    )

    data["Return"] = (
        data["Close"]
        .pct_change()
    )

    data["Target"] = (
        data["Close"]
        .shift(-1)
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna()

    return data


# =========================================================
# FUTURE FEATURES
# =========================================================

def create_future_features(values):

    if len(values) < 10:
        return None

    previous = values[-2]

    if previous != 0:

        daily_return = (
            values[-1]
            - previous
        ) / previous

    else:

        daily_return = 0

    return pd.DataFrame(
        [
            {
                "Lag1": values[-1],
                "Lag2": values[-2],
                "Lag3": values[-3],
                "Lag5": values[-5],
                "MA5": np.mean(
                    values[-5:]
                ),
                "MA10": np.mean(
                    values[-10:]
                ),
                "Return": daily_return,
            }
        ]
    )


# =========================================================
# CLASSICAL FORECAST
# =========================================================

def classical_forecast(
    model,
    history,
    days=30,
):

    values = (
        history["Close"]
        .dropna()
        .astype(float)
        .tolist()
    )

    predictions = []

    for _ in range(days):

        features = create_future_features(
            values,
        )

        if features is None:
            break

        prediction = float(
            model.predict(
                features,
            )[0]
        )

        prediction = max(
            prediction,
            0,
        )

        predictions.append(
            prediction,
        )

        values.append(
            prediction,
        )

    return predictions


# =========================================================
# LSTM MODEL
# =========================================================

def build_lstm_model(sequence_length):

    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    1,
                )
            ),

            LSTM(
                64,
                return_sequences=True,
            ),

            Dropout(
                0.20,
            ),

            LSTM(
                32,
            ),

            Dropout(
                0.20,
            ),

            Dense(
                16,
                activation="relu",
            ),

            Dense(
                1,
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    return model


# =========================================================
# ANALYZE MODELS
# =========================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def analyze_models(df):

    result = {
        "models": [],
        "best_model": "N/A",
        "best_rmse": None,
        "forecast": [],
    }

    # Need enough history
    if df is None or len(df) < 40:
        return result

    # =====================================================
    # LINEAR + RANDOM FOREST
    # =====================================================

    ml_data = prepare_ml_data(
        df,
    )

    if len(ml_data) >= 30:

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

        # =====================================
        # LINEAR REGRESSION
        # =====================================

        linear = LinearRegression()

        linear.fit(
            X_train,
            y_train,
        )

        linear_predictions = (
            linear.predict(
                X_test,
            )
        )

        linear_metrics = calculate_metrics(
            y_test,
            linear_predictions,
        )

        result["models"].append(
            {
                "Model": "Linear Regression",
                "MAE": linear_metrics["MAE"],
                "MSE": linear_metrics["MSE"],
                "RMSE": linear_metrics["RMSE"],
                "R² Score": linear_metrics["R2"],
            }
        )

        # =====================================
        # RANDOM FOREST
        # =====================================

        forest = RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        forest.fit(
            X_train,
            y_train,
        )

        forest_predictions = (
            forest.predict(
                X_test,
            )
        )

        forest_metrics = calculate_metrics(
            y_test,
            forest_predictions,
        )

        result["models"].append(
            {
                "Model": "Random Forest",
                "MAE": forest_metrics["MAE"],
                "MSE": forest_metrics["MSE"],
                "RMSE": forest_metrics["RMSE"],
                "R² Score": forest_metrics["R2"],
            }
        )

    # =====================================================
    # LSTM
    # =====================================================

    lstm_model = None
    lstm_scaler = None

    sequence_length = 60

    if (
        TENSORFLOW_AVAILABLE
        and
        len(df) >= 120
    ):

        try:

            values = (
                df["Close"]
                .values
                .reshape(-1, 1)
            )

            split_index = int(
                len(values)
                * 0.80
            )

            scaler = MinMaxScaler(
                feature_range=(0, 1),
            )

            scaler.fit(
                values[:split_index],
            )

            scaled = scaler.transform(
                values,
            )

            X_train = []
            y_train = []

            X_test = []
            y_test = []

            for i in range(
                sequence_length,
                len(scaled),
            ):

                sequence = scaled[
                    i - sequence_length:i
                ]

                target = scaled[
                    i
                ]

                if i < split_index:

                    X_train.append(
                        sequence,
                    )

                    y_train.append(
                        target,
                    )

                else:

                    X_test.append(
                        sequence,
                    )

                    y_test.append(
                        target,
                    )

            X_train = np.array(
                X_train,
            )

            y_train = np.array(
                y_train,
            )

            X_test = np.array(
                X_test,
            )

            y_test = np.array(
                y_test,
            )

            if (
                len(X_train) >= 20
                and
                len(X_test) >= 5
            ):

                lstm_model = build_lstm_model(
                    sequence_length,
                )

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                )

                lstm_model.fit(
                    X_train,
                    y_train,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.15,
                    callbacks=[
                        early_stop,
                    ],
                    verbose=0,
                )

                predicted_scaled = (
                    lstm_model.predict(
                        X_test,
                        verbose=0,
                    )
                )

                predicted = (
                    scaler
                    .inverse_transform(
                        predicted_scaled,
                    )
                    .reshape(-1)
                )

                actual = (
                    scaler
                    .inverse_transform(
                        y_test,
                    )
                    .reshape(-1)
                )

                lstm_metrics = calculate_metrics(
                    actual,
                    predicted,
                )

                result["models"].append(
                    {
                        "Model": "LSTM Deep Learning",
                        "MAE": lstm_metrics["MAE"],
                        "MSE": lstm_metrics["MSE"],
                        "RMSE": lstm_metrics["RMSE"],
                        "R² Score": lstm_metrics["R2"],
                    }
                )

                lstm_scaler = scaler

        except Exception:

            lstm_model = None
            lstm_scaler = None

    # =====================================================
    # SELECT BEST MODEL
    # =====================================================

    if not result["models"]:
        return result

    best = min(
        result["models"],
        key=lambda item: item["RMSE"],
    )

    result["best_model"] = (
        best["Model"]
    )

    result["best_rmse"] = (
        best["RMSE"]
    )

    # =====================================================
    # 30-DAY FORECAST
    # =====================================================

    if result["best_model"] == "Linear Regression":

        data = prepare_ml_data(
            df,
        )

        features = [
            "Lag1",
            "Lag2",
            "Lag3",
            "Lag5",
            "MA5",
            "MA10",
            "Return",
        ]

        model = LinearRegression()

        model.fit(
            data[features],
            data["Target"],
        )

        result["forecast"] = (
            classical_forecast(
                model,
                df,
                days=30,
            )
        )

    elif result["best_model"] == "Random Forest":

        data = prepare_ml_data(
            df,
        )

        features = [
            "Lag1",
            "Lag2",
            "Lag3",
            "Lag5",
            "MA5",
            "MA10",
            "Return",
        ]

        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            data[features],
            data["Target"],
        )

        result["forecast"] = (
            classical_forecast(
                model,
                df,
                days=30,
            )
        )

    elif (
        result["best_model"]
        == "LSTM Deep Learning"
        and
        TENSORFLOW_AVAILABLE
    ):

        try:

            values = (
                df["Close"]
                .values
                .reshape(-1, 1)
            )

            scaler = MinMaxScaler(
                feature_range=(0, 1),
            )

            scaled = scaler.fit_transform(
                values,
            )

            X_full = []
            y_full = []

            for i in range(
                sequence_length,
                len(scaled),
            ):

                X_full.append(
                    scaled[
                        i - sequence_length:i
                    ]
                )

                y_full.append(
                    scaled[i]
                )

            X_full = np.array(
                X_full,
            )

            y_full = np.array(
                y_full,
            )

            final_model = build_lstm_model(
                sequence_length,
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )

            final_model.fit(
                X_full,
                y_full,
                epochs=30,
                batch_size=32,
                validation_split=0.15,
                callbacks=[
                    early_stop,
                ],
                verbose=0,
            )

            sequence = (
                scaled[
                    -sequence_length:
                ]
                .reshape(
                    1,
                    sequence_length,
                    1,
                )
            )

            forecast = []

            for _ in range(30):

                predicted_scaled = float(
                    final_model.predict(
                        sequence,
                        verbose=0,
                    )[0][0]
                )

                predicted_price = float(
                    scaler.inverse_transform(
                        np.array(
                            [[predicted_scaled]]
                        )
                    )[0][0]
                )

                predicted_price = max(
                    predicted_price,
                    0,
                )

                forecast.append(
                    predicted_price,
                )

                new_value = np.array(
                    [
                        [
                            [
                                predicted_scaled
                            ]
                        ]
                    ]
                )

                sequence = np.concatenate(
                    (
                        sequence[:, 1:, :],
                        new_value,
                    ),
                    axis=1,
                )

            result["forecast"] = forecast

        except Exception:

            result["forecast"] = []

    return result


# =========================================================
# AI RECOMMENDATION
# =========================================================

def create_ai_recommendation(
    info,
    technical,
    forecast_change,
):

    latest = technical.iloc[-1]

    price = safe_float(
        latest.get("Close")
    )

    ma20 = safe_float(
        latest.get("MA20")
    )

    ma50 = safe_float(
        latest.get("MA50")
    )

    rsi = safe_float(
        latest.get("RSI")
    )

    macd = safe_float(
        latest.get("MACD")
    )

    macd_signal = safe_float(
        latest.get("MACD_Signal")
    )

    pe = safe_float(
        info.get("pe_ratio")
    )

    eps = safe_float(
        info.get("eps")
    )

    score = 0.0

    # =====================================================
    # PRICE VS MA20
    # =====================================================

    if (
        price is not None
        and
        ma20 is not None
    ):

        if price > ma20:
            score += 1
        else:
            score -= 1

    # =====================================================
    # MA20 VS MA50
    # =====================================================

    if (
        ma20 is not None
        and
        ma50 is not None
    ):

        if ma20 > ma50:
            score += 1.5
        else:
            score -= 1.5

    # =====================================================
    # RSI
    # =====================================================

    if rsi is not None:

        if rsi < 30:
            score += 1

        elif rsi > 70:
            score -= 1

        elif 45 <= rsi <= 65:
            score += 0.5

    # =====================================================
    # MACD
    # =====================================================

    if (
        macd is not None
        and
        macd_signal is not None
    ):

        if macd > macd_signal:
            score += 1

        else:
            score -= 1

    # =====================================================
    # EPS
    # =====================================================

    if eps is not None:

        if eps > 0:
            score += 1

        elif eps < 0:
            score -= 1.5

    # =====================================================
    # PE
    # =====================================================

    if pe is not None and pe > 0:

        if pe <= 20:
            score += 1

        elif pe <= 35:
            score += 0.5

        elif pe > 50:
            score -= 1

    # =====================================================
    # AI FORECAST
    # =====================================================

    if forecast_change is not None:

        if forecast_change >= 8:
            score += 2.5

        elif forecast_change >= 3:
            score += 1.5

        elif forecast_change <= -8:
            score -= 2.5

        elif forecast_change <= -3:
            score -= 1.5

    # =====================================================
    # FINAL RESULT
    # =====================================================

    if score >= 3:
        recommendation = "BUY"

    elif score <= -3:
        recommendation = "SELL"

    else:
        recommendation = "HOLD"

    confidence = (
        50
        +
        (
            min(
                abs(score),
                10,
            )
            / 10
        )
        * 45
    )

    confidence = min(
        round(confidence),
        95,
    )

    return (
        recommendation,
        confidence,
        score,
    )


# =========================================================
# HTML REPORT
# =========================================================

def create_html_report(
    info,
    technical,
    model_results,
    recommendation,
    confidence,
    ai_score,
    forecast_price,
    forecast_change,
):

    latest = technical.iloc[-1]

    symbol = info.get(
        "symbol",
        "N/A",
    )

    company = info.get(
        "name",
        "N/A",
    )

    model_table = ""

    for model in model_results["models"]:

        model_table += f"""
        <tr>
            <td>{model["Model"]}</td>
            <td>{model["MAE"]:.2f}</td>
            <td>{model["MSE"]:.2f}</td>
            <td>{model["RMSE"]:.2f}</td>
            <td>{model["R² Score"]:.4f}</td>
        </tr>
        """

    recent_history = technical.tail(
        30
    )

    history_rows = ""

    for _, row in recent_history.iterrows():

        history_rows += f"""
        <tr>
            <td>{row["Date"].strftime("%Y-%m-%d")}</td>
            <td>{display_number(row.get("Open"))}</td>
            <td>{display_number(row.get("High"))}</td>
            <td>{display_number(row.get("Low"))}</td>
            <td>{display_number(row.get("Close"))}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <meta charset="UTF-8">

        <title>{symbol} Stock Analysis Report</title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                background: #0e1117;
                color: #ffffff;
                padding: 40px;
            }}

            h1 {{
                margin-bottom: 5px;
            }}

            h2 {{
                margin-top: 35px;
                border-bottom: 1px solid #444;
                padding-bottom: 8px;
            }}

            .card {{
                background: #161b22;
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}

            th,
            td {{
                padding: 10px;
                border: 1px solid #333;
                text-align: left;
            }}

            th {{
                background: #21262d;
            }}

            .disclaimer {{
                margin-top: 40px;
                font-size: 12px;
                color: #aaaaaa;
            }}

        </style>

    </head>

    <body>

        <h1>TEAM7 Stock Analysis Report</h1>

        <p>
            {company} ({symbol})
        </p>

        <h2>Company Information</h2>

        <div class="card">

            <p>
                <strong>Company:</strong>
                {company}
            </p>

            <p>
                <strong>Symbol:</strong>
                {symbol}
            </p>

            <p>
                <strong>Sector:</strong>
                {info.get("sector", "N/A")}
            </p>

            <p>
                <strong>Industry:</strong>
                {info.get("industry", "N/A")}
            </p>

            <p>
                <strong>Country:</strong>
                {info.get("country", "N/A")}
            </p>

        </div>

        <h2>Market Information</h2>

        <div class="card">

            <p>
                <strong>Current Price:</strong>
                {display_number(info.get("price"))}
                {info.get("currency", "")}
            </p>

            <p>
                <strong>Market Cap:</strong>
                {info.get("market_cap", "N/A")}
            </p>

            <p>
                <strong>P/E Ratio:</strong>
                {display_number(info.get("pe_ratio"))}
            </p>

            <p>
                <strong>EPS:</strong>
                {display_number(info.get("eps"))}
            </p>

            <p>
                <strong>52 Week High:</strong>
                {display_number(info.get("high_52"))}
            </p>

            <p>
                <strong>52 Week Low:</strong>
                {display_number(info.get("low_52"))}
            </p>

        </div>

        <h2>Technical Analysis</h2>

        <table>

            <tr>
                <th>Indicator</th>
                <th>Value</th>
            </tr>

            <tr>
                <td>RSI</td>
                <td>{display_number(latest.get("RSI"))}</td>
            </tr>

            <tr>
                <td>MA20</td>
                <td>{display_number(latest.get("MA20"))}</td>
            </tr>

            <tr>
                <td>MA50</td>
                <td>{display_number(latest.get("MA50"))}</td>
            </tr>

            <tr>
                <td>MA100</td>
                <td>{display_number(latest.get("MA100"))}</td>
            </tr>

            <tr>
                <td>MACD</td>
                <td>{display_number(latest.get("MACD"))}</td>
            </tr>

            <tr>
                <td>MACD Signal</td>
                <td>{display_number(latest.get("MACD_Signal"))}</td>
            </tr>

            <tr>
                <td>Annualized Volatility</td>
                <td>{display_number(latest.get("Volatility"))}%</td>
            </tr>

        </table>

        <h2>AI Investment Recommendation</h2>

        <div class="card">

            <h3>{recommendation}</h3>

            <p>
                <strong>Confidence:</strong>
                {confidence}%
            </p>

            <p>
                <strong>AI Score:</strong>
                {ai_score:.1f}
            </p>

        </div>

        <h2>Machine Learning Performance</h2>

        <table>

            <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>MSE</th>
                <th>RMSE</th>
                <th>R²</th>
            </tr>

            {model_table}

        </table>

        <div class="card">

            <p>
                <strong>Best Model:</strong>
                {model_results["best_model"]}
            </p>

            <p>
                <strong>Best RMSE:</strong>
                {display_number(model_results["best_rmse"])}
            </p>

            <p>
                <strong>30-Day Predicted Price:</strong>
                {display_number(forecast_price)}
            </p>

            <p>
                <strong>Expected Change:</strong>
                {display_number(forecast_change)}%
            </p>

        </div>

        <h2>Recent Historical Data</h2>

        <table>

            <tr>
                <th>Date</th>
                <th>Open</th>
                <th>High</th>
                <th>Low</th>
                <th>Close</th>
            </tr>

            {history_rows}

        </table>

        <p class="disclaimer">

            This report was generated by TEAM7 using historical
            market data, technical indicators and machine-learning
            models. Predictions and recommendations are for
            educational purposes only and should not be considered
            financial advice.

        </p>

    </body>

    </html>
    """

    return html


# =========================================================
# MAIN REPORT
# =========================================================

def dashboard_report(info, history):

    st.subheader(
        "Advanced Stock Analysis Report"
    )

    history = clean_history(
        history,
    )

    if history is None or history.empty:

        st.warning(
            "Historical data is not available."
        )

        return

    technical = calculate_indicators(
        history,
    )

    # =====================================================
    # MODEL ANALYSIS
    # =====================================================

    with st.spinner(
        "Preparing AI report..."
    ):

        model_results = analyze_models(
            history,
        )

    current_price = safe_float(
        technical["Close"].iloc[-1]
    )

    forecast_price = None
    forecast_change = None

    if model_results["forecast"]:

        forecast_price = float(
            model_results["forecast"][-1]
        )

        if (
            current_price is not None
            and
            current_price != 0
        ):

            forecast_change = (
                (
                    forecast_price
                    - current_price
                )
                / current_price
            ) * 100

    # =====================================================
    # AI RECOMMENDATION
    # =====================================================

    recommendation, confidence, ai_score = (
        create_ai_recommendation(
            info,
            technical,
            forecast_change,
        )
    )

    # =====================================================
    # REPORT SUMMARY
    # =====================================================

    st.markdown(
        "### Report Summary"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "Recommendation",
            recommendation,
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence}%",
        )

    with col3:

        st.metric(
            "Best Model",
            model_results["best_model"],
        )

    with col4:

        if forecast_change is not None:

            st.metric(
                "30-Day Outlook",
                f"{forecast_change:.2f}%",
            )

        else:

            st.metric(
                "30-Day Outlook",
                "N/A",
            )

    # =====================================================
    # COMPANY + MARKET TABLE
    # =====================================================

    latest = technical.iloc[-1]

    summary = {
        "Company": info.get(
            "name",
            "N/A",
        ),

        "Symbol": info.get(
            "symbol",
            "N/A",
        ),

        "Current Price": info.get(
            "price",
            "N/A",
        ),

        "Currency": info.get(
            "currency",
            "N/A",
        ),

        "Market Cap": info.get(
            "market_cap",
            "N/A",
        ),

        "P/E Ratio": info.get(
            "pe_ratio",
            "N/A",
        ),

        "EPS": info.get(
            "eps",
            "N/A",
        ),

        "Dividend Yield": info.get(
            "dividend_yield",
            "N/A",
        ),

        "Volume": info.get(
            "volume",
            "N/A",
        ),

        "52 Week High": info.get(
            "high_52",
            "N/A",
        ),

        "52 Week Low": info.get(
            "low_52",
            "N/A",
        ),

        "Sector": info.get(
            "sector",
            "N/A",
        ),

        "Industry": info.get(
            "industry",
            "N/A",
        ),

        "Country": info.get(
            "country",
            "N/A",
        ),

        "RSI": display_number(
            latest.get("RSI")
        ),

        "MA20": display_number(
            latest.get("MA20")
        ),

        "MA50": display_number(
            latest.get("MA50")
        ),

        "MACD": display_number(
            latest.get("MACD")
        ),

        "Volatility (%)": display_number(
            latest.get("Volatility")
        ),

        "AI Recommendation": recommendation,

        "AI Confidence": (
            f"{confidence}%"
        ),

        "AI Score": (
            f"{ai_score:.1f}"
        ),

        "Best AI Model": (
            model_results[
                "best_model"
            ]
        ),

        "Best RMSE": display_number(
            model_results[
                "best_rmse"
            ]
        ),

        "30-Day Predicted Price": (
            display_number(
                forecast_price
            )
        ),

        "Expected Change (%)": (
            display_number(
                forecast_change
            )
        ),
    }

    summary_df = pd.DataFrame(
        summary.items(),
        columns=[
            "Metric",
            "Value",
        ],
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    if model_results["models"]:

        st.markdown(
            "### AI Model Performance"
        )

        models_df = pd.DataFrame(
            model_results["models"]
        )

        st.dataframe(
            models_df.style.format(
                {
                    "MAE": "{:.2f}",
                    "MSE": "{:.2f}",
                    "RMSE": "{:.2f}",
                    "R² Score": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Select a longer historical period "
            "such as 1Y or 5Y to include full AI "
            "model performance in the report."
        )

    # =====================================================
    # CSV DOWNLOAD
    # =====================================================

    csv = summary_df.to_csv(
        index=False,
    ).encode(
        "utf-8",
    )

    symbol = info.get(
        "symbol",
        "stock",
    )

    st.download_button(
        label="Download Summary Report (CSV)",
        data=csv,
        file_name=(
            f"{symbol}_TEAM7_report.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    # =====================================================
    # FULL HTML REPORT DOWNLOAD
    # =====================================================

    html_report = create_html_report(
        info=info,
        technical=technical,
        model_results=model_results,
        recommendation=recommendation,
        confidence=confidence,
        ai_score=ai_score,
        forecast_price=forecast_price,
        forecast_change=forecast_change,
    )

    st.download_button(
        label="Download Full AI Report (HTML)",
        data=html_report.encode(
            "utf-8"
        ),
        file_name=(
            f"{symbol}_TEAM7_AI_report.html"
        ),
        mime="text/html",
        use_container_width=True,
    )

    # =====================================================
    # HISTORICAL DATA DOWNLOAD
    # =====================================================

    historical_csv = (
        technical.to_csv(
            index=False,
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        label="Download Historical Analysis (CSV)",
        data=historical_csv,
        file_name=(
            f"{symbol}_historical_analysis.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    st.success(
        "Advanced report generated successfully."
    )

    st.caption(
        "For the most complete AI report, use a historical "
        "period of 1Y or longer."
    )