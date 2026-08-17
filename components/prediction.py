import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# TENSORFLOW / LSTM
# =========================================================

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True

except ImportError:
    TENSORFLOW_AVAILABLE = False


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(actual, predicted):

    actual = np.array(actual).reshape(-1)
    predicted = np.array(predicted).reshape(-1)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    try:

        r2 = r2_score(
            actual,
            predicted
        )

    except Exception:

        r2 = 0

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# PREPARE HISTORY
# =========================================================

def clean_history(history):

    if history is None or history.empty:
        return None

    df = history.copy()

    if "Date" not in df.columns:
        df = df.reset_index()

    if "Date" not in df.columns:
        return None

    if "Close" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Close"]
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


# =========================================================
# CLASSICAL ML DATA
# =========================================================

def prepare_classical_data(history):

    df = clean_history(history)

    if df is None:
        return None

    df["Lag_1"] = (
        df["Close"].shift(1)
    )

    df["Lag_2"] = (
        df["Close"].shift(2)
    )

    df["Lag_3"] = (
        df["Close"].shift(3)
    )

    df["Lag_5"] = (
        df["Close"].shift(5)
    )

    df["MA_5"] = (
        df["Close"]
        .rolling(window=5)
        .mean()
    )

    df["MA_10"] = (
        df["Close"]
        .rolling(window=10)
        .mean()
    )

    df["Return_1"] = (
        df["Close"]
        .pct_change()
    )

    # Predict next day's closing price
    df["Target"] = (
        df["Close"].shift(-1)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    return df


# =========================================================
# CREATE FUTURE CLASSICAL FEATURES
# =========================================================

def create_future_features(close_values):

    if len(close_values) < 10:
        return None

    lag_1 = close_values[-1]
    lag_2 = close_values[-2]
    lag_3 = close_values[-3]
    lag_5 = close_values[-5]

    ma_5 = np.mean(
        close_values[-5:]
    )

    ma_10 = np.mean(
        close_values[-10:]
    )

    previous_price = close_values[-2]

    if previous_price != 0:

        return_1 = (
            close_values[-1]
            - previous_price
        ) / previous_price

    else:

        return_1 = 0

    return pd.DataFrame(
        [
            {
                "Lag_1": lag_1,
                "Lag_2": lag_2,
                "Lag_3": lag_3,
                "Lag_5": lag_5,
                "MA_5": ma_5,
                "MA_10": ma_10,
                "Return_1": return_1
            }
        ]
    )


# =========================================================
# CLASSICAL ML FUTURE FORECAST
# =========================================================

def classical_future_forecast(
    model,
    history,
    days=30
):

    df = clean_history(history)

    if df is None:
        return []

    close_values = (
        df["Close"]
        .astype(float)
        .tolist()
    )

    predictions = []

    for _ in range(days):

        features = create_future_features(
            close_values
        )

        if features is None:
            break

        predicted_price = float(
            model.predict(features)[0]
        )

        predicted_price = max(
            predicted_price,
            0
        )

        predictions.append(
            predicted_price
        )

        close_values.append(
            predicted_price
        )

    return predictions


# =========================================================
# BUILD LSTM MODEL
# =========================================================

def build_lstm_model(sequence_length):

    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    1
                )
            ),

            LSTM(
                64,
                return_sequences=True
            ),

            Dropout(
                0.20
            ),

            LSTM(
                32
            ),

            Dropout(
                0.20
            ),

            Dense(
                16,
                activation="relu"
            ),

            Dense(
                1
            )
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# =========================================================
# PREPARE LSTM DATA
# =========================================================

def prepare_lstm_data(
    history,
    sequence_length=60
):

    df = clean_history(history)

    if df is None:
        return None

    if len(df) < sequence_length + 40:
        return None

    values = (
        df["Close"]
        .values
        .reshape(-1, 1)
    )

    dates = (
        df["Date"]
        .reset_index(drop=True)
    )

    split_index = int(
        len(values) * 0.80
    )

    if split_index <= sequence_length:
        return None

    # -----------------------------------------
    # Prevent future information leakage
    # -----------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    scaler.fit(
        values[:split_index]
    )

    scaled_values = scaler.transform(
        values
    )

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    test_dates = []

    for i in range(
        sequence_length,
        len(scaled_values)
    ):

        sequence = scaled_values[
            i - sequence_length:i
        ]

        target = scaled_values[i]

        if i < split_index:

            X_train.append(
                sequence
            )

            y_train.append(
                target
            )

        else:

            X_test.append(
                sequence
            )

            y_test.append(
                target
            )

            test_dates.append(
                dates.iloc[i]
            )

    X_train = np.array(
        X_train
    )

    y_train = np.array(
        y_train
    )

    X_test = np.array(
        X_test
    )

    y_test = np.array(
        y_test
    )

    if (
        len(X_train) < 20
        or
        len(X_test) < 5
    ):
        return None

    return {
        "df": df,
        "values": values,
        "scaled_values": scaled_values,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_dates": test_dates,
        "sequence_length": sequence_length
    }


# =========================================================
# TRAIN LSTM
# =========================================================

def train_lstm(lstm_data):

    model = build_lstm_model(
        lstm_data["sequence_length"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    model.fit(
        lstm_data["X_train"],
        lstm_data["y_train"],
        epochs=40,
        batch_size=32,
        validation_split=0.15,
        callbacks=[
            early_stopping
        ],
        verbose=0
    )

    return model


# =========================================================
# LSTM FUTURE FORECAST
# =========================================================

def lstm_future_forecast(
    model,
    scaler,
    history,
    sequence_length=60,
    days=30
):

    df = clean_history(
        history
    )

    if df is None:
        return []

    close_values = (
        df["Close"]
        .values
        .reshape(-1, 1)
    )

    if len(close_values) < sequence_length:
        return []

    scaled_values = scaler.transform(
        close_values
    )

    current_sequence = (
        scaled_values[
            -sequence_length:
        ]
        .reshape(
            1,
            sequence_length,
            1
        )
    )

    future_predictions = []

    for _ in range(days):

        predicted_scaled = float(
            model.predict(
                current_sequence,
                verbose=0
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
            0
        )

        future_predictions.append(
            predicted_price
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

        current_sequence = np.concatenate(
            (
                current_sequence[:, 1:, :],
                new_value
            ),
            axis=1
        )

    return future_predictions


# =========================================================
# MAIN PREDICTION
# =========================================================

def prediction(history):

    st.subheader(
        "AI Price Prediction"
    )

    history_df = clean_history(
        history
    )

    if (
        history_df is None
        or
        len(history_df) < 40
    ):

        st.warning(
            "Not enough historical data for AI prediction. "
            "Select a longer time period such as 1Y, 2Y or 5Y."
        )

        return

    # =====================================================
    # CLASSICAL ML
    # =====================================================

    df = prepare_classical_data(
        history_df
    )

    if df is None or len(df) < 30:

        st.warning(
            "Not enough historical data to train AI models."
        )

        return

    features = [
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_5",
        "MA_5",
        "MA_10",
        "Return_1"
    ]

    X = df[features]
    y = df["Target"]

    split_index = int(
        len(df) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    test_dates = df["Date"].iloc[
        split_index:
    ]

    # =====================================================
    # LINEAR REGRESSION
    # =====================================================

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_predictions = (
        linear_model.predict(
            X_test
        )
    )

    linear_metrics = calculate_metrics(
        y_test,
        linear_predictions
    )

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_predictions = (
        random_forest.predict(
            X_test
        )
    )

    rf_metrics = calculate_metrics(
        y_test,
        rf_predictions
    )

    # =====================================================
    # LSTM
    # =====================================================

    lstm_model = None
    lstm_metrics = None
    lstm_predictions = None
    lstm_test_dates = None
    lstm_scaler = None

    if TENSORFLOW_AVAILABLE:

        lstm_data = prepare_lstm_data(
            history_df,
            sequence_length=60
        )

        if lstm_data is not None:

            with st.spinner(
                "Training LSTM deep learning model..."
            ):

                lstm_model = train_lstm(
                    lstm_data
                )

            predicted_scaled = (
                lstm_model.predict(
                    lstm_data["X_test"],
                    verbose=0
                )
            )

            lstm_predictions = (
                lstm_data["scaler"]
                .inverse_transform(
                    predicted_scaled
                )
                .reshape(-1)
            )

            actual_lstm = (
                lstm_data["scaler"]
                .inverse_transform(
                    lstm_data["y_test"]
                )
                .reshape(-1)
            )

            lstm_metrics = calculate_metrics(
                actual_lstm,
                lstm_predictions
            )

            lstm_test_dates = (
                lstm_data["test_dates"]
            )

            lstm_scaler = (
                lstm_data["scaler"]
            )

    # =====================================================
    # MODEL COMPARISON TABLE
    # =====================================================

    st.markdown(
        "### Model Performance"
    )

    model_rows = [
        {
            "Model": "Linear Regression",
            "MAE": linear_metrics["MAE"],
            "MSE": linear_metrics["MSE"],
            "RMSE": linear_metrics["RMSE"],
            "R² Score": linear_metrics["R2"]
        },
        {
            "Model": "Random Forest",
            "MAE": rf_metrics["MAE"],
            "MSE": rf_metrics["MSE"],
            "RMSE": rf_metrics["RMSE"],
            "R² Score": rf_metrics["R2"]
        }
    ]

    if lstm_metrics is not None:

        model_rows.append(
            {
                "Model": "LSTM Deep Learning",
                "MAE": lstm_metrics["MAE"],
                "MSE": lstm_metrics["MSE"],
                "RMSE": lstm_metrics["RMSE"],
                "R² Score": lstm_metrics["R2"]
            }
        )

    comparison = pd.DataFrame(
        model_rows
    )

    comparison = comparison.sort_values(
        by="RMSE"
    )

    st.dataframe(
        comparison.style.format(
            {
                "MAE": "{:.2f}",
                "MSE": "{:.2f}",
                "RMSE": "{:.2f}",
                "R² Score": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # CHOOSE BEST MODEL
    # =====================================================

    model_scores = {
        "Linear Regression":
            linear_metrics["RMSE"],

        "Random Forest":
            rf_metrics["RMSE"]
    }

    if lstm_metrics is not None:

        model_scores[
            "LSTM Deep Learning"
        ] = lstm_metrics["RMSE"]

    best_model_name = min(
        model_scores,
        key=model_scores.get
    )

    best_rmse = model_scores[
        best_model_name
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Best AI Model",
            best_model_name
        )

    with col2:

        st.metric(
            "Best RMSE",
            f"{best_rmse:.2f}"
        )

    # =====================================================
    # ACTUAL VS MODEL GRAPH
    # =====================================================

    st.markdown(
        "### Actual vs AI Predictions"
    )

    fig_compare = go.Figure()

    fig_compare.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test,
            mode="lines",
            name="Actual Price"
        )
    )

    fig_compare.add_trace(
        go.Scatter(
            x=test_dates,
            y=linear_predictions,
            mode="lines",
            name="Linear Regression"
        )
    )

    fig_compare.add_trace(
        go.Scatter(
            x=test_dates,
            y=rf_predictions,
            mode="lines",
            name="Random Forest"
        )
    )

    if (
        lstm_predictions is not None
        and
        lstm_test_dates is not None
    ):

        fig_compare.add_trace(
            go.Scatter(
                x=lstm_test_dates,
                y=lstm_predictions,
                mode="lines",
                name="LSTM"
            )
        )

    fig_compare.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    # =====================================================
    # TRAIN BEST MODEL ON FULL DATA
    # =====================================================

    forecast = []

    if best_model_name == "Linear Regression":

        best_model = LinearRegression()

        best_model.fit(
            X,
            y
        )

        forecast = classical_future_forecast(
            best_model,
            history_df,
            days=30
        )

    elif best_model_name == "Random Forest":

        best_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        best_model.fit(
            X,
            y
        )

        forecast = classical_future_forecast(
            best_model,
            history_df,
            days=30
        )

    elif (
        best_model_name
        == "LSTM Deep Learning"
    ):

        full_values = (
            history_df["Close"]
            .values
            .reshape(-1, 1)
        )

        full_scaler = MinMaxScaler(
            feature_range=(0, 1)
        )

        full_scaled = full_scaler.fit_transform(
            full_values
        )

        sequence_length = 60

        X_full = []
        y_full = []

        for i in range(
            sequence_length,
            len(full_scaled)
        ):

            X_full.append(
                full_scaled[
                    i - sequence_length:i
                ]
            )

            y_full.append(
                full_scaled[i]
            )

        X_full = np.array(
            X_full
        )

        y_full = np.array(
            y_full
        )

        full_lstm_model = build_lstm_model(
            sequence_length
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )

        with st.spinner(
            "Training final LSTM forecast model..."
        ):

            full_lstm_model.fit(
                X_full,
                y_full,
                epochs=40,
                batch_size=32,
                validation_split=0.15,
                callbacks=[
                    early_stopping
                ],
                verbose=0
            )

        forecast = lstm_future_forecast(
            full_lstm_model,
            full_scaler,
            history_df,
            sequence_length=60,
            days=30
        )

    # =====================================================
    # FORECAST CHECK
    # =====================================================

    if len(forecast) == 0:

        st.warning(
            "Unable to generate the 30-day forecast."
        )

        return

    # =====================================================
    # FUTURE DATES
    # =====================================================

    last_date = pd.to_datetime(
        history_df["Date"].iloc[-1]
    )

    future_dates = pd.bdate_range(
        start=last_date
        + pd.Timedelta(days=1),
        periods=len(forecast)
    )

    # =====================================================
    # FUTURE GRAPH
    # =====================================================

    st.markdown(
        "### 30 Trading Day AI Forecast"
    )

    forecast_fig = go.Figure()

    recent_history = (
        history_df.tail(90)
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=recent_history["Date"],
            y=recent_history["Close"],
            mode="lines",
            name="Historical Price"
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast,
            mode="lines+markers",
            name="AI Forecast"
        )
    )

    forecast_fig.update_layout(
        title=(
            f"30-Day Forecast Using "
            f"{best_model_name}"
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

    # =====================================================
    # FORECAST RESULTS
    # =====================================================

    current_price = float(
        history_df["Close"]
        .iloc[-1]
    )

    predicted_price = float(
        forecast[-1]
    )

    expected_change = (
        (
            predicted_price
            - current_price
        )
        / current_price
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Price",
            f"{current_price:.2f}"
        )

    with col2:

        st.metric(
            "30-Day Predicted Price",
            f"{predicted_price:.2f}"
        )

    with col3:

        st.metric(
            "Expected Change",
            f"{expected_change:.2f}%"
        )

    # =====================================================
    # AI TREND
    # =====================================================

    st.markdown(
        "### AI Market Outlook"
    )

    if expected_change >= 5:

        st.success(
            f"Bullish Forecast — "
            f"{best_model_name} predicts "
            f"approximately {expected_change:.2f}% "
            f"growth over the next 30 trading days."
        )

    elif expected_change <= -5:

        st.error(
            f"Bearish Forecast — "
            f"{best_model_name} predicts "
            f"approximately {abs(expected_change):.2f}% "
            f"decline over the next 30 trading days."
        )

    else:

        st.info(
            f"Stable Forecast — "
            f"{best_model_name} predicts "
            f"approximately {expected_change:.2f}% "
            f"movement over the next 30 trading days."
        )

    # =====================================================
    # TENSORFLOW MESSAGE
    # =====================================================

    if not TENSORFLOW_AVAILABLE:

        st.warning(
            "TensorFlow is not installed, so LSTM was skipped. "
            "Install TensorFlow to enable the deep learning model."
        )

    elif lstm_metrics is None:

        st.info(
            "LSTM requires more historical data. "
            "Try selecting 1Y, 2Y or 5Y."
        )

    st.caption(
        "Predictions are generated from historical data "
        "for educational purposes and are not financial advice."
    )