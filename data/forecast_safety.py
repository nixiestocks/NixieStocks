from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


HORIZON_DAYS = 30
FEATURE_COLUMNS = [
    "Close",
    "Return1",
    "Return5",
    "MA5",
    "MA10",
    "MA20",
    "Momentum10",
    "Momentum20",
    "Volatility20",
]


def _safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _clean_history(history):
    if history is None or getattr(history, "empty", True):
        return None

    df = history.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return None

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    if len(df) < 90:
        return None

    return df.reset_index(drop=True)


def _feature_frame(history):
    df = history.copy()
    close = df["Close"].astype(float)

    df["Return1"] = close.pct_change()
    df["Return5"] = close.pct_change(5)
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["Momentum10"] = close.pct_change(10)
    df["Momentum20"] = close.pct_change(20)
    df["Volatility20"] = close.pct_change().rolling(20).std()
    df["Target"] = close.shift(-HORIZON_DAYS)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _plausible_change_limit(close):
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return 0.20

    recent = returns.tail(90)

    # Winsorize extreme one-day jumps (splits/data glitches) before
    # estimating normal volatility.
    if len(recent) >= 20:
        low = recent.quantile(0.05)
        high = recent.quantile(0.95)
        recent = recent.clip(lower=low, upper=high)

    daily_vol = _safe_float(recent.std()) or 0.02
    volatility_limit = 3.5 * daily_vol * np.sqrt(HORIZON_DAYS)

    horizon_returns = close.pct_change(HORIZON_DAYS).dropna().tail(120)
    historical_limit = 0.0

    if not horizon_returns.empty:
        historical_limit = float(horizon_returns.abs().quantile(0.95)) * 1.5

    # Minimum room prevents over-clamping normal stocks; the hard maximum
    # prevents recursive/extrapolation explosions from ever reaching the UI.
    return float(np.clip(max(volatility_limit, historical_limit, 0.12), 0.12, 0.50))


def _candidate_models(train_x, train_y, test_x, test_y):
    models = [
        (
            "Linear Regression (30D Direct)",
            LinearRegression(),
        ),
        (
            "Random Forest (30D Direct)",
            RandomForestRegressor(
                n_estimators=140,
                max_depth=9,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    candidates = []

    for name, model in models:
        try:
            model.fit(train_x, train_y)
            test_prediction = model.predict(test_x)

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        test_y,
                        test_prediction,
                    )
                )
            )

            scale = float(np.nanmedian(np.abs(test_y)))
            normalized_rmse = rmse / scale if scale > 0 else np.inf

            candidates.append(
                {
                    "name": name,
                    "model": model,
                    "nrmse": normalized_rmse,
                }
            )
        except Exception:
            continue

    return sorted(candidates, key=lambda item: item["nrmse"])


def _trend_fallback(close, limit):
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return float(close.iloc[-1])

    recent_20 = returns.tail(20)
    recent_60 = returns.tail(60)

    short_drift = float(recent_20.median()) if not recent_20.empty else 0.0
    medium_drift = float(recent_60.median()) if not recent_60.empty else short_drift
    drift = 0.65 * short_drift + 0.35 * medium_drift

    daily_vol = _safe_float(recent_60.std()) or 0.02
    daily_drift_cap = min(0.004, max(0.0008, daily_vol * 0.20))
    drift = float(np.clip(drift, -daily_drift_cap, daily_drift_cap))

    projected_change = (1.0 + drift) ** HORIZON_DAYS - 1.0
    fallback_limit = min(0.25, limit * 0.65)
    projected_change = float(
        np.clip(
            projected_change,
            -fallback_limit,
            fallback_limit,
        )
    )

    return float(close.iloc[-1]) * (1.0 + projected_change)


def build_safe_30_day_forecast(history):
    """Return a stable direct 30-trading-day forecast.

    The original model predicted one day at a time and fed each prediction
    back into the next step. Linear Regression can become numerically unstable
    in that recursive loop. This implementation predicts the 30-day target
    directly, validates it against historical volatility, and falls back to a
    conservative trend estimate if every ML candidate is implausible.
    """

    history = _clean_history(history)
    if history is None:
        return None

    features = _feature_frame(history)
    latest_features = features[FEATURE_COLUMNS].iloc[[-1]].dropna()

    training = features.dropna(subset=FEATURE_COLUMNS + ["Target"])

    if latest_features.empty or len(training) < 70:
        return None

    split = int(len(training) * 0.80)

    if split < 45 or len(training) - split < 8:
        return None

    train_x = training[FEATURE_COLUMNS].iloc[:split]
    train_y = training["Target"].iloc[:split]
    test_x = training[FEATURE_COLUMNS].iloc[split:]
    test_y = training["Target"].iloc[split:]

    current_price = float(history["Close"].iloc[-1])
    if not np.isfinite(current_price) or current_price <= 0:
        return None

    limit = _plausible_change_limit(history["Close"].astype(float))
    candidates = _candidate_models(train_x, train_y, test_x, test_y)

    # Refit candidates on all available training observations, then accept the
    # best test performer whose 30-day forecast stays inside a data-driven band.
    all_x = training[FEATURE_COLUMNS]
    all_y = training["Target"]

    for candidate in candidates:
        if not np.isfinite(candidate["nrmse"]) or candidate["nrmse"] > 0.30:
            continue

        try:
            model = candidate["model"]
            model.fit(all_x, all_y)
            forecast_price = float(model.predict(latest_features)[0])
        except Exception:
            continue

        if not np.isfinite(forecast_price) or forecast_price <= 0:
            continue

        forecast_change = (forecast_price - current_price) / current_price

        if abs(forecast_change) <= limit:
            return {
                "forecast_price": forecast_price,
                "forecast_change": forecast_change * 100.0,
                "best_model": candidate["name"],
                "forecast_guarded": False,
                "forecast_limit_pct": limit * 100.0,
            }

    forecast_price = _trend_fallback(history["Close"].astype(float), limit)
    forecast_change = (forecast_price - current_price) / current_price

    return {
        "forecast_price": forecast_price,
        "forecast_change": forecast_change * 100.0,
        "best_model": "Stabilized Trend",
        "forecast_guarded": True,
        "forecast_limit_pct": limit * 100.0,
    }


def stabilize_analysis(history, result):
    if not isinstance(result, dict):
        return result

    updated = result.copy()

    safe_forecast = build_safe_30_day_forecast(history)

    if safe_forecast is None:
        # Never let an obviously broken recursive forecast survive merely
        # because there was not enough data to build the replacement model.
        current_price = _safe_float(updated.get("price"))
        forecast_price = _safe_float(updated.get("forecast_price"))

        if current_price and forecast_price:
            raw_change = abs((forecast_price - current_price) / current_price)
            if raw_change > 0.50:
                updated["forecast_price"] = None
                updated["forecast_change"] = None
                updated["best_model"] = "Forecast unavailable"
                updated["forecast_guarded"] = True

        return updated

    updated.update(safe_forecast)
    return updated
