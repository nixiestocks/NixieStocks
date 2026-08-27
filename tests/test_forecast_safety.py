import numpy as np
import pandas as pd

from data import forecast_safety


def history(rows=180, start=100.0, daily_change=0.001):
    dates = pd.bdate_range("2025-01-01", periods=rows)
    close = start * np.power(1.0 + daily_change, np.arange(rows))
    return pd.DataFrame({"Close": close}, index=dates)


def test_clean_history_sorts_dates_and_keeps_latest_duplicate():
    frame = history(100)
    duplicate = frame.iloc[[40]].copy()
    duplicate["Close"] = 777.0
    dirty = pd.concat([frame.iloc[::-1], duplicate])

    cleaned = forecast_safety._clean_history(dirty)

    assert cleaned is not None
    assert len(cleaned) == 100
    assert cleaned.loc[40, "Close"] == 777.0


def test_clean_history_rejects_ambiguous_close_columns():
    frame = pd.DataFrame(
        np.ones((100, 2)),
        columns=pd.MultiIndex.from_tuples([("Close", "A"), ("Close", "B")]),
    )

    assert forecast_safety._clean_history(frame) is None


def test_validation_split_has_full_forecast_horizon_embargo(monkeypatch):
    captured = {}

    def capture(train_x, train_y, test_x, test_y):
        captured["train_last"] = train_x.index[-1]
        captured["test_first"] = test_x.index[0]
        return []

    monkeypatch.setattr(forecast_safety, "_candidate_models", capture)

    result = forecast_safety.build_safe_30_day_forecast(history(220))

    assert result["forecast_guarded"] is True
    assert captured["test_first"] - captured["train_last"] > forecast_safety.HORIZON_DAYS


def test_short_history_rejects_legacy_forecast_outside_dynamic_limit():
    frame = history(60, daily_change=0.0002)
    original = {
        "price": float(frame["Close"].iloc[-1]),
        "forecast_price": float(frame["Close"].iloc[-1]) * 1.20,
        "forecast_change": 20.0,
        "best_model": "Legacy",
    }

    result = forecast_safety.stabilize_analysis(frame, original)

    assert result["forecast_price"] is None
    assert result["forecast_change"] is None
    assert result["forecast_guarded"] is True
    assert result["forecast_limit_pct"] == 12.0


def test_short_history_keeps_legacy_forecast_inside_dynamic_limit():
    frame = history(60, daily_change=0.0002)
    current = float(frame["Close"].iloc[-1])
    original = {
        "price": current,
        "forecast_price": current * 1.05,
        "forecast_change": 5.0,
        "best_model": "Legacy",
    }

    assert forecast_safety.stabilize_analysis(frame, original) == original


def test_horizon_jump_is_winsorized_before_limit_is_calculated():
    frame = history(180)
    frame.iloc[80, frame.columns.get_loc("Close")] *= 20

    limit = forecast_safety._plausible_change_limit(frame["Close"])

    assert 0.12 <= limit < 0.50


def test_invalid_current_price_never_produces_forecast():
    frame = history(180)
    frame.iloc[-1, frame.columns.get_loc("Close")] = 0

    assert forecast_safety.build_safe_30_day_forecast(frame) is None
