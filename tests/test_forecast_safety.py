import numpy as np

from data.forecast_safety import is_forecast_path_plausible


def test_accepts_reasonable_forecast_path():
    forecast = np.linspace(100.0, 110.0, 30)

    assert is_forecast_path_plausible(100.0, forecast)


def test_rejects_explosive_forecast_from_reported_case():
    # The screenshot's estimate rises from about 1,246 to about 27,016.
    assert not is_forecast_path_plausible(1246.40, [27016.35])


def test_rejects_large_single_step_jump():
    assert not is_forecast_path_plausible(
        100.0,
        [100.0, 135.0],
        max_total_change=0.50,
        max_daily_change=0.30,
    )


def test_rejects_zero_negative_and_non_finite_values():
    assert not is_forecast_path_plausible(100.0, [100.0, 0.0])
    assert not is_forecast_path_plausible(100.0, [100.0, -1.0])
    assert not is_forecast_path_plausible(100.0, [100.0, np.inf])
    assert not is_forecast_path_plausible(100.0, [100.0, np.nan])


def test_rejects_missing_or_invalid_inputs():
    assert not is_forecast_path_plausible(100.0, [])
    assert not is_forecast_path_plausible(0.0, [100.0])
    assert not is_forecast_path_plausible(float("nan"), [100.0])


def test_direct_endpoint_can_skip_daily_step_check():
    # A direct 30-day estimate is an endpoint, not a one-day move.
    assert is_forecast_path_plausible(
        100.0,
        [125.0],
        max_daily_change=None,
    )
