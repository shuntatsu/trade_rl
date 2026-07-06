import numpy as np

from mars_lite.eval.drift_monitor import DriftMonitor, DriftMonitorConfig


def test_drift_monitor_alerts_and_flattens_on_synthetic_shift():
    reference = np.zeros((120, 2), dtype=float)
    monitor = DriftMonitor(
        reference,
        config=DriftMonitorConfig(psi_threshold=0.1, ks_threshold=0.3),
    )
    current = np.vstack(
        [
            np.zeros((20, 2), dtype=float),
            np.ones((100, 2), dtype=float) * 3.0,
        ]
    )

    report = monitor.evaluate(current)
    first_alert = monitor.first_alert_step(current, window_size=20)

    assert report.should_flatten is True
    assert any(alert.metric == "psi" for alert in report.alerts)
    assert any(alert.metric == "ks" for alert in report.alerts)
    assert first_alert is not None
    assert first_alert <= 100


def test_prediction_distribution_is_monitored_separately():
    monitor = DriftMonitor(
        np.zeros((100, 1), dtype=float),
        reference_predictions=np.zeros(100, dtype=float),
        config=DriftMonitorConfig(prediction_psi_threshold=0.1),
    )

    report = monitor.evaluate(
        np.zeros((100, 1), dtype=float),
        predictions=np.ones(100, dtype=float) * 2.0,
    )

    assert report.should_flatten is True
    assert any(alert.metric == "prediction_psi" for alert in report.alerts)


def test_drift_monitor_invalid_inputs():
    import pytest

    from mars_lite.eval.drift_monitor import population_stability_index

    # 1. Non 2D reference features
    with pytest.raises(ValueError, match="reference_features must be a 2D array"):
        DriftMonitor(np.zeros(10))

    # 2. Empty reference features
    with pytest.raises(ValueError, match="reference_features cannot be empty"):
        DriftMonitor(np.zeros((0, 2)))

    # 3. Invalid config values
    with pytest.raises(ValueError, match="psi_threshold"):
        DriftMonitor(np.zeros((10, 2)), config=DriftMonitorConfig(psi_threshold=0.0))
    with pytest.raises(ValueError, match="ks_threshold"):
        DriftMonitor(np.zeros((10, 2)), config=DriftMonitorConfig(ks_threshold=-0.1))
    with pytest.raises(ValueError, match="ks_threshold"):
        DriftMonitor(np.zeros((10, 2)), config=DriftMonitorConfig(ks_threshold=1.5))
    with pytest.raises(ValueError, match="prediction_psi_threshold"):
        DriftMonitor(
            np.zeros((10, 2)), config=DriftMonitorConfig(prediction_psi_threshold=-0.5)
        )
    with pytest.raises(ValueError, match="bins"):
        DriftMonitor(np.zeros((10, 2)), config=DriftMonitorConfig(bins=0))

    monitor = DriftMonitor(np.zeros((10, 2)))

    # 4. Non 2D evaluate features
    with pytest.raises(ValueError, match="current_features must be a 2D array"):
        monitor.evaluate(np.zeros(10))

    # 5. Empty evaluate features
    with pytest.raises(ValueError, match="current_features cannot be empty"):
        monitor.evaluate(np.zeros((0, 2)))

    # 6. Dimension mismatch
    with pytest.raises(ValueError, match="feature dimensions do not match"):
        monitor.evaluate(np.zeros((10, 3)))

    # 7. Invalid first_alert_step window_size
    with pytest.raises(ValueError, match="window_size must be positive"):
        monitor.first_alert_step(np.zeros((10, 2)), window_size=0)

    # 8. window_size exceeds features length
    with pytest.raises(
        ValueError, match="current_features length must be at least window_size"
    ):
        monitor.first_alert_step(np.zeros((5, 2)), window_size=10)

    # 9. population_stability_index bins <= 0
    with pytest.raises(ValueError, match="bins must be positive"):
        population_stability_index([1, 2], [1, 2], bins=0)
