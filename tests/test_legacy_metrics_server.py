import pytest

from mars_lite.server.metrics_server import create_app


def test_legacy_metrics_server_refuses_start_without_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("TRADE_RL_ENABLE_LEGACY_METRICS_SERVER", raising=False)

    with pytest.raises(RuntimeError, match="development-only"):
        create_app()


def test_legacy_metrics_server_accepts_explicit_factory_opt_in() -> None:
    app = create_app(development_only=True)

    assert app.title == "MarS Lite Training Server"


def test_legacy_metrics_server_accepts_exact_environment_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("TRADE_RL_ENABLE_LEGACY_METRICS_SERVER", "1")

    assert create_app().title == "MarS Lite Training Server"


@pytest.mark.parametrize("value", ["true", "yes", "01", " 1"])
def test_legacy_metrics_server_rejects_non_exact_environment_opt_in(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("TRADE_RL_ENABLE_LEGACY_METRICS_SERVER", value)

    with pytest.raises(RuntimeError, match="development-only"):
        create_app()
