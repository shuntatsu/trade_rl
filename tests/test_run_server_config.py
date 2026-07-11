import pytest

from scripts.run_server import build_app_from_env


def test_serving_token_is_required() -> None:
    with pytest.raises(RuntimeError, match="TRADE_RL_SERVING_TOKEN"):
        build_app_from_env({})
