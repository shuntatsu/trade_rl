from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts import materialize_universal_postgres_cache as module
from trade_rl.integrations.postgres_universal_source import UniversalSourceScope


def test_cli_defaults_are_the_maintained_real_data_scope() -> None:
    args = module._parser().parse_args(
        [
            "--postgres-url",
            "postgresql://db",
            "--report-root",
            "artifacts/report",
        ]
    )

    assert args.cache_id == (
        "binance-usds-m-native-indicators-15x-20241113-20260705-v1"
    )
    assert args.start == "2024-11-13T00:00:00Z"
    assert args.end == "2026-07-05T00:00:00Z"


def test_source_builds_are_loaded_one_symbol_at_a_time(monkeypatch) -> None:
    start = datetime(2024, 11, 13, tzinfo=UTC)
    scope = UniversalSourceScope(
        symbols=("ETHUSDT", "BTCUSDT"),
        start=start,
        end=start + timedelta(minutes=60),
    )
    loaded: list[tuple[str, ...]] = []
    sentinel_builds: list[object] = []

    def load_source(connection, *, scope):
        del connection
        loaded.append(scope.symbols)
        return {scope.symbols[0]: object()}

    def build_cache(source, *, scope):
        assert tuple(source) == scope.symbols
        build = object()
        sentinel_builds.append(build)
        return build

    combined = object()

    def combine(builds, *, scope):
        assert tuple(builds) == tuple(sentinel_builds)
        assert scope.symbols == ("ETHUSDT", "BTCUSDT")
        return combined

    monkeypatch.setattr(module, "load_postgres_universal_source", load_source)
    monkeypatch.setattr(module, "build_native_indicator_cache", build_cache)
    monkeypatch.setattr(module, "combine_native_indicator_builds", combine)

    assert module._build_source_streaming(object(), scope=scope) is combined
    assert loaded == [("ETHUSDT",), ("BTCUSDT",)]
