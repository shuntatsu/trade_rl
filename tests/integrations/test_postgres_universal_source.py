from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.integrations.postgres_universal_source import (
    UniversalSourceScope,
    load_postgres_universal_source,
)


class FakeCursor:
    def __init__(self, database: FakeSourceDatabase) -> None:
        self.database = database
        self.rows: list[tuple[object, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        assert isinstance(params, tuple)
        self.database.queries.append((query, params))
        symbol = str(params[1])
        if "public.rl_klines" in query:
            self.rows = list(self.database.klines[symbol])
        elif "public.rl_funding_rate" in query:
            self.rows = list(self.database.funding[symbol])
        elif "public.rl_derivatives" in query:
            self.rows = list(self.database.derivatives[symbol])
        elif "public.rl_orderflow_1m" in query:
            self.rows = list(self.database.orderflow[symbol])
        else:  # pragma: no cover - proves query routing is explicit
            raise AssertionError(query)

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class FakeSourceDatabase:
    def __init__(self, scope: UniversalSourceScope) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        timestamps = tuple(scope.start + timedelta(minutes=index) for index in range(3))
        self.klines = {
            symbol: [
                (timestamp, 10.0 + index, 11.0 + index, 9.0 + index, 10.5 + index, 2.0)
                for index, timestamp in enumerate(timestamps)
            ]
            for symbol in scope.symbols
        }
        self.funding = {symbol: [(timestamps[1], 0.0001)] for symbol in scope.symbols}
        self.derivatives = {
            symbol: [(timestamps[0], 100.0, 1.1, 2.0, 0.0002)]
            for symbol in scope.symbols
        }
        self.orderflow = {
            symbol: [(timestamps[2], 3.0, 2.0, 7, 0.5, 0.2)] for symbol in scope.symbols
        }

    @classmethod
    def complete(cls, scope: UniversalSourceScope) -> FakeSourceDatabase:
        return cls(scope)

    @classmethod
    def mutated(cls, scope: UniversalSourceScope, mutation: str) -> FakeSourceDatabase:
        database = cls(scope)
        symbol = scope.symbols[0]
        rows = database.klines[symbol]
        if mutation == "gap":
            rows.pop(1)
        elif mutation == "duplicate":
            rows.insert(1, rows[0])
        elif mutation == "bad_ohlc":
            row = list(rows[1])
            row[2] = 8.0
            rows[1] = tuple(row)
        elif mutation == "nan":
            row = list(rows[1])
            row[4] = float("nan")
            rows[1] = tuple(row)
        else:  # pragma: no cover - test helper misuse
            raise AssertionError(mutation)
        return database

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


def _short_scope() -> UniversalSourceScope:
    start = datetime(2024, 11, 13, tzinfo=UTC)
    return UniversalSourceScope(
        symbols=("ETHUSDT", "BTCUSDT"),
        start=start,
        end=start + timedelta(minutes=3),
    )


def test_source_loader_uses_half_open_interval_and_declared_symbol_order() -> None:
    maintained = UniversalSourceScope.maintained()
    assert maintained.start == datetime(2024, 11, 13, tzinfo=UTC)
    assert maintained.end == datetime(2026, 7, 5, tzinfo=UTC)
    assert len(maintained.symbols) == 15

    scope = _short_scope()
    database = FakeSourceDatabase.complete(scope)
    rows = load_postgres_universal_source(database, scope=scope)

    assert tuple(rows) == scope.symbols
    assert all(
        item.timestamps[0] == np.datetime64("2024-11-13T00:00:00")
        for item in rows.values()
    )
    assert all(
        item.timestamps[-1] < np.datetime64("2024-11-13T00:03:00")
        for item in rows.values()
    )
    assert len(rows["ETHUSDT"].derivative_timestamps) == 1
    assert len(rows["ETHUSDT"].orderflow_timestamps) == 1
    assert len(database.queries) == len(scope.symbols) * 4
    for query, params in database.queries:
        assert "timestamp >= %s" in query
        assert "timestamp < %s" in query
        assert "DELETE" not in query.upper()
        assert params[0] == "binance"
        assert params[-2:] == (scope.start, scope.end)
    kline_queries = [query for query, _ in database.queries if "rl_klines" in query]
    assert all("timeframe = '1m'" in query for query in kline_queries)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("gap", "contiguous"),
        ("duplicate", "unique"),
        ("bad_ohlc", "OHLCV"),
        ("nan", "finite"),
    ),
)
def test_source_loader_rejects_invalid_raw_rows(mutation: str, message: str) -> None:
    scope = _short_scope()
    with pytest.raises(ValueError, match=message):
        load_postgres_universal_source(
            FakeSourceDatabase.mutated(scope, mutation), scope=scope
        )


def test_source_scope_and_auxiliary_rows_fail_closed() -> None:
    start = datetime(2024, 11, 13, tzinfo=UTC)
    with pytest.raises(ValueError, match="unique"):
        UniversalSourceScope(
            symbols=("BTCUSDT", "BTCUSDT"),
            start=start,
            end=start + timedelta(minutes=3),
        )

    scope = _short_scope()
    database = FakeSourceDatabase.complete(scope)
    database.derivatives[scope.symbols[0]].append(
        database.derivatives[scope.symbols[0]][0]
    )
    with pytest.raises(ValueError, match="derivative timestamps.*increasing"):
        load_postgres_universal_source(database, scope=scope)


def test_derivative_observation_seconds_are_preserved_for_causal_asof() -> None:
    scope = _short_scope()
    database = FakeSourceDatabase.complete(scope)
    row = list(database.derivatives[scope.symbols[0]][0])
    row[0] = row[0] + timedelta(seconds=1)
    database.derivatives[scope.symbols[0]][0] = tuple(row)

    item = load_postgres_universal_source(database, scope=scope)[scope.symbols[0]]

    assert item.derivative_timestamps[0] == np.datetime64("2024-11-13T00:00:01")
