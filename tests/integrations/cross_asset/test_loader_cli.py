"""Loader, normaliser, CLI and package-boundary contracts."""

from __future__ import annotations

import ast
import io
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import trade_rl.integrations.cross_asset.cli as cli_module
from tests.integrations.cross_asset.support import (
    FETCHED_AT,
    OPT_IN_ALL,
    FakeTransport,
    clock,
    fred_payload,
    random_walk,
    seed_store,
    weekdays,
)
from trade_rl.integrations.cross_asset.catalog import series_by_key
from trade_rl.integrations.cross_asset.cli import main, status_report
from trade_rl.integrations.cross_asset.features import (
    COMPOSITE_FEATURE_NAMES,
    FEATURE_SET_ID,
)
from trade_rl.integrations.cross_asset.fetch import fetch_all
from trade_rl.integrations.cross_asset.loader import (
    load_cross_asset_features,
    regular_grid,
)
from trade_rl.integrations.cross_asset.normalization import FeatureNormalizer

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "trade_rl" / "integrations" / "cross_asset"


def _grid(timeframe: str) -> np.ndarray:
    return regular_grid(
        datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 3, 20, tzinfo=UTC), timeframe
    )


def test_regular_grid_is_utc_and_half_open() -> None:
    grid = _grid("4h")
    assert grid[0] == np.datetime64("2024-03-01T00:00:00")
    assert grid[-1] == np.datetime64("2024-03-19T20:00:00")
    assert set(np.diff(grid).astype("timedelta64[s]").astype(int)) == {4 * 3600}
    with pytest.raises(ValueError, match="before"):
        regular_grid(
            datetime(2024, 3, 2, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC), "1d"
        )
    with pytest.raises(ValueError, match="timezone"):
        regular_grid(datetime(2024, 3, 1), datetime(2024, 3, 2, tzinfo=UTC), "1d")


def test_values_do_not_depend_on_the_grid_resolution(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    four_hour = load_cross_asset_features(store.root, _grid("4h"))
    daily = load_cross_asset_features(store.root, _grid("1d"))
    midnight = np.isin(four_hour.decision_ns, daily.decision_ns)
    assert midnight.sum() == daily.decision_ns.size
    assert four_hour.names == daily.names
    assert np.array_equal(four_hour.values[midnight], daily.values, equal_nan=True)
    assert np.array_equal(four_hour.available[midnight], daily.available)
    assert four_hour.input_digest == daily.input_digest


def test_as_of_selects_the_vintage_known_at_that_time(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    first = load_cross_asset_features(store.root, _grid("1d"), opt_in_series=("SP500",))
    days = weekdays(420)
    revised = random_walk(420, seed=0)  # SP500 is seed offset 0 in seed_store
    revised[300:310] *= 1.2  # a revision inside the grid period
    later = FETCHED_AT + timedelta(days=3)
    (record,) = fetch_all(
        store,
        (series_by_key("SP500"),),
        FakeTransport({"id=SP500": fred_payload("SP500", days, revised)}),
        now=clock(later),
        sleep=lambda _s: None,
    )
    assert record["status"] == "ok"
    assert record["revision"]["changed"] == 10  # type: ignore[index]
    old = load_cross_asset_features(
        store.root,
        _grid("1d"),
        as_of=FETCHED_AT + timedelta(hours=1),
        opt_in_series=("SP500",),
    )
    new = load_cross_asset_features(store.root, _grid("1d"), opt_in_series=("SP500",))
    assert old.input_digest == first.input_digest
    assert new.input_digest != old.input_digest
    column = old.column("SP500__ret1")
    assert np.array_equal(old.values, first.values, equal_nan=True)
    difference = np.abs(old.values[:, column] - new.values[:, column])
    assert np.nanmax(difference) > 0.05
    with pytest.raises(FileNotFoundError):
        load_cross_asset_features(
            store.root, _grid("1d"), as_of=FETCHED_AT - timedelta(days=1)
        )


def test_loader_can_restrict_series_and_select_features(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    frame = load_cross_asset_features(
        store.root,
        _grid("1d"),
        series=("SP500", "NASDAQCOM"),
        opt_in_series=("SP500", "NASDAQCOM"),
        feature_names=("SP500__ret1", "dollar_strength"),
    )
    assert frame.names == ("SP500__ret1", "dollar_strength")
    assert frame.series_present == ("NASDAQCOM", "SP500")
    assert not frame.available[:, 1].any()  # too few dollar members


def test_loader_refuses_a_tampered_raw_file(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    record = store.read_manifest()[0]
    (tmp_path / str(record["path"])).write_bytes(b"observation_date,X\n")
    with pytest.raises(ValueError, match="digest"):
        load_cross_asset_features(store.root, _grid("1d"))


def test_loader_parses_exactly_the_raw_bytes_it_digest_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw file changed after digest verification must not leak new values."""
    from trade_rl.integrations.cross_asset.observations import load_obs_series

    store = seed_store(tmp_path)
    record = next(r for r in store.read_manifest() if r["key"] == "SP500")
    expected = load_obs_series(store, {"SP500": record})["SP500"]
    original = store.read_raw(record)
    rows = original.splitlines(keepends=True)
    assert len(rows) > 2 and b"," in rows[1]
    rows[1] = rows[1].split(b",", 1)[0] + b",999999.0\n"
    tampered = b"".join(rows)
    assert tampered != original
    target = tmp_path / str(record["path"])
    real_read = store.read_raw
    calls = 0

    def read_then_modify_file(item: Mapping[str, object]) -> bytes:
        nonlocal calls
        payload = real_read(item)
        calls += 1
        if calls == 1:
            target.write_bytes(tampered)
        return payload

    monkeypatch.setattr(store, "read_raw", read_then_modify_file)
    actual = load_obs_series(store, {"SP500": record})["SP500"]
    assert calls == 1
    assert actual.sha256 == expected.sha256 == record["sha256"]
    np.testing.assert_array_equal(actual.dates, expected.dates)
    np.testing.assert_array_equal(actual.values, expected.values)
    assert target.read_bytes() == tampered
    # A subsequent call may never admit the now-altered on-disk source.
    with pytest.raises(ValueError, match="digest"):
        load_obs_series(store, {"SP500": record})


@pytest.mark.parametrize("failure", ("missing", "tampered"))
def test_loader_refuses_invalid_raw_before_calling_source_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Retain the existing missing/digest halt without parsing untrusted bytes."""
    from trade_rl.integrations.cross_asset.observations import load_obs_series
    from trade_rl.integrations.cross_asset.parsers import PARSERS

    store = seed_store(tmp_path)
    record = next(r for r in store.read_manifest() if r["key"] == "SP500")
    target = tmp_path / str(record["path"])
    if failure == "missing":
        target.unlink()
    else:
        target.write_bytes(b"invalid cached source payload")
    parser_name, _ = PARSERS["fred"]

    def unexpected_parse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unverified source must never reach the parser")

    monkeypatch.setitem(PARSERS, "fred", (parser_name, unexpected_parse))
    with pytest.raises(ValueError, match="raw file for SP500 is missing or fails its digest"):
        load_obs_series(store, {"SP500": record})


def test_normalizer_statistics_come_only_from_the_fit_window(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    grid = regular_grid(
        datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 7, 1, tzinfo=UTC), "1d"
    )
    frame = load_cross_asset_features(store.root, grid, opt_in_series=OPT_IN_ALL)
    names = ("dollar_strength", "risk_appetite", "SP500__ret5")
    chosen = frame.select(names)
    start, end = grid[0], grid[100]
    fitted = FeatureNormalizer.fit(chosen, fit_start=start, fit_end=end, min_count=20)
    # Corrupt every row outside [start, end): statistics must not move.
    corrupted_values = chosen.values.copy()
    outside = ~(
        (chosen.decision_ns >= start.astype(np.int64))
        & (chosen.decision_ns < end.astype(np.int64))
    )
    corrupted_values[outside] = 1e6
    corrupted = type(chosen)(
        feature_set=chosen.feature_set,
        definitions_digest=chosen.definitions_digest,
        input_digest=chosen.input_digest,
        names=chosen.names,
        decision_ns=chosen.decision_ns,
        values=np.where(chosen.available, corrupted_values, np.nan),
        available=chosen.available,
        staleness_hours=chosen.staleness_hours,
        series_present=chosen.series_present,
    )
    again = FeatureNormalizer.fit(corrupted, fit_start=start, fit_end=end, min_count=20)
    assert np.array_equal(fitted.mean, again.mean, equal_nan=True)
    assert np.array_equal(fitted.scale, again.scale, equal_nan=True)
    assert fitted.digest() == again.digest()
    # Inside the window the standardised data has zero mean and unit spread.
    z = fitted.transform(chosen)
    inside = (chosen.decision_ns >= start.astype(np.int64)) & (
        chosen.decision_ns < end.astype(np.int64)
    )
    for column in range(len(names)):
        ok = z.available[inside, column]
        if ok.sum() >= 20:
            sample = z.values[inside, column][ok]
            assert abs(sample.mean()) < 0.35 and 0.5 < sample.std(ddof=1) < 1.2
    assert np.nanmax(np.abs(z.values)) <= fitted.clip + 1e-12
    shifted = FeatureNormalizer.fit(
        chosen, fit_start=grid[10], fit_end=grid[110], min_count=20
    )
    assert shifted.digest() != fitted.digest()


def test_normalizer_rejects_bad_windows_and_mismatched_frames(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    grid = _grid("1d")
    frame = load_cross_asset_features(store.root, grid, feature_names=("curve_slope",))
    other = load_cross_asset_features(
        store.root, grid, feature_names=("risk_appetite",)
    )
    with pytest.raises(ValueError, match="fit_start < fit_end"):
        FeatureNormalizer.fit(frame, fit_start=grid[5], fit_end=grid[5])
    with pytest.raises(ValueError, match="timezone"):
        FeatureNormalizer.fit(
            frame, fit_start=datetime(2024, 3, 1), fit_end=datetime(2024, 3, 5)
        )
    normalizer = FeatureNormalizer.fit(
        frame, fit_start=grid[0], fit_end=grid[-1], min_count=5
    )
    with pytest.raises(ValueError, match="do not match"):
        normalizer.transform(other)
    starved = FeatureNormalizer.fit(
        frame, fit_start=grid[0], fit_end=grid[3], min_count=30
    )
    assert not starved.transform(frame).available.any()


def _fake_cli_transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    days = weekdays(100, start="2024-01-01")
    transport = FakeTransport(
        {
            "id=DGS10": fred_payload("DGS10", days, random_walk(100, 1) / 25.0),
            "stooq.com": b"<!DOCTYPE html><script>fetch('/__verify')</script>",
        }
    )
    monkeypatch.setattr(cli_module, "UrllibTransport", lambda: transport)
    return transport


def test_cli_fetch_status_and_build_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _fake_cli_transport(monkeypatch)
    out = io.StringIO()
    code = main(
        [
            "--root",
            str(tmp_path),
            "fetch",
            "--sources",
            "fred,stooq",
            "--series",
            "DGS10,stq_spx,stq_ndq",
            "--delay",
            "0",
        ],
        out,
    )
    assert code == 0
    text = out.getvalue()
    assert "DGS10" in text and "ok" in text and "bot-detection" in text
    assert "1/3 series usable" in text
    assert len([u for u in transport.requested if "stooq" in u]) == 1

    status = io.StringIO()
    assert main(["--root", str(tmp_path), "status"], status) == 0
    report = status_report(cli_module.CrossAssetStore(tmp_path))
    by_key = {row["key"]: row for row in report}
    assert by_key["DGS10"]["state"] == "ok" and by_key["DGS10"]["n_obs"] > 0
    assert by_key["stq_spx"]["state"] == "unavailable"
    assert by_key["SP500"]["state"] == "never fetched"
    assert FEATURE_SET_ID in status.getvalue()

    built = io.StringIO()
    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "build",
                "--timeframe",
                "1d",
                "--start",
                "2024-03-01",
                "--end",
                "2024-04-01",
            ],
            built,
        )
        == 0
    )
    outputs = sorted((tmp_path / "built" / FEATURE_SET_ID).glob("*.json"))
    assert len(outputs) == 1
    meta = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert meta["names"] == list(COMPOSITE_FEATURE_NAMES)
    assert meta["rows"] == 31
    with np.load(outputs[0].with_suffix(".npz")) as archive:
        assert archive["values"].shape == (31, len(COMPOSITE_FEATURE_NAMES))


def test_cli_fetch_fails_when_nothing_is_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_cli_transport(monkeypatch)
    out = io.StringIO()
    code = main(
        [
            "--root",
            str(tmp_path),
            "fetch",
            "--sources",
            "stooq",
            "--series",
            "stq_spx",
            "--delay",
            "0",
        ],
        out,
    )
    assert code == 2


def test_status_flags_a_series_that_lags_its_configured_release_rule(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    # SP500 lag is three business days; fetched 2025-07-10 12:00Z (Thursday), the
    # data ends 2024-08, far behind: the lag audit must flag it.
    row = next(r for r in status_report(store) if r["key"] == "SP500")
    assert str(row["lag_audit"]).startswith("BEHIND")


def test_package_never_touches_crypto_data_or_other_layers() -> None:
    forbidden = (
        "trade_rl.integrations.binance",
        "trade_rl.integrations.market_store",
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "trade_rl.simulation",
    )
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            for name in names:
                assert not name.startswith(forbidden), (path.name, name)
    text = "\n".join(p.read_text(encoding="utf-8") for p in PACKAGE.glob("*.py"))
    for token in ("BTCUSDT", "ETHUSDT", "klines", "api_key", "apikey", "secret"):
        assert token not in text, token


def test_no_other_production_module_imports_the_overlay_package() -> None:
    offenders = []
    for path in (ROOT / "trade_rl").rglob("*.py"):
        if PACKAGE in path.parents:
            continue
        if "integrations.cross_asset" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_all_source_urls_are_https_and_keyless() -> None:
    from trade_rl.integrations.cross_asset.catalog import CATALOG

    for spec in CATALOG:
        url = spec.url()
        assert url.startswith("https://")
        assert "apikey" not in url.lower() and "api_key" not in url.lower()


@pytest.mark.skipif(
    os.environ.get("CROSS_ASSET_NETWORK_TESTS") != "1",
    reason="set CROSS_ASSET_NETWORK_TESTS=1 to run live key-less fetches",
)
def test_live_fred_fetch_parses(tmp_path: Path) -> None:
    from trade_rl.integrations.cross_asset.fetch import UrllibTransport

    store = cli_module.CrossAssetStore(tmp_path)
    (record,) = fetch_all(
        store,
        (series_by_key("T10Y2Y"),),
        UrllibTransport(),
        sleep=lambda _s: None,
    )
    assert record["status"] == "ok"
    assert int(str(record["n_obs"])) > 1_000
