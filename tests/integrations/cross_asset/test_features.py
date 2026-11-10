"""cross_asset_features_v1: trailing-window oracles, alignment and leakage."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import numpy as np
import pytest

from tests.integrations.cross_asset.support import (
    OPT_IN_ALL,
    obs_series,
    random_walk,
    weekdays,
)
from trade_rl.integrations.cross_asset import features as features_module
from trade_rl.integrations.cross_asset.catalog import CATALOG, SeriesKind
from trade_rl.integrations.cross_asset.features import (
    COMPOSITE_FEATURE_NAMES,
    COMPOSITES,
    FEATURE_SET_ID,
    compute_frame,
    definitions,
    definitions_digest,
    series_feature_arrays,
    series_suffixes,
)
from trade_rl.integrations.cross_asset.pit import knowable_at_utc

N = 400
CATALOG_BY_KEY = {spec.key: spec for spec in CATALOG}


def _nan_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def test_price_features_match_an_independent_oracle() -> None:
    days = weekdays(N)
    x = random_walk(N, seed=3)
    arrays = series_feature_arrays(obs_series("NASDAQCOM", days, x))
    logx = np.log(x)
    step = np.diff(logx, prepend=np.nan)
    i = 300
    for window in (1, 5, 20, 60):
        assert math.isclose(arrays[f"ret{window}"][i], logx[i] - logx[i - window])
    assert np.isnan(arrays["ret60"][59]) and not np.isnan(arrays["ret60"][60])
    assert math.isclose(arrays["rvol20"][i], np.std(step[i - 19 : i + 1], ddof=1))
    window = logx[i - 251 : i + 1]
    assert math.isclose(
        arrays["z252"][i], (logx[i] - window.mean()) / window.std(ddof=1)
    )
    scale = np.std(step[i - 251 : i + 1], ddof=1)
    expected = (logx[i] - logx[i - 20]) / (scale * math.sqrt(20))
    assert math.isclose(arrays["sm20"][i], float(np.clip(expected, -5, 5)))
    assert arrays["abv60"][i] == float(x[i] > x[i - 59 : i + 1].mean())
    assert arrays["abv200"][i] == float(x[i] > x[i - 199 : i + 1].mean())
    assert np.isnan(arrays["abv200"][198]) and not np.isnan(arrays["abv200"][199])


def test_level_features_use_differences_and_expose_the_level() -> None:
    days = weekdays(N)
    level = 3.0 + np.cumsum(0.05 * np.random.default_rng(5).standard_normal(N))
    arrays = series_feature_arrays(obs_series("DGS10", days, level))
    assert set(arrays) == set(series_suffixes(SeriesKind.LEVEL))
    i = 250
    assert math.isclose(arrays["chg5"][i], level[i] - level[i - 5])
    assert np.array_equal(arrays["lvl"], level)
    assert "abv60" not in arrays and "ret1" not in arrays


def test_every_feature_is_trailing_so_prefixes_reproduce_exactly() -> None:
    days = weekdays(N)
    x = random_walk(N, seed=11)
    full = series_feature_arrays(obs_series("SP500", days, x))
    for cut in (70, 130, 260, 399):
        prefix = series_feature_arrays(obs_series("SP500", days[:cut], x[:cut]))
        for name, array in prefix.items():
            assert _nan_equal(array, full[name][:cut]), (name, cut)


def test_changing_the_future_never_changes_earlier_rows() -> None:
    days = weekdays(N)
    x = random_walk(N, seed=12)
    base = series_feature_arrays(obs_series("VIXCLS", days, x))
    changed_values = x.copy()
    changed_values[300:] *= 1.7
    changed = series_feature_arrays(obs_series("VIXCLS", days, changed_values))
    for name in base:
        assert _nan_equal(base[name][:300], changed[name][:300]), name


def test_non_positive_prices_do_not_produce_inf_or_leak_into_other_rows() -> None:
    days = weekdays(N)
    x = random_walk(N, seed=13)
    x[200] = -37.0
    arrays = series_feature_arrays(obs_series("DCOILWTICO", days, x))
    for name, array in arrays.items():
        assert not np.isinf(array).any(), name
    assert np.isnan(arrays["ret1"][200]) and np.isnan(arrays["ret1"][201])
    assert not np.isnan(arrays["ret1"][202])
    assert not np.isnan(arrays["ret5"][204]) and np.isnan(arrays["ret5"][205])
    assert not np.isnan(arrays["ret5"][206])


def _grid(start: str, end: str, step_hours: int) -> np.ndarray:
    return np.arange(
        np.datetime64(start, "ns"),
        np.datetime64(end, "ns"),
        np.timedelta64(step_hours, "h"),
    )


def _equity_book(seed: int = 0, count: int = N) -> dict[str, object]:
    days = weekdays(count)
    keys = (
        "SP500",
        "NASDAQCOM",
        "NIKKEI225",
        "DTWEXBGS",
        "DEXJPUS",
        "DEXUSEU",
        "VIXCLS",
        "T10Y2Y",
    )
    book = {}
    for offset, key in enumerate(keys):
        values = random_walk(count, seed=seed + offset)
        if key == "T10Y2Y":
            values = 0.4 + 0.02 * (values - 100.0)
        book[key] = obs_series(key, days, values)
    return book  # type: ignore[return-value]


def test_aligned_value_is_the_latest_row_released_by_the_decision_time() -> None:
    book = _equity_book()
    series = book["NASDAQCOM"]
    grid = _grid("2024-03-01T00:00", "2024-03-12T00:00", 4)  # spans a weekend
    frame = compute_frame({"NASDAQCOM": series}, grid, opt_in_series=OPT_IN_ALL)  # type: ignore[arg-type]
    arrays = series_feature_arrays(series)  # type: ignore[arg-type]
    column = frame.column("NASDAQCOM__ret1")
    stale_col = frame.column("NASDAQCOM__rvol20")
    for row, instant in enumerate(grid):
        t = instant.astype(np.int64)
        released = np.flatnonzero(series.knowable_ns <= t)  # type: ignore[attr-defined]
        assert released.size
        last = released[-1]
        assert frame.values[row, column] == arrays["ret1"][last]
        hours = (t - series.close_ns[last]) / 3.6e12  # type: ignore[attr-defined]
        assert math.isclose(frame.staleness_hours[row, stale_col], hours)
        # The observation of calendar date d is never visible before the end of
        # the next publisher business day.
        obs_day = series.dates[last].astype(date)  # type: ignore[attr-defined]
        stamp = knowable_at_utc(
            obs_day,
            venue_tz=series.spec.venue_tz,  # type: ignore[attr-defined]
            close_local=series.spec.close_local,  # type: ignore[attr-defined]
            rule=series.spec.release,  # type: ignore[attr-defined]
        )
        assert stamp <= datetime.fromtimestamp(t / 1e9, UTC)


def test_weekend_bars_carry_the_last_released_value_with_growing_staleness() -> None:
    book = _equity_book()
    series = book["NASDAQCOM"]
    grid = _grid("2024-03-08T00:00", "2024-03-12T00:00", 4)
    frame = compute_frame({"NASDAQCOM": series}, grid, opt_in_series=OPT_IN_ALL)  # type: ignore[arg-type]
    column = frame.column("NASDAQCOM__ret1")
    values = frame.values[:, column]
    stale = frame.staleness_hours[:, column]
    assert np.isfinite(values).all()
    # Value changes only at release instants; staleness rises in between.
    changes = np.flatnonzero(np.diff(values) != 0.0) + 1
    for start, stop in zip(
        [0, *changes.tolist()], [*changes.tolist(), grid.size], strict=True
    ):
        assert np.all(np.diff(stale[start:stop]) > 0.0)
    drops = np.flatnonzero(np.diff(stale) < 0.0)
    assert set(drops + 1) <= set(changes.tolist())


def test_values_older_than_the_staleness_limit_become_unavailable() -> None:
    book = _equity_book()
    series = book["NASDAQCOM"]
    last_close = series.close_ns[-1]  # type: ignore[attr-defined]
    last_release = series.knowable_ns[-1]  # type: ignore[attr-defined]
    limit = series.spec.max_staleness_hours  # type: ignore[attr-defined]
    grid = np.asarray(
        [
            last_release - 1,
            last_release,
            last_close + int(limit * 3.6e12) - 1,
            last_close + int(limit * 3.6e12) + 10**9,
        ],
        dtype="datetime64[ns]",
    )
    frame = compute_frame({"NASDAQCOM": series}, grid, opt_in_series=OPT_IN_ALL)  # type: ignore[arg-type]
    available = frame.available[:, frame.column("NASDAQCOM__ret1")]
    # Before the last release the previous row is used; after the limit, nothing.
    assert available.tolist() == [True, True, True, False]
    assert np.isnan(frame.values[3, frame.column("NASDAQCOM__ret1")])


def test_decision_times_before_the_first_release_are_unavailable() -> None:
    book = _equity_book()
    series = book["NASDAQCOM"]
    early = np.asarray([series.knowable_ns[0] - 1], dtype="datetime64[ns]")  # type: ignore[attr-defined]
    frame = compute_frame({"NASDAQCOM": series}, early, opt_in_series=OPT_IN_ALL)  # type: ignore[arg-type]
    assert not frame.available.any()


def test_no_feature_row_changes_when_unreleased_observations_change() -> None:
    book = _equity_book(seed=21)
    grid = _grid("2024-04-01T00:00", "2024-10-01T00:00", 24)
    cutoff = np.datetime64("2024-07-01T00:00", "ns")
    cutoff_ns = int(cutoff.astype(np.int64))
    base = compute_frame(book, grid, opt_in_series=OPT_IN_ALL)  # type: ignore[arg-type]
    rng = np.random.default_rng(99)
    mutated = {}
    for key, series in book.items():
        values = series.values.copy()  # type: ignore[attr-defined]
        future = series.knowable_ns > cutoff_ns  # type: ignore[attr-defined]
        assert future.any()
        values[future] = values[future] * (1.0 + rng.uniform(0.2, 2.0, future.sum()))
        mutated[key] = obs_series(key, series.dates, values)  # type: ignore[attr-defined]
    after = compute_frame(mutated, grid, opt_in_series=OPT_IN_ALL)
    rows = grid <= cutoff
    assert rows.any() and (~rows).any()
    assert _nan_equal(base.values[rows], after.values[rows])
    assert np.array_equal(base.available[rows], after.available[rows])
    assert _nan_equal(base.staleness_hours[rows], after.staleness_hours[rows])
    assert not _nan_equal(base.values[~rows], after.values[~rows])


def _trend_book(signs: dict[str, int]) -> dict[str, object]:
    days = weekdays(N)
    rng = np.random.default_rng(4)
    book = {}
    for key, sign in signs.items():
        noise = 0.004 * rng.standard_normal(N)
        trend = np.zeros(N)
        trend[-60:] = sign * 0.012
        book[key] = obs_series(key, days, 100.0 * np.exp(np.cumsum(noise + trend)))
    return book  # type: ignore[return-value]


def _last_row(book: dict[str, object]) -> tuple[object, int]:
    end = min(s.knowable_ns[-1] for s in book.values())  # type: ignore[attr-defined]
    grid = np.asarray([end], dtype="datetime64[ns]")
    return compute_frame(book, grid, opt_in_series=OPT_IN_ALL), 0  # type: ignore[arg-type]


def test_dollar_strength_has_the_economic_sign_for_each_member() -> None:
    strong = _trend_book(
        {"DTWEXBGS": 1, "DEXJPUS": 1, "DEXCHUS": 1, "DEXUSEU": -1, "DEXUSUK": -1}
    )
    frame, row = _last_row(strong)
    column = frame.column("dollar_strength")  # type: ignore[attr-defined]
    assert frame.available[row, column]  # type: ignore[attr-defined]
    assert frame.values[row, column] > 1.0  # type: ignore[attr-defined]
    weak = _trend_book(
        {"DTWEXBGS": -1, "DEXJPUS": -1, "DEXCHUS": -1, "DEXUSEU": 1, "DEXUSUK": 1}
    )
    frame, row = _last_row(weak)
    assert frame.values[row, frame.column("dollar_strength")] < -1.0  # type: ignore[attr-defined]


def test_composites_need_their_minimum_member_count() -> None:
    book = _trend_book({"DTWEXBGS": 1, "DEXJPUS": 1})
    frame, row = _last_row(book)
    column = frame.column("dollar_strength")  # type: ignore[attr-defined]
    assert not frame.available[row, column]  # type: ignore[attr-defined]
    assert np.isnan(frame.values[row, column])  # type: ignore[attr-defined]
    assert not frame.available[row, frame.column("risk_appetite")]  # type: ignore[attr-defined]


def test_risk_appetite_rises_with_equities_and_falls_with_volatility() -> None:
    risk_on = _trend_book({"SP500": 1, "NASDAQCOM": 1, "NIKKEI225": 1, "VIXCLS": -1})
    frame, row = _last_row(risk_on)
    assert frame.values[row, frame.column("risk_appetite")] > 1.0  # type: ignore[attr-defined]
    risk_off = _trend_book({"SP500": -1, "NASDAQCOM": -1, "NIKKEI225": -1, "VIXCLS": 1})
    frame, row = _last_row(risk_off)
    assert frame.values[row, frame.column("risk_appetite")] < -1.0  # type: ignore[attr-defined]


def test_trend_breadth_is_the_share_of_series_above_their_moving_average() -> None:
    members = (
        "SP500",
        "NASDAQCOM",
        "NIKKEI225",
        "DTWEXBGS",
        "DCOILWTICO",
        "DCOILBRENTEU",
    )
    rising = {key: 1 for key in members}
    frame, row = _last_row(_trend_book(rising))
    assert frame.values[row, frame.column("trend_breadth_60")] == 1.0  # type: ignore[attr-defined]
    mixed = {key: (1 if index < 4 else -1) for index, key in enumerate(members)}
    frame, row = _last_row(_trend_book(mixed))
    assert math.isclose(
        frame.values[row, frame.column("trend_breadth_60")],  # type: ignore[attr-defined]
        4 / 6,
    )
    sparse = {key: 1 for key in members[:3]}
    frame, row = _last_row(_trend_book(sparse))
    assert not frame.available[row, frame.column("trend_breadth_60")]  # type: ignore[attr-defined]


def test_curve_slope_aliases_the_released_spread_level() -> None:
    book = _equity_book()
    frame, row = _last_row(book)
    level = book["T10Y2Y"].values[-1]  # type: ignore[attr-defined]
    assert frame.values[row, frame.column("curve_slope")] == level  # type: ignore[attr-defined]
    assert (
        frame.values[row, frame.column("curve_slope")]
        == frame.values[  # type: ignore[attr-defined]
            row, frame.column("T10Y2Y__lvl")  # type: ignore[attr-defined]
        ]
    )


def test_frame_lists_composites_first_then_series_in_catalog_order() -> None:
    book = _equity_book()
    frame, _ = _last_row(book)
    names = frame.names  # type: ignore[attr-defined]
    assert names[: len(COMPOSITE_FEATURE_NAMES)] == COMPOSITE_FEATURE_NAMES
    present = [spec for spec in CATALOG if spec.key in book]
    expected = [
        f"{spec.key}__{suffix}"
        for spec in present
        for suffix in series_suffixes(spec.kind)
    ]
    assert list(names[len(COMPOSITE_FEATURE_NAMES) :]) == expected
    assert frame.series_present == tuple(sorted(book))  # type: ignore[attr-defined]
    assert frame.feature_set == FEATURE_SET_ID  # type: ignore[attr-defined]


def test_composite_members_are_known_catalog_series_and_suffixes() -> None:
    keys = {spec.key: spec for spec in CATALOG}
    for composite in COMPOSITES:
        assert composite.min_members >= 1
        for key, suffix, sign in composite.members:
            assert key in keys, (composite.name, key)
            assert suffix in series_suffixes(keys[key].kind), (composite.name, key)
            assert sign in (-1, 1)


def test_definitions_digest_is_stable_and_binds_the_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = definitions_digest()
    assert len(digest) == 64 and digest == definitions_digest()
    assert definitions()["change_windows"] == [1, 5, 20, 60]
    monkeypatch.setattr(features_module, "CHANGE_WINDOWS", (1, 5, 20, 120))
    assert definitions_digest() != digest


def test_frame_rejects_invalid_decision_grids_and_unknown_names() -> None:
    book = _equity_book()
    with pytest.raises(ValueError, match="datetime64"):
        compute_frame(book, np.asarray([1, 2, 3]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NaT"):
        compute_frame(book, np.asarray(["NaT"], dtype="datetime64[ns]"))  # type: ignore[arg-type]
    frame = compute_frame(book, _grid("2024-06-01T00:00", "2024-06-03T00:00", 24))  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        frame.select(["nope"])
    assert frame.select(["dollar_strength"]).values.shape == (2, 1)


def _default_book() -> dict[str, object]:
    keys = (
        "DEXJPUS",
        "DEXCHUS",
        "DEXUSEU",
        "DEXUSUK",
        "DEXUSAL",
        "T10Y2Y",
        "NIKKEI225",
        "VIXCLS",
    )
    days = weekdays(N)
    book = {}
    for offset, key in enumerate(keys):
        book[key] = obs_series(key, days, random_walk(N, seed=40 + offset))
    return book  # type: ignore[return-value]


def test_default_composites_ignore_series_that_are_not_low_revision_risk() -> None:
    low_only = _default_book()
    extra = {}
    days = weekdays(N)
    for offset, key in enumerate(
        (
            "DTWEXBGS",
            "NASDAQCOM",
            "SP500",
            "BAMLH0A0HYM2",
            "DCOILWTICO",
            "y_dxy",
            "y_gdaxi",
            "y_hsi",
        )
    ):
        extra[key] = obs_series(key, days, random_walk(N, seed=70 + offset))
    book = {**low_only, **extra}
    assert all(not CATALOG_BY_KEY[key].default_member for key in extra)
    end = min(s.knowable_ns[-1] for s in book.values())  # type: ignore[attr-defined]
    grid = np.asarray([end], dtype="datetime64[ns]")
    plain = compute_frame(low_only, grid)  # type: ignore[arg-type]
    default = compute_frame(book, grid)  # type: ignore[arg-type]
    for name in COMPOSITE_FEATURE_NAMES:
        assert _nan_equal(
            default.values[:, default.column(name)], plain.values[:, plain.column(name)]
        ), name
    assert default.opt_in_series == ()
    assert default.available[0, default.column("dollar_strength")]
    assert default.available[0, default.column("curve_slope")]
    # Too few default members: these composites stay unavailable by default.
    for name in (
        "risk_appetite",
        "equity_momentum20",
        "commodity_momentum20",
        "trend_breadth_60",
    ):
        assert not default.available[0, default.column(name)], name
    opted = compute_frame(
        book, grid, opt_in_series=("NASDAQCOM", "SP500", "y_gdaxi", "y_hsi")
    )  # type: ignore[arg-type]
    assert opted.opt_in_series == ("NASDAQCOM", "SP500", "y_gdaxi", "y_hsi")
    assert opted.available[0, opted.column("equity_momentum20")]
    assert opted.available[0, opted.column("risk_appetite")]
    assert not _nan_equal(
        opted.values[:, opted.column("risk_appetite")],
        default.values[:, default.column("risk_appetite")],
    )
    # Non-low series have no columns at all unless opted in.
    assert "NASDAQCOM__ret1" not in default.names
    assert "BAMLH0A0HYM2__ret1" not in default.names
    assert "NASDAQCOM__ret1" in opted.names and "SP500__ret1" in opted.names
    assert "y_dxy__ret1" not in opted.names  # not opted in
    with pytest.raises(ValueError, match="unknown opt-in"):
        compute_frame(book, grid, opt_in_series=("nope",))  # type: ignore[arg-type]


def test_revision_risk_labels_carry_evidence_and_futures_are_absent() -> None:
    for spec in CATALOG:
        assert spec.revision_risk in {"low", "possible", "unknown"}
        assert spec.revision_evidence
        if spec.revision_risk == "low":
            assert "ALFRED" in spec.revision_evidence and "of" in spec.revision_evidence
        if spec.source == "yahoo":
            assert spec.revision_risk == "unknown"
        assert not spec.symbol.endswith("=F"), spec.key
    default = {spec.key for spec in CATALOG if spec.default_member}
    assert default == {
        "DGS2",
        "DGS10",
        "T10Y2Y",
        "DFII10",
        "DEXJPUS",
        "DEXUSEU",
        "DEXUSUK",
        "DEXCHUS",
        "DEXUSAL",
        "VIXCLS",
        "NIKKEI225",
    }
    for key in (
        "DTWEXBGS",
        "NASDAQCOM",
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
        "DCOILWTICO",
        "DCOILBRENTEU",
    ):
        assert CATALOG_BY_KEY[key].revision_risk == "possible", key
    assert CATALOG_BY_KEY["SP500"].revision_risk == "unknown"
