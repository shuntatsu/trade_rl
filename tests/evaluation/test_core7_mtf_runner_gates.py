"""Fail-closed runner gates: authorization, causality probe, ledger reconcile, drawdown stop.

Synthetic inputs only. No market data is read.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.data.test_mtf6_signals import _make_alternating_dataset
from trade_rl.data.features.mtf6_signals import (
    Mtf6SignalPack,
    build_mtf6_signal_pack,
    enforce_pre_fs1_boundary,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import (
    INVALID,
    PASS_PARITY_ONLY,
    STOP_FAIL_NO_RESCUE,
    SYNTHETIC_ONLY_NOT_EVIDENCE,
    UNAUTHENTICATED_NOT_EVIDENCE,
    AuthorizationReceipt,
    CellSummary,
    decide,
    protocol,
    run_causality_probe,
    run_study,
)
from trade_rl.evaluation.core7_mtf_study import runner as study_runner
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.rl.shadow_anchor_mtf import (
    ShadowAnchorMlpWeights,
    ShadowAnchorMtfConfig,
    ShadowAnchorMtfPolicy,
)

AUTH_URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1"


def _synthetic(ds: MarketDataset) -> MarketDataset:
    return protocol.mark_synthetic_dataset(ds)


def _digest() -> str:
    return protocol.protocol_digest()


def _receipt(**over: Any) -> AuthorizationReceipt:
    fields: dict[str, Any] = {
        "authorization_url": AUTH_URL,
        "comment_id": 1,
        "author_login": "shuntatsu",
        "author_id": 148511409,
        "body_sha256": "a" * 64,
        "fetched_at": "2026-11-20T00:00:00+00:00",
        "quoted_protocol_digest": protocol.protocol_digest(),
    }
    fields.update(over)
    return AuthorizationReceipt(**fields)


# --------------------------------------------------------------------------- B1


def test_synthetic_marker_is_content_bound() -> None:
    ds = _make_alternating_dataset(n_bars=30, pairs=("BTCUSDT", "ETHUSDT"), seed=3)
    assert protocol.is_synthetic_dataset(ds) is False
    marked = _synthetic(ds)
    assert protocol.is_synthetic_dataset(marked) is True
    # any content change breaks the marker
    close = np.asarray(marked.close, dtype=np.float64).copy()
    close[5, 0] *= 1.0000001
    close[5, 0] = min(close[5, 0], float(marked.high[5, 0]))
    tampered = replace(marked, close=close)
    assert protocol.is_synthetic_dataset(tampered) is False
    # ... including arrays outside the OHLCV core: mark_price, available_at, features,
    # funding_price_rate
    mark = np.asarray(marked.resolved_array("mark_price"), dtype=np.float64).copy()
    mark[5, 0] = mark[5, 0] * 1.001
    assert protocol.is_synthetic_dataset(replace(marked, mark_price=mark)) is False
    feats = np.asarray(marked.features).copy()
    feats[3, 0, 0] += 1.0
    assert protocol.is_synthetic_dataset(replace(marked, features=feats)) is False
    ts = np.asarray(marked.timestamps, dtype="datetime64[ns]")
    late = np.broadcast_to(ts[:, None], (marked.n_bars, marked.n_symbols)).copy()
    late[7, :] += np.timedelta64(1, "h")
    info = np.ones((marked.n_bars, marked.n_symbols), dtype=bool)
    info[7, :] = False
    delayed = replace(marked, available_at=late, information_available=info)
    assert protocol.is_synthetic_dataset(delayed) is False
    fpr = np.zeros((marked.n_bars, marked.n_symbols), dtype=np.float64)
    assert (
        protocol.is_synthetic_dataset(replace(marked, funding_price_rate=fpr)) is False
    )


def test_run_study_refuses_synthetic_bypass_on_unmarked_dataset(
    tmp_path: Path,
) -> None:
    ds = _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    with pytest.raises(ValueError, match="synthetic"):
        run_study(
            ds,
            output_dir=tmp_path / "unauth",
            expected_protocol_digest=_digest(),
            allow_synthetic_unauthenticated=True,
            start_index=8,
            stop_index=48,
            fit_stop_index=8,
        )
    # post-embargo data with the embargo switched off and a well-formed URL
    ds_future = _make_alternating_dataset(
        n_bars=40,
        pairs=("BTCUSDT", "ETHUSDT"),
        start_iso="2026-09-01T00:00:00",
        seed=88,
    )
    with pytest.raises(ValueError, match="always enforces the Pre-FS1 embargo"):
        run_study(
            ds_future,
            output_dir=tmp_path / "embargo_off",
            expected_protocol_digest=_digest(),
            authorization_url=AUTH_URL,
            enforce_pre_fs1=False,
            start_index=8,
            stop_index=38,
            fit_stop_index=8,
        )
    assert not (tmp_path / "unauth").exists()


def test_marked_post_embargo_url_less_run_is_refused(tmp_path: Path) -> None:
    """Haiku's case: payload-free, post-embargo, marked, URL-less, full scenarios."""
    ds_future = _synthetic(
        _make_alternating_dataset(
            n_bars=40,
            pairs=("BTCUSDT", "ETHUSDT"),
            start_iso="2026-09-01T00:00:00",
            seed=88,
        )
    )
    assert protocol.is_synthetic_dataset(ds_future)
    common: dict[str, Any] = {
        "expected_protocol_digest": _digest(),
        "allow_synthetic_unauthenticated": True,
        "scenarios": protocol.SCENARIOS,
        "start_index": 8,
        "stop_index": 38,
        "fit_stop_index": 8,
    }
    with pytest.raises(ValueError, match="Pre-FS1 embargo"):
        run_study(ds_future, output_dir=tmp_path / "default", **common)
    with pytest.raises(ValueError, match="always enforces the Pre-FS1 embargo"):
        run_study(
            ds_future, output_dir=tmp_path / "off", enforce_pre_fs1=False, **common
        )
    # available_at at/after the embargo with in-window timestamps is refused too
    ds = _synthetic(
        _make_alternating_dataset(n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    )
    ts = np.asarray(ds.timestamps, dtype="datetime64[ns]")
    late = np.broadcast_to(ts[:, None], (ds.n_bars, ds.n_symbols)).copy()
    late[-1, :] = np.datetime64("2026-08-02T00:00:00", "ns")
    info = np.ones((ds.n_bars, ds.n_symbols), dtype=bool)
    info[-1, :] = False
    late_ds = _synthetic(replace(ds, available_at=late, information_available=info))
    with pytest.raises(ValueError, match="Pre-FS1 embargo"):
        run_study(late_ds, output_dir=tmp_path / "late_available", **common)
    assert not (tmp_path / "default").exists()
    assert not (tmp_path / "off").exists()
    assert not (tmp_path / "embargo_off").exists()


def test_authorized_run_requires_full_scenarios_roster_and_embargo(
    tmp_path: Path,
) -> None:
    ds = _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    with pytest.raises(ValueError, match="scenarios must equal"):
        run_study(
            ds,
            output_dir=tmp_path / "subset",
            expected_protocol_digest=_digest(),
            authorization_url=AUTH_URL,
            scenarios=("BASE",),
            start_index=8,
            stop_index=48,
            fit_stop_index=8,
        )
    with pytest.raises(ValueError, match="scenarios must equal"):
        run_study(
            ds,
            output_dir=tmp_path / "reordered",
            expected_protocol_digest=_digest(),
            authorization_url=AUTH_URL,
            scenarios=tuple(reversed(protocol.SCENARIOS)),
            start_index=8,
            stop_index=48,
            fit_stop_index=8,
        )
    # an authorized run can neither waive the full roster nor the embargo
    with pytest.raises(ValueError, match="ROSTER"):
        run_study(
            ds,
            output_dir=tmp_path / "roster",
            expected_protocol_digest=_digest(),
            authorization_url=AUTH_URL,
            require_full_roster=False,
            start_index=8,
            stop_index=48,
            fit_stop_index=8,
        )


def test_protocol_digest_binding_is_required_and_checked(tmp_path: Path) -> None:
    ds = _synthetic(
        _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    )
    kwargs: dict[str, Any] = {
        "allow_synthetic_unauthenticated": True,
        "start_index": 8,
        "stop_index": 48,
        "fit_stop_index": 8,
    }
    with pytest.raises(TypeError):
        run_study(ds, output_dir=tmp_path / "missing", **kwargs)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="protocol digest"):
        run_study(
            ds,
            output_dir=tmp_path / "wrong",
            expected_protocol_digest="0" * 64,
            **kwargs,
        )
    assert not (tmp_path / "wrong").exists()


def test_marked_synthetic_run_succeeds_and_records_binding(tmp_path: Path) -> None:
    ds = _synthetic(
        _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    )
    res = run_study(
        ds,
        output_dir=tmp_path / "ok",
        expected_protocol_digest=_digest(),
        allow_synthetic_unauthenticated=True,
        start_index=8,
        stop_index=48,
        fit_stop_index=8,
    )
    # a URL-less run is never parity/stop evidence, whatever the cells say
    assert res["outcome"] == SYNTHETIC_ONLY_NOT_EVIDENCE
    assert res["underlying_outcome"] == PASS_PARITY_ONLY
    assert res["evidence_eligible"] is False
    assert res["synthetic_only"] is True
    assert res["protocol_digest"] == _digest()
    assert res["scenarios"] == list(protocol.SCENARIOS)
    import json

    manifest = json.loads((tmp_path / "ok" / "study-manifest.json").read_text())
    decision_file = json.loads((tmp_path / "ok" / "decision.json").read_text())
    assert manifest["synthetic_only"] is True
    assert decision_file["synthetic_only"] is True
    assert decision_file["outcome"] == SYNTHETIC_ONLY_NOT_EVIDENCE


def test_decide_url_less_never_reports_stop_or_pass() -> None:
    grid = {
        (arm, sc): _good_cell(arm, sc)
        for arm in protocol.ARMS
        for sc in protocol.SCENARIOS
    }
    res = decide(grid, evidence_eligible=False)
    assert res["outcome"] == SYNTHETIC_ONLY_NOT_EVIDENCE
    grid[("CORE7_SPARSE_PPO", "BASE")] = _good_cell(
        "CORE7_SPARSE_PPO", "BASE", max_drawdown=0.35
    )
    stopped = decide(grid, evidence_eligible=False)
    assert stopped["outcome"] == SYNTHETIC_ONLY_NOT_EVIDENCE
    assert stopped["underlying_outcome"] == STOP_FAIL_NO_RESCUE
    grid[("CASH", "BASE")] = _good_cell("CASH", "BASE", ledger_reconciled=False)
    assert decide(grid, evidence_eligible=False)["outcome"] == INVALID
    assert decide(grid, evidence_eligible=True)["outcome"] == INVALID


def test_decide_refuses_pass_without_the_full_scenario_set() -> None:
    cells = {(arm, "BASE"): _good_cell(arm, "BASE") for arm in protocol.ARMS}
    assert (
        decide(cells, scenarios=("BASE",), evidence_eligible=True)["outcome"] == INVALID
    )


# --------------------------------------------------------------------------- B7


def _good_cell(arm: str, scenario: str, **over: Any) -> CellSummary:
    base: dict[str, Any] = {
        "arm": arm,
        "scenario": scenario,
        "total_return": 0.05,
        "annual_sharpe": 1.0,
        "max_drawdown": 0.04,
        "total_cost": 10.0,
        "funding_pnl": 1.0,
        "turnover_total": 2.0,
        "complete": True,
        "terminal_flat": True,
        "execution_valid": True,
        "causality_passed": True,
        "ledger_reconciled": True,
        "stop_reason": None,
        "invalid_reasons": (),
    }
    base.update(over)
    return CellSummary(**base)


def _full_grid(**over_for_cell: Any) -> dict[tuple[str, str], CellSummary]:
    return {
        (arm, sc): _good_cell(arm, sc)
        for arm in protocol.ARMS
        for sc in protocol.SCENARIOS
    } | {
        ("CORE7_SPARSE_PPO", "BASE"): _good_cell(
            "CORE7_SPARSE_PPO", "BASE", **over_for_cell
        )
    }


def test_twenty_percent_drawdown_is_a_hard_stop_not_a_pass() -> None:
    drawdown_stop = protocol.risk_config().drawdown_stop
    assert drawdown_stop == 0.20
    assert (
        decide(_full_grid(max_drawdown=0.19), evidence_eligible=True)["outcome"]
        == PASS_PARITY_ONLY
    )
    for breach in (drawdown_stop, 0.35, 1.0):
        res = decide(_full_grid(max_drawdown=breach), evidence_eligible=True)
        assert res["outcome"] == STOP_FAIL_NO_RESCUE, breach
        assert res["stopped_cells"] == ["CORE7_SPARSE_PPO__BASE"]
    breached = _good_cell("CORE7_SPARSE_PPO", "BASE", max_drawdown=0.35)
    assert breached.to_dict()["cell_valid"] is False
    # an integrity failure still dominates a stop
    inv = decide(
        _full_grid(max_drawdown=0.35, ledger_reconciled=False), evidence_eligible=True
    )
    assert inv["outcome"] == INVALID


# --------------------------------------------------------------------------- B5


def _leaky_factory() -> Any:
    """Policy factory that reads pack row ``index + 1`` whatever the signal lag."""

    class _Leaky(ShadowAnchorMtfPolicy):
        def _validate_snapshot(self, snapshot: Any) -> int:
            super()._validate_snapshot(snapshot)
            return min(snapshot.index + 1, int(self._pack.timestamps.shape[0]) - 1)

    def factory(
        pack: Mtf6SignalPack,
        config: ShadowAnchorMtfConfig,
        weights: ShadowAnchorMlpWeights | None,
    ) -> ShadowAnchorMtfPolicy:
        return _Leaky(signal_pack=pack, config=config, weights=weights)

    return factory


def test_causality_probe_snapshots_use_a_nontrivial_account_path() -> None:
    ds = _make_alternating_dataset(n_bars=80, pairs=("BTCUSDT", "ETHUSDT"), seed=7)
    snaps = study_runner._build_synthetic_prefix_snapshots(
        ds, start_index=8, stop_inclusive=70
    )
    weights = np.asarray([s.account.weights for s in snaps])
    pv = np.asarray([s.account.portfolio_value for s in snaps])
    peak = np.asarray([s.account.peak_value for s in snaps])
    assert np.ptp(weights, axis=0).max() > 0.05  # weights actually move
    dd = 1.0 - pv / peak
    assert dd.max() >= 0.10 > dd.min()  # both sides of drawdown_start occur


@pytest.mark.parametrize("mode", ["anc_bidir_75_25", "core7_sparse_ppo"])
@pytest.mark.parametrize("lag", [0, 1])
def test_causality_probe_catches_row_plus_one_policy_at_any_cut(
    mode: str, lag: int
) -> None:
    ds = _make_alternating_dataset(n_bars=60, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    weights = (
        ShadowAnchorMlpWeights.zero_residual(fit_stop_index=8)
        if mode == "core7_sparse_ppo"
        else None
    )
    kwargs: dict[str, Any] = {
        "policy_mode": mode,
        "start_index": 8,
        "fit_stop_index": 8,
        "weights": weights,
        "signal_lag_bars": lag,
    }
    # a cut that is NOT rebalance-aligned: (28 - 8) % 6 == 2
    clean = run_causality_probe(ds, cut_index=28, **kwargs)
    assert clean.passed and clean.proposal_max_abs_diff == 0.0
    leaky = run_causality_probe(
        ds, cut_index=28, policy_factory=_leaky_factory(), **kwargs
    )
    assert not leaky.passed
    assert leaky.proposal_max_abs_diff > 0.0
    assert leaky.signal_max_abs_diff == 0.0  # the pack itself is causal


def test_causality_probe_compares_hold_tuples() -> None:
    ds = _make_alternating_dataset(n_bars=60, pairs=("BTCUSDT", "ETHUSDT"), seed=88)

    class _HoldLeak(ShadowAnchorMtfPolicy):
        def decide_portfolio(self, snapshot: Any) -> Any:  # type: ignore[override]
            proposal = super().decide_portfolio(snapshot)
            row = min(snapshot.index + 1, int(self._pack.timestamps.shape[0]) - 1)
            if int(abs(float(self._pack.perp_ret_causal[row, 0])) * 1e9) % 2 == 0:
                return proposal
            return replace(proposal, hold_current_quantity_symbols=())

    def factory(
        pack: Mtf6SignalPack,
        config: ShadowAnchorMtfConfig,
        weights: ShadowAnchorMlpWeights | None,
    ) -> ShadowAnchorMtfPolicy:
        return _HoldLeak(signal_pack=pack, config=config, weights=weights)

    probe = run_causality_probe(
        ds,
        cut_index=40,
        policy_mode="anc_bidir_75_25",
        start_index=8,
        fit_stop_index=8,
        policy_factory=factory,
    )
    assert not probe.passed


def test_run_study_flags_a_leaky_policy_per_scenario_lag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _synthetic(
        _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    )
    original = ShadowAnchorMtfPolicy._validate_snapshot

    def leaky(self: ShadowAnchorMtfPolicy, snapshot: Any) -> int:
        original(self, snapshot)
        return min(snapshot.index + 1, int(self._pack.timestamps.shape[0]) - 1)

    monkeypatch.setattr(ShadowAnchorMtfPolicy, "_validate_snapshot", leaky)
    res = run_study(
        ds,
        output_dir=tmp_path / "leaky",
        expected_protocol_digest=_digest(),
        allow_synthetic_unauthenticated=True,
        start_index=8,
        stop_index=48,
        fit_stop_index=8,
    )
    assert res["outcome"] == INVALID
    cells = res["cells"]
    assert isinstance(cells, dict)
    for arm in ("ANC_BIDIR_75_25", "ANC_REGIME_GATE_NONRL", "CORE7_SPARSE_PPO"):
        for sc in protocol.SCENARIOS:
            assert cells[f"{arm}__{sc}"]["causality_passed"] is False, (arm, sc)


def test_run_study_probes_every_scenario_lag(tmp_path: Path) -> None:
    ds = _synthetic(
        _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    )
    run_study(
        ds,
        output_dir=tmp_path / "clean",
        expected_protocol_digest=_digest(),
        allow_synthetic_unauthenticated=True,
        start_index=8,
        stop_index=48,
        fit_stop_index=8,
    )
    import json

    manifest = json.loads((tmp_path / "clean" / "study-manifest.json").read_text())
    probes = manifest["causality_probes"]
    for arm in protocol.ARMS:
        for sc in protocol.SCENARIOS:
            entry = probes[f"{arm}__{sc}"]
            assert entry["passed"] is True
            assert entry["signal_lag_bars"] == protocol.scenario_signal_delay_bars(sc)


# --------------------------------------------------------------------------- B6


def _replayed_ledger(*, tight_volume: bool = True) -> tuple[Any, MarketDataset]:
    ds_raw = _make_alternating_dataset(
        n_bars=48, pairs=("BTCUSDT", "ETHUSDT"), seed=103
    )
    vol = np.asarray(ds_raw.volume, dtype=np.float64).copy()
    if tight_volume:
        vol[14:18, :] = 150.0
    fr = np.asarray(ds_raw.funding_rate, dtype=np.float64).copy()
    fr[:, 0::2] = 0.0
    fr[:, 1::2] = 0.00025
    ds = replace(ds_raw, dataset_id="c" * 64, volume=vol, funding_rate=fr)
    pack = build_mtf6_signal_pack(ds)
    result = run_shared_portfolio_replay(
        ds,
        ShadowAnchorMtfPolicy(
            signal_pack=pack,
            config=ShadowAnchorMtfConfig(
                mode="core7_sparse_ppo", start_index=10, fit_stop_index=10
            ),
            weights=ShadowAnchorMlpWeights.zero_residual(fit_stop_index=10),
        ),
        start_index=10,
        stop_index=46,
        initial_capital=100_000.0,
        execution_cost=protocol.scenario_cost("BASE"),
        risk=PreTradeRisk(protocol.risk_config()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert result.ledger_evidence is not None
    return result.ledger_evidence, ds


def _corrupt_interval(ledger: Any, position: int, **fields: float) -> Any:
    intervals = list(ledger.intervals.intervals)
    intervals[position] = replace(intervals[position], **fields)
    return replace(
        ledger, intervals=replace(ledger.intervals, intervals=tuple(intervals))
    )


def test_ledger_reconcile_accepts_the_true_ledger_and_rejects_each_corruption() -> None:
    ledger, ds = _replayed_ledger()
    assert study_runner._check_ledger_reconciled(ledger, ds, 100_000.0) is True
    intervals = ledger.intervals.intervals
    assert any(ev.filled_quantity != 0.0 for iv in intervals for ev in iv.order_events)
    mid = len(intervals) // 2
    garbage = {
        "cash_after": intervals[mid].cash_after + 1_000.0,
        "cash_before": intervals[mid].cash_before - 1_000.0,
        "portfolio_value_after": intervals[mid].portfolio_value_after * 1.01,
        "portfolio_value_before": intervals[mid].portfolio_value_before * 0.99,
        "turnover_total_after": intervals[mid].turnover_total_after + 5.0,
        "total_cost_after": intervals[mid].total_cost_after + 50.0,
        "funding_pnl_after": intervals[mid].funding_pnl_after + 50.0,
        "interval_cost": intervals[mid].interval_cost + 50.0,
        "interval_funding": intervals[mid].interval_funding + 50.0,
    }
    for name, value in garbage.items():
        corrupted = _corrupt_interval(ledger, mid, **{name: value})
        assert (
            study_runner._check_ledger_reconciled(corrupted, ds, 100_000.0) is False
        ), name
    # first-interval cash must start at the initial capital
    first = _corrupt_interval(ledger, 0, cash_before=99_000.0)
    assert study_runner._check_ledger_reconciled(first, ds, 100_000.0) is False
    assert study_runner._check_ledger_reconciled(ledger, ds, 90_000.0) is False
    # terminal aggregates
    for name in ("final_cash", "final_portfolio_value", "final_total_cost"):
        bad = replace(
            ledger,
            intervals=replace(
                ledger.intervals, **{name: getattr(ledger.intervals, name) + 10.0}
            ),
        )
        assert study_runner._check_ledger_reconciled(bad, ds, 100_000.0) is False, name
    bad_turn = replace(
        ledger,
        intervals=replace(
            ledger.intervals,
            final_turnover_total=ledger.intervals.final_turnover_total + 1.0,
        ),
    )
    assert study_runner._check_ledger_reconciled(bad_turn, ds, 100_000.0) is False


def test_ledger_reconcile_detects_a_dropped_or_phantom_fill() -> None:
    ledger, ds = _replayed_ledger()
    intervals = ledger.intervals.intervals
    pos = next(
        i
        for i, iv in enumerate(intervals)
        if any(ev.filled_quantity != 0.0 for ev in iv.order_events)
    )
    events = list(intervals[pos].order_events)
    idx = next(i for i, ev in enumerate(events) if ev.filled_quantity != 0.0)
    events[idx] = replace(
        events[idx], filled_quantity=events[idx].filled_quantity * 1.5
    )
    bad = _corrupt_interval(ledger, pos, order_events=tuple(events))
    assert study_runner._check_ledger_reconciled(bad, ds, 100_000.0) is False
    assert study_runner._check_ledger_reconciled(None, ds, 100_000.0) is False


# --------------------------------------------------------------------------- N4 / N2


def test_pre_fs1_boundary_also_checks_available_at() -> None:
    ds = _make_alternating_dataset(n_bars=20, pairs=("BTCUSDT", "ETHUSDT"), seed=5)
    enforce_pre_fs1_boundary(ds)
    late = np.asarray(ds.timestamps, dtype="datetime64[ns]")[:, None] + np.zeros(
        (1, ds.n_symbols), dtype="timedelta64[ns]"
    )
    late = late.copy()
    late[-1, :] = np.datetime64("2026-08-02T00:00:00", "ns")
    info = np.ones((ds.n_bars, ds.n_symbols), dtype=bool)
    info[-1, :] = False  # delayed rows are not "available on time"
    leaked = replace(ds, available_at=late, information_available=info)
    with pytest.raises(ValueError, match="Pre-FS1 embargo"):
        enforce_pre_fs1_boundary(leaked)


def test_micro_arrays_require_shape_and_causal_availability() -> None:
    ds = _make_alternating_dataset(n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=5)
    t_len, p_len = ds.n_bars, 2
    m15 = np.zeros((t_len, p_len, 3))
    m1h = np.zeros((t_len, p_len, 2))
    ts = np.asarray(ds.timestamps, dtype="datetime64[ns]")[:, None] + np.zeros(
        (1, p_len), dtype="timedelta64[ns]"
    )
    with pytest.raises(ValueError, match="micro_available_at"):
        build_mtf6_signal_pack(ds, micro_15m=m15, micro_1h=m1h)
    with pytest.raises(ValueError, match="micro_available_at"):
        build_mtf6_signal_pack(
            ds, micro_15m=m15, micro_1h=m1h, micro_available_at=ts[:5]
        )
    late = ts.copy()
    late[10, :] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="after the 4h row"):
        build_mtf6_signal_pack(ds, micro_15m=m15, micro_1h=m1h, micro_available_at=late)
    with pytest.raises(ValueError, match="finite"):
        bad = m15.copy()
        bad[3, 0, 0] = np.inf
        build_mtf6_signal_pack(ds, micro_15m=bad, micro_1h=m1h, micro_available_at=ts)
    pack = build_mtf6_signal_pack(
        ds, micro_15m=m15, micro_1h=m1h, micro_available_at=ts
    )
    assert pack.core7_signals.shape[0] == t_len


# ------------------------------------------------------- identity-carrying datasets


def test_probe_and_authorized_study_accept_identity_payload_datasets(
    tmp_path: Path,
) -> None:
    """A real dataset carries an identity payload; the perturbed copy must not trip it."""
    import json

    ds = _make_alternating_dataset(n_bars=60, pairs=protocol.ROSTER, seed=21)
    real = ds.with_content_identity({"source": "unit-test-fixture"})
    assert real.identity_payload_json is not None and real.identity_verified
    original_id = real.dataset_id

    probe = run_causality_probe(
        real,
        cut_index=30,
        policy_mode="anc_bidir_75_25",
        start_index=8,
        fit_stop_index=8,
    )
    assert probe.passed
    assert real.dataset_id == original_id  # identity of the input is untouched

    # an identity-bearing dataset can never be relabelled synthetic
    with pytest.raises(ValueError, match="identity"):
        protocol.mark_synthetic_dataset(real)
    assert protocol.is_synthetic_dataset(real) is False

    res = run_study(
        real,
        output_dir=tmp_path / "authorized",
        expected_protocol_digest=_digest(),
        authorization_url=AUTH_URL,
        authorization_receipt=_receipt(),
        start_index=8,
        stop_index=58,
        fit_stop_index=8,
    )
    assert res["outcome"] == PASS_PARITY_ONLY
    assert res["synthetic_only"] is False
    assert res["authorization_authenticated"] is True
    assert res["dataset_id"] == original_id
    manifest = json.loads(
        (tmp_path / "authorized" / "study-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_id"] == original_id
    assert manifest["authorization_url"] == AUTH_URL
    assert manifest["authorization_check"] == "receipt_shape_and_binding"
    assert manifest["authorization_authenticated"] is True
    assert manifest["authorization_receipt"]["author_id"] == 148511409
    assert manifest["evidence_eligible"] is True


def test_fabricated_url_without_receipt_is_not_evidence(tmp_path: Path) -> None:
    """Haiku's case: unmarked 11-symbol pre-embargo data, well-formed fabricated URL."""
    import json

    ds = _make_alternating_dataset(
        n_bars=60, pairs=protocol.ROSTER, seed=21, start_iso="2026-05-01T00:00:00"
    )
    assert protocol.is_synthetic_dataset(ds) is False
    res = run_study(
        ds,
        output_dir=tmp_path / "fabricated",
        expected_protocol_digest=_digest(),
        authorization_url="https://github.com/shuntatsu/trade_rl/issues/1#issuecomment-999",
        start_index=8,
        stop_index=58,
        fit_stop_index=8,
    )
    assert res["outcome"] == UNAUTHENTICATED_NOT_EVIDENCE
    assert res["underlying_outcome"] == PASS_PARITY_ONLY
    assert res["evidence_eligible"] is False
    assert res["authorization_authenticated"] is False
    manifest = json.loads(
        (tmp_path / "fabricated" / "study-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["authorization_check"] == "url_format_only"
    assert manifest["authorization_authenticated"] is False
    assert manifest["authorization_receipt"] is None
    assert manifest["evidence_eligible"] is False


def test_authorization_receipt_must_bind_url_digest_and_have_a_valid_shape(
    tmp_path: Path,
) -> None:
    ds = _make_alternating_dataset(
        n_bars=60, pairs=protocol.ROSTER, seed=21, start_iso="2026-05-01T00:00:00"
    )
    other_url = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-2"

    def attempt(name: str, receipt: Any, url: str | None = AUTH_URL) -> None:
        run_study(
            ds,
            output_dir=tmp_path / name,
            expected_protocol_digest=_digest(),
            authorization_url=url,
            authorization_receipt=receipt,
            start_index=8,
            stop_index=58,
            fit_stop_index=8,
        )

    with pytest.raises(ValueError, match="receipt URL does not match"):
        attempt("url_mismatch", _receipt(authorization_url=other_url, comment_id=2))
    with pytest.raises(ValueError, match="quoted protocol digest"):
        attempt("digest_mismatch", _receipt(quoted_protocol_digest="0" * 64))
    with pytest.raises(ValueError, match="comment_id does not match"):
        attempt("id_mismatch", _receipt(comment_id=7))
    for name, over in (
        ("bad_author", {"author_login": "bad login!"}),
        ("bad_author_id", {"author_id": 0}),
        ("bool_id", {"author_id": True}),
        ("bad_body", {"body_sha256": "xyz"}),
        ("naive_time", {"fetched_at": "2026-11-20T00:00:00"}),
        ("bad_time", {"fetched_at": "yesterday"}),
    ):
        with pytest.raises(ValueError, match="authorization receipt"):
            attempt(name, _receipt(**over))
    with pytest.raises(TypeError, match="AuthorizationReceipt"):
        attempt("not_a_receipt", {"comment_id": 1})
    with pytest.raises(ValueError, match="requires authorization_url"):
        attempt("receipt_without_url", _receipt(), url=None)
    for name in (
        "url_mismatch",
        "digest_mismatch",
        "id_mismatch",
        "receipt_without_url",
        "not_a_receipt",
    ):
        assert not (tmp_path / name).exists()


def test_decide_evidence_eligible_is_required() -> None:
    with pytest.raises(TypeError, match="evidence_eligible"):
        decide({})  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="non-evidence label"):
        decide({}, evidence_eligible=False, non_evidence_label="PASS_PARITY_ONLY")
    grid = {
        (arm, sc): _good_cell(arm, sc)
        for arm in protocol.ARMS
        for sc in protocol.SCENARIOS
    }
    res = decide(
        grid,
        evidence_eligible=False,
        non_evidence_label=UNAUTHENTICATED_NOT_EVIDENCE,
    )
    assert res["outcome"] == UNAUTHENTICATED_NOT_EVIDENCE
    assert res["underlying_outcome"] == PASS_PARITY_ONLY


def test_pre_fs1_boundary_refuses_misaligned_row_arrays() -> None:
    ds = _make_alternating_dataset(n_bars=20, pairs=("BTCUSDT", "ETHUSDT"), seed=5)
    fake = SimpleNamespace(
        timestamps=ds.timestamps,
        available_at=None,
        features=np.zeros((ds.n_bars + 1, ds.n_symbols, 1)),
        global_features=ds.global_features,
        funding_rate=ds.funding_rate,
        funding_price_rate=None,
    )
    with pytest.raises(ValueError, match="features rows are not aligned"):
        enforce_pre_fs1_boundary(fake)  # type: ignore[arg-type]
    fake.features = ds.features
    fake.funding_price_rate = np.zeros((ds.n_bars - 1, ds.n_symbols))
    with pytest.raises(ValueError, match="funding_price_rate rows are not aligned"):
        enforce_pre_fs1_boundary(fake)  # type: ignore[arg-type]
    fake.funding_price_rate = None
    enforce_pre_fs1_boundary(fake)  # type: ignore[arg-type]


def test_micro_available_at_is_embargoed() -> None:
    ds = _make_alternating_dataset(n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=5)
    ts = np.asarray(ds.timestamps, dtype="datetime64[ns]")[:, None] + np.zeros(
        (1, 2), dtype="timedelta64[ns]"
    )
    late = ts.copy()
    late[10, :] = np.datetime64("2026-08-02T00:00:00", "ns")
    with pytest.raises(ValueError, match="micro_available_at must be strictly before"):
        build_mtf6_signal_pack(
            ds,
            micro_15m=np.zeros((40, 2, 3)),
            micro_1h=np.zeros((40, 2, 2)),
            micro_available_at=late,
        )


def test_run_causality_probe_has_no_embargo_switch() -> None:
    import inspect

    assert "enforce_pre_fs1" not in inspect.signature(run_causality_probe).parameters


def _shift_turnover(ledger: Any, from_position: int, delta: float) -> Any:
    """Shift cumulative turnover from ``from_position`` on, keeping every chain intact."""
    intervals = list(ledger.intervals.intervals)
    for i in range(from_position, len(intervals)):
        iv = intervals[i]
        before_shift = delta if i > from_position else 0.0
        intervals[i] = replace(
            iv,
            turnover_total_before=iv.turnover_total_before + before_shift,
            turnover_total_after=iv.turnover_total_after + delta,
        )
    inner = replace(
        ledger.intervals,
        intervals=tuple(intervals),
        final_turnover_total=ledger.intervals.final_turnover_total + delta,
    )
    return replace(ledger, intervals=inner)


def test_ledger_reconcile_turnover_rebuild_is_load_bearing() -> None:
    """A chain-consistent turnover error is caught only by the per-interval rebuild."""
    ledger, ds = _replayed_ledger()
    intervals = ledger.intervals.intervals
    pos = next(
        i
        for i, iv in enumerate(intervals)
        if any(ev.filled_quantity != 0.0 for ev in iv.order_events)
    )
    shifted = _shift_turnover(ledger, pos, 3.0)
    # chain and terminal aggregates still agree with each other
    assert (
        shifted.intervals.intervals[-1].turnover_total_after
        == shifted.intervals.final_turnover_total
    )
    assert study_runner._check_ledger_reconciled(shifted, ds, 100_000.0) is False
    assert study_runner._check_ledger_reconciled(ledger, ds, 100_000.0) is True


def _fake_replay(max_drawdown: float, termination: tuple[str, ...] = ()) -> Any:
    return SimpleNamespace(
        ledger_evidence=None,
        book=SimpleNamespace(
            quantities=(0.0, 0.0),
            insolvent=False,
            max_drawdown=max_drawdown,
            portfolio_value=90_000.0,
        ),
        diagnostics=SimpleNamespace(
            termination_reasons=termination,
            total_cost=1.0,
            funding_pnl=0.0,
            turnover_total=1.0,
        ),
        returns=SimpleNamespace(values=(0.01, -0.01, 0.0)),
    )


def test_runner_emits_max_drawdown_hard_stop_reason() -> None:
    ds = _make_alternating_dataset(n_bars=20, pairs=("BTCUSDT", "ETHUSDT"), seed=1)

    def summarize(result: Any) -> CellSummary:
        return study_runner._summarize_replay_cell(
            arm="CORE7_SPARSE_PPO",
            scenario="BASE",
            result=result,
            dataset=ds,
            initial_capital=100_000.0,
            causality_passed=True,
        )

    breached = summarize(_fake_replay(0.35))
    assert breached.stop_reason == "max_drawdown_hard_stop"
    assert breached.complete is True  # a stop is not an integrity failure
    healthy = summarize(_fake_replay(0.19))
    assert healthy.stop_reason is None
    # an engine termination reason takes precedence over the derived one
    engine = summarize(_fake_replay(0.35, ("insolvent",)))
    assert engine.stop_reason == "insolvent"
