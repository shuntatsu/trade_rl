from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import trade_rl.workflows.universal_causal_alpha_v10_stage_entry as stage_entry
from trade_rl.learning.causal_alpha_v10 import (
    CausalAlphaV10Candidate,
    CausalAlphaV10Config,
)
from trade_rl.learning.causal_alpha_v10_hierarchy import CausalAlphaV10BoundaryMode
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.workflows.universal_causal_alpha_v10_stage_entry import (
    _execution_rebalance_contract,
    _path,
    _require_execution_rebalance_contract,
    _training_rows,
    causal_alpha_v10_stage_config_digest,
)


def test_v10_training_rows_bind_fast_and_slow_labels_without_symbol_features() -> None:
    decisions = np.arange(10, 90, 10, dtype=np.int64)
    local = SimpleNamespace(
        feature_names=("return_4h", "volatility"),
        values=np.column_stack((decisions, decisions + 1.0)),
        available=np.ones((len(decisions), 2), dtype=np.bool_),
    )
    global_context = SimpleNamespace(
        feature_names=("market_return",),
        values=decisions[:, None] + 2.0,
        available=np.ones((len(decisions), 1), dtype=np.bool_),
    )
    sample = SimpleNamespace(
        decision_indices=decisions,
        label_end_indices_4h=decisions + 2,
        label_end_indices_72h=decisions + 20,
        labels_4h=decisions / 1_000.0,
        labels_72h=decisions / 100.0,
        local_context=local,
        global_context=global_context,
    )
    prepared = SimpleNamespace(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": sample},
    )

    rows = _training_rows(prepared)["AAAUSDT"]

    assert rows.fast_label_end_indices.tolist() == (decisions + 2).tolist()
    assert rows.slow_label_end_indices.tolist() == (decisions + 20).tolist()
    assert rows.fast_labels.tolist() == (decisions / 1_000.0).tolist()
    assert rows.slow_labels.tolist() == (decisions / 100.0).tolist()
    assert rows.feature_names == (
        "local:return_4h",
        "local:volatility",
        "global:market_return",
    )
    assert all("symbol" not in name for name in rows.feature_names)


def _risk_config(
    *, exit_threshold: float = 0.03, drawdown_start: float = 0.12
) -> PreTradeRiskConfig:
    return PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=0.40,
        max_turnover=0.75,
        entry_threshold=0.10,
        exit_threshold=exit_threshold,
        no_trade_band=0.05,
        drawdown_start=drawdown_start,
        drawdown_stop=0.25,
        emergency_turnover_override=True,
        fail_closed_tolerance=1e-9,
    )


def test_v10_resolves_full_execution_contract_from_environment_and_closes_it() -> None:
    closed: list[bool] = []

    class Environment:
        pre_trade_risk = SimpleNamespace(config=_risk_config())

        def close(self) -> None:
            closed.append(True)

    prepared = SimpleNamespace(
        prepared_v3=SimpleNamespace(
            environment_factories={"BTCUSDT": Environment},
        )
    )

    contract = _execution_rebalance_contract(prepared, "BTCUSDT")

    assert contract.entry_threshold == 0.10
    assert contract.exit_threshold == 0.03
    assert contract.no_trade_band == 0.05
    assert contract.max_turnover == 0.75
    assert contract.drawdown_start == 0.12
    assert contract.drawdown_stop == 0.25
    assert contract.fail_closed_tolerance == 1e-9
    assert closed == [True]


def test_v10_execution_contract_digest_binds_non_entry_pretrade_fields() -> None:
    class EnvironmentA:
        pre_trade_risk = SimpleNamespace(config=_risk_config(drawdown_start=0.12))

        def close(self) -> None:
            pass

    class EnvironmentB:
        pre_trade_risk = SimpleNamespace(config=_risk_config(drawdown_start=0.13))

        def close(self) -> None:
            pass

    prepared = SimpleNamespace(
        prepared_v3=SimpleNamespace(
            environment_factories={
                "AAAUSDT": EnvironmentA,
                "BBBUSDT": EnvironmentB,
            },
        )
    )

    first = _execution_rebalance_contract(prepared, "AAAUSDT")
    second = _execution_rebalance_contract(prepared, "BBBUSDT")

    assert first.entry_threshold == second.entry_threshold == 0.10
    assert first.no_trade_band == second.no_trade_band == 0.05
    assert first.digest != second.digest


def test_v10_replay_rejects_execution_contract_drift_in_exit_threshold() -> None:
    environment = SimpleNamespace(
        pre_trade_risk=SimpleNamespace(config=_risk_config(exit_threshold=0.02))
    )
    expected_environment = SimpleNamespace(
        pre_trade_risk=SimpleNamespace(config=_risk_config(exit_threshold=0.03))
    )
    expected = stage_entry._environment_rebalance_contract(expected_environment)

    with pytest.raises(ValueError, match="execution rebalance contract drifted"):
        _require_execution_rebalance_contract(environment, expected)


def test_v10_closed_loop_replay_uses_new_leaf_schema() -> None:
    assert stage_entry._REPLAY_LEAF_SCHEMA == "causal_alpha_v10_replay_leaf_v2"


def test_v10_hierarchical_replay_has_dedicated_closed_loop_path() -> None:
    assert callable(getattr(stage_entry, "_build_hierarchical_replay", None))


def test_v10_leaf_paths_are_candidate_symbol_episode_scoped() -> None:
    assert _path(
        CausalAlphaV10Candidate.HIERARCHICAL_WAVE, "BTCUSDT", 8
    ).as_posix() == ("selection/replays/08/BTCUSDT/hierarchical_wave.json")


def test_v11_boundary_mode_is_bound_into_stage_identity() -> None:
    source = SimpleNamespace(
        calibration=SimpleNamespace(digest="a" * 64),
        target=SimpleNamespace(digest="b" * 64),
    )
    v8 = SimpleNamespace(digest="c" * 64)
    v9 = SimpleNamespace(digest="d" * 64)
    v10 = CausalAlphaV10Config()

    inherited = causal_alpha_v10_stage_config_digest(
        source,
        v8,
        v9,
        v10,
        boundary_mode=CausalAlphaV10BoundaryMode.INHERIT_CONFIRM,
    )
    boundary_flat = causal_alpha_v10_stage_config_digest(
        source,
        v8,
        v9,
        v10,
        boundary_mode=CausalAlphaV10BoundaryMode.FLATTEN_THEN_RESET,
    )
    neutral_expiry = causal_alpha_v10_stage_config_digest(
        source,
        v8,
        v9,
        v10,
        boundary_mode=CausalAlphaV10BoundaryMode.NEUTRAL_FAST_EXPIRY,
    )
    risk_flatten = causal_alpha_v10_stage_config_digest(
        source,
        v8,
        v9,
        v10,
        boundary_mode=CausalAlphaV10BoundaryMode.FLATTEN_ON_RISK_BREACH,
    )
    fast_only = causal_alpha_v10_stage_config_digest(
        source,
        v8,
        v9,
        v10,
        boundary_mode=CausalAlphaV10BoundaryMode.FAST_ONLY_OWNERSHIP,
    )
    flat_start = causal_alpha_v10_stage_config_digest(
        source,
        v8,
        v9,
        v10,
        boundary_mode=CausalAlphaV10BoundaryMode.FLAT_START_ACTIVATION,
    )

    assert inherited != boundary_flat
    assert inherited != neutral_expiry
    assert boundary_flat != neutral_expiry
    assert risk_flatten not in {inherited, boundary_flat, neutral_expiry}
    assert fast_only not in {inherited, boundary_flat, neutral_expiry, risk_flatten}
    assert flat_start not in {
        inherited,
        boundary_flat,
        neutral_expiry,
        risk_flatten,
        fast_only,
    }
