from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation import ppo_normalization_execution as module
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.ppo_normalization_execution import (
    EXECUTION_ACTIVATION_SCHEMA,
    SEALED_PROTOCOL_SHA256,
    claim_replication_slot,
    execute_replication_slot,
    fit_replication_strategy,
    prepare_replication_execution,
    recompute_replication_decision,
    record_consumed_failure,
    record_prefit_failure,
    replication_arm_specs,
    replication_slot_state,
    replication_strategy_factory,
    verify_replication_slot,
)


def test_replication_roster_is_exactly_ten_fresh_matched_seed_slots() -> None:
    specs = replication_arm_specs()

    assert tuple(spec.slot for spec in specs) == (
        "control_raw_seed0",
        "control_raw_seed1",
        "control_raw_seed2",
        "control_raw_seed3",
        "control_raw_seed4",
        "candidate_normalized_seed0",
        "candidate_normalized_seed1",
        "candidate_normalized_seed2",
        "candidate_normalized_seed3",
        "candidate_normalized_seed4",
    )
    assert len({spec.slot for spec in specs}) == 10

    for seed in range(5):
        control = specs[seed]
        candidate = specs[seed + 5]
        assert control.seed == candidate.seed == seed
        assert control.protocol_arm == "control_raw"
        assert candidate.protocol_arm == "candidate_normalized"
        assert control.normalize_features is False
        assert candidate.normalize_features is True


def test_fit_replication_strategy_changes_only_normalization_for_matched_seed(
    monkeypatch,
) -> None:
    timestamps = np.asarray(
        [
            "2022-12-31T21:00:00",
            "2022-12-31T22:00:00",
            "2022-12-31T23:00:00",
            "2023-01-01T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    dataset = SimpleNamespace(timestamps=timestamps, n_bars=len(timestamps))
    config = SimpleNamespace(
        feature_indices=(1, 3),
        fit_symbol_indices=(0, 2),
        fit_cutoff="2023-01-01T00:00:00",
    )
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_fit(_dataset, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(module, "fit_ppo_strategy", fake_fit)
    control, candidate = replication_arm_specs()[0], replication_arm_specs()[5]

    assert fit_replication_strategy(dataset, config, control) is sentinel
    assert fit_replication_strategy(dataset, config, candidate) is sentinel

    first, second = calls
    common_keys = set(first) | set(second)
    changed = {key for key in common_keys if first.get(key) != second.get(key)}
    assert changed == {"normalize_features"}
    assert first["normalize_features"] is False
    assert second["normalize_features"] is True

    for call in calls:
        assert call["feature_indices"] == (1, 3)
        assert call["fit_symbol_indices"] == (0, 2)
        assert call["start_index"] == 0
        assert call["stop_index"] == 2
        assert call["gross_budget"] == 0.1
        assert call["total_timesteps"] == 262_144
        assert call["seed"] == 0
        assert call["initial_capital"] == 10_000.0
        assert call["execution_cost"] is DIRECTIONAL_BASE_EXECUTION_COST
        assert call["training_layout"] == "sequential"
        assert call["risk_config"] is None
        assert call["settle_terminal_position"] is True


def test_replication_strategy_factory_preserves_raw_feature_schema() -> None:
    policy = object()
    frozen = SimpleNamespace(
        policy=policy,
        feature_indices=(2, 4),
        feature_names=("raw_2", "raw_4"),
        feature_normalizer=None,
    )

    factory = replication_strategy_factory(frozen)
    first = factory()
    second = factory()

    assert first is not second
    assert first.policy is second.policy is policy
    assert first.feature_indices == second.feature_indices == (2, 4)
    assert first.feature_names == second.feature_names == ("raw_2", "raw_4")
    assert first.feature_normalizer is second.feature_normalizer is None


def test_replication_strategy_factory_preserves_normalized_feature_schema() -> None:
    class FakeNormalizer:
        feature_names = ("norm_2", "norm_4")

        def validate_features(self, feature_indices) -> None:
            assert feature_indices == (2, 4)

    policy = object()
    normalizer = FakeNormalizer()
    frozen = SimpleNamespace(
        policy=policy,
        feature_indices=(2, 4),
        feature_names=("norm_2", "norm_4"),
        feature_normalizer=normalizer,
    )

    factory = replication_strategy_factory(frozen)
    first = factory()
    second = factory()

    assert first is not second
    assert first.policy is second.policy is policy
    assert first.feature_indices == second.feature_indices == (2, 4)
    assert first.feature_names == second.feature_names == ("norm_2", "norm_4")
    assert first.feature_normalizer is second.feature_normalizer is normalizer

def test_slot_boundary_distinguishes_prefit_from_consumed_failure(tmp_path) -> None:
    spec = replication_arm_specs()[0]

    record_prefit_failure(
        tmp_path,
        spec,
        attempt_id="runtime-preflight-1",
        error="trainer runtime missing",
    )
    before = replication_slot_state(tmp_path, spec)
    assert before["consumed"] is False
    assert before["failed"] is False
    assert before["prefit_failure_count"] == 1

    claim_replication_slot(
        tmp_path,
        spec,
        activation_digest="a" * 64,
        implementation_digest="b" * 64,
    )
    claimed = replication_slot_state(tmp_path, spec)
    assert claimed["consumed"] is True
    assert claimed["failed"] is False

    with pytest.raises(ValueError, match="consumed"):
        claim_replication_slot(
            tmp_path,
            spec,
            activation_digest="a" * 64,
            implementation_digest="b" * 64,
        )

    record_consumed_failure(tmp_path, spec, error="fit started then failed")
    failed = replication_slot_state(tmp_path, spec)
    assert failed["consumed"] is True
    assert failed["failed"] is True
    assert failed["prefit_failure_count"] == 1


def _screen_row(
    total_return: float,
    *,
    year_return: float,
    with_stress: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "metrics": {"total_return": total_return},
        "ledger_max_drawdown": 0.1,
        "terminal_flat": True,
        "termination_reasons": [],
        "start_index": 0,
        "stop_index": 17_544,
        "returns": [0.0] * 17_544,
        "year_returns": {"2023": year_return, "2024": year_return},
        "qualified": False,
    }
    if with_stress:
        stress_row = {
            "metrics": {"total_return": 0.01},
            "ledger_max_drawdown": 0.1,
            "terminal_flat": True,
            "termination_reasons": [],
            "start_index": 0,
            "stop_index": 17_544,
            "returns": [0.0] * 17_544,
            "year_returns": {"2023": -0.5, "2024": -0.5},
        }
        row["stress"] = [
            {**stress_row, "cost_multiplier": 2.0, "latency_bars": 0},
            {**stress_row, "cost_multiplier": 1.0, "latency_bars": 1},
        ]
        row["by_symbol"] = {
            symbol: {}
            for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
        }
    return row



def test_candidate_base_pass_runs_registered_stress_and_symbol_diagnostics(
    monkeypatch,
) -> None:
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    dataset = SimpleNamespace(symbols=symbols)
    candidate = replication_arm_specs()[5]
    calls: list[dict[str, object]] = []

    def fake_evaluate(_dataset, _factory, **kwargs):
        calls.append(kwargs)
        return _screen_row(0.01, year_return=0.01, with_stress=False)

    monkeypatch.setattr(module, "evaluate_directional_arm", fake_evaluate)

    result = module._evaluate_replication_result(
        dataset,
        lambda: object(),
        start_index=0,
        stop_index=17_544,
        spec=candidate,
    )

    assert len(calls) == 8
    assert [(row["cost_multiplier"], row["latency_bars"]) for row in result["stress"]] == [
        (2.0, 0),
        (1.0, 1),
    ]
    assert set(result["by_symbol"]) == set(symbols)
    assert calls[0]["initial_capital"] == 10_000.0
    assert calls[0]["gross_budget"] == 0.1


def test_control_and_unqualified_candidate_do_not_open_stress_evidence(
    monkeypatch,
) -> None:
    dataset = SimpleNamespace(
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    )
    calls: list[dict[str, object]] = []

    def fake_evaluate(_dataset, _factory, **kwargs):
        calls.append(kwargs)
        return _screen_row(-0.01, year_return=-0.01, with_stress=False)

    monkeypatch.setattr(module, "evaluate_directional_arm", fake_evaluate)

    control = module._evaluate_replication_result(
        dataset,
        lambda: object(),
        start_index=0,
        stop_index=17_544,
        spec=replication_arm_specs()[0],
    )
    candidate = module._evaluate_replication_result(
        dataset,
        lambda: object(),
        start_index=0,
        stop_index=17_544,
        spec=replication_arm_specs()[5],
    )

    assert len(calls) == 2
    assert "stress" not in control and "by_symbol" not in control
    assert "stress" not in candidate and "by_symbol" not in candidate


def test_slot_state_rejects_tampered_consumed_claim(tmp_path) -> None:
    spec = replication_arm_specs()[0]
    claim_replication_slot(
        tmp_path,
        spec,
        activation_digest="a" * 64,
        implementation_digest="b" * 64,
    )
    claim = tmp_path / "slots" / spec.slot / "consumed.json"
    payload = {
        "schema": "ppo_normalization_replication_slot_v1",
        "slot": "candidate_normalized_seed4",
        "protocol_arm": spec.protocol_arm,
        "seed": spec.seed,
        "normalize_features": spec.normalize_features,
        "activation_digest": "a" * 64,
        "implementation_digest": "b" * 64,
        "consumed": True,
    }
    claim.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="claim"):
        replication_slot_state(tmp_path, spec)

def test_decision_is_recomputed_without_trusting_qualified_flags() -> None:
    control = {
        seed: _screen_row(0.0, year_return=0.0, with_stress=False) for seed in range(5)
    }
    candidate = {
        seed: _screen_row(0.01, year_return=0.01, with_stress=True) for seed in range(5)
    }

    report = recompute_replication_decision(control, candidate)

    assert report["paired_win_count"] == 5
    assert report["relative_improvement"] is True
    assert report["candidate_base_pass_count"] == 5
    assert report["candidate_base_and_stress_pass_count"] == 5
    assert report["decision"] == "PROSPECTIVE_PAPER_REQUIRED"


def test_relative_improvement_does_not_claim_absolute_profitability() -> None:
    control = {
        seed: _screen_row(-0.02, year_return=-0.01, with_stress=False)
        for seed in range(5)
    }
    candidate = {
        seed: _screen_row(-0.01, year_return=-0.01, with_stress=False)
        for seed in range(5)
    }

    report = recompute_replication_decision(control, candidate)

    assert report["relative_improvement"] is True
    assert report["candidate_base_pass_count"] == 0
    assert report["decision"] == "RELATIVE_IMPROVEMENT_ONLY"


def _activation(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "schema": EXECUTION_ACTIVATION_SCHEMA,
        "protocol_sha256": SEALED_PROTOCOL_SHA256,
        "implementation_digest": "b" * 64,
        "provenance": provenance,
        "economic_execution_authorized": True,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def test_prepare_execute_and_verify_uses_saved_bundle_without_refit(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = {
        "schema_version": "test-provenance",
        "implementation_digest": "b" * 64,
    }
    activation = _activation(provenance)
    activation_digest = content_digest(activation)
    root = tmp_path / "execution"
    prepare_replication_execution(
        root,
        activation,
        expected_activation_digest=activation_digest,
    )

    dataset = SimpleNamespace(feature_names=("signal", "carry"))
    config = SimpleNamespace()
    monkeypatch.setattr(
        module,
        "build_candidate_run_provenance",
        lambda: provenance,
    )
    monkeypatch.setattr(
        module,
        "_load_replication_context",
        lambda _source: (dataset, config, 10, 20),
    )

    fit_calls: list[str] = []
    saved = SimpleNamespace(
        policy=SimpleNamespace(num_timesteps=262_144),
        feature_indices=(0,),
        feature_names=("signal",),
        feature_normalizer=None,
    )

    def fake_fit(_dataset, _config, spec):
        fit_calls.append(spec.slot)
        return saved

    monkeypatch.setattr(module, "fit_replication_strategy", fake_fit)

    loaded = SimpleNamespace(
        policy=SimpleNamespace(num_timesteps=262_144),
        feature_indices=(0,),
        feature_names=("signal",),
        feature_normalizer=None,
    )

    def fake_save(bundle_root, _strategy, *, feature_names):
        assert feature_names == ("signal", "carry")
        bundle_root.mkdir(parents=True)
        (bundle_root / "manifest.json").write_text(
            '{"normalizer":null,"policy_sha256":"' + "c" * 64 + '"}',
            encoding="utf-8",
        )
        return "d" * 64

    monkeypatch.setattr(module, "save_ppo_inference_bundle", fake_save)
    monkeypatch.setattr(
        module,
        "load_ppo_inference_bundle",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        module,
        "_manifest_for_bundle",
        lambda _store, _spec: {
            "normalizer": None,
            "policy_sha256": "c" * 64,
        },
    )

    replay = {
        "metrics": {"total_return": 0.01},
        "ledger_max_drawdown": 0.05,
        "terminal_flat": True,
        "termination_reasons": [],
        "start_index": 10,
        "stop_index": 20,
        "returns": [0.001],
        "year_returns": {"2023": 0.01, "2024": 0.01},
    }
    monkeypatch.setattr(
        module,
        "evaluate_directional_arm",
        lambda *_a, **_k: replay,
    )

    slot = replication_arm_specs()[0].slot
    published = execute_replication_slot(
        tmp_path / "source",
        root,
        slot,
        expected_activation_digest=activation_digest,
    )
    assert fit_calls == [slot]
    assert published["bundle_digest"] == "d" * 64
    assert published["realized_timesteps"] == 262_144
    assert replication_slot_state(root, replication_arm_specs()[0]) == {
        "slot": slot,
        "consumed": True,
        "failed": False,
        "result_published": True,
        "prefit_failure_count": 0,
    }

    monkeypatch.setattr(
        module,
        "fit_replication_strategy",
        lambda *_a, **_k: pytest.fail("verifier must not refit"),
    )
    verified = verify_replication_slot(
        tmp_path / "source",
        root,
        slot,
        expected_activation_digest=activation_digest,
    )
    assert verified == published
    assert fit_calls == [slot]

    result_path = root / "slots" / slot / "result.json"
    semantic_result = json.loads(result_path.read_bytes())
    result_path.write_text(
        json.dumps(semantic_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        verify_replication_slot(
            tmp_path / "source",
            root,
            slot,
            expected_activation_digest=activation_digest,
        )


def test_execution_root_rejects_noncanonical_protocol_bytes(tmp_path) -> None:
    provenance = {
        "schema_version": "test-provenance",
        "implementation_digest": "b" * 64,
    }
    activation = _activation(provenance)
    activation_digest = content_digest(activation)
    root = tmp_path / "noncanonical-protocol"
    prepare_replication_execution(
        root,
        activation,
        expected_activation_digest=activation_digest,
    )

    protocol_path = root / "protocol.json"
    semantic_protocol = json.loads(protocol_path.read_bytes())
    protocol_path.write_text(
        json.dumps(semantic_protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    monkeypatch_provenance = module.build_candidate_run_provenance
    try:
        module.build_candidate_run_provenance = lambda: provenance
        with pytest.raises(ValueError, match="canonical"):
            module._validate_execution_root(
                root,
                expected_activation_digest=activation_digest,
            )
    finally:
        module.build_candidate_run_provenance = monkeypatch_provenance


def test_activation_rejects_result_or_unused_data_authority(tmp_path) -> None:
    provenance = {
        "schema_version": "test-provenance",
        "implementation_digest": "b" * 64,
    }
    for field in (
        "economic_result_inspected",
        "unused_data_accessed",
        "final_test_accessed",
        "production_eligible",
        "live_trading_authorized",
    ):
        activation = _activation(provenance)
        activation[field] = True
        with pytest.raises(ValueError, match=field):
            prepare_replication_execution(
                tmp_path / field,
                activation,
                expected_activation_digest=content_digest(activation),
            )


def test_activation_rejects_implementation_digest_drift(tmp_path) -> None:
    provenance = {
        "schema_version": "test-provenance",
        "implementation_digest": "c" * 64,
    }
    activation = _activation(provenance)
    with pytest.raises(ValueError, match="implementation digest"):
        prepare_replication_execution(
            tmp_path / "identity-drift",
            activation,
            expected_activation_digest=content_digest(activation),
        )
