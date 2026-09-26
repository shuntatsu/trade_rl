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
    execute_replication_slot,
    fit_replication_strategy,
    prepare_replication_execution,
    publish_replication_decision,
    recompute_replication_decision,
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


def test_slot_boundary_distinguishes_prefit_from_consumed_failure(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "slot-boundary",
    )
    spec = replication_arm_specs()[0]

    module._record_prefit_failure(
        root,
        spec,
        attempt_id="runtime-preflight-1",
        error="trainer runtime missing",
    )
    before = replication_slot_state(root, spec)
    assert before["consumed"] is False
    assert before["failed"] is False
    assert before["prefit_failure_count"] == 1

    module._claim_replication_slot(root, spec)
    claimed = replication_slot_state(root, spec)
    assert claimed["consumed"] is True
    assert claimed["failed"] is False

    with pytest.raises(ValueError, match="consumed"):
        module._claim_replication_slot(root, spec)

    module._record_consumed_failure(root, spec, error="fit started then failed")
    failed = replication_slot_state(root, spec)
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


def test_digest_payload_pair_resumes_digest_only_and_rejects_tamper(
    tmp_path,
) -> None:
    store = module.StudyStore(tmp_path)
    payload = {"schema": "test_pair_v1", "value": 1}
    raw = canonical_json_bytes(payload)
    store.publish_json_once(
        "proof.sha256.json",
        {"sha256": module.sha256(raw).hexdigest()},
    )

    module._publish_json_with_sha256_pair(
        store,
        "proof.json",
        payload,
        field="proof",
    )
    assert (tmp_path / "proof.json").read_bytes() == raw

    module._publish_json_with_sha256_pair(
        store,
        "proof.json",
        payload,
        field="proof",
    )

    (tmp_path / "proof.json").write_bytes(
        canonical_json_bytes({"schema": "test_pair_v1", "value": 2})
    )
    with pytest.raises(ValueError, match="differs"):
        module._publish_json_with_sha256_pair(
            store,
            "proof.json",
            payload,
            field="proof",
        )


def test_result_digest_without_result_does_not_make_slot_retryable(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "result-digest",
    )
    spec = replication_arm_specs()[0]
    module._claim_replication_slot(root, spec)
    store = module.StudyStore(root)
    store.publish_json_once(
        module._slot_relative(spec, "result.sha256.json"),
        {"sha256": "c" * 64},
    )

    state = replication_slot_state(root, spec)
    assert state["consumed"] is True
    assert state["result_published"] is False
    assert state["failed"] is False

    module._record_consumed_failure(root, spec, error="crash before result publication")
    failed = replication_slot_state(root, spec)
    assert failed["consumed"] is True
    assert failed["failed"] is True
    assert failed["result_published"] is False
    with pytest.raises(ValueError, match="consumed"):
        module._claim_replication_slot(root, spec)


def test_candidate_base_pass_runs_registered_stress_and_symbol_diagnostics(
    monkeypatch,
) -> None:
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    dataset = SimpleNamespace(symbols=symbols)
    candidate = replication_arm_specs()[5]
    calls: list[dict[str, object]] = []

    def fake_evaluate(_dataset, _factory, **kwargs):
        calls.append(kwargs)
        row = _screen_row(0.01, year_return=0.01, with_stress=False)
        row["cost_multiplier"] = kwargs.get("cost_multiplier", 1.0)
        row["latency_bars"] = kwargs.get("latency_bars", 0)
        return row

    monkeypatch.setattr(module, "evaluate_directional_arm", fake_evaluate)

    result = module._evaluate_replication_result(
        dataset,
        lambda: object(),
        start_index=0,
        stop_index=17_544,
        spec=candidate,
    )

    assert len(calls) == 8
    assert [
        (row["cost_multiplier"], row["latency_bars"]) for row in result["stress"]
    ] == [
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


def test_slot_state_rejects_tampered_consumed_claim(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "tampered-claim",
    )
    spec = replication_arm_specs()[0]
    module._claim_replication_slot(root, spec)
    claim = root / "slots" / spec.slot / "consumed.json"
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
        replication_slot_state(root, spec)


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


def _provenance(
    *,
    implementation_digest: str = "b" * 64,
    python_version: str = "3.12.7",
    stable_baselines3: str = "2.3.2",
    os_release: str = "test-kernel-1",
    machine: str = "x86_64",
) -> dict[str, object]:
    return {
        "schema_version": "test-provenance",
        "implementation_digest": implementation_digest,
        "runtime_environment": {
            "schema_version": "candidate_run_runtime_v1",
            "python": {
                "implementation": "CPython",
                "version": python_version,
            },
            "os": {
                "family": "TestOS",
                "release": os_release,
            },
            "machine": machine,
            "packages": {
                "trade-rl": "0.1.0",
                "numpy": "2.1.3",
                "gymnasium": "0.29.1",
                "lightgbm": "4.5.0",
                "stable-baselines3": stable_baselines3,
                "torch": "2.4.1",
            },
        },
        "research_context_digest": None,
    }


def _activation(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "schema": EXECUTION_ACTIVATION_SCHEMA,
        "protocol_sha256": SEALED_PROTOCOL_SHA256,
        "implementation_digest": "b" * 64,
        "implementation_seal_sha256": "c" * 64,
        "fresh_reconstruction_sha256": "d" * 64,
        "assurance_review_sha256": "e" * 64,
        "provenance": provenance,
        "economic_execution_authorized": True,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def _seal_activation_authority(tmp_path, monkeypatch, activation) -> str:
    digest = content_digest(activation)
    authority = tmp_path / "activation-authority.json"
    authority.write_bytes(
        canonical_json_bytes(
            {
                "schema": module.ACTIVATION_AUTHORITY_SCHEMA,
                "activation_sha256": digest,
            }
        )
    )
    monkeypatch.setattr(module, "_ACTIVATION_AUTHORITY_PATH", authority)
    return digest


def _prepare_test_root(
    tmp_path,
    monkeypatch,
    name: str,
):
    provenance = _provenance()
    activation = _activation(provenance)
    activation_digest = _seal_activation_authority(tmp_path, monkeypatch, activation)
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: provenance)
    root = tmp_path / name
    prepare_replication_execution(root, activation)
    return root, provenance, activation_digest


def test_activation_authority_is_outside_python_implementation_identity() -> None:
    provenance = module.build_candidate_run_provenance()
    implementation = provenance["implementation"]
    assert isinstance(implementation, dict)
    files = implementation["files"]
    assert isinstance(files, list)
    paths = {
        item["path"]
        for item in files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }

    assert "evaluation/ppo_normalization_execution.py" in paths
    assert "evaluation/ppo_normalization_activation.json" not in paths


def test_execution_is_fail_closed_without_sealed_activation(tmp_path) -> None:
    provenance = _provenance()
    activation = _activation(provenance)

    authority = json.loads(module._ACTIVATION_AUTHORITY_PATH.read_bytes())
    assert authority == {
        "schema": module.ACTIVATION_AUTHORITY_SCHEMA,
        "activation_sha256": None,
    }
    with pytest.raises(RuntimeError, match="not sealed"):
        prepare_replication_execution(tmp_path / "execution", activation)
    assert not (tmp_path / "execution").exists()


def test_activation_authority_rejects_noncanonical_or_wrong_schema(
    tmp_path,
    monkeypatch,
) -> None:
    authority = tmp_path / "activation-authority.json"
    monkeypatch.setattr(module, "_ACTIVATION_AUTHORITY_PATH", authority)

    authority.write_text(
        json.dumps(
            {
                "schema": module.ACTIVATION_AUTHORITY_SCHEMA,
                "activation_sha256": "a" * 64,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        module._sealed_execution_activation_digest()

    authority.write_bytes(
        canonical_json_bytes(
            {
                "schema": "unexpected_activation_authority",
                "activation_sha256": "a" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="shape"):
        module._sealed_execution_activation_digest()


def test_prepare_execute_and_verify_uses_saved_bundle_without_refit(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance()
    activation = _activation(provenance)
    _seal_activation_authority(
        tmp_path,
        monkeypatch,
        activation,
    )
    root = tmp_path / "execution"
    monkeypatch.setattr(
        module,
        "build_candidate_run_provenance",
        lambda: provenance,
    )
    prepare_replication_execution(root, activation)

    dataset = SimpleNamespace(feature_names=("signal", "carry"))
    config = SimpleNamespace()
    monkeypatch.setattr(
        module,
        "_load_replication_context",
        lambda _source: (dataset, config, 10, 20),
    )
    monkeypatch.setattr(
        module,
        "_source_identity_snapshot",
        lambda _source: {"source": {"sha256": "a" * 64, "size_bytes": 1}},
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
        lambda _store, _spec, **_kwargs: {
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
    publication_order: list[str] = []
    original_publish_json_once = module.StudyStore.publish_json_once

    def record_publication(self, relative, payload):
        publication_order.append(str(relative).replace("\\", "/"))
        return original_publish_json_once(self, relative, payload)

    monkeypatch.setattr(module.StudyStore, "publish_json_once", record_publication)

    slot = replication_arm_specs()[0].slot
    published = execute_replication_slot(
        tmp_path / "source",
        root,
        slot,
    )
    assert fit_calls == [slot]
    assert published["bundle_digest"] == "d" * 64
    assert published["realized_timesteps"] == 262_144
    result_prefix = f"slots/{slot}/"
    assert publication_order.index(result_prefix + "result.sha256.json") < (
        publication_order.index(result_prefix + "result.json")
    )
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
    )
    assert verified == published
    assert fit_calls == [slot]
    verification_path = root / "slots" / slot / "verified.json"
    verification = json.loads(verification_path.read_bytes())
    assert verification["no_refit"] is True
    assert verification["replay_verified"] is True
    verification_raw = verification_path.read_bytes()
    assert json.loads(
        (root / "slots" / slot / "verified.sha256.json").read_bytes()
    ) == {"sha256": module.sha256(verification_raw).hexdigest()}

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
        )


def _publish_synthetic_result(
    root,
    spec,
    *,
    activation_digest: str,
    provenance: dict[str, object],
    verified: bool,
) -> None:
    module._claim_replication_slot(root, spec)
    economic = _screen_row(
        0.0 if spec.protocol_arm == "control_raw" else 0.01,
        year_return=0.0 if spec.protocol_arm == "control_raw" else 0.01,
        with_stress=False,
    )
    payload = {
        **economic,
        "schema": module.SLOT_RESULT_SCHEMA,
        "slot": spec.slot,
        "protocol_arm": spec.protocol_arm,
        "seed": spec.seed,
        "normalize_features": spec.normalize_features,
        "protocol_sha256": SEALED_PROTOCOL_SHA256,
        "activation_digest": activation_digest,
        "implementation_digest": "b" * 64,
        "bundle_digest": "d" * 64,
        "bundle_policy_sha256": "c" * 64,
        "realized_timesteps": 262_144,
        "provenance": provenance,
    }
    store = module.StudyStore(root)
    result_raw = canonical_json_bytes(payload)
    store.publish_json_once(module._slot_relative(spec, "result.json"), payload)
    store.publish_json_once(
        module._slot_relative(spec, "result.sha256.json"),
        {"sha256": module.sha256(result_raw).hexdigest()},
    )
    if not verified:
        return
    verification = {
        "schema": module.SLOT_VERIFICATION_SCHEMA,
        "slot": spec.slot,
        "protocol_arm": spec.protocol_arm,
        "seed": spec.seed,
        "normalize_features": spec.normalize_features,
        "protocol_sha256": SEALED_PROTOCOL_SHA256,
        "activation_digest": activation_digest,
        "implementation_digest": "b" * 64,
        "result_sha256": module.sha256(result_raw).hexdigest(),
        "bundle_digest": "d" * 64,
        "bundle_policy_sha256": "c" * 64,
        "verifier_provenance": provenance,
        "no_refit": True,
        "replay_verified": True,
    }
    verification_raw = canonical_json_bytes(verification)
    store.publish_json_once(module._slot_relative(spec, "verified.json"), verification)
    store.publish_json_once(
        module._slot_relative(spec, "verified.sha256.json"),
        {"sha256": module.sha256(verification_raw).hexdigest()},
    )


def _publish_verifier_authority(
    root,
    *,
    activation_digest: str,
    implementation_digest: str,
) -> dict[str, object]:
    records = []
    for spec in replication_arm_specs():
        raw = (root / "slots" / spec.slot / "verified.json").read_bytes()
        records.append(
            {
                "slot": spec.slot,
                "sha256": module.sha256(raw).hexdigest(),
            }
        )
    verification_set_sha256 = content_digest(
        {
            "schema": module._VERIFICATION_SET_SCHEMA,
            "records": records,
        }
    )
    authority = {
        "schema": module.VERIFIER_ARTIFACT_AUTHORITY_SCHEMA,
        "repository_id": 123,
        "run_id": 456,
        "artifact_id": 789,
        "artifact_sha256": "f" * 64,
        "artifact_api_digest": "sha256:" + "f" * 64,
        "code_sha": "1" * 40,
        "workflow_sha": "2" * 40,
        "activation_digest": activation_digest,
        "implementation_digest": implementation_digest,
        "verification_set_sha256": verification_set_sha256,
    }
    store = module.StudyStore(root)
    raw = canonical_json_bytes(authority)
    store.publish_json_once("verifier-authority.json", authority)
    store.publish_json_once(
        "verifier-authority.sha256.json",
        {"sha256": module.sha256(raw).hexdigest()},
    )
    return authority


def test_comparison_requires_all_ten_durable_verifications(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance()
    activation = _activation(provenance)
    activation_digest = _seal_activation_authority(
        tmp_path,
        monkeypatch,
        activation,
    )
    monkeypatch.setattr(
        module,
        "build_candidate_run_provenance",
        lambda: provenance,
    )
    root = tmp_path / "verified-comparison"
    prepare_replication_execution(root, activation)

    specs = replication_arm_specs()
    for spec in specs[:-1]:
        _publish_synthetic_result(
            root,
            spec,
            activation_digest=activation_digest,
            provenance=provenance,
            verified=True,
        )
    _publish_synthetic_result(
        root,
        specs[-1],
        activation_digest=activation_digest,
        provenance=provenance,
        verified=False,
    )

    with pytest.raises(ValueError, match="not independently verified"):
        publish_replication_decision(root)
    assert not (root / "comparison.json").exists()

    store = module.StudyStore(root)
    last = specs[-1]
    result_path = root / "slots" / last.slot / "result.json"
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw)
    verification = {
        "schema": module.SLOT_VERIFICATION_SCHEMA,
        "slot": last.slot,
        "protocol_arm": last.protocol_arm,
        "seed": last.seed,
        "normalize_features": last.normalize_features,
        "protocol_sha256": SEALED_PROTOCOL_SHA256,
        "activation_digest": activation_digest,
        "implementation_digest": "b" * 64,
        "result_sha256": module.sha256(result_raw).hexdigest(),
        "bundle_digest": result["bundle_digest"],
        "bundle_policy_sha256": result["bundle_policy_sha256"],
        "verifier_provenance": provenance,
        "no_refit": True,
        "replay_verified": True,
    }
    verification_raw = canonical_json_bytes(verification)
    store.publish_json_once(module._slot_relative(last, "verified.json"), verification)
    store.publish_json_once(
        module._slot_relative(last, "verified.sha256.json"),
        {"sha256": module.sha256(verification_raw).hexdigest()},
    )

    with pytest.raises(ValueError, match="verifier artifact authority"):
        publish_replication_decision(root)
    _publish_verifier_authority(
        root,
        activation_digest=activation_digest,
        implementation_digest="b" * 64,
    )

    publication_order: list[str] = []
    original_publish_json_once = module.StudyStore.publish_json_once

    def record_publication(self, relative, payload):
        publication_order.append(str(relative).replace("\\", "/"))
        return original_publish_json_once(self, relative, payload)

    monkeypatch.setattr(module.StudyStore, "publish_json_once", record_publication)

    report = publish_replication_decision(root)
    assert publication_order.index("comparison.sha256.json") < publication_order.index(
        "comparison.json"
    )
    assert report["all_slots_independently_verified"] is True
    assert report["verified_slots"] == [spec.slot for spec in specs]
    assert report["verifier_run_id"] == 456
    assert report["verifier_artifact_id"] == 789
    assert isinstance(report["verifier_authority_sha256"], str)
    assert isinstance(report["verification_set_sha256"], str)
    assert report["decision"] == "RELATIVE_IMPROVEMENT_ONLY"
    paired = report["paired_return_deltas"]
    assert isinstance(paired, dict)
    assert set(paired) == {"0", "1", "2", "3", "4"}
    assert (root / "comparison.json").exists()

    published_count = len(publication_order)
    assert publish_replication_decision(root) == report
    assert len(publication_order) == published_count


def test_execution_root_rejects_noncanonical_protocol_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance()
    activation = _activation(provenance)
    _seal_activation_authority(
        tmp_path,
        monkeypatch,
        activation,
    )
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: provenance)
    root = tmp_path / "noncanonical-protocol"
    prepare_replication_execution(root, activation)

    protocol_path = root / "protocol.json"
    semantic_protocol = json.loads(protocol_path.read_bytes())
    protocol_path.write_text(
        json.dumps(semantic_protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        module._validate_execution_root(root)


def test_activation_rejects_result_or_unused_data_authority(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance()
    for field in (
        "economic_result_inspected",
        "unused_data_accessed",
        "final_test_accessed",
        "production_eligible",
        "live_trading_authorized",
    ):
        activation = _activation(provenance)
        activation[field] = True
        _seal_activation_authority(tmp_path, monkeypatch, activation)
        with pytest.raises(ValueError, match=field):
            prepare_replication_execution(tmp_path / field, activation)


@pytest.mark.parametrize(
    ("provenance", "message"),
    (
        (_provenance(stable_baselines3="2.4.0"), "stable-baselines3"),
        (_provenance(python_version="3.13.0"), "Python runtime"),
    ),
)
def test_activation_rejects_runtime_drift(
    tmp_path,
    monkeypatch,
    provenance: dict[str, object],
    message: str,
) -> None:
    activation = _activation(provenance)
    _seal_activation_authority(tmp_path, monkeypatch, activation)
    with pytest.raises(ValueError, match=message):
        prepare_replication_execution(tmp_path / "runtime-drift", activation)


def test_activation_rejects_implementation_digest_drift(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance(implementation_digest="c" * 64)
    activation = _activation(provenance)
    _seal_activation_authority(tmp_path, monkeypatch, activation)
    with pytest.raises(ValueError, match="implementation digest"):
        prepare_replication_execution(tmp_path / "identity-drift", activation)


def test_slot_state_requires_an_existing_prepared_root(tmp_path) -> None:
    root = tmp_path / "missing-execution"
    spec = replication_arm_specs()[0]

    with pytest.raises(ValueError, match="prepared|root"):
        replication_slot_state(root, spec)

    assert not root.exists()


def test_prepare_rejects_non_current_provenance_before_root_creation(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance()
    activation = _activation(provenance)
    _seal_activation_authority(tmp_path, monkeypatch, activation)
    monkeypatch.setattr(
        module,
        "build_candidate_run_provenance",
        lambda: _provenance(implementation_digest="d" * 64),
    )
    root = tmp_path / "wrong-current-provenance"

    with pytest.raises(ValueError, match="source/runtime|current provenance"):
        prepare_replication_execution(root, activation)

    assert not root.exists()


def test_prepare_failure_never_publishes_partial_root(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance()
    activation = _activation(provenance)
    _seal_activation_authority(tmp_path, monkeypatch, activation)
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: provenance)
    root = tmp_path / "atomic-prepare"
    original = module.StudyStore.publish_json_once

    def fail_mid_prepare(self, relative, payload):
        if str(relative).replace("\\", "/") == "activation.json":
            raise RuntimeError("injected prepare failure")
        return original(self, relative, payload)

    monkeypatch.setattr(module.StudyStore, "publish_json_once", fail_mid_prepare)

    with pytest.raises(RuntimeError, match="injected prepare failure"):
        prepare_replication_execution(root, activation)

    assert not root.exists()
    assert not list(tmp_path.glob(".atomic-prepare.staging-*"))


def test_prefit_failure_is_forbidden_after_slot_claim(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "prefit-chronology",
    )
    spec = replication_arm_specs()[0]
    module._claim_replication_slot(root, spec)

    with pytest.raises(ValueError, match="pre-fit failure|consumption"):
        module._record_prefit_failure(
            root,
            spec,
            attempt_id="too-late",
            error="must not be appended",
        )


def test_prefit_failure_record_is_validated_against_filename_and_arm(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "prefit-integrity",
    )
    spec = replication_arm_specs()[0]
    module._record_prefit_failure(
        root,
        spec,
        attempt_id="attempt-1",
        error="runtime unavailable",
    )
    path = root / "prefit-failures" / spec.slot / "attempt-1.json"
    payload = json.loads(path.read_bytes())
    payload["attempt_id"] = "other-attempt"
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="prefit failure evidence"):
        replication_slot_state(root, spec)


def test_consumed_failure_record_is_bound_to_immutable_claim(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "consumed-failure-integrity",
    )
    spec = replication_arm_specs()[0]
    module._claim_replication_slot(root, spec)
    module._record_consumed_failure(root, spec, error="fit crashed")
    path = root / "slots" / spec.slot / "failed.json"
    payload = json.loads(path.read_bytes())
    payload["seed"] = 4
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="consumed failure evidence"):
        replication_slot_state(root, spec)


def test_activation_requires_result_blind_assurance_evidence_chain(
    tmp_path,
    monkeypatch,
) -> None:
    provenance = _provenance()
    activation = _activation(provenance)
    activation.pop("fresh_reconstruction_sha256")
    _seal_activation_authority(tmp_path, monkeypatch, activation)
    root = tmp_path / "missing-assurance"

    with pytest.raises(ValueError, match="shape"):
        prepare_replication_execution(root, activation)

    assert not root.exists()


def test_verifier_runtime_ignores_kernel_release_but_not_stable_identity(
    tmp_path,
    monkeypatch,
) -> None:
    root, provenance, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "verifier-runtime",
    )
    verifier = _provenance(os_release="other-kernel")
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: verifier)

    module._validate_execution_root(root, runtime_contract="verifier")

    wrong_machine = _provenance(os_release="other-kernel", machine="arm64")
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: wrong_machine)
    with pytest.raises(ValueError, match="stable runtime identity"):
        module._validate_execution_root(root, runtime_contract="verifier")


def test_execute_rejects_realized_timestep_drift_before_bundle_publication(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "timestep-drift",
    )
    dataset = SimpleNamespace(feature_names=("signal", "carry"))
    monkeypatch.setattr(
        module,
        "_load_replication_context",
        lambda _source: (dataset, SimpleNamespace(), 10, 20),
    )
    monkeypatch.setattr(
        module,
        "_source_identity_snapshot",
        lambda _source: {"source": {"sha256": "a" * 64, "size_bytes": 1}},
    )
    fitted = SimpleNamespace(
        policy=SimpleNamespace(num_timesteps=262_143),
        feature_indices=(0,),
        feature_names=("signal",),
        feature_normalizer=None,
    )
    monkeypatch.setattr(module, "fit_replication_strategy", lambda *_args: fitted)
    monkeypatch.setattr(
        module,
        "save_ppo_inference_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "bundle publication must not start with a wrong training budget"
        ),
    )

    slot = replication_arm_specs()[0].slot
    with pytest.raises(ValueError, match="timesteps|training budget"):
        execute_replication_slot(tmp_path / "source", root, slot)


def test_verify_validates_bundle_manifest_before_policy_deserialization(
    tmp_path,
    monkeypatch,
) -> None:
    root, provenance, activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "manifest-before-load",
    )
    spec = replication_arm_specs()[0]
    _publish_synthetic_result(
        root,
        spec,
        activation_digest=activation_digest,
        provenance=provenance,
        verified=False,
    )
    dataset = SimpleNamespace(feature_names=("signal", "carry"))
    monkeypatch.setattr(
        module,
        "_load_replication_context",
        lambda _source: (dataset, SimpleNamespace(), 10, 20),
    )
    monkeypatch.setattr(
        module,
        "_source_identity_snapshot",
        lambda _source: {"source": {"sha256": "a" * 64, "size_bytes": 1}},
    )
    monkeypatch.setattr(
        module,
        "_manifest_for_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("unsafe bundle manifest")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_ppo_inference_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "policy deserialization happened before manifest validation"
        ),
    )

    with pytest.raises(ValueError, match="unsafe bundle manifest"):
        verify_replication_slot(tmp_path / "source", root, spec.slot)


def _install_successful_execution_fakes(monkeypatch):
    dataset = SimpleNamespace(feature_names=("signal", "carry"))
    fitted = SimpleNamespace(
        policy=SimpleNamespace(num_timesteps=262_144),
        feature_indices=(0,),
        feature_names=("signal",),
        feature_normalizer=None,
    )
    loaded = SimpleNamespace(
        policy=SimpleNamespace(num_timesteps=262_144),
        feature_indices=(0,),
        feature_names=("signal",),
        feature_normalizer=None,
    )
    monkeypatch.setattr(
        module,
        "_load_replication_context",
        lambda _source: (dataset, SimpleNamespace(), 10, 20),
    )
    monkeypatch.setattr(module, "fit_replication_strategy", lambda *_args: fitted)

    def fake_save(bundle_root, _strategy, *, feature_names):
        bundle_root.mkdir(parents=True)
        return "d" * 64

    monkeypatch.setattr(module, "save_ppo_inference_bundle", fake_save)
    monkeypatch.setattr(
        module,
        "_manifest_for_bundle",
        lambda *_args, **_kwargs: {
            "normalizer": None,
            "policy_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        module,
        "load_ppo_inference_bundle",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        module,
        "_evaluate_replication_result",
        lambda *_args, **_kwargs: {
            "metrics": {"total_return": 0.01},
            "ledger_max_drawdown": 0.01,
            "terminal_flat": True,
            "termination_reasons": [],
            "start_index": 10,
            "stop_index": 20,
            "returns": [0.001],
            "year_returns": {"2023": 0.01, "2024": 0.01},
        },
    )


def test_execute_rechecks_runtime_identity_before_result_publication(
    tmp_path,
    monkeypatch,
) -> None:
    root, provenance, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "runtime-recheck",
    )
    _install_successful_execution_fakes(monkeypatch)
    monkeypatch.setattr(
        module,
        "_source_identity_snapshot",
        lambda _source: {"source": {"sha256": "a" * 64, "size_bytes": 1}},
    )
    calls = 0

    def current_provenance():
        nonlocal calls
        calls += 1
        if calls <= 2:
            return provenance
        return _provenance(implementation_digest="f" * 64)

    monkeypatch.setattr(module, "build_candidate_run_provenance", current_provenance)
    spec = replication_arm_specs()[0]

    with pytest.raises(ValueError, match="source/runtime changed"):
        execute_replication_slot(tmp_path / "source", root, spec.slot)

    state = replication_slot_state(root, spec)
    assert state["failed"] is True
    assert state["result_published"] is False


def test_execute_rechecks_source_bytes_before_result_publication(
    tmp_path,
    monkeypatch,
) -> None:
    root, _provenance_value, _activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "source-recheck",
    )
    _install_successful_execution_fakes(monkeypatch)
    snapshots = iter(
        (
            {"source": {"sha256": "a" * 64, "size_bytes": 1}},
            {"source": {"sha256": "a" * 64, "size_bytes": 1}},
            {"source": {"sha256": "b" * 64, "size_bytes": 1}},
        )
    )
    monkeypatch.setattr(
        module, "_source_identity_snapshot", lambda _source: next(snapshots)
    )
    spec = replication_arm_specs()[0]

    with pytest.raises(ValueError, match="source changed during fit or replay"):
        execute_replication_slot(tmp_path / "source", root, spec.slot)

    state = replication_slot_state(root, spec)
    assert state["failed"] is True
    assert state["result_published"] is False


def test_verify_rejects_symlinked_bundle_parent_before_policy_deserialization(
    tmp_path,
    monkeypatch,
) -> None:
    root, provenance, activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "bundle-parent-symlink",
    )
    spec = replication_arm_specs()[0]
    _publish_synthetic_result(
        root,
        spec,
        activation_digest=activation_digest,
        provenance=provenance,
        verified=False,
    )
    dataset = SimpleNamespace(feature_names=("signal", "carry"))
    monkeypatch.setattr(
        module,
        "_source_identity_snapshot",
        lambda _source: {"source": {"sha256": "a" * 64, "size_bytes": 1}},
    )
    monkeypatch.setattr(
        module,
        "_load_replication_context",
        lambda _source: (dataset, SimpleNamespace(), 10, 20),
    )
    bundle = root / "slots" / spec.slot / "bundle"
    outside = tmp_path / "outside-bundle"
    outside.mkdir()
    (outside / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema": "ppo_inference_bundle_v1",
                "observation": {},
                "feature_indices": [0],
                "feature_names": ["signal"],
                "normalizer": None,
                "policy_sha256": "c" * 64,
            }
        )
    )
    try:
        bundle.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")
    monkeypatch.setattr(
        module,
        "load_ppo_inference_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "policy deserialization followed a symlinked bundle parent"
        ),
    )

    with pytest.raises(Exception, match="symlink"):
        verify_replication_slot(tmp_path / "source", root, spec.slot)


def test_verify_revalidates_the_exact_provenance_it_publishes(
    tmp_path,
    monkeypatch,
) -> None:
    root, provenance, activation_digest = _prepare_test_root(
        tmp_path,
        monkeypatch,
        "verifier-publication-race",
    )
    spec = replication_arm_specs()[0]
    _publish_synthetic_result(
        root,
        spec,
        activation_digest=activation_digest,
        provenance=provenance,
        verified=False,
    )
    result = json.loads((root / "slots" / spec.slot / "result.json").read_bytes())
    replay = {
        key: value for key, value in result.items() if key not in module._RESULT_METADATA
    }
    dataset = SimpleNamespace(feature_names=("signal", "carry"))
    monkeypatch.setattr(
        module,
        "_source_identity_snapshot",
        lambda _source: {"source": {"sha256": "a" * 64, "size_bytes": 1}},
    )
    monkeypatch.setattr(
        module,
        "_load_replication_context",
        lambda _source: (dataset, SimpleNamespace(), 10, 20),
    )
    monkeypatch.setattr(
        module,
        "_manifest_for_bundle",
        lambda *_args, **_kwargs: {
            "normalizer": None,
            "policy_sha256": "c" * 64,
        },
    )
    loaded = SimpleNamespace(
        policy=SimpleNamespace(num_timesteps=262_144),
        feature_indices=(0,),
        feature_names=("signal",),
        feature_normalizer=None,
    )
    monkeypatch.setattr(
        module,
        "load_ppo_inference_bundle",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        module,
        "_evaluate_replication_result",
        lambda *_args, **_kwargs: replay,
    )
    calls = 0

    def current_provenance():
        nonlocal calls
        calls += 1
        if calls <= 2:
            return provenance
        return _provenance(implementation_digest="f" * 64)

    monkeypatch.setattr(module, "build_candidate_run_provenance", current_provenance)

    with pytest.raises(ValueError, match="verifier implementation digest"):
        verify_replication_slot(tmp_path / "source", root, spec.slot)

    assert not (root / "slots" / spec.slot / "verified.json").exists()
