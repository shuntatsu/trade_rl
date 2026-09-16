from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.capacity_robustness_prereg import (
    CapacityRobustnessProtocol,
    canonical_capacity_robustness_protocol,
    load_capacity_robustness_protocol,
)


def test_capacity_robustness_protocol_freezes_pre_successor_authority() -> None:
    protocol = canonical_capacity_robustness_protocol()

    assert protocol.source_baseline_artifact_id == 10301701698
    assert protocol.source_baseline_artifact_digest == (
        "f814fe4e205f8714c4344238911aae16e89ce0279908265feb1fdc85069b2a0a"
    )
    assert protocol.source_baseline_evidence_fingerprint == (
        "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
    )
    assert protocol.source_dataset_id == (
        "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
    )
    assert protocol.source_dataset_artifact_digest == (
        "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
    )
    assert protocol.source_study_digest == (
        "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
    )
    assert protocol.source_eligibility_run_id == 34794576828
    assert protocol.source_eligibility_artifact_id == 10329551558
    assert protocol.source_eligibility_artifact_digest == (
        "907599e2330e52fe832872b3b11a5f0ad62316fe4a170acd4453ab39ba1a0800"
    )
    assert protocol.source_eligibility_digest == (
        "cb92548157200067907abdb221bcfb9cea5fa527dd22193ae065fcad9c4fdbe7"
    )

    assert protocol.capacity_result_run_id == 34766830666
    assert protocol.causal_capacity_head_sha == (
        "db8193c81dd0ca334a15ffc1895d75884238f8a9"
    )
    assert protocol.identity_layer_head_sha == (
        "3e111a59153b7353e26080bd569b9775190577be"
    )
    assert protocol.identity_layer_ci_run_id == 34803217815
    assert protocol.successor_bundle_run_id == 34803217815
    assert protocol.successor_bundle_artifact_id == 10331899302
    assert protocol.successor_bundle_artifact_digest == (
        "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
    )
    assert protocol.successor_dataset_id == (
        "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
    )
    assert protocol.successor_dataset_artifact_digest == (
        "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
    )
    assert protocol.successor_study_digest == (
        "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
    )
    assert protocol.successor_dataset_tree_digest == (
        "2dd187a2cf4634e1c902f58592a15b4942127190a9eee16e0efe004da8a470f0"
    )
    assert protocol.successor_study_tree_digest == (
        "5fda054135f90ff6b35353a145dfe488dbeb45d43f9a5d391bc7e394c2dcc151"
    )
    assert protocol.successor_materialization_index_sha256 == (
        "9bddde683d4ca8d3596fb62a5612b1b4487aaa1dc11ef631923b5e51e030ba3c"
    )
    assert protocol.successor_runtime_environment_digest == (
        "6dbc9cffd844837e17741ae30681f09a6d84b0a49dec011a4fc328f97ade1d85"
    )
    assert protocol.successor_fresh_verification_run_id == 34803432434
    assert protocol.successor_fresh_verification_artifact_id == 10332575500
    assert protocol.successor_fresh_verification_artifact_digest == (
        "2067c38da3c1f45927748e2cc7481f11268710b2b386a5bd3b9a289394937589"
    )
    assert protocol.schema_version == "calibrated_capacity_robustness_prereg_v3"
    assert protocol.successor_execution_overlay == (
        "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity"
    )

    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert protocol.capacity_caps == (
        0.0021629560553901974,
        0.002044685341258238,
        0.002184898995567895,
        0.0020480213652913385,
        0.002346308308284808,
    )
    assert protocol.fit_cutoff == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_stop_exclusive == datetime(2025, 1, 1, tzinfo=UTC)
    assert protocol.ppo_seeds == (0, 1, 2, 3, 4)

    assert protocol.core_strategies == (
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
    )
    assert protocol.control_strategies == (
        "cash",
        "constant_long",
        "constant_short",
    )
    assert protocol.ppo_disclosed is True
    assert protocol.ppo_in_formal_decision is False
    assert protocol.source_profitable_core_strategies == ()
    assert protocol.pre_successor_suite_status == "NO_PROFITABLE_CORE_BASELINE"
    assert protocol.stage_b_role == "diagnostic_only"

    assert protocol.stage_a_required is True
    assert protocol.stage_a_failure_status == "INVALID_IMPLEMENTATION_DRIFT"
    assert protocol.source_profitability_required_positive_symbols == 5
    assert protocol.robust_positive_required_positive_symbols == 5
    assert protocol.execution_fragile_max_positive_symbols == 3
    assert protocol.integration_gate_satisfied is False
    assert protocol.execution_authorized is False
    assert protocol.stage_a_executed is False
    assert protocol.stage_b_executed is False
    assert protocol.successor_pnl_inspected is False
    assert protocol.experiment_0004_issue == 529
    assert protocol.experiment_0004_must_not_influence is True
    assert len(protocol.digest) == 64


def test_source_returns_and_empty_eligibility_are_frozen() -> None:
    protocol = canonical_capacity_robustness_protocol()

    assert protocol.source_core_total_returns == (
        (
            "trend",
            (
                -0.4693596383984787,
                -0.6745119541152158,
                -0.6985296047775885,
                -0.6610604847435877,
                -0.6995926449531766,
            ),
        ),
        (
            "mean_reversion",
            (
                -0.3865837097474639,
                -0.19975352370855237,
                -0.08364864809522887,
                -0.3984753309722041,
                -0.3792367092172487,
            ),
        ),
        (
            "ridge24",
            (
                -0.3817900601958202,
                -0.28212108986352513,
                -0.2640984345173577,
                -0.7338319054368865,
                0.2262786724769308,
            ),
        ),
        (
            "lightgbm24",
            (
                -0.54697432035348,
                -0.26915263313070426,
                0.10253875097178011,
                -0.11188398892435136,
                0.8215258151078024,
            ),
        ),
    )
    assert protocol.source_profitable_core_strategies == ()
    assert protocol.pre_successor_suite_status == "NO_PROFITABLE_CORE_BASELINE"


def test_protocol_rejects_semantic_drift() -> None:
    protocol = canonical_capacity_robustness_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"source_baseline_artifact_id": 1},
        {"source_baseline_artifact_digest": "0" * 64},
        {"source_eligibility_digest": "0" * 64},
        {"source_implementation_sha": "0" * 40},
        {"source_runtime_environment_digest": "0" * 64},
        {"capacity_result_digest": "0" * 64},
        {"capacity_result_run_id": 1},
        {"capacity_result_artifact_id": 1},
        {"capacity_verifier_artifact_digest": "0" * 64},
        {"successor_bundle_artifact_id": 1},
        {"successor_execution_overlay": "zero_overlay_dataset_fields_authoritative"},
        {"source_profitable_core_strategies": ("lightgbm24",)},
        {"pre_successor_suite_status": "CAPACITY_ROBUST"},
        {"stage_b_role": "formal_profit_selection"},
        {"causal_capacity_head_sha": "0" * 40},
        {"identity_layer_head_sha": "0" * 40},
        {"successor_dataset_id": "0" * 64},
        {"successor_study_digest": "0" * 64},
        {"successor_dataset_tree_digest": "0" * 64},
        {"successor_study_tree_digest": "0" * 64},
        {"successor_materialization_index_sha256": "0" * 64},
        {"successor_runtime_environment_digest": "0" * 64},
        {"successor_fresh_verification_run_id": 1},
        {"successor_fresh_verification_artifact_id": 1},
        {"successor_fresh_verification_artifact_digest": "0" * 64},
        {"capacity_caps": (0.05,) * 5},
        {"core_strategies": ("lightgbm24",)},
        {"ppo_in_formal_decision": True},
        {"source_profitability_required_positive_symbols": 4},
        {"robust_positive_required_positive_symbols": 4},
        {"execution_fragile_max_positive_symbols": 2},
        {"stage_a_required": False},
        {"integration_gates": ("pr_531_merged",)},
        {"stage_a_failure_status": "KEEP_GOING"},
        {"robust_positive_requires_positive_median": False},
        {"experiment_0004_issue": 1},
        {
            "source_core_total_returns": (
                ("trend", (-0.4, -0.6, -0.6, -0.6, -0.6)),
                *protocol.source_core_total_returns[1:],
            )
        },
        {"integration_gate_satisfied": True},
        {"execution_authorized": True},
        {"stage_a_executed": True},
        {"stage_b_executed": True},
        {"successor_pnl_inspected": True},
        {"experiment_0004_must_not_influence": False},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_protocol_payload_is_strict_and_contains_no_successor_result(tmp_path) -> None:
    protocol = canonical_capacity_robustness_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "successor_total_returns",
        "stage_b_total_returns",
        "successor_pnl",
        "pnl",
        "capacity_robustness_result",
        "experiment_0004_result",
        "candidate_result",
    }
    assert forbidden.isdisjoint(payload)
    assert "successor_identity_index_digest" not in payload
    assert payload["successor_materialization_index_sha256"] == (
        "9bddde683d4ca8d3596fb62a5612b1b4487aaa1dc11ef631923b5e51e030ba3c"
    )

    path = tmp_path / "capacity-robustness-prereg.json"
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_capacity_robustness_protocol(path)
    assert isinstance(loaded, CapacityRobustnessProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["successor_total_returns"] = {"lightgbm24": [1.0] * 5}
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_capacity_robustness_protocol(path)

    legacy = dict(payload)
    legacy["successor_identity_index_digest"] = "0" * 64
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_capacity_robustness_protocol(path)


def test_loader_rejects_nonfinite_source_returns_and_naive_clock(tmp_path) -> None:
    protocol = canonical_capacity_robustness_protocol()
    payload = protocol.to_payload()
    path = tmp_path / "capacity-robustness-prereg.json"

    broken = dict(payload)
    returns = json.loads(json.dumps(payload["source_core_total_returns"]))
    returns[0]["total_returns"][0] = float("nan")
    broken["source_core_total_returns"] = returns
    path.write_text(json.dumps(broken, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_capacity_robustness_protocol(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, evaluation_start=datetime(2023, 1, 1))
