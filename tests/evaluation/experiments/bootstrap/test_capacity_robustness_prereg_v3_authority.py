from trade_rl.evaluation.experiments.bootstrap.capacity_robustness_prereg import (
    canonical_capacity_robustness_protocol,
)


def test_v3_binds_final_publisher_bound_successor_authority() -> None:
    protocol = canonical_capacity_robustness_protocol()

    assert protocol.schema_version == "calibrated_capacity_robustness_prereg_v3"
    assert protocol.capacity_result_run_id == 34766830666
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
    assert protocol.successor_dataset_tree_digest == (
        "2dd187a2cf4634e1c902f58592a15b4942127190a9eee16e0efe004da8a470f0"
    )
    assert protocol.successor_study_digest == (
        "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
    )
    assert protocol.successor_study_tree_digest == (
        "5fda054135f90ff6b35353a145dfe488dbeb45d43f9a5d391bc7e394c2dcc151"
    )
    assert protocol.successor_materialization_index_sha256 == (
        "9bddde683d4ca8d3596fb62a5612b1b4487aaa1dc11ef631923b5e51e030ba3c"
    )
    assert protocol.successor_fresh_verification_run_id == 34803432434
    assert protocol.successor_fresh_verification_artifact_id == 10332575500
    assert protocol.successor_fresh_verification_artifact_digest == (
        "2067c38da3c1f45927748e2cc7481f11268710b2b386a5bd3b9a289394937589"
    )
    assert protocol.integration_gate_satisfied is False
    assert protocol.execution_authorized is False
    assert protocol.stage_a_executed is False
    assert protocol.stage_b_executed is False
    assert protocol.successor_pnl_inspected is False
