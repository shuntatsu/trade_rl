from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_artifact import (
    CAUSAL_SCENARIO_ARRAYS_NAME,
    CAUSAL_SCENARIO_MANIFEST_NAME,
    CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA,
    load_causal_scenario_value_artifact,
    write_causal_scenario_value_artifact,
)
from trade_rl.evaluation.causal_scenario_values import (
    CausalQuerySnapshot,
    CausalScenarioEvaluatorConfig,
    CausalScenarioSet,
    ProjectedResidualCandidate,
    ScenarioRolloutEvidence,
    evaluate_causal_scenario_actions,
)


def sha(char: str) -> str:
    return char * 64


class Rollout:
    def __init__(self, query: CausalQuerySnapshot, coefficient: float) -> None:
        self.query = query
        self.coefficient = coefficient

    def run(self, candidate, *, horizon_decisions, zero_residual_after_first):
        log_return = self.coefficient * float(candidate.raw_action[0])
        terminal = self.query.starting_equity * math.exp(log_return)
        payload = {
            "feasible": True,
            "fill_ratio": 1.0,
            "filled_turnover": float(abs(candidate.raw_action[0])),
            "interval_cost": 0.0,
            "reported_log_return": log_return,
            "schema_version": "scenario_rollout_evidence_v1",
            "terminal_equity": terminal,
            "termination_reason": "horizon",
        }
        return ScenarioRolloutEvidence(
            terminal_equity=terminal,
            reported_log_return=log_return,
            filled_turnover=float(abs(candidate.raw_action[0])),
            interval_cost=0.0,
            fill_ratio=1.0,
            feasible=True,
            termination_reason="horizon",
            evidence_digest=content_digest(payload),
        )


class Factory:
    def project_candidate(self, query, raw_action):
        raw = np.asarray(raw_action, dtype=np.float64)
        target = np.clip(query.baseline_target + 0.25 * raw, -0.45, 0.45)
        execution = content_digest({"target": target.tolist()})
        digest = content_digest(
            {
                "execution_intent_digest": execution,
                "projected_target": target.tolist(),
                "schema_version": "projected_residual_candidate_v1",
            }
        )
        return ProjectedResidualCandidate(
            raw_action=raw,
            projected_target=target,
            execution_intent_digest=execution,
            candidate_digest=digest,
            expected_turnover_hint=float(np.abs(raw).sum()),
            is_zero=bool(np.all(raw == 0.0)),
        )

    def create_rollout(self, query, scenario_index, scenario_id):
        return Rollout(query, 0.01 + scenario_index * 0.0001)


def valid_result():
    count = 64
    query = CausalQuerySnapshot(
        dataset_id=sha("a"),
        fold_digest=sha("b"),
        train_start=0,
        train_stop=1000,
        query_index=1001,
        query_timestamp_ns=1,
        source_commit="c" * 40,
        query_digest=sha("1"),
        state_snapshot_digest=sha("2"),
        observation_digest=sha("3"),
        environment_digest=sha("4"),
        action_spec_digest=sha("5"),
        execution_policy_digest=sha("6"),
        risk_digest=sha("7"),
        trend_digest=sha("8"),
        starting_equity=100.0,
        baseline_target=np.asarray([0.0]),
    )
    scenarios = CausalScenarioSet(
        scenario_ids=tuple(f"s-{i}" for i in range(count)),
        probabilities=np.full(count, 1.0 / count),
        anchor_indices=np.arange(count, dtype=np.int64),
        distances=np.arange(count, dtype=np.float64),
        query_condition=np.asarray([0.0]),
        anchor_conditions=np.zeros((count, 1)),
        library_digest=sha("9"),
    )
    return evaluate_causal_scenario_actions(
        query=query,
        scenarios=scenarios,
        config=CausalScenarioEvaluatorConfig(action_dimension=1),
        rollout_factory=Factory(),
    )


def test_artifact_round_trip_is_deterministic(tmp_path: Path) -> None:
    result = valid_result()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_digest = write_causal_scenario_value_artifact(first, result)
    second_digest = write_causal_scenario_value_artifact(second, result)
    assert CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA == "causal_scenario_value_artifact_v1"
    assert first_digest == second_digest
    assert (first / CAUSAL_SCENARIO_MANIFEST_NAME).read_bytes() == (
        second / CAUSAL_SCENARIO_MANIFEST_NAME
    ).read_bytes()
    assert (first / CAUSAL_SCENARIO_ARRAYS_NAME).read_bytes() == (
        second / CAUSAL_SCENARIO_ARRAYS_NAME
    ).read_bytes()
    loaded = load_causal_scenario_value_artifact(first)
    assert loaded.result_digest == result.result_digest
    np.testing.assert_array_equal(
        loaded.baseline_relative_advantages,
        result.baseline_relative_advantages,
    )
    assert not loaded.raw_candidate_actions.flags.writeable


def test_writer_rejects_nonempty_destination(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "existing").write_text("x")
    with pytest.raises(FileExistsError):
        write_causal_scenario_value_artifact(root, valid_result())


def test_loader_rejects_file_closure_and_tampering(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_value_artifact(root, valid_result())
    (root / "extra").write_text("x")
    with pytest.raises(ValueError, match="closure"):
        load_causal_scenario_value_artifact(root)
    (root / "extra").unlink()

    manifest_path = root / CAUSAL_SCENARIO_MANIFEST_NAME
    raw = json.loads(manifest_path.read_text())
    raw["query_index"] += 1
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="manifest digest"):
        load_causal_scenario_value_artifact(root)


def test_loader_rejects_arrays_digest_tampering(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_value_artifact(root, valid_result())
    arrays_path = root / CAUSAL_SCENARIO_ARRAYS_NAME
    arrays_path.write_bytes(arrays_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="arrays digest"):
        load_causal_scenario_value_artifact(root)


def test_evaluation_package_exports_causal_scenario_api() -> None:
    from trade_rl import evaluation

    assert evaluation.CAUSAL_SCENARIO_EVALUATOR_SCHEMA == (
        "causal_scenario_action_evaluator_v1"
    )
    assert evaluation.CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA == (
        "causal_scenario_value_artifact_v1"
    )
    assert (
        evaluation.evaluate_causal_scenario_actions is evaluate_causal_scenario_actions
    )
    assert (
        evaluation.write_causal_scenario_value_artifact
        is write_causal_scenario_value_artifact
    )


def _rewrite_manifest(root: Path, mutate) -> dict[str, object]:
    from trade_rl.artifacts.codec import canonical_json_bytes
    from trade_rl.artifacts.hashing import content_digest

    path = root / CAUSAL_SCENARIO_MANIFEST_NAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("artifact_digest")
    mutate(raw)
    raw["artifact_digest"] = content_digest(raw)
    path.write_bytes(canonical_json_bytes(raw))
    return raw


def _loaded_arrays(root: Path) -> dict[str, np.ndarray]:
    import io

    path = root / CAUSAL_SCENARIO_ARRAYS_NAME
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_artifact_private_validators_reject_wrong_types(tmp_path: Path) -> None:
    from trade_rl.evaluation import causal_scenario_artifact as module

    with pytest.raises(ValueError, match="mapping"):
        module._mapping([], field="mapping")
    with pytest.raises(ValueError, match="non-empty string"):
        module._string("", field="text")
    with pytest.raises(ValueError, match="integer"):
        module._integer(True, field="integer")
    with pytest.raises(ValueError, match="finite real"):
        module._finite_float("1.0", field="float")
    with pytest.raises(ValueError, match="finite real"):
        module._finite_float(float("inf"), field="float")
    with pytest.raises(ValueError, match="sequence"):
        module._sequence("not-a-sequence", field="sequence")
    with pytest.raises(FileNotFoundError, match="directory"):
        module._verify_exact_files(tmp_path / "missing")

    root = tmp_path / "invalid-entry"
    root.mkdir()
    (root / CAUSAL_SCENARIO_MANIFEST_NAME).mkdir()
    (root / CAUSAL_SCENARIO_ARRAYS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="invalid file entry"):
        module._verify_exact_files(root)


def test_loader_rejects_schema_metadata_and_manifest_closure(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_value_artifact(root, valid_result())

    _rewrite_manifest(root, lambda raw: raw.__setitem__("schema_version", "bad"))
    with pytest.raises(ValueError, match="schema"):
        load_causal_scenario_value_artifact(root)

    root = tmp_path / "metadata"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(root, lambda raw: raw["array_metadata"].pop("score"))
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_causal_scenario_value_artifact(root)

    root = tmp_path / "arrays-file"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(root, lambda raw: raw.__setitem__("arrays_file", "wrong.npz"))
    with pytest.raises(ValueError, match="arrays file identity"):
        load_causal_scenario_value_artifact(root)

    root = tmp_path / "manifest-closure"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(root, lambda raw: raw.__setitem__("unexpected", True))
    with pytest.raises(ValueError, match="field closure"):
        load_causal_scenario_value_artifact(root)


def test_loader_rejects_array_names_metadata_shape_and_dtype(tmp_path: Path) -> None:
    from trade_rl.evaluation import causal_scenario_artifact as module

    def replace_arrays(root: Path, arrays: dict[str, np.ndarray]) -> None:
        payload = module._deterministic_npz(arrays)
        (root / CAUSAL_SCENARIO_ARRAYS_NAME).write_bytes(payload)
        _rewrite_manifest(
            root,
            lambda raw: raw.__setitem__("arrays_digest", module._sha256_bytes(payload)),
        )

    root = tmp_path / "names"
    write_causal_scenario_value_artifact(root, valid_result())
    arrays = _loaded_arrays(root)
    arrays.pop("score")
    replace_arrays(root, arrays)
    with pytest.raises(ValueError, match="array names"):
        load_causal_scenario_value_artifact(root)

    root = tmp_path / "meta-keys"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(
        root,
        lambda raw: raw["array_metadata"]["score"].__setitem__("extra", 1),
    )
    with pytest.raises(ValueError, match="metadata invalid"):
        load_causal_scenario_value_artifact(root)

    root = tmp_path / "shape"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(
        root,
        lambda raw: raw["array_metadata"]["score"].__setitem__("shape", [999]),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_causal_scenario_value_artifact(root)

    root = tmp_path / "dtype"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(
        root,
        lambda raw: raw["array_metadata"]["score"].__setitem__("dtype", "<i8"),
    )
    with pytest.raises(ValueError, match="dtype mismatch"):
        load_causal_scenario_value_artifact(root)


def test_loader_rejects_config_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(root, lambda raw: raw.__setitem__("config_digest", "f" * 64))
    with pytest.raises(ValueError, match="config digest"):
        load_causal_scenario_value_artifact(root)


def test_loader_rejects_config_payload_field_closure(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_value_artifact(root, valid_result())
    _rewrite_manifest(root, lambda raw: raw["config_payload"].pop("cvar_alpha"))
    with pytest.raises(ValueError, match="config payload field closure"):
        load_causal_scenario_value_artifact(root)
