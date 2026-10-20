from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tests.tools.market_b4_oracle._native_replay_fixture import _public_native_fixture


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _frozen_h_plan(tmp_path: Path):
    import tools.market_b4_outer_runner as outer
    from tools.market_b4_oracle.outer_witness import _make_frozen_h_runtime_pin

    observer = tmp_path / "observer"
    child = tmp_path / "frozen-h"
    python = child / ".venv" / "Scripts" / "python.exe"
    runtime = {
        "schema_version": "candidate_run_runtime_v1",
        "python": {"implementation": "CPython", "version": "3.12.3"},
        "os": {"family": "Windows", "release": "11"},
        "machine": "AMD64",
        "packages": {
            "trade-rl": "0.3.0",
            "numpy": "1.26.4",
            "gymnasium": "0.29.1",
            "lightgbm": "4.7.0",
        },
    }
    runtime_bytes = _canonical(runtime)
    runtime_digest = hashlib.sha256(runtime_bytes).hexdigest()
    environment_policy = outer._frozen_h_environment_policy(
        child, outer._observer_environment()
    )
    module_resolution = {
        "working_directory": str(child),
        "sys_path0": str(child),
        "trade_rl_origin": str(child / "trade_rl" / "__init__.py"),
        "tier_d_nonrl_main": str(child / "tools" / "tier_d_nonrl" / "__main__.py"),
    }
    probe = {
        "schema": "issue810_frozen_h_runtime_probe_v1",
        "source_root": str(child),
        "python_executable": str(python),
        "python_executable_sha256": "a" * 64,
        "module_resolution": module_resolution,
        "environment_policy": environment_policy,
        "runtime_environment_sha256": runtime_digest,
    }
    probe_bytes = _canonical(probe)
    runtime_pin = _make_frozen_h_runtime_pin(
        source_root=child,
        python_executable=python,
        python_executable_sha256="a" * 64,
        runtime_manifest_canonical=runtime_bytes,
        runtime_environment_sha256=runtime_digest,
        probe_facts_canonical=probe_bytes,
        probe_facts_sha256=hashlib.sha256(probe_bytes).hexdigest(),
    )
    attempt_uuid = "1" * 32
    attempt_root = child / "output" / "tier-d-nonrl" / "attempt"
    output_root = attempt_root / "attempt"
    witness_root = observer / "output" / "verify" / "attempt"
    protocol = child / "protocol-v3.json"
    review = child / "review.json"
    argv = outer._frozen_h_public_argv(
        child_python=python,
        attempt_root=attempt_root,
        protocol_path=protocol,
        review_path=review,
        max_workers=4,
    )
    child_binding = {
        "scope": "FROZEN_H_PUBLIC_CLI_PLAN_ONLY",
        "source_root": str(child),
        "cwd": str(child),
        "python_executable": str(python),
        "runtime_pin": runtime_pin.authority_payload(),
        "environment_policy": environment_policy,
        "argv": list(argv),
        "attempt_uuid": attempt_uuid,
        "supervisor_attempt_root": str(attempt_root),
        "output_root": str(output_root),
        "protocol_v3": {
            "path": str(protocol),
            "bytes": 2,
            "sha256": "b" * 64,
        },
        "protocol_v4": {"bytes": 3, "sha256": "c" * 64},
        "review_record": {"path": str(review), "bytes": 4, "sha256": "d" * 64},
        "native_source": {"head_sha": "e" * 40, "snapshot_sha256": "f" * 64},
    }
    authority = {
        "scope": "NATIVE_H3_V4_STATIC_PINS",
        "child_binding": child_binding,
        "observer_source": {
            "root": str(observer),
            "closure_sha256": "0" * 64,
            "repository_head": "1" * 40,
            "working_tree_clean": False,
            "git_status_sha256": "2" * 64,
        },
    }
    plan = outer.FrozenHLaunchPlan(
        observer_repository=observer,
        observer_source_sha256="0" * 64,
        child_source_root=child,
        child_python=python,
        runtime_pin=runtime_pin,
        supervisor_attempt_root=attempt_root,
        output_root=output_root,
        witness_root=witness_root,
        protocol_v3_path=protocol,
        review_record_path=review,
        protocol_v3_pin=outer.WitnessFilePin(2, "b" * 64),
        protocol_v4_pin=outer.WitnessFilePin(3, "c" * 64),
        review_record_pin=outer.WitnessFilePin(4, "d" * 64),
        attempt_uuid=attempt_uuid,
        argv=argv,
        input_authority_canonical=_canonical(authority),
        _plan_token=outer._FROZEN_H_PLAN_TOKEN,
    )
    start = {
        "schema": "issue810_root_owned_attempt_start_v1",
        "attempt_uuid": attempt_uuid,
        "argv": list(argv),
        "cwd": str(child),
        "output_root": str(output_root),
        "input_authority": authority,
        "child_environment_policy": environment_policy,
    }
    return plan, start


def test_frozen_h_start_witness_matches_the_typed_child_plan(tmp_path: Path) -> None:
    import tools.market_b4_outer_runner as outer

    plan, start = _frozen_h_plan(tmp_path)
    validator = getattr(outer, "_assert_frozen_h_start_matches_plan", None)

    assert callable(validator), "typed frozen-H start validator is missing"
    validator(plan, start)


@pytest.mark.parametrize(
    "field", ["attempt_uuid", "argv", "cwd", "runtime_pin", "environment_policy"]
)
def test_frozen_h_start_witness_rejects_child_binding_drift(
    tmp_path: Path, field: str
) -> None:
    import tools.market_b4_outer_runner as outer

    plan, start = _frozen_h_plan(tmp_path)
    validator = getattr(outer, "_assert_frozen_h_start_matches_plan", None)
    assert callable(validator), "typed frozen-H start validator is missing"
    changed = json.loads(json.dumps(start))
    if field == "runtime_pin":
        changed["input_authority"]["child_binding"]["runtime_pin"][
            "runtime_environment_sha256"
        ] = "9" * 64
    elif field == "environment_policy":
        changed["input_authority"]["child_binding"]["environment_policy"][
            "worker_environment"
        ]["PYTHONPATH"] = str(tmp_path / "foreign")
    elif field == "argv":
        changed["argv"][-1] = "5"
    elif field == "cwd":
        changed["cwd"] = str(tmp_path / "different-child")
    else:
        changed[field] = "9" * 32

    with pytest.raises(ValueError, match="frozen H start witness differs"):
        validator(plan, changed)


def test_frozen_h_spawn_uses_only_planned_identity_and_child_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    plan, _ = _frozen_h_plan(tmp_path)
    plan.observer_repository.mkdir(parents=True)
    plan.child_source_root.mkdir(parents=True)
    plan.supervisor_attempt_root.parent.mkdir(parents=True)
    plan.witness_root.parent.mkdir(parents=True)
    identity = {"closure_sha256": "0" * 64}
    calls: list[tuple[list[str], dict[str, object]]] = []
    runtime = plan.runtime_pin

    def forbidden_native_spawn(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))
        raise RuntimeError("synthetic spawn trap")

    monkeypatch.setattr(outer, "implementation_source_identity", lambda _: identity)
    monkeypatch.setattr(outer, "_git_readback", lambda _: ("1" * 40, False, "2" * 64))
    monkeypatch.setattr(outer, "_assert_frozen_h_runtime_pin_current", lambda *_: None)
    monkeypatch.setattr(outer, "capture_frozen_h_runtime_pin", lambda _: runtime)
    monkeypatch.setattr(outer.subprocess, "Popen", forbidden_native_spawn)

    with pytest.raises(RuntimeError, match="synthetic spawn trap"):
        outer._run_frozen_h_attempt(plan, timeout_seconds=2.0)

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == list(plan.argv)
    assert kwargs["cwd"] == plan.child_source_root
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert not {
        "PYTHONPATH",
        "PYTHONHOME",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    } & set(kwargs["env"])
    start = json.loads((plan.witness_root / "start.json").read_text(encoding="utf-8"))
    assert start["attempt_uuid"] == plan.attempt_uuid
    assert start["cwd"] == str(plan.child_source_root)
    assert start["argv"] == list(plan.argv)
    assert not (plan.witness_root / "finish.json").exists()
    assert (plan.witness_root / "failure.json").is_file()


def test_frozen_h_environment_policy_excludes_import_and_thread_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "foreign"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "foreign-home"))
    monkeypatch.setenv("OMP_NUM_THREADS", "77")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "88")

    environment = outer._observer_environment()
    policy = outer._frozen_h_environment_policy(tmp_path, environment)

    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "OMP_NUM_THREADS" not in environment
    assert "OPENBLAS_NUM_THREADS" not in environment
    assert policy["worker_environment"]["PYTHONPATH"] == str(tmp_path)
    assert policy["absent_numerical_threads"]


def test_frozen_h_matrix_entry_rejects_incomplete_roster_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_oracle.offline_replay_consumer as consumer
    import tools.market_b4_outer_runner as outer

    plan, _ = _frozen_h_plan(tmp_path)
    static_inputs = object.__new__(consumer.StaticInputReadback)
    object.__setattr__(static_inputs, "source_root", plan.child_source_root)
    monkeypatch.setattr(
        consumer,
        "derive_frozen_native_market_roster",
        lambda **_: tuple(object() for _ in range(47)),
    )
    monkeypatch.setattr(
        outer,
        "_run_frozen_h_attempt",
        lambda **_: pytest.fail("incomplete roster must not spawn the H supervisor"),
    )

    with pytest.raises(ValueError, match="exact 48-arm roster"):
        outer.run_and_consume_frozen_h_market_matrix(
            plan=plan,
            protocol_pair=object(),
            content=object(),
            static_inputs=static_inputs,
            timeout_seconds=1.0,
        )


def test_artificial_matrix_join_unit_keeps_authority_flags_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub both sides of the outer join; no H child or matrix scan runs."""
    from types import SimpleNamespace

    import tools.market_b4_oracle.offline_replay_consumer as consumer
    import tools.market_b4_outer_runner as outer

    plan, start = _frozen_h_plan(tmp_path)
    authority = outer._frozen_h_plan_authority(plan)
    arms = tuple(
        SimpleNamespace(
            root_attempt_observed=True,
            outer_attempt_authenticated=False,
            full_b4_authorized=False,
        )
        for _ in range(48)
    )
    comparison = SimpleNamespace(
        status="PINNED_INPUT_MATRIX_COMPARISON_COMPLETE",
        arms=arms,
        root_attempt_observed=True,
        outer_attempt_authenticated=False,
        full_b4_authorized=False,
    )
    manifest = SimpleNamespace(bytes=37, sha256="a" * 64)
    output_index = SimpleNamespace(_root=plan.output_root, manifest_pin=manifest)
    start_pin = outer.WitnessFilePin(1, "b" * 64)
    finish_pin = outer.WitnessFilePin(2, "c" * 64)
    finish = {
        "child": {"exit_code": 0, "owned_tree_empty": True},
        "output_manifest": {"bytes": manifest.bytes, "sha256": manifest.sha256},
    }

    class StubRootWitness:
        def readback(self, *, expected_start_pin, expected_finish_pin):
            assert expected_start_pin == start_pin
            assert expected_finish_pin == finish_pin
            return start, finish

    # The required scope marker is a test-owned stub, not observed H execution.
    execution = outer.RootObservedExecution(
        output_index=output_index,
        root_witness=StubRootWitness(),
        expected_start_pin=start_pin,
        expected_finish_pin=finish_pin,
        input_authority=authority,
        scope="NATIVE_H3_V4_ROOT_OBSERVED",
    )
    monkeypatch.setattr(outer, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        outer, "_assert_frozen_h_matrix_plan", lambda *_args, **_kwargs: arms
    )
    monkeypatch.setattr(
        outer, "_run_frozen_h_attempt", lambda *_args, **_kwargs: execution
    )
    monkeypatch.setattr(outer, "_assert_frozen_h_start_matches_plan", lambda *_: None)
    monkeypatch.setattr(
        consumer, "consume_root_observed_market_matrix", lambda **_: comparison
    )
    monkeypatch.setattr(
        outer.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("synthetic orchestration must not spawn"),
    )

    result = outer.run_and_consume_frozen_h_market_matrix(
        plan=plan,
        protocol_pair=object(),
        content=object(),
        static_inputs=object(),
        timeout_seconds=1.0,
    )

    assert result.execution is execution
    assert result.comparison is comparison
    assert result.root_attempt_observed is True
    assert result.outer_attempt_authenticated is False
    assert result.full_b4_authorized is False
    assert len(result.comparison.arms) == 48


@pytest.mark.skipif(
    __import__("os").name != "nt",
    reason="frozen root-owned process-tree observer is Windows-only",
)
def test_outer_dummy_child_consumes_only_artificial_comparison_fixture(
    tmp_path,
) -> None:
    import sys

    from tools.market_b4_outer_runner import run_and_consume_market_arm

    owned, _, native, _, _ = _public_native_fixture(
        tmp_path,
        0,
        scenario="BASEL0",
        role="nonrl",
        horizon=72,
    )
    forecast_bytes = (tmp_path / "stream-0-BASEL0-nonrl-h72.json").read_bytes()
    fixture_root = tmp_path / "old-artificial-fixture"
    fixture_files = {
        "receipts/arm.json.gz": gzip.compress(
            json.dumps(
                native, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8"),
            mtime=0,
        ),
        "forecasts/arm.json": forecast_bytes,
    }
    for name, raw in fixture_files.items():
        path = fixture_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = json.dumps(
        {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in fixture_files.items()
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (fixture_root / "output-manifest.json").write_bytes(manifest)

    output_root = tmp_path / "fresh-child-output"
    witness_root = tmp_path / "fresh-child-witness"
    child_code = (
        "import pathlib,shutil,sys,time; source=pathlib.Path(sys.argv[1]); "
        "target=pathlib.Path(sys.argv[2]); target.mkdir(); "
        "shutil.copytree(source/'receipts',target/'receipts'); "
        "shutil.copytree(source/'forecasts',target/'forecasts'); "
        "shutil.copyfile(source/'output-manifest.json',target/'output-manifest.json'); "
        "time.sleep(3)"
    )
    repository = Path(__file__).resolve().parents[3]
    observed = run_and_consume_market_arm(
        argv=[
            sys.executable,
            "-B",
            "-c",
            child_code,
            str(fixture_root),
            str(output_root),
        ],
        repository=repository,
        output_root=output_root,
        witness_root=witness_root,
        content=owned.content,
        policy=owned.policy,
        binding=owned.binding,
        role="NONRL",
        reserve_plan_digest=owned.reserve_plan_digest,
    )

    assert observed.execution.scope == "ARTIFICIAL_COMPARISON_ONLY"
    assert (
        observed.execution.root_witness.readback(
            expected_start_pin=observed.execution.expected_start_pin,
            expected_finish_pin=observed.execution.expected_finish_pin,
        )[1]["child"]["owned_tree_empty"]
        is True
    )
    assert observed.comparison.status == "ARTIFICIAL_COMPARISON_COMPLETE"
    assert observed.comparison.root_attempt_observed is True
    assert observed.comparison.outer_attempt_authenticated is False
    assert observed.full_b4_authorized is False
