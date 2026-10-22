"""Root-owned observer for frozen market comparison attempts.

The CLI defaults to an offline plan. Process creation is available only through
the typed Python entry point, which binds the observed output to a frozen arm
and invokes the comparison consumer after the owned process tree has stopped.
This software evidence does not authorize research or a Full-B4 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from tomllib import loads as toml_loads
from typing import Any, Literal, Sequence
from uuid import uuid4

from tools.market_b4_oracle.artificial_spec import is_artificial_h_component_spec
from tools.market_b4_oracle.outer_witness import (
    FrozenHRuntimePin,
    RootObservedAttempt,
    WitnessFilePin,
    _canonical,
    _make_frozen_h_runtime_pin,
    _make_root_observed_attempt,
    implementation_source_identity,
)
from tools.market_b4_oracle.output_index import (
    ArtifactPin,
    NativeOutputIndex,
    capture_output_manifest_pin,
    load_pinned_output_index,
)
from tools.market_replay_oracle.source import NativeSourcePin, verify_native_source

_MAX_ARGV_ITEMS = 128
_MAX_ARGV_STRING_BYTES = 8192
_DEFAULT_TIMEOUT_SECONDS = 180
_SAMPLE_INTERVAL_SECONDS = 0.15
_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUTF8",
)
_IMPORT_RESOLUTION_ENVIRONMENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "PYTHONSTARTUP",
)
_NUMERICAL_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_FROZEN_H_RUNTIME_PROBE = """import importlib.metadata as metadata
import importlib.util
import json
import sys
from pathlib import Path

import trade_rl
from trade_rl.evaluation.runs.provenance import runtime_environment_manifest

module_spec = importlib.util.find_spec("tools.tier_d_nonrl.__main__")
if module_spec is None or module_spec.origin is None:
    raise RuntimeError("frozen H CLI module cannot be resolved")

def distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "schema": "issue810_frozen_h_runtime_probe_v1",
    "source_root": str(Path.cwd().resolve()),
    "python_executable": str(Path(sys.executable).resolve()),
    "module_root": str(Path(trade_rl.__file__).resolve().parent.parent),
    "module_resolution": {
        "working_directory": str(Path.cwd().resolve()),
        "sys_path0": str(Path(sys.path[0] or Path.cwd()).resolve()),
        "trade_rl_origin": str(Path(trade_rl.__file__).resolve()),
        "tier_d_nonrl_main": str(Path(module_spec.origin).resolve()),
    },
    "runtime_environment": runtime_environment_manifest(),
    "support_packages": {
        name: distribution_version(name)
        for name in ("scikit-learn", "scipy")
    },
}, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
"""


def _root_owned_process_tree_supported() -> bool:
    """Return whether this host has the frozen Windows process-tree observer."""
    return os.name == "nt"


@dataclass(frozen=True, slots=True)
class RootObservedExecution:
    """An output index paired with external root-owned start/finish pins."""

    output_index: NativeOutputIndex
    root_witness: RootObservedAttempt
    expected_start_pin: WitnessFilePin
    expected_finish_pin: WitnessFilePin
    input_authority: dict[str, object]
    scope: Literal[
        "ARTIFICIAL_COMPARISON_ONLY",
        "ARTIFICIAL_H_COMPONENT_PROCESS_OBSERVED_ONLY",
        "NATIVE_H3_V4_ROOT_OBSERVED",
    ] = "ARTIFICIAL_COMPARISON_ONLY"
    full_b4_authorized: Literal[False] = False


@dataclass(frozen=True, slots=True)
class RootObservedMarketArm:
    """A completed software comparison still barred from Full-B4 status."""

    execution: RootObservedExecution
    comparison: Any
    full_b4_authorized: Literal[False] = False


@dataclass(frozen=True, slots=True)
class RootObservedHMarketMatrix:
    """One fresh H supervisor plus a complete native comparison-only matrix."""

    execution: RootObservedExecution
    comparison: Any
    root_attempt_observed: Literal[True] = True
    outer_attempt_authenticated: Literal[False] = False
    full_b4_authorized: Literal[False] = False


@dataclass(frozen=True, slots=True)
class RootObservedArtificialHComponentPair:
    """Independent Q replay of both fixed artificial roles; never Full-B4."""

    execution: RootObservedExecution
    comparisons: tuple[Any, Any]
    scope: Literal["ARTIFICIAL_H_COMPONENT_Q_PAIR_COMPARISON_COMPLETE"] = (
        "ARTIFICIAL_H_COMPONENT_Q_PAIR_COMPARISON_COMPLETE"
    )
    outer_attempt_authenticated: Literal[False] = False
    full_b4_authorized: Literal[False] = False
    research_authorized: Literal[False] = False


_FROZEN_H_PLAN_TOKEN = object()
_ARTIFICIAL_H_PLAN_TOKEN = object()

_ARTIFICIAL_H_OPERATOR_RELATIVE = Path(
    "tools/market_b4_artificial_component_operator.py"
)
_ARTIFICIAL_H_PLAN_SCHEMA = "issue810_artificial_h_component_plan_v1"


@dataclass(frozen=True, slots=True)
class FrozenHLaunchPlan:
    """Typed, no-process plan for H's exact public supervised market CLI."""

    observer_repository: Path
    observer_source_sha256: str
    child_source_root: Path
    child_python: Path
    runtime_pin: FrozenHRuntimePin
    supervisor_attempt_root: Path
    output_root: Path
    witness_root: Path
    protocol_v3_path: Path
    review_record_path: Path
    protocol_v3_pin: WitnessFilePin
    protocol_v4_pin: WitnessFilePin
    review_record_pin: WitnessFilePin
    attempt_uuid: str
    argv: tuple[str, ...]
    input_authority_canonical: bytes
    scope: Literal["FROZEN_H_PUBLIC_CLI_PLAN_ONLY"] = "FROZEN_H_PUBLIC_CLI_PLAN_ONLY"
    process_started: Literal[False] = False
    full_b4_authorized: Literal[False] = False
    _plan_token: object = dataclass_field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if (
            self._plan_token is not _FROZEN_H_PLAN_TOKEN
            or self.scope != "FROZEN_H_PUBLIC_CLI_PLAN_ONLY"
            or self.process_started is not False
            or self.full_b4_authorized is not False
            or self.supervisor_attempt_root.parent
            != self.child_source_root / "output" / "tier-d-nonrl"
            or type(self.runtime_pin) is not FrozenHRuntimePin
            or self.runtime_pin.source_root != self.child_source_root
            or self.runtime_pin.python_executable != self.child_python
            or self.output_root != self.supervisor_attempt_root / "attempt"
            or len(self.attempt_uuid) != 32
            or any(char not in "0123456789abcdef" for char in self.attempt_uuid)
            or type(self.input_authority_canonical) is not bytes
        ):
            raise ValueError("frozen H launch plan is not an exact plan-only record")
        expected = _frozen_h_public_argv(
            child_python=self.child_python,
            attempt_root=self.supervisor_attempt_root,
            protocol_path=self.protocol_v3_path,
            review_path=self.review_record_path,
            max_workers=int(self.argv[-1]),
        )
        if self.argv != expected:
            raise ValueError("frozen H plan differs from its fixed public CLI")


@dataclass(frozen=True, slots=True)
class ArtificialHComponentPlan:
    """Typed plan for a fixed synthetic H component operator, never native H."""

    observer_repository: Path
    observer_source_sha256: str
    observer_head_sha: str
    child_source_root: Path
    child_source_pin: NativeSourcePin
    child_python: Path
    runtime_pin: FrozenHRuntimePin
    operator_path: Path
    operator_sha256: str
    component_spec_canonical: bytes
    component_spec_sha256: str
    input_plan_path: Path
    input_plan_canonical: bytes
    output_root: Path
    witness_root: Path
    attempt_uuid: str
    argv: tuple[str, ...]
    input_authority_canonical: bytes
    scope: Literal["ARTIFICIAL_H_COMPONENT_PLAN_ONLY"] = (
        "ARTIFICIAL_H_COMPONENT_PLAN_ONLY"
    )
    process_started: Literal[False] = False
    outer_attempt_authenticated: Literal[False] = False
    full_b4_authorized: Literal[False] = False
    _plan_token: object = dataclass_field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if (
            self._plan_token is not _ARTIFICIAL_H_PLAN_TOKEN
            or self.scope != "ARTIFICIAL_H_COMPONENT_PLAN_ONLY"
            or self.process_started is not False
            or self.outer_attempt_authenticated is not False
            or self.full_b4_authorized is not False
            or type(self.child_source_pin) is not NativeSourcePin
            or type(self.runtime_pin) is not FrozenHRuntimePin
            or self.runtime_pin.source_root != self.child_source_root
            or self.runtime_pin.python_executable != self.child_python
            or self.child_python
            != self.child_source_root / ".venv" / "Scripts" / "python.exe"
            or self.operator_path
            != self.observer_repository / _ARTIFICIAL_H_OPERATOR_RELATIVE
            or not _valid_sha256(self.observer_source_sha256)
            or not _valid_sha256(self.operator_sha256)
            or not _valid_sha256(self.component_spec_sha256)
            or type(self.component_spec_canonical) is not bytes
            or hashlib.sha256(self.component_spec_canonical).hexdigest()
            != self.component_spec_sha256
            or type(self.input_plan_canonical) is not bytes
            or type(self.input_authority_canonical) is not bytes
            or not _is_uuid_hex(self.attempt_uuid)
            or self.output_root == self.witness_root
            or self.output_root.is_relative_to(self.witness_root)
            or self.witness_root.is_relative_to(self.output_root)
        ):
            raise ValueError("artificial H component plan is malformed")
        spec = _parse_canonical_object(self.component_spec_canonical)
        document = _parse_canonical_object(self.input_plan_canonical)
        authority = _parse_canonical_object(self.input_authority_canonical)
        authority_dataset = authority.get("dataset")
        if (
            not _is_artificial_h_component_spec(spec)
            or document.get("schema") != _ARTIFICIAL_H_PLAN_SCHEMA
            or document.get("scope") != "ARTIFICIAL_H_COMPONENT_ONLY"
            or document.get("component_spec") != spec
            or document.get("component_spec_sha256") != self.component_spec_sha256
            or document.get("attempt_uuid") != self.attempt_uuid
            or document.get("output_root") != str(self.output_root)
            or document.get("outer_attempt_authenticated") is not False
            or document.get("full_b4_authorized") is not False
            or authority.get("scope") != "ARTIFICIAL_H_COMPONENT_ONLY"
            or authority.get("component_spec_sha256") != self.component_spec_sha256
            or not _valid_sha256(authority.get("binding_digest"))
            or type(authority_dataset) is not dict
            or set(authority_dataset) != {"dataset_id"}
            or not _valid_sha256(authority_dataset.get("dataset_id"))
            or authority.get("attempt_uuid") != self.attempt_uuid
            or authority.get("outer_attempt_authenticated") is not False
            or authority.get("full_b4_authorized") is not False
            or self.argv
            != _artificial_h_component_argv(
                child_python=self.child_python,
                operator_path=self.operator_path,
                input_plan_path=self.input_plan_path,
            )
        ):
            raise ValueError(
                "artificial H component plan differs from its pinned input"
            )

    @property
    def component_spec(self) -> dict[str, object]:
        return json.loads(self.component_spec_canonical)

    @property
    def input_authority(self) -> dict[str, object]:
        return json.loads(self.input_authority_canonical)


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_uuid_hex(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_canonical_object(raw: bytes) -> dict[str, object]:
    value = json.loads(raw)
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("artificial H plan JSON is not a canonical object")
    return value


def _is_artificial_h_component_spec(value: object) -> bool:
    return is_artificial_h_component_spec(value)


def _artificial_h_component_argv(
    *, child_python: Path, operator_path: Path, input_plan_path: Path
) -> tuple[str, ...]:
    if any(
        not isinstance(path, Path) or not path.is_absolute()
        for path in (child_python, operator_path, input_plan_path)
    ):
        raise ValueError("artificial H component paths must be absolute")
    return (
        str(child_python),
        "-I",
        "-B",
        str(operator_path),
        "--plan",
        str(input_plan_path),
    )


def _frozen_h_public_argv(
    *,
    child_python: Path,
    attempt_root: Path,
    protocol_path: Path,
    review_path: Path,
    max_workers: int,
) -> tuple[str, ...]:
    if (
        any(
            not isinstance(path, Path) or not path.is_absolute()
            for path in (
                child_python,
                attempt_root,
                protocol_path,
                review_path,
            )
        )
        or type(max_workers) is not int
        or not 1 <= max_workers <= 6
    ):
        raise ValueError("frozen H public CLI paths or worker count are malformed")
    return (
        str(child_python),
        "-B",
        "-m",
        "tools.tier_d_nonrl",
        "--mode",
        "market",
        "--attempt-root",
        str(attempt_root),
        "--protocol",
        str(protocol_path),
        "--review-record",
        str(review_path),
        "--max-workers",
        str(max_workers),
    )


def _assert_frozen_h_start_matches_plan(
    plan: FrozenHLaunchPlan, start: dict[str, object]
) -> None:
    """Require an external start record to bind the exact typed frozen-H plan."""
    if (
        type(plan) is not FrozenHLaunchPlan
        or plan._plan_token is not _FROZEN_H_PLAN_TOKEN
        or type(start) is not dict
        or start.get("schema") != "issue810_root_owned_attempt_start_v1"
        or start.get("attempt_uuid") != plan.attempt_uuid
        or start.get("argv") != list(plan.argv)
        or start.get("cwd") != str(plan.child_source_root)
        or start.get("output_root") != str(plan.output_root)
    ):
        raise ValueError("frozen H start witness differs from its typed plan")

    authority = start.get("input_authority")
    if type(authority) is not dict:
        raise ValueError("frozen H start witness differs from its typed plan")
    try:
        plan_authority = json.loads(plan.input_authority_canonical)
        if (
            type(plan_authority) is not dict
            or _canonical(plan_authority) != plan.input_authority_canonical
            or _canonical(authority) != plan.input_authority_canonical
        ):
            raise ValueError("noncanonical plan authority")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            "frozen H start witness differs from its typed plan"
        ) from error

    child = plan_authority.get("child_binding")
    if (
        plan_authority.get("scope") != "NATIVE_H3_V4_STATIC_PINS"
        or type(child) is not dict
    ):
        raise ValueError("frozen H start witness differs from its typed plan")
    protocol_v3 = child.get("protocol_v3")
    protocol_v4 = child.get("protocol_v4")
    review_record = child.get("review_record")
    native_source = child.get("native_source")
    runtime_authority = plan.runtime_pin.authority_payload()
    runtime_facts = runtime_authority.get("probe_facts")
    if type(runtime_facts) is not dict:
        raise ValueError("frozen H start witness differs from its typed plan")
    environment_policy = runtime_facts.get("environment_policy")
    expected_child_values = {
        "scope": "FROZEN_H_PUBLIC_CLI_PLAN_ONLY",
        "source_root": str(plan.child_source_root),
        "cwd": str(plan.child_source_root),
        "python_executable": str(plan.child_python),
        "runtime_pin": runtime_authority,
        "environment_policy": environment_policy,
        "argv": list(plan.argv),
        "attempt_uuid": plan.attempt_uuid,
        "supervisor_attempt_root": str(plan.supervisor_attempt_root),
        "output_root": str(plan.output_root),
    }
    expected_v3 = {
        "path": str(plan.protocol_v3_path),
        "bytes": plan.protocol_v3_pin.byte_count,
        "sha256": plan.protocol_v3_pin.sha256,
    }
    expected_v4 = {
        "bytes": plan.protocol_v4_pin.byte_count,
        "sha256": plan.protocol_v4_pin.sha256,
    }
    expected_review = {
        "path": str(plan.review_record_path),
        "bytes": plan.review_record_pin.byte_count,
        "sha256": plan.review_record_pin.sha256,
    }
    observer = plan_authority.get("observer_source")
    if (
        any(child.get(key) != value for key, value in expected_child_values.items())
        or start.get("child_environment_policy") != environment_policy
        or protocol_v3 != expected_v3
        or protocol_v4 != expected_v4
        or review_record != expected_review
        or type(native_source) is not dict
        or type(native_source.get("head_sha")) is not str
        or len(native_source["head_sha"]) != 40
        or any(char not in "0123456789abcdef" for char in native_source["head_sha"])
        or type(native_source.get("snapshot_sha256")) is not str
        or len(native_source["snapshot_sha256"]) != 64
        or any(
            char not in "0123456789abcdef" for char in native_source["snapshot_sha256"]
        )
        or type(observer) is not dict
        or observer.get("root") != str(plan.observer_repository)
        or observer.get("closure_sha256") != plan.observer_source_sha256
    ):
        raise ValueError("frozen H start witness differs from its typed plan")


def _frozen_h_plan_authority(plan: FrozenHLaunchPlan) -> dict[str, object]:
    if (
        type(plan) is not FrozenHLaunchPlan
        or plan._plan_token is not _FROZEN_H_PLAN_TOKEN
    ):
        raise ValueError("frozen H launch requires an exact typed plan")
    try:
        value = json.loads(plan.input_authority_canonical)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen H plan authority is malformed") from error
    if type(value) is not dict or _canonical(value) != plan.input_authority_canonical:
        raise ValueError("frozen H plan authority is not canonical")
    return value


def _frozen_h_child_environment_policy(
    authority: dict[str, object],
) -> dict[str, object]:
    child_binding = authority.get("child_binding")
    if type(child_binding) is not dict:
        raise ValueError("frozen H plan lacks a child binding")
    environment_policy = child_binding.get("environment_policy")
    if type(environment_policy) is not dict:
        raise ValueError("frozen H plan lacks a child environment policy")
    return environment_policy


def _assert_frozen_h_plan_observer_current(plan: FrozenHLaunchPlan) -> None:
    authority = _frozen_h_plan_authority(plan)
    observer = authority.get("observer_source")
    if type(observer) is not dict:
        raise ValueError("frozen H plan lacks the observer source binding")
    identity = implementation_source_identity(plan.observer_repository)
    head, clean, status_sha256 = _git_readback(plan.observer_repository)
    if (
        observer.get("root") != str(plan.observer_repository)
        or observer.get("closure_sha256") != identity.get("closure_sha256")
        or observer.get("repository_head") != head
        or observer.get("working_tree_clean") is not clean
        or observer.get("git_status_sha256") != status_sha256
        or identity.get("closure_sha256") != plan.observer_source_sha256
    ):
        raise ValueError("frozen H observer source changed after plan creation")


def _run_frozen_h_attempt(
    plan: FrozenHLaunchPlan, *, timeout_seconds: float
) -> RootObservedExecution:
    """Start only the plan's fixed H CLI and bind it to the root witness."""
    authority = _frozen_h_plan_authority(plan)
    if not _root_owned_process_tree_supported():
        raise RuntimeError("frozen H market supervision requires Windows")
    _assert_frozen_h_runtime_pin_current(plan.runtime_pin, plan.child_source_root)
    reprobed_runtime = capture_frozen_h_runtime_pin(plan.child_source_root)
    if reprobed_runtime.authority_payload() != plan.runtime_pin.authority_payload():
        raise ValueError("frozen H runtime facts changed after plan creation")
    _assert_frozen_h_plan_observer_current(plan)
    start_binding: dict[str, object] = {
        "schema": "issue810_root_owned_attempt_start_v1",
        "attempt_uuid": plan.attempt_uuid,
        "argv": list(plan.argv),
        "cwd": str(plan.child_source_root),
        "output_root": str(plan.output_root),
        "input_authority": authority,
        "child_environment_policy": _frozen_h_child_environment_policy(authority),
    }
    _assert_frozen_h_start_matches_plan(plan, start_binding)
    return _run_root_observed_attempt(
        argv=plan.argv,
        repository=plan.observer_repository,
        output_root=plan.output_root,
        witness_root=plan.witness_root,
        input_authority=authority,
        timeout_seconds=timeout_seconds,
        attempt_uuid=plan.attempt_uuid,
        child_cwd=plan.child_source_root,
        frozen_h_plan=plan,
        _completion_scope="NATIVE_H3_V4_ROOT_OBSERVED",
    )


def build_frozen_h_launch_plan(
    *,
    observer_repository: Path,
    supervisor_attempt_root: Path,
    witness_root: Path,
    content: Any,
    policy: Any,
    binding: Any,
    runtime_pin: FrozenHRuntimePin,
    static_inputs: Any,
    review_record_path: Path,
    review_record_pin: WitnessFilePin,
) -> FrozenHLaunchPlan:
    """Validate source pins and describe H's public command without spawning it."""
    if getattr(binding, "comparison_only", None) is True:
        raise ValueError("artificial inputs cannot produce an H market launch plan")
    if os.name != "nt":
        raise RuntimeError("frozen H market CLI planning requires its Windows profile")
    authority = _frozen_input_authority(
        content=content,
        policy=policy,
        binding=binding,
        static_inputs=static_inputs,
    )
    from tools.market_b4_oracle.offline_replay_consumer import StaticInputReadback
    from tools.market_b4_oracle.outer_witness import _read_source_file

    if type(static_inputs) is not StaticInputReadback:
        raise ValueError("frozen H plan requires an exact static input readback")
    if type(review_record_pin) is not WitnessFilePin:
        raise ValueError("frozen H plan requires an independently supplied review pin")
    if type(runtime_pin) is not FrozenHRuntimePin:
        raise ValueError("frozen H plan requires an independently captured runtime pin")
    observer_repository = _safe_target(observer_repository, must_exist=True)
    child_source_root = _safe_target(static_inputs.source_root, must_exist=True)
    if observer_repository == child_source_root:
        raise ValueError("comparison source and frozen H child source must be separate")
    _assert_frozen_h_runtime_pin_current(runtime_pin, child_source_root)
    supervisor_attempt_root = _safe_target(supervisor_attempt_root, must_exist=False)
    witness_root = _fresh_h_witness_root(observer_repository, witness_root)
    expected_output_root = child_source_root / "output" / "tier-d-nonrl"
    if (
        supervisor_attempt_root.parent != expected_output_root
        or supervisor_attempt_root.exists()
        or witness_root == supervisor_attempt_root
        or witness_root.is_relative_to(supervisor_attempt_root)
        or supervisor_attempt_root.is_relative_to(witness_root)
    ):
        raise ValueError(
            "frozen H output and root witness require separate fresh roots"
        )
    protocol_v3_path = Path(static_inputs.protocol_v3_path).absolute()
    review_record_path = Path(review_record_path).absolute()
    from tools.market_b4_oracle.outer_witness import _source_components

    _source_components(protocol_v3_path, directory=False)
    _source_components(Path(static_inputs.protocol_v4_path).absolute(), directory=False)
    _, review_bytes = _read_source_file(review_record_path)
    if (
        len(review_bytes) != review_record_pin.byte_count
        or hashlib.sha256(review_bytes).hexdigest() != review_record_pin.sha256
    ):
        raise ValueError("external review record differs from its independent byte pin")
    child_python = runtime_pin.python_executable
    protocol = binding.protocol_document
    concurrency = protocol.get("concurrency")
    if type(concurrency) is not dict or type(concurrency.get("max_workers")) is not int:
        raise ValueError("frozen H protocol has no exact worker count")
    command = _frozen_h_public_argv(
        child_python=child_python,
        attempt_root=supervisor_attempt_root,
        protocol_path=protocol_v3_path,
        review_path=review_record_path,
        max_workers=concurrency["max_workers"],
    )
    observer_identity = implementation_source_identity(observer_repository)
    observer_head, observer_clean, observer_status_sha256 = _git_readback(
        observer_repository
    )
    attempt_uuid = uuid4().hex
    authority.update(
        {
            "child_binding": {
                "scope": "FROZEN_H_PUBLIC_CLI_PLAN_ONLY",
                "source_root": str(child_source_root),
                "cwd": str(child_source_root),
                "python_executable": str(child_python),
                "runtime_pin": runtime_pin.authority_payload(),
                "environment_policy": runtime_pin.authority_payload()[
                    "probe_facts"
                ].get("environment_policy"),
                "argv": list(command),
                "attempt_uuid": attempt_uuid,
                "supervisor_attempt_root": str(supervisor_attempt_root),
                "output_root": str(supervisor_attempt_root / "attempt"),
                "protocol_v3": {
                    "path": str(protocol_v3_path),
                    "bytes": static_inputs.protocol_v3_pin.raw_bytes,
                    "sha256": static_inputs.protocol_v3_pin.sha256,
                },
                "protocol_v4": {
                    "bytes": static_inputs.protocol_v4_pin.raw_bytes,
                    "sha256": static_inputs.protocol_v4_pin.sha256,
                },
                "review_record": {
                    "path": str(review_record_path),
                    "bytes": review_record_pin.byte_count,
                    "sha256": review_record_pin.sha256,
                },
                "native_source": {
                    "head_sha": static_inputs.source_pin.head_sha,
                    "snapshot_sha256": static_inputs.source_pin.snapshot_sha256,
                },
            },
            "observer_source": {
                "root": str(observer_repository),
                "closure_sha256": observer_identity["closure_sha256"],
                "repository_head": observer_head,
                "working_tree_clean": observer_clean,
                "git_status_sha256": observer_status_sha256,
            },
        }
    )
    return FrozenHLaunchPlan(
        observer_repository=observer_repository,
        observer_source_sha256=str(observer_identity["closure_sha256"]),
        child_source_root=child_source_root,
        child_python=child_python,
        runtime_pin=runtime_pin,
        supervisor_attempt_root=supervisor_attempt_root,
        output_root=supervisor_attempt_root / "attempt",
        witness_root=witness_root,
        protocol_v3_path=protocol_v3_path,
        review_record_path=review_record_path,
        protocol_v3_pin=WitnessFilePin(
            static_inputs.protocol_v3_pin.raw_bytes,
            static_inputs.protocol_v3_pin.sha256,
        ),
        protocol_v4_pin=WitnessFilePin(
            static_inputs.protocol_v4_pin.raw_bytes,
            static_inputs.protocol_v4_pin.sha256,
        ),
        review_record_pin=review_record_pin,
        attempt_uuid=attempt_uuid,
        argv=command,
        input_authority_canonical=_canonical(authority),
        _plan_token=_FROZEN_H_PLAN_TOKEN,
    )


def _artificial_h_environment_policy(
    child_source_root: Path, operator_path: Path, environment: dict[str, str]
) -> dict[str, object]:
    if (
        type(environment) is not dict
        or set(environment) - set(_ENVIRONMENT_ALLOWLIST)
        or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
        or environment.get("PYTHONUTF8") != "1"
        or set(_IMPORT_RESOLUTION_ENVIRONMENT) & set(environment)
        or set(_NUMERICAL_THREAD_ENVIRONMENT) & set(environment)
    ):
        raise ValueError("artificial H operator environment violates its fixed policy")
    return {
        "schema": "issue810_artificial_h_environment_policy_v1",
        "supervisor_values_sha256": {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in sorted(environment.items())
        },
        "absent_import_resolution": list(_IMPORT_RESOLUTION_ENVIRONMENT),
        "absent_numerical_threads": list(_NUMERICAL_THREAD_ENVIRONMENT),
        "worker": {
            "inherits": "supervisor_allowlist_only",
            "isolated_python": True,
            "python_executable": str(
                child_source_root / ".venv" / "Scripts" / "python.exe"
            ),
            "cwd": str(child_source_root),
            "operator": str(operator_path),
            "pythonpath": "absent",
            "source_root_added_after_operator_validation": str(child_source_root),
        },
    }


def _artificial_h_plan_document(
    *,
    component_spec: dict[str, object],
    component_spec_sha256: str,
    child_source_root: Path,
    child_source_pin: NativeSourcePin,
    runtime_pin: FrozenHRuntimePin,
    operator_path: Path,
    operator_sha256: str,
    output_root: Path,
    attempt_uuid: str,
) -> dict[str, object]:
    return {
        "schema": _ARTIFICIAL_H_PLAN_SCHEMA,
        "scope": "ARTIFICIAL_H_COMPONENT_ONLY",
        "attempt_uuid": attempt_uuid,
        "child_source": {
            "root": str(child_source_root),
            "head_sha": child_source_pin.head_sha,
            "snapshot_sha256": child_source_pin.snapshot_sha256,
        },
        "runtime_pin": runtime_pin.authority_payload(),
        "operator": {"path": str(operator_path), "sha256": operator_sha256},
        "component_spec": component_spec,
        "component_spec_sha256": component_spec_sha256,
        "output_root": str(output_root),
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
    }


def _artificial_h_plan_authority(
    *,
    observer_repository: Path,
    observer_source_sha256: str,
    observer_head_sha: str,
    child_source_root: Path,
    child_source_pin: NativeSourcePin,
    runtime_pin: FrozenHRuntimePin,
    operator_path: Path,
    operator_sha256: str,
    component_spec_sha256: str,
    binding_digest: str,
    dataset_id: str,
    input_plan_path: Path,
    input_plan_sha256: str,
    output_root: Path,
    witness_root: Path,
    attempt_uuid: str,
    argv: tuple[str, ...],
    environment_policy: dict[str, object],
) -> dict[str, object]:
    return {
        "scope": "ARTIFICIAL_H_COMPONENT_ONLY",
        "observer_source": {
            "root": str(observer_repository),
            "head_sha": observer_head_sha,
            "closure_sha256": observer_source_sha256,
            "working_tree_clean": True,
        },
        "child_source": {
            "root": str(child_source_root),
            "head_sha": child_source_pin.head_sha,
            "snapshot_sha256": child_source_pin.snapshot_sha256,
        },
        "runtime_pin": runtime_pin.authority_payload(),
        "operator": {"path": str(operator_path), "sha256": operator_sha256},
        "component_spec_sha256": component_spec_sha256,
        "binding_digest": binding_digest,
        "dataset": {"dataset_id": dataset_id},
        "input_plan": {
            "path": str(input_plan_path),
            "sha256": input_plan_sha256,
        },
        "attempt_uuid": attempt_uuid,
        "argv": list(argv),
        "cwd": str(child_source_root),
        "output_root": str(output_root),
        "witness_root": str(witness_root),
        "environment_policy": environment_policy,
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
    }


def build_artificial_h_component_plan(
    *,
    observer_repository: Path,
    child_source_root: Path,
    source_pin: NativeSourcePin,
    runtime_pin: FrozenHRuntimePin,
    component_spec: dict[str, object],
    input_plan_path: Path,
    output_root: Path,
    witness_root: Path,
) -> ArtificialHComponentPlan:
    """Build an offline typed plan for the fixed artificial H component path."""
    if not _is_artificial_h_component_spec(component_spec):
        raise ValueError("artificial H component specification is not the fixed scope")
    if (
        type(source_pin) is not NativeSourcePin
        or type(runtime_pin) is not FrozenHRuntimePin
    ):
        raise ValueError(
            "artificial H component requires exact source and runtime pins"
        )
    observer_repository = _safe_target(observer_repository, must_exist=True)
    child_source_root = _safe_target(child_source_root, must_exist=True)
    if (
        observer_repository == child_source_root
        or runtime_pin.source_root != child_source_root
    ):
        raise ValueError(
            "artificial H source must be separately pinned from the observer"
        )
    child_python = runtime_pin.python_executable
    _assert_frozen_h_runtime_pin_current(runtime_pin, child_source_root)
    observed_runtime = capture_frozen_h_runtime_pin(child_source_root)
    if observed_runtime.authority_payload() != runtime_pin.authority_payload():
        raise ValueError("artificial H runtime changed before plan creation")
    source_readback = verify_native_source(child_source_root, source_pin)
    if (
        source_readback.get("head_sha") != source_pin.head_sha
        or source_readback.get("snapshot_sha256") != source_pin.snapshot_sha256
        or source_readback.get("scope") != "STATIC_SOURCE_BYTES_NOT_RUNTIME_ATTESTATION"
    ):
        raise ValueError("artificial H source does not match its static byte pin")

    operator_path = observer_repository / _ARTIFICIAL_H_OPERATOR_RELATIVE
    operator_path = operator_path.absolute()
    from tools.market_b4_oracle.outer_witness import _read_source_file

    _, operator_bytes = _read_source_file(operator_path)
    if not 0 < len(operator_bytes) <= 4 * 1024**2:
        raise ValueError("artificial H operator source is outside its bounded pin")
    operator_sha256 = hashlib.sha256(operator_bytes).hexdigest()

    verify_root = _safe_target(
        observer_repository / "output" / "verify", must_exist=True
    )
    input_plan_path = _safe_target(input_plan_path, must_exist=False)
    output_root = _safe_target(output_root, must_exist=False)
    witness_root = _safe_target(witness_root, must_exist=False)
    if (
        any(
            path == verify_root or not path.is_relative_to(verify_root)
            for path in (input_plan_path, output_root, witness_root)
        )
        or output_root == witness_root
        or output_root.is_relative_to(witness_root)
        or witness_root.is_relative_to(output_root)
    ):
        raise ValueError("artificial H plans and artifacts require fresh verify roots")
    if (
        input_plan_path == output_root
        or input_plan_path.is_relative_to(output_root)
        or output_root.is_relative_to(input_plan_path)
    ):
        raise ValueError("artificial H input plan must be separate from child output")
    observer_identity = implementation_source_identity(observer_repository)
    observer_head, observer_clean, observer_status_sha256 = _git_readback(
        observer_repository
    )
    if (
        observer_clean is not True
        or type(observer_identity.get("closure_sha256")) is not str
        or not _valid_sha256(observer_identity.get("closure_sha256"))
    ):
        raise ValueError("artificial H plan requires a clean pinned observer source")

    component_spec_canonical = _canonical(component_spec)
    component_spec_sha256 = hashlib.sha256(component_spec_canonical).hexdigest()
    from tools.market_b4_oracle.artificial_component import (
        artificial_component_binding,
        build_artificial_component_inputs,
    )

    q_inputs = build_artificial_component_inputs(component_spec)
    if q_inputs.spec_sha256 != component_spec_sha256:
        raise ValueError("Q artificial input identity differs from the fixed plan")
    nonrl_binding, _ = artificial_component_binding(q_inputs, "NONRL")
    cash_binding, _ = artificial_component_binding(q_inputs, "CASH")
    if nonrl_binding.protocol_canonical != cash_binding.protocol_canonical:
        raise ValueError("artificial pair roles do not share one protocol binding")
    binding_digest = hashlib.sha256(nonrl_binding.protocol_canonical).hexdigest()
    dataset_id = q_inputs.content.dataset_id
    attempt_uuid = uuid4().hex
    argv = _artificial_h_component_argv(
        child_python=child_python,
        operator_path=operator_path,
        input_plan_path=input_plan_path,
    )
    input_document = _artificial_h_plan_document(
        component_spec=component_spec,
        component_spec_sha256=component_spec_sha256,
        child_source_root=child_source_root,
        child_source_pin=source_pin,
        runtime_pin=runtime_pin,
        operator_path=operator_path,
        operator_sha256=operator_sha256,
        output_root=output_root,
        attempt_uuid=attempt_uuid,
    )
    input_plan_canonical = _canonical(input_document)
    environment_policy = _artificial_h_environment_policy(
        child_source_root, operator_path, _observer_environment()
    )
    input_authority = _artificial_h_plan_authority(
        observer_repository=observer_repository,
        observer_source_sha256=str(observer_identity["closure_sha256"]),
        observer_head_sha=observer_head,
        child_source_root=child_source_root,
        child_source_pin=source_pin,
        runtime_pin=runtime_pin,
        operator_path=operator_path,
        operator_sha256=operator_sha256,
        component_spec_sha256=component_spec_sha256,
        binding_digest=binding_digest,
        dataset_id=dataset_id,
        input_plan_path=input_plan_path,
        input_plan_sha256=hashlib.sha256(input_plan_canonical).hexdigest(),
        output_root=output_root,
        witness_root=witness_root,
        attempt_uuid=attempt_uuid,
        argv=argv,
        environment_policy=environment_policy,
    )
    return ArtificialHComponentPlan(
        observer_repository=observer_repository,
        observer_source_sha256=str(observer_identity["closure_sha256"]),
        observer_head_sha=observer_head,
        child_source_root=child_source_root,
        child_source_pin=source_pin,
        child_python=child_python,
        runtime_pin=runtime_pin,
        operator_path=operator_path,
        operator_sha256=operator_sha256,
        component_spec_canonical=component_spec_canonical,
        component_spec_sha256=component_spec_sha256,
        input_plan_path=input_plan_path,
        input_plan_canonical=input_plan_canonical,
        output_root=output_root,
        witness_root=witness_root,
        attempt_uuid=attempt_uuid,
        argv=argv,
        input_authority_canonical=_canonical(input_authority),
        _plan_token=_ARTIFICIAL_H_PLAN_TOKEN,
    )


def _artificial_h_plan_authority_payload(
    plan: ArtificialHComponentPlan,
) -> dict[str, object]:
    if (
        type(plan) is not ArtificialHComponentPlan
        or plan._plan_token is not _ARTIFICIAL_H_PLAN_TOKEN
    ):
        raise ValueError("artificial H execution requires an exact typed plan")
    authority = _parse_canonical_object(plan.input_authority_canonical)
    if _canonical(authority) != plan.input_authority_canonical:
        raise ValueError("artificial H plan authority is not canonical")
    return authority


def _assert_artificial_h_plan_current(
    plan: ArtificialHComponentPlan, *, input_plan_written: bool = False
) -> None:
    authority = _artificial_h_plan_authority_payload(plan)
    observer = authority.get("observer_source")
    if type(observer) is not dict:
        raise ValueError("artificial H plan lacks its observer source binding")
    observer_identity = implementation_source_identity(plan.observer_repository)
    observer_head, observer_clean, _ = _git_readback(plan.observer_repository)
    if (
        observer.get("root") != str(plan.observer_repository)
        or observer.get("closure_sha256") != observer_identity.get("closure_sha256")
        or observer.get("head_sha") != observer_head
        or observer_head != plan.observer_head_sha
        or observer_clean is not True
        or observer.get("working_tree_clean") is not True
    ):
        raise ValueError("artificial H observer source changed after plan creation")
    source_readback = verify_native_source(
        plan.child_source_root, plan.child_source_pin
    )
    if (
        source_readback.get("head_sha") != plan.child_source_pin.head_sha
        or source_readback.get("snapshot_sha256")
        != plan.child_source_pin.snapshot_sha256
        or source_readback.get("scope") != "STATIC_SOURCE_BYTES_NOT_RUNTIME_ATTESTATION"
    ):
        raise ValueError("artificial H source changed after plan creation")
    _assert_frozen_h_runtime_pin_current(plan.runtime_pin, plan.child_source_root)
    current_runtime = capture_frozen_h_runtime_pin(plan.child_source_root)
    if current_runtime.authority_payload() != plan.runtime_pin.authority_payload():
        raise ValueError("artificial H runtime changed after plan creation")
    from tools.market_b4_oracle.outer_witness import _read_source_file

    _, operator_bytes = _read_source_file(plan.operator_path)
    if hashlib.sha256(operator_bytes).hexdigest() != plan.operator_sha256:
        raise ValueError("artificial H operator changed after plan creation")
    if authority.get("environment_policy") != _artificial_h_environment_policy(
        plan.child_source_root, plan.operator_path, _observer_environment()
    ):
        raise ValueError("artificial H launch environment changed after plan creation")
    if input_plan_written:
        _, raw = _read_source_file(plan.input_plan_path)
        if raw != plan.input_plan_canonical:
            raise ValueError("artificial H input plan changed after it was written")
    elif plan.input_plan_path.exists():
        raise FileExistsError("artificial H input plan path must be fresh")


def _artificial_h_start_record(plan: ArtificialHComponentPlan) -> dict[str, object]:
    authority = _artificial_h_plan_authority_payload(plan)
    child = authority.get("child_source")
    if type(child) is not dict:
        raise ValueError("artificial H plan lacks a child source binding")
    return {
        "schema": "issue810_root_owned_attempt_start_v1",
        "attempt_uuid": plan.attempt_uuid,
        "argv": list(plan.argv),
        "cwd": str(plan.child_source_root),
        "output_root": str(plan.output_root),
        "input_authority": authority,
        "child_environment_policy": authority.get("environment_policy"),
    }


def _assert_artificial_h_start_matches_plan(
    plan: ArtificialHComponentPlan, start: dict[str, object]
) -> None:
    expected = _artificial_h_start_record(plan)
    if (
        type(start) is not dict
        or start.get("schema") != expected["schema"]
        or start.get("attempt_uuid") != expected["attempt_uuid"]
        or start.get("argv") != expected["argv"]
        or start.get("cwd") != expected["cwd"]
        or start.get("output_root") != expected["output_root"]
        or start.get("input_authority") != expected["input_authority"]
        or start.get("child_environment_policy") != expected["child_environment_policy"]
    ):
        raise ValueError("artificial H start witness differs from its typed plan")


def _assert_artificial_h_operator_runtime_report(
    plan: ArtificialHComponentPlan, output_index: NativeOutputIndex
) -> None:
    try:
        raw = output_index.read_pinned_bytes("operator-runtime.json", maximum=16 * 1024)
        report = _parse_canonical_object(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "artificial H operator runtime report is absent or malformed"
        ) from error
    runtime_pin = plan.runtime_pin.authority_payload()
    runtime = runtime_pin.get("runtime_environment")
    packages = runtime.get("packages") if type(runtime) is dict else None
    h_site = plan.child_source_root / ".venv" / "Lib" / "site-packages"
    try:
        numpy_origin = Path(str(report.get("numpy_origin"))).absolute()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "artificial H operator imported an invalid NumPy origin"
        ) from error
    expected = {
        "schema": "issue810_artificial_h_operator_runtime_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "attempt_uuid": plan.attempt_uuid,
        "input_plan_sha256": hashlib.sha256(plan.input_plan_canonical).hexdigest(),
        "operator_sha256": plan.operator_sha256,
        "python_executable": str(plan.child_python),
        "prefix": str(plan.child_source_root / ".venv"),
        "source_root": str(plan.child_source_root),
        "runtime_environment": runtime,
        "runtime_environment_sha256": runtime_pin.get("runtime_environment_sha256"),
        "numpy_version": packages.get("numpy") if type(packages) is dict else None,
        "trade_rl_origin": str(plan.child_source_root / "trade_rl" / "__init__.py"),
        "tier_d_nonrl_origin": str(
            plan.child_source_root / "tools" / "tier_d_nonrl" / "__init__.py"
        ),
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
    }
    if (
        set(report)
        != {
            "schema",
            "scope",
            "attempt_uuid",
            "input_plan_sha256",
            "operator_sha256",
            "python_executable",
            "prefix",
            "source_root",
            "runtime_environment",
            "runtime_environment_sha256",
            "numpy_version",
            "numpy_origin",
            "trade_rl_origin",
            "tier_d_nonrl_origin",
            "outer_attempt_authenticated",
            "full_b4_authorized",
        }
        or any(report.get(key) != value for key, value in expected.items())
        or not numpy_origin.is_relative_to(h_site)
    ):
        raise ValueError("artificial H runtime report differs from its typed plan")


def capture_frozen_h_runtime_pin(child_source_root: Path) -> FrozenHRuntimePin:
    """Probe the locked H venv without starting its supervisor or market."""
    child_source_root = _safe_target(child_source_root, must_exist=True)
    child_python = child_source_root / ".venv" / "Scripts" / "python.exe"
    _safe_target(child_python.parent, must_exist=True)
    from tools.market_b4_oracle.outer_witness import (
        _read_source_file,
        _same_file_stamp,
    )

    executable_before, executable_bytes = _read_source_file(child_python)
    if len(executable_bytes) > 4 * 1024**2:
        raise ValueError("frozen H Python executable exceeds its bounded pin")
    project_before: dict[str, tuple[os.stat_result, bytes]] = {}
    for name in ("pyproject.toml", "uv.lock"):
        project_before[name] = _read_source_file(child_source_root / name)
    project_hashes = {
        name: hashlib.sha256(raw).hexdigest()
        for name, (_, raw) in project_before.items()
    }

    project = toml_loads(project_before["pyproject.toml"][1].decode("utf-8"))
    project_document = project.get("project")
    dependencies = (
        project_document.get("dependencies") if type(project_document) is dict else None
    )
    optional = (
        project_document.get("optional-dependencies")
        if type(project_document) is dict
        else None
    )
    forecast_gbm = optional.get("forecast-gbm") if type(optional) is dict else None
    if (
        type(project_document) is not dict
        or project_document.get("requires-python") != ">=3.12,<3.13"
        or type(dependencies) is not list
        or "numpy>=1.26.0,<2.0.0" not in dependencies
        or "gymnasium==0.29.1" not in dependencies
        or type(forecast_gbm) is not list
        or "lightgbm[scikit-learn]==4.7.0" not in forecast_gbm
    ):
        raise ValueError("frozen H project runtime requirements changed")

    lock = toml_loads(project_before["uv.lock"][1].decode("utf-8"))
    locked_packages = lock.get("package")
    if type(locked_packages) is not list:
        raise ValueError("frozen H uv lock has no package roster")
    expected_locked: dict[str, str] = {}
    for name in ("numpy", "gymnasium", "lightgbm", "scikit-learn", "scipy"):
        matches = [
            row
            for row in locked_packages
            if type(row) is dict and row.get("name") == name
        ]
        if len(matches) != 1 or type(matches[0].get("version")) is not str:
            raise ValueError("frozen H runtime lock is not a single exact package set")
        expected_locked[name] = matches[0]["version"]

    probe_environment = _observer_environment()
    environment_policy = _frozen_h_environment_policy(
        child_source_root, probe_environment
    )
    completed = subprocess.run(
        [str(child_python), "-B", "-c", _FROZEN_H_RUNTIME_PROBE],
        cwd=child_source_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=probe_environment,
        close_fds=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
        timeout=30,
    )
    if completed.returncode != 0 or len(completed.stdout) > 16 * 1024:
        raise ValueError("frozen H runtime probe failed or exceeded its byte bound")
    try:
        probe = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen H runtime probe returned malformed JSON") from error
    runtime_manifest = probe.get("runtime_environment") if type(probe) is dict else None
    runtime_packages = (
        runtime_manifest.get("packages") if type(runtime_manifest) is dict else None
    )
    support_packages = probe.get("support_packages") if type(probe) is dict else None
    if (
        type(probe) is not dict
        or set(probe)
        != {
            "schema",
            "source_root",
            "python_executable",
            "module_root",
            "module_resolution",
            "runtime_environment",
            "support_packages",
        }
        or probe.get("schema") != "issue810_frozen_h_runtime_probe_v1"
        or probe.get("source_root") != str(child_source_root)
        or probe.get("python_executable") != str(child_python)
        or probe.get("module_root") != str(child_source_root)
        or probe.get("module_resolution")
        != {
            "working_directory": str(child_source_root),
            "sys_path0": str(child_source_root),
            "trade_rl_origin": str(child_source_root / "trade_rl" / "__init__.py"),
            "tier_d_nonrl_main": str(
                child_source_root / "tools" / "tier_d_nonrl" / "__main__.py"
            ),
        }
        or type(runtime_manifest) is not dict
        or type(runtime_packages) is not dict
        or type(support_packages) is not dict
        or set(support_packages) != {"scikit-learn", "scipy"}
    ):
        raise ValueError("frozen H runtime probe did not resolve the exact source tree")
    for name in ("numpy", "gymnasium", "lightgbm", "scikit-learn", "scipy"):
        observed = (
            runtime_packages.get(name)
            if name in runtime_packages
            else support_packages.get(name)
        )
        if observed != expected_locked[name]:
            raise ValueError("frozen H runtime differs from its exact uv lock")
    runtime_python = runtime_manifest.get("python")
    if (
        runtime_manifest.get("schema_version") != "candidate_run_runtime_v1"
        or type(runtime_python) is not dict
        or runtime_python.get("implementation") != "CPython"
        or not str(runtime_python.get("version", "")).startswith("3.12.")
        or type(runtime_packages.get("trade-rl")) is not str
    ):
        raise ValueError(
            "frozen H runtime does not satisfy its declared Python profile"
        )

    runtime_bytes = _canonical(runtime_manifest)
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    executable_sha256 = hashlib.sha256(executable_bytes).hexdigest()
    facts = {
        "schema": "issue810_frozen_h_runtime_probe_v1",
        "source_root": str(child_source_root),
        "python_executable": str(child_python),
        "python_executable_sha256": executable_sha256,
        "module_root": probe["module_root"],
        "module_resolution": probe["module_resolution"],
        "environment_policy": environment_policy,
        "runtime_environment_sha256": runtime_sha256,
        "project_inputs_sha256": project_hashes,
        "locked_support_packages": {
            name: support_packages.get(name) for name in ("scikit-learn", "scipy")
        },
    }
    for name in ("pyproject.toml", "uv.lock"):
        current_stamp, current_bytes = _read_source_file(child_source_root / name)
        if (
            not _same_file_stamp(current_stamp, project_before[name][0])
            or current_bytes != project_before[name][1]
        ):
            raise ValueError("frozen H runtime project files changed during probing")
    executable_after, executable_after_bytes = _read_source_file(child_python)
    if (
        not _same_file_stamp(executable_after, executable_before)
        or executable_after_bytes != executable_bytes
    ):
        raise ValueError("frozen H Python executable changed during probing")
    facts_bytes = _canonical(facts)
    return _make_frozen_h_runtime_pin(
        source_root=child_source_root,
        python_executable=child_python,
        python_executable_sha256=executable_sha256,
        runtime_manifest_canonical=runtime_bytes,
        runtime_environment_sha256=runtime_sha256,
        probe_facts_canonical=facts_bytes,
        probe_facts_sha256=hashlib.sha256(facts_bytes).hexdigest(),
    )


def _assert_frozen_h_runtime_pin_current(
    runtime_pin: FrozenHRuntimePin, source_root: Path
) -> None:
    if (
        type(runtime_pin) is not FrozenHRuntimePin
        or runtime_pin.source_root != source_root
        or runtime_pin.python_executable
        != source_root / ".venv" / "Scripts" / "python.exe"
    ):
        raise ValueError("frozen H runtime pin differs from its source checkout")
    from tools.market_b4_oracle.outer_witness import _read_source_file

    _, executable_bytes = _read_source_file(runtime_pin.python_executable)
    facts = json.loads(runtime_pin.probe_facts_canonical)
    expected_module_resolution = {
        "working_directory": str(source_root),
        "sys_path0": str(source_root),
        "trade_rl_origin": str(source_root / "trade_rl" / "__init__.py"),
        "tier_d_nonrl_main": str(
            source_root / "tools" / "tier_d_nonrl" / "__main__.py"
        ),
    }
    if facts.get("module_resolution") != expected_module_resolution:
        raise ValueError("frozen H import resolution differs from its source root")
    if facts.get("environment_policy") != _frozen_h_environment_policy(
        source_root, _observer_environment()
    ):
        raise ValueError("frozen H launch environment changed after runtime probing")
    for name, expected in facts["project_inputs_sha256"].items():
        _, content_bytes = _read_source_file(source_root / name)
        if hashlib.sha256(content_bytes).hexdigest() != expected:
            raise ValueError("frozen H runtime project inputs changed after probing")
    if (
        hashlib.sha256(executable_bytes).hexdigest()
        != runtime_pin.python_executable_sha256
    ):
        raise ValueError("frozen H Python executable changed after probing")


def _git_readback(repository: Path) -> tuple[str, bool, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=15,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=15,
    ).stdout.encode("utf-8")
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise ValueError("outer observer could not read back an exact Git HEAD")
    return head, not status, hashlib.sha256(status).hexdigest()


def _safe_target(path: Path, *, must_exist: bool) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("outer attempt path must be an absolute non-aliased path")
    target = path.absolute()
    components = target.parts
    current = Path(target.anchor)
    for index, component in enumerate(components[1:], start=1):
        current = current / component
        final_component = index == len(components) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if final_component:
                if must_exist:
                    raise ValueError("outer attempt directory does not exist") from None
                break
            raise ValueError("outer attempt parent directory does not exist") from None
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("outer attempt path contains a link or reparse point")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("outer attempt path component is not a directory")
        if final_component and not must_exist:
            raise FileExistsError("outer attempt paths are fresh and never overwritten")
    return target


def _fresh_h_witness_root(observer_repository: Path, witness_root: Path) -> Path:
    """Keep H launch records under the observer's ignored verification tree."""
    verify_root = _safe_target(
        observer_repository / "output" / "verify", must_exist=True
    )
    target = _safe_target(witness_root, must_exist=False)
    if target == verify_root or not target.is_relative_to(verify_root):
        raise ValueError(
            "frozen H witness root must stay in the ignored verification scope"
        )
    return target


def _write_exclusive_record(path: Path, value: dict[str, object]) -> WitnessFilePin:
    raw = _canonical(value)
    if not 0 < len(raw) <= 1024 * 1024:
        raise ValueError("outer witness record exceeds its bounded size")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise ValueError(
            "outer witness record could not be written exclusively"
        ) from error
    return WitnessFilePin(len(raw), hashlib.sha256(raw).hexdigest())


def _frozen_input_authority(
    *,
    content: Any,
    policy: Any,
    binding: Any,
    static_inputs: Any,
) -> dict[str, object]:
    protocol_digest = hashlib.sha256(binding.protocol_canonical).hexdigest()
    dataset: dict[str, object] = {"dataset_id": content.dataset_id}
    if binding.comparison_only:
        if (
            binding.protocol_document.get("scope") != "ARTIFICIAL_COMPARISON_ONLY"
            or static_inputs is not None
        ):
            raise ValueError(
                "synthetic root observation requires artificial-only inputs"
            )
        return {
            "scope": "ARTIFICIAL_COMPARISON_ONLY",
            "binding_digest": protocol_digest,
            "dataset": dataset,
            "protocols": None,
            "native_source": None,
        }

    from tools.market_b4_oracle.offline_replay_consumer import (
        _readback_static_inputs,
    )

    if static_inputs is None:
        raise ValueError("native root observation requires H/v3 and V4 source pins")
    pair, _, _ = _readback_static_inputs(
        pins=static_inputs, binding=binding, content=content, policy=policy
    )
    return {
        "scope": "NATIVE_H3_V4_STATIC_PINS",
        "binding_digest": protocol_digest,
        "dataset": {
            "dataset_id": static_inputs.dataset_pin.dataset_id,
            "manifest_bytes": static_inputs.dataset_pin.manifest_bytes,
            "manifest_sha256": static_inputs.dataset_pin.manifest_sha256,
            "arrays_bytes": static_inputs.dataset_pin.arrays_bytes,
            "arrays_sha256": static_inputs.dataset_pin.arrays_sha256,
        },
        "protocols": {
            "h3": {
                "bytes": pair.native_pin.raw_bytes,
                "sha256": pair.native_pin.sha256,
                "source_head": pair.native_pin.source_head,
            },
            "v4": {
                "bytes": pair.comparison_pin.raw_bytes,
                "sha256": pair.comparison_pin.sha256,
                "source_head": pair.comparison_pin.source_head,
            },
        },
        "native_source": {
            "head_sha": static_inputs.source_pin.head_sha,
            "snapshot_sha256": static_inputs.source_pin.snapshot_sha256,
        },
    }


def _require_artificial_launch(binding: Any, static_inputs: Any) -> None:
    if getattr(binding, "comparison_only", None) is True:
        if static_inputs is not None:
            raise ValueError("artificial launch cannot claim native static inputs")
        return
    raise ValueError(
        "native execution requires a frozen H source-root and exact native CLI "
        "binding; arbitrary Python commands are comparison-only"
    )


def _observer_environment() -> dict[str, str]:
    """Keep only the bounded process environment used by the H public CLI."""
    environment = {
        name: os.environ[name]
        for name in _ENVIRONMENT_ALLOWLIST[:5]
        if name in os.environ
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"
    return environment


def _frozen_h_environment_policy(
    child_source_root: Path, environment: dict[str, str]
) -> dict[str, object]:
    if (
        not isinstance(child_source_root, Path)
        or not child_source_root.is_absolute()
        or type(environment) is not dict
        or set(environment) - set(_ENVIRONMENT_ALLOWLIST)
        or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
        or environment.get("PYTHONUTF8") != "1"
        or set(_IMPORT_RESOLUTION_ENVIRONMENT) & set(environment)
        or set(_NUMERICAL_THREAD_ENVIRONMENT) & set(environment)
    ):
        raise ValueError("frozen H process environment violates its fixed policy")
    return {
        "schema": "issue810_frozen_h_environment_policy_v1",
        "supervisor_values_sha256": {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in sorted(environment.items())
        },
        "forced_values": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
        },
        "absent_import_resolution": list(_IMPORT_RESOLUTION_ENVIRONMENT),
        "absent_numerical_threads": list(_NUMERICAL_THREAD_ENVIRONMENT),
        "worker_environment": {
            "inherits": "supervisor_allowlist_only",
            "python_executable": str(
                child_source_root / ".venv" / "Scripts" / "python.exe"
            ),
            "PYTHONPATH": str(child_source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }


def _validate_argv(
    argv: Sequence[str], *, expected_executable: Path | None = None
) -> tuple[str, ...]:
    if (
        isinstance(argv, (str, bytes))
        or not 1 <= len(argv) <= _MAX_ARGV_ITEMS
        or any(
            type(value) is not str
            or not value
            or "\x00" in value
            or len(value.encode("utf-8")) > _MAX_ARGV_STRING_BYTES
            for value in argv
        )
    ):
        raise ValueError("outer command must be a bounded explicit argv sequence")
    executable = Path(argv[0]).absolute()
    expected = (
        Path(sys.executable).absolute()
        if expected_executable is None
        else expected_executable.absolute()
    )
    if not executable.is_absolute() or executable != expected:
        raise ValueError("outer child must use its exact pinned Python runtime")
    return tuple(argv)


def _new_process_tree(pid: int) -> Any:
    if os.name != "nt":
        raise RuntimeError(
            "root-owned process-tree observation requires the frozen Windows observer"
        )
    from tools.tier_d_nonrl.tier_d_process_tree import ProcessTree

    return ProcessTree(pid)


def _stop_owned_tree(process: subprocess.Popen[Any], tree: Any) -> None:
    from tools.tier_d_nonrl.tier_d_process_tree import terminate_owned_tree

    terminate_owned_tree(process, tree)


def _observe_owned_tree(
    process: subprocess.Popen[Any], tree: Any, *, timeout_seconds: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        sample = tree.sample()
        ambiguous_members = getattr(sample, "ambiguous_members", None)
        if ambiguous_members is None:
            raise RuntimeError(
                "root-observed process tree sample omitted ambiguity classification"
            )
        if ambiguous_members:
            raise RuntimeError(
                "root-observed process tree contains ambiguous process identities"
            )
        if getattr(sample, "members", None) is None:
            raise RuntimeError("root-observed process tree sample omitted members")
        if process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("root-observed child exceeded its bounded runtime")
        time.sleep(_SAMPLE_INTERVAL_SECONDS)
    exit_code = process.wait()
    while True:
        sample = tree.sample()
        ambiguous_members = getattr(sample, "ambiguous_members", None)
        if ambiguous_members is None:
            raise RuntimeError(
                "root-observed process tree sample omitted ambiguity classification"
            )
        if ambiguous_members:
            raise RuntimeError(
                "root-observed process tree contains ambiguous process identities"
            )
        members = getattr(sample, "members", None)
        if members is None:
            raise RuntimeError("root-observed process tree sample omitted members")
        if not members:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("root-observed child descendants did not exit")
        time.sleep(_SAMPLE_INTERVAL_SECONDS)
    if tree.root_identity is None or len(tree.owned) > 64:
        raise ValueError("root observer did not capture the child PID birth identity")
    return {
        "pid": tree.root_identity[0],
        "birth_ticks": tree.root_identity[1],
        "exit_code": exit_code,
        "owned_tree_empty": True,
        "observed_process_identities": [
            list(identity) for identity in sorted(tree.owned)
        ],
    }


def _child_observation(process: Any, tree: Any) -> dict[str, object]:
    observation_error: str | None = None
    tree_observation_failed = False
    sample = getattr(tree, "last_sample", None)
    try:
        sample = tree.sample()
    except Exception as error:
        tree_observation_failed = True
        observation_error = f"{type(error).__name__}: {str(error)[:300]}"
    root_identity = getattr(tree, "root_identity", None)
    identities = getattr(tree, "owned", {})
    process_id = getattr(process, "pid", None)
    root_pid, birth_ticks = (
        root_identity
        if isinstance(root_identity, tuple) and len(root_identity) == 2
        else (process_id, None)
    )
    return_code = None
    if process is not None:
        try:
            return_code = process.poll()
        except Exception as error:
            observed_returncode = getattr(process, "returncode", None)
            if type(observed_returncode) is int:
                return_code = observed_returncode
            observation_error = observation_error or (
                f"exit poll {type(error).__name__}: {str(error)[:300]}"
            )
    members = getattr(sample, "members", None)
    ambiguous_members = getattr(sample, "ambiguous_members", None)
    if ambiguous_members:
        owned_tree_empty = False
        observation_error = observation_error or (
            "ambiguous process identities prevent an empty-tree observation"
        )
    elif tree_observation_failed or members is None or ambiguous_members is None:
        owned_tree_empty = None
        observation_error = observation_error or (
            "process-tree sample omitted member or ambiguity classification"
        )
    else:
        owned_tree_empty = not members
    return {
        "pid": root_pid,
        "birth_ticks": birth_ticks,
        "exit_code": return_code,
        "exit_observed": return_code is not None,
        "owned_tree_empty": owned_tree_empty,
        "observed_process_identities": [
            list(identity) for identity in sorted(identities)
        ],
        "observation_error": observation_error,
    }


def _cleanup_owned_tree(process: Any, tree: Any) -> dict[str, object]:
    errors: list[str] = []

    def poll_process() -> int | None:
        if process is None:
            return None
        try:
            return process.poll()
        except Exception as error:
            errors.append(f"root exit poll {type(error).__name__}: {str(error)[:300]}")
            value = getattr(process, "returncode", None)
            return value if type(value) is int else None

    if process is not None:
        if tree is not None:
            try:
                _stop_owned_tree(process, tree)
            except Exception as error:
                errors.append(f"tree stop {type(error).__name__}: {str(error)[:300]}")
        if poll_process() is None:
            try:
                process.terminate()
            except Exception as error:
                errors.append(
                    f"root terminate {type(error).__name__}: {str(error)[:300]}"
                )
        try:
            if poll_process() is None:
                process.wait(timeout=30)
            else:
                process.wait()
        except Exception as error:
            errors.append(f"root reap {type(error).__name__}: {str(error)[:300]}")
    deadline = time.monotonic() + 5.0
    tree_empty: bool | None = tree is None
    if tree is not None:
        tree_empty = False
        while time.monotonic() < deadline:
            try:
                sample = tree.sample()
            except Exception as error:
                errors.append(
                    f"final tree sample {type(error).__name__}: {str(error)[:300]}"
                )
                break
            ambiguous_members = getattr(sample, "ambiguous_members", None)
            if ambiguous_members is None:
                errors.append("final tree sample omitted ambiguity classification")
                break
            if ambiguous_members:
                errors.append("final tree sample contains ambiguous process identities")
                break
            members = getattr(sample, "members", None)
            if members is None:
                errors.append("final tree sample omitted process members")
                break
            if not members:
                tree_empty = True
                break
            time.sleep(_SAMPLE_INTERVAL_SECONDS)
    exit_observed = process is None or poll_process() is not None
    if not exit_observed:
        errors.append("root process exit was not observed")
    if tree_empty is not True:
        errors.append("owned process tree was not observed empty")
    return {
        "status": "REAPED_AND_TREE_EMPTY"
        if exit_observed and tree_empty is True and not errors
        else "CLEANUP_INCOMPLETE",
        "root_exit_observed": exit_observed,
        "owned_tree_empty": tree_empty,
        "errors": errors,
    }


def _write_failure_record(
    *,
    witness_root: Path,
    attempt_uuid: str,
    start_pin: WitnessFilePin,
    command: Sequence[str],
    output_root: Path,
    input_authority: dict[str, object],
    process: Any,
    tree: Any,
    original_error: BaseException,
    cleanup: dict[str, object],
) -> None:
    failure_value: dict[str, object] = {
        "schema": "issue810_root_owned_attempt_failure_v1",
        "attempt_uuid": attempt_uuid,
        "clock": {"wall_utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()},
        "argv_sha256": hashlib.sha256(_canonical(list(command))).hexdigest(),
        "output_root": str(output_root),
        "input_authority_sha256": hashlib.sha256(
            _canonical(input_authority)
        ).hexdigest(),
        "start_record": {"bytes": start_pin.byte_count, "sha256": start_pin.sha256},
        "failure": {
            "type": type(original_error).__name__,
            "message": str(original_error)[:500],
        },
        "cleanup": cleanup,
        "child": _child_observation(process, tree),
    }
    try:
        _write_exclusive_record(witness_root / "failure.json", failure_value)
    except Exception:
        # Preserve the original failure; a failed diagnostic write is secondary.
        return


def _run_root_observed_attempt(
    *,
    argv: Sequence[str],
    repository: Path,
    output_root: Path,
    witness_root: Path,
    input_authority: dict[str, object],
    timeout_seconds: float,
    attempt_uuid: str | None = None,
    child_cwd: Path | None = None,
    frozen_h_plan: FrozenHLaunchPlan | None = None,
    artificial_h_plan: ArtificialHComponentPlan | None = None,
    _completion_scope: Literal[
        "ARTIFICIAL_COMPARISON_ONLY", "NATIVE_H3_V4_ROOT_OBSERVED"
    ]
    | None = None,
) -> RootObservedExecution:
    if frozen_h_plan is not None and type(frozen_h_plan) is not FrozenHLaunchPlan:
        raise ValueError("fixed child binding requires an exact frozen H plan")
    if (
        artificial_h_plan is not None
        and type(artificial_h_plan) is not ArtificialHComponentPlan
    ):
        raise ValueError("artificial child binding requires an exact component plan")
    if frozen_h_plan is not None and artificial_h_plan is not None:
        raise ValueError(
            "one process attempt cannot combine native and artificial plans"
        )
    expected_executable = (
        frozen_h_plan.child_python
        if frozen_h_plan is not None
        else artificial_h_plan.child_python
        if artificial_h_plan is not None
        else None
    )
    command = _validate_argv(argv, expected_executable=expected_executable)
    repository = _safe_target(repository, must_exist=True)
    witness_root = _safe_target(witness_root, must_exist=False)
    if frozen_h_plan is None and artificial_h_plan is None:
        output_root = _safe_target(output_root, must_exist=False)
    elif frozen_h_plan is not None:
        supervisor_root = _safe_target(
            frozen_h_plan.supervisor_attempt_root, must_exist=False
        )
        if (
            supervisor_root.exists()
            or Path(output_root).absolute() != frozen_h_plan.output_root
            or frozen_h_plan.output_root != supervisor_root / "attempt"
        ):
            raise ValueError("frozen H output root is not the exact fresh plan target")
        output_root = frozen_h_plan.output_root
    else:
        assert artificial_h_plan is not None
        output_root = _safe_target(output_root, must_exist=False)
        if (
            output_root != artificial_h_plan.output_root
            or Path(witness_root) != artificial_h_plan.witness_root
            or Path(repository) != artificial_h_plan.observer_repository
        ):
            raise ValueError("artificial H output roots differ from the typed plan")
        output_root = artificial_h_plan.output_root
    if frozen_h_plan is not None:
        execution_cwd = _safe_target(child_cwd or repository, must_exist=True)
        if _completion_scope != "NATIVE_H3_V4_ROOT_OBSERVED":
            raise ValueError(
                "native H process requires the fixed native completion scope"
            )
        if (
            type(frozen_h_plan) is not FrozenHLaunchPlan
            or frozen_h_plan._plan_token is not _FROZEN_H_PLAN_TOKEN
            or frozen_h_plan.observer_repository != repository
            or frozen_h_plan.output_root != output_root
            or frozen_h_plan.witness_root != witness_root
            or frozen_h_plan.child_source_root != execution_cwd
            or frozen_h_plan.argv != command
            or frozen_h_plan.attempt_uuid != attempt_uuid
            or _canonical(_frozen_h_plan_authority(frozen_h_plan))
            != _canonical(input_authority)
        ):
            raise ValueError("frozen H process fields differ from the typed plan")
        _assert_frozen_h_runtime_pin_current(
            frozen_h_plan.runtime_pin, frozen_h_plan.child_source_root
        )
        _assert_frozen_h_start_matches_plan(
            frozen_h_plan,
            {
                "schema": "issue810_root_owned_attempt_start_v1",
                "attempt_uuid": attempt_uuid,
                "argv": list(command),
                "cwd": str(execution_cwd),
                "output_root": str(output_root),
                "input_authority": input_authority,
                "child_environment_policy": _frozen_h_child_environment_policy(
                    _frozen_h_plan_authority(frozen_h_plan)
                ),
            },
        )
    elif artificial_h_plan is not None:
        if attempt_uuid is not None or child_cwd is not None:
            raise ValueError("artificial H child fields are fixed by its typed plan")
        execution_cwd = _safe_target(
            artificial_h_plan.child_source_root, must_exist=True
        )
        attempt_uuid = artificial_h_plan.attempt_uuid
        if (
            artificial_h_plan._plan_token is not _ARTIFICIAL_H_PLAN_TOKEN
            or artificial_h_plan.observer_repository != repository
            or artificial_h_plan.output_root != output_root
            or artificial_h_plan.witness_root != witness_root
            or artificial_h_plan.child_source_root != execution_cwd
            or artificial_h_plan.argv != command
            or _canonical(_artificial_h_plan_authority_payload(artificial_h_plan))
            != _canonical(input_authority)
            or _completion_scope != "ARTIFICIAL_COMPARISON_ONLY"
        ):
            raise ValueError("artificial H process fields differ from the typed plan")
        _assert_artificial_h_plan_current(artificial_h_plan)
        _assert_artificial_h_start_matches_plan(
            artificial_h_plan, _artificial_h_start_record(artificial_h_plan)
        )
    else:
        execution_cwd = repository
        if attempt_uuid is not None or child_cwd is not None:
            raise ValueError("fixed child binding requires a frozen H launch plan")
        attempt_uuid = uuid4().hex
    if _completion_scope not in (
        None,
        "ARTIFICIAL_COMPARISON_ONLY",
        "NATIVE_H3_V4_ROOT_OBSERVED",
    ):
        raise ValueError("root observer completion scope is unsupported")
    if frozen_h_plan is None and _completion_scope == "NATIVE_H3_V4_ROOT_OBSERVED":
        raise ValueError("native completion scope requires its exact frozen H plan")
    if (
        artificial_h_plan is None
        and frozen_h_plan is None
        and _completion_scope
        not in (
            None,
            "ARTIFICIAL_COMPARISON_ONLY",
        )
    ):
        raise ValueError("unplanned process attempts remain artificial only")
    if (
        output_root == witness_root
        or output_root.is_relative_to(witness_root)
        or witness_root.is_relative_to(output_root)
        or timeout_seconds <= 0.0
        or timeout_seconds > 3600.0
    ):
        raise ValueError("outer attempt roots or bounded timeout are invalid")
    source_before = implementation_source_identity(repository)
    head_before, clean_before, status_before = _git_readback(repository)
    source_record = {
        **source_before,
        "repository_head": head_before,
        "working_tree_clean": clean_before,
        "git_status_sha256": status_before,
    }
    assert attempt_uuid is not None
    process_environment = _observer_environment()
    child_environment_policy = (
        _frozen_h_environment_policy(execution_cwd, process_environment)
        if frozen_h_plan is not None
        else _artificial_h_environment_policy(
            execution_cwd, artificial_h_plan.operator_path, process_environment
        )
        if artificial_h_plan is not None
        else {
            "schema": "issue810_root_environment_policy_v1",
            "values_sha256": {
                key: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for key, value in sorted(process_environment.items())
            },
        }
    )
    if frozen_h_plan is not None:
        planned_environment_policy = _frozen_h_child_environment_policy(
            _frozen_h_plan_authority(frozen_h_plan)
        )
        if child_environment_policy != planned_environment_policy:
            raise ValueError("frozen H launch environment differs from its plan")
    if artificial_h_plan is not None:
        planned_authority = _artificial_h_plan_authority_payload(artificial_h_plan)
        if child_environment_policy != planned_authority.get("environment_policy"):
            raise ValueError("artificial H process environment differs from its plan")
    witness_root.mkdir(parents=False, exist_ok=False)
    start_path = witness_root / "start.json"
    finish_path = witness_root / "finish.json"
    start_value: dict[str, object] = {
        "schema": "issue810_root_owned_attempt_start_v1",
        "attempt_uuid": attempt_uuid,
        "clock": {"wall_utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()},
        "argv": list(command),
        "cwd": str(execution_cwd),
        "output_root": str(output_root),
        "input_authority": input_authority,
        "child_environment_policy": child_environment_policy,
        "implementation_source": source_record,
        "environment_allowlist": [
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONUTF8",
        ],
    }
    if artificial_h_plan is not None:
        plan_document = _parse_canonical_object(artificial_h_plan.input_plan_canonical)
        plan_pin = _write_exclusive_record(
            artificial_h_plan.input_plan_path, plan_document
        )
        if (
            plan_pin.byte_count != len(artificial_h_plan.input_plan_canonical)
            or plan_pin.sha256
            != hashlib.sha256(artificial_h_plan.input_plan_canonical).hexdigest()
        ):
            raise ValueError("artificial H input plan write differs from its typed pin")
    start_pin = _write_exclusive_record(start_path, start_value)
    process = None
    tree = None
    child: dict[str, object] | None = None
    output_index: NativeOutputIndex | None = None
    manifest_pin: ArtifactPin | None = None
    finish_source: dict[str, object] | None = None
    original_error: BaseException | None = None
    original_traceback = None
    try:
        if not _root_owned_process_tree_supported():
            raise RuntimeError(
                "root-owned process-tree observation is unavailable on this OS"
            )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            list(command),
            cwd=execution_cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=process_environment,
            close_fds=True,
            creationflags=creation_flags,
        )
        tree = _new_process_tree(process.pid)
        child = _observe_owned_tree(process, tree, timeout_seconds=timeout_seconds)
        if child["exit_code"] != 0:
            raise RuntimeError("root-observed child exited unsuccessfully")
        manifest_pin = capture_output_manifest_pin(output_root)
        output_index = load_pinned_output_index(output_root, manifest_pin)
        if artificial_h_plan is not None:
            _assert_artificial_h_operator_runtime_report(
                artificial_h_plan, output_index
            )
        source_after = implementation_source_identity(repository)
        head_after, clean_after, status_after = _git_readback(repository)
        finish_source = {
            **source_after,
            "repository_head": head_after,
            "working_tree_clean": clean_after,
            "git_status_sha256": status_after,
        }
        if (
            head_before != head_after
            or clean_before != clean_after
            or status_before != status_after
            or source_before["closure_sha256"] != source_after["closure_sha256"]
        ):
            raise ValueError("root-observed source or Git state changed during attempt")
        if frozen_h_plan is not None:
            _assert_frozen_h_runtime_pin_current(
                frozen_h_plan.runtime_pin, frozen_h_plan.child_source_root
            )
            runtime_after = capture_frozen_h_runtime_pin(
                frozen_h_plan.child_source_root
            )
            if (
                runtime_after.authority_payload()
                != frozen_h_plan.runtime_pin.authority_payload()
            ):
                raise ValueError("frozen H runtime or import resolution changed")
        if artificial_h_plan is not None:
            _assert_artificial_h_plan_current(
                artificial_h_plan, input_plan_written=True
            )
    except BaseException as error:
        original_error = error
        original_traceback = error.__traceback__
    try:
        cleanup = _cleanup_owned_tree(process, tree)
    except BaseException as error:
        cleanup = {
            "status": "CLEANUP_EXCEPTION",
            "root_exit_observed": getattr(process, "returncode", None) is not None,
            "owned_tree_empty": None,
            "errors": [f"{type(error).__name__}: {str(error)[:300]}"],
        }
        if original_error is None:
            original_error = error
            original_traceback = error.__traceback__
    if original_error is None and cleanup["status"] != "REAPED_AND_TREE_EMPTY":
        original_error = RuntimeError("root-observed child cleanup was incomplete")
        original_traceback = original_error.__traceback__
    if original_error is not None:
        _write_failure_record(
            witness_root=witness_root,
            attempt_uuid=attempt_uuid,
            start_pin=start_pin,
            command=command,
            output_root=output_root,
            input_authority=input_authority,
            process=process,
            tree=tree,
            original_error=original_error,
            cleanup=cleanup,
        )
        raise original_error.with_traceback(original_traceback)
    assert child is not None and output_index is not None and manifest_pin is not None
    assert finish_source is not None and process is not None
    finish_value: dict[str, object] = {
        "schema": "issue810_root_owned_attempt_finish_v1",
        "attempt_uuid": attempt_uuid,
        "clock": {"wall_utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()},
        "argv_sha256": hashlib.sha256(_canonical(list(command))).hexdigest(),
        "output_root": str(output_root),
        "implementation_source": finish_source,
        "input_authority_sha256": hashlib.sha256(
            _canonical(input_authority)
        ).hexdigest(),
        "child": child,
        "output_manifest": {
            "bytes": manifest_pin.bytes,
            "sha256": manifest_pin.sha256,
        },
    }
    # Validate all semantic fields before publishing the success finish record.
    from tools.market_b4_oracle.outer_witness import _validate_pair

    _validate_pair(start_value, finish_value)
    finish_pin = _write_exclusive_record(finish_path, finish_value)
    root_witness = _make_root_observed_attempt(
        start_path,
        finish_path,
        repository_head=head_before,
        implementation_source_sha256=source_before["closure_sha256"],
    )
    start_readback, finish_readback = root_witness.readback(
        expected_start_pin=start_pin,
        expected_finish_pin=finish_pin,
    )
    if (
        finish_readback.get("input_authority_sha256")
        != hashlib.sha256(_canonical(start_readback["input_authority"])).hexdigest()
    ):
        raise ValueError(
            "root-observed source, inputs or Git state changed during attempt"
        )
    return RootObservedExecution(
        output_index,
        root_witness,
        start_pin,
        finish_pin,
        input_authority,
        scope=(
            "NATIVE_H3_V4_ROOT_OBSERVED"
            if frozen_h_plan is not None
            else "ARTIFICIAL_H_COMPONENT_PROCESS_OBSERVED_ONLY"
            if artificial_h_plan is not None
            else "ARTIFICIAL_COMPARISON_ONLY"
        ),
    )


def run_artificial_h_component_attempt(
    *, plan: ArtificialHComponentPlan, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
) -> RootObservedExecution:
    """Observe only the exact fixed artificial H child process and output tree."""
    if type(plan) is not ArtificialHComponentPlan:
        raise ValueError("artificial H execution requires its exact typed plan")
    authority = _artificial_h_plan_authority_payload(plan)
    _assert_artificial_h_start_matches_plan(plan, _artificial_h_start_record(plan))
    return _run_root_observed_attempt(
        argv=plan.argv,
        repository=plan.observer_repository,
        output_root=plan.output_root,
        witness_root=plan.witness_root,
        input_authority=authority,
        timeout_seconds=timeout_seconds,
        artificial_h_plan=plan,
        _completion_scope="ARTIFICIAL_COMPARISON_ONLY",
    )


def _artificial_receipt_plan_digest(
    *,
    output_index: NativeOutputIndex,
    receipt_name: str,
    inputs: Any,
    binding: Any,
    policy: Any,
) -> str:
    """Read the receipt's plan identity, then validate its semantics from Q."""
    from tools.market_b4_oracle.frozen_market_replay import FrozenMarketReplay
    from tools.market_b4_oracle.offline_replay_consumer import _metadata_arm

    reader = output_index.receipt_rows(receipt_name, phase="actor")
    for _ in reader:
        pass
    metadata = reader.metadata
    plan = metadata.get("plan")
    if type(plan) is not dict:
        raise ValueError("artificial receipt omits its terminal plan")
    digest = hashlib.sha256(_canonical(plan)).hexdigest()
    replay = FrozenMarketReplay(
        content=inputs.content,
        policy=policy,
        binding=binding,
        reserve_plan_digest=digest,
    )
    # The child plan digest is only an order-identity key. Its schema, source
    # clocks, costs, capital, risk and role must independently satisfy Q's
    # fixed spec before that digest is used to compare any receipt row.
    _metadata_arm(
        metadata,
        replay,
        digest,
        expected_forecast_sha256=hashlib.sha256(
            inputs.forecast_payload_canonical
        ).hexdigest(),
    )
    return digest


def run_and_consume_artificial_h_component_pair(
    *,
    plan: ArtificialHComponentPlan,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RootObservedArtificialHComponentPair:
    """Run H's fixed synthetic operator, then independently replay both roles."""
    if type(plan) is not ArtificialHComponentPlan:
        raise ValueError("artificial pair requires its exact typed plan")
    authority = _artificial_h_plan_authority_payload(plan)
    spec = plan.component_spec
    from tools.market_b4_oracle.artificial_component import (
        artificial_component_binding,
        build_artificial_component_inputs,
        verify_artificial_component_forecast,
    )

    inputs = build_artificial_component_inputs(spec)
    if (
        inputs.spec_sha256 != plan.component_spec_sha256
        or authority.get("component_spec_sha256") != inputs.spec_sha256
        or authority.get("scope") != "ARTIFICIAL_H_COMPONENT_ONLY"
        or authority.get("outer_attempt_authenticated") is not False
        or authority.get("full_b4_authorized") is not False
    ):
        raise ValueError("Q artificial source differs from the exact typed plan")

    execution = run_artificial_h_component_attempt(
        plan=plan, timeout_seconds=timeout_seconds
    )
    if (
        execution.scope != "ARTIFICIAL_H_COMPONENT_PROCESS_OBSERVED_ONLY"
        or execution.full_b4_authorized is not False
        or execution.input_authority != authority
        or execution.output_index._root.resolve() != plan.output_root.resolve()
    ):
        raise ValueError("artificial child process differs from its observed plan")
    start, finish = execution.root_witness.readback(
        expected_start_pin=execution.expected_start_pin,
        expected_finish_pin=execution.expected_finish_pin,
    )
    _assert_artificial_h_start_matches_plan(plan, start)
    child = finish.get("child")
    manifest = finish.get("output_manifest")
    if (
        type(child) is not dict
        or child.get("exit_code") != 0
        or child.get("owned_tree_empty") is not True
        or type(manifest) is not dict
        or manifest
        != {
            "bytes": execution.output_index.manifest_pin.bytes,
            "sha256": execution.output_index.manifest_pin.sha256,
        }
        or start.get("input_authority") != authority
    ):
        raise ValueError("artificial output lacks its exact root-owned process pins")

    expected_files = {
        "component-result.json",
        "forecasts/simple-return-stream.json",
        "operator-runtime.json",
        "receipts/cash.json.gz",
        "receipts/nonrl.json.gz",
    }
    if set(execution.output_index.files) != expected_files:
        raise ValueError("artificial H output does not match the exact pair roster")
    execution.output_index.recheck()
    _assert_artificial_h_operator_runtime_report(plan, execution.output_index)
    forecast_sha256 = verify_artificial_component_forecast(
        output_index=execution.output_index,
        inputs=inputs,
    )
    result = _parse_canonical_object(
        execution.output_index.read_pinned_bytes(
            "component-result.json", maximum=4 * 1024**2
        )
    )
    expected_result_fields = {
        "schema",
        "scope",
        "component_spec_sha256",
        "synthetic_dataset_id",
        "synthetic_protocol_id",
        "roles",
        "terminal_plan_digests",
        "outer_attempt_authenticated",
        "full_b4_authorized",
        "research_authorized",
        "forecast_stream",
    }
    roles = result.get("roles")
    plan_digests = result.get("terminal_plan_digests")
    forecast_stream = result.get("forecast_stream")
    if (
        set(result) != expected_result_fields
        or result.get("schema") != "issue810_artificial_h_component_result_v1"
        or result.get("scope") != "ARTIFICIAL_COMPARISON_ONLY"
        or result.get("component_spec_sha256") != inputs.spec_sha256
        or result.get("synthetic_dataset_id") != inputs.content.dataset_id
        or result.get("synthetic_protocol_id") != spec.get("protocol_id")
        or result.get("outer_attempt_authenticated") is not False
        or result.get("full_b4_authorized") is not False
        or result.get("research_authorized") is not False
        or type(roles) is not dict
        or set(roles) != {"nonrl", "cash"}
        or type(plan_digests) is not dict
        or set(plan_digests) != {"NONRL", "CASH"}
        or type(forecast_stream) is not dict
        or forecast_stream
        != {
            "path": "forecasts/simple-return-stream.json",
            "sha256": forecast_sha256,
            "dataset_id": inputs.content.dataset_id,
        }
    ):
        raise ValueError("artificial H summary differs from the fixed pair inputs")

    from tools.market_b4_oracle.offline_replay_consumer import (
        consume_root_observed_market_arm,
    )

    comparisons = []
    for role in ("NONRL", "CASH"):
        role_key = role.lower()
        role_result = roles.get(role_key)
        summary_plan_digest = plan_digests.get(role)
        if (
            type(role_result) is not dict
            or set(role_result)
            != {
                "actor_rows",
                "reserve_rows",
                "actor_fill_processing_indices",
                "reserve_fill_processing_indices",
                "terminal_flat",
                "active_orders",
                "terminal_plan_digest",
            }
            or type(summary_plan_digest) is not str
            or role_result.get("terminal_plan_digest") != summary_plan_digest
        ):
            raise ValueError("artificial role summary is malformed")
        binding, policy = artificial_component_binding(inputs, role)
        receipt_plan_digest = _artificial_receipt_plan_digest(
            output_index=execution.output_index,
            receipt_name=f"receipts/{role_key}.json.gz",
            inputs=inputs,
            binding=binding,
            policy=policy,
        )
        if summary_plan_digest != receipt_plan_digest:
            raise ValueError("artificial role summary plan is not a receipt crosscheck")
        comparison = consume_root_observed_market_arm(
            output_index=execution.output_index,
            receipt_name=f"receipts/{role_key}.json.gz",
            forecast_name="forecasts/simple-return-stream.json",
            content=inputs.content,
            policy=policy,
            binding=binding,
            role=role,
            reserve_plan_digest=receipt_plan_digest,
            artificial_inputs=inputs,
            root_witness=execution.root_witness,
            expected_start_pin=execution.expected_start_pin,
            expected_finish_pin=execution.expected_finish_pin,
        )
        if (
            comparison.status != "ARTIFICIAL_COMPARISON_COMPLETE"
            or comparison.account_id != "H24_RIDGE_A1_BTCUSDT"
            or comparison.scenario != "BASEL0"
            or comparison.role != role_key
            or comparison.actor_rows != 6527
            or comparison.reserve_rows != 72
            or comparison.plan_digest != receipt_plan_digest
            or comparison.forecast_sha256 != forecast_sha256
            or comparison.output_index_rechecked is not True
            or comparison.static_inputs_rechecked is not False
            or comparison.root_attempt_observed is not True
            or comparison.outer_attempt_authenticated is not False
            or comparison.full_b4_authorized is not False
            or role_result.get("actor_rows") != comparison.actor_rows
            or role_result.get("reserve_rows") != comparison.reserve_rows
            or role_result.get("actor_fill_processing_indices")
            != list(comparison.actor_fill_processing_indices)
            or role_result.get("reserve_fill_processing_indices")
            != list(comparison.reserve_fill_processing_indices)
            or role_result.get("terminal_flat") is not True
            or role_result.get("active_orders") != 0
        ):
            raise ValueError("Q replay differs from the artificial H role summary")
        comparisons.append(comparison)

    nonrl, cash = comparisons
    if (
        cash.actor_fill_processing_indices
        or cash.reserve_fill_processing_indices
        or not all(
            any(
                start < processing < stop
                for processing in nonrl.actor_fill_processing_indices
            )
            for start, stop in zip(
                (2903, 5087, 7295),
                (5087, 7295, 9430),
                strict=True,
            )
        )
        or 9430 not in nonrl.actor_fill_processing_indices
        or (not nonrl.actor_fill_processing_indices)
        or max(nonrl.actor_fill_processing_indices) != 9430
        or 9431 not in nonrl.reserve_fill_processing_indices
    ):
        raise ValueError(
            "Q replay omitted a per-vintage fill or the actor/reserve boundary fill"
        )
    execution.output_index.recheck()
    return RootObservedArtificialHComponentPair(
        execution,
        (nonrl, cash),
    )


def run_and_consume_market_arm(
    *,
    argv: Sequence[str],
    repository: Path,
    output_root: Path,
    witness_root: Path,
    content: Any,
    policy: Any,
    binding: Any,
    role: Literal["NONRL", "CASH"],
    reserve_plan_digest: str,
    static_inputs: Any = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RootObservedMarketArm:
    """Observe one artificial child and consume its output as comparison only."""
    _require_artificial_launch(binding, static_inputs)
    authority = _frozen_input_authority(
        content=content,
        policy=policy,
        binding=binding,
        static_inputs=static_inputs,
    )
    execution = _run_root_observed_attempt(
        argv=argv,
        repository=repository,
        output_root=output_root,
        witness_root=witness_root,
        input_authority=authority,
        timeout_seconds=timeout_seconds,
    )
    from tools.market_b4_oracle.offline_replay_consumer import (
        consume_root_observed_market_arm,
    )

    comparison = consume_root_observed_market_arm(
        output_index=execution.output_index,
        receipt_name="receipts/arm.json.gz",
        forecast_name="forecasts/arm.json",
        content=content,
        policy=policy,
        binding=binding,
        role=role,
        reserve_plan_digest=reserve_plan_digest,
        static_inputs=static_inputs,
        root_witness=execution.root_witness,
        expected_start_pin=execution.expected_start_pin,
        expected_finish_pin=execution.expected_finish_pin,
    )
    return RootObservedMarketArm(execution, comparison)


def _assert_frozen_h_matrix_plan(
    plan: FrozenHLaunchPlan,
    *,
    protocol_pair: Any,
    content: Any,
    static_inputs: Any,
) -> tuple[Any, ...]:
    """Check the exact 48-arm static authority before one H supervisor launch."""
    from tools.market_b4_oracle.offline_replay_consumer import (
        StaticInputReadback,
        derive_frozen_native_market_roster,
    )

    if (
        type(plan) is not FrozenHLaunchPlan
        or type(static_inputs) is not StaticInputReadback
        or plan._plan_token is not _FROZEN_H_PLAN_TOKEN
        or plan.child_source_root != static_inputs.source_root
    ):
        raise ValueError("frozen H matrix requires its exact typed plan and inputs")
    roster = derive_frozen_native_market_roster(
        protocol_pair=protocol_pair,
        content=content,
    )
    if (
        len(roster) != 48
        or any(arm.binding.comparison_only for arm in roster)
        or {arm.role for arm in roster} != {"NONRL", "CASH"}
        or len({arm.account_id for arm in roster}) != 8
    ):
        raise ValueError("frozen H matrix differs from the exact 48-arm roster")
    first = roster[0]
    expected_authority = _frozen_input_authority(
        content=content,
        policy=first.policy,
        binding=first.binding,
        static_inputs=static_inputs,
    )
    plan_authority = _frozen_h_plan_authority(plan)
    if any(
        plan_authority.get(key) != value for key, value in expected_authority.items()
    ):
        raise ValueError("frozen H matrix source authority differs from its plan")
    child = plan_authority.get("child_binding")
    if type(child) is not dict:
        raise ValueError("frozen H matrix child plan is missing")
    if (
        plan.protocol_v3_path != Path(static_inputs.protocol_v3_path).absolute()
        or plan.protocol_v3_pin.byte_count != static_inputs.protocol_v3_pin.raw_bytes
        or plan.protocol_v3_pin.sha256 != static_inputs.protocol_v3_pin.sha256
        or plan.protocol_v4_pin.byte_count != static_inputs.protocol_v4_pin.raw_bytes
        or plan.protocol_v4_pin.sha256 != static_inputs.protocol_v4_pin.sha256
        or child.get("native_source")
        != {
            "head_sha": static_inputs.source_pin.head_sha,
            "snapshot_sha256": static_inputs.source_pin.snapshot_sha256,
        }
        or child.get("dataset")
        != {
            "dataset_id": static_inputs.dataset_pin.dataset_id,
            "manifest_bytes": static_inputs.dataset_pin.manifest_bytes,
            "manifest_sha256": static_inputs.dataset_pin.manifest_sha256,
            "arrays_bytes": static_inputs.dataset_pin.arrays_bytes,
            "arrays_sha256": static_inputs.dataset_pin.arrays_sha256,
        }
    ):
        raise ValueError("frozen H matrix static pins differ from its launch plan")
    _assert_frozen_h_runtime_pin_current(plan.runtime_pin, plan.child_source_root)
    observed_runtime = capture_frozen_h_runtime_pin(plan.child_source_root)
    if observed_runtime.authority_payload() != plan.runtime_pin.authority_payload():
        raise ValueError("frozen H runtime changed after matrix plan creation")
    _assert_frozen_h_plan_observer_current(plan)
    return roster


def run_and_consume_frozen_h_market_matrix(
    *,
    plan: FrozenHLaunchPlan,
    protocol_pair: Any,
    content: Any,
    static_inputs: Any,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RootObservedHMarketMatrix:
    """Run one exact frozen-H CLI and compare the source-derived 48-arm matrix.

    This returns root-observed software evidence only. It does not authenticate a
    research attempt or authorize Full-B4.
    """
    roster = _assert_frozen_h_matrix_plan(
        plan,
        protocol_pair=protocol_pair,
        content=content,
        static_inputs=static_inputs,
    )
    if len(roster) != 48:
        raise ValueError("frozen H matrix plan lost its complete arm roster")
    if not _root_owned_process_tree_supported():
        raise RuntimeError("frozen H market supervision requires Windows")
    execution = _run_frozen_h_attempt(plan, timeout_seconds=timeout_seconds)
    if (
        execution.scope != "NATIVE_H3_V4_ROOT_OBSERVED"
        or execution.full_b4_authorized is not False
        or execution.input_authority != _frozen_h_plan_authority(plan)
        or execution.output_index._root.resolve() != plan.output_root.resolve()
    ):
        raise ValueError("frozen H execution differs from its root-observed plan")
    start, finish = execution.root_witness.readback(
        expected_start_pin=execution.expected_start_pin,
        expected_finish_pin=execution.expected_finish_pin,
    )
    _assert_frozen_h_start_matches_plan(plan, start)
    child = finish.get("child")
    manifest = finish.get("output_manifest")
    if (
        type(child) is not dict
        or child.get("exit_code") != 0
        or child.get("owned_tree_empty") is not True
        or type(manifest) is not dict
        or manifest
        != {
            "bytes": execution.output_index.manifest_pin.bytes,
            "sha256": execution.output_index.manifest_pin.sha256,
        }
    ):
        raise ValueError("frozen H output lacks an observed successful tree and pin")
    from tools.market_b4_oracle.offline_replay_consumer import (
        consume_root_observed_market_matrix,
    )

    comparison = consume_root_observed_market_matrix(
        output_index=execution.output_index,
        protocol_pair=protocol_pair,
        content=content,
        static_inputs=static_inputs,
        runtime_pin=plan.runtime_pin,
        root_witness=execution.root_witness,
        expected_start_pin=execution.expected_start_pin,
        expected_finish_pin=execution.expected_finish_pin,
        expected_input_authority_canonical=plan.input_authority_canonical,
    )
    if (
        comparison.status != "PINNED_INPUT_MATRIX_COMPARISON_COMPLETE"
        or len(comparison.arms) != 48
        or comparison.root_attempt_observed is not True
        or comparison.outer_attempt_authenticated is not False
        or comparison.full_b4_authorized is not False
        or any(
            arm.root_attempt_observed is not True
            or arm.outer_attempt_authenticated is not False
            or arm.full_b4_authorized is not False
            for arm in comparison.arms
        )
    ):
        raise ValueError("frozen H matrix comparison did not remain comparison-only")
    return RootObservedHMarketMatrix(execution, comparison)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an offline root-observed Full-B4 comparison plan."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--witness-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("synthetic",), default="synthetic")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args(argv)
    repository = Path(__file__).absolute().parents[1]
    output_root = options.output_root.absolute()
    witness_root = options.witness_root.absolute()
    if options.execute:
        parser.error(
            "CLI execution requires a typed frozen binding and exact receipt pins; "
            "use run_and_consume_market_arm from an offline caller"
        )
    plan = {
        "status": "OFFLINE_PLAN_ONLY",
        "process_started": False,
        "mode": options.mode,
        "repository": str(repository),
        "output_root": str(output_root),
        "witness_root": str(witness_root),
        "full_b4_authorized": False,
    }
    sys.stdout.write(json.dumps(plan, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
