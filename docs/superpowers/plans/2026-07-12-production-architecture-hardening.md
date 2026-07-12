# Production Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it mechanically impossible for an overridden, incompletely evaluated, incompletely risk-configured, code-mismatched, or unverified model bundle to become the model served as Production.

**Architecture:** Introduce one immutable `ReleaseEligibility` contract in the Control Plane, one validated release-risk loader, and fail-closed checks at candidate construction, bundle validation, deployment approval, runtime refresh, and post-activation verification. Preserve research execution through `--no-register`, but require exact bundle identity and running-code identity for every Production activation.

**Tech Stack:** Python 3.11+, dataclasses, JSON, argparse, FastAPI, pytest, GitHub Actions, existing `ServingBundle`, `Registry`, `ProductionPipeline`, and deployment evidence modules.

## Global Constraints

- Production remains **NO-GO** after this increment.
- `--force`, `--skip-p0`, `--skip-wf`, and `--skip-gate` are release-disqualifying.
- `--skip-pbt` is recorded but is not release-disqualifying by itself.
- Release candidates require a non-empty sealed holdout used by final Gate 2.
- Release candidates require explicit, finite, validated pre-trade risk limits.
- `TRADE_RL_RELEASE_GIT_SHA` must be exactly 40 hexadecimal characters in strict serving mode.
- Deployment succeeds only when `/ready` reports the approved version and bundle digest.
- No automatic rollback is added in this increment.
- Existing research commands remain usable with `--no-register`.
- No new third-party runtime dependency is introduced.

---

## File structure

Create:

- `mars_lite/pipeline/release_eligibility.py` — pure release classification and immutable metadata serialization.
- `mars_lite/pipeline/release_risk.py` — load and validate the release-only risk JSON document.
- `scripts/verify_served_identity.py` — poll `/ready` and compare the live identity with the approved bundle.
- `tests/test_release_eligibility.py` — pure eligibility tests.
- `tests/test_release_risk.py` — release-risk validation tests.
- `tests/test_verify_served_identity.py` — readiness polling and mismatch tests.

Modify:

- `mars_lite/pipeline/cli.py` — add `--risk-config` and release-intent validation.
- `mars_lite/pipeline/production_pipeline.py` — enforce sealed holdout, derive eligibility, and deny candidate construction for research-only runs.
- `mars_lite/serving/candidate.py` — require validated eligibility and risk policy.
- `mars_lite/serving/bundle.py` — validate release metadata and expose manifest Git SHA and digest.
- `mars_lite/server/deployment_gate.py` — reject ineligible bundles before activation.
- `mars_lite/serving/runtime.py` — enforce optional strict running-Git-SHA binding while preserving the previous healthy bundle.
- `mars_lite/server/signal_server.py` — include release SHA, active version, and bundle digest in readiness.
- `scripts/run_server.py` — require and pass the Production release SHA in strict mode.
- `.github/workflows/deploy.yml` — activate on persistent stage storage and verify the live served identity.
- `tests/test_production_pipeline.py` — pipeline-level release denial and holdout tests.
- `tests/test_serving_bundle.py` — malformed/ineligible metadata tests.
- `tests/test_serving_runtime.py` — matching and mismatching Git SHA hot-swap tests.
- `tests/test_deployment_activation.py` — workflow ordering and expected identity arguments.
- `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md`, `docs/PRODUCTION_READINESS.md`, `README.md` — normative behavior and continued NO-GO status.

---

### Task 1: Immutable release eligibility contract

**Files:**
- Create: `mars_lite/pipeline/release_eligibility.py`
- Create: `tests/test_release_eligibility.py`

**Interfaces:**
- Produces: `GateState`, `ReleaseEligibility`, and `derive_release_eligibility(...) -> ReleaseEligibility`.
- Consumers: candidate construction, bundle validation, deployment gate, and Production pipeline.

- [ ] **Step 1: Write the failing unit tests**

```python
from mars_lite.pipeline.release_eligibility import derive_release_eligibility


def test_normal_run_is_release_eligible() -> None:
    result = derive_release_eligibility(
        forced=False,
        skip_p0=False,
        skip_pbt=False,
        skip_wf=False,
        skip_gate=False,
        sealed_holdout_used=True,
        p0_passed=True,
        walk_forward_passed=True,
        gate2_passed=True,
        significance_passed=None,
    )
    assert result.eligible is True
    assert result.skipped_gates == ()
    assert result.required_gates["significance"] == "not_required"


def test_forced_run_is_not_release_eligible() -> None:
    result = derive_release_eligibility(
        forced=True,
        skip_p0=False,
        skip_pbt=False,
        skip_wf=False,
        skip_gate=False,
        sealed_holdout_used=True,
        p0_passed=True,
        walk_forward_passed=True,
        gate2_passed=True,
        significance_passed=True,
    )
    assert result.eligible is False
    assert result.forced is True


def test_skipped_pbt_is_recorded_but_not_disqualifying() -> None:
    result = derive_release_eligibility(
        forced=False,
        skip_p0=False,
        skip_pbt=True,
        skip_wf=False,
        skip_gate=False,
        sealed_holdout_used=True,
        p0_passed=True,
        walk_forward_passed=True,
        gate2_passed=True,
        significance_passed=True,
    )
    assert result.eligible is True
    assert result.optimization_steps_skipped == ("pbt",)


def test_missing_holdout_is_not_release_eligible() -> None:
    result = derive_release_eligibility(
        forced=False,
        skip_p0=False,
        skip_pbt=False,
        skip_wf=False,
        skip_gate=False,
        sealed_holdout_used=False,
        p0_passed=True,
        walk_forward_passed=True,
        gate2_passed=True,
        significance_passed=True,
    )
    assert result.eligible is False
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```bash
pytest tests/test_release_eligibility.py -v
```

Expected: collection fails with `ModuleNotFoundError: mars_lite.pipeline.release_eligibility`.

- [ ] **Step 3: Implement the pure eligibility model**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

GateState = Literal["passed", "failed", "skipped", "not_required"]


@dataclass(frozen=True)
class ReleaseEligibility:
    eligible: bool
    forced: bool
    skipped_gates: tuple[str, ...]
    optimization_steps_skipped: tuple[str, ...]
    sealed_holdout_used: bool
    required_gates: dict[str, GateState]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def derive_release_eligibility(
    *,
    forced: bool,
    skip_p0: bool,
    skip_pbt: bool,
    skip_wf: bool,
    skip_gate: bool,
    sealed_holdout_used: bool,
    p0_passed: bool,
    walk_forward_passed: bool,
    gate2_passed: bool,
    significance_passed: bool | None,
) -> ReleaseEligibility:
    skipped_gates = tuple(
        name
        for name, skipped in (
            ("p0", skip_p0),
            ("walk_forward", skip_wf),
            ("gate2", skip_gate),
        )
        if skipped
    )
    required_gates: dict[str, GateState] = {
        "p0": "skipped" if skip_p0 else "passed" if p0_passed else "failed",
        "walk_forward": (
            "skipped" if skip_wf else "passed" if walk_forward_passed else "failed"
        ),
        "gate2": "skipped" if skip_gate else "passed" if gate2_passed else "failed",
        "significance": (
            "not_required"
            if significance_passed is None
            else "passed" if significance_passed else "failed"
        ),
    }
    eligible = (
        not forced
        and not skipped_gates
        and sealed_holdout_used
        and all(state in {"passed", "not_required"} for state in required_gates.values())
    )
    return ReleaseEligibility(
        eligible=eligible,
        forced=forced,
        skipped_gates=skipped_gates,
        optimization_steps_skipped=("pbt",) if skip_pbt else (),
        sealed_holdout_used=sealed_holdout_used,
        required_gates=required_gates,
    )
```

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/test_release_eligibility.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add mars_lite/pipeline/release_eligibility.py tests/test_release_eligibility.py
git commit -m "feat: add immutable release eligibility contract"
```

---

### Task 2: Mandatory release-risk configuration

**Files:**
- Create: `mars_lite/pipeline/release_risk.py`
- Create: `tests/test_release_risk.py`
- Modify: `mars_lite/pipeline/cli.py`
- Modify: `mars_lite/serving/candidate.py`

**Interfaces:**
- Produces: `ReleaseRiskPolicy`, `load_release_risk_policy(path, symbols) -> ReleaseRiskPolicy`.
- Candidate builder consumes `risk_policy: ReleaseRiskPolicy` and writes `risk.json` from `risk_policy.to_dict()`.

- [ ] **Step 1: Write failing risk-policy tests**

```python
import json
from pathlib import Path

import pytest

from mars_lite.pipeline.release_risk import load_release_risk_policy


def _valid_policy() -> dict[str, object]:
    return {
        "max_leverage": 1.0,
        "max_single_weight": 0.20,
        "max_net_exposure": 0.60,
        "max_worst_case_notional": 100000.0,
        "min_order_notional": 10.0,
        "symbol_liquidity_caps": {"BTC": 50000.0, "ETH": 30000.0},
        "forbidden_symbols": [],
    }


def test_loads_complete_release_risk_policy(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(_valid_policy()), encoding="utf-8")
    policy = load_release_risk_policy(path, symbols=("BTC", "ETH"))
    assert policy.max_single_weight == 0.20
    assert policy.symbol_liquidity_caps["BTC"] == 50000.0


def test_rejects_missing_symbol_cap(tmp_path: Path) -> None:
    payload = _valid_policy()
    payload["symbol_liquidity_caps"] = {"BTC": 50000.0}
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing liquidity caps.*ETH"):
        load_release_risk_policy(path, symbols=("BTC", "ETH"))


def test_rejects_unbounded_single_weight(tmp_path: Path) -> None:
    payload = _valid_policy()
    payload["max_single_weight"] = 1.5
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="max_single_weight"):
        load_release_risk_policy(path, symbols=("BTC", "ETH"))
```

- [ ] **Step 2: Verify tests fail**

```bash
pytest tests/test_release_risk.py -v
```

Expected: module import failure.

- [ ] **Step 3: Implement strict parsing and semantic validation**

```python
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseRiskPolicy:
    max_leverage: float
    max_single_weight: float
    max_net_exposure: float
    max_worst_case_notional: float
    min_order_notional: float
    symbol_liquidity_caps: dict[str, float]
    forbidden_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _positive_finite(name: str, value: object) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def load_release_risk_policy(
    path: str | Path,
    *,
    symbols: tuple[str, ...],
) -> ReleaseRiskPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "max_leverage",
        "max_single_weight",
        "max_net_exposure",
        "max_worst_case_notional",
        "min_order_notional",
        "symbol_liquidity_caps",
        "forbidden_symbols",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"missing release risk fields: {', '.join(missing)}")

    max_leverage = _positive_finite("max_leverage", payload["max_leverage"])
    max_single_weight = _positive_finite(
        "max_single_weight", payload["max_single_weight"]
    )
    max_net_exposure = _positive_finite(
        "max_net_exposure", payload["max_net_exposure"]
    )
    if max_single_weight > 1.0:
        raise ValueError("max_single_weight must be <= 1.0")
    if max_net_exposure > max_leverage:
        raise ValueError("max_net_exposure must be <= max_leverage")

    raw_caps = payload["symbol_liquidity_caps"]
    if not isinstance(raw_caps, dict):
        raise ValueError("symbol_liquidity_caps must be an object")
    missing_caps = sorted(set(symbols) - set(raw_caps))
    if missing_caps:
        raise ValueError(f"missing liquidity caps for: {', '.join(missing_caps)}")
    caps = {symbol: _positive_finite(f"liquidity cap {symbol}", raw_caps[symbol]) for symbol in symbols}

    forbidden = payload["forbidden_symbols"]
    if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
        raise ValueError("forbidden_symbols must be a list of strings")

    return ReleaseRiskPolicy(
        max_leverage=max_leverage,
        max_single_weight=max_single_weight,
        max_net_exposure=max_net_exposure,
        max_worst_case_notional=_positive_finite(
            "max_worst_case_notional", payload["max_worst_case_notional"]
        ),
        min_order_notional=_positive_finite(
            "min_order_notional", payload["min_order_notional"]
        ),
        symbol_liquidity_caps=caps,
        forbidden_symbols=tuple(forbidden),
    )
```

- [ ] **Step 4: Add the release-only CLI input**

Add to the Production pipeline parser:

```python
parser.add_argument(
    "--risk-config",
    type=Path,
    default=None,
    help="Required validated JSON risk policy when a release candidate may be registered.",
)
```

At resolved-argument validation, enforce:

```python
release_intent = not args.no_register
if release_intent and args.risk_config is None:
    parser.error("--risk-config is required unless --no-register is supplied")
```

- [ ] **Step 5: Require validated objects in candidate construction**

Change candidate construction so it receives:

```python
risk_policy: ReleaseRiskPolicy
release_eligibility: ReleaseEligibility
```

Before writing any file:

```python
if not release_eligibility.eligible:
    raise ValueError("ineligible research run cannot create a release candidate")
```

Write the immutable files using:

```python
write_json(candidate_dir / "risk.json", risk_policy.to_dict())
metadata["release_eligibility"] = release_eligibility.to_dict()
```

Remove the Production-path fallback that serializes an empty `PreTradeRiskConfig()`.

- [ ] **Step 6: Run focused tests**

```bash
pytest tests/test_release_risk.py tests/test_serving_bundle.py -v
```

Expected: all tests pass after adapting existing candidate fixtures to supply explicit policy and eligibility.

- [ ] **Step 7: Commit**

```bash
git add mars_lite/pipeline/release_risk.py mars_lite/pipeline/cli.py mars_lite/serving/candidate.py tests/test_release_risk.py tests/test_serving_bundle.py
git commit -m "feat: require explicit release risk policy"
```

---

### Task 3: Fail-closed Production pipeline and sealed holdout

**Files:**
- Modify: `mars_lite/pipeline/production_pipeline.py`
- Modify: `tests/test_production_pipeline.py`

**Interfaces:**
- Consumes: `derive_release_eligibility` and `load_release_risk_policy`.
- Produces: candidate creation only for fully eligible release runs; research reports remain available under `--no-register`.

- [ ] **Step 1: Add failing pipeline tests**

Add tests using existing pipeline fixtures and monkeypatches:

```python
@pytest.mark.parametrize("flag", ["force", "skip_p0", "skip_wf", "skip_gate"])
def test_release_disqualifying_override_never_registers_candidate(
    tmp_path: Path,
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_pipeline_config(tmp_path, no_register=False, risk_config=write_valid_risk(tmp_path))
    setattr(config, flag, True)
    registered: list[Path] = []
    monkeypatch.setattr("mars_lite.pipeline.production_pipeline.Registry.register", lambda self, path: registered.append(path))

    result = run_pipeline_with_stubbed_training(config)

    assert result.candidate_path is None
    assert registered == []
    assert result.release_eligibility.eligible is False


def test_release_run_rejects_missing_sealed_holdout(tmp_path: Path) -> None:
    config = make_pipeline_config(tmp_path, no_register=False, risk_config=write_valid_risk(tmp_path))
    with pytest.raises(RuntimeError, match="sealed holdout"):
        run_pipeline_with_insufficient_holdout(config)


def test_research_run_may_continue_without_holdout_when_not_registering(tmp_path: Path) -> None:
    config = make_pipeline_config(tmp_path, no_register=True, risk_config=None)
    result = run_pipeline_with_insufficient_holdout(config)
    assert result.candidate_path is None
```

- [ ] **Step 2: Verify focused tests fail against current permissive behavior**

```bash
pytest tests/test_production_pipeline.py -k "override or sealed_holdout or research_run" -v
```

Expected: at least one test observes candidate creation/registration or no exception for missing holdout.

- [ ] **Step 3: Resolve release intent before expensive training**

Near pipeline entry:

```python
release_intent = not config.no_register
release_disqualifying_override = any(
    (config.force, config.skip_p0, config.skip_wf, config.skip_gate)
)
if release_intent and release_disqualifying_override:
    logger.warning(
        "release-disqualifying override detected; candidate creation and registration are disabled"
    )
    release_intent = False
```

Do not silently mutate the original configuration; keep the resolved intent in the pipeline result and evidence.

- [ ] **Step 4: Make holdout failure conditional on release intent**

After purge-aware split validation:

```python
sealed_holdout_used = holdout_features is not None and len(holdout_features) > 0
if release_intent and not sealed_holdout_used:
    raise RuntimeError(
        "release candidate requires a non-empty sealed holdout after purge"
    )
```

Ensure final Gate 2 receives only the sealed holdout and sets `sealed_holdout_used=True` only after that evaluation completes successfully.

- [ ] **Step 5: Derive eligibility once and pass it to candidate construction**

```python
eligibility = derive_release_eligibility(
    forced=config.force,
    skip_p0=config.skip_p0,
    skip_pbt=config.skip_pbt,
    skip_wf=config.skip_wf,
    skip_gate=config.skip_gate,
    sealed_holdout_used=sealed_holdout_used and gate2_used_holdout,
    p0_passed=p0_result.passed,
    walk_forward_passed=walk_forward_result.passed,
    gate2_passed=gate2_result.passed,
    significance_passed=(
        significance_result.passed if significance_result is not None else None
    ),
)
```

Candidate construction and Registry registration must be inside:

```python
if release_intent and eligibility.eligible:
    risk_policy = load_release_risk_policy(config.risk_config, symbols=tuple(symbols))
    candidate_path = build_candidate(..., risk_policy=risk_policy, release_eligibility=eligibility)
    registry.register(candidate_path)
else:
    candidate_path = None
```

- [ ] **Step 6: Run pipeline tests**

```bash
pytest tests/test_production_pipeline.py tests/test_release_eligibility.py tests/test_release_risk.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add mars_lite/pipeline/production_pipeline.py tests/test_production_pipeline.py
git commit -m "fix: fail closed for ineligible production runs"
```

---

### Task 4: Bundle and deployment-gate enforcement

**Files:**
- Modify: `mars_lite/serving/bundle.py`
- Modify: `mars_lite/server/deployment_gate.py`
- Modify: `tests/test_serving_bundle.py`
- Modify: `tests/test_deployment_activation.py`

**Interfaces:**
- Produces: `ServingBundle.release_eligibility`, `ServingBundle.bundle_digest`, and `validate_release_eligibility(metadata)`.
- Deployment gate consumes the validated bundle rather than trusting evidence text alone.

- [ ] **Step 1: Write failing bundle validation tests**

```python
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eligible", False),
        ("forced", True),
        ("skipped_gates", ["p0"]),
        ("sealed_holdout_used", False),
    ],
)
def test_bundle_rejects_ineligible_release_metadata(
    valid_bundle_dir: Path,
    field: str,
    value: object,
) -> None:
    metadata = read_json(valid_bundle_dir / "metadata.json")
    metadata["release_eligibility"][field] = value
    rewrite_json_and_manifest(valid_bundle_dir, "metadata.json", metadata)
    with pytest.raises(BundleValidationError, match="release eligibility"):
        ServingBundle.load(valid_bundle_dir)


def test_bundle_rejects_failed_required_gate(valid_bundle_dir: Path) -> None:
    metadata = read_json(valid_bundle_dir / "metadata.json")
    metadata["release_eligibility"]["required_gates"]["gate2"] = "failed"
    rewrite_json_and_manifest(valid_bundle_dir, "metadata.json", metadata)
    with pytest.raises(BundleValidationError, match="gate2"):
        ServingBundle.load(valid_bundle_dir)
```

- [ ] **Step 2: Verify tests fail**

```bash
pytest tests/test_serving_bundle.py -k "release_metadata or required_gate" -v
```

Expected: current bundle loader accepts the recomputed but semantically ineligible metadata.

- [ ] **Step 3: Add semantic validation after hash/schema validation**

```python
_ACCEPTED_GATE_STATES = {"passed", "not_required"}
_REQUIRED_GATE_NAMES = {"p0", "walk_forward", "gate2", "significance"}


def validate_release_eligibility(metadata: dict[str, object]) -> dict[str, object]:
    value = metadata.get("release_eligibility")
    if not isinstance(value, dict):
        raise BundleValidationError("missing release eligibility metadata")
    if value.get("eligible") is not True:
        raise BundleValidationError("release eligibility must be true")
    if value.get("forced") is not False:
        raise BundleValidationError("forced bundle is not release eligible")
    if value.get("skipped_gates") != []:
        raise BundleValidationError("release eligibility contains skipped gates")
    if value.get("sealed_holdout_used") is not True:
        raise BundleValidationError("sealed holdout was not used")
    gates = value.get("required_gates")
    if not isinstance(gates, dict) or set(gates) != _REQUIRED_GATE_NAMES:
        raise BundleValidationError("release eligibility has invalid required gates")
    for name, state in gates.items():
        if state not in _ACCEPTED_GATE_STATES:
            raise BundleValidationError(f"required gate {name} is {state}")
    return value
```

Expose the validated metadata and manifest values as read-only bundle properties.

- [ ] **Step 4: Require bundle eligibility in deployment approval**

After existing evidence validation:

```python
bundle = ServingBundle.load(bundle_path)
validate_release_eligibility(bundle.metadata)
if bundle.version != evidence.model_version:
    raise DeploymentGateError("evidence model version does not match bundle")
if bundle.bundle_digest != evidence.bundle_digest:
    raise DeploymentGateError("evidence digest does not match bundle")
```

- [ ] **Step 5: Run focused tests**

```bash
pytest tests/test_serving_bundle.py tests/test_deployment_activation.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add mars_lite/serving/bundle.py mars_lite/server/deployment_gate.py tests/test_serving_bundle.py tests/test_deployment_activation.py
git commit -m "feat: enforce release eligibility at bundle and deployment gates"
```

---

### Task 5: Bind Serving code identity to bundle identity

**Files:**
- Modify: `mars_lite/serving/runtime.py`
- Modify: `mars_lite/server/signal_server.py`
- Modify: `scripts/run_server.py`
- Modify: `tests/test_serving_runtime.py`

**Interfaces:**
- `ServingRuntime(..., release_git_sha: str | None = None, strict_release_binding: bool = False)`.
- `ServingRuntime.readiness() -> dict[str, object]` includes `active_version`, `bundle_digest`, and `release_git_sha`.

- [ ] **Step 1: Add failing runtime tests**

```python
def test_strict_runtime_rejects_bundle_from_different_git_sha(
    runtime_fixture: RuntimeFixture,
) -> None:
    runtime = runtime_fixture.runtime(
        release_git_sha="a" * 40,
        strict_release_binding=True,
    )
    runtime_fixture.activate_bundle(git_sha="b" * 40)

    runtime.refresh()

    assert runtime.active_version == runtime_fixture.previous_version
    readiness = runtime.readiness()
    assert readiness["status"] == "degraded"
    assert "git sha mismatch" in str(readiness["last_refresh_error"]).lower()


def test_strict_runtime_loads_matching_git_sha(runtime_fixture: RuntimeFixture) -> None:
    runtime = runtime_fixture.runtime(
        release_git_sha="a" * 40,
        strict_release_binding=True,
    )
    bundle = runtime_fixture.activate_bundle(git_sha="a" * 40)
    runtime.refresh()
    readiness = runtime.readiness()
    assert readiness["active_version"] == bundle.version
    assert readiness["bundle_digest"] == bundle.bundle_digest
    assert readiness["release_git_sha"] == "a" * 40
```

- [ ] **Step 2: Verify tests fail**

```bash
pytest tests/test_serving_runtime.py -k "git_sha or release_git_sha" -v
```

Expected: constructor or readiness does not support the new identity fields.

- [ ] **Step 3: Validate the configured running SHA**

```python
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def validate_release_git_sha(value: str) -> str:
    if _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError("release_git_sha must be a 40-character hexadecimal SHA")
    return value.lower()
```

Store it on the runtime only after validation.

- [ ] **Step 4: Enforce binding during refresh before replacing the cached bundle**

```python
candidate = ServingBundle.load(active_path)
if self.strict_release_binding:
    assert self.release_git_sha is not None
    if candidate.git_sha.lower() != self.release_git_sha:
        raise BundleValidationError(
            f"bundle git sha mismatch: bundle={candidate.git_sha} running={self.release_git_sha}"
        )
```

Keep the existing exception boundary that preserves the previous healthy bundle and records the refresh error.

- [ ] **Step 5: Extend readiness without exposing secrets**

Return:

```python
{
    "status": status,
    "active_version": self.active_version,
    "bundle_digest": self.active_bundle.bundle_digest if self.active_bundle else None,
    "release_git_sha": self.release_git_sha,
    "last_refresh_error": self.last_refresh_error,
}
```

The `/ready` route must return this object and retain its existing HTTP readiness semantics.

- [ ] **Step 6: Make strict binding the Production entrypoint default**

In `scripts/run_server.py`:

```python
release_git_sha = os.environ.get("TRADE_RL_RELEASE_GIT_SHA")
if release_git_sha is None:
    raise RuntimeError("TRADE_RL_RELEASE_GIT_SHA is required for Production serving")

runtime = ServingRuntime(
    ...,
    release_git_sha=release_git_sha,
    strict_release_binding=True,
)
```

Do not change isolated unit-test defaults; strictness is selected explicitly by the Production entrypoint.

- [ ] **Step 7: Run focused tests**

```bash
pytest tests/test_serving_runtime.py tests/test_signal_server.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add mars_lite/serving/runtime.py mars_lite/server/signal_server.py scripts/run_server.py tests/test_serving_runtime.py tests/test_signal_server.py
git commit -m "feat: bind serving runtime to release git sha"
```

---

### Task 6: Verify the live served identity after activation

**Files:**
- Create: `scripts/verify_served_identity.py`
- Create: `tests/test_verify_served_identity.py`
- Modify: `.github/workflows/deploy.yml`
- Modify: `tests/test_deployment_activation.py`

**Interfaces:**
- Produces CLI: `python scripts/verify_served_identity.py --url URL --version VERSION --digest DIGEST --release-git-sha SHA --attempts 12 --interval-seconds 5`.
- Exit code `0` only for an exact identity match; exit code `1` for exhausted mismatch or transport failure.

- [ ] **Step 1: Write failing verification tests**

```python
from scripts.verify_served_identity import identity_matches


def test_identity_matches_ready_or_degraded_exact_identity() -> None:
    payload = {
        "status": "degraded",
        "active_version": "v123",
        "bundle_digest": "d" * 64,
        "release_git_sha": "a" * 40,
    }
    assert identity_matches(
        payload,
        expected_version="v123",
        expected_digest="d" * 64,
        expected_release_git_sha="a" * 40,
    )


def test_identity_rejects_previous_active_bundle() -> None:
    payload = {
        "status": "degraded",
        "active_version": "previous",
        "bundle_digest": "e" * 64,
        "release_git_sha": "a" * 40,
    }
    assert not identity_matches(
        payload,
        expected_version="v123",
        expected_digest="d" * 64,
        expected_release_git_sha="a" * 40,
    )
```

- [ ] **Step 2: Verify tests fail**

```bash
pytest tests/test_verify_served_identity.py -v
```

Expected: module import failure.

- [ ] **Step 3: Implement dependency-free polling**

```python
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def identity_matches(
    payload: dict[str, Any],
    *,
    expected_version: str,
    expected_digest: str,
    expected_release_git_sha: str,
) -> bool:
    return (
        payload.get("status") in {"ready", "degraded"}
        and payload.get("active_version") == expected_version
        and payload.get("bundle_digest") == expected_digest
        and payload.get("release_git_sha") == expected_release_git_sha
    )


def fetch_json(url: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise RuntimeError(f"readiness returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def verify_with_retries(
    *,
    url: str,
    expected_version: str,
    expected_digest: str,
    expected_release_git_sha: str,
    attempts: int,
    interval_seconds: float,
) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            payload = fetch_json(url)
            if identity_matches(
                payload,
                expected_version=expected_version,
                expected_digest=expected_digest,
                expected_release_git_sha=expected_release_git_sha,
            ):
                return True
            print(f"attempt {attempt}: served identity mismatch: {payload}")
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
            print(f"attempt {attempt}: readiness check failed: {exc}")
        if attempt < attempts:
            time.sleep(interval_seconds)
    return False
```

Add argparse and `raise SystemExit(0 if verify_with_retries(...) else 1)`.

- [ ] **Step 4: Update deploy workflow target requirements**

Use a stage environment and a runner that can access the persistent registry. The workflow must validate configuration before mutation:

```yaml
- name: Validate deployment target
  shell: bash
  run: |
    test -n "${TRADE_RL_REGISTRY_DIR:-}"
    test "${TRADE_RL_REGISTRY_DIR#/}" != "${TRADE_RL_REGISTRY_DIR}"
    test -n "${TRADE_RL_SERVING_READY_URL:-}"
```

Activation remains after evidence and bundle validation.

- [ ] **Step 5: Add post-activation verification immediately after activation**

```yaml
- name: Verify served identity
  run: >-
    python scripts/verify_served_identity.py
    --url "${TRADE_RL_SERVING_READY_URL}"
    --version "${MODEL_VERSION}"
    --digest "${BUNDLE_DIGEST}"
    --release-git-sha "${RELEASE_GIT_SHA}"
    --attempts 12
    --interval-seconds 5
```

The workflow must obtain `MODEL_VERSION`, `BUNDLE_DIGEST`, and `RELEASE_GIT_SHA` from the validated immutable bundle/evidence, not from manually typed dispatch inputs.

- [ ] **Step 6: Strengthen workflow structure test**

Assert the workflow text/order includes:

```python
assert workflow.index("manage_registry.py activate") < workflow.index("verify_served_identity.py")
assert "TRADE_RL_SERVING_READY_URL" in workflow
assert "--digest" in workflow
assert "--release-git-sha" in workflow
```

- [ ] **Step 7: Run focused tests**

```bash
pytest tests/test_verify_served_identity.py tests/test_deployment_activation.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/verify_served_identity.py tests/test_verify_served_identity.py .github/workflows/deploy.yml tests/test_deployment_activation.py
git commit -m "feat: verify live served identity after activation"
```

---

### Task 7: Normative documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/PRODUCTION_READINESS.md`

**Interfaces:**
- Documents the exact enforced contracts; does not change code interfaces.

- [ ] **Step 1: Update architecture documentation**

Add the following normative rules:

```text
A Production-capable bundle is eligible only when no release-disqualifying override was used, all mandatory gates passed, a sealed holdout was used for final Gate 2, and a complete release-risk policy is embedded in the immutable bundle.

Serving in strict mode rejects a bundle whose manifest Git SHA differs from TRADE_RL_RELEASE_GIT_SHA. The previously loaded healthy bundle remains active after a rejected refresh.
```

- [ ] **Step 2: Update operations documentation**

Document exact operator behavior:

```text
Deployment is not complete when Registry activation returns successfully. Completion requires the live /ready endpoint to report the approved active_version, bundle_digest, and release_git_sha. A mismatch or unreachable endpoint is a failed deployment and requires explicit operator rollback using the existing Registry rollback command.
```

- [ ] **Step 3: Keep readiness state NO-GO**

Ensure `README.md` and `docs/PRODUCTION_READINESS.md` still state **Production NO-GO** and list operational, security, exchange, legal, monitoring, rollback drill, and GameDay evidence as unresolved owner responsibilities.

- [ ] **Step 4: Run formatting, static checks, and all tests**

```bash
ruff check .
ruff format --check .
mypy mars_lite
pytest -q
```

Expected: all commands exit `0`.

- [ ] **Step 5: Inspect the final diff for scope and safety**

```bash
git diff main...HEAD --stat
git diff main...HEAD -- . ':!docs/superpowers/specs/*' ':!docs/superpowers/plans/*'
```

Confirm:

- no destructive serving endpoint was added;
- training never activates a bundle;
- research overrides never create a registrable candidate;
- risk policy and eligibility are protected by the bundle manifest digest;
- failed strict refresh preserves the prior healthy runtime;
- deployment verification occurs after activation;
- documentation still reports Production NO-GO.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/ARCHITECTURE.md docs/OPERATIONS.md docs/PRODUCTION_READINESS.md
git commit -m "docs: document production release safety contracts"
```

- [ ] **Step 7: Run final verification from the committed tree**

```bash
ruff check .
ruff format --check .
mypy mars_lite
pytest -q
git status --short
```

Expected: all quality commands exit `0` and `git status --short` is empty.

---

## Plan self-review

- Spec coverage: eligibility, override denial, sealed holdout, mandatory risk, bundle semantics, deployment gate, strict Git SHA binding, readiness identity, live verification, documentation, and full checks are each assigned to a task.
- Scope: no exchange adapter, distributed registry, TLS layer, container platform, or automatic rollback is introduced.
- Type consistency: `ReleaseEligibility.to_dict()`, `ReleaseRiskPolicy.to_dict()`, `ServingRuntime.release_git_sha`, and readiness keys use one spelling throughout.
- Failure behavior: every Production-path ambiguity fails closed; research-only execution remains available with no candidate registration.
- Test order: each behavior starts with a focused failing test before implementation and ends with a separate commit.
