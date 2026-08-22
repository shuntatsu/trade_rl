from __future__ import annotations

import json
from pathlib import Path

from tests.architecture.import_references import (
    causal_scenario_dependency_violations,
    forbidden_json_key_paths,
    module_name_from_path,
    scan_import_references,
)
from trade_rl.workflows.causal_scenario import (
    CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA,
    CausalConditionConfig,
    CausalScenarioLibraryConfig,
    CausalScenarioReplayIdentity,
    build_causal_scenario_library,
    load_causal_scenario_library_artifact,
    materialize_causal_scenario_dataset,
    select_causal_scenarios,
    write_causal_scenario_library_artifact,
)

_PACKAGE_ROOT = Path("trade_rl")
_CAUSAL_SCENARIO_PREFIX = "trade_rl.workflows.causal_scenario"
_CAUSAL_SCENARIO_ROOT = _PACKAGE_ROOT / "workflows" / "causal_scenario"
_INTEGRATIONS_ROOT = _PACKAGE_ROOT / "integrations"
_BOOTSTRAP_EXTENSION_BOUNDARY = _INTEGRATIONS_ROOT / "runtime_factory.py"
_PROTECTED_ROOTS = (
    _PACKAGE_ROOT / "rl",
    _PACKAGE_ROOT / "serving",
    _PACKAGE_ROOT / "release",
    _PACKAGE_ROOT / "workflows",
)
_WALK_FORWARD_CONFIG = Path("examples/binance-multitimeframe/walk-forward-full.json")


def _runtime_dependency_violations() -> list[str]:
    violations = list(
        causal_scenario_dependency_violations(
            protected_roots=_PROTECTED_ROOTS,
            excluded_root=_CAUSAL_SCENARIO_ROOT,
            package_root=_PACKAGE_ROOT,
            root_package="trade_rl",
            prohibited_prefix=_CAUSAL_SCENARIO_PREFIX,
        )
    )
    violations.extend(
        causal_scenario_dependency_violations(
            protected_roots=(_INTEGRATIONS_ROOT,),
            excluded_root=_BOOTSTRAP_EXTENSION_BOUNDARY,
            package_root=_PACKAGE_ROOT,
            root_package="trade_rl",
            prohibited_prefix=_CAUSAL_SCENARIO_PREFIX,
        )
    )
    return violations


def test_c2_public_api_is_available() -> None:
    assert CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA.endswith("_v1")
    assert CausalConditionConfig is not None
    assert CausalScenarioLibraryConfig is not None
    assert CausalScenarioReplayIdentity is not None
    assert build_causal_scenario_library is not None
    assert select_causal_scenarios is not None
    assert materialize_causal_scenario_dataset is not None
    assert write_causal_scenario_library_artifact is not None
    assert load_causal_scenario_library_artifact is not None


def test_runtime_factory_is_one_explicit_dynamic_bootstrap_extension_boundary() -> None:
    assert _BOOTSTRAP_EXTENSION_BOUNDARY.is_file()
    module_name = module_name_from_path(
        _BOOTSTRAP_EXTENSION_BOUNDARY,
        package_root=_PACKAGE_ROOT,
        root_package="trade_rl",
    )
    references = scan_import_references(
        _BOOTSTRAP_EXTENSION_BOUNDARY,
        module_name=module_name,
    )
    causal_scenario_references = tuple(
        reference
        for reference in references
        if reference.target is not None
        and (
            reference.target == _CAUSAL_SCENARIO_PREFIX
            or reference.target.startswith(f"{_CAUSAL_SCENARIO_PREFIX}.")
        )
    )
    unresolved_references = tuple(
        reference for reference in references if reference.unresolved
    )

    assert causal_scenario_references == ()
    assert len(unresolved_references) == 1
    assert unresolved_references[0].kind == "dynamic"
    assert unresolved_references[0].target is None


def test_causal_scenario_library_remains_outside_runtime_paths() -> None:
    violations = _runtime_dependency_violations()
    if _WALK_FORWARD_CONFIG.exists():
        payload = json.loads(_WALK_FORWARD_CONFIG.read_text(encoding="utf-8"))
        violations.extend(
            f"{_WALK_FORWARD_CONFIG}:{path}:causal_scenario_library"
            for path in forbidden_json_key_paths(
                payload,
                key="causal_scenario_library",
            )
        )
    assert violations == []
