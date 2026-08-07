"""Maintained single-instrument Binance research pipeline.

The historical multi-asset implementation remains in
``full_research_pipeline_legacy`` for read-only artifact compatibility. This
module fixes the maintained runtime to one BTCUSDT instrument and deliberately
does not expose triplet-selection APIs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from trade_rl.data.market import MarketDataset
from trade_rl.rl.observations import ObservationBuilder
from trade_rl.rl.sequence_observations import SequenceObservationBuilder

_RUNTIME_MODULE_NAME = "_trade_rl_single_symbol_full_research_pipeline_runtime"


def _load_legacy_runtime() -> ModuleType:
    path = Path(__file__).with_name("full_research_pipeline_legacy.py")
    spec = importlib.util.spec_from_file_location(_RUNTIME_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("single-symbol pipeline legacy runtime could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNTIME_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_RUNTIME_MODULE_NAME, None)
        raise
    return module


_legacy = _load_legacy_runtime()

_SYMBOLS = ("BTCUSDT",)
_NATIVE_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_FEATURE_TIMEFRAMES = ("1h", "4h", "1d")
_START = _legacy._START
_END = _legacy._END
_EXPECTED_15M_BARS = _legacy._EXPECTED_15M_BARS
_TRAIN_RUN_COMMAND = _legacy._TRAIN_RUN_COMMAND
_WALK_FORWARD_RUN_COMMAND = _legacy._WALK_FORWARD_RUN_COMMAND

# Only the private runtime copy is rebound. Importing the historical module by
# its public name continues to expose the original three-symbol state.
_legacy._SYMBOLS = _SYMBOLS
_legacy._SLOT_SYMBOLS = _SYMBOLS
_legacy._SYMBOL_POOL = _SYMBOLS
_legacy._ACTIVE_SYMBOL_TRIPLET = None


def _policy_observation_count(dataset: MarketDataset) -> int:
    """Return the exact flattened plus sequence observation width."""

    n_symbols = dataset.n_symbols
    flat = (
        ObservationBuilder(
            action_size=n_symbols,
            n_factors=0,
            finite_horizon=True,
        )
        .layout(dataset)
        .size
    )
    sequence = SequenceObservationBuilder().schema_payload(dataset)
    raw_windows = sequence.get("windows")
    if not isinstance(raw_windows, (tuple, list)):
        raise RuntimeError("sequence schema windows must be ordered")
    return flat + sum(
        n_symbols
        * int(dict(window)["length"])
        * len(tuple(dict(window)["feature_names"]))
        * 3
        for window in raw_windows
    )


def _require_single_symbol_run_payload(payload: dict[str, object]) -> None:
    action = payload.get("action")
    if not isinstance(action, dict):
        raise ValueError("maintained training requires an action object")
    if action.get("mode") != "target_weight" or action.get("target_weight_count") != 1:
        raise ValueError(
            "maintained training requires exactly one target-weight action"
        )


def _materialize_candidate_run_files(
    payload: dict[str, object],
    *,
    template_path: Path,
    validate_runs: bool,
) -> bool:
    candidates = payload.get("candidates", ())
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("walk-forward candidates must be an ordered list")
    changed = False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("walk-forward candidate must be an object")
        run = candidate.get("run")
        run_file = candidate.get("run_file")
        if run is not None and run_file is not None:
            raise ValueError("walk-forward candidate cannot define run and run_file")
        if run_file is not None:
            if not isinstance(run_file, str) or not run_file:
                raise ValueError("walk-forward candidate run_file must be a path")
            resolved = (template_path.parent / run_file).resolve()
            if resolved.parent != template_path.parent.resolve() or not resolved.is_file():
                raise ValueError("walk-forward candidate run_file is not maintained")
            run = _legacy._load_json(resolved)
            candidate.pop("run_file", None)
            candidate["run"] = run
            changed = True
        if run is None:
            continue
        if not isinstance(run, dict):
            raise ValueError("walk-forward candidate run must be an object")
        if validate_runs:
            _require_single_symbol_run_payload(run)
    return changed


def _write_run_config(*, template_path: Path, output_path: Path) -> Path:
    """Materialize relative candidate files and bind packaged Git provenance."""

    payload = _legacy._load_json(template_path)
    git_commit, git_dirty = _legacy._packaged_git_provenance()
    payload["git_commit"] = git_commit
    payload["git_dirty"] = git_dirty
    if "training" in payload:
        _require_single_symbol_run_payload(payload)
    _materialize_candidate_run_files(
        payload,
        template_path=template_path,
        validate_runs=True,
    )
    candidates = payload.get("candidates", ())
    assert isinstance(candidates, (list, tuple))
    for candidate in candidates:
        assert isinstance(candidate, dict)
        run = candidate.get("run")
        if run is None:
            continue
        assert isinstance(run, dict)
        run["git_commit"] = git_commit
        run["git_dirty"] = git_dirty
    _legacy._write_json(output_path, payload)
    return output_path


def _selected_walk_forward_recipe(
    walk_forward_path: Path,
    walk_forward_config_path: Path,
    output_path: Path,
) -> tuple[str, tuple[int, ...], Path]:
    """Select a recipe from embedded or relative-file candidate configuration."""

    payload = _legacy._load_json(walk_forward_config_path)
    if not _materialize_candidate_run_files(
        payload,
        template_path=walk_forward_config_path,
        validate_runs=False,
    ):
        return _legacy._selected_walk_forward_recipe(
            walk_forward_path,
            walk_forward_config_path,
            output_path,
        )
    materialized = output_path.parent / f".{output_path.name}.walk-forward-config.json"
    _legacy._write_json(materialized, payload)
    try:
        return _legacy._selected_walk_forward_recipe(
            walk_forward_path,
            materialized,
            output_path,
        )
    finally:
        materialized.unlink(missing_ok=True)


_legacy._policy_observation_count = _policy_observation_count

# Public maintained API. These helpers retain their existing semantics while
# resolving the one-symbol globals above.
parse_utc = _legacy._parse_utc
align_workflow_to_full_dataset = _legacy._align_workflow_to_full_dataset
policy_observation_count = _policy_observation_count
write_json = _legacy._write_json
load_json = _legacy._load_json
training_policy_digest = _legacy._training_policy_digest
prepare_run_roots = _legacy._prepare_run_roots
write_run_config = _write_run_config
run_cli = _legacy._run_cli
resolve_metadata = _legacy._resolve_metadata
build_dataset = _legacy._build_dataset
require_file = _legacy._require_file
verify_training = _legacy._verify_training
evaluate_walk_forward_research_gate = _legacy._evaluate_walk_forward_research_gate
execution_sensitivity_gate = _legacy._execution_sensitivity_gate
selected_walk_forward_recipe = _selected_walk_forward_recipe
finalize_research_run = _legacy._finalize_research_run
validate_maintained_dataset_preset = _legacy.validate_maintained_dataset_preset

# Private aliases retained for focused tests and the current state runner.
_parse_utc = parse_utc
_align_workflow_to_full_dataset = align_workflow_to_full_dataset
_write_json = write_json
_load_json = load_json
_training_policy_digest = training_policy_digest
_prepare_run_roots = prepare_run_roots
_write_run_config = write_run_config
_run_cli = run_cli
_resolve_metadata = resolve_metadata
_build_dataset = build_dataset
_require_file = require_file
_verify_training = verify_training
_evaluate_walk_forward_research_gate = evaluate_walk_forward_research_gate
_execution_sensitivity_gate = execution_sensitivity_gate
_selected_walk_forward_recipe = selected_walk_forward_recipe
_finalize_research_run = finalize_research_run
_load_rule_history = _legacy._load_rule_history
_selection_stability_passed = _legacy._selection_stability_passed

__all__ = [
    "_END",
    "_FEATURE_TIMEFRAMES",
    "_NATIVE_TIMEFRAMES",
    "_START",
    "_SYMBOLS",
    "_TRAIN_RUN_COMMAND",
    "_WALK_FORWARD_RUN_COMMAND",
    "align_workflow_to_full_dataset",
    "build_dataset",
    "evaluate_walk_forward_research_gate",
    "execution_sensitivity_gate",
    "finalize_research_run",
    "load_json",
    "parse_utc",
    "policy_observation_count",
    "prepare_run_roots",
    "require_file",
    "resolve_metadata",
    "run_cli",
    "selected_walk_forward_recipe",
    "training_policy_digest",
    "validate_maintained_dataset_preset",
    "verify_training",
    "write_json",
    "write_run_config",
]
