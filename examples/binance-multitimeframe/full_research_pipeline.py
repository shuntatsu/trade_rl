"""Maintained single-instrument Binance research pipeline.

The historical multi-asset implementation remains in
``full_research_pipeline_legacy`` for read-only artifact compatibility.  This
module fixes the maintained runtime to one BTCUSDT instrument and deliberately
does not expose triplet-selection APIs.
"""

from __future__ import annotations

import importlib

from trade_rl.data.market import MarketDataset
from trade_rl.rl.observations import ObservationBuilder
from trade_rl.rl.sequence_observations import SequenceObservationBuilder

_legacy = importlib.import_module("full_research_pipeline_legacy")

_SYMBOLS = ("BTCUSDT",)
_NATIVE_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_FEATURE_TIMEFRAMES = ("1h", "4h", "1d")
_START = _legacy._START
_END = _legacy._END
_EXPECTED_15M_BARS = _legacy._EXPECTED_15M_BARS
_TRAIN_RUN_COMMAND = _legacy._TRAIN_RUN_COMMAND
_WALK_FORWARD_RUN_COMMAND = _legacy._WALK_FORWARD_RUN_COMMAND

# The legacy implementation resolves these globals at call time.  Bind them to
# the maintained one-instrument contract before reusing its stable IO, evidence,
# metadata and research-gate helpers.
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


_legacy._policy_observation_count = _policy_observation_count

# Public maintained API.  These helpers retain their existing semantics while
# resolving the one-symbol globals above.
parse_utc = _legacy._parse_utc
align_workflow_to_full_dataset = _legacy._align_workflow_to_full_dataset
policy_observation_count = _policy_observation_count
write_json = _legacy._write_json
load_json = _legacy._load_json
training_policy_digest = _legacy._training_policy_digest
prepare_run_roots = _legacy._prepare_run_roots
write_run_config = _legacy._write_run_config
run_cli = _legacy._run_cli
resolve_metadata = _legacy._resolve_metadata
build_dataset = _legacy._build_dataset
require_file = _legacy._require_file
verify_training = _legacy._verify_training
evaluate_walk_forward_research_gate = _legacy._evaluate_walk_forward_research_gate
execution_sensitivity_gate = _legacy._execution_sensitivity_gate
selected_walk_forward_recipe = _legacy._selected_walk_forward_recipe
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
