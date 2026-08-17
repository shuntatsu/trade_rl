from __future__ import annotations

import ast
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_universal_causal_alpha_v3_signal as cli_module
from tests.workflows.test_universal_causal_alpha_v3_signal_forensics_v2_sidecars import (
    _complete_sidecars,
)
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.workflows import (
    universal_causal_alpha_v3_signal_forensics_v2_analysis as analysis_module,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticRealizedRow,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2 import (
    load_causal_alpha_v3_signal_forensics_v2,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_loader import (
    CausalAlphaV3SignalForensicsV2BoundScope,
    load_causal_alpha_v3_signal_forensics_v2_sidecars,
)


def _file_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _required(value: float | None) -> float:
    assert value is not None
    return value


def test_v2_report_is_absolute_path_independent_and_source_read_only(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left" / "run"
    right = tmp_path / "different" / "nested" / "run"
    _complete_sidecars(left)
    _complete_sidecars(right)
    before = _file_digests(left)

    left_report = load_causal_alpha_v3_signal_forensics_v2(left)
    after = _file_digests(left)
    right_report = load_causal_alpha_v3_signal_forensics_v2(right)

    assert after == before
    assert left_report.digest == right_report.digest
    assert left_report.to_payload() == right_report.to_payload()


def test_v2_72h_diagnostics_use_24h_equivalent_units_not_raw_72h_units(
    tmp_path: Path,
) -> None:
    _complete_sidecars(tmp_path)
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    analysis = analysis_module.build_causal_alpha_v3_signal_forensics_v2_analysis(bound)
    rows = bound[0].diagnostic.realized_72h_rows

    equivalent = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([row.prediction for row in rows], dtype=np.float64),
        np.asarray([row.realized_return for row in rows], dtype=np.float64),
    )
    raw = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([_required(row.raw_prediction) for row in rows], dtype=np.float64),
        np.asarray(
            [_required(row.raw_realized_return) for row in rows], dtype=np.float64
        ),
    )

    assert (
        analysis.scope_summaries[0].horizon_72h.to_payload() == equivalent.to_payload()
    )
    assert analysis.scope_summaries[0].horizon_72h.to_payload() != raw.to_payload()


def test_v2_undefined_correlation_remains_explicitly_undefined() -> None:
    rows = (
        CausalAlphaV3SignalDiagnosticRealizedRow(
            decision_index=10,
            label_end_index=11,
            available_feature_count=2,
            available_feature_fraction=1.0,
            prediction=0.1,
            realized_return=0.2,
        ),
        CausalAlphaV3SignalDiagnosticRealizedRow(
            decision_index=11,
            label_end_index=12,
            available_feature_count=2,
            available_feature_fraction=1.0,
            prediction=0.1,
            realized_return=0.3,
        ),
    )

    diagnostics = analysis_module._diagnostics_from_rows(rows)

    assert diagnostics.pearson_correlation is None
    assert diagnostics.rank_correlation is None
    assert diagnostics.undefined_correlation_reason == "constant_prediction"


def test_v2_paired_horizon_comparison_uses_exact_decision_intersection(
    tmp_path: Path,
) -> None:
    _complete_sidecars(tmp_path)
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    first = bound[0]
    diagnostic = replace(
        first.diagnostic,
        realized_72h_rows=first.diagnostic.realized_72h_rows[:-1],
        digest="",
    )
    modified = CausalAlphaV3SignalForensicsV2BoundScope(
        metric=first.metric,
        diagnostic=diagnostic,
    )
    expected = tuple(
        sorted(
            {row.decision_index for row in diagnostic.realized_24h_rows}
            & {row.decision_index for row in diagnostic.realized_72h_rows}
        )
    )

    paired = analysis_module._paired_horizons(modified)

    assert paired.decision_indices == expected
    assert paired.sample_count == len(expected)


def test_v2_chronological_prediction_std_uses_all_prediction_rows(
    tmp_path: Path,
) -> None:
    _complete_sidecars(tmp_path)
    original = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    target = original[0]
    target_diagnostic = replace(
        target.diagnostic,
        realized_24h_rows=target.diagnostic.realized_24h_rows[:-1],
        digest="",
    )
    modified = (
        CausalAlphaV3SignalForensicsV2BoundScope(
            metric=target.metric,
            diagnostic=target_diagnostic,
        ),
        *original[1:],
    )

    analysis = analysis_module.build_causal_alpha_v3_signal_forensics_v2_analysis(
        modified
    )
    series = next(
        item
        for item in analysis.chronological_horizon_series
        if item.fit_config_digest == target.metric.fit_config_digest
        and item.horizon == "24h"
    )
    interval = (target.metric.contract_start, target.metric.contract_stop)
    interval_index = series.contract_intervals.index(interval)
    cluster = tuple(
        scope
        for scope in modified
        if scope.metric.fit_config_digest == target.metric.fit_config_digest
        and (scope.metric.contract_start, scope.metric.contract_stop) == interval
    )
    all_predictions = np.asarray(
        [
            row.prediction_24h
            for scope in cluster
            for row in scope.diagnostic.prediction_rows
        ],
        dtype=np.float64,
    )

    assert series.prediction_standard_deviation.values[interval_index] == pytest.approx(
        float(np.std(all_predictions, dtype=np.float64))
    )


def test_v2_single_episode_chronological_slope_is_zero() -> None:
    summary = analysis_module._chronological_metric((1.25,))

    assert summary.count == 1
    assert summary.defined_count == 1
    assert summary.early_mean == pytest.approx(1.25)
    assert summary.late_mean == pytest.approx(1.25)
    assert summary.slope == pytest.approx(0.0)


def test_v2_analysis_layer_has_no_training_or_promotion_dependencies() -> None:
    module_path = analysis_module.__file__
    assert module_path is not None
    syntax = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
    forbidden_prefixes = (
        "trade_rl.integrations",
        "trade_rl.release",
        "trade_rl.rl",
        "trade_rl.workflows.universal_causal_alpha_fitting",
        "trade_rl.workflows.universal_causal_alpha_v3_runtime",
        "trade_rl.workflows.universal_causal_alpha_v3_selection",
        "trade_rl.workflows.universal_causal_alpha_v3_teacher",
    )

    assert not {
        module
        for module in imported_modules
        if any(module.startswith(prefix) for prefix in forbidden_prefixes)
    }


def test_v2_cli_rejects_output_inside_source_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(ValueError, match="outside"):
        cli_module.main(
            [
                str(run_root),
                "--schema",
                "v2",
                "--output",
                str(run_root / "signal" / "forensics-v2.json"),
            ]
        )
