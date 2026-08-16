from __future__ import annotations

import importlib
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
    signal_scope_metric_from_payload,
)


_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_WEAK_RANKS = (-0.08, -0.02, 0.01, 0.03)
_STRONG_RANKS = (0.01, 0.03, 0.07, 0.09)
_WEAK_SPREADS = (-0.003, -0.001, 0.0005, 0.001)
_STRONG_SPREADS = (0.001, 0.002, 0.003, 0.004)
_WEAK_DIRECTIONS = (-0.03, -0.01, 0.0, 0.01)
_STRONG_DIRECTIONS = (-0.01, 0.01, 0.03, 0.05)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _target_payload() -> dict[str, object]:
    return {
        "target_magnitudes": [0.0, 0.05, 0.1],
        "uncertainty_multiplier": 1.0,
        "execution_cost_multiplier": 1.5,
        "edge_margin": 0.001,
        "alpha_rebalance_decisions": 4,
        "strong_reversal_threshold": 0.02,
        "max_target_delta": 0.125,
    }


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_causal_alpha_v3_research_config_v2",
        "nested_selection": {
            "signal_contract_count": 4,
            "minimum_economic_contract_count": 1,
        },
        "signal_gate": {
            "minimum_independent_episode_count": 4,
            "minimum_raw_scope_coverage": 1.0,
            "minimum_rank_ic_lower_ci": 1.0,
            "minimum_top_bottom_spread_lower_ci": 1.0,
            "minimum_direction_accuracy_excess_lower_ci": 1.0,
            "bootstrap_resamples": 16,
            "bootstrap_seed": 20260816,
            "bootstrap_block_size": 2,
        },
        "selection_gate": {
            "minimum_mean_gross_return": 0.0,
            "minimum_mean_net_return": 0.0,
            "minimum_symbol_episode_net_return": -0.05,
            "maximum_mean_turnover_per_day": 1.0,
            "maximum_unexplained_execution_rejections": 0,
            "minimum_positive_gross_episode_fraction": 0.5,
        },
        "candidates": [
            {
                "name": "ridge-weak",
                "fit": {"ridge_strength": 0.1},
                "target": _target_payload(),
            },
            {
                "name": "ridge-strong",
                "fit": {"ridge_strength": 1.0},
                "target": _target_payload(),
            },
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_values(
    *,
    fit_index: int,
    episode_index: int,
    symbol_index: int,
) -> tuple[float, float, float]:
    rank_means = (_WEAK_RANKS, _STRONG_RANKS)[fit_index]
    spread_means = (_WEAK_SPREADS, _STRONG_SPREADS)[fit_index]
    direction_means = (_WEAK_DIRECTIONS, _STRONG_DIRECTIONS)[fit_index]
    sign = -1.0 if symbol_index == 0 else 1.0
    return (
        rank_means[episode_index] + sign * 0.01,
        spread_means[episode_index] + sign * 0.0002,
        0.5 + direction_means[episode_index] + sign * 0.01,
    )


def _build_run(root: Path) -> dict[str, Any]:
    config_payload = _config_payload()
    config = CausalAlphaV3ResearchConfig.from_mapping(config_payload)
    manifest = CausalAlphaV3RunManifestV2(
        train_symbols=_SYMBOLS,
        config_digest=config.digest,
        catalog_digest=_digest("catalog"),
        partition_digest=_digest("partition"),
        split_manifest_digest=_digest("split"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        generator_code_digest=_digest("generator"),
        nested_partition_digest=_digest("nested"),
        execution_identity_digest=_digest("execution"),
        training_contract_digest=_digest("training-contract"),
        instrument_context_schema_digest=_digest("instrument-context"),
    )
    _write_json(root / "run-manifest.json", manifest.to_payload())
    _write_json(root / "authored-config.json", config_payload)

    fit_configs = []
    metrics_by_fit: dict[str, list[CausalAlphaV3SignalScopeMetric]] = {}
    paths_by_identity: dict[
        tuple[str, str, int], Path
    ] = {}
    for fit_index, candidate in enumerate(config.candidates):
        fit_digest = candidate.fit.digest
        fit_configs.append(fit_digest)
        metrics_by_fit[fit_digest] = []
        for symbol_index, symbol in enumerate(_SYMBOLS):
            for episode_index in range(4):
                start = 1000 + episode_index * 100
                stop = start + 80
                rank, spread, direction = _metric_values(
                    fit_index=fit_index,
                    episode_index=episode_index,
                    symbol_index=symbol_index,
                )
                sample_count = 4 + episode_index + 2 * symbol_index
                metric = CausalAlphaV3SignalScopeMetric(
                    run_manifest_digest=manifest.digest,
                    fit_config_digest=fit_digest,
                    symbol=symbol,
                    episode_index=episode_index,
                    contract_start=start,
                    contract_stop=stop,
                    contract_digest=_digest(
                        f"contract:{symbol}:{episode_index}"
                    ),
                    fit_digest=_digest(
                        f"pooled-fit:{fit_index}:{episode_index}"
                    ),
                    forecast_digest=_digest(
                        f"forecast:{fit_index}:{symbol}:{episode_index}"
                    ),
                    sample_count=sample_count,
                    rank_correlation=rank,
                    direction_accuracy=direction,
                    top_bottom_realized_spread=spread,
                    cohort_indices=tuple(range(start, start + sample_count)),
                )
                path = (
                    root
                    / "signal"
                    / "records"
                    / fit_digest
                    / symbol
                    / f"{episode_index}.json"
                )
                _write_json(path, metric.to_payload())
                metrics_by_fit[fit_digest].append(metric)
                paths_by_identity[metric.identity] = path

    fit_results: list[dict[str, object]] = []
    for fit_digest in fit_configs:
        metrics = tuple(
            sorted(
                metrics_by_fit[fit_digest],
                key=lambda item: (
                    _SYMBOLS.index(item.symbol),
                    item.episode_index,
                ),
            )
        )
        evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
            metrics,
            expected_raw_scope_count=8,
            expected_independent_episode_count=4,
            gate=config.signal_gate,
        )
        assert not evidence.passed
        fit_results.append(
            {
                "evidence": evidence.to_payload(),
                "fit_config_digest": fit_digest,
                "passed": False,
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_fit_signal_result_v2",
                "unavailable_scope_contract_digests": [],
            }
        )
    rejection_body: dict[str, object] = {
        "fit_results": fit_results,
        "promotion_eligible": False,
        "schema_version": "causal_alpha_v3_signal_rejection_v2",
    }
    rejection_payload = {
        **rejection_body,
        "artifact_digest": content_digest(rejection_body),
    }
    rejection_path = root / "signal" / "rejection.json"
    _write_json(rejection_path, rejection_payload)
    return {
        "config": config,
        "fit_configs": tuple(fit_configs),
        "manifest": manifest,
        "metrics_by_fit": metrics_by_fit,
        "paths_by_identity": paths_by_identity,
        "rejection_path": rejection_path,
    }


def _api() -> Any:
    return importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_signal_forensics"
    )


def _load_metric(path: Path) -> CausalAlphaV3SignalScopeMetric:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return signal_scope_metric_from_payload(raw)


def _rewrite_metric(path: Path, metric: CausalAlphaV3SignalScopeMetric) -> None:
    _write_json(path, metric.to_payload())


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    return {**body, "artifact_digest": content_digest(body)}


def test_signal_forensics_aggregates_fit_episode_and_symbol_views(
    tmp_path: Path,
) -> None:
    built = _build_run(tmp_path)
    report = _api().load_causal_alpha_v3_signal_forensics(tmp_path)

    assert report.schema_version == "causal_alpha_v3_signal_forensics_v1"
    assert report.raw_scope_count == 16
    assert report.independent_episode_count == 4
    assert report.source_signal_status == "rejected"
    assert report.promotion_eligible is False
    assert report.research_only is True
    assert len(report.fit_summaries) == 2
    assert len(report.episode_summaries) == 8
    assert len(report.symbol_summaries) == 4

    strong_digest = built["fit_configs"][1]
    strong = next(
        item for item in report.fit_summaries if item.fit_config_digest == strong_digest
    )
    assert strong.candidate_names == ("ridge-strong",)
    assert strong.ridge_strength == pytest.approx(1.0)
    assert strong.raw_scope_count == 8
    assert strong.independent_episode_count == 4
    assert strong.sample_count.minimum == pytest.approx(4.0)
    assert strong.sample_count.mean == pytest.approx(6.5)
    assert strong.sample_count.maximum == pytest.approx(9.0)
    assert strong.episode_rank_ic.mean == pytest.approx(0.05)
    assert strong.episode_top_bottom_spread.mean == pytest.approx(0.0025)
    assert strong.episode_direction_accuracy_excess.mean == pytest.approx(0.02)
    assert strong.rank_ic_trend.early_mean == pytest.approx(0.02)
    assert strong.rank_ic_trend.late_mean == pytest.approx(0.08)
    assert strong.rank_ic_trend.slope == pytest.approx(0.028)
    assert strong.fit_digest_unique_count == 4
    assert strong.fit_digest_transition_count == 3

    first_episode = next(
        item
        for item in report.episode_summaries
        if item.fit_config_digest == strong_digest and item.contract_start == 1000
    )
    assert first_episode.symbol_count == 2
    assert first_episode.total_sample_count == 10
    assert first_episode.mean_sample_count == pytest.approx(5.0)
    assert first_episode.rank_ic == pytest.approx(0.01)
    assert first_episode.top_bottom_spread == pytest.approx(0.001)
    assert first_episode.direction_accuracy == pytest.approx(0.49)
    assert first_episode.direction_accuracy_excess == pytest.approx(-0.01)
    assert first_episode.negative_rank_symbol_count == 1
    assert first_episode.negative_spread_symbol_count == 0
    assert first_episode.negative_direction_excess_symbol_count == 1

    btc = next(
        item
        for item in report.symbol_summaries
        if item.fit_config_digest == strong_digest and item.symbol == "BTCUSDT"
    )
    assert btc.episode_count == 4
    assert btc.sample_count.minimum == pytest.approx(4.0)
    assert btc.sample_count.maximum == pytest.approx(7.0)
    assert btc.rank_ic.mean == pytest.approx(0.04)
    assert btc.direction_accuracy_excess.mean == pytest.approx(0.01)


def test_signal_forensics_compares_fits_on_paired_episodes(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    report = _api().load_causal_alpha_v3_signal_forensics(tmp_path)

    assert len(report.fit_comparisons) == 1
    comparison = report.fit_comparisons[0]
    assert comparison.left_fit_config_digest == built["fit_configs"][0]
    assert comparison.right_fit_config_digest == built["fit_configs"][1]
    assert comparison.common_episode_count == 4
    assert comparison.mean_rank_ic_delta == pytest.approx(-0.065)
    assert comparison.mean_top_bottom_spread_delta == pytest.approx(-0.003125)
    assert comparison.mean_direction_accuracy_excess_delta == pytest.approx(-0.0275)
    assert comparison.left_rank_ic_win_fraction == pytest.approx(0.0)
    assert comparison.left_top_bottom_spread_win_fraction == pytest.approx(0.0)
    assert comparison.left_direction_accuracy_excess_win_fraction == pytest.approx(0.0)


def test_signal_forensics_is_deterministic_and_truthful_about_missing_inputs(
    tmp_path: Path,
) -> None:
    _build_run(tmp_path)
    first = _api().load_causal_alpha_v3_signal_forensics(tmp_path)
    second = _api().load_causal_alpha_v3_signal_forensics(tmp_path)

    assert first.digest == second.digest
    assert first.to_payload() == second.to_payload()
    unavailable = {item.analysis: item.reason for item in first.unavailable_analyses}
    assert set(unavailable) == {
        "horizon_24h_vs_72h",
        "coefficient_cosine_similarity",
        "coefficient_sign_flip_rate",
        "prediction_distribution",
        "residual_rmse_by_episode",
    }
    assert all(unavailable.values())


def test_signal_forensics_rejects_corrupt_leaf_digest(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    path = next(iter(built["paths_by_identity"].values()))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["artifact_digest"] = "0" * 64
    _write_json(path, raw)

    with pytest.raises(ValueError):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_wrong_run_identity(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    path = next(iter(built["paths_by_identity"].values()))
    metric = _load_metric(path)
    _rewrite_metric(
        path,
        replace(metric, run_manifest_digest=_digest("foreign-run"), digest=""),
    )

    with pytest.raises(ValueError, match="run manifest"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_wrong_record_path(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    path = next(iter(built["paths_by_identity"].values()))
    path.rename(path.with_name("99.json"))

    with pytest.raises(ValueError, match="path"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_unknown_symbol(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    path = next(iter(built["paths_by_identity"].values()))
    metric = _load_metric(path)
    replacement = replace(metric, symbol="DOGEUSDT", digest="")
    path.unlink()
    replacement_path = (
        tmp_path
        / "signal"
        / "records"
        / replacement.fit_config_digest
        / replacement.symbol
        / f"{replacement.episode_index}.json"
    )
    _rewrite_metric(replacement_path, replacement)

    with pytest.raises(ValueError, match="symbol"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_unknown_fit_config(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    path = next(iter(built["paths_by_identity"].values()))
    metric = _load_metric(path)
    replacement = replace(metric, fit_config_digest=_digest("unknown-fit"), digest="")
    path.unlink()
    replacement_path = (
        tmp_path
        / "signal"
        / "records"
        / replacement.fit_config_digest
        / replacement.symbol
        / f"{replacement.episode_index}.json"
    )
    _rewrite_metric(replacement_path, replacement)

    with pytest.raises(ValueError, match="fit config"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_incomplete_episode_cluster(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    fit_digest = built["fit_configs"][0]
    missing = built["paths_by_identity"][(fit_digest, "ETHUSDT", 2)]
    missing.unlink()

    with pytest.raises(ValueError, match="complete symbol scope"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_mixed_cluster_fit_identity(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    fit_digest = built["fit_configs"][0]
    path = built["paths_by_identity"][(fit_digest, "ETHUSDT", 1)]
    metric = _load_metric(path)
    _rewrite_metric(
        path,
        replace(metric, fit_digest=_digest("mixed-pooled-fit"), digest=""),
    )

    with pytest.raises(ValueError, match="pooled fit"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_rejection_metric_digest_drift(
    tmp_path: Path,
) -> None:
    built = _build_run(tmp_path)
    path = built["rejection_path"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    evidence = raw["fit_results"][0]["evidence"]
    evidence["metric_digests"][0] = _digest("foreign-metric")
    raw["fit_results"][0]["evidence"] = _rehash(evidence)
    _write_json(path, _rehash(raw))

    with pytest.raises(ValueError, match="metric digests"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)


def test_signal_forensics_rejects_rejection_outer_digest_drift(
    tmp_path: Path,
) -> None:
    built = _build_run(tmp_path)
    path = built["rejection_path"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["artifact_digest"] = "f" * 64
    _write_json(path, raw)

    with pytest.raises(ValueError, match="rejection digest"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)
