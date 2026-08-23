"""Thin facade and CLI boundary for research-only Causal Alpha V4."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v4 import CausalAlphaV4FitConfig, CausalAlphaV4TargetConfig
from trade_rl.workflows.universal_causal_alpha_v4_pipeline import (
    CausalAlphaV4AdmissionRejected,
    CausalAlphaV4ResearchPackage,
    CausalAlphaV4SelectionRejected,
    CausalAlphaV4SignalRejected,
)
from trade_rl.workflows.universal_causal_alpha_v4_signal import CausalAlphaV4SignalGateConfig

CAUSAL_ALPHA_V4_RESEARCH_CONFIG_SCHEMA: Final = "universal_causal_alpha_v4_research_config_v1"


def _exact_dict(value: object, *, field: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result = dict(value)
    if set(result) != keys:
        raise ValueError(
            f"{field} fields mismatch; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}"
        )
    return result


@dataclass(frozen=True, slots=True)
class CausalAlphaV4ResearchConfig:
    fit: CausalAlphaV4FitConfig
    target: CausalAlphaV4TargetConfig
    signal_gate: CausalAlphaV4SignalGateConfig
    schema_version: str = CAUSAL_ALPHA_V4_RESEARCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.fit, CausalAlphaV4FitConfig):
            raise TypeError("V4 research fit config is invalid")
        if not isinstance(self.target, CausalAlphaV4TargetConfig):
            raise TypeError("V4 research target config is invalid")
        if not isinstance(self.signal_gate, CausalAlphaV4SignalGateConfig):
            raise TypeError("V4 research signal gate config is invalid")
        if self.schema_version != CAUSAL_ALPHA_V4_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V4 research config schema")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "fit_digest": self.fit.digest,
                "schema_version": self.schema_version,
                "signal_gate_digest": self.signal_gate.digest,
                "target_digest": self.target.digest,
            }
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CausalAlphaV4ResearchConfig":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("V4 research config JSON is invalid") from error
        root = _exact_dict(
            raw,
            field="V4 research config",
            keys=frozenset({"schema_version", "fit", "target", "signal_gate"}),
        )
        if root["schema_version"] != CAUSAL_ALPHA_V4_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V4 research config schema")
        fit = _exact_dict(
            root["fit"],
            field="V4 fit config",
            keys=frozenset(
                {"market_ridge_strength", "residual_ridge_strength", "direction_ridge_strength"}
            ),
        )
        target = _exact_dict(
            root["target"],
            field="V4 target config",
            keys=frozenset(
                {
                    "slow_target_magnitudes",
                    "fast_deviation_magnitudes",
                    "uncertainty_multiplier",
                    "execution_cost_multiplier",
                    "edge_margin",
                    "slow_rebalance_decisions",
                    "fast_rebalance_decisions",
                    "maximum_final_target_delta",
                    "maximum_fast_absolute_deviation",
                }
            ),
        )
        gate = _exact_dict(
            root["signal_gate"],
            field="V4 signal gate config",
            keys=frozenset(
                {
                    "independent_episode_count",
                    "minimum_rank_ic_lower_ci",
                    "minimum_top_bottom_spread_lower_ci",
                    "minimum_direction_accuracy_excess_lower_ci",
                    "bootstrap_resamples",
                    "bootstrap_seed",
                    "bootstrap_block_size",
                }
            ),
        )
        slow = target["slow_target_magnitudes"]
        fast = target["fast_deviation_magnitudes"]
        if not isinstance(slow, list) or not isinstance(fast, list):
            raise ValueError("V4 target magnitudes must be arrays")
        return cls(
            fit=CausalAlphaV4FitConfig(
                market_ridge_strength=float(fit["market_ridge_strength"]),
                residual_ridge_strength=float(fit["residual_ridge_strength"]),
                direction_ridge_strength=float(fit["direction_ridge_strength"]),
            ),
            target=CausalAlphaV4TargetConfig(
                slow_target_magnitudes=tuple(float(value) for value in slow),
                fast_deviation_magnitudes=tuple(float(value) for value in fast),
                uncertainty_multiplier=float(target["uncertainty_multiplier"]),
                execution_cost_multiplier=float(target["execution_cost_multiplier"]),
                edge_margin=float(target["edge_margin"]),
                slow_rebalance_decisions=int(target["slow_rebalance_decisions"]),
                fast_rebalance_decisions=int(target["fast_rebalance_decisions"]),
                maximum_final_target_delta=float(target["maximum_final_target_delta"]),
                maximum_fast_absolute_deviation=float(
                    target["maximum_fast_absolute_deviation"]
                ),
            ),
            signal_gate=CausalAlphaV4SignalGateConfig.from_mapping(gate),
        )


def run_universal_causal_alpha_v4_research_from_paths(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV4ResearchPackage:
    """Resolve immutable path inputs and execute the concrete V4 research runner.

    The concrete runtime/stage assembly is imported lazily so the CLI remains a thin
    boundary and unit tests can inject a side-effect-free runner.
    """

    from trade_rl.workflows.universal_causal_alpha_v4_stage_entry import (
        run_causal_alpha_v4_stage_entry,
    )

    return run_causal_alpha_v4_stage_entry(
        config_path=Path(config_path),
        run_config_path=Path(run_config_path),
        runtime_manifest_path=Path(runtime_manifest_path),
        v4_context_manifest_path=Path(v4_context_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
        output_root=Path(output_root),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research-only Causal Alpha V4")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--v4-context-manifest", required=True, type=Path)
    parser.add_argument("--frozen-metadata-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def cli_main(
    argv: Sequence[str] | None = None,
    *,
    run_from_paths: Callable[..., object] = run_universal_causal_alpha_v4_research_from_paths,
) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_from_paths(
            config_path=args.config,
            run_config_path=args.run_config,
            runtime_manifest_path=args.runtime_manifest,
            v4_context_manifest_path=args.v4_context_manifest,
            frozen_metadata_root=args.frozen_metadata_root,
            output_root=args.output_root,
        )
    except CausalAlphaV4SignalRejected as rejection:
        _emit(
            {
                "artifact_digest": rejection.digest,
                "promotion_eligible": False,
                "status": "signal_rejected",
            }
        )
        return 2
    except CausalAlphaV4SelectionRejected as rejection:
        _emit(
            {
                "artifact_digest": rejection.digest,
                "promotion_eligible": False,
                "status": "selection_rejected",
            }
        )
        return 3
    except CausalAlphaV4AdmissionRejected as rejection:
        _emit(
            {
                "artifact_digest": rejection.digest,
                "promotion_eligible": False,
                "status": "admission_rejected",
            }
        )
        return 4
    if not isinstance(result, CausalAlphaV4ResearchPackage):
        raise TypeError("V4 runner returned an invalid research package")
    _emit(
        {
            "artifact_digest": result.digest,
            "promotion_eligible": result.promotion_eligible,
            "research_only": result.research_only,
            "status": "admitted",
        }
    )
    return 0


__all__ = [
    "CAUSAL_ALPHA_V4_RESEARCH_CONFIG_SCHEMA",
    "CausalAlphaV4ResearchConfig",
    "cli_main",
    "run_universal_causal_alpha_v4_research_from_paths",
]
