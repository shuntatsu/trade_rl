"""Strict config and CLI boundary for research-only Causal Alpha V7."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6TargetConfig
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7CalibrationConfig
from trade_rl.workflows.universal_causal_alpha_v6_runner import (
    _TARGET_KEYS,
    _integer,
    _magnitudes,
    _number,
    _ordered_mapping,
)
from trade_rl.workflows.universal_causal_alpha_v7_pipeline import (
    CausalAlphaV7ResearchPackage,
    CausalAlphaV7StageRejected,
)
from trade_rl.workflows.universal_causal_alpha_v7_stage_entry import (
    causal_alpha_v7_stage_config_digest,
)

CAUSAL_ALPHA_V7_RESEARCH_CONFIG_SCHEMA: Final = (
    "universal_causal_alpha_v7_research_config_v1"
)
_ROOT_KEYS: Final = ("schema_version", "calibration", "target")
_CALIBRATION_KEYS: Final = (
    "calibration_fraction",
    "forward_block_count",
    "ridge_strength",
    "minimum_pooled_support",
    "minimum_symbol_support",
    "working_memory_rows",
)


@dataclass(frozen=True, slots=True)
class CausalAlphaV7ResearchConfig:
    calibration: CausalAlphaV7CalibrationConfig
    target: CausalAlphaV6TargetConfig
    schema_version: str = CAUSAL_ALPHA_V7_RESEARCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, CausalAlphaV7CalibrationConfig):
            raise TypeError("V7 research calibration config is invalid")
        if not isinstance(self.target, CausalAlphaV6TargetConfig):
            raise TypeError("V7 research target config is invalid")
        if self.schema_version != CAUSAL_ALPHA_V7_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V7 research config schema")

    @property
    def stage_config_digest(self) -> str:
        return causal_alpha_v7_stage_config_digest(
            calibration=self.calibration,
            target=self.target,
        )

    @property
    def digest(self) -> str:
        return self.stage_config_digest

    @classmethod
    def from_json(cls, path: str | Path) -> CausalAlphaV7ResearchConfig:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("V7 research config JSON is invalid") from error
        root = _ordered_mapping(raw, keys=_ROOT_KEYS, field="V7 research config")
        if root["schema_version"] != CAUSAL_ALPHA_V7_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V7 research config schema")
        calibration = _ordered_mapping(
            root["calibration"],
            keys=_CALIBRATION_KEYS,
            field="V7 calibration config",
        )
        target = _ordered_mapping(
            root["target"],
            keys=_TARGET_KEYS,
            field="V7 target config",
        )
        return cls(
            calibration=CausalAlphaV7CalibrationConfig(
                calibration_fraction=_number(
                    calibration["calibration_fraction"],
                    field="calibration_fraction",
                ),
                forward_block_count=_integer(
                    calibration["forward_block_count"],
                    field="forward_block_count",
                ),
                ridge_strength=_number(
                    calibration["ridge_strength"],
                    field="ridge_strength",
                ),
                minimum_pooled_support=_integer(
                    calibration["minimum_pooled_support"],
                    field="minimum_pooled_support",
                ),
                minimum_symbol_support=_integer(
                    calibration["minimum_symbol_support"],
                    field="minimum_symbol_support",
                ),
                working_memory_rows=_integer(
                    calibration["working_memory_rows"],
                    field="working_memory_rows",
                ),
            ),
            target=CausalAlphaV6TargetConfig(
                target_magnitudes=_magnitudes(target["target_magnitudes"]),
                maximum_absolute_target=_number(
                    target["maximum_absolute_target"],
                    field="maximum_absolute_target",
                ),
                maximum_target_delta=_number(
                    target["maximum_target_delta"],
                    field="maximum_target_delta",
                ),
                fast_rebalance_decisions=_integer(
                    target["fast_rebalance_decisions"],
                    field="fast_rebalance_decisions",
                ),
                slow_context_decisions=_integer(
                    target["slow_context_decisions"],
                    field="slow_context_decisions",
                ),
                uncertainty_multiplier=_number(
                    target["uncertainty_multiplier"],
                    field="uncertainty_multiplier",
                ),
                execution_cost_multiplier=_number(
                    target["execution_cost_multiplier"],
                    field="execution_cost_multiplier",
                ),
                edge_margin=_number(target["edge_margin"], field="edge_margin"),
                confirmation_count=_integer(
                    target["confirmation_count"],
                    field="confirmation_count",
                ),
                strong_reversal_threshold=_number(
                    target["strong_reversal_threshold"],
                    field="strong_reversal_threshold",
                ),
                liquidity_lookback_decisions=_integer(
                    target["liquidity_lookback_decisions"],
                    field="liquidity_lookback_decisions",
                ),
                liquidity_lower_quantile=_number(
                    target["liquidity_lower_quantile"],
                    field="liquidity_lower_quantile",
                ),
                liquidity_safety_multiplier=_number(
                    target["liquidity_safety_multiplier"],
                    field="liquidity_safety_multiplier",
                ),
            ),
        )


def run_universal_causal_alpha_v7_research_from_paths(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV7ResearchPackage:
    from trade_rl.workflows.universal_causal_alpha_v7_stage_execution import (
        run_causal_alpha_v7_stage_entry,
    )

    return run_causal_alpha_v7_stage_entry(
        config_path=config_path,
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
        output_root=output_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research-only Causal Alpha V7")
    for name in (
        "config",
        "run-config",
        "runtime-manifest",
        "v4-context-manifest",
        "frozen-metadata-root",
        "output-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    return parser


def cli_main(
    argv: Sequence[str] | None = None,
    *,
    run_from_paths: Callable[
        ..., object
    ] = run_universal_causal_alpha_v7_research_from_paths,
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
    except CausalAlphaV7StageRejected as rejection:
        print(
            json.dumps(
                {
                    "artifact_digest": rejection.digest,
                    "promotion_eligible": False,
                    "status": f"{rejection.stage}_rejected",
                },
                sort_keys=True,
            )
        )
        return rejection.exit_code
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "error": f"{type(error).__name__}:{error}",
                    "promotion_eligible": False,
                    "status": "invalid_preparation_or_config",
                },
                sort_keys=True,
            )
        )
        return 5
    if not isinstance(result, CausalAlphaV7ResearchPackage):
        raise TypeError("V7 runner returned an invalid research package")
    print(
        json.dumps(
            {
                "artifact_digest": result.digest,
                "promotion_eligible": False,
                "research_only": True,
                "status": "admitted",
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CAUSAL_ALPHA_V7_RESEARCH_CONFIG_SCHEMA",
    "CausalAlphaV7ResearchConfig",
    "cli_main",
    "run_universal_causal_alpha_v7_research_from_paths",
]
