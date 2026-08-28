"""Strict config and CLI boundary for research-only Causal Alpha V6."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6TargetConfig
from trade_rl.workflows.universal_causal_alpha_v6_pipeline import (
    CausalAlphaV6ResearchPackage,
    CausalAlphaV6StageRejected,
)

CAUSAL_ALPHA_V6_RESEARCH_CONFIG_SCHEMA: Final = (
    "universal_causal_alpha_v6_research_config_v1"
)
_ROOT_KEYS: Final = ("schema_version", "target")
_TARGET_KEYS: Final = (
    "target_magnitudes",
    "maximum_absolute_target",
    "maximum_target_delta",
    "fast_rebalance_decisions",
    "slow_context_decisions",
    "uncertainty_multiplier",
    "execution_cost_multiplier",
    "edge_margin",
    "confirmation_count",
    "strong_reversal_threshold",
    "liquidity_lookback_decisions",
    "liquidity_lower_quantile",
    "liquidity_safety_multiplier",
)


def _ordered_mapping(
    value: object,
    *,
    keys: tuple[str, ...],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result = {str(key): item for key, item in value.items()}
    if tuple(result) != keys:
        raise ValueError(
            f"{field} fields/order mismatch; expected={list(keys)}, observed={list(result)}"
        )
    return result


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric and not boolean")
    return float(value)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer and not boolean")
    return value


def _magnitudes(value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError("target_magnitudes must be an array")
    return tuple(_number(item, field="target_magnitudes") for item in value)


@dataclass(frozen=True, slots=True)
class CausalAlphaV6ResearchConfig:
    target: CausalAlphaV6TargetConfig
    schema_version: str = CAUSAL_ALPHA_V6_RESEARCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.target, CausalAlphaV6TargetConfig):
            raise TypeError("V6 research target config is invalid")
        if self.schema_version != CAUSAL_ALPHA_V6_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V6 research config schema")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": self.schema_version,
                "target_digest": self.target.digest,
            }
        )

    @classmethod
    def from_json(cls, path: str | Path) -> CausalAlphaV6ResearchConfig:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("V6 research config JSON is invalid") from error
        root = _ordered_mapping(raw, keys=_ROOT_KEYS, field="V6 research config")
        if root["schema_version"] != CAUSAL_ALPHA_V6_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V6 research config schema")
        values = _ordered_mapping(
            root["target"],
            keys=_TARGET_KEYS,
            field="V6 target config",
        )
        return cls(
            target=CausalAlphaV6TargetConfig(
                target_magnitudes=_magnitudes(values["target_magnitudes"]),
                maximum_absolute_target=_number(
                    values["maximum_absolute_target"],
                    field="maximum_absolute_target",
                ),
                maximum_target_delta=_number(
                    values["maximum_target_delta"],
                    field="maximum_target_delta",
                ),
                fast_rebalance_decisions=_integer(
                    values["fast_rebalance_decisions"],
                    field="fast_rebalance_decisions",
                ),
                slow_context_decisions=_integer(
                    values["slow_context_decisions"],
                    field="slow_context_decisions",
                ),
                uncertainty_multiplier=_number(
                    values["uncertainty_multiplier"],
                    field="uncertainty_multiplier",
                ),
                execution_cost_multiplier=_number(
                    values["execution_cost_multiplier"],
                    field="execution_cost_multiplier",
                ),
                edge_margin=_number(values["edge_margin"], field="edge_margin"),
                confirmation_count=_integer(
                    values["confirmation_count"],
                    field="confirmation_count",
                ),
                strong_reversal_threshold=_number(
                    values["strong_reversal_threshold"],
                    field="strong_reversal_threshold",
                ),
                liquidity_lookback_decisions=_integer(
                    values["liquidity_lookback_decisions"],
                    field="liquidity_lookback_decisions",
                ),
                liquidity_lower_quantile=_number(
                    values["liquidity_lower_quantile"],
                    field="liquidity_lower_quantile",
                ),
                liquidity_safety_multiplier=_number(
                    values["liquidity_safety_multiplier"],
                    field="liquidity_safety_multiplier",
                ),
            )
        )


def run_universal_causal_alpha_v6_research_from_paths(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV6ResearchPackage:
    from trade_rl.workflows.universal_causal_alpha_v6_stage_execution import (
        run_causal_alpha_v6_stage_entry,
    )

    return run_causal_alpha_v6_stage_entry(
        config_path=config_path,
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
        output_root=output_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research-only Causal Alpha V6")
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
    ] = run_universal_causal_alpha_v6_research_from_paths,
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
    except CausalAlphaV6StageRejected as rejection:
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
    if not isinstance(result, CausalAlphaV6ResearchPackage):
        raise TypeError("V6 runner returned an invalid research package")
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
    "CAUSAL_ALPHA_V6_RESEARCH_CONFIG_SCHEMA",
    "CausalAlphaV6ResearchConfig",
    "cli_main",
    "run_universal_causal_alpha_v6_research_from_paths",
]
