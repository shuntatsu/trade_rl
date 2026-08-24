"""Strict config and CLI boundary for research-only Causal Alpha V5."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v5 import CausalAlphaV5CalibrationConfig
from trade_rl.workflows.universal_causal_alpha_v5_pipeline import (
    CausalAlphaV5ResearchPackage,
    CausalAlphaV5StageRejected,
)

CAUSAL_ALPHA_V5_RESEARCH_CONFIG_SCHEMA: Final = (
    "universal_causal_alpha_v5_research_config_v1"
)
_ROOT_KEYS: Final = ("schema_version", "calibration")
_CALIBRATION_KEYS: Final = (
    "calibration_fraction",
    "forward_block_count",
    "ridge_strength",
    "minimum_pooled_support",
    "minimum_symbol_support",
    "minimum_selective_confidence",
    "minimum_active_coverage",
    "minimum_scope_active_fraction",
    "minimum_scope_active_count",
    "execution_cost_multiplier",
    "edge_margin",
    "epsilon",
)


def _ordered_mapping(
    value: object, *, keys: tuple[str, ...], field: str
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


@dataclass(frozen=True, slots=True)
class CausalAlphaV5ResearchConfig:
    calibration: CausalAlphaV5CalibrationConfig
    epsilon: float = 1e-12
    schema_version: str = CAUSAL_ALPHA_V5_RESEARCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, CausalAlphaV5CalibrationConfig):
            raise TypeError("V5 research calibration config is invalid")
        if self.epsilon != 1e-12:
            raise ValueError("V5 research epsilon must remain 1e-12")
        if self.schema_version != CAUSAL_ALPHA_V5_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V5 research config schema")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "calibration_digest": self.calibration.digest,
                "epsilon": self.epsilon,
                "schema_version": self.schema_version,
            }
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CausalAlphaV5ResearchConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("V5 research config JSON is invalid") from error
        root = _ordered_mapping(raw, keys=_ROOT_KEYS, field="V5 research config")
        if root["schema_version"] != CAUSAL_ALPHA_V5_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported V5 research config schema")
        values = _ordered_mapping(
            root["calibration"], keys=_CALIBRATION_KEYS, field="V5 calibration config"
        )
        return cls(
            calibration=CausalAlphaV5CalibrationConfig(
                calibration_fraction=_number(
                    values["calibration_fraction"], field="calibration_fraction"
                ),
                forward_block_count=_integer(
                    values["forward_block_count"], field="forward_block_count"
                ),
                ridge_strength=_number(
                    values["ridge_strength"], field="ridge_strength"
                ),
                minimum_pooled_support=_integer(
                    values["minimum_pooled_support"], field="minimum_pooled_support"
                ),
                minimum_symbol_support=_integer(
                    values["minimum_symbol_support"], field="minimum_symbol_support"
                ),
                minimum_selective_confidence=_number(
                    values["minimum_selective_confidence"],
                    field="minimum_selective_confidence",
                ),
                minimum_active_coverage=_number(
                    values["minimum_active_coverage"], field="minimum_active_coverage"
                ),
                minimum_scope_active_fraction=_number(
                    values["minimum_scope_active_fraction"],
                    field="minimum_scope_active_fraction",
                ),
                minimum_scope_active_count=_integer(
                    values["minimum_scope_active_count"],
                    field="minimum_scope_active_count",
                ),
                execution_cost_multiplier=_number(
                    values["execution_cost_multiplier"],
                    field="execution_cost_multiplier",
                ),
                edge_margin=_number(values["edge_margin"], field="edge_margin"),
            ),
            epsilon=_number(values["epsilon"], field="epsilon"),
        )


def run_universal_causal_alpha_v5_research_from_paths(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV5ResearchPackage:
    from trade_rl.workflows.universal_causal_alpha_v5_stage_execution import (
        run_causal_alpha_v5_stage_entry,
    )

    return run_causal_alpha_v5_stage_entry(
        config_path=config_path,
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
        output_root=output_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research-only Causal Alpha V5")
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
    ] = run_universal_causal_alpha_v5_research_from_paths,
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
    except CausalAlphaV5StageRejected as rejection:
        codes = {"signal": 2, "selection": 3, "admission": 4, "calibration": 5}
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
        return codes[rejection.stage]
    if not isinstance(result, CausalAlphaV5ResearchPackage):
        raise TypeError("V5 runner returned an invalid research package")
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
    "CAUSAL_ALPHA_V5_RESEARCH_CONFIG_SCHEMA",
    "CausalAlphaV5ResearchConfig",
    "cli_main",
    "run_universal_causal_alpha_v5_research_from_paths",
]
