"""Build complete immutable serving-bundle candidates from trained artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mars_lite.serving.bundle import build_manifest, load_bundle


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def create_candidate_bundle(
    *,
    destination: str | Path,
    model_source: str | Path,
    version: str,
    git_sha: str,
    symbols: Sequence[str],
    feature_names: Sequence[str],
    global_feature_names: Sequence[str],
    feature_norm: str,
    feature_mask: Sequence[bool] | None,
    observation_dim: int,
    observation_schema_version: int,
    post_processor: Mapping[str, Any],
    run_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    pre_trade: Mapping[str, Any],
    rank_window: int = 250,
    rank_min_periods: int = 40,
) -> Path:
    """Create and validate a single-model or ensemble candidate directory."""
    if run_config.get("observation_progress_mode") != "zero":
        raise ValueError(
            "observation_progress_mode must be 'zero' for serving-compatible models"
        )
    if not version or not git_sha:
        raise ValueError("version and git_sha are required")
    ordered_symbols = tuple(symbols)
    ordered_features = tuple(feature_names)
    ordered_globals = tuple(global_feature_names)
    if not ordered_symbols or len(set(ordered_symbols)) != len(ordered_symbols):
        raise ValueError("symbols must be unique and non-empty")
    if not ordered_features or len(set(ordered_features)) != len(ordered_features):
        raise ValueError("feature_names must be unique and non-empty")
    if len(set(ordered_globals)) != len(ordered_globals):
        raise ValueError("global_feature_names must be unique")
    if feature_norm not in {"none", "rank_gauss"}:
        raise ValueError("feature_norm must be none or rank_gauss")
    if feature_mask is not None:
        if len(feature_mask) != len(ordered_features) or not all(
            isinstance(item, bool) for item in feature_mask
        ):
            raise ValueError("feature_mask must match feature_names")
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if observation_schema_version != 1:
        raise ValueError("unsupported observation_schema_version")

    source = Path(model_source)
    if not source.exists():
        raise FileNotFoundError(source)
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"candidate destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        if source.is_dir():
            shutil.copytree(source, root / "ensemble")
            model_kind = "ensemble"
        else:
            shutil.copy2(source, root / "model.zip")
            model_kind = "single"

        _write_json(
            root / "metadata.json",
            {
                "schema_version": 1,
                "model_version": version,
                "git_sha": git_sha,
                "model_kind": model_kind,
                "symbols": list(ordered_symbols),
                "observation_schema_version": observation_schema_version,
                "observation_dim": observation_dim,
                "observation_progress_mode": "zero",
                "serving_progress": 0.0,
                "post_processor": dict(post_processor),
                "run_config": dict(run_config),
                "metrics": dict(metrics),
            },
        )
        _write_json(
            root / "preprocessing.json",
            {
                "feature_names": list(ordered_features),
                "global_feature_names": list(ordered_globals),
                "feature_norm": feature_norm,
                "feature_mask": (
                    list(feature_mask) if feature_mask is not None else None
                ),
                "post_mask_dim": len(ordered_features),
                "rank_window": rank_window,
                "rank_min_periods": rank_min_periods,
            },
        )
        _write_json(
            root / "risk.json",
            {"guardrails": dict(guardrails), "pre_trade": dict(pre_trade)},
        )
        build_manifest(root)
        load_bundle(root)
        return root
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
