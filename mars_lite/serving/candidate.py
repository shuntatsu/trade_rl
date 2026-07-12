"""Build complete immutable serving-bundle candidates from trained artifacts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mars_lite.pipeline.release_eligibility import ReleaseEligibility
from mars_lite.pipeline.release_risk import ReleaseRiskPolicy
from mars_lite.serving.bundle import build_manifest, load_bundle

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,49}$")
_GIT_SHA_RE = re.compile(r"^[a-fA-F0-9]{40}$")
_ACTION_SCHEMAS = {"direct_weights_v1", "baseline_residual_v1"}
_RESIDUAL_POLICY_MODES = {"ppo_residual_ensemble", "baseline_only"}


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
    model_source: str | Path | None,
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
    risk_policy: ReleaseRiskPolicy,
    release_eligibility: ReleaseEligibility,
    rank_window: int = 250,
    rank_min_periods: int = 40,
    action_schema: str = "direct_weights_v1",
    policy_mode: str | None = None,
    residual_alpha_source: str | Path | None = None,
    trend_family_config: Mapping[str, Any] | None = None,
    composer_config: Mapping[str, Any] | None = None,
    shadow_baseline: str = "base_trend_v2",
) -> Path:
    """Create and validate a release-eligible model bundle candidate."""

    if not release_eligibility.eligible:
        raise ValueError("ineligible research run cannot create a release candidate")
    if run_config.get("observation_progress_mode") != "zero":
        raise ValueError(
            "observation_progress_mode must be 'zero' for serving-compatible models"
        )
    if not _VERSION_RE.fullmatch(version):
        raise ValueError("version must be a valid identifier of at most 50 characters")
    if not _GIT_SHA_RE.fullmatch(git_sha):
        raise ValueError("git_sha must be a 40-character hexadecimal commit hash")
    if action_schema not in _ACTION_SCHEMAS:
        raise ValueError(f"unsupported action_schema: {action_schema}")

    ordered_symbols = tuple(symbols)
    ordered_features = tuple(feature_names)
    ordered_globals = tuple(global_feature_names)
    if not ordered_symbols or len(set(ordered_symbols)) != len(ordered_symbols):
        raise ValueError("symbols must be unique and non-empty")
    if set(risk_policy.symbol_liquidity_caps) != set(ordered_symbols):
        raise ValueError(
            "risk policy liquidity caps must exactly match candidate symbols"
        )
    if set(risk_policy.forbidden_symbols) - set(ordered_symbols):
        raise ValueError(
            "risk policy forbidden symbols must belong to candidate symbols"
        )
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
    expected_observation_schema = 2 if action_schema == "baseline_residual_v1" else 1
    if observation_schema_version != expected_observation_schema:
        raise ValueError(
            f"{action_schema} requires observation_schema_version "
            f"{expected_observation_schema}"
        )
    if rank_window <= 0 or rank_min_periods <= 0 or rank_min_periods > rank_window:
        raise ValueError("rank normalization window settings are invalid")

    source = Path(model_source) if model_source is not None else None
    alpha_source = (
        Path(residual_alpha_source) if residual_alpha_source is not None else None
    )
    if action_schema == "baseline_residual_v1":
        policy_mode = policy_mode or (
            "baseline_only" if source is None else "ppo_residual_ensemble"
        )
        if policy_mode not in _RESIDUAL_POLICY_MODES:
            raise ValueError(f"unsupported residual policy_mode: {policy_mode}")
        if alpha_source is None or not alpha_source.is_file():
            raise ValueError("baseline_residual_v1 requires residual_alpha_source")
        if policy_mode == "baseline_only":
            if source is not None:
                raise ValueError("baseline_only must not include a policy model")
            model_kind = "baseline_only"
        else:
            if source is None or not source.is_dir():
                raise ValueError("ppo_residual_ensemble requires an ensemble directory")
            if not any(source.glob("seed_*.zip")):
                raise ValueError(
                    "residual ensemble source requires at least one seed_*.zip file"
                )
            model_kind = "ensemble"
    else:
        if source is None or not source.exists():
            raise FileNotFoundError(source or "missing direct model_source")
        if source.is_dir() and not any(source.glob("seed_*.zip")):
            raise ValueError(
                "ensemble model source requires at least one seed_*.zip file"
            )
        model_kind = "ensemble" if source.is_dir() else "single"
        policy_mode = policy_mode or (
            "direct_ensemble" if model_kind == "ensemble" else "direct_single"
        )

    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"candidate destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        if model_kind == "ensemble":
            assert source is not None
            shutil.copytree(source, root / "ensemble")
        elif model_kind == "single":
            assert source is not None
            shutil.copy2(source, root / "model.zip")
        if alpha_source is not None:
            shutil.copy2(alpha_source, root / "residual_alpha.json")

        metadata: dict[str, Any] = {
            "schema_version": 1,
            "model_version": version,
            "git_sha": git_sha,
            "model_kind": model_kind,
            "policy_mode": policy_mode,
            "action_schema": action_schema,
            "symbols": list(ordered_symbols),
            "observation_schema_version": observation_schema_version,
            "observation_dim": observation_dim,
            "observation_progress_mode": "zero",
            "serving_progress": 0.0,
            "post_processor": dict(post_processor),
            "run_config": dict(run_config),
            "metrics": dict(metrics),
            "release_eligibility": release_eligibility.to_dict(),
        }
        if action_schema == "baseline_residual_v1":
            metadata.update(
                {
                    "trend_family": dict(trend_family_config or {}),
                    "composer": dict(composer_config or {}),
                    "shadow_baseline": shadow_baseline,
                    "residual_alpha_file": "residual_alpha.json",
                }
            )
        _write_json(root / "metadata.json", metadata)
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
            {"guardrails": dict(guardrails), "pre_trade": risk_policy.to_dict()},
        )
        build_manifest(root)
        load_bundle(root)
        return root
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
