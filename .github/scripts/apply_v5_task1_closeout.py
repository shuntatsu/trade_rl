from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _write(relative: str, content: str) -> None:
    (ROOT / relative).write_text(content, encoding="utf-8", newline="\n")


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def _patch_production() -> None:
    path = "trade_rl/learning/causal_alpha_v5.py"
    source = _read(path)
    source = _replace_once(
        source,
        '''        pooled_support = sum(count for _, count in support)
        if pooled_support < self.config.minimum_pooled_support:
            raise ValueError("V5 calibration pooled support is insufficient")

        block_support = tuple(self.calibration_block_support)
''',
        '''        pooled_support = sum(count for _, count in support)
        if pooled_support < self.config.minimum_pooled_support:
            raise ValueError("V5 calibration pooled support is insufficient")
        if self.model.sample_count != pooled_support:
            raise ValueError(
                "V5 calibration model sample_count must match pooled support"
            )

        block_support = tuple(self.calibration_block_support)
''',
        label="production model-support check",
    )
    source = _replace_once(
        source,
        '''    actionable_mask: object,
    calibration_fit: CausalAlphaV5CalibrationFit,
    slow_direction_override: object | None = None,
) -> CausalAlphaV5SelectiveForecast:
''',
        '''    actionable_mask: object,
    calibration_fit: CausalAlphaV5CalibrationFit,
) -> CausalAlphaV5SelectiveForecast:
''',
        label="production function signature",
    )
    source = _replace_once(
        source,
        '''    if slow_direction_override is None:
        raw_direction = 0.5 * (
            np.asarray(directions["24h"], dtype=np.float64)
            + np.asarray(directions["72h"], dtype=np.float64)
        )
    else:
        raw_direction = _aligned_vector(
            slow_direction_override,
            rows=rows,
            dtype=np.float64,
            field="V5 slow direction override",
        )
''',
        '''    raw_direction = 0.5 * (
        np.asarray(directions["24h"], dtype=np.float64)
        + np.asarray(directions["72h"], dtype=np.float64)
    )
''',
        label="production direction source",
    )
    _write(path, source)


def _patch_tests() -> None:
    path = "tests/learning/test_causal_alpha_v5_calibration.py"
    source = _read(path)
    source = _replace_once(
        source,
        "import math\n",
        "import inspect\nimport json\nimport math\nfrom pathlib import Path\n",
        label="test imports",
    )
    source = _replace_once(
        source,
        '''def _ridge_model(*, intercept: float = 0.0) -> CausalAlphaRidgeModel:
''',
        '''def _ridge_model(
    *,
    intercept: float = 0.0,
    sample_count: int = 288,
) -> CausalAlphaRidgeModel:
''',
        label="test ridge helper signature",
    )
    source = _replace_once(
        source,
        "        sample_count=256,\n",
        "        sample_count=sample_count,\n",
        label="test ridge sample count",
    )
    source = _replace_once(
        source,
        "        eligible_indices=np.arange(256, dtype=np.int64),\n",
        "        eligible_indices=np.arange(sample_count, dtype=np.int64),\n",
        label="test ridge eligible indices",
    )
    source = _replace_once(
        source,
        '''def _forecast(*, rows: int = 3):
''',
        '''def _forecast(
    *,
    rows: int = 3,
    direction_24h: float = 0.8,
    direction_72h: float = 0.6,
):
''',
        label="test forecast helper signature",
    )
    source = _replace_once(
        source,
        '''        "24h": np.full(rows, 0.8),
        "72h": np.full(rows, 0.6),
''',
        '''        "24h": np.full(rows, direction_24h),
        "72h": np.full(rows, direction_72h),
''',
        label="test forecast directions",
    )
    anchor = '''def test_v5_calibration_fit_rejects_insufficient_symbol_support() -> None:
'''
    addition = '''def test_v5_calibration_fit_rejects_model_support_mismatch() -> None:
    fit = _fit()

    with pytest.raises(ValueError, match="model sample_count"):
        CausalAlphaV5CalibrationFit(
            v4_fit_digest=fit.v4_fit_digest,
            v4_fit_config_digest=fit.v4_fit_config_digest,
            v4_sample_scope_digest=fit.v4_sample_scope_digest,
            calibration_start=fit.calibration_start,
            train_stop=fit.train_stop,
            model=_ridge_model(sample_count=287),
            forward_model_digests=fit.forward_model_digests,
            forward_residual_digests=fit.forward_residual_digests,
            final_weight_digest=fit.final_weight_digest,
            forward_weight_digests=fit.forward_weight_digests,
            per_symbol_support=fit.per_symbol_support,
            calibration_block_support=fit.calibration_block_support,
            forward_block_symbol_counts=fit.forward_block_symbol_counts,
            calibration_residual_rmse=fit.calibration_residual_rmse,
            direction_score_rmse=fit.direction_score_rmse,
            config=fit.config,
        )


def test_selective_forecast_has_no_direction_override_escape_hatch() -> None:
    parameters = inspect.signature(
        build_causal_alpha_v5_selective_forecast
    ).parameters

    assert "slow_direction_override" not in parameters


'''
    source = _replace_once(
        source,
        anchor,
        addition + anchor,
        label="test contract insertion",
    )
    source = _replace_once(
        source,
        "    forecast = _forecast(rows=2)\n",
        (
            "    forecast = _forecast(\n"
            "        rows=2,\n"
            "        direction_24h=-1.0,\n"
            "        direction_72h=-1.0,\n"
            "    )\n"
        ),
        label="test disagreement forecast",
    )
    source = _replace_once(
        source,
        "        calibration_fit=fit,\n        slow_direction_override=np.asarray([-1.0, 1.0]),\n",
        "        calibration_fit=fit,\n",
        label="test override removal",
    )
    if "test_v5_calibration_config_rejects_boolean_integer_fields" in source:
        raise SystemExit("test closeout block already exists")
    source += '''


@pytest.mark.parametrize(
    "field",
    ("forward_block_count", "minimum_pooled_support", "minimum_symbol_support"),
)
def test_v5_calibration_config_rejects_boolean_integer_fields(field: str) -> None:
    with pytest.raises(ValueError):
        CausalAlphaV5CalibrationConfig(**{field: True})


def test_v5_example_json_freezes_task1_hypothesis() -> None:
    payload = json.loads(
        Path("examples/binance/universal-causal-alpha-v5-research.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload == {
        "schema_version": "universal_causal_alpha_v5_research_config_v1",
        "calibration": {
            "calibration_fraction": 0.2,
            "forward_block_count": 4,
            "ridge_strength": 1.0,
            "minimum_pooled_support": 256,
            "minimum_symbol_support": 16,
            "minimum_selective_confidence": 1.0,
            "minimum_active_coverage": 0.25,
            "minimum_scope_active_fraction": 0.2,
            "minimum_scope_active_count": 3,
            "execution_cost_multiplier": 1.5,
            "edge_margin": 0.001,
            "epsilon": 1e-12,
        },
    }


def test_learning_package_exports_v5_contracts() -> None:
    import trade_rl.learning as learning

    for name in (
        "CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES",
        "CausalAlphaV5CalibrationConfig",
        "CausalAlphaV5CalibrationFit",
        "CausalAlphaV5SelectiveForecast",
        "V5SelectiveState",
        "build_causal_alpha_v5_selective_forecast",
    ):
        assert getattr(learning, name) is not None
'''
    _write(path, source)


def _patch_exports() -> None:
    path = "trade_rl/learning/__init__.py"
    source = _read(path)
    import_anchor = '''from trade_rl.learning.evaluation import (
'''
    imports = '''from trade_rl.learning.causal_alpha_v5 import (
    CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV5CalibrationConfig,
    CausalAlphaV5CalibrationFit,
    CausalAlphaV5SelectiveForecast,
    V5SelectiveState,
    build_causal_alpha_v5_selective_forecast,
)
'''
    source = _replace_once(
        source,
        import_anchor,
        imports + import_anchor,
        label="learning export imports",
    )
    list_anchor = '''__all__ = [
'''
    export_lines = '''__all__ = [
    "CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES",
    "CausalAlphaV5CalibrationConfig",
    "CausalAlphaV5CalibrationFit",
    "CausalAlphaV5SelectiveForecast",
    "V5SelectiveState",
    "build_causal_alpha_v5_selective_forecast",
'''
    source = _replace_once(
        source,
        list_anchor,
        export_lines,
        label="learning export list",
    )
    _write(path, source)


def _write_config() -> None:
    path = ROOT / "examples/binance/universal-causal-alpha-v5-research.json"
    if path.exists():
        raise SystemExit("V5 example config unexpectedly already exists")
    payload = {
        "schema_version": "universal_causal_alpha_v5_research_config_v1",
        "calibration": {
            "calibration_fraction": 0.2,
            "forward_block_count": 4,
            "ridge_strength": 1.0,
            "minimum_pooled_support": 256,
            "minimum_symbol_support": 16,
            "minimum_selective_confidence": 1.0,
            "minimum_active_coverage": 0.25,
            "minimum_scope_active_fraction": 0.2,
            "minimum_scope_active_count": 3,
            "execution_cost_multiplier": 1.5,
            "edge_margin": 0.001,
            "epsilon": 1e-12,
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    _patch_production()
    _patch_tests()
    _patch_exports()
    _write_config()


if __name__ == "__main__":
    main()
