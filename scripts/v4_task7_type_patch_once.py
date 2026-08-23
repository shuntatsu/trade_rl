from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/workflows/universal_causal_alpha_v4_fitting.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one match, got {count}: {old[:120]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "from types import MappingProxyType\n\nimport numpy as np\n",
    "from types import MappingProxyType\nfrom typing import TypeVar\n\nimport numpy as np\n",
)
replace_once(
    '_V4_WEIGHT_SCHEMA = "universal_causal_alpha_v4_weight_v1"\n',
    '_V4_WEIGHT_SCHEMA = "universal_causal_alpha_v4_weight_v1"\n_T = TypeVar("_T")\n',
)
replace_once(
    '''def _mapping_exact(\n    value: Mapping[str, object], *, field: str\n) -> dict[str, object]:\n    resolved = dict(value)\n    if tuple(resolved) != _V4_HORIZONS:\n        raise ValueError(f"{field} must use the canonical V4 horizon order")\n    return resolved\n''',
    '''def _mapping_exact(\n    value: Mapping[str, _T], *, field: str\n) -> dict[str, _T]:\n    resolved = dict(value)\n    if tuple(resolved) != _V4_HORIZONS:\n        raise ValueError(f"{field} must use the canonical V4 horizon order")\n    return resolved\n''',
)
replace_once(
    '''        for mapping, field in (\n            (market_models, "market_models"),\n            (residual_models, "residual_models"),\n            (direction_models, "direction_models"),\n        ):\n            if any(\n                not isinstance(model, CausalAlphaRidgeModel)\n                for model in mapping.values()\n            ):\n                raise TypeError(f"V4 {field} must contain ridge models")\n''',
    '''        for model_mapping, field_name in (\n            (market_models, "market_models"),\n            (residual_models, "residual_models"),\n            (direction_models, "direction_models"),\n        ):\n            if any(\n                not isinstance(model, CausalAlphaRidgeModel)\n                for model in model_mapping.values()\n            ):\n                raise TypeError(f"V4 {field_name} must contain ridge models")\n''',
)
replace_once(
    '''        digest_maps: dict[str, dict[str, str]] = {}\n        for field in (\n            "market_weight_digests",\n            "residual_weight_digests",\n            "direction_weight_digests",\n        ):\n            resolved = _mapping_exact(getattr(self, field), field=field)\n            typed: dict[str, str] = {}\n            for horizon, raw in resolved.items():\n                value = str(raw)\n                if len(value) != 64:\n                    raise ValueError(f"V4 {field}[{horizon}] is invalid")\n                typed[horizon] = value\n            digest_maps[field] = typed\n\n        numeric_maps: dict[str, dict[str, float]] = {}\n        for field in ("market_rmse", "residual_rmse", "direction_rmse"):\n            resolved = _mapping_exact(getattr(self, field), field=field)\n            typed_float: dict[str, float] = {}\n            for horizon, raw in resolved.items():\n                value = float(raw)\n                if not math.isfinite(value) or value < 0.0:\n                    raise ValueError(f"V4 {field}[{horizon}] must be non-negative")\n                typed_float[horizon] = value\n            numeric_maps[field] = typed_float\n''',
    '''        digest_maps: dict[str, dict[str, str]] = {}\n        for field_name in (\n            "market_weight_digests",\n            "residual_weight_digests",\n            "direction_weight_digests",\n        ):\n            resolved = _mapping_exact(getattr(self, field_name), field=field_name)\n            typed: dict[str, str] = {}\n            for horizon, raw in resolved.items():\n                digest_value = str(raw)\n                if len(digest_value) != 64:\n                    raise ValueError(f"V4 {field_name}[{horizon}] is invalid")\n                typed[horizon] = digest_value\n            digest_maps[field_name] = typed\n\n        numeric_maps: dict[str, dict[str, float]] = {}\n        for field_name in ("market_rmse", "residual_rmse", "direction_rmse"):\n            resolved = _mapping_exact(getattr(self, field_name), field=field_name)\n            typed_float: dict[str, float] = {}\n            for horizon, raw in resolved.items():\n                numeric_value = float(raw)\n                if not math.isfinite(numeric_value) or numeric_value < 0.0:\n                    raise ValueError(\n                        f"V4 {field_name}[{horizon}] must be non-negative"\n                    )\n                typed_float[horizon] = numeric_value\n            numeric_maps[field_name] = typed_float\n''',
)
replace_once(
    '''        for field, mapping in digest_maps.items():\n            object.__setattr__(self, field, MappingProxyType(mapping))\n        for field, mapping in numeric_maps.items():\n            object.__setattr__(self, field, MappingProxyType(mapping))\n''',
    '''        for field_name, digest_mapping in digest_maps.items():\n            object.__setattr__(self, field_name, MappingProxyType(digest_mapping))\n        for field_name, numeric_mapping in numeric_maps.items():\n            object.__setattr__(self, field_name, MappingProxyType(numeric_mapping))\n''',
)
path.write_text(text, encoding="utf-8")
