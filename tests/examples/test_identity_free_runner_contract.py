from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from trade_rl.integrations.binance import binance_multitimeframe_feature_specs

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def _runner_namespace() -> dict[str, object]:
    if str(EXAMPLE_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_ROOT))
    return vars(importlib.import_module("full_research_pipeline"))


def test_postgres_runner_uses_identity_free_native_features() -> None:
    namespace = _runner_namespace()
    validate = namespace["validate_maintained_dataset_preset"]
    assert callable(validate)
    feature_names = tuple(
        spec.name
        for spec in binance_multitimeframe_feature_specs(
            base_timeframe="15m",
            feature_timeframes=("1h", "4h", "1d"),
        )
    )
    dataset = SimpleNamespace(
        n_bars=55_392,
        symbols=("SLOT0", "SLOT1", "SLOT2"),
        feature_names=feature_names,
    )

    validate(dataset, use_postgres=True)

    assert len(feature_names) == 226
    assert not any("symbol_id" in name for name in feature_names)


def test_full_runner_identity_free_policy_observation_count_is_fixed() -> None:
    namespace = _runner_namespace()

    assert namespace["_EXPECTED_POLICY_OBSERVATIONS"] == 217_886
