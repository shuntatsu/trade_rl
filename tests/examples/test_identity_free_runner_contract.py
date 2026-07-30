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


def test_full_runner_activates_only_symbol_disjoint_train_triplets(
    tmp_path: Path,
) -> None:
    from trade_rl.workflows.symbol_disjoint_manifest import (
        load_symbol_disjoint_manifest,
    )
    from trade_rl.workflows.symbol_disjoint_triplet_manifest import (
        load_symbol_disjoint_triplet_manifest,
    )

    namespace = _runner_namespace()
    activate = namespace["activate_symbol_triplet"]
    assert callable(activate)
    selected = activate(work_root=tmp_path, seed=31, train_slot=83)
    source = load_symbol_disjoint_manifest(tmp_path / "symbol-disjoint.json")
    manifest = load_symbol_disjoint_triplet_manifest(
        tmp_path / "symbol-triplets.json", source=source
    )

    assert len(manifest.slots_for("train")) == 84
    assert set(selected.symbols) <= set(source.train_symbols)
    assert set(selected.symbols).isdisjoint(source.validation_symbols)
    assert set(selected.symbols).isdisjoint(source.test_symbols)
