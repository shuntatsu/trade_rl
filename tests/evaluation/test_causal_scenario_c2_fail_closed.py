from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.trend import TrendConfig, TrendStrategy
from trade_rl.workflows.causal_scenario import conditions
from trade_rl.workflows.causal_scenario import library as library_module
from trade_rl.workflows.causal_scenario import library_artifact as artifact_module
from trade_rl.workflows.causal_scenario.conditions import (
    CausalConditionConfig,
    CausalConditionLayout,
    TrainRobustConditionNormalizer,
    build_causal_condition_layout,
    compute_raw_causal_condition,
    fit_train_condition_normalizer,
)
from trade_rl.workflows.causal_scenario.library import (
    CausalScenarioLibraryConfig,
    CausalScenarioSelection,
    FrozenCausalScenarioLibrary,
    RelativeScenarioBlock,
    build_causal_scenario_library,
    select_causal_scenarios,
)
from trade_rl.workflows.causal_scenario.library_artifact import (
    load_causal_scenario_library_artifact,
    write_causal_scenario_library_artifact,
)
from trade_rl.workflows.causal_scenario.replay import (
    CausalScenarioReplayIdentity,
    materialize_causal_scenario_dataset,
)


def _strategy() -> TrendStrategy:
    return TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96))


def _small_library(factory: Any) -> tuple[MarketDataset, FrozenCausalScenarioLibrary]:
    dataset = factory(n_bars=820)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 780),
        _strategy(),
        CausalScenarioLibraryConfig(horizon_decisions=8),
    )
    return dataset, library


def _copy_dataset(dataset: MarketDataset, **updates: Any) -> MarketDataset:
    values = {
        name: getattr(dataset, name)
        for name, field in dataset.__dataclass_fields__.items()
        if field.init and not name.startswith("_")
    }
    values["identity_payload_json"] = None
    values.update(updates)
    return MarketDataset(**values)


def test_condition_helper_and_contract_fail_closed_branches(
    market_dataset_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    for bad in (True, "x", 0.0, float("nan")):
        with pytest.raises(ValueError):
            CausalConditionConfig(volatility_hours=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shorter"):
        CausalConditionConfig(volatility_hours=48, correlation_hours=24)
    with pytest.raises(ValueError, match="schema"):
        CausalConditionConfig(schema_version="bad")
    with pytest.raises(ValueError, match="numeric"):
        conditions._readonly_float("x", object(), ndim=1)
    with pytest.raises(ValueError, match="rank"):
        conditions._readonly_float("x", np.zeros((1, 1)), ndim=1)
    with pytest.raises(ValueError, match="finite"):
        conditions._readonly_float("x", np.asarray([np.nan]), ndim=1)
    with pytest.raises(ValueError, match="boolean"):
        conditions._readonly_bool("x", np.asarray([1]), ndim=1)

    layout = build_causal_condition_layout(("BTCUSDT",))
    with pytest.raises(ValueError, match="shape"):
        CausalConditionLayout(
            symbols=layout.symbols,
            feature_names=layout.feature_names,
            continuous_mask=np.ones(1, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="schema"):
        replace(layout, schema_version="bad")
    with pytest.raises(ValueError, match="match feature_names"):
        TrainRobustConditionNormalizer(
            feature_names=layout.feature_names,
            continuous_mask=layout.continuous_mask,
            median=np.zeros(1),
            scale=np.ones(1),
            train_view_digest="b" * 64,
        )
    median = np.zeros(len(layout.feature_names))
    scale = np.ones(len(layout.feature_names))
    median[~layout.continuous_mask] = 1.0
    with pytest.raises(ValueError, match="binary normalizer"):
        TrainRobustConditionNormalizer(
            feature_names=layout.feature_names,
            continuous_mask=layout.continuous_mask,
            median=median,
            scale=scale,
            train_view_digest="b" * 64,
        )
    with pytest.raises(ValueError, match="schema"):
        TrainRobustConditionNormalizer(
            feature_names=layout.feature_names,
            continuous_mask=layout.continuous_mask,
            median=np.zeros(len(layout.feature_names)),
            scale=np.ones(len(layout.feature_names)),
            train_view_digest="b" * 64,
            schema_version="bad",
        )
    normalizer = fit_train_condition_normalizer(
        np.zeros((2, len(layout.feature_names))), layout, "b" * 64
    )
    with pytest.raises(ValueError, match="shape"):
        normalizer.transform(np.zeros(1))
    overflow = TrainRobustConditionNormalizer(
        feature_names=layout.feature_names,
        continuous_mask=layout.continuous_mask,
        median=np.zeros(len(layout.feature_names)),
        scale=np.where(layout.continuous_mask, 1e-300, 1.0),
        train_view_digest="b" * 64,
    )
    raw = np.zeros(len(layout.feature_names))
    raw[np.flatnonzero(layout.continuous_mask)[0]] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="finite"):
        overflow.transform(raw)

    dataset = market_dataset_factory()
    with pytest.raises(ValueError, match="outside"):
        compute_raw_causal_condition(dataset, True, _strategy())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="insufficient"):
        compute_raw_causal_condition(dataset, 100, _strategy())
    constant_close = np.full_like(dataset.close, 100.0)
    constant = _copy_dataset(
        dataset,
        open=constant_close,
        high=constant_close,
        low=constant_close,
        close=constant_close,
        mark_price=constant_close,
        index_price=constant_close,
    )
    vector = compute_raw_causal_condition(constant, 800, _strategy())
    corr_indices = [
        index
        for index, name in enumerate(
            build_causal_condition_layout(dataset.symbols).feature_names
        )
        if name.startswith("corr_7d:")
    ]
    np.testing.assert_array_equal(vector[corr_indices], 0.0)

    class BadHistory:
        close = np.asarray([[1.0], [0.0]])

        @staticmethod
        def lookback_index(index: int, hours: float) -> int:
            return 0

    with pytest.raises(ValueError, match="invalid close"):
        conditions._log_returns(BadHistory(), 1, 1.0)  # type: ignore[arg-type]

    monkeypatch.setattr(
        MarketDataset,
        "market_notional",
        lambda self, index, prices=None, volume=None: np.full(self.n_symbols, np.nan),
    )
    with pytest.raises(ValueError, match="invalid"):
        compute_raw_causal_condition(dataset, 800, _strategy())

    with pytest.raises(ValueError, match="shape"):
        fit_train_condition_normalizer(
            np.zeros((0, len(layout.feature_names))), layout, "b" * 64
        )
    bad_matrix = np.zeros((2, len(layout.feature_names)))
    bad_matrix[:, np.flatnonzero(~layout.continuous_mask)[0]] = 0.25
    with pytest.raises(ValueError, match="binary anchor"):
        fit_train_condition_normalizer(bad_matrix, layout, "b" * 64)


def _block_values(block: RelativeScenarioBlock) -> dict[str, Any]:
    return {
        "anchor_index": block.anchor_index,
        "source_start": block.source_start,
        "source_stop": block.source_stop,
        "elapsed_ns": block.elapsed_ns,
        "raw_condition": block.raw_condition,
        "normalized_condition": block.normalized_condition,
        "price_relatives": dict(block.price_relatives),
        "volume_relative": block.volume_relative,
        "market_notional_relative": block.market_notional_relative,
        "future_arrays": dict(block.future_arrays),
        "block_digest": block.block_digest,
        "schema_version": block.schema_version,
    }


def test_library_helper_and_block_fail_closed_branches(
    market_dataset_factory: Any,
) -> None:
    for bad in (True, "x", 0, -1):
        with pytest.raises(ValueError):
            library_module._positive_int("x", bad)
    for bad in (True, "x", -1):
        with pytest.raises(ValueError):
            library_module._non_negative_int("x", bad)
    for bad in (True, "x", 0.0, float("inf")):
        with pytest.raises(ValueError):
            library_module._finite_positive("x", bad)
    with pytest.raises(ValueError, match="array"):
        library_module._readonly_array("x", object(), dtype=np.dtype(np.float64))
    with pytest.raises(ValueError, match="rank"):
        library_module._readonly_array("x", np.zeros((1, 1)), ndim=1)
    with pytest.raises(ValueError, match="finite"):
        library_module._readonly_array("x", np.asarray([np.nan]))
    assert library_module._normal_payload((1,)) == (1,)
    assert library_module._normal_payload([1]) == [1]
    assert library_module._normal_payload({"x": 1}) == {"x": 1}

    with pytest.raises(ValueError, match="condition"):
        CausalScenarioLibraryConfig(condition="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema"):
        CausalScenarioLibraryConfig(schema_version="bad")

    dataset, library = _small_library(market_dataset_factory)
    query_index = 800
    selection = select_causal_scenarios(
        library,
        MarketDatasetView(dataset, 0, query_index + 1),
        query_index,
        _strategy(),
    )
    block = selection.blocks[0]
    values = _block_values(block)
    invalid_cases: list[tuple[str, Any, str]] = [
        ("source_start", block.source_start + 1, "immediately"),
        ("elapsed_ns", np.zeros(block.horizon_decisions, dtype=np.int64), "elapsed"),
        ("raw_condition", np.asarray([], dtype=np.float64), "condition"),
        ("volume_relative", block.volume_relative[:-1], "volume"),
        ("volume_relative", -np.ones_like(block.volume_relative), "non-negative"),
        ("schema_version", "bad", "schema"),
        ("block_digest", "f" * 64, "block_digest"),
    ]
    for name, value, message in invalid_cases:
        case = dict(values)
        case[name] = value
        with pytest.raises(ValueError, match=message):
            RelativeScenarioBlock(**case)
    case = dict(values)
    prices = dict(block.price_relatives)
    prices.pop("open")
    case["price_relatives"] = prices
    with pytest.raises(ValueError, match="keys"):
        RelativeScenarioBlock(**case)
    case = dict(values)
    prices = dict(block.price_relatives)
    prices["open"] = np.ones((1,))
    case["price_relatives"] = prices
    with pytest.raises(ValueError, match="horizon"):
        RelativeScenarioBlock(**case)
    case = dict(values)
    prices = dict(block.price_relatives)
    prices["open"] = np.ones((block.horizon_decisions, 1))
    case["price_relatives"] = prices
    with pytest.raises(ValueError, match="price relatives"):
        RelativeScenarioBlock(**case)
    case = dict(values)
    prices = dict(block.price_relatives)
    prices["dividend"] = np.ones((block.horizon_decisions, 1))
    case["price_relatives"] = prices
    with pytest.raises(ValueError, match="dividend"):
        RelativeScenarioBlock(**case)
    case = dict(values)
    future = dict(block.future_arrays)
    future.pop("features")
    case["future_arrays"] = future
    with pytest.raises(ValueError, match="keys"):
        RelativeScenarioBlock(**case)


def test_library_and_selection_fail_closed_branches(
    market_dataset_factory: Any,
) -> None:
    dataset, library = _small_library(market_dataset_factory)
    base = {
        "dataset_id": library.dataset_id,
        "train_view_digest": library.train_view_digest,
        "train_start": library.train_start,
        "train_stop": library.train_stop,
        "symbols": library.symbols,
        "feature_names": library.feature_names,
        "global_feature_names": library.global_feature_names,
        "config": library.config,
        "trend_config_payload": library.trend_config_payload,
        "layout": library.layout,
        "normalizer": library.normalizer,
        "anchor_indices": library.anchor_indices,
        "raw_conditions": library.raw_conditions,
        "normalized_conditions": library.normalized_conditions,
        "library_digest": library.library_digest,
        "schema_version": library.schema_version,
    }
    range_case = dict(base)
    range_case["train_start"] = 1
    range_case["train_stop"] = 1
    with pytest.raises(ValueError, match="train range"):
        FrozenCausalScenarioLibrary(**range_case)

    mismatched_normalizer = replace(library.normalizer, train_view_digest="f" * 64)
    for name, value, message in (
        ("config", "bad", "config"),
        ("layout", build_causal_condition_layout(("OTHER",)), "layout"),
        ("normalizer", mismatched_normalizer, "normalizer"),
        ("schema_version", "bad", "schema"),
        ("library_digest", "f" * 64, "library_digest"),
    ):
        case = dict(base)
        case[name] = value
        with pytest.raises(ValueError, match=message):
            FrozenCausalScenarioLibrary(**case)

    too_small = dict(base)
    too_small["anchor_indices"] = library.anchor_indices[:1]
    too_small["raw_conditions"] = library.raw_conditions[:1]
    too_small["normalized_conditions"] = library.normalized_conditions[:1]
    too_small["library_digest"] = ""
    with pytest.raises(ValueError, match="at least 64"):
        FrozenCausalScenarioLibrary(**too_small)

    duplicate = dict(base)
    duplicate_anchors = library.anchor_indices.copy()
    duplicate_anchors[1] = duplicate_anchors[0]
    duplicate["anchor_indices"] = duplicate_anchors
    duplicate["library_digest"] = ""
    with pytest.raises(ValueError, match="unique and ascending"):
        FrozenCausalScenarioLibrary(**duplicate)

    outside = dict(base)
    outside_anchors = library.anchor_indices.copy()
    outside_anchors[-1] = library.train_stop
    outside["anchor_indices"] = outside_anchors
    outside["library_digest"] = ""
    with pytest.raises(ValueError, match="outside"):
        FrozenCausalScenarioLibrary(**outside)

    raw_shape = dict(base)
    raw_shape["raw_conditions"] = library.raw_conditions[:-1]
    raw_shape["library_digest"] = ""
    with pytest.raises(ValueError, match="invalid shapes"):
        FrozenCausalScenarioLibrary(**raw_shape)

    normalized_shape = dict(base)
    normalized_shape["normalized_conditions"] = library.normalized_conditions[:, :-1]
    normalized_shape["library_digest"] = ""
    with pytest.raises(ValueError, match="invalid shapes"):
        FrozenCausalScenarioLibrary(**normalized_shape)

    normalized_case = dict(base)
    normalized_case["normalized_conditions"] = library.normalized_conditions + 1.0
    normalized_case["library_digest"] = ""
    with pytest.raises(ValueError, match="normalized conditions"):
        FrozenCausalScenarioLibrary(**normalized_case)

    mask_case = dict(base)
    flipped_mask = ~library.layout.continuous_mask
    mask_case["normalizer"] = TrainRobustConditionNormalizer(
        feature_names=library.layout.feature_names,
        continuous_mask=flipped_mask,
        median=np.zeros(len(library.layout.feature_names)),
        scale=np.ones(len(library.layout.feature_names)),
        train_view_digest=library.train_view_digest,
    )
    mask_case["library_digest"] = ""
    with pytest.raises(ValueError, match="normalizer"):
        FrozenCausalScenarioLibrary(**mask_case)

    trend_case = dict(base)
    trend_case["trend_config_payload"] = []
    trend_case["library_digest"] = ""
    with pytest.raises(ValueError, match="trend"):
        FrozenCausalScenarioLibrary(**trend_case)

    with pytest.raises(ValueError, match="64 complete"):
        build_causal_scenario_library(
            MarketDatasetView(dataset, 0, 740),
            _strategy(),
            CausalScenarioLibraryConfig(horizon_decisions=8),
        )

    query_index = 790
    selection = select_causal_scenarios(
        library,
        MarketDatasetView(dataset, 0, query_index + 1),
        query_index,
        _strategy(),
    )
    with pytest.raises(ValueError, match="dataset identity"):
        changed = _copy_dataset(dataset, dataset_id="f" * 64)
        select_causal_scenarios(
            library,
            MarketDatasetView(changed, 0, query_index + 1),
            query_index,
            _strategy(),
        )
    changed_features = _copy_dataset(
        dataset,
        feature_names=("other_a", "other_b"),
    )
    with pytest.raises(ValueError, match="feature schema"):
        select_causal_scenarios(
            library,
            MarketDatasetView(changed_features, 0, query_index + 1),
            query_index,
            _strategy(),
        )
    with pytest.raises(ValueError, match="trend"):
        select_causal_scenarios(
            library,
            MarketDatasetView(dataset, 0, query_index + 1),
            query_index,
            TrendStrategy(TrendConfig(fast_hours=6, base_hours=48, slow_hours=96)),
        )

    scenario = selection.scenario_set
    with pytest.raises(ValueError, match="library digest"):
        CausalScenarioSelection(
            library_digest="f" * 64,
            query_index=selection.query_index,
            query_timestamp_ns=selection.query_timestamp_ns,
            raw_query_condition=selection.raw_query_condition,
            normalized_query_condition=selection.normalized_query_condition,
            scenario_set=scenario,
            blocks=selection.blocks,
        )
    with pytest.raises(ValueError, match="scenario_set"):
        CausalScenarioSelection(
            library_digest=selection.library_digest,
            query_index=selection.query_index,
            query_timestamp_ns=selection.query_timestamp_ns,
            raw_query_condition=selection.raw_query_condition,
            normalized_query_condition=selection.normalized_query_condition,
            scenario_set=object(),  # type: ignore[arg-type]
            blocks=selection.blocks,
        )
    with pytest.raises(ValueError, match="selection blocks"):
        CausalScenarioSelection(
            library_digest=selection.library_digest,
            query_index=selection.query_index,
            query_timestamp_ns=selection.query_timestamp_ns,
            raw_query_condition=selection.raw_query_condition,
            normalized_query_condition=selection.normalized_query_condition,
            scenario_set=scenario,
            blocks=(*selection.blocks[:-1], object()),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="conditions"):
        CausalScenarioSelection(
            library_digest=selection.library_digest,
            query_index=selection.query_index,
            query_timestamp_ns=selection.query_timestamp_ns,
            raw_query_condition=np.zeros(1),
            normalized_query_condition=np.zeros(2),
            scenario_set=scenario,
            blocks=selection.blocks,
        )
    with pytest.raises(ValueError, match="schema"):
        replace(selection, schema_version="bad")
    with pytest.raises(ValueError, match="selection_digest"):
        replace(selection, selection_digest="f" * 64)

    bad_anchors = replace(scenario, anchor_indices=scenario.anchor_indices[::-1])
    with pytest.raises(ValueError, match="anchors"):
        replace(selection, scenario_set=bad_anchors, selection_digest="")

    bad_distances = replace(scenario, distances=scenario.distances + 1.0)
    with pytest.raises(ValueError, match="distances"):
        replace(selection, scenario_set=bad_distances, selection_digest="")

    bad_anchor_conditions = replace(
        scenario, anchor_conditions=scenario.anchor_conditions + 1.0
    )
    with pytest.raises(ValueError, match="anchor conditions"):
        replace(
            selection,
            scenario_set=bad_anchor_conditions,
            selection_digest="",
        )

    bad_query = replace(scenario, query_condition=scenario.query_condition + 1.0)
    with pytest.raises(ValueError, match="query condition"):
        replace(selection, scenario_set=bad_query, selection_digest="")

    changed_ids = ("e" * 64, *scenario.scenario_ids[1:])
    bad_ids = replace(scenario, scenario_ids=changed_ids)
    with pytest.raises(ValueError, match="scenario IDs"):
        replace(selection, scenario_set=bad_ids, selection_digest="")


def _rewrite_manifest(root: Path, mutate: Any) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    base = dict(manifest)
    base.pop("artifact_digest", None)
    manifest["artifact_digest"] = content_digest(base)
    path.write_bytes(canonical_json_bytes(manifest))


def test_artifact_helpers_and_manifest_fail_closed_branches(
    tmp_path: Path,
    market_dataset_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="mapping"):
        artifact_module._mapping([], field="x")
    with pytest.raises(ValueError, match="sequence"):
        artifact_module._sequence("x", field="x")
    with pytest.raises(ValueError, match="string"):
        artifact_module._string("", field="x")
    with pytest.raises(ValueError, match="integer"):
        artifact_module._integer(True, field="x")
    with pytest.raises(ValueError, match="number"):
        artifact_module._float(True, field="x")
    with pytest.raises(ValueError, match="number"):
        artifact_module._float(float("nan"), field="x")
    with pytest.raises(ValueError, match="condition config"):
        artifact_module._condition_from_payload({})
    with pytest.raises(ValueError, match="library config"):
        artifact_module._config_from_payload({})

    _, library = _small_library(market_dataset_factory)
    with pytest.raises(FileNotFoundError, match="missing"):
        load_causal_scenario_library_artifact(tmp_path / "missing")

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "manifest.json").write_text("placeholder", encoding="utf-8")
    (allowed / "arrays.npz").write_bytes(b"placeholder")
    write_causal_scenario_library_artifact(allowed, library)

    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    (symlink_root / "target").write_text("x", encoding="utf-8")
    (symlink_root / "manifest.json").symlink_to(symlink_root / "target")
    with pytest.raises(ValueError, match="invalid entries"):
        write_causal_scenario_library_artifact(symlink_root, library)

    directory_root = tmp_path / "directory-root"
    directory_root.mkdir()
    (directory_root / "manifest.json").mkdir()
    with pytest.raises(ValueError, match="invalid entries"):
        write_causal_scenario_library_artifact(directory_root, library)

    expected_names = artifact_module._expected_array_names
    monkeypatch.setattr(
        artifact_module,
        "_expected_array_names",
        lambda: (*expected_names(), "extra"),
    )
    with pytest.raises(RuntimeError, match="array closure"):
        artifact_module._library_arrays(library)
    monkeypatch.setattr(artifact_module, "_expected_array_names", expected_names)

    invalid_root = tmp_path / "file-root"
    invalid_root.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        write_causal_scenario_library_artifact(invalid_root, library)
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "extra").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid entries"):
        write_causal_scenario_library_artifact(dirty, library)

    variants = [
        ("field closure", lambda m: m.pop("symbols")),
        ("arrays file", lambda m: m.__setitem__("arrays_file", "bad.npz")),
        ("unsupported", lambda m: m.__setitem__("schema_version", "bad")),
    ]
    for index, (message, mutate) in enumerate(variants):
        root = tmp_path / f"manifest-{index}"
        write_causal_scenario_library_artifact(root, library)
        _rewrite_manifest(root, mutate)
        with pytest.raises(ValueError, match=message):
            load_causal_scenario_library_artifact(root)


def test_artifact_array_fail_closed_branches(
    tmp_path: Path,
    market_dataset_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, library = _small_library(market_dataset_factory)
    root = tmp_path / "artifact"
    write_causal_scenario_library_artifact(root, library)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = (root / "arrays.npz").read_bytes()
    metadata = manifest["array_metadata"]

    bad_meta = dict(metadata)
    bad_meta.pop("anchor_indices")
    with pytest.raises(ValueError, match="metadata closure"):
        artifact_module._load_arrays(payload, bad_meta)
    bad_meta = {name: dict(value) for name, value in metadata.items()}
    bad_meta["anchor_indices"] = {"dtype": "<i8"}
    with pytest.raises(ValueError, match="metadata invalid"):
        artifact_module._load_arrays(payload, bad_meta)
    bad_meta = {name: dict(value) for name, value in metadata.items()}
    bad_meta["anchor_indices"]["dtype"] = "<f8"
    with pytest.raises(ValueError, match="dtype"):
        artifact_module._load_arrays(payload, bad_meta)

    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = dict(arrays)
    missing.pop("anchor_indices")
    with pytest.raises(ValueError, match="name closure"):
        artifact_module._load_arrays(
            artifact_module._deterministic_npz(missing), metadata
        )

    root_count = tmp_path / "count"
    write_causal_scenario_library_artifact(root_count, library)
    _rewrite_manifest(
        root_count, lambda m: m.__setitem__("anchor_count", m["anchor_count"] + 1)
    )
    with pytest.raises(ValueError, match="anchor count"):
        load_causal_scenario_library_artifact(root_count)

    valid_manifest, valid_payload = artifact_module._load_manifest(root)
    valid_arrays = artifact_module._load_arrays(
        valid_payload, valid_manifest["array_metadata"]
    )
    mismatched_arrays = dict(valid_arrays)
    mismatched_arrays["raw_conditions"] = valid_arrays["raw_conditions"][:-1]
    monkeypatch.setattr(
        artifact_module,
        "_load_manifest",
        lambda ignored: (valid_manifest, valid_payload),
    )
    monkeypatch.setattr(
        artifact_module,
        "_load_arrays",
        lambda _ignored_payload, _ignored_metadata: mismatched_arrays,
    )
    with pytest.raises(ValueError, match="raw condition array shape"):
        load_causal_scenario_library_artifact(root)

    mismatched_normalized = dict(valid_arrays)
    mismatched_normalized["normalized_conditions"] = valid_arrays[
        "normalized_conditions"
    ][:-1]
    monkeypatch.setattr(
        artifact_module,
        "_load_arrays",
        lambda _ignored_payload, _ignored_metadata: mismatched_normalized,
    )
    with pytest.raises(ValueError, match="normalized condition array shape"):
        load_causal_scenario_library_artifact(root)


def test_replay_remaining_fail_closed_branches(
    market_dataset_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = market_dataset_factory(n_bars=1_000)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 780),
        _strategy(),
        CausalScenarioLibraryConfig(horizon_decisions=8),
    )
    query_index = 900
    prefix = MarketDatasetView(dataset, 0, query_index + 1)
    selection = select_causal_scenarios(library, prefix, query_index, _strategy())
    with pytest.raises(ValueError):
        materialize_causal_scenario_dataset(
            library, selection, prefix, selected_rank=-1
        )
    with pytest.raises(ValueError, match="schema"):
        CausalScenarioReplayIdentity(
            query_dataset_id=dataset.dataset_id,
            library_digest=library.library_digest,
            selection_digest=selection.selection_digest,
            block_digest=selection.blocks[0].block_digest,
            scenario_id=selection.scenario_set.scenario_ids[0],
            query_index=query_index,
            selected_rank=0,
            schema_version="bad",
        )

    changed_selection = object.__new__(CausalScenarioSelection)
    for field in selection.__dataclass_fields__:
        object.__setattr__(changed_selection, field, getattr(selection, field))
    object.__setattr__(changed_selection, "library_digest", "f" * 64)
    with pytest.raises(ValueError, match="selection library"):
        materialize_causal_scenario_dataset(
            library, changed_selection, prefix, selected_rank=0
        )

    changed_dataset = _copy_dataset(dataset, dataset_id="f" * 64)
    with pytest.raises(ValueError, match="query dataset"):
        materialize_causal_scenario_dataset(
            library,
            selection,
            MarketDatasetView(changed_dataset, 0, query_index + 1),
            selected_rank=0,
        )

    missing_anchor = object.__new__(RelativeScenarioBlock)
    for field, value in _block_values(selection.blocks[0]).items():
        object.__setattr__(missing_anchor, field, value)
    object.__setattr__(missing_anchor, "anchor_index", library.train_stop + 1)
    missing_anchor_selection = object.__new__(CausalScenarioSelection)
    for field in selection.__dataclass_fields__:
        object.__setattr__(missing_anchor_selection, field, getattr(selection, field))
    object.__setattr__(
        missing_anchor_selection,
        "blocks",
        (missing_anchor,) + selection.blocks[1:],
    )
    with pytest.raises(ValueError, match="not contained"):
        materialize_causal_scenario_dataset(
            library, missing_anchor_selection, prefix, selected_rank=0
        )

    source = selection.blocks[0]
    missing_valid = RelativeScenarioBlock(
        anchor_index=library.train_stop + 1,
        source_start=library.train_stop + 2,
        source_stop=library.train_stop + 2 + source.horizon_decisions,
        elapsed_ns=source.elapsed_ns,
        raw_condition=source.raw_condition,
        normalized_condition=source.normalized_condition,
        price_relatives=dict(source.price_relatives),
        volume_relative=source.volume_relative,
        market_notional_relative=source.market_notional_relative,
        future_arrays=dict(source.future_arrays),
    )
    missing_valid_selection = object.__new__(CausalScenarioSelection)
    for field in selection.__dataclass_fields__:
        object.__setattr__(missing_valid_selection, field, getattr(selection, field))
    object.__setattr__(
        missing_valid_selection,
        "blocks",
        (missing_valid,) + selection.blocks[1:],
    )
    with pytest.raises(ValueError, match="not contained"):
        materialize_causal_scenario_dataset(
            library, missing_valid_selection, prefix, selected_rank=0
        )

    altered_notional = source.market_notional_relative.copy()
    altered_notional[0, 0] += 0.25
    altered_valid = RelativeScenarioBlock(
        anchor_index=source.anchor_index,
        source_start=source.source_start,
        source_stop=source.source_stop,
        elapsed_ns=source.elapsed_ns,
        raw_condition=source.raw_condition,
        normalized_condition=source.normalized_condition,
        price_relatives=dict(source.price_relatives),
        volume_relative=source.volume_relative,
        market_notional_relative=altered_notional,
        future_arrays=dict(source.future_arrays),
    )
    altered_selection = object.__new__(CausalScenarioSelection)
    for field in selection.__dataclass_fields__:
        object.__setattr__(altered_selection, field, getattr(selection, field))
    object.__setattr__(
        altered_selection, "blocks", (altered_valid,) + selection.blocks[1:]
    )
    with pytest.raises(ValueError, match="not contained"):
        materialize_causal_scenario_dataset(
            library, altered_selection, prefix, selected_rank=0
        )

    alien = object.__new__(RelativeScenarioBlock)
    for field, value in _block_values(selection.blocks[0]).items():
        object.__setattr__(alien, field, value)
    object.__setattr__(alien, "block_digest", "e" * 64)
    alien_selection = object.__new__(CausalScenarioSelection)
    for field in selection.__dataclass_fields__:
        object.__setattr__(alien_selection, field, getattr(selection, field))
    object.__setattr__(alien_selection, "blocks", (alien,) + selection.blocks[1:])
    with pytest.raises(ValueError, match="not contained"):
        materialize_causal_scenario_dataset(
            library, alien_selection, prefix, selected_rank=0
        )

    bad_notional = selection.blocks[0].market_notional_relative.copy()
    bad_notional += 1.0
    object.__setattr__(selection.blocks[0], "market_notional_relative", bad_notional)
    with pytest.raises(ValueError, match="not contained"):
        materialize_causal_scenario_dataset(library, selection, prefix, selected_rank=0)

    fresh_selection = select_causal_scenarios(library, prefix, query_index, _strategy())
    original_market_notional = MarketDataset.market_notional
    calls = 0

    def inconsistent_notional(
        self: MarketDataset,
        index: int,
        prices: np.ndarray | None = None,
        volume: np.ndarray | None = None,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        result = original_market_notional(self, index, prices=prices, volume=volume)
        if calls == library.config.horizon_decisions + 2:
            result = result.copy()
            result[0] *= 2.0
        return result

    monkeypatch.setattr(MarketDataset, "market_notional", inconsistent_notional)
    with pytest.raises(ValueError, match="market notional"):
        materialize_causal_scenario_dataset(
            library, fresh_selection, prefix, selected_rank=0
        )
