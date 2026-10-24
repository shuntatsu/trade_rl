from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data import (
    MarketCalendarKind,
    MarketDataset,
    publish_market_dataset_artifact,
)
from trade_rl.evaluation.runs import (
    RidgeArtifactComparisonConfig,
    compare_ridge_artifact_shared_cash,
    load_ridge_comparison_artifact,
    publish_ridge_comparison_artifact,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.forecasts import (
    RidgeForecastArtifactBinding,
    RidgeForecastModel,
    publish_ridge_forecast_artifact,
)

_DIGEST_DOMAIN = b"trade_rl.ridge_shared_cash_comparison_artifact_v1\0"


def _dataset() -> MarketDataset:
    offsets_minutes = np.asarray([0, 30, 120, 180, 360, 390], dtype=np.int64)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        offsets_minutes * np.timedelta64(1, "m")
    )
    signal = np.asarray([-0.20, -0.15, 0.20, 0.25, -0.20, -0.25], dtype=np.float32)
    secondary = np.asarray([0.4, 0.3, 0.2, 0.1, -0.1, -0.2], dtype=np.float32)
    row_features = np.stack((signal, secondary), axis=1)
    features = np.repeat(row_features[:, None, :], 2, axis=1)
    close = np.asarray(
        [
            [100.0, 200.0],
            [101.0, 198.0],
            [102.0, 196.0],
            [103.0, 194.0],
            [104.0, 192.0],
            [105.0, 190.0],
        ],
        dtype=np.float64,
    )
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((6, 2), 100_000.0, dtype=np.float64),
        funding_rate=np.zeros((6, 2), dtype=np.float64),
        tradable=np.ones((6, 2), dtype=np.bool_),
        feature_available=np.ones((6, 2, 2), dtype=np.bool_),
        feature_names=("momentum", "secondary"),
        global_feature_names=("regime",),
        periods_per_year=252,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
    )
    return dataset.with_content_identity({"fixture": "ridge-comparison-artifact"})


def _model(dataset: MarketDataset) -> RidgeForecastModel:
    return RidgeForecastModel(
        feature_indices=(1, 0),
        feature_mean=np.asarray([0.0, 0.0], dtype=np.float64),
        feature_scale=np.asarray([2.0, 1.0], dtype=np.float64),
        coefficients=np.asarray([0.0, 1.0], dtype=np.float64),
        intercept=0.0,
        horizon_hours=2,
        alpha=0.5,
        n_samples=4,
        fit_cutoff=dataset.timestamps[2].astype("datetime64[ns]"),
    )


def _binding(
    dataset: MarketDataset,
    schema_version: str,
    dataset_digest: str,
    model: RidgeForecastModel,
) -> RidgeForecastArtifactBinding:
    symbol_indices = (1, 0)
    return RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=schema_version,
        dataset_artifact_sha256=dataset_digest,
        fit_symbol_indices=symbol_indices,
        fit_symbol_names=tuple(dataset.symbols[index] for index in symbol_indices),
        feature_names=dataset.feature_names,
        feature_indices=model.feature_indices,
        fit_cutoff_ns=int(model.fit_cutoff.astype("int64")),
        horizon_hours=model.horizon_hours,
        alpha_hex=float(model.alpha).hex(),
        n_samples=model.n_samples,
        vintage_id="literal-comparison-artifact-fixture",
        source_commit="a" * 40,
        implementation_sha256="b" * 64,
        runtime_sha256="c" * 64,
    )


def _comparison(tmp_path: Path, *, market_order_profile=None):
    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    dataset_artifact = publish_market_dataset_artifact(dataset_root, dataset)
    model = _model(dataset)
    binding = _binding(
        dataset,
        dataset_artifact.schema_version,
        dataset_artifact.artifact_digest,
        model,
    )
    ridge_root = tmp_path / "ridge-model"
    ridge_digest = publish_ridge_forecast_artifact(ridge_root, model, binding)
    config = RidgeArtifactComparisonConfig(
        start_index=2,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=10_000.0,
        entry_threshold=0.10,
        exit_threshold=0.02,
        one_way_switch_cost=0.01,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.001,
            spread_rate=0.0005,
            impact_rate=0.0001,
            max_participation_rate=1.0,
        ),
        risk=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        ),
        market_order_profile=market_order_profile,
        minimum_hold_bars=0,
    )
    comparison = compare_ridge_artifact_shared_cash(
        dataset_root=dataset_root,
        ridge_artifact_root=ridge_root,
        expected_ridge_digest=ridge_digest,
        expected_binding=binding,
        config=config,
    )
    return comparison, build_candidate_run_provenance()


def _market_profile(dataset: MarketDataset):
    from tests.integrations.test_binance_forward_rules import symbol
    from trade_rl.integrations.binance.market_order_profile import (
        build_usdm_market_order_profile,
    )
    from trade_rl.integrations.binance.metadata import BinanceExchangeInfoSnapshot

    payload = {"symbols": [symbol("perpetual", name) for name in dataset.symbols]}
    raw = json.dumps(payload).encode()
    snapshot = BinanceExchangeInfoSnapshot(
        payload=payload,
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return build_usdm_market_order_profile(
        dataset,
        snapshot,
        selected_symbols=dataset.symbols,
        account_mode="one_way",
        reduce_only_exits=True,
    )


def _digest(raw: bytes) -> str:
    return hashlib.sha256(_DIGEST_DOMAIN + raw).hexdigest()


def _mutable_payload(loaded) -> dict[str, object]:
    return json.loads(canonical_json_bytes(loaded.payload))


def _reseal(root: Path, payload: dict[str, object]) -> str:
    raw = canonical_json_bytes(payload)
    (root / "comparison.json").write_bytes(raw)
    return _digest(raw)


def test_publish_load_preserves_complete_four_arm_context_and_ledgers(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"

    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(root, expected_digest=digest)

    payload = loaded.payload
    assert loaded.root == root
    assert loaded.artifact_digest == digest
    assert payload["schema_version"] == "ridge_shared_cash_comparison_v1"
    dataset = payload["dataset"]
    assert dataset["dataset_id"] == comparison.dataset_id
    assert dataset["schema_version"] == comparison.dataset_artifact.schema_version
    assert dataset["artifact_digest"] == comparison.dataset_artifact.artifact_digest
    assert dataset["symbols"] == ("BTCUSDT", "ETHUSDT")
    assert dataset["evaluation_start_index"] == 2
    assert dataset["evaluation_stop_index"] == 5
    assert dataset["evaluation_start_ns"] == comparison.evaluation_start_ns
    assert dataset["evaluation_stop_ns"] == comparison.evaluation_stop_ns
    assert payload["model"]["artifact_digest"] == comparison.ridge_artifact_digest
    assert payload["model"]["binding"]["feature_names"] == (
        "momentum",
        "secondary",
    )
    assert payload["config"]["gross_budget"] == 0.5
    assert payload["config"]["execution_cost"]["fee_rate"] == 0.001
    assert canonical_json_bytes(payload["provenance"]) == canonical_json_bytes(
        provenance
    )

    arms = payload["arms"]
    assert tuple(arm["name"] for arm in arms) == (
        "cash",
        "legacy_hysteresis",
        "position_aware_zero_cost",
        "position_aware_cost",
    )
    for arm, entry in zip(arms, comparison.entries, strict=True):
        ledger = arm["ledger"]
        expected_ledger = json.loads(
            canonical_json_bytes(entry.replay.ledger_evidence.to_mapping())
        )
        assert canonical_json_bytes(ledger) == canonical_json_bytes(expected_ledger)
        assert ledger["schema_version"] == "shared_cash_replay_ledger_v2"
        assert ledger["dataset_id"] == comparison.dataset_id
        assert len(ledger["intervals"]) == 3
        assert len(ledger["decisions"]) == 2
        assert ledger["terminal_exact_quantities"] == ("0", "0")
        assert ledger["active_order_remainders"] == ()
        assert len(arm["returns"]["values"]) == 3
        assert arm["metrics"]["n_periods"] == 3
        assert arm["diagnostics"]["termination_reasons"] == ()

    with pytest.raises(TypeError):
        payload["schema_version"] = "changed"
    with pytest.raises(TypeError):
        dataset["dataset_id"] = "changed"


def test_profile_remains_a_descriptive_declaration_after_reload(tmp_path: Path) -> None:
    profile = _market_profile(_dataset())
    comparison, provenance = _comparison(tmp_path, market_order_profile=profile)
    root = tmp_path / "comparison"

    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(root, expected_digest=digest)

    saved = loaded.payload["config"]["market_order_profile"]
    assert saved["digest"] == profile.digest
    assert canonical_json_bytes(saved["declaration"]) == canonical_json_bytes(
        profile.canonical_payload()
    )
    assert not isinstance(saved, type(profile))


Mutation = Callable[[dict[str, object]], None]


def _mutations() -> tuple[tuple[str, Mutation], ...]:
    def unknown_field(payload):
        payload["unexpected"] = 1

    def reordered_arms(payload):
        payload["arms"].reverse()

    def boolean_number(payload):
        payload["config"]["gross_budget"] = True

    def malformed_dataset_id(payload):
        payload["dataset"]["dataset_id"] = "not-a-sha"

    def mismatched_model_dataset(payload):
        payload["model"]["binding"]["dataset_id"] = "e" * 64

    def changed_return(payload):
        payload["arms"][0]["returns"]["values"][0] += 0.01

    def changed_metrics(payload):
        payload["arms"][0]["metrics"]["total_return"] += 0.01

    def broken_cash_chain(payload):
        payload["arms"][0]["ledger"]["intervals"][0]["cash_after"] += 1.0

    def incomplete_coverage(payload):
        payload["arms"][0]["ledger"]["intervals"].pop()

    def nonflat_terminal(payload):
        payload["arms"][0]["ledger"]["terminal_exact_quantities"][0] = "1"

    def bad_event_symbol(payload):
        event = payload["arms"][1]["ledger"]["intervals"][0]["order_events"][0]
        event["symbol_index"] = True

    def changed_implementation_digest(payload):
        payload["provenance"]["implementation_digest"] = "f" * 64

    def changed_policy_digest(payload):
        payload["arms"][0]["ledger"]["execution_policy_digest"] = "d" * 64

    def changed_decision_quantity(payload):
        payload["arms"][0]["ledger"]["decisions"][0]["position_quantity_after"][0] += (
            1.0
        )

    return (
        ("unknown field", unknown_field),
        ("arm order", reordered_arms),
        ("boolean numeric", boolean_number),
        ("dataset identity", malformed_dataset_id),
        ("model Dataset binding", mismatched_model_dataset),
        ("return consistency", changed_return),
        ("metric consistency", changed_metrics),
        ("ledger cash chain", broken_cash_chain),
        ("coverage", incomplete_coverage),
        ("terminal position", nonflat_terminal),
        ("nested event", bad_event_symbol),
        ("implementation provenance", changed_implementation_digest),
        ("execution policy binding", changed_policy_digest),
        ("decision quantity", changed_decision_quantity),
    )


@pytest.mark.parametrize(
    ("case", "mutate"),
    _mutations(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_loader_rejects_semantic_mutations_even_with_recomputed_digest(
    tmp_path: Path,
    case: str,
    mutate: Mutation,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(
        root,
        expected_digest=_digest((root / "comparison.json").read_bytes()),
    )
    payload = _mutable_payload(loaded)
    mutate(payload)
    recomputed_digest = _reseal(root, payload)

    with pytest.raises((TypeError, ValueError), match="."):
        load_ridge_comparison_artifact(root, expected_digest=recomputed_digest)


def test_loader_rejects_wrong_external_digest_raw_tampering_and_extra_files(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)

    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest="f" * 64)

    raw = (root / "comparison.json").read_bytes()
    marker = b'"schema_version":"ridge_shared_cash_comparison_v1"'
    assert marker in raw
    (root / "comparison.json").write_bytes(
        raw.replace(marker, b'"schema_version":"x"', 1)
    )
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=digest)

    (root / "comparison.json").write_bytes(raw)
    (root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=digest)


def test_loader_rejects_duplicate_keys_and_noncanonical_json(tmp_path: Path) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    raw = (root / "comparison.json").read_bytes()
    duplicate = raw[:-1] + b',"schema_version":"ridge_shared_cash_comparison_v1"}'
    (root / "comparison.json").write_bytes(duplicate)
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=_digest(duplicate))

    (root / "comparison.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=_digest(raw + b" "))


def test_publish_is_write_once_and_preserves_existing_artifact(tmp_path: Path) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    before = (root / "comparison.json").read_bytes()

    with pytest.raises(FileExistsError):
        publish_ridge_comparison_artifact(root, comparison, provenance=provenance)

    assert (root / "comparison.json").read_bytes() == before
    assert (
        load_ridge_comparison_artifact(root, expected_digest=digest).artifact_digest
        == digest
    )


@pytest.mark.parametrize("failure_point", ["write", "rename"])
def test_publish_failure_cleans_only_owned_staging_and_leaves_no_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    unowned_stage = tmp_path / ".comparison.staging-unowned"
    unowned_stage.mkdir()
    sentinel = unowned_stage / "sentinel"
    sentinel.write_bytes(b"preserve")
    if failure_point == "write":

        def fail(*_args, **_kwargs):
            raise OSError("injected write failure")

        monkeypatch.setattr(
            "trade_rl.evaluation.runs.ridge_comparison_artifact.atomic_write_bytes",
            fail,
        )
    else:

        def fail(*_args, **_kwargs):
            raise OSError("injected rename failure")

        monkeypatch.setattr(
            "trade_rl.evaluation.runs.ridge_comparison_artifact.atomic_rename_directory",
            fail,
        )

    with pytest.raises(OSError):
        publish_ridge_comparison_artifact(root, comparison, provenance=provenance)

    assert not root.exists()
    assert sentinel.read_bytes() == b"preserve"
    assert tuple(tmp_path.glob(".comparison.staging-*")) == (unowned_stage,)
