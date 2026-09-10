from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.bootstrap.test_binance import (
    _config,
    _FakeLiveTransport,
)
from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    CanonicalM2BootstrapResult,
    bootstrap_canonical_m2_study,
    inspect_canonical_m2_bootstrap,
)
from trade_rl.integrations.binance import (
    BinanceDatasetBuildResult,
    BinanceTransportMode,
)


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "bootstrap-config.json"
    path.write_text(json.dumps(_config().to_payload()), encoding="utf-8")
    return path


def _synthetic_dataset(metadata_evidence: object):
    periods = 60 * 24
    timestamps = np.datetime64("2024-01-01T01:00:00", "ns") + np.arange(
        periods
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(periods, dtype=np.float64) * 0.01
    raw_by_symbol = {}
    for offset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        shifted = close + offset * 10.0
        open_price = np.concatenate([shifted[:1], shifted[:-1]])
        raw_by_symbol[symbol] = RawMarketSeries(
            timestamps=timestamps,
            open=open_price,
            high=np.maximum(open_price, shifted) + 1.0,
            low=np.minimum(open_price, shifted) - 1.0,
            close=shifted,
            volume=np.full(periods, 1_000_000.0),
            funding_rate=np.zeros(periods),
            tradable=np.ones(periods, dtype=np.bool_),
        )
    contracts = tuple(
        InstrumentContract(symbol=symbol, listed_at=datetime(2024, 1, 1, tzinfo=UTC))
        for symbol in ("BTCUSDT", "ETHUSDT")
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="1h__log_return_24bar",
                    kind=FeatureKind.LOG_RETURN,
                    lookback=1,
                ),
            ),
        )
    ).build(
        InMemoryMarketDataSource(raw_by_symbol),
        contracts,
        identity_provenance=metadata_evidence,
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    from trade_rl.evaluation.experiments.bootstrap import workflow as workflow_module

    real_freeze = workflow_module._freeze_binance_source

    def freeze(config, source_root):
        source_root = Path(source_root)
        live = _FakeLiveTransport(source_root / "vision-cache")
        return real_freeze(config, source_root, live_transport=live)

    def build(**kwargs):
        assert kwargs["transport_mode"] is BinanceTransportMode.VISION
        transport = kwargs["transport"]
        assert transport.market_data.allow_network is False
        metadata_evidence = kwargs["metadata_evidence"]
        dataset = _synthetic_dataset(metadata_evidence)
        return BinanceDatasetBuildResult(
            dataset=dataset,
            metadata=(),
            sources_used=("frozen:exchange-info", "vision"),
            feature_timeframes=("1h",),
        )

    monkeypatch.setattr(workflow_module, "_freeze_binance_source", freeze)
    monkeypatch.setattr(workflow_module, "build_binance_market_dataset", build)


def test_bootstrap_publishes_dataset_and_study_atomically_before_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    config_path = _config_path(tmp_path)
    output = tmp_path / "canonical-m2"

    result = bootstrap_canonical_m2_study(config_path, output)

    assert isinstance(result, CanonicalM2BootstrapResult)
    assert result.root == output
    assert result.config_digest == _config().digest
    assert len(result.bootstrap_digest) == 64
    assert len(result.dataset_id) == 64
    assert len(result.dataset_artifact_digest) == 64
    assert len(result.study_digest) == 64
    assert {entry.name for entry in output.iterdir()} == {
        "bootstrap.json",
        "bootstrap-manifest.json",
        "source",
        "dataset",
        "study",
    }
    snapshot = inspect_study(output / "study")
    assert snapshot.baseline is None
    assert snapshot.experiment_sequences == ()
    assert not (output / "study" / "baseline").exists()
    assert inspect_canonical_m2_bootstrap(output) == result


def test_bootstrap_manifest_binds_source_dataset_study_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    output = tmp_path / "canonical-m2"
    result = bootstrap_canonical_m2_study(_config_path(tmp_path), output)
    manifest = json.loads(
        (output / "bootstrap-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["bootstrap_digest"] == result.bootstrap_digest
    assert manifest["bootstrap_config_digest"] == result.config_digest
    assert manifest["dataset_id"] == result.dataset_id
    assert manifest["dataset_artifact_digest"] == result.dataset_artifact_digest
    assert manifest["study_digest"] == result.study_digest
    assert manifest["vision_plan_digest"]
    assert manifest["raw_source_roster"]
    assert manifest["raw_source_roster_digest"]
    assert manifest["metadata_evidence"]["raw_payload_sha256"]
    assert manifest["implementation_digest"]
    assert manifest["runtime_environment_digest"]


def test_existing_output_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    output = tmp_path / "canonical-m2"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        bootstrap_canonical_m2_study(_config_path(tmp_path), output)

    assert marker.read_text(encoding="utf-8") == "untouched"
    assert {entry.name for entry in output.iterdir()} == {"keep.txt"}


def test_mid_bootstrap_failure_leaves_no_final_or_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    from trade_rl.evaluation.experiments.bootstrap import workflow as workflow_module

    def fail_build(**_kwargs):
        raise RuntimeError("dataset build failed")

    monkeypatch.setattr(workflow_module, "build_binance_market_dataset", fail_build)
    output = tmp_path / "canonical-m2"

    with pytest.raises(RuntimeError, match="dataset build failed"):
        bootstrap_canonical_m2_study(_config_path(tmp_path), output)

    assert not output.exists()
    assert not any(
        path.name.startswith(".canonical-m2.staging-") for path in tmp_path.iterdir()
    )


def test_provenance_drift_prevents_final_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    from trade_rl.evaluation.experiments.bootstrap import workflow as workflow_module

    real_provenance = workflow_module.build_candidate_run_provenance
    original = real_provenance()
    drifted = dict(original)
    drifted["runtime_environment_digest"] = "f" * 64
    values = iter((original, drifted))
    monkeypatch.setattr(
        workflow_module,
        "build_candidate_run_provenance",
        lambda: next(values),
    )
    output = tmp_path / "canonical-m2"

    with pytest.raises(ValueError, match="provenance.*drift|runtime provenance"):
        bootstrap_canonical_m2_study(_config_path(tmp_path), output)

    assert not output.exists()
    assert not any(
        path.name.startswith(".canonical-m2.staging-") for path in tmp_path.iterdir()
    )
