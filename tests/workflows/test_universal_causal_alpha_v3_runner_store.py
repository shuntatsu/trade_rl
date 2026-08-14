from __future__ import annotations

import json
from dataclasses import replace

import pytest

from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_store import CausalAlphaV3RecordStore


def _metric() -> CausalAlphaV3ReplayMetric:
    return CausalAlphaV3ReplayMetric(
        run_manifest_digest="1" * 64,
        freeze_digest="2" * 64,
        candidate_digest="3" * 64,
        symbol="BTCUSDT",
        episode_index=4,
        contract_digest="4" * 64,
        fit_digest="5" * 64,
        forecast_digest="6" * 64,
        target_path_digest="7" * 64,
        gross_return=0.01,
        net_return=0.008,
        turnover_per_day=0.2,
        total_execution_cost=10.0,
        trade_count=4,
        submitted_change_count=3,
        sign_flip_count=1,
        liquidity_deleveraging_count=0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(("max_abs_weight", 1),),
        target_reason_counts=(("hold", 5), ("rebalance", 3)),
        hard_risk_violation=False,
    )


def test_record_store_round_trips_only_expected_scopes(tmp_path) -> None:
    metric = _metric()
    store = CausalAlphaV3RecordStore(
        tmp_path,
        run_manifest_digest=metric.run_manifest_digest,
        freeze_digest=metric.freeze_digest,
    )

    path = store.write_replay_metric(metric)
    loaded = store.load_replay_metrics(
        expected_contract_digests={metric.identity: metric.contract_digest}
    )

    assert path.is_file()
    assert loaded == {metric.identity: metric}
    assert store.write_replay_metric(metric) == path


def test_record_store_rejects_tampering_unknown_scope_and_identity_drift(
    tmp_path,
) -> None:
    metric = _metric()
    store = CausalAlphaV3RecordStore(
        tmp_path,
        run_manifest_digest=metric.run_manifest_digest,
        freeze_digest=metric.freeze_digest,
    )
    path = store.write_replay_metric(metric)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["net_return"] = 0.5
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        store.load_replay_metrics(
            expected_contract_digests={metric.identity: metric.contract_digest}
        )

    path.unlink()
    store.write_replay_metric(metric)
    with pytest.raises(ValueError, match="scope"):
        store.load_replay_metrics(expected_contract_digests={})

    other = CausalAlphaV3RecordStore(
        tmp_path,
        run_manifest_digest="9" * 64,
        freeze_digest=metric.freeze_digest,
    )
    with pytest.raises(ValueError, match="run manifest"):
        other.load_replay_metrics(
            expected_contract_digests={metric.identity: metric.contract_digest}
        )


def test_record_store_rejects_path_traversal_in_artifacts_and_symbols(tmp_path) -> None:
    metric = _metric()
    store = CausalAlphaV3RecordStore(
        tmp_path,
        run_manifest_digest=metric.run_manifest_digest,
        freeze_digest=metric.freeze_digest,
    )

    with pytest.raises(ValueError, match="under the store root"):
        store.write_exact_artifact("../escape.json", {"schema_version": "test_v1"})

    unsafe = replace(metric, symbol="../escape", digest="")
    with pytest.raises(ValueError, match="safe artifact path segment"):
        store.write_replay_metric(unsafe)
