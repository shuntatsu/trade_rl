from __future__ import annotations

import pytest

from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)


def _sha(token: str) -> str:
    return token * 64


def test_signal_resume_loader_rejects_leaf_copied_from_another_run(tmp_path) -> None:
    source_root = tmp_path / "source-run"
    target_root = tmp_path / "target-run"
    metric = CausalAlphaV3SignalScopeMetric(
        run_manifest_digest=_sha("1"),
        fit_config_digest=_sha("f"),
        symbol="BTCUSDT",
        episode_index=0,
        contract_start=10,
        contract_stop=20,
        contract_digest=_sha("c"),
        fit_digest=_sha("a"),
        forecast_digest=_sha("b"),
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.6,
        top_bottom_realized_spread=0.01,
        cohort_indices=(10, 15),
    )
    source_store = CausalAlphaV3ArtifactStore(
        source_root, run_manifest_digest=metric.run_manifest_digest
    )
    source_store.write_signal_scope_metric(metric)
    relative_path = (
        source_root
        / "signal"
        / "records"
        / metric.fit_config_digest
        / metric.symbol
        / f"{metric.episode_index}.json"
    ).relative_to(source_root)
    target_path = target_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes((source_root / relative_path).read_bytes())

    target_store = CausalAlphaV3ArtifactStore(
        target_root, run_manifest_digest=_sha("2")
    )
    with pytest.raises(ValueError, match="run manifest"):
        target_store.load_signal_scope_metrics(
            expected={metric.identity: metric.contract_digest}
        )
