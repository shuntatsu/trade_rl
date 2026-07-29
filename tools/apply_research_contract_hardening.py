from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.strip() + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        return
    path.write_text(normalized, encoding="utf-8")


def main() -> None:
    write(
        "tests/data/test_economic_semantics.py",
        r'''
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trade_rl.data.contracts import InstrumentContract, InstrumentExecutionRule
from trade_rl.data.economic_semantics import build_market_economic_semantics


def test_economic_semantics_are_explicit_point_in_time_and_immutable() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = np.arange(
        np.datetime64("2026-01-01T00:15:00", "ns"),
        np.datetime64("2026-01-01T01:15:00", "ns"),
        np.timedelta64(15, "m"),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=start,
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
        execution_rules=(
            InstrumentExecutionRule(
                effective_at=start,
                tick_size=0.1,
                lot_size=0.001,
                minimum_notional=5.0,
            ),
            InstrumentExecutionRule(
                effective_at=start + timedelta(minutes=45),
                tick_size=0.2,
                lot_size=0.002,
                minimum_notional=10.0,
            ),
        ),
    )
    shape = (len(timestamps), 1)
    semantics = build_market_economic_semantics(
        timestamps=timestamps,
        instruments=(contract,),
        row_present=np.ones(shape, dtype=np.bool_),
        raw_tradable=np.ones(shape, dtype=np.bool_),
        source_information_available=np.ones(shape, dtype=np.bool_),
        available_at=np.broadcast_to(timestamps[:, None], shape),
        close=np.full(shape, 100.0),
        funding_event_count=np.zeros(shape, dtype=np.int32),
    )

    assert semantics.tick_size[:, 0].tolist() == [0.1, 0.1, 0.2, 0.2]
    assert semantics.lot_size[:, 0].tolist() == [0.001, 0.001, 0.002, 0.002]
    assert semantics.minimum_notional[:, 0].tolist() == [5.0, 5.0, 10.0, 10.0]
    assert np.all(semantics.max_participation_rate == 1.0)
    assert np.all(semantics.borrow_available)
    assert np.all(semantics.buy_allowed)
    assert np.all(semantics.sell_allowed)
    np.testing.assert_array_equal(semantics.mark_price, semantics.index_price)
    np.testing.assert_array_equal(semantics.mark_price, np.full(shape, 100.0))
    for value in semantics.market_dataset_kwargs().values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable


def test_vision_and_postgres_builders_use_the_same_economic_constructor() -> None:
    builder = (Path(__file__).resolve().parents[2] / "trade_rl/data/builder.py").read_text()
    postgres = (
        Path(__file__).resolve().parents[2]
        / "trade_rl/integrations/postgres_market_dataset.py"
    ).read_text()
    assert "build_market_economic_semantics" in builder
    assert "build_market_economic_semantics" in postgres
''',
    )
    write(
        "tests/learning/test_oracle_bc_causal_gate_contract.py",
        r'''
from __future__ import annotations

import numpy as np

from trade_rl.learning.evaluation import deterministic_bootstrap_upper_bound


def test_bootstrap_upper_bound_is_deterministic_and_one_sided() -> None:
    values = np.array([0.01, 0.03, 0.02, 0.08, 0.04], dtype=np.float64)
    first = deterministic_bootstrap_upper_bound(
        values,
        confidence_level=0.95,
        resamples=2_000,
        seed_material="a" * 64,
    )
    second = deterministic_bootstrap_upper_bound(
        values,
        confidence_level=0.95,
        resamples=2_000,
        seed_material="a" * 64,
    )
    assert first == second
    assert first >= float(np.mean(values))


def test_maintained_target_weight_profiles_require_nontrivial_causal_evidence() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    names = (
        "training-target-weight-growth-ppo.json",
        "training-target-weight-constrained-growth.json",
        "training-target-weight-constrained-growth-discounted.json",
    )
    for name in names:
        payload = json.loads(
            (root / "examples/binance-multitimeframe" / name).read_text()
        )
        training = payload["training"]
        assert training["behavior_cloning_required_relative_improvement"] > 0.0
        assert training["behavior_cloning_min_causal_holdout_trades"] >= 30
        assert training["behavior_cloning_causal_holdout_bootstrap_resamples"] >= 2_000
        assert training["behavior_cloning_causal_holdout_confidence_level"] >= 0.95
''',
    )
    write(
        "tests/examples/test_dataset_publication_order_contract.py",
        r'''
from __future__ import annotations

import ast
from pathlib import Path


def test_maintained_dataset_is_validated_before_immutable_publication() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/binance-multitimeframe/full_research_pipeline.py"
    )
    module = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_build_dataset"
    )
    calls: list[str] = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
    assert "validate_maintained_dataset_preset" in calls
    assert calls.index("validate_maintained_dataset_preset") < calls.index(
        "publish_market_dataset_artifact"
    )
''',
    )
    write(
        "tests/integrations/test_binance_cache_integrity_contract.py",
        r'''
from __future__ import annotations

import json
import urllib.request

import pytest

from trade_rl.integrations.binance import BinancePublicTransport, BinanceTransportError


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.headers = {"ETag": '"fixture"', "Last-Modified": "Wed, 29 Jul 2026 00:00:00 GMT"}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_vision_cache_has_verified_content_sidecar(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"official-archive"
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response(payload),
    )
    transport = BinancePublicTransport(cache_root=tmp_path, max_attempts=1)
    url = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/file.zip"

    assert transport._request_bytes(url) == payload
    binary = next(tmp_path.rglob("*.bin"))
    sidecar = binary.with_suffix(".json")
    evidence = json.loads(sidecar.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "binance_vision_raw_cache_v1"
    assert evidence["url"] == url
    assert evidence["size_bytes"] == len(payload)
    assert len(evidence["sha256"]) == 64
    assert evidence["etag"] == '"fixture"'

    binary.write_bytes(b"tampered")
    with pytest.raises(BinanceTransportError, match="digest|size"):
        transport._request_bytes(url)
''',
    )
    write(
        "tests/rl/test_change_intensity_contract.py",
        r'''
from __future__ import annotations

import numpy as np
import torch

from trade_rl.rl.action_telemetry import hierarchical_action_stage_metrics
from trade_rl.rl.policies import HierarchicalActorOutputs


def test_gate_output_is_exposed_as_change_intensity_without_checkpoint_rename() -> None:
    intensity = torch.tensor([[0.25, 0.75]])
    outputs = HierarchicalActorOutputs(
        gate_logits=torch.zeros_like(intensity),
        gate_probabilities=intensity,
        target_actions=torch.ones_like(intensity),
        composed_actions=torch.full_like(intensity, 0.5),
        mean_logits=torch.zeros_like(intensity),
        current_weights=torch.zeros_like(intensity),
        active_mask=torch.ones_like(intensity, dtype=torch.bool),
    )
    assert outputs.change_intensity is outputs.gate_probabilities


def test_action_stage_metrics_measure_exploration_and_effective_action() -> None:
    metrics = hierarchical_action_stage_metrics(
        deterministic_composed=np.array([0.1, 0.2]),
        sampled_policy_action=np.array([0.4, 0.0]),
        submitted_target=np.array([0.3, 0.0]),
        effective_filled_weights=np.array([0.25, 0.05]),
    )
    assert metrics["exploration_l1"] == 0.5
    assert metrics["submission_l1"] == 0.4
    assert metrics["effective_action_l1"] == 0.3
''',
    )
    write(
        "tests/test_maintained_documentation_v3_contract.py",
        r'''
from __future__ import annotations

import json
from pathlib import Path


def test_maintained_docs_match_config_v3_and_structured_export_v2() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    architecture = (root / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    configuration = (root / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    binance = (root / "docs/BINANCE.md").read_text(encoding="utf-8")

    assert "training_run_config_v3" in readme
    assert "training_run_config_v2" not in architecture
    assert "structured_policy_export_v2" in architecture
    assert "# Training Configuration v3" in configuration
    assert "training_run_config_v2" not in configuration
    assert "structured_policy_export_v1" not in configuration
    assert "change intensity" in architecture.lower()
    assert "constraint cost" in architecture.lower()
    assert "binance_vision_raw_cache_v1" in binance
    assert "runner classification" in architecture.lower()
    assert "offline_signing" in architecture
    assert "#193" in architecture


def test_quickstart_pins_hybrid_reward_instead_of_inheriting_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples/quickstart/training.json").read_text())
    reward = payload["reward"]
    assert reward["absolute_growth_weight"] == 1.0
    assert reward["incremental_drawdown_weight"] == 0.05
    assert reward["baseline_underperformance_weight"] == 0.10
    assert reward["terminal_equity_weight"] == 1.0
    assert reward["margin_deficit_weight"] == 1.0
''',
    )


if __name__ == "__main__":
    main()
