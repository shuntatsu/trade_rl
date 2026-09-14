from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.issue541_stage_a import StageABridgeError, verify_stage_a_bridge

STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture(root: Path, *, implementation: str, fingerprint: str) -> None:
    _write_json(root / "study" / "plan.json", {"schema_version": "plan", "x": 1})
    evidence = root / "study" / "baseline" / "evidence"
    _write_json(
        evidence / "manifest.json",
        {
            "schema_version": "controlled_evidence_set_v1",
            "fingerprint": fingerprint,
            "ppo_seeds": [0, 1],
            "research_context_digest": "c" * 64,
            "semantic_config": {
                "execution_overlay": "zero_overlay_dataset_fields_authoritative"
            },
            "semantic_config_digest": "d" * 64,
            "run_digests": [
                {"ppo_seed": 0, "artifact_digest": "0" * 64},
                {"ppo_seed": 1, "artifact_digest": "1" * 64},
            ],
        },
    )
    for seed in (0, 1):
        run = evidence / "runs" / f"seed-{seed}"
        strategies = []
        for index, name in enumerate(STRATEGIES):
            is_cash = name == "cash"
            strategies.append(
                {
                    "name": name,
                    "return_key": f"symbol_0_strategy_{index}",
                    "fill_count": 0 if is_cash else 1,
                    "final_portfolio_value": 100000.0,
                    "diagnostics": {
                        "borrow_cost": 0.0,
                        "funding_pnl": 0.0,
                        "n_trades": 0 if is_cash else 1,
                        "rebalance_events": 0 if is_cash else 1,
                        "termination_reasons": [],
                        "total_cost": 0.0 if is_cash else 1.0,
                        "turnover_total": 0.0 if is_cash else 1.0,
                    },
                    "metrics": {
                        "borrow_cost": 0.0,
                        "funding_pnl": 0.0,
                        "max_drawdown": 0.0,
                        "n_periods": 3,
                        "n_trades": 0 if is_cash else 1,
                        "periods_per_year": 8760,
                        "rebalance_events": 0 if is_cash else 1,
                        "return_kind": "base_bar",
                        "sharpe": 0.0,
                        "sortino": 0.0,
                        "termination_count": 0,
                        "total_cost": 0.0 if is_cash else 1.0,
                        "total_return": 0.0,
                        "turnover_total": 0.0 if is_cash else 1.0,
                    },
                }
            )
        _write_json(
            run / "summary.json",
            {
                "schema_version": "lean_candidate_result_v2",
                "dataset_id": "a" * 64,
                "dataset_artifact": {
                    "schema_version": "market_dataset_artifact_v3",
                    "artifact_digest": "b" * 64,
                },
                "candidate_config": {"ppo_seed": seed},
                "evaluation": {
                    "execution_overlay": "zero_overlay_dataset_fields_authoritative"
                },
                "ppo_observation": {"schema_version": "ppo_observation_v2"},
                "symbols": ["BTCUSDT"],
                "by_symbol": [
                    {"symbol": "BTCUSDT", "symbol_index": 0, "strategies": strategies}
                ],
            },
        )
        arrays = {
            f"symbol_0_strategy_{i}": np.asarray(
                [0.0, i / 1000.0, 0.0], dtype=np.float64
            )
            for i in range(len(STRATEGIES))
        }
        np.savez(run / "returns.npz", **arrays)
        _write_json(
            run / "provenance.json",
            {
                "schema_version": "candidate_run_provenance_v1",
                "implementation_digest": implementation,
                "runtime_environment_digest": "e" * 64,
                "research_context_digest": "c" * 64,
            },
        )
    _write_json(
        root / "study" / "baseline" / "analysis.json",
        {
            "schema_version": "controlled_evidence_analysis_binding_v1",
            "evidence_fingerprint": fingerprint,
            "analysis_digest": "f" * 64,
            "analysis": {
                "schema_version": "controlled_evidence_analysis_v1",
                "analysis_digest": "9" * 64,
                "seeds": [0, 1],
                "by_symbol": {"BTCUSDT": {"x": 1}},
            },
        },
    )


def test_bridge_allows_only_implementation_provenance_change(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    _write_fixture(original, implementation="1" * 64, fingerprint="2" * 64)
    _write_fixture(replay, implementation="3" * 64, fingerprint="4" * 64)

    report = verify_stage_a_bridge(original, replay)

    assert report["status"] == "PASS"
    assert report["raw_return_arrays_checked"] == 16
    assert report["summaries_checked"] == 2
    assert report["implementation_digest_changed"] is True
    assert report["runtime_environment_digest_match"] is True


def test_bridge_rejects_any_raw_return_drift(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    _write_fixture(original, implementation="1" * 64, fingerprint="2" * 64)
    _write_fixture(replay, implementation="3" * 64, fingerprint="4" * 64)
    np.savez(
        replay / "study" / "baseline" / "evidence" / "runs" / "seed-1" / "returns.npz",
        **{
            f"symbol_0_strategy_{i}": np.asarray(
                [0.0, 9.0 if i == 3 else i / 1000.0, 0.0], dtype=np.float64
            )
            for i in range(len(STRATEGIES))
        },
    )

    with pytest.raises(StageABridgeError, match="raw return"):
        verify_stage_a_bridge(original, replay)


def test_bridge_rejects_economic_summary_drift(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    _write_fixture(original, implementation="1" * 64, fingerprint="2" * 64)
    _write_fixture(replay, implementation="3" * 64, fingerprint="4" * 64)
    path = (
        replay / "study" / "baseline" / "evidence" / "runs" / "seed-0" / "summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["by_symbol"][0]["strategies"][3]["metrics"]["total_return"] = 0.1
    _write_json(path, payload)

    with pytest.raises(StageABridgeError, match="summary"):
        verify_stage_a_bridge(original, replay)


def test_bridge_rejects_runtime_drift_and_invalid_cost_semantics(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    _write_fixture(original, implementation="1" * 64, fingerprint="2" * 64)
    _write_fixture(replay, implementation="3" * 64, fingerprint="4" * 64)
    provenance = (
        replay
        / "study"
        / "baseline"
        / "evidence"
        / "runs"
        / "seed-0"
        / "provenance.json"
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["runtime_environment_digest"] = "8" * 64
    _write_json(provenance, payload)

    with pytest.raises(StageABridgeError, match="runtime environment"):
        verify_stage_a_bridge(original, replay)

    _write_fixture(replay, implementation="3" * 64, fingerprint="4" * 64)
    summary = (
        replay / "study" / "baseline" / "evidence" / "runs" / "seed-0" / "summary.json"
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["by_symbol"][0]["strategies"][3]["metrics"]["total_cost"] = 0.0
    payload["by_symbol"][0]["strategies"][3]["diagnostics"]["total_cost"] = 0.0
    original_summary = (
        original
        / "study"
        / "baseline"
        / "evidence"
        / "runs"
        / "seed-0"
        / "summary.json"
    )
    original_payload = json.loads(original_summary.read_text(encoding="utf-8"))
    original_payload["by_symbol"][0]["strategies"][3]["metrics"]["total_cost"] = 0.0
    original_payload["by_symbol"][0]["strategies"][3]["diagnostics"]["total_cost"] = 0.0
    _write_json(summary, payload)
    _write_json(original_summary, original_payload)

    with pytest.raises(StageABridgeError, match="positive trading cost"):
        verify_stage_a_bridge(original, replay)
