"""No-market-input admission and fixed native comparison wiring tests."""

from __future__ import annotations

import gzip
import hashlib

import numpy as np
import pytest

from tests.strategies.test_divergence_cloud_follow import _native_market
from tools import cloud_regime_switch_dev as study


def test_frozen_roster_and_protocol_identity():
    assert study.PROTOCOL.parent == study.REPO_ROOT / "report"
    assert hashlib.sha256(study.PROTOCOL.read_bytes()).hexdigest() == (
        "4a8d3c70c10332bff9d023b493baa88a03f2ee513b11a3d777a7f563605cb0cf"
    )
    p = study.protocol()
    assert len(p["arms"]) == 11
    assert (
        len(p["symbols"]) * len(p["years"]) * len(p["arms"]) * len(p["scenarios"])
        == 198
    )
    assert p["primary"].startswith("S2,")
    assert set(p["datasets"]) == set(p["symbols"])
    assert (
        p["scenarios"]["double_cost"]["fee_rate"]
        == 2 * p["scenarios"]["normal"]["fee_rate"]
    )
    assert p["scenarios"]["latency_1h"]["order_latency_bars"] == 1


@pytest.mark.parametrize("arm", ["CASH", "BH25"])
def test_native_controls_and_ledger(arm):
    ds = _native_market()
    m, z = study.evaluate_cell(ds, arm, "normal", start=0, stop=ds.n_bars - 1)
    raw = gzip.decompress(z)
    assert hashlib.sha256(raw).hexdigest() == m["ledger_sha256"]
    assert m["terminal_ok"] and m["valid"]
    if arm == "CASH":
        assert m["net_return_pct"] == 0 and m["native_fills"] == 0
    else:
        assert m["native_fills"] >= 2
        assert m["pre_interval_long_hours"] > 0
        assert m["pre_interval_short_hours"] == 0


def test_git_lookup_is_independent_of_protocol_location(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "PROTOCOL", tmp_path / "protocol.json")
    assert study._git("rev-parse", "--show-toplevel") == str(study.REPO_ROOT)


def test_admission_refuses_dirty_or_mismatched_head(monkeypatch):
    monkeypatch.setattr(
        study, "_git", lambda *args: "dirty" if args[0] == "status" else "a" * 40
    )
    with pytest.raises(ValueError, match="clean exact"):
        study._authorize({"identity": {}}, 1, 2, "unused")


def test_all_strategy_modes_use_canonical_replay_on_synthetic_warmup():
    from tests.data.test_divergence_cloud_mtf import _bars
    from trade_rl.data.market import MarketDataset

    raw = _bars(24 * 90)
    n = len(raw.close)
    ramp = np.maximum(np.arange(n, dtype=float) - 24 * 85, 0.0)
    ds = MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=raw.timestamps,
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=(raw.open + ramp)[:, None],
        high=(raw.high + ramp)[:, None],
        low=(raw.low + ramp)[:, None],
        close=(raw.close + ramp)[:, None],
        volume=raw.volume[:, None],
        funding_rate=raw.funding_rate[:, None],
        tradable=raw.tradable[:, None],
        feature_available=np.ones((n, 1, 1), dtype=bool),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8760,
    )
    for arm in study.protocol()["arms"]:
        metrics, _ = study.evaluate_cell(ds, arm, "normal", start=n - 48, stop=n - 1)
        assert metrics["terminal_ok"] and metrics["valid"], arm


def _synthetic_cells():
    p = study.protocol()
    cells = {}
    for symbol in p["symbols"]:
        for year in p["years"]:
            for arm in p["arms"]:
                for scenario in p["scenarios"]:
                    cells[(symbol, year, arm, scenario)] = {
                        "valid": True,
                        "net_return_pct": 3.0 if arm == "S2" else 1.0,
                        "max_drawdown_pct": 5.0 if arm == "S2" else 10.0,
                    }
    return cells


def test_summary_requires_all_cells_and_does_not_select_another_winner():
    from tools.cloud_regime_switch_summary import decide_comparison

    cells = _synthetic_cells()
    assert (
        decide_comparison(study.protocol(), cells)["disposition"]
        == "DEVELOPMENT_PROMISING"
    )
    cells.pop(("BTCUSDT", 2022, "S1", "latency_1h"))
    assert decide_comparison(study.protocol(), cells)["disposition"] == "INCONCLUSIVE"
    cells = _synthetic_cells()
    cells[("BTCUSDT", 2022, "S2", "normal")]["valid"] = False
    assert decide_comparison(study.protocol(), cells)["disposition"] == "INCONCLUSIVE"
    cells = _synthetic_cells()
    for key, row in cells.items():
        if key[2] == "S1":
            row["net_return_pct"] = 4.0
    assert decide_comparison(study.protocol(), cells)["disposition"] == "NO_WINNER"


def test_summary_requires_strict_aggregate_and_cohort_hold_and_drawdown():
    from tools.cloud_regime_switch_summary import decide_comparison

    p = study.protocol()
    cells = _synthetic_cells()
    for key, row in cells.items():
        if key[2] == "C2":
            row["net_return_pct"] = 3.0
    assert decide_comparison(p, cells)["disposition"] == "NO_WINNER"
    cells = _synthetic_cells()
    for key in [("BTCUSDT", y, "BH25", "double_cost") for y in p["years"]]:
        cells[key]["net_return_pct"] = 3.1
    assert decide_comparison(p, cells)["disposition"] == "NO_WINNER"
    cells = _synthetic_cells()
    cells[("ETHUSDT", 2024, "S2", "normal")]["max_drawdown_pct"] = 10.01
    assert decide_comparison(p, cells)["disposition"] == "NO_WINNER"


def test_summary_latency_failure_and_inactive_hold_are_not_robustness_pass():
    from tools.cloud_regime_switch_summary import decide_comparison

    cells = _synthetic_cells()
    cells[("BTCUSDT", 2022, "BH25", "latency_1h")]["valid"] = False
    out = decide_comparison(study.protocol(), cells)
    assert out["disposition"] == "DEVELOPMENT_PROMISING"
    assert out["latency_all_valid"] is False
    assert out["latency_robustness_established"] is False
    cells[("BTCUSDT", 2022, "C3", "normal")]["net_return_pct"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decide_comparison(study.protocol(), cells)


def test_result_readback_rejects_changed_ledger_or_metadata(tmp_path):
    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    admission = {
        "head": "a" * 40,
        "packet_digest": "b" * 64,
        "protocol_sha256": "c" * 64,
    }
    name = "BTCUSDT_2022_BH25_normal"
    metrics, z = study.evaluate_cell(ds, "BH25", "normal", start=0, stop=ds.n_bars - 1)
    path = tmp_path / name
    path.mkdir()
    payload = {
        "admission": admission,
        "cell": name,
        "dataset_id": ds.dataset_id,
        "metrics": metrics,
        "ledger_gzip_sha256": hashlib.sha256(z).hexdigest(),
    }
    (path / "attempt.json").write_bytes(
        canonical_json_bytes({"admission": admission, "cell": name})
    )
    (path / "result.json").write_bytes(canonical_json_bytes(payload))
    (path / "ledger.json.gz").write_bytes(z)
    assert (
        read_cell(
            path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
        )
        == metrics
    )
    (path / "ledger.json.gz").write_bytes(z + b"changed")
    with pytest.raises(ValueError, match="digest"):
        read_cell(
            path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
        )
    (path / "ledger.json.gz").write_bytes(z)
    metrics["net_return_pct"] += 10.0
    (path / "result.json").write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="net_return_pct"):
        read_cell(
            path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
        )


def test_summary_command_is_blocked_before_reading_results_without_review(monkeypatch):
    import sys

    def refuse(*args, **kwargs):
        raise ValueError("review missing")

    def forbidden(*args, **kwargs):
        raise AssertionError("attempted read before admission")

    monkeypatch.setattr(study, "_authorize", refuse)
    monkeypatch.setattr(study, "summarize_recorded_cells", forbidden)
    monkeypatch.setattr(study, "_load_fixed", forbidden)
    monkeypatch.setattr(study, "evaluate_cell", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study",
            "--packet",
            str(study.PROTOCOL),
            "--pr",
            "1",
            "--review-id",
            "2",
            "--owner-comment-url",
            "unused",
            "--summarize",
        ],
    )
    with pytest.raises(ValueError, match="review missing"):
        study.main()


def test_empty_result_root_is_inconclusive_without_market_read(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "OUTPUT_ROOT", tmp_path / "not_created")
    result = study.summarize_recorded_cells({})
    assert result["disposition"] == "INCONCLUSIVE"
    assert result["recorded_cells"] == 0
    assert result["required_cells"] == 198
    assert not study.OUTPUT_ROOT.exists()


def test_cell_mode_refuses_symlinked_output_root(monkeypatch, tmp_path):
    import sys

    target = tmp_path / "other_study"
    target.mkdir()
    alias = tmp_path / "output"
    alias.symlink_to(target, target_is_directory=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("attempted market load through aliased root")

    monkeypatch.setattr(study, "OUTPUT_ROOT", alias)
    monkeypatch.setattr(study, "_authorize", lambda *args: {})
    monkeypatch.setattr(study, "_load_fixed", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study",
            "--packet",
            str(study.PROTOCOL),
            "--pr",
            "1",
            "--review-id",
            "2",
            "--owner-comment-url",
            "synthetic-only",
            "--symbol",
            "BTCUSDT",
            "--year",
            "2022",
            "--arm",
            "CASH",
            "--scenario",
            "normal",
        ],
    )
    with pytest.raises(ValueError, match="output root must not be a symlink"):
        study.main()
    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "push",
        "wrong_pr",
        "missing",
        "skipped",
        "wrong_head",
        "duplicate",
        "pending",
        "changed_attempt",
    ],
)
def test_exact_full_ci_admission(monkeypatch, fault):
    from copy import deepcopy

    head = "a" * 40
    run = dict(
        id=123,
        run_attempt=1,
        name="CI",
        path=".github/workflows/ci.yml",
        head_sha=head,
        event="pull_request_review",
        status="completed",
        conclusion="success",
        pull_requests=[{"number": 1355, "head": {"sha": head}}],
    )
    jobs = [
        dict(
            name=name,
            head_sha=head,
            run_id=123,
            run_attempt=1,
            status="completed",
            conclusion="success",
        )
        for name in (
            "Lean Core",
            "PPO Runtime",
            "Human Guide",
            "Market Store PostgreSQL",
            "Generic Independent Research Review",
        )
    ]
    if fault == "push":
        run["event"] = "push"
    if fault == "wrong_pr":
        run["pull_requests"][0]["number"] = 999
    if fault == "missing":
        del jobs[3]
    if fault == "skipped":
        jobs[3]["conclusion"] = "skipped"
    if fault == "wrong_head":
        jobs[0]["head_sha"] = "b" * 40
    if fault == "duplicate":
        jobs.append(deepcopy(jobs[0]))
    runs = [run]
    if fault == "pending":
        runs.append(dict(run, id=124, status="in_progress", conclusion=None))

    def api(path):
        if "/actions/runs?" in path:
            return {"total_count": len(runs), "workflow_runs": deepcopy(runs)}
        if path.endswith("/attempts/1/jobs?per_page=100"):
            return {"total_count": len(jobs), "jobs": deepcopy(jobs)}
        if path.endswith("/actions/runs/123"):
            return dict(run, run_attempt=2) if fault == "changed_attempt" else run
        pytest.fail(f"unexpected API request: {path}")

    monkeypatch.setattr(study, "_github_api_read", api)
    if fault is None:
        assert study._require_full_ci(head, 1355) == (123, 1)
    else:
        with pytest.raises(ValueError, match="CI"):
            study._require_full_ci(head, 1355)
