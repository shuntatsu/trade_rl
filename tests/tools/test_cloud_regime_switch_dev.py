"""No-market-input admission and fixed native comparison wiring tests."""

from __future__ import annotations

import gzip
import hashlib

import numpy as np
import pytest

from tests.strategies.test_divergence_cloud_follow import _native_market
from tools import cloud_regime_switch_dev as study
from trade_rl.simulation import ExecutionCostConfig, runtime_execution_policy_digest


def _expected_policy(ds, scenario):
    return runtime_execution_policy_digest(
        ExecutionCostConfig(
            **study.protocol()["scenarios"][scenario],
            max_participation_rate=0.05,
            allow_short=True,
            max_leverage=1.0,
        ),
        funding_price_rate_present=ds.funding_price_rate is not None,
    )


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


@pytest.mark.parametrize("mutation", ["fee", "roster", "whitespace"])
def test_frozen_protocol_refuses_replacement_before_admission(
    monkeypatch, tmp_path, mutation
):
    import json

    raw = study.PROTOCOL.read_bytes()
    value = json.loads(raw)
    if mutation == "fee":
        value["scenarios"]["normal"]["fee_rate"] *= 2
        raw = json.dumps(value).encode()
    elif mutation == "roster":
        value["arms"].remove("C2")
        raw = json.dumps(value).encode()
    else:
        raw += b" "
    replacement = tmp_path / "protocol.json"
    replacement.write_bytes(raw)
    monkeypatch.setattr(study, "PROTOCOL", replacement)
    head = "a" * 40
    packet = {
        "identity": {"implementation_head": head, "protocol_head": head},
        "thesis": {"limitations": hashlib.sha256(raw).hexdigest()},
    }
    monkeypatch.setattr(study, "_git", lambda *a: "" if a[0] == "status" else head)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid protocol reached review, market or result access")

    monkeypatch.setattr(study, "verify_github_review", forbidden)
    monkeypatch.setattr(study, "_github_api_read", forbidden)
    monkeypatch.setattr(study, "load_published_market_dataset_artifact", forbidden)
    for entry in (
        study.protocol,
        lambda: study.authorization_body(packet),
        lambda: study._authorize(packet, 1355, 123, "unused"),
        lambda: study._load_fixed("BTCUSDT"),
        lambda: study.summarize_recorded_cells({}),
    ):
        with pytest.raises(ValueError, match="frozen protocol digest"):
            entry()


def test_git_lookup_is_independent_of_protocol_location(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "PROTOCOL", tmp_path / "protocol.json")
    assert study._git("rev-parse", "--show-toplevel") == str(study.REPO_ROOT)


def test_admission_refuses_dirty_or_mismatched_head(monkeypatch):
    monkeypatch.setattr(
        study, "_git", lambda *args: "dirty" if args[0] == "status" else "a" * 40
    )
    with pytest.raises(ValueError, match="clean exact"):
        study._authorize({"identity": {}}, 1, 2, "unused")


@pytest.mark.parametrize("scenario", ["normal", "double_cost", "latency_1h"])
def test_all_strategy_modes_use_canonical_replay_on_synthetic_warmup(
    tmp_path, scenario
):
    from tests.data.test_divergence_cloud_mtf import _bars
    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes
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
        metrics, compressed = study.evaluate_cell(
            ds, arm, scenario, start=n - 48, stop=n - 1
        )
        assert metrics["terminal_ok"] and metrics["valid"], arm
        path = tmp_path / arm
        path.mkdir()
        admission = {"synthetic_only": True}
        attempt = {"admission": admission, "cell": path.name}
        (path / "attempt.json").write_bytes(canonical_json_bytes(attempt))
        (path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": metrics,
                    "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
        )
        (path / "ledger.json.gz").write_bytes(compressed)
        assert (
            read_cell(
                path,
                admission=admission,
                dataset_id=ds.dataset_id,
                arm=arm,
                scenario=scenario,
                expected_start=n - 48,
                expected_stop=n - 1,
                expected_symbol=ds.symbols[0],
                expected_execution_policy_digest=_expected_policy(ds, scenario),
            )
            == metrics
        )


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
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
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
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
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
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
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
        "missing_updated_at",
        "malformed_updated_at",
    ],
)
def test_exact_full_ci_admission(monkeypatch, fault):
    from copy import deepcopy

    head = "a" * 40
    run = dict(
        id=123,
        run_attempt=1,
        updated_at="2026-11-24T02:00:00Z",
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
    if fault == "missing_updated_at":
        del run["updated_at"]
    if fault == "malformed_updated_at":
        run["updated_at"] = "2026-11-24Z"
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
        assert study._require_full_ci(head, 1355) == (123, 1, "2026-11-24T02:00:00Z")
    else:
        with pytest.raises(ValueError, match="CI|timestamp"):
            study._require_full_ci(head, 1355)


@pytest.mark.parametrize(
    "created,updated,accepted",
    [
        ("2026-11-24T03:00:00Z", "2026-11-24T03:00:00Z", True),
        ("2026-11-24T00:00:00Z", "2026-11-24T00:00:00Z", False),
        ("2026-11-24T01:30:00Z", "2026-11-24T01:30:00Z", False),
        ("2026-11-24T02:00:00Z", "2026-11-24T02:00:00Z", False),
        ("2026-11-24T03:00:00Z", "2026-11-24T04:00:00Z", False),
    ],
)
@pytest.mark.parametrize(
    "late_fault",
    [
        None,
        "pr_head",
        "local_head",
        "local_dirty",
        "ci_attempt",
        "owner_body",
        "owner_timestamp",
    ],
)
def test_owner_comment_created_after_gates(
    monkeypatch, created, updated, accepted, late_fault
):
    from types import SimpleNamespace

    head = "a" * 40
    packet = {
        "identity": {"implementation_head": head, "protocol_head": head},
        "thesis": {"limitations": study._sha(study.PROTOCOL)},
    }
    url = "https://github.com/shuntatsu/trade_rl/pull/1355#issuecomment-456"
    body = study.authorization_body(packet)

    owner_seen = False
    final_review_seen = False

    def api(path):
        nonlocal owner_seen
        if path.endswith("/reviews/123"):
            return {
                "id": 123,
                "commit_id": head,
                "submitted_at": "2026-11-24T01:00:00Z",
            }
        if path.endswith("/issues/comments/456"):
            owner_seen = True
            return {
                "id": 456,
                "html_url": url,
                "issue_url": "https://api.github.com/repos/shuntatsu/trade_rl/issues/1355",
                "user": {"id": 148511409, "login": "shuntatsu", "type": "User"},
                "body": body + "\nREVOKED"
                if final_review_seen and late_fault == "owner_body"
                else body,
                "created_at": created,
                "updated_at": "2026-11-24T05:00:00Z"
                if final_review_seen and late_fault == "owner_timestamp"
                else updated,
            }
        pytest.fail(path)

    def verified(*args, api_read, **kwargs):
        nonlocal final_review_seen
        final_review_seen = owner_seen
        api_read("repos/shuntatsu/trade_rl/pulls/1355/reviews/123")
        return SimpleNamespace(
            status="BLOCKED" if owner_seen and late_fault == "pr_head" else "PASS"
        )

    monkeypatch.setattr(
        study,
        "_git",
        lambda *a: (
            ("dirty" if owner_seen and late_fault == "local_dirty" else "")
            if a[0] == "status"
            else ("b" * 40 if owner_seen and late_fault == "local_head" else head)
        ),
    )
    monkeypatch.setattr(study, "_github_api_read", api)
    monkeypatch.setattr(study, "verify_github_review", verified)
    monkeypatch.setattr(
        study,
        "_require_full_ci",
        lambda *a: (
            789,
            2 if owner_seen and late_fault == "ci_attempt" else 1,
            "2026-11-24T02:00:00Z",
        ),
    )
    if accepted and late_fault:
        with pytest.raises(ValueError, match="changed during admission"):
            study._authorize(packet, 1355, 123, url)
    elif accepted:
        assert study._authorize(packet, 1355, 123, url)["ci_run_id"] == 789
    else:
        with pytest.raises(ValueError, match="owner authorization"):
            study._authorize(packet, 1355, 123, url)


@pytest.mark.parametrize("mutation", ["terminal_dd", "understated_path_dd"])
def test_readback_rejects_contradictory_drawdown(tmp_path, mutation):
    import json

    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    admission = {"synthetic_only": True}
    name = "BTCUSDT_2022_BH25_normal"
    metrics, compressed = study.evaluate_cell(
        ds, "BH25", "normal", start=0, stop=ds.n_bars - 1
    )
    ledger = json.loads(gzip.decompress(compressed))
    assert ledger["final_max_drawdown"] > 0
    ledger["final_max_drawdown"] = 0.0
    metrics["max_drawdown_pct"] = 0.0
    if mutation == "understated_path_dd":
        for row in ledger["intervals"]:
            row["max_drawdown_before"] = row["max_drawdown_after"] = 0.0
    raw = canonical_json_bytes(ledger)
    compressed = gzip.compress(raw, mtime=0)
    metrics["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / name
    path.mkdir()
    attempt = {"admission": admission, "cell": name}
    (path / "attempt.json").write_bytes(canonical_json_bytes(attempt))
    (path / "result.json").write_bytes(
        canonical_json_bytes(
            {
                **attempt,
                "dataset_id": ds.dataset_id,
                "metrics": metrics,
                "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
            }
        )
    )
    (path / "ledger.json.gz").write_bytes(compressed)
    with pytest.raises(ValueError, match="drawdown"):
        read_cell(
            path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
        )


@pytest.mark.parametrize(
    "field",
    [
        "sharpe_hourly",
        "native_fills",
        "directional_equity_buckets",
        "mean_pre_interval_gross",
    ],
)
def test_readback_rejects_tampered_diagnostic_scalar(tmp_path, field):
    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, z = study.evaluate_cell(ds, "BH25", "normal", start=0, stop=ds.n_bars - 1)
    admission = {"head": "a" * 40}
    attempt = {"admission": admission, "cell": tmp_path.name}
    record = {
        **attempt,
        "dataset_id": ds.dataset_id,
        "metrics": metrics,
        "ledger_gzip_sha256": hashlib.sha256(z).hexdigest(),
    }
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))
    (tmp_path / "ledger.json.gz").write_bytes(z)

    def read():
        (tmp_path / "result.json").write_bytes(canonical_json_bytes(record))
        return read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
        )

    assert read() == metrics
    if field == "directional_equity_buckets":
        metrics[field]["net_long_usdt"] += 1000000
    else:
        metrics[field] = 999999
    with pytest.raises(ValueError, match=field):
        read()


@pytest.mark.parametrize(
    ("final_field", "metric_field"),
    [
        ("final_total_cost", "total_cost_usdt"),
        ("final_funding_pnl", "funding_pnl_usdt"),
        ("final_borrow_cost", "borrow_cost_usdt"),
        ("final_turnover_total", "turnover"),
    ],
)
def test_readback_rejects_rehashed_final_flow(tmp_path, final_field, metric_field):
    import json

    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, z = study.evaluate_cell(ds, "BH25", "normal", start=0, stop=ds.n_bars - 1)
    ledger = json.loads(gzip.decompress(z))
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": tmp_path.name}
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))

    def read():
        raw = canonical_json_bytes(ledger)
        compressed = gzip.compress(raw, mtime=0)
        metrics["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / "ledger.json.gz").write_bytes(compressed)
        (tmp_path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": metrics,
                    "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
        )
        return read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
        )

    assert read() == metrics
    ledger[final_field] += 100.0
    metrics[metric_field] += 100.0
    with pytest.raises(ValueError, match="native flow"):
        read()


@pytest.mark.parametrize(
    "mutation",
    [
        "terminal_quantity",
        "terminal_cash",
        "cash_chain",
        "termination",
        "interval_return",
    ],
)
def test_readback_rejects_contradictory_terminal_evidence(tmp_path, mutation):
    import json

    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, z = study.evaluate_cell(ds, "BH25", "normal", start=0, stop=ds.n_bars - 1)
    ledger = json.loads(gzip.decompress(z))
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": tmp_path.name}
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))

    def read():
        raw = canonical_json_bytes(ledger)
        compressed = gzip.compress(raw, mtime=0)
        metrics["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / "ledger.json.gz").write_bytes(compressed)
        (tmp_path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": metrics,
                    "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
        )
        return read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
        )

    assert read() == metrics
    rows = ledger["intervals"]
    if mutation == "terminal_quantity":
        assert rows[-1]["exact_quantities_after"] == ["0"]
        assert float(rows[-1]["portfolio_value_before"]) > 0
        rows[-1]["exact_quantities_after"] = ["1"]
    elif mutation == "terminal_cash":
        ledger["final_cash"] += 100.0
    elif mutation == "cash_chain":
        rows[1]["cash_before"] += 100.0
    elif mutation == "termination":
        rows[-1]["termination_reason"] = "synthetic_contradiction"
    else:
        rows[1]["interval_net_return"] += 0.25
        returns = np.asarray([row["interval_net_return"] for row in rows])
        metrics["sharpe_hourly"] = float(
            np.mean(returns) / np.std(returns) * np.sqrt(8760)
        )
    with pytest.raises(ValueError, match="native (terminal|cash|interval)"):
        read()


@pytest.mark.parametrize(
    "mutation",
    ["cash_relabel", "hold_initial_weight", "hold_rebalance", "hold_missing_decision"],
)
def test_readback_rejects_contradictory_control_identity(tmp_path, mutation):
    import json

    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, z = study.evaluate_cell(ds, "BH25", "normal", start=0, stop=ds.n_bars - 1)
    ledger = json.loads(gzip.decompress(z))
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": tmp_path.name}
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))

    def read():
        raw = canonical_json_bytes(ledger)
        compressed = gzip.compress(raw, mtime=0)
        metrics["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / "ledger.json.gz").write_bytes(compressed)
        (tmp_path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": metrics,
                    "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
        )
        return read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm=metrics["arm"],
            scenario="normal",
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
        )

    assert read() == metrics
    if mutation == "cash_relabel":
        metrics["arm"] = "CASH"
    elif mutation == "hold_initial_weight":
        ledger["portfolio_decisions"][0]["proposal_weights"] = [0.5]
    elif mutation == "hold_missing_decision":
        ledger["portfolio_decisions"].pop()
    else:
        assert ledger["portfolio_decisions"][1]["requested_hold_symbols"] == [0]
        ledger["portfolio_decisions"][1]["requested_hold_symbols"] = []
    with pytest.raises(ValueError, match="native control"):
        read()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate_index",
        "target_shape",
        "hold_roster",
        "target_over_cap",
        "unsolicited_hold",
        "changed_weight_hold",
        "dropped_unchanged_hold",
    ],
)
def test_readback_rejects_noncontrol_decision_contradictions(tmp_path, mutation):
    import json

    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, z = study.evaluate_cell(ds, "S2", "normal", start=0, stop=ds.n_bars - 1)
    ledger = json.loads(gzip.decompress(z))
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": tmp_path.name}
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))

    def read():
        raw = canonical_json_bytes(ledger)
        compressed = gzip.compress(raw, mtime=0)
        metrics["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / "ledger.json.gz").write_bytes(compressed)
        (tmp_path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": metrics,
                    "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
        )
        return read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="S2",
            scenario="normal",
            expected_start=0,
            expected_stop=ds.n_bars - 1,
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
        )

    assert read() == metrics
    decisions = ledger["portfolio_decisions"]
    if mutation == "missing":
        decisions.pop()
    elif mutation == "duplicate_index":
        decisions[1]["index"] = decisions[0]["index"]
    elif mutation == "target_shape":
        decisions[0]["target_weights"] = []
    elif mutation == "target_over_cap":
        decisions[0]["target_weights"] = [1.5]
    elif mutation == "unsolicited_hold":
        decisions[0]["requested_hold_symbols"] = []
        decisions[0]["effective_hold_symbols"] = [0]
    elif mutation == "changed_weight_hold":
        decisions[0]["requested_hold_symbols"] = [0]
        decisions[0]["effective_hold_symbols"] = [0]
        decisions[0]["proposal_weights"] = [0.0]
        decisions[0]["target_weights"] = [0.5]
    elif mutation == "dropped_unchanged_hold":
        decisions[0]["requested_hold_symbols"] = [0]
        decisions[0]["effective_hold_symbols"] = []
        decisions[0]["proposal_weights"] = [0.0]
        decisions[0]["target_weights"] = [0.0]
    else:
        decisions[0]["requested_hold_symbols"] = [1]
    with pytest.raises(ValueError, match="native decision"):
        read()


@pytest.mark.parametrize(
    "mutation",
    [
        "dataset",
        "symbol",
        "policy",
        "processing",
        "malformed",
        "sequence_duplicate",
        "sequence_gap",
        "sequence_reordered",
    ],
)
def test_readback_rejects_order_event_identity(tmp_path, mutation):
    import json

    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, z = study.evaluate_cell(ds, "BH25", "normal", start=0, stop=ds.n_bars - 1)
    ledger = json.loads(gzip.decompress(z))
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": tmp_path.name}
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))

    def read():
        raw = canonical_json_bytes(ledger)
        compressed = gzip.compress(raw, mtime=0)
        metrics["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / "ledger.json.gz").write_bytes(compressed)
        (tmp_path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": metrics,
                    "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
        )
        return read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
            expected_start=0,
            expected_stop=ds.n_bars - 1,
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
        )

    assert read() == metrics
    row = ledger["intervals"][0]
    event = next(e for e in row["order_events"] if e["event_type"] == "filled")
    if mutation == "dataset":
        event["dataset_id"] = "different-dataset"
    elif mutation == "symbol":
        event["symbol_index"] = 1
    elif mutation == "policy":
        event["execution_policy_digest"] = "f" * 64
    elif mutation == "processing":
        event["processing_index"] = row["next_index"] + 1
    elif mutation == "sequence_duplicate":
        event["sequence"] = row["order_events"][0]["sequence"]
    elif mutation == "sequence_gap":
        event["sequence"] += 1
    elif mutation == "sequence_reordered":
        row["order_events"][0], row["order_events"][1] = (
            row["order_events"][1],
            row["order_events"][0],
        )
    else:
        del event["execution_price"]
    with pytest.raises(ValueError, match="native order event"):
        read()


@pytest.mark.parametrize(
    "mutation", ["drawdown_over_one", "negative_turnover", "bool_window"]
)
def test_readback_rejects_impossible_native_bounds(tmp_path, mutation):
    import json

    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, z = study.evaluate_cell(ds, "BH25", "normal", start=0, stop=ds.n_bars - 1)
    ledger = json.loads(gzip.decompress(z))
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": tmp_path.name}
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))

    def read():
        raw = canonical_json_bytes(ledger)
        compressed = gzip.compress(raw, mtime=0)
        metrics["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / "ledger.json.gz").write_bytes(compressed)
        (tmp_path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": metrics,
                    "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
        )
        return read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm="BH25",
            scenario="normal",
            expected_start=0,
            expected_stop=ds.n_bars - 1,
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
        )

    assert read() == metrics
    if mutation == "drawdown_over_one":
        ledger["intervals"][-1]["max_drawdown_after"] = 1.5
        ledger["final_max_drawdown"] = 1.5
        metrics["max_drawdown_pct"] = 150.0
    elif mutation == "negative_turnover":
        ledger["intervals"][-1]["turnover_total_after"] = -1.0
        ledger["final_turnover_total"] = -1.0
        metrics["turnover"] = -1.0
    else:
        ledger["start_index"] = False
    with pytest.raises(ValueError, match="native (drawdown|flow|ledger window)"):
        read()


@pytest.mark.parametrize("arm", ["CASH", "BH25"])
def test_readback_rejects_relabelled_double_cost_as_normal(tmp_path, arm):
    from tools.cloud_regime_switch_summary import read_cell
    from trade_rl.artifacts import canonical_json_bytes

    ds = _native_market()
    metrics, compressed = study.evaluate_cell(
        ds, arm, "double_cost", start=0, stop=ds.n_bars - 1
    )
    metrics["scenario"] = "normal"
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": tmp_path.name}
    (tmp_path / "attempt.json").write_bytes(canonical_json_bytes(attempt))
    (tmp_path / "ledger.json.gz").write_bytes(compressed)
    (tmp_path / "result.json").write_bytes(
        canonical_json_bytes(
            {
                **attempt,
                "dataset_id": ds.dataset_id,
                "metrics": metrics,
                "ledger_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
            }
        )
    )
    with pytest.raises(ValueError, match="execution policy"):
        read_cell(
            tmp_path,
            admission=admission,
            dataset_id=ds.dataset_id,
            arm=arm,
            scenario="normal",
            expected_symbol=ds.symbols[0],
            expected_execution_policy_digest=_expected_policy(ds, "normal"),
            expected_start=0,
            expected_stop=ds.n_bars - 1,
        )


@pytest.mark.parametrize("present", [False, True])
def test_independent_funding_presence_from_published_manifest_only(
    tmp_path, monkeypatch, present
):
    import json

    from tests.data.test_market_dataset_artifact import _dataset, _dataset_v7
    from trade_rl.data import write_market_dataset_files

    ds = _dataset_v7() if present else _dataset()
    source = tmp_path / "synthetic-artifact"
    write_market_dataset_files(source, ds)
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    bound = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_digest": manifest["artifact_digest"],
        "dataset_id": ds.dataset_id,
        "arrays_npz_sha256": manifest["arrays_digest"],
    }
    metadata_root = tmp_path / "metadata-only"
    symbol_dir = metadata_root / ds.symbols[0]
    symbol_dir.mkdir(parents=True)
    (symbol_dir / "manifest.json").write_bytes(raw)
    monkeypatch.setattr(study, "INPUT_ROOT", metadata_root)
    assert not (symbol_dir / "arrays.npz").exists()
    assert (
        study._read_frozen_funding_presence(
            {"datasets": {ds.symbols[0]: bound}}, ds.symbols[0]
        )
        is present
    )
    with pytest.raises(ValueError, match="manifest digest"):
        study._verified_funding_price_rate_present(
            raw + b" ", bound=bound, symbol=ds.symbols[0]
        )
    with pytest.raises(ValueError, match="manifest identity"):
        study._verified_funding_price_rate_present(raw, bound=bound, symbol="ETHUSDT")


@pytest.mark.parametrize(
    "mutation", ["funding_schema", "scalar_shadow", "missing_identity"]
)
def test_pinned_manifest_refuses_contradictory_metadata(tmp_path, mutation):
    import json

    from tests.data.test_market_dataset_artifact import _dataset_v7
    from trade_rl.artifacts import canonical_json_bytes, content_digest
    from trade_rl.data import write_market_dataset_files

    ds = _dataset_v7()
    write_market_dataset_files(tmp_path / "artifact", ds)
    manifest = json.loads((tmp_path / "artifact" / "manifest.json").read_bytes())
    if mutation == "funding_schema":
        manifest["arrays"].pop("funding_price_rate")
    elif mutation == "scalar_shadow":
        manifest["dataset"]["funding_price_rate"] = None
    else:
        manifest["dataset"].pop("identity_payload_json")
    manifest.pop("artifact_digest")
    manifest["artifact_digest"] = content_digest(manifest)
    raw = canonical_json_bytes(manifest)
    bound = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_digest": manifest["artifact_digest"],
        "dataset_id": ds.dataset_id,
        "arrays_npz_sha256": manifest["arrays_digest"],
    }
    with pytest.raises(ValueError, match="frozen manifest"):
        study._verified_funding_price_rate_present(
            raw, bound=bound, symbol=ds.symbols[0]
        )


def test_metadata_only_year_summary_binds_native_exposure_and_scenario(
    tmp_path, monkeypatch
):
    """Actual generated native replay, then independent manifest/receipt-only readback."""
    import json

    from trade_rl.artifacts import canonical_json_bytes
    from trade_rl.data import write_market_dataset_files
    from trade_rl.data.market import MarketDataset

    p = study.protocol()
    n = 8761
    close = np.full((n, 1), 100.0)
    ds = MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2023-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.full((n, 1), 100000.0),
        funding_rate=np.zeros((n, 1)),
        funding_price_rate=np.zeros((n, 1)),
        tradable=np.ones((n, 1), dtype=bool),
        feature_available=np.ones((n, 1, 1), dtype=bool),
        feature_names=("synthetic",),
        global_feature_names=("synthetic",),
        periods_per_year=8760,
    ).with_content_identity({"source": "synthetic-summary-integration"})
    metadata = tmp_path / "metadata"
    source = metadata / "BTCUSDT"
    source.mkdir(parents=True)
    write_market_dataset_files(source, ds)
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    # Keep only independently pinned metadata. The summary must never
    # require the array file, load a Dataset, or execute another book.
    (source / "arrays.npz").unlink()
    p["datasets"] = {
        "BTCUSDT": {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "artifact_digest": manifest["artifact_digest"],
            "dataset_id": ds.dataset_id,
            "arrays_npz_sha256": manifest["arrays_digest"],
        }
    }
    p["symbols"] = ["BTCUSDT"]
    p["years"] = [2023]
    p["warmup_start"] = "2022-12-31T23:00:00Z"
    monkeypatch.setattr(study, "protocol", lambda: p)
    monkeypatch.setattr(study, "INPUT_ROOT", metadata)
    monkeypatch.setattr(study, "OUTPUT_ROOT", tmp_path / "results")
    path = study.OUTPUT_ROOT / "BTCUSDT_2023_CASH_normal"
    path.mkdir(parents=True)
    admission = {"synthetic_only": True}
    attempt = {"admission": admission, "cell": path.name}
    (path / "attempt.json").write_bytes(canonical_json_bytes(attempt))
    metrics, compressed = study.evaluate_cell(ds, "CASH", "normal", start=0, stop=n - 1)
    assert metrics["valid"] is True
    assert metrics["mean_pre_interval_gross"] == 0.0

    def publish(payload, ledger):
        (path / "ledger.json.gz").write_bytes(ledger)
        (path / "result.json").write_bytes(
            canonical_json_bytes(
                {
                    **attempt,
                    "dataset_id": ds.dataset_id,
                    "metrics": payload,
                    "ledger_gzip_sha256": hashlib.sha256(ledger).hexdigest(),
                }
            )
        )

    def forbidden(*args, **kwargs):
        pytest.fail("summary attempted market-array loading or another replay")

    monkeypatch.setattr(study, "load_published_market_dataset_artifact", forbidden)
    monkeypatch.setattr(study, "run_shared_portfolio_replay", forbidden)
    publish(metrics, compressed)
    result = study.summarize_recorded_cells(admission)
    assert result["recorded_cells"] == 1
    assert result["disposition"] == "INCONCLUSIVE"
    assert not (source / "arrays.npz").exists()

    publish({**metrics, "mean_pre_interval_gross": 1.0}, compressed)
    with pytest.raises(ValueError, match="mean_pre_interval_gross"):
        study.summarize_recorded_cells(admission)

    # Preserve the flat outcome and rehash every affected identity. The
    # trusted expected cost is independently derived from the fixed scenario.
    ledger = json.loads(gzip.decompress(compressed))
    wrong = _expected_policy(ds, "double_cost")
    assert wrong != ledger["execution_policy_digest"]
    ledger["execution_policy_digest"] = wrong
    for interval in ledger["intervals"]:
        for event in interval["order_events"]:
            event["execution_policy_digest"] = wrong
    forged = canonical_json_bytes(ledger)
    publish(
        {**metrics, "ledger_sha256": hashlib.sha256(forged).hexdigest()},
        gzip.compress(forged, mtime=0),
    )
    with pytest.raises(ValueError, match="execution policy"):
        study.summarize_recorded_cells(admission)
