"""The committed result and provenance files are consistent with the frozen rules."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.evaluation.residual_rl.cli import REPORT_PREFIX
from trade_rl.evaluation.residual_rl.evaluate import (
    control_verdict,
    final_label,
    primary_verdict,
)
from trade_rl.evaluation.residual_rl.protocol import (
    FROZEN_PROTOCOL_SHA256,
    PROTOCOL_V1,
)

REPORT = Path(__file__).resolve().parents[3] / "report"
RESULTS = REPORT / f"{REPORT_PREFIX}-results.json"
PROVENANCE = REPORT / f"{REPORT_PREFIX}-provenance.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stack(regimes: Any, regime: str, key: str) -> np.ndarray:
    return np.stack(
        [
            np.asarray(regimes[regime][f"RESIDUAL_PPO_seed{seed}"][key])
            for seed in PROTOCOL_V1.ppo.seeds
        ]
    )


def test_results_label_is_reproduced_from_the_stored_per_seed_numbers() -> None:
    results = _load(RESULTS)
    assert results["protocol_sha256"] == FROZEN_PROTOCOL_SHA256
    assert results["held_out_seeds"] == list(PROTOCOL_V1.seeds.held_out)
    regimes = results["regimes"]
    planted = regimes["planted_alpha"]
    zero = regimes["zero_alpha"]
    primary = primary_verdict(
        _stack(regimes, "planted_alpha", "net_log_return_by_seed"),
        np.asarray(planted["ALLOCATOR_KAPPA_1"]["net_log_return_by_seed"]),
        PROTOCOL_V1.rules,
    )
    control = control_verdict(
        _stack(regimes, "zero_alpha", "daily_turnover_by_seed"),
        _stack(regimes, "zero_alpha", "net_log_return_by_seed"),
        np.asarray(zero["ALLOCATOR_KAPPA_1"]["daily_turnover_by_seed"]),
        np.asarray(zero["ALLOCATOR_KAPPA_1"]["net_log_return_by_seed"]),
        PROTOCOL_V1.rules,
    )
    assert primary == results["primary_planted_alpha"]
    assert control == results["control_zero_alpha_fee_only"]
    label, reasons = final_label(
        bool(primary["passed"]), bool(control["passed"]), PROTOCOL_V1.rules
    )
    assert results["label"] == label
    assert results["label_reasons"] == reasons
    assert label in {"SYNTHETIC_PASS", "SYNTHETIC_FAIL"}
    assert "Never a real-data profitability claim" in str(results["label_scope"])


def test_cash_baseline_is_exactly_flat() -> None:
    regimes = _load(RESULTS)["regimes"]
    for regime in ("planted_alpha", "zero_alpha"):
        cash = regimes[regime]["CASH"]
        assert set(cash["net_log_return_by_seed"]) == {0.0}
        assert set(cash["daily_turnover_by_seed"]) == {0.0}


def test_provenance_lists_every_run_and_binds_the_results_file() -> None:
    provenance = _load(PROVENANCE)
    runs = provenance["runs"]
    modes = [run["mode"] for run in runs]
    assert "scratch_precommit_calibration" in modes
    assert modes.count("scratch_precommit_ppo_pilot") == 2
    assert "smoke" in modes
    evidence = [run for run in runs if run.get("evidence") is True]
    assert len(evidence) == 1 and evidence[0]["mode"] == "full"
    full = evidence[0]
    assert all(run["synthetic_only"] is True for run in runs)
    # Scratch and smoke runs precede the single evidence run in the file.
    assert runs[-1] is not None and runs[-1]["mode"] == "full"
    raw = RESULTS.read_bytes()
    assert full["results_file_sha256"] == sha256(raw).hexdigest()
    assert full["label"] == _load(RESULTS)["label"]
    assert full["protocol_sha256_computed"] == FROZEN_PROTOCOL_SHA256
    assert full["protocol_sha256_frozen_in_source"] == FROZEN_PROTOCOL_SHA256
    assert full["git"]["dirty_outside_report_files"] is False
    assert full["git"]["commit"] == _load(RESULTS)["git"]["commit"]
    assert b"\r" not in raw
