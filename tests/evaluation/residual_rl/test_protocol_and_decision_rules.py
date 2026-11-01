"""Frozen-protocol, decision-rule and provenance contracts (no PPO runtime needed)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.evaluation.residual_rl.cli import (
    REPORT_PREFIX,
    append_provenance,
    write_json,
)
from trade_rl.evaluation.residual_rl.env import EnvConfig
from trade_rl.evaluation.residual_rl.evaluate import (
    control_verdict,
    env_config,
    final_label,
    paired_bootstrap_median_lower,
    primary_verdict,
)
from trade_rl.evaluation.residual_rl.market import Regime
from trade_rl.evaluation.residual_rl.protocol import (
    FROZEN_PROTOCOL_SHA256,
    PROTOCOL_V1,
    DecisionRules,
    SeedPlan,
    protocol_digest,
    require_frozen_protocol,
)

RULES = PROTOCOL_V1.rules
ROOT = Path(__file__).resolve().parents[3]


def test_protocol_digest_is_frozen() -> None:
    assert protocol_digest() == FROZEN_PROTOCOL_SHA256
    assert require_frozen_protocol() == FROZEN_PROTOCOL_SHA256


def test_any_protocol_change_breaks_the_freeze() -> None:
    changed = replace(PROTOCOL_V1, kappa_primary=2.0)
    assert protocol_digest(changed) != FROZEN_PROTOCOL_SHA256
    with pytest.raises(RuntimeError, match="frozen digest"):
        require_frozen_protocol(changed)
    margin = replace(PROTOCOL_V1, rules=replace(RULES, margin=0.0))
    assert protocol_digest(margin) != FROZEN_PROTOCOL_SHA256


def test_committed_protocol_file_matches_the_source_digest() -> None:
    path = ROOT / "report" / f"{REPORT_PREFIX}-protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == FROZEN_PROTOCOL_SHA256
    assert payload["frozen_protocol_sha256_in_source"] == FROZEN_PROTOCOL_SHA256


def test_protocol_declares_the_pre_run_choices() -> None:
    assert PROTOCOL_V1.synthetic_only is True
    assert "blocked behind the fee-only learnability check" in (
        PROTOCOL_V1.real_data_rl_gate
    )
    assert PROTOCOL_V1.multipliers == (0.0, 0.5, 1.0, 1.5)
    assert 3 <= PROTOCOL_V1.market.n_assets <= 5
    assert PROTOCOL_V1.costs.fee_rate == 0.0005
    assert PROTOCOL_V1.costs.spread_rate == 0.0002
    assert PROTOCOL_V1.kappa_primary == 1.0 and PROTOCOL_V1.kappa_comparison == 3.0
    assert PROTOCOL_V1.ppo.total_timesteps % PROTOCOL_V1.ppo.n_steps == 0
    assert PROTOCOL_V1.ppo.n_steps % PROTOCOL_V1.ppo.batch_size == 0
    # The env built from the protocol exposes exactly this action set.
    config = env_config(PROTOCOL_V1, Regime.PLANTED_ALPHA)
    assert isinstance(config, EnvConfig)
    assert config.multipliers == PROTOCOL_V1.multipliers
    assert config.allocator.kappa == 1.0
    assert env_config(PROTOCOL_V1, Regime.ZERO_ALPHA, kappa=3.0).allocator.kappa == 3.0


def test_seed_pools_are_disjoint() -> None:
    plan = PROTOCOL_V1.seeds
    assert not set(plan.train) & set(plan.held_out)
    assert not set(plan.train) & set(plan.calibration)
    assert not set(plan.held_out) & set(plan.calibration)
    with pytest.raises(ValueError, match="disjoint"):
        SeedPlan(train=(1, 2), held_out=(2, 3), calibration=(4,))


def _advantage(deltas: list[float], n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    allocator = np.linspace(0.05, 0.25, n)
    residual = np.stack([allocator + delta for delta in deltas])
    return residual, allocator


def test_primary_passes_only_when_margin_sign_and_bootstrap_all_hold() -> None:
    residual, allocator = _advantage([0.03, 0.02, 0.025])
    verdict = primary_verdict(residual, allocator, RULES)
    assert verdict["passed"] is True
    assert verdict["margin_met"] and verdict["every_ppo_seed_positive"]

    below, allocator = _advantage([0.004, 0.003, 0.005])
    assert primary_verdict(below, allocator, RULES)["margin_met"] is False
    assert primary_verdict(below, allocator, RULES)["passed"] is False

    one_negative, allocator = _advantage([0.03, 0.03, -0.001])
    verdict = primary_verdict(one_negative, allocator, RULES)
    assert verdict["margin_met"] is True
    assert verdict["every_ppo_seed_positive"] is False
    assert verdict["passed"] is False


def test_primary_fails_when_the_bootstrap_lower_bound_is_not_positive() -> None:
    rng = np.random.default_rng(0)
    allocator = np.zeros(40)
    noise = rng.normal(0.0, 0.2, size=40)
    # Median advantage clears the margin only by chance in a very noisy sample.
    residual = np.stack([noise + 0.012] * 3)
    verdict = primary_verdict(residual, allocator, RULES)
    assert verdict["bootstrap_lower_bound_positive"] is False
    assert verdict["passed"] is False


def test_primary_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        primary_verdict(np.zeros(5), np.zeros(5), RULES)
    with pytest.raises(ValueError):
        primary_verdict(np.zeros((3, 5)), np.zeros(4), RULES)


def test_bootstrap_is_deterministic_and_ordered() -> None:
    values = np.linspace(-0.01, 0.05, 40)
    first = paired_bootstrap_median_lower(values, resamples=500, seed=1, quantile=0.05)
    again = paired_bootstrap_median_lower(values, resamples=500, seed=1, quantile=0.05)
    assert first == again
    assert first < float(np.median(values))
    with pytest.raises(ValueError):
        paired_bootstrap_median_lower(
            np.array([1.0]), resamples=10, seed=1, quantile=0.05
        )


def test_control_requires_low_turnover_and_losing_less() -> None:
    allocator_turnover = np.full(40, 0.40)
    allocator_returns = np.full(40, -0.05)
    low = np.full((3, 40), 0.05)
    better = np.full((3, 40), -0.01)
    passed = control_verdict(low, better, allocator_turnover, allocator_returns, RULES)
    assert passed["passed"] is True
    assert passed["median_turnover_ratio_to_allocator"] == pytest.approx(0.125)

    busy = control_verdict(
        np.full((3, 40), 0.2), better, allocator_turnover, allocator_returns, RULES
    )
    assert busy["turnover_ratio_met"] is False and busy["passed"] is False

    worse = control_verdict(
        low, np.full((3, 40), -0.09), allocator_turnover, allocator_returns, RULES
    )
    assert worse["loses_less_than_allocator"] is False and worse["passed"] is False

    idle_allocator = control_verdict(
        low, better, np.zeros(40), allocator_returns, RULES
    )
    assert idle_allocator["passed"] is False  # nothing to throttle: control is void


def test_label_is_pass_only_when_both_falsifiers_survive() -> None:
    assert final_label(True, True, RULES) == ("SYNTHETIC_PASS", [])
    label, reasons = final_label(False, True, RULES)
    assert label == "SYNTHETIC_FAIL" and reasons == [
        "F1_primary_advantage_not_established"
    ]
    label, reasons = final_label(True, False, RULES)
    assert label == "SYNTHETIC_FAIL" and reasons == [
        "F2_fee_only_learnability_control_failed"
    ]
    assert final_label(False, False, RULES)[0] == "SYNTHETIC_FAIL"
    assert DecisionRules().label_pass == "SYNTHETIC_PASS"


def test_report_json_is_lf_only_and_provenance_accumulates(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    digest = write_json(path, {"b": 1, "a": [1.5]})
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert len(digest) == 64

    provenance = tmp_path / "provenance.json"
    append_provenance(provenance, {"mode": "calibration", "evidence": False})
    append_provenance(provenance, {"mode": "smoke", "evidence": False})
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert [run["mode"] for run in payload["runs"]] == ["calibration", "smoke"]
    assert payload["protocol_id"] == PROTOCOL_V1.protocol_id
    assert b"\r" not in provenance.read_bytes()


def test_full_study_refuses_an_unfrozen_protocol_before_any_work() -> None:
    from trade_rl.evaluation.residual_rl.study import run_full_study

    with pytest.raises(RuntimeError, match="frozen digest"):
        run_full_study(replace(PROTOCOL_V1, kappa_comparison=4.0), workers=1)
