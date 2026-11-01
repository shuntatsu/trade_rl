"""FS1 registry: the frozen parameters, their digest and the committed export.

These are contract tests of a design recorded BEFORE the forward window exists. They
read no dataset and no result. Changing any frozen value changes the digest and fails
the pinned-digest test on purpose: that would be a new study, not an edit.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from trade_rl.evaluation.dev_realdata_study.phase3_registry import P3_TRIALS
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.forward_study import registry as fs1_registry
from trade_rl.evaluation.forward_study.registry import (
    ARM_FS1_A,
    ARM_FS1_B,
    ARM_FS1_C,
    FROZEN_PARAMS_DIGEST,
    FS1_ARMS,
    FS1_REGISTRY,
    LABELS,
    REGISTRY_REPORT_PATH,
    registry_json_bytes,
    registry_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Independent literal copy of the frozen digest: updating it is a deliberate re-seal.
PINNED_PARAMS_DIGEST = (
    "9350734ca6ba920ebfcb47cd7b81b633df5457bfcfe7da37eba6d96fab5f1c56"
)


def test_the_frozen_digest_is_pinned_twice() -> None:
    assert FS1_REGISTRY.digest() == PINNED_PARAMS_DIGEST
    assert FROZEN_PARAMS_DIGEST == PINNED_PARAMS_DIGEST


def test_the_roster_is_cash_two_candidates_and_one_passive_control() -> None:
    assert [arm.role for arm in FS1_ARMS] == [
        "cash_control",
        "primary_candidate",
        "secondary_candidate",
        "passive_control",
    ]
    assert FS1_REGISTRY.candidate_arms == (ARM_FS1_A, ARM_FS1_B)
    assert FS1_REGISTRY.passive_arm == ARM_FS1_C
    assert FS1_REGISTRY.bonferroni_k == 2
    assert len({arm.fs1_id for arm in FS1_ARMS}) == 4
    assert LABELS == ("FORWARD_PASS", "FORWARD_FAIL", "FORWARD_INVALID")


def test_each_arm_is_the_named_frozen_phase3_mechanism() -> None:
    by_arm = {spec.arm: spec for spec in P3_TRIALS}
    expected = {
        ARM_FS1_A: ("P3-6", 3.0, 0.12, False, "zblend_t2_t4"),
        ARM_FS1_B: ("P3-3", 1.0, 0.12, False, "t2_xsmom"),
        ARM_FS1_C: ("P3-7", 1.0, 0.12, False, None),
    }
    for fs1_id, (trial, kappa, target, long_only, source) in expected.items():
        arm = FS1_REGISTRY.arm(fs1_id)
        spec = by_arm[arm.phase3_arm]
        assert arm.lineage_trial_id == trial == spec.trial_id
        assert (spec.kappa, spec.vol_target, spec.long_only, spec.source) == (
            kappa,
            target,
            long_only,
            source,
        )
        assert spec.decision_interval_hours == 24
        assert spec.decision_anchor_hour_utc == 0
    assert FS1_REGISTRY.arm("cash").phase3_arm == "cash"
    with pytest.raises(ValueError):
        FS1_REGISTRY.arm("fs1_d")


def test_window_clock_and_account_are_the_frozen_ones() -> None:
    registry = FS1_REGISTRY
    assert registry.symbols == ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    assert registry.development_data_end == "2026-08-01T00:00:00"
    assert registry.warmup_only_start == "2026-08-01T00:00:00"
    assert registry.scoring_start == "2026-11-01T00:00:00"
    assert registry.scoring_end == "2027-05-01T00:00:00"
    assert registry.development_data_end < registry.scoring_start
    span = datetime.fromisoformat(registry.scoring_end) - datetime.fromisoformat(
        registry.scoring_start
    )
    assert span.days == 181
    protocol = registry.protocol("a" * 64)
    assert (protocol.evaluation_start, protocol.evaluation_end) == (
        registry.scoring_start,
        registry.scoring_end,
    )
    assert registry.scoring_start < protocol.causality_cut < registry.scoring_end
    assert protocol.initial_capital == 10_000.0
    assert (protocol.fee_rate, protocol.spread_rate) == (0.0005, 0.0002)
    assert protocol.max_participation_rate == 0.05
    assert protocol.cost_stress_multiplier == 2.0 and protocol.latency_stress_bars == 1
    assert protocol.refit_interval_days == 30 and protocol.ridge_alpha == 10.0
    assert (protocol.decision_interval_hours, protocol.decision_anchor_hour_utc) == (
        24,
        0,
    )
    assert (
        protocol.bootstrap_block_days,
        protocol.bootstrap_resamples,
        protocol.bootstrap_seed,
        protocol.bootstrap_alpha,
    ) == (20, 2000, 0, 0.05)
    assert protocol.bonferroni_comparisons == 2
    assert protocol.max_drawdown_limit == 0.20
    assert protocol.scenarios == ("base", "cost_x2", "latency_1bar")


def test_falsifier_constants() -> None:
    registry = FS1_REGISTRY
    assert registry.max_drawdown_limit_pct == 20.0
    assert registry.drawdown_scenarios == ("base", "cost_x2")
    assert registry.excess_scenario == "cost_x2"
    assert registry.reported_scenarios == ("base", "cost_x2", "latency_1bar")


def test_digest_ignores_the_dataset_id_but_not_any_frozen_value() -> None:
    assert FS1_REGISTRY.protocol("a" * 64) != FS1_REGISTRY.protocol("b" * 64)
    assert FS1_REGISTRY.digest() == PINNED_PARAMS_DIGEST
    changed = (
        replace(FS1_REGISTRY, scoring_end="2027-04-30T00:00:00"),
        replace(FS1_REGISTRY, max_drawdown_limit_pct=25.0),
        replace(FS1_REGISTRY, bonferroni_k=3),
        replace(FS1_REGISTRY, causality_cut="2026-12-02T00:00:00"),
        replace(FS1_REGISTRY, min_trades=2),
        replace(FS1_REGISTRY, base_protocol=replace(PROTOCOL, gamma=6.0)),
        replace(FS1_REGISTRY, base_protocol=replace(PROTOCOL, fee_rate=0.0006)),
        replace(FS1_REGISTRY, arms=FS1_ARMS[:3]),
    )
    digests = {candidate.digest() for candidate in changed}
    assert PINNED_PARAMS_DIGEST not in digests
    assert len(digests) == len(changed)


def test_a_changed_phase3_mechanism_or_constant_changes_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.dev_realdata_study.trials import Phase2Constants

    monkeypatch.setattr(
        fs1_registry, "CONSTANTS", replace(Phase2Constants(), xsmom_lookback_days=60)
    )
    assert FS1_REGISTRY.digest() != PINNED_PARAMS_DIGEST


def test_the_committed_registry_json_is_the_exact_export() -> None:
    path = REPO_ROOT / REGISTRY_REPORT_PATH
    assert path.read_bytes() == registry_json_bytes()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["params_digest"] == PINNED_PARAMS_DIGEST
    assert {k: payload[k] for k in registry_payload()} == registry_payload()
    assert payload["scoring_window"] == ["2026-11-01T00:00:00", "2027-05-01T00:00:00"]
    assert "WINNER" not in json.dumps(payload["labels"])


def test_claim_limits_disclose_selection_contamination_and_low_power() -> None:
    text = " ".join(FS1_REGISTRY.claim_limits)
    for required in (
        "selected from public development",
        "cannot re-select",
        "not demonstrated",
        "BETTER_THAN_PASSIVE",
        "no live or paper order",
    ):
        assert required in text


def test_registry_module_reads_no_clock_dataset_or_result() -> None:
    source = (
        REPO_ROOT / "trade_rl" / "evaluation" / "forward_study" / "registry.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "datetime.now",
        "time.time",
        " open(",
        ".read_bytes",
        "load_dataset",
        "load_published",
    ):
        assert forbidden not in source


PINNED_BUILD_POLICY_DIGEST = (
    "13d404e9a8b0a87d02cf7a90f9af2bbbad412d439dae42f730e4f416efd03a34"
)


def test_the_dataset_build_policy_is_pinned_and_part_of_the_params_digest() -> None:
    assert FS1_REGISTRY.build_policy_digest() == PINNED_BUILD_POLICY_DIGEST
    policy = FS1_REGISTRY.build_policy
    assert policy["gap_policy"] == "non_tradable_rows"
    assert policy["base_interval"] == "1h" and policy["market"] == "usds_m"
    assert policy["start_time"] == "2021-01-01T00:00:00+00:00"
    assert policy["end_time"] == "2027-05-01T00:00:00+00:00"
    assert tuple(policy["symbols"]) == FS1_REGISTRY.symbols  # type: ignore[arg-type]
    assert "dataset_build" in str(policy["builder"])
    assert "never while scoring" in str(policy["network"])
    changed = replace(
        FS1_REGISTRY, build_policy={**policy, "gap_policy": "forward_fill_tradable"}
    )
    assert changed.build_policy_digest() != PINNED_BUILD_POLICY_DIGEST
    assert changed.digest() != PINNED_PARAMS_DIGEST


def test_the_settlement_margin_is_frozen_and_changes_the_digest() -> None:
    assert FS1_REGISTRY.settlement_margin_hours == 24
    assert (
        replace(FS1_REGISTRY, settlement_margin_hours=0).digest()
        != PINNED_PARAMS_DIGEST
    )
