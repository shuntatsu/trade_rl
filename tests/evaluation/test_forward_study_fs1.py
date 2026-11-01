"""Sealed FS1 evaluator on SYNTHETIC data only.

The frozen registry is never scored here: a scaled-down registry (same mechanism
builders, three synthetic symbols, a two-month window) is injected through the
evaluator's explicit ``registry`` seam. No real data, no economic claim.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import (
    SYMBOLS,
    small_protocol,
    synthetic_market,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.forward_study import cli, fs1, seal
from trade_rl.evaluation.forward_study.registry import (
    ARM_FS1_A,
    ARM_FS1_B,
    ARM_FS1_C,
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    Fs1Registry,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-222222222"
FETCH_AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-111111111"
SHA = "a" * 40
NOW = datetime(2027, 5, 2, tzinfo=UTC)

SYN: Fs1Registry = replace(
    FS1_REGISTRY,
    symbols=SYMBOLS,
    dataset_first_row_latest="2021-01-01T00:00:00",
    development_data_end="2021-08-01T00:00:00",
    warmup_only_start="2021-08-01T00:00:00",
    scoring_start="2021-10-04T00:00:00",
    scoring_end="2021-11-27T00:00:00",
    causality_cut="2021-10-25T00:00:00",
    base_protocol=small_protocol(),
)


@pytest.fixture(scope="module")
def dataset() -> MarketDataset:
    return synthetic_market().with_content_identity()


def manifest_for(data: MarketDataset) -> tuple[bytes, str]:
    raw = seal.manifest_bytes_from_dataset(
        data,
        source_manifest={"synthetic.zip": "1" * 64},
        build_policy_digest=SYN.build_policy_digest(),
        authorization_reference=FETCH_AUTH,
        fetched_utc="2027-05-02T01:00:00",
        sealed_utc="2027-05-02T02:00:00",
    )
    return raw, hashlib.sha256(raw).hexdigest()


def make_permit(data: MarketDataset) -> seal.ScoringPermit:
    """A permit for the synthetic registry through the private factory."""

    raw, digest = manifest_for(data)
    return seal._authorize(
        registry=SYN,
        frozen_params_digest=SYN.digest(),
        now=NOW,
        authorization=AUTH,
        expected_params_digest=SYN.digest(),
        sealed_code_sha=SHA,
        code_identity=seal.CodeIdentity(SHA, True),
        manifest_bytes=raw,
        expected_manifest_sha256=digest,
    )


def make_env(
    data: MarketDataset,
    *,
    clock: datetime = NOW,
    identity: seal.CodeIdentity | None = None,
    frozen: str | None = None,
    loader: Callable[[Path], MarketDataset] | None = None,
) -> fs1._Environment:
    return fs1._Environment(
        registry=SYN,
        frozen_params_digest=frozen or SYN.digest(),
        clock=lambda: clock,
        code_identity=lambda: identity or seal.CodeIdentity(SHA, True),
        load_dataset=loader or (lambda path: data),
    )


def run_kwargs(
    data: MarketDataset,
    out: Path,
    *,
    with_manifest: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    raw, digest = manifest_for(data)
    manifest = out.parent / "manifest.json"
    if with_manifest:
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(raw)
    kwargs: dict[str, Any] = {
        "authorization": AUTH,
        "expected_params_digest": SYN.digest(),
        "sealed_code_sha": SHA,
        "manifest_path": manifest,
        "expected_manifest_sha256": digest,
        "dataset_path": out.parent / "dataset",
        "output_path": out,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module")
def scored(
    dataset: MarketDataset, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, Any], Path]:
    out = tmp_path_factory.mktemp("fs1") / "result.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fs1, "_environment", lambda: make_env(dataset))
        result = fs1.run_fs1(**run_kwargs(dataset, out))
    return result, out


@pytest.fixture(scope="module")
def roster(
    dataset: MarketDataset,
) -> tuple[Any, dict[str, fs1.ArmRun]]:
    return fs1.run_roster(make_permit(dataset), dataset)


# --- the scoring path on synthetic data ------------------------------------


def test_scoring_a_sealed_synthetic_dataset_writes_a_complete_result_once(
    scored: tuple[dict[str, Any], Path],
) -> None:
    result, out = scored
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
        json.dumps(result, allow_nan=True)
    )
    assert result["schema_version"] == "forward_study_fs1_result_v1"
    assert result["scoring_authorization"] == AUTH
    assert result["fetch_authorization"] == FETCH_AUTH
    assert result["params_digest"] == SYN.digest() and result["code_sha"] == SHA
    assert set(result["arms"]) == {arm.fs1_id for arm in SYN.arms}
    for arm, scenarios in result["arms"].items():
        assert set(scenarios) == {"base", "cost_x2", "latency_1bar"}, arm
    assert set(result["labels"]) == {ARM_FS1_A, ARM_FS1_B}
    for row in result["labels"].values():
        assert row["label"] in SYN.labels
    assert "WINNER" not in json.dumps(result["labels"])
    assert result["controls_valid"] is True
    assert all(not reasons for reasons in result["invalid_reasons"].values())


def test_every_scenario_runs_on_one_continuous_account_over_the_whole_window(
    scored: tuple[dict[str, Any], Path],
) -> None:
    result, _ = scored
    for scenarios in result["arms"].values():
        for summary in scenarios.values():
            assert summary["reconciliation"]["passed"] is True
            assert summary["reconciliation"]["terminal_flat"] is True
            assert summary["terminated"] == []
            assert summary["n_daily_returns"] == 54
    assert result["arms"]["cash"]["base"]["net_pnl_usdt"] == 0.0
    base = result["arms"][ARM_FS1_A]["base"]
    assert base["net_pnl_usdt"] != 0.0


def test_the_causality_matrix_is_identical_for_every_non_cash_arm_on_synthetic_data(
    scored: tuple[dict[str, Any], Path],
) -> None:
    result, _ = scored
    matrix = result["causality_matrix"]
    assert set(matrix) == {ARM_FS1_A, ARM_FS1_B, ARM_FS1_C}
    for arm, cells in matrix.items():
        assert set(cells) == {"base", "cost_x2", "latency_1bar"}
        for scenario, cell in cells.items():
            assert cell["identical_pre_cut_trace"] is True, (arm, scenario)


def test_bootstrap_uses_the_frozen_pairing_and_level(
    scored: tuple[dict[str, Any], Path],
) -> None:
    result, _ = scored
    row = result["bootstrap"]["cost_x2"][ARM_FS1_A]
    assert set(row) == {"vs_cash", "vs_passive"}
    interval = row["vs_cash"]
    assert interval["ci95_low_annualized"] <= interval["annualized_mean_excess"]
    assert interval["annualized_mean_excess"] <= interval["ci95_high_annualized"]
    # Bonferroni k=2 is wider than the unadjusted 95% interval.
    assert interval["bonferroni_ci_low_annualized"] <= interval["ci95_low_annualized"]
    assert "vs_passive" not in result["bootstrap"]["cost_x2"][ARM_FS1_C]
    label = result["labels"][ARM_FS1_A]
    assert (label["label"] == "FORWARD_PASS") == (
        interval["ci95_low_annualized"] > 0.0
        and result["arms"][ARM_FS1_A]["base"]["max_drawdown_pct"] <= 20.0
        and result["arms"][ARM_FS1_A]["cost_x2"]["max_drawdown_pct"] <= 20.0
    )


def test_the_result_is_write_once_and_a_second_run_never_loads_data(
    dataset: MarketDataset,
    scored: tuple[dict[str, Any], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, out = scored
    before = out.read_bytes()
    loads: list[int] = []

    def loader(path: Path) -> MarketDataset:
        loads.append(1)
        return dataset

    monkeypatch.setattr(fs1, "_environment", lambda: make_env(dataset, loader=loader))
    with pytest.raises(seal.Fs1Refusal) as caught:
        fs1.run_fs1(**run_kwargs(dataset, out))
    assert caught.value.reasons == ("output_exists",)
    assert loads == [] and out.read_bytes() == before


# --- refusals never read a dataset and never write a result ----------------


@pytest.mark.parametrize(
    ("kwargs_override", "env_override", "reason"),
    [
        (
            {},
            {"clock": datetime(2021, 11, 27, 23, tzinfo=UTC)},
            "scoring_window_not_closed",
        ),
        (
            {},
            {"clock": datetime(2021, 11, 27, 0, tzinfo=UTC)},
            "scoring_window_not_closed",
        ),
        ({"authorization": None}, {}, "authorization_missing"),
        ({"authorization": "ok"}, {}, "authorization_malformed"),
        (
            {
                "authorization": "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1"
            },
            {},
            "authorization_malformed",
        ),
        ({"expected_params_digest": "0" * 64}, {}, "params_digest_mismatch"),
        ({"sealed_code_sha": "b" * 40}, {}, "code_sha_mismatch"),
        ({}, {"identity": seal.CodeIdentity(SHA, False)}, "code_tree_dirty"),
        ({}, {"identity": seal.CodeIdentity(None, False)}, "code_identity_unavailable"),
        ({"with_manifest": False}, {}, "manifest_missing"),
        ({"expected_manifest_sha256": "0" * 64}, {}, "manifest_sha256_mismatch"),
        ({}, {"frozen": "0" * 64}, "registry_digest_differs_from_frozen"),
    ],
)
def test_each_refusal_path_stops_before_loading_or_writing(
    dataset: MarketDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs_override: dict[str, Any],
    env_override: dict[str, Any],
    reason: str,
) -> None:
    out = tmp_path / "never.json"
    loads: list[int] = []

    def loader(path: Path) -> MarketDataset:
        loads.append(1)
        return dataset

    monkeypatch.setattr(
        fs1, "_environment", lambda: make_env(dataset, loader=loader, **env_override)
    )
    with pytest.raises(seal.Fs1Refusal) as caught:
        fs1.run_fs1(**run_kwargs(dataset, out, **kwargs_override))
    assert reason in caught.value.reasons
    assert loads == [] and not out.exists()


def test_the_public_entrypoint_always_uses_the_frozen_registry_and_digest() -> None:
    parameters = set(inspect.signature(fs1.run_fs1).parameters)
    assert parameters == {
        "authorization",
        "expected_params_digest",
        "sealed_code_sha",
        "manifest_path",
        "expected_manifest_sha256",
        "dataset_path",
        "output_path",
        "progress",
    }
    env = fs1._environment()
    assert env.registry is FS1_REGISTRY
    assert env.frozen_params_digest == FROZEN_PARAMS_DIGEST
    assert env.code_identity is seal.current_code_identity
    assert env.clock().tzinfo is not None


def test_the_real_registry_is_refused_by_the_real_entrypoint_before_the_window_closes(
    dataset: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic arguments against the FROZEN registry: refused, nothing read."""

    def no_read(path: Path) -> MarketDataset:
        raise AssertionError("the dataset must not be read after a refusal")

    real = fs1._environment()
    monkeypatch.setattr(
        fs1,
        "_environment",
        lambda: replace(
            real,
            clock=lambda: datetime(2026, 10, 30, tzinfo=UTC),
            code_identity=lambda: seal.CodeIdentity(SHA, True),
            load_dataset=no_read,
        ),
    )
    out = tmp_path / "frozen.json"
    with pytest.raises(seal.Fs1Refusal) as caught:
        fs1.run_fs1(**run_kwargs(dataset, out))
    assert "scoring_window_not_closed" in caught.value.reasons
    assert "params_digest_mismatch" in caught.value.reasons
    assert not out.exists()


def refuse_dataset(
    data: MarketDataset,
    loaded: MarketDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, ...]:
    out = tmp_path / "dataset-refusal.json"
    monkeypatch.setattr(
        fs1, "_environment", lambda: make_env(data, loader=lambda path: loaded)
    )
    with pytest.raises(seal.Fs1Refusal) as caught:
        fs1.run_fs1(**run_kwargs(data, out))
    assert not out.exists()
    return caught.value.reasons


def test_a_dataset_that_is_not_the_sealed_one_is_refused(
    dataset: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = synthetic_market(seed=22).with_content_identity()
    assert "dataset_id_mismatch" in refuse_dataset(
        dataset, other, tmp_path, monkeypatch
    )


def test_a_dataset_without_verified_identity_is_refused(
    dataset: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unverified = replace(dataset, identity_payload_json=None)
    assert "dataset_identity_unverified" in refuse_dataset(
        dataset, unverified, tmp_path, monkeypatch
    )


def test_a_dataset_with_edited_arrays_is_refused(
    dataset: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered_volume = dataset.volume.copy()
    tampered_volume[100] *= 1.5
    # The dataset type itself rejects an edit that keeps the sealed identity ...
    with pytest.raises(ValueError, match="dataset_id does not match"):
        replace(dataset, volume=tampered_volume)
    # ... and an edit that drops the identity payload is refused by the evaluator.
    stripped = replace(dataset, volume=tampered_volume, identity_payload_json=None)
    assert "dataset_identity_unverified" in refuse_dataset(
        dataset, stripped, tmp_path, monkeypatch
    )


def test_dataset_checks_cover_roster_window_cadence_and_tradability(
    dataset: MarketDataset,
) -> None:
    permit = make_permit(dataset)
    assert seal.dataset_problems(permit, dataset) == ()
    swapped = replace(dataset, symbols=tuple(reversed(dataset.symbols)))
    assert "dataset_symbols_mismatch" in seal.dataset_problems(permit, swapped)
    shifted = replace(permit, registry=replace(SYN, scoring_end="2021-11-26T00:00:00"))
    assert "dataset_last_row_mismatch" in seal.dataset_problems(shifted, dataset)
    gappy = np.asarray(dataset.tradable).copy()
    window = (dataset.timestamps >= np.datetime64("2021-08-01")) & (
        dataset.timestamps < np.datetime64("2021-11-27")
    )
    rows = np.flatnonzero(window)[::10]
    gappy[rows, 1] = False
    holes = replace(dataset, tradable=gappy, identity_payload_json=None)
    problems = seal.dataset_problems(permit, holes)
    assert f"dataset_tradable_fraction_low_{SYMBOLS[1]}" in problems
    assert all(SYMBOLS[0] not in problem for problem in problems)


def test_scoring_requires_a_permit_not_just_arguments(dataset: MarketDataset) -> None:
    with pytest.raises(TypeError):
        fs1.score_fs1(object(), dataset)  # type: ignore[arg-type]


# --- validity and labels over the finished runs ----------------------------


def assemble(
    roster: tuple[Any, dict[str, fs1.ArmRun]], mutate: Callable[[dict], None]
) -> dict[str, Any]:
    protocol, runs = roster
    copied = copy.deepcopy(runs)
    mutate(copied)
    return fs1.assemble_result(copied, protocol, SYN)


def test_the_unmutated_roster_has_valid_arms_and_a_label_per_candidate(
    roster: tuple[Any, dict[str, fs1.ArmRun]],
) -> None:
    out = assemble(roster, lambda runs: None)
    assert out["controls_valid"] is True
    assert set(out["labels"]) == {ARM_FS1_A, ARM_FS1_B}
    assert all(not reasons for reasons in out["invalid_reasons"].values())


def test_a_failed_replay_makes_that_candidate_invalid(
    roster: tuple[Any, dict[str, fs1.ArmRun]],
) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        del runs[ARM_FS1_A].outcomes["cost_x2"]
        runs[ARM_FS1_A].failures["cost_x2"] = {"status": "FAILED_TO_RUN"}

    out = assemble(roster, mutate)
    assert out["labels"][ARM_FS1_A]["label"] == "FORWARD_INVALID"
    assert "replay_failed_cost_x2" in out["invalid_reasons"][ARM_FS1_A]
    assert out["labels"][ARM_FS1_B]["label"] != "FORWARD_INVALID"
    assert out["arms"][ARM_FS1_A]["cost_x2"] == {"status": "FAILED_TO_RUN"}


def test_an_unreconciled_ledger_or_termination_is_invalid(
    roster: tuple[Any, dict[str, fs1.ArmRun]],
) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        runs[ARM_FS1_B].outcomes["base"].summary["reconciliation"]["passed"] = False
        runs[ARM_FS1_A].outcomes["latency_1bar"].summary["terminated"] = ["margin"]

    out = assemble(roster, mutate)
    assert "ledger_not_reconciled_base" in out["invalid_reasons"][ARM_FS1_B]
    assert "account_terminated_latency_1bar" in out["invalid_reasons"][ARM_FS1_A]
    assert {row["label"] for row in out["labels"].values()} == {"FORWARD_INVALID"}


def test_a_short_equity_path_is_incomplete_and_invalid(
    roster: tuple[Any, dict[str, fs1.ArmRun]],
) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        outcome = runs[ARM_FS1_A].outcomes["base"]
        outcome.equity = outcome.equity[:-48]
        outcome.timestamps_ns = outcome.timestamps_ns[:-48]

    out = assemble(roster, mutate)
    assert "incomplete_daily_equity_base" in out["invalid_reasons"][ARM_FS1_A]
    assert out["labels"][ARM_FS1_A]["label"] == "FORWARD_INVALID"


def test_pipeline_degradation_guards_make_the_arm_invalid(
    roster: tuple[Any, dict[str, fs1.ArmRun]],
) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        runs[ARM_FS1_A].outcomes["base"].summary["n_trades"] = 0
        decisions = runs[ARM_FS1_B].outcomes["cost_x2"].summary["allocator_decisions"]
        decisions["fail_closed_hold_count"] = decisions["n_decisions"]

    out = assemble(roster, mutate)
    assert "too_few_trades_base" in out["invalid_reasons"][ARM_FS1_A]
    assert "fail_closed_fraction_high_cost_x2" in out["invalid_reasons"][ARM_FS1_B]


def test_a_non_identical_causality_cell_is_invalid(
    roster: tuple[Any, dict[str, fs1.ArmRun]],
) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        matrix = runs[ARM_FS1_B].probe_matrix
        assert matrix is not None
        matrix["cost_x2"]["identical_pre_cut_trace"] = False
        runs[ARM_FS1_A].probe_matrix = None

    out = assemble(roster, mutate)
    assert "causality_probe_not_identical_cost_x2" in out["invalid_reasons"][ARM_FS1_B]
    assert "causality_matrix_missing" in out["invalid_reasons"][ARM_FS1_A]


def test_a_summary_without_allocator_decisions_is_invalid_not_a_silent_pass(
    roster: tuple[Any, dict[str, fs1.ArmRun]],
) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        for scenario in ("base", "latency_1bar"):
            del runs[ARM_FS1_A].outcomes[scenario].summary["allocator_decisions"]
        del runs[ARM_FS1_C].outcomes["base"].summary["allocator_decisions"]

    out = assemble(roster, mutate)
    assert "allocator_decisions_missing_base" in out["invalid_reasons"][ARM_FS1_A]
    assert (
        "allocator_decisions_missing_latency_1bar" in out["invalid_reasons"][ARM_FS1_A]
    )
    assert out["labels"][ARM_FS1_A]["label"] == "FORWARD_INVALID"
    # the passive control is held to the same rule, which invalidates every candidate
    assert "allocator_decisions_missing_base" in out["invalid_reasons"][ARM_FS1_C]
    assert out["labels"][ARM_FS1_B]["label"] == "FORWARD_INVALID"
    # CASH has no allocator and is not required to expose decisions
    assert not any("allocator" in r for r in out["invalid_reasons"]["cash"])


@pytest.mark.parametrize("control", ["cash", ARM_FS1_C])
def test_an_invalid_control_invalidates_every_candidate(
    roster: tuple[Any, dict[str, fs1.ArmRun]], control: str
) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        runs[control].outcomes["base"].summary["reconciliation"]["passed"] = False

    out = assemble(roster, mutate)
    assert out["controls_valid"] is False
    for label in out["labels"].values():
        assert label["label"] == "FORWARD_INVALID"
        assert "control_invalid" in label["invalid_reasons"]


def test_cash_must_be_exactly_flat(roster: tuple[Any, dict[str, fs1.ArmRun]]) -> None:
    def mutate(runs: dict[str, fs1.ArmRun]) -> None:
        runs["cash"].outcomes["base"].summary["net_pnl_usdt"] = 12.0

    out = assemble(roster, mutate)
    assert "cash_not_flat_base" in out["invalid_reasons"]["cash"]
    assert out["controls_valid"] is False


# --- G2: the causality probe has power on FS1 arms --------------------------


class LookaheadPolicy:
    """Deliberately non-causal: goes long what is higher 24 bars LATER."""

    def __init__(self, data: MarketDataset) -> None:
        self._close = data.close

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        ahead = min(snapshot.index + 24, self._close.shape[0] - 1)
        move = self._close[ahead] / self._close[snapshot.index] - 1.0
        weights = tuple(0.3 if value > 0.0 else -0.3 for value in move)
        return PortfolioTargetProposal(target_weights=weights)


def test_a_lookahead_arm_is_caught_as_invalid_by_the_fs1_probe(
    dataset: MarketDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_builder = fs1.build_fs1_arm

    def leaky(
        arm: str,
        data: MarketDataset,
        protocol: Any,
        context: Any,
        *,
        registry: Fs1Registry = FS1_REGISTRY,
    ) -> runner.BuiltArm:
        if arm == ARM_FS1_A:
            return runner.BuiltArm(LookaheadPolicy(data), None)
        return real_builder(arm, data, protocol, context, registry=registry)

    monkeypatch.setattr(fs1, "build_fs1_arm", leaky)
    permit = make_permit(dataset)
    protocol, runs = fs1.run_roster(permit, dataset)
    out = fs1.assemble_result(runs, protocol, SYN)
    reasons = out["invalid_reasons"][ARM_FS1_A]
    assert any(
        reason.startswith("causality_probe_not_identical_") for reason in reasons
    )
    assert out["labels"][ARM_FS1_A]["label"] == "FORWARD_INVALID"
    assert not out["invalid_reasons"][ARM_FS1_B]


# --- CLI ----------------------------------------------------------------------


def cli_args(tmp_path: Path, **overrides: str) -> list[str]:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}")
    values = {
        "--dataset": str(tmp_path / "no-such-dataset"),
        "--manifest": str(manifest),
        "--manifest-sha256": "0" * 64,
        "--authorization": AUTH,
        "--params-digest": FS1_REGISTRY.digest(),
        "--sealed-code-sha": SHA,
        "--output": str(tmp_path / "out" / "result.json"),
    }
    values.update(overrides)
    return [item for pair in values.items() for item in pair]


def real_registry_env(monkeypatch: pytest.MonkeyPatch, clock: datetime) -> None:
    """The frozen registry with a fake clock/git and a loader that must not run."""

    def no_read(path: Path) -> MarketDataset:
        raise AssertionError("the dataset must not be read after a refusal")

    real = fs1._environment()
    monkeypatch.setattr(
        fs1,
        "_environment",
        lambda: replace(
            real,
            clock=lambda: clock,
            code_identity=lambda: seal.CodeIdentity(SHA, True),
            load_dataset=no_read,
        ),
    )


def test_cli_refuses_with_exit_code_two_and_reads_no_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_registry_env(monkeypatch, datetime(2026, 10, 30, tzinfo=UTC))
    assert cli.main(cli_args(tmp_path)) == 2
    error = capsys.readouterr().err
    assert "scoring_window_not_closed" in error and "manifest" in error
    assert not (tmp_path / "out").exists()


def test_cli_without_a_manifest_file_is_a_refusal_not_a_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_registry_env(monkeypatch, NOW)
    args = cli_args(tmp_path, **{"--manifest": str(tmp_path / "missing.json")})
    assert cli.main(args) == 2
    assert "manifest_missing" in capsys.readouterr().err


def test_cli_passes_the_operator_values_through_and_prints_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"labels": {ARM_FS1_A: {"label": "FORWARD_FAIL"}}}

    monkeypatch.setattr(cli, "run_fs1", fake_run)
    assert cli.main(cli_args(tmp_path)) == 0
    assert "FORWARD_FAIL" in capsys.readouterr().out
    assert seen["authorization"] == AUTH and seen["sealed_code_sha"] == SHA
    assert seen["manifest_path"] == tmp_path / "manifest.json"
    assert seen["output_path"] == tmp_path / "out" / "result.json"
    assert set(seen) == set(inspect.signature(fs1.run_fs1).parameters)


def test_cli_main_has_no_test_seams() -> None:
    assert list(inspect.signature(cli.main).parameters) == ["argv"]


def test_cli_has_no_tunable_parameters() -> None:
    options = {
        action.dest for action in cli.build_parser()._actions if action.dest != "help"
    }
    assert options == {
        "dataset",
        "manifest",
        "manifest_sha256",
        "authorization",
        "params_digest",
        "sealed_code_sha",
        "output",
    }


# --- LOW-6: load errors and write races end as refusals, never partial output ----------


def test_a_dataset_load_error_is_a_refusal_and_leaves_no_output(
    dataset: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> MarketDataset:
        raise OSError("artifact unreadable")

    monkeypatch.setattr(fs1, "_environment", lambda: make_env(dataset, loader=broken))
    out = tmp_path / "result.json"
    with pytest.raises(seal.Fs1Refusal) as caught:
        fs1.run_fs1(**run_kwargs(dataset, out))
    assert caught.value.reasons == ("dataset_load_failed",)
    assert "OSError: artifact unreadable" in caught.value.detail
    assert not out.exists()


def test_a_failure_while_scoring_removes_the_claimed_output(
    dataset: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert out.exists(), "the output path is claimed before scoring starts"
        raise RuntimeError("scoring bug")

    monkeypatch.setattr(fs1, "_environment", lambda: make_env(dataset))
    monkeypatch.setattr(fs1, "score_fs1", explode)
    out = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="scoring bug"):
        fs1.run_fs1(**run_kwargs(dataset, out))
    assert not out.exists()


def test_a_concurrent_writer_is_refused_before_scoring_and_never_overwritten(
    dataset: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "result.json"
    real_exists = Path.exists
    loads: list[int] = []

    def loader(path: Path) -> MarketDataset:
        loads.append(1)
        return dataset

    # The pre-check sees no file (the race window); the file then appears.
    monkeypatch.setattr(
        Path, "exists", lambda self: False if self == out else real_exists(self)
    )
    monkeypatch.setattr(fs1, "_environment", lambda: make_env(dataset, loader=loader))
    kwargs = run_kwargs(dataset, out)
    out.write_text("someone else's result", encoding="utf-8")
    with pytest.raises(seal.Fs1Refusal) as caught:
        fs1.run_fs1(**kwargs)
    assert caught.value.reasons == ("output_exists",)
    assert loads == []
    assert out.read_text(encoding="utf-8") == "someone else's result"


def test_cli_turns_a_load_error_into_exit_two_with_a_clear_message(
    dataset: MarketDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(path: Path) -> MarketDataset:
        raise ValueError("digest mismatch in artifact")

    monkeypatch.setattr(fs1, "_environment", lambda: make_env(dataset, loader=broken))
    raw, digest = manifest_for(dataset)
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "out" / "result.json"
    args = cli_args(
        tmp_path,
        **{
            "--manifest": str(manifest),
            "--manifest-sha256": digest,
            "--params-digest": SYN.digest(),
            "--output": str(out),
        },
    )
    manifest.write_bytes(raw)  # cli_args wrote a placeholder; use the real manifest
    assert cli.main(args) == 2
    error = capsys.readouterr().err
    assert "dataset_load_failed" in error and "digest mismatch in artifact" in error
    assert not out.exists()


def test_cli_turns_a_write_race_into_exit_two(
    dataset: MarketDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw, digest = manifest_for(dataset)
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "out" / "result.json"
    out.parent.mkdir()
    out.write_text("first writer", encoding="utf-8")
    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda self: False if self == out else real_exists(self)
    )
    monkeypatch.setattr(fs1, "_environment", lambda: make_env(dataset))
    args = cli_args(
        tmp_path,
        **{
            "--manifest": str(manifest),
            "--manifest-sha256": digest,
            "--params-digest": SYN.digest(),
            "--output": str(out),
        },
    )
    manifest.write_bytes(raw)
    assert cli.main(args) == 2
    assert "output_exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "first writer"
