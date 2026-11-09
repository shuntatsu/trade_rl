"""Durable, tamper-evident failure receipt for blocked TD8 label populations (#1025).

When every (or any) paired label candidate of a block is ineligible, the study
must abort before fitting. These tests drive the *real* label collector and
study runner (only the paired replay result is flagged ineligible after a real
replay) and verify the versioned, write-once receipt that is left behind.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_decision_value_dev_study import _compact_market
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import (
    PublishedDatasetArtifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import decision_value_dev_study as study
from trade_rl.evaluation import decision_value_reachable_labels as labels
from trade_rl.evaluation.decision_value_plan import DecisionValueStudyPlan
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.accounting import EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)

_RECEIPT = "incomplete-pairs.json"
_REAL_PAIRS: list[Any] = []
_HOLD = (ACTION_HOLD, ACTION_HOLD)
_LONG = (ACTION_LONG, ACTION_HOLD)
_SHORT = (ACTION_SHORT, ACTION_HOLD)


@dataclass(frozen=True)
class _Setup:
    dataset: MarketDataset
    artifact: PublishedDatasetArtifact
    protocol: study.DecisionValueDevStudyProtocol


def _setup(base: Path) -> _Setup:
    dataset = _compact_market(90).with_content_identity(
        {"synthetic_only": True, "generator": "issue1025-failure-receipt"}
    )
    artifact = publish_market_dataset_artifact(base / "dataset", dataset)
    plan = DecisionValueStudyPlan(
        data_scope="synthetic_only",
        account_mode="shared_portfolio",
        symbols=dataset.symbols,
        initial_capital=100_000.0,
        horizon_bars=4,
        terminal_reserve_bars=1,
        candidate_actions=(_HOLD, _LONG, _SHORT),
        fee_rate=0.0005,
        spread_rate=0.0002,
        max_policy_improvements=1,
        learner_roster=("linear", "conservative_linear", "shallow"),
        real_scope_authorization_digest=None,
    )
    protocol = study.DecisionValueDevStudyProtocol(
        study_plan=plan,
        dataset_id=dataset.dataset_id,
        expected_artifact_digest=artifact.artifact_digest,
        blocks=(
            study.DecisionValueBlockSpec(
                block_id="receipt_block",
                train_start_index=0,
                train_stop_index=24,
                train_step=4,
                fit_cutoff_index=30,
                eval_start_index=35,
                eval_stop_index=55,
            ),
        ),
        target_abs_weights=(0.25, 0.25),
        risk_config=PreTradeRiskConfig(
            max_gross=0.50,
            max_abs_weight=0.25,
            max_turnover=1.0,
            drawdown_start=0.15,
            drawdown_stop=0.20,
        ),
        base_execution_cost=ExecutionCostConfig(
            fee_rate=0.0005,
            spread_rate=0.0002,
            impact_rate=0.0001,
            max_participation_rate=0.05,
            order_latency_bars=0,
        ),
        ridge_feature_indices=(0, 1),
        ridge_horizon_hours=4,
        ridge_alpha=1.0,
        ridge_entry_threshold=0.001,
        ridge_exit_threshold=0.0005,
        prior_usage_disclosure="compact synthetic failure-receipt disclosure",
        label_prefix_bars=4,
        reference_min_fit_bars=6,
    )
    return _Setup(dataset=dataset, artifact=artifact, protocol=protocol)


def _flag_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    *,
    when: Callable[[int], bool],
    insolvent_candidate: bool = False,
) -> dict[str, Any]:
    """Run the real paired replay, then mark selected pairs not label-eligible.

    The replay itself is real and its endpoint values are recorded as produced.
    With ``insolvent_candidate`` the candidate book is *injected* as
    insolvent/terminated **after** that replay (the real candidate is not
    insolvent), so the collector derives the insolvency reason codes from the
    injected flags, not from an organically insolvent account.
    """
    real = labels.run_paired_portfolio_action
    counter: dict[str, Any] = {"calls": 0, "flagged": 0, "pairs": []}

    def wrapped(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        pair = real(*args, **kwargs)  # type: ignore[arg-type]
        index = counter["calls"]
        counter["calls"] += 1
        if when(index):
            counter["flagged"] += 1
            if insolvent_candidate:
                book = pair.candidate_result.book
                book.insolvent = True
                book.termination_reason = EconomicTerminationReason.INSOLVENCY
            counter["pairs"].append(pair)
            return replace(
                pair, eligible_for_label=False, delta_per_initial_capital=None
            )
        return pair

    monkeypatch.setattr(labels, "run_paired_portfolio_action", wrapped)
    return counter


def _valid_endpoint(equity: str) -> dict[str, object]:
    """A complete, reader-valid endpoint record for hand-built source rows."""
    return {
        "terminal_marked_equity": equity,
        "observed_equity_change": "0.0",
        "initial_capital": equity,
        "requested_stop_index": 7,
        "observed_stop_index": 7,
        "terminal_cash": equity,
        "terminal_exact_quantities": ["0", "0"],
        "total_cost": "0.0",
        "funding_pnl": "0.0",
        "insolvent": False,
        "termination_reason": None,
        "observed_intervals": 7,
        "expected_intervals": 7,
        "pending_order_remainders": [],
        "settled_label_eligible": False,
    }


def _receipt_kwargs(setup: _Setup, **overrides: Any) -> dict[str, Any]:
    provenance = build_candidate_run_provenance()
    values: dict[str, Any] = {
        "dataset_id": setup.dataset.dataset_id,
        "protocol_digest": setup.protocol.protocol_digest,
        "actual_artifact_digest": setup.artifact.artifact_digest,
        "implementation_digest": provenance["implementation_digest"],
        "runtime_environment_digest": provenance["runtime_environment_digest"],
        "phase": "training",
        "observed_view_digest": None,
    }
    values.update(overrides)
    return values


def _forbid_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("a blocked label population must never be fitted")

    monkeypatch.setattr(study, "_fit_all_learners", boom)


@pytest.fixture(scope="module")
def failed_all(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[_Setup, Path, study.IncompletePairedLabelsError, int]]:
    base = tmp_path_factory.mktemp("all-ineligible")
    setup = _setup(base)
    destination = base / "attempt"
    with pytest.MonkeyPatch.context() as patch:
        counter = _flag_ineligible(
            patch, when=lambda _i: True, insolvent_candidate=True
        )
        _forbid_fit(patch)
        with pytest.raises(study.IncompletePairedLabelsError) as caught:
            study.run_decision_value_dev_study(
                setup.dataset,
                setup.protocol,
                actual_artifact_digest=setup.artifact.artifact_digest,
                artifact_root=setup.artifact.root,
                output_dir=destination,
            )
    assert counter["calls"] == counter["flagged"] > 0
    _REAL_PAIRS[:] = counter["pairs"]
    yield setup, destination, caught.value, counter["flagged"]


def test_all_candidates_ineligible_persists_identity_bound_receipt(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
) -> None:
    setup, destination, failure, flagged = failed_all
    # Only the receipt exists: no protocol/result/summary of a "success".
    assert sorted(p.name for p in destination.iterdir()) == [_RECEIPT]
    assert failure.receipt_digest is not None
    receipt = study.load_incomplete_pair_attempt(
        destination, expected_receipt_digest=failure.receipt_digest
    )
    provenance = build_candidate_run_provenance()
    assert receipt["schema_version"] == "decision_value_incomplete_pair_attempt_v3"
    assert receipt["verdict"] == "INVALID / BLOCKED"
    assert receipt["dataset_id"] == setup.dataset.dataset_id
    assert receipt["protocol_digest"] == setup.protocol.protocol_digest
    assert receipt["actual_artifact_digest"] == setup.artifact.artifact_digest
    assert receipt["implementation_digest"] == provenance["implementation_digest"]
    assert (
        receipt["runtime_environment_digest"]
        == provenance["runtime_environment_digest"]
    )
    assert receipt["block_id"] == "receipt_block"
    assert receipt["ineligible_pairs"] == flagged == len(failure.evidence)
    pairs = receipt["pairs"]
    assert pairs == [record.to_mapping() for record in failure.evidence]
    assert receipt["pairs_digest"] == content_digest(pairs)
    for pair in pairs:
        assert pair["schema"] == "decision_value_ineligible_pair_v1"
        assert pair["admissible_for_fit"] is False
        for endpoint in (pair["baseline"], pair["candidate"]):
            for key in (
                "terminal_marked_equity",
                "terminal_cash",
                "terminal_exact_quantities",
                "total_cost",
                "funding_pnl",
                "insolvent",
                "termination_reason",
                "pending_order_remainders",
                "observed_intervals",
                "expected_intervals",
            ):
                assert key in endpoint, key
    assert receipt["fit_admitted"] is False
    assert receipt["retry_policy"] == "write_once_no_restart"
    assert receipt["phase"] == "training"
    assert receipt["observed_view_digest"] is None
    # Endpoint fidelity: each recorded endpoint equals the real replay result,
    # and the reason codes are the ones derived from the candidate whose insolvency
    # was injected after the real replay.
    assert len(_REAL_PAIRS) == len(pairs)
    for real_pair, recorded in zip(_REAL_PAIRS, pairs, strict=True):
        for arm, result in (
            ("baseline", real_pair.baseline_result),
            ("candidate", real_pair.candidate_result),
        ):
            book = result.book
            endpoint = recorded[arm]
            assert endpoint["terminal_marked_equity"] == str(book.portfolio_value)
            assert endpoint["terminal_cash"] == str(book.cash)
            assert endpoint["terminal_exact_quantities"] == [
                str(q) for q in book.exact_quantities
            ]
            assert endpoint["total_cost"] == str(book.total_cost)
            assert endpoint["funding_pnl"] == str(book.funding_pnl)
            assert endpoint["insolvent"] is bool(book.insolvent)
            assert endpoint["observed_intervals"] == len(
                result.ledger_evidence.to_mapping()["intervals"]
            )
        assert recorded["baseline"]["insolvent"] is False
        assert recorded["baseline"]["termination_reason"] is None
        assert recorded["candidate"]["insolvent"] is True
        assert recorded["candidate"]["termination_reason"] == "insolvency"
        assert recorded["candidate"]["settled_label_eligible"] is False
        assert "candidate_insolvent" in recorded["reason_codes"]
        assert "candidate_terminated" in recorded["reason_codes"]
        assert "baseline_insolvent" not in recorded["reason_codes"]


def test_second_attempt_on_same_root_is_rejected_and_receipt_is_unchanged(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup, destination, failure, _ = failed_all
    before = (destination / _RECEIPT).read_bytes()

    def no_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("a restart into a failed root must be rejected pre-fit")

    monkeypatch.setattr(study, "_train_block", no_training)
    with pytest.raises(FileExistsError):
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
            output_dir=destination,
        )
    # Even bypassing the pre-flight check the writer refuses to replace it.
    with pytest.raises(FileExistsError):
        study._persist_incomplete_pair_attempt(
            destination, failure, **_receipt_kwargs(setup)
        )
    assert (destination / _RECEIPT).read_bytes() == before
    assert sorted(p.name for p in destination.iterdir()) == [_RECEIPT]


def test_partial_block_still_aborts_without_fitting_survivors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some eligible + some ineligible pairs: same block abort as before."""
    setup = _setup(tmp_path)
    counter = _flag_ineligible(monkeypatch, when=lambda i: i in (1, 4))
    _forbid_fit(monkeypatch)
    destination = tmp_path / "partial"
    with pytest.raises(study.IncompletePairedLabelsError) as caught:
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
            output_dir=destination,
        )
    assert counter["flagged"] == 2 < counter["calls"]
    receipt = study.load_incomplete_pair_attempt(
        destination, expected_receipt_digest=caught.value.receipt_digest
    )
    # Only the ineligible records are retained; survivors are never relabelled
    # as fitted evidence and invalid endpoints never become labels.
    assert receipt["ineligible_pairs"] == 2
    assert len(receipt["pairs"]) == 2
    assert all(pair["admissible_for_fit"] is False for pair in receipt["pairs"])
    assert sorted(p.name for p in destination.iterdir()) == [_RECEIPT]
    # Valid replays forced ineligible fall back to the generic reason and still
    # record the real endpoint values. A partial block now writes this receipt
    # and locks the root (previously nothing was written).
    assert receipt["phase"] == "training"
    for real_pair, recorded in zip(counter["pairs"], receipt["pairs"], strict=True):
        assert recorded["reason_codes"] == ["invalid_endpoint"]
        assert recorded["candidate"]["terminal_marked_equity"] == str(
            real_pair.candidate_result.book.portfolio_value
        )


def test_no_output_dir_still_aborts_without_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup(tmp_path)
    _flag_ineligible(monkeypatch, when=lambda _i: True)
    _forbid_fit(monkeypatch)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(study.IncompletePairedLabelsError) as caught:
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
        )
    assert caught.value.receipt_digest is None
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def _copy_receipt(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _rewrite(
    root: Path, mutate: Callable[[dict[str, object]], None], *, redigest: bool
) -> None:
    path = root / _RECEIPT
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    if redigest:
        payload["pairs_digest"] = content_digest(payload["pairs"])
        body = {k: v for k, v in payload.items() if k != "receipt_digest"}
        payload["receipt_digest"] = content_digest(body)
    path.write_bytes(canonical_json_bytes(payload))


def test_strict_reload_accepts_the_untouched_receipt(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
) -> None:
    _setup_, destination, failure, _ = failed_all
    # Without an external pin the self-digest and structure are still enforced.
    assert study.load_incomplete_pair_attempt(destination)["verdict"] == (
        "INVALID / BLOCKED"
    )
    assert failure.receipt_digest is not None


def _mutate_endpoint(payload: dict[str, object]) -> None:
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    pairs[0]["candidate"]["terminal_marked_equity"] = "123456789.0"


def _mutate_dataset(payload: dict[str, object]) -> None:
    payload["dataset_id"] = "0" * 64


def _mutate_checkout(payload: dict[str, object]) -> None:
    payload["implementation_digest"] = "0" * 64


def _mutate_verdict(payload: dict[str, object]) -> None:
    payload["verdict"] = "SOFTWARE_ONLY"


def _mutate_admit(payload: dict[str, object]) -> None:
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    pairs[0]["admissible_for_fit"] = True


def _mutate_drop_pair(payload: dict[str, object]) -> None:
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    pairs.pop()


def _mutate_retry(payload: dict[str, object]) -> None:
    payload["retry_policy"] = "restart_allowed"


def _mutate_extra_key(payload: dict[str, object]) -> None:
    payload["unreviewed"] = True


_MUTATIONS = (
    _mutate_endpoint,
    _mutate_dataset,
    _mutate_checkout,
    _mutate_verdict,
    _mutate_admit,
    _mutate_drop_pair,
    _mutate_retry,
    _mutate_extra_key,
)


@pytest.mark.parametrize("mutate", _MUTATIONS, ids=lambda f: f.__name__)
def test_tampered_receipt_is_rejected_on_strict_reload(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    _setup_, destination, failure, _ = failed_all
    # (a) edited without re-sealing: the self digest catches it.
    broken = _copy_receipt(destination, tmp_path / "unsealed")
    _rewrite(broken, mutate, redigest=False)
    with pytest.raises(ValueError):
        study.load_incomplete_pair_attempt(broken)
    # (b) edited and re-sealed: structural/semantic checks or the externally
    # pinned digest must still reject it.
    resealed = _copy_receipt(destination, tmp_path / "resealed")
    _rewrite(resealed, mutate, redigest=True)
    with pytest.raises(ValueError):
        study.load_incomplete_pair_attempt(
            resealed, expected_receipt_digest=failure.receipt_digest
        )


def test_resealed_forgery_is_rejected_by_the_pinned_digest(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
    tmp_path: Path,
) -> None:
    _setup_, destination, failure, _ = failed_all
    forged = _copy_receipt(destination, tmp_path / "forged")
    _rewrite(forged, _mutate_endpoint, redigest=True)
    # Self-consistent and structurally valid, so only the pin can reject it.
    assert study.load_incomplete_pair_attempt(forged)["pairs"][0]["candidate"][
        "terminal_marked_equity"
    ] == ("123456789.0")
    with pytest.raises(ValueError, match="pinned"):
        study.load_incomplete_pair_attempt(
            forged, expected_receipt_digest=failure.receipt_digest
        )


def test_non_canonical_or_foreign_files_are_rejected(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
    tmp_path: Path,
) -> None:
    _setup_, destination, _failure, _ = failed_all
    pretty = _copy_receipt(destination, tmp_path / "pretty")
    payload = json.loads((pretty / _RECEIPT).read_text(encoding="utf-8"))
    (pretty / _RECEIPT).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        study.load_incomplete_pair_attempt(pretty)

    extra = _copy_receipt(destination, tmp_path / "extra")
    (extra / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="contents"):
        study.load_incomplete_pair_attempt(extra)

    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ValueError, match="contents"):
        study.load_incomplete_pair_attempt(missing)

    with pytest.raises(ValueError, match="pinned|sha256|digest"):
        study.load_incomplete_pair_attempt(destination, expected_receipt_digest="xyz")


def test_probe_only_ineligibility_persists_probe_scoped_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label failure in the causality-probe re-run must not abort receipt-less."""
    setup = _setup(tmp_path)
    state = {"trains": 0}
    real_train = study._train_block

    def counting_train(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        result = real_train(*args, **kwargs)
        state["trains"] += 1
        return result

    monkeypatch.setattr(study, "_train_block", counting_train)
    counter = _flag_ineligible(
        monkeypatch, when=lambda _i: state["trains"] >= 1, insolvent_candidate=True
    )
    destination = tmp_path / "probe"
    with pytest.raises(study.IncompletePairedLabelsError) as caught:
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
            output_dir=destination,
        )
    # The primary block fitted normally; only the probe re-run was blocked.
    assert state["trains"] == 1
    assert counter["flagged"] == counter["calls"] - counter["flagged"] > 0
    assert caught.value.receipt_digest is not None
    receipt = study.load_incomplete_pair_attempt(
        destination, expected_receipt_digest=caught.value.receipt_digest
    )
    assert receipt["phase"] == "causality_probe"
    assert receipt["block_id"] == "receipt_block"
    assert receipt["dataset_id"] == setup.dataset.dataset_id
    assert receipt["ineligible_pairs"] == counter["flagged"]
    view_digest = receipt["observed_view_digest"]
    assert isinstance(view_digest, str) and len(view_digest) == 64
    # The recorded view identity is the perturbed (not the original) source.
    perturbed = study._perturb_after(
        setup.dataset, setup.protocol.blocks[0].fit_cutoff_index
    )
    assert view_digest == study._dataset_view_digest(perturbed)
    assert view_digest != study._dataset_view_digest(setup.dataset)
    assert sorted(p.name for p in destination.iterdir()) == [_RECEIPT]

    # Same root is not retryable and nothing was published or replaced.
    before = (destination / _RECEIPT).read_bytes()
    monkeypatch.setattr(study, "_train_block", lambda *a, **k: pytest.fail("retry"))
    with pytest.raises(FileExistsError):
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
            output_dir=destination,
        )
    assert (destination / _RECEIPT).read_bytes() == before
    assert not (destination / "result.json").exists()


def test_receipt_writer_does_not_need_provenance_on_the_abort_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identities are fixed before the run; the abort path cannot lose the receipt."""
    setup = _setup(tmp_path)
    kwargs = _receipt_kwargs(setup)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("provenance failed on the abort path")

    monkeypatch.setattr(study, "build_candidate_run_provenance", boom)
    row = labels.IneligiblePairOutcome(
        intervention_index=1,
        prefix_start_index=0,
        terminal_index=7,
        candidate_action=(1, 0),
        reason_codes=("candidate_terminated",),
        baseline=_valid_endpoint("1000"),
        candidate=_valid_endpoint("900"),
    )
    failure = study.IncompletePairedLabelsError("b1", (row,))
    destination = tmp_path / "writer"
    digest = study._persist_incomplete_pair_attempt(destination, failure, **kwargs)
    assert (destination / _RECEIPT).is_file()
    assert len(digest) == 64
    # Whatever the writer persists must pass strict reload.
    receipt = study.load_incomplete_pair_attempt(
        destination, expected_receipt_digest=digest
    )
    assert receipt["pairs"][0]["candidate"]["terminal_marked_equity"] == "900"


def test_provenance_failure_aborts_before_any_work_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("provenance unavailable")

    monkeypatch.setattr(study, "build_candidate_run_provenance", boom)
    monkeypatch.setattr(study, "_train_block", lambda *a, **k: pytest.fail("trained"))
    destination = tmp_path / "never"
    with pytest.raises(RuntimeError, match="provenance unavailable"):
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
            output_dir=destination,
        )
    assert not destination.exists()


def _on_pair(payload: dict[str, object], fn: Callable[[dict[str, Any]], None]) -> None:
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    fn(pairs[0])


def _drop_equity(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p["baseline"].pop("terminal_marked_equity"))


def _nan_equity(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p["candidate"].update(terminal_marked_equity="nan"))


def _inf_cost(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p["candidate"].update(total_cost="inf"))


def _bad_flag(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p["candidate"].update(insolvent="yes"))


def _bad_quantities(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p["baseline"].update(terminal_exact_quantities="0"))


def _bad_pending(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p["baseline"].update(pending_order_remainders=[[1]]))


def _bad_index(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p.update(intervention_index=True))


def _bad_reasons(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p.update(reason_codes="candidate_terminated"))


def _no_reasons(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p.update(reason_codes=[]))


def _bad_action(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p.update(candidate_action=["a"]))


def _missing_endpoint(payload: dict[str, object]) -> None:
    _on_pair(payload, lambda p: p.pop("candidate"))


def _bad_phase(payload: dict[str, object]) -> None:
    payload["phase"] = "fit"


def _training_with_view(payload: dict[str, object]) -> None:
    payload["observed_view_digest"] = "a" * 64


def _probe_without_view(payload: dict[str, object]) -> None:
    payload["phase"] = "causality_probe"


_STRUCTURAL = (
    _drop_equity,
    _nan_equity,
    _inf_cost,
    _bad_flag,
    _bad_quantities,
    _bad_pending,
    _bad_index,
    _bad_reasons,
    _no_reasons,
    _bad_action,
    _missing_endpoint,
    _bad_phase,
    _training_with_view,
    _probe_without_view,
)


@pytest.mark.parametrize("mutate", _STRUCTURAL, ids=lambda f: f.__name__)
def test_resealed_structurally_invalid_content_is_rejected_without_a_pin(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    _setup_, destination, _failure, _ = failed_all
    resealed = _copy_receipt(destination, tmp_path / "resealed")
    _rewrite(resealed, mutate, redigest=True)
    with pytest.raises(ValueError):
        study.load_incomplete_pair_attempt(resealed)


# The receipt reader is a trust boundary: parsing attacker-supplied numeric
# strings must be bounded *before* Fraction expands their exponent.
@pytest.mark.parametrize("value", ("1e4097", "1e-4097"))
def test_incomplete_receipt_numeric_parser_rejects_unbounded_exponents(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="finite numeric string"):
        study._require_finite_decimal(value, field="terminal_cash")


@pytest.mark.parametrize(
    "value",
    ("0", "-1/3", "3.5", "+2.5", "1e-308", "1e308", "1e4096"),
)
def test_incomplete_receipt_numeric_parser_preserves_bounded_exact_forms(
    value: str,
) -> None:
    study._require_finite_decimal(value, field="terminal_cash")


@pytest.mark.parametrize("value", ("1e+-1", "1e--1", "1e-+1", "1e++1"))
def test_incomplete_receipt_numeric_parser_rejects_malformed_signs(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="finite numeric string"):
        study._require_finite_decimal(value, field="terminal_cash")


@pytest.mark.parametrize(
    "value",
    (
        pytest.param("1e99999999", id="huge-positive-exponent"),
        pytest.param("1e-99999999", id="huge-negative-exponent"),
        pytest.param("1" * 20_000, id="oversized-mantissa"),
    ),
)
def test_incomplete_receipt_numeric_parser_rejects_resource_exhaustion(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="finite numeric string"):
        study._require_finite_decimal(value, field="terminal_cash")


def test_unpinned_resealed_receipt_rejects_numeric_exponent_bomb(
    failed_all: tuple[_Setup, Path, study.IncompletePairedLabelsError, int],
    tmp_path: Path,
) -> None:
    _setup_, destination, _failure, _ = failed_all
    resealed = _copy_receipt(destination, tmp_path / "exponent-bomb")
    _rewrite(
        resealed,
        lambda payload: _on_pair(
            payload,
            lambda pair: pair["candidate"].update(terminal_cash="1e99999999"),
        ),
        redigest=True,
    )
    with pytest.raises(ValueError, match="finite numeric string"):
        study.load_incomplete_pair_attempt(resealed)
