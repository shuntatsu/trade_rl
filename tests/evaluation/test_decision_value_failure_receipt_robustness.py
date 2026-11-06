"""Robustness contract of the TD8 incomplete-pair failure receipt (#1025).

Issue #1025 requires a durable, write-once receipt *before* a blocked run
aborts. These tests pin the three properties that make that dependable:

* the writer applies the reader's validator, so every receipt it persists
  passes strict reload (non-finite or unrepresentable endpoint values are
  encoded as the defined ``nonfinite:<kind>`` form instead of failing);
* no value can make serialisation fail between the abort decision and the root
  claim, so the root is never left retryable;
* the strict reader bounds numeric-string parsing before ``Fraction`` sees it.

The ``IneligiblePairOutcome`` rows here are hand-built source records; the
end-to-end tests drive the real label collector and runner.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.evaluation.test_decision_value_dev_study import _compact_market
from tests.evaluation.test_decision_value_failure_receipt import (
    _copy_receipt,
    _flag_ineligible,
    _forbid_fit,
    _rewrite,
    _setup,
)
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation import decision_value_dev_study as study
from trade_rl.evaluation import decision_value_reachable_labels as labels

_RECEIPT = "incomplete-pairs.json"
_V2 = "decision_value_incomplete_pair_attempt_v2"
_V3 = "decision_value_incomplete_pair_attempt_v3"
_IDENTITY: dict[str, Any] = {
    "dataset_id": "1" * 64,
    "protocol_digest": "f" * 64,
    "actual_artifact_digest": "2" * 64,
    "implementation_digest": "3" * 64,
    "runtime_environment_digest": "4" * 64,
    "phase": "training",
    "observed_view_digest": None,
}


def _endpoint(**overrides: Any) -> dict[str, object]:
    value: dict[str, object] = {
        "terminal_marked_equity": "99000.5",
        "observed_equity_change": "-999.5",
        "initial_capital": "100000.0",
        "requested_stop_index": 8,
        "observed_stop_index": 6,
        "terminal_cash": "99000.5",
        "terminal_exact_quantities": ["0", "1/3"],
        "total_cost": "12.25",
        "funding_pnl": "-0.0",
        "insolvent": False,
        "termination_reason": None,
        "observed_intervals": 2,
        "expected_intervals": 4,
        "pending_order_remainders": [["a" * 64, 4.0]],
        "settled_label_eligible": False,
    }
    value.update(overrides)
    return value


def _row(
    *,
    baseline: dict[str, object] | None = None,
    candidate: dict[str, object] | None = None,
    action: tuple[int, ...] = (1, 0),
) -> labels.IneligiblePairOutcome:
    return labels.IneligiblePairOutcome(
        intervention_index=1,
        prefix_start_index=0,
        terminal_index=7,
        candidate_action=action,
        reason_codes=("candidate_terminated",),
        baseline=_endpoint() if baseline is None else baseline,
        candidate=_endpoint() if candidate is None else candidate,
    )


def _persist(root: Path, *rows: labels.IneligiblePairOutcome, **overrides: Any) -> str:
    failure = study.IncompletePairedLabelsError("b1", tuple(rows))
    return study._persist_incomplete_pair_attempt(
        root, failure, **{**_IDENTITY, **overrides}
    )


def _assert_locked_and_reloadable(root: Path, digest: str) -> dict[str, object]:
    assert sorted(p.name for p in root.iterdir()) == [_RECEIPT]
    receipt = study.load_incomplete_pair_attempt(root, expected_receipt_digest=digest)
    # The root stays consumed: a retry is rejected and leaves the receipt as is.
    before = (root / _RECEIPT).read_bytes()
    with pytest.raises(FileExistsError):
        _persist(root, _row())
    assert (root / _RECEIPT).read_bytes() == before
    return receipt


# --------------------------------------------------------------------- Low 1
_NONFINITE_TEXT = (
    ("nan", "nonfinite:nan"),
    ("NaN", "nonfinite:nan"),
    ("-nan", "nonfinite:nan"),
    ("inf", "nonfinite:inf"),
    ("+inf", "nonfinite:inf"),
    ("Infinity", "nonfinite:inf"),
    ("-inf", "nonfinite:-inf"),
    ("-Infinity", "nonfinite:-inf"),
    ("1e99999999", "nonfinite:oversized"),
    ("1" * 20_000, "nonfinite:oversized"),
    ("not-a-number", "nonfinite:unparseable"),
    ("", "nonfinite:unparseable"),
    (" 1.5", "nonfinite:unparseable"),
    ("1/0", "nonfinite:unparseable"),
)
_FINITE_TEXT = ("0", "-0.0", "1e308", "-1e-324", "1/3", "-7/2", ".5", "5.", "1E5")
_DECIMAL_FIELDS = (
    "terminal_marked_equity",
    "observed_equity_change",
    "initial_capital",
    "terminal_cash",
    "total_cost",
    "funding_pnl",
)


@pytest.mark.parametrize("field", _DECIMAL_FIELDS)
@pytest.mark.parametrize(("raw", "encoded"), _NONFINITE_TEXT)
def test_nonfinite_endpoint_strings_are_persisted_and_strictly_reloadable(
    tmp_path: Path, field: str, raw: str, encoded: str
) -> None:
    root = tmp_path / "attempt"
    digest = _persist(root, _row(candidate=_endpoint(**{field: raw})))
    receipt = _assert_locked_and_reloadable(root, digest)
    pair = receipt["pairs"][0]  # type: ignore[index]
    # Defined encoding, never a silent drop or a different value.
    assert pair["candidate"][field] == encoded
    assert pair["baseline"][field] == _endpoint()[field]
    assert receipt["schema_version"] == _V3
    assert receipt["fit_admitted"] is False


@pytest.mark.parametrize("raw", _FINITE_TEXT)
def test_finite_endpoint_strings_are_preserved_verbatim(
    tmp_path: Path, raw: str
) -> None:
    root = tmp_path / "attempt"
    digest = _persist(
        root,
        _row(
            candidate=_endpoint(terminal_cash=raw, terminal_exact_quantities=[raw, "0"])
        ),
    )
    receipt = _assert_locked_and_reloadable(root, digest)
    candidate = receipt["pairs"][0]["candidate"]  # type: ignore[index]
    assert candidate["terminal_cash"] == raw
    assert candidate["terminal_exact_quantities"] == [raw, "0"]


@pytest.mark.parametrize(
    ("raw", "encoded"),
    (
        pytest.param(float("nan"), "nonfinite:nan", id="nan"),
        pytest.param(float("inf"), "nonfinite:inf", id="inf"),
        pytest.param(float("-inf"), "nonfinite:-inf", id="-inf"),
        pytest.param(10**5000, "nonfinite:oversized", id="huge-int"),
        pytest.param("nan", "nonfinite:nan", id="nan-text"),
    ),
)
def test_nonfinite_pending_remainders_are_encoded_not_fatal(
    tmp_path: Path, raw: object, encoded: str
) -> None:
    root = tmp_path / "attempt"
    digest = _persist(
        root,
        _row(
            candidate=_endpoint(
                pending_order_remainders=[["a" * 64, raw], ["b" * 64, 2.5]]
            )
        ),
    )
    receipt = _assert_locked_and_reloadable(root, digest)
    pending = receipt["pairs"][0]["candidate"]["pending_order_remainders"]  # type: ignore[index]
    assert pending == [["a" * 64, encoded], ["b" * 64, 2.5]]


def test_nonfinite_quantity_and_negative_zero_survive_round_trip(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    digest = _persist(
        root,
        _row(
            baseline=_endpoint(
                terminal_exact_quantities=["-0.0", "inf", "-inf", "nan", "3/0"]
            ),
            candidate=_endpoint(pending_order_remainders=[["a" * 64, -0.0]]),
        ),
    )
    receipt = _assert_locked_and_reloadable(root, digest)
    pair = receipt["pairs"][0]  # type: ignore[index]
    assert pair["baseline"]["terminal_exact_quantities"] == [
        "-0.0",
        "nonfinite:inf",
        "nonfinite:-inf",
        "nonfinite:nan",
        "nonfinite:unparseable",
    ]
    assert str(pair["candidate"]["pending_order_remainders"][0][1]) == "-0.0"


def test_writer_and_reader_share_one_pair_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real = study._validate_incomplete_pair_record

    def spy(pair: object, **kwargs: object) -> None:
        calls.append("call")
        real(pair, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(study, "_validate_incomplete_pair_record", spy)
    root = tmp_path / "attempt"
    digest = _persist(root, _row())
    after_write = len(calls)
    assert after_write >= 1, "writer must run the reader's validator"
    study.load_incomplete_pair_attempt(root, expected_receipt_digest=digest)
    assert len(calls) > after_write


def test_unrepresentable_record_is_quarantined_not_lost_and_not_retryable(
    tmp_path: Path,
) -> None:
    """A structurally malformed record still locks the root with a valid receipt."""
    root = tmp_path / "attempt"
    malformed = _row(
        baseline={"terminal_marked_equity": "893.0"},
        candidate={"terminal_marked_equity": "nan", "surprise": object()},
        action=(1, 10**30),
    )
    digest = _persist(root, malformed)
    receipt = _assert_locked_and_reloadable(root, digest)
    pair = receipt["pairs"][0]  # type: ignore[index]
    assert pair["admissible_for_fit"] is False
    assert pair["reason_codes"][0] == "receipt:unvalidated_source_record"
    source = pair["unvalidated_source_record"]
    # The original evidence is preserved as bounded text with its digest.
    assert "893.0" in source["text"]
    assert source["encoding"] in {"canonical_json", "repr"}
    assert len(source["sha256"]) == 64
    for arm in ("baseline", "candidate"):
        assert pair[arm]["settled_label_eligible"] is False
        assert pair[arm]["terminal_marked_equity"] == "nonfinite:unparseable"


def test_whole_receipt_self_check_falls_back_to_quarantine_before_claiming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if pair building misbehaves, a valid receipt claims the root."""
    monkeypatch.setattr(
        study,
        "_receipt_pairs",
        lambda evidence: [{"schema": "bogus"} for _ in evidence],
    )
    root = tmp_path / "attempt"
    digest = _persist(root, _row(), _row())
    receipt = _assert_locked_and_reloadable(root, digest)
    assert receipt["ineligible_pairs"] == 2
    assert all(
        pair["reason_codes"][0] == "receipt:unvalidated_source_record"
        for pair in receipt["pairs"]  # type: ignore[attr-defined]
    )


_TEXTS = st.one_of(
    st.sampled_from(
        [raw for raw, _ in _NONFINITE_TEXT]
        + list(_FINITE_TEXT)
        + ["1e4096", "1e4097", "1e-4097", "0e9999", "١٢٣", "1_0", "0x10", "\ud800"]
    ),
    st.floats(allow_nan=True, allow_infinity=True).map(str),
    st.floats(allow_nan=True, allow_infinity=True).map(repr),
    st.integers(min_value=-(10**40), max_value=10**40).map(str),
    st.text(max_size=40),
)
_ANY_VALUE = st.one_of(
    _TEXTS,
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(),
    st.integers(min_value=10**5000, max_value=10**5001),
    st.booleans(),
    st.none(),
    st.lists(st.integers(), max_size=3),
)


@st.composite
def _adversarial_endpoint(draw: st.DrawFn) -> dict[str, object]:
    value = _endpoint()
    for name in _DECIMAL_FIELDS:
        value[name] = draw(_ANY_VALUE)
    value["terminal_exact_quantities"] = draw(st.lists(_ANY_VALUE, max_size=4))
    value["pending_order_remainders"] = draw(
        st.lists(st.tuples(st.text(max_size=8), _ANY_VALUE), max_size=3)
    )
    for name in ("requested_stop_index", "observed_intervals"):
        value[name] = draw(st.one_of(st.integers(), st.just(10**5000)))
    value["termination_reason"] = draw(st.one_of(st.none(), st.text(max_size=8)))
    return value


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(baseline=_adversarial_endpoint(), candidate=_adversarial_endpoint())
def test_every_persisted_receipt_passes_strict_reload(
    baseline: dict[str, object], candidate: dict[str, object]
) -> None:
    """Invariant: whatever the writer persists, the strict reader accepts."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "attempt"
        digest = _persist(root, _row(baseline=baseline, candidate=candidate))
        assert sorted(p.name for p in root.iterdir()) == [_RECEIPT]
        receipt = study.load_incomplete_pair_attempt(
            root, expected_receipt_digest=digest
        )
        assert receipt["ineligible_pairs"] == 1
        assert receipt["fit_admitted"] is False
        # Re-reading the persisted bytes is stable (canonical, digest bound).
        raw_bytes = (root / _RECEIPT).read_bytes()
        assert raw_bytes == canonical_json_bytes(json.loads(raw_bytes))
        with pytest.raises(FileExistsError):
            _persist(root, _row())


# --------------------------------------------------------------------- Low 2
def test_nonfinite_pending_remainder_does_not_replace_the_failure_or_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced NaN remainder through the real runner: receipt written, root locked."""
    setup = _setup(tmp_path)
    _flag_ineligible(monkeypatch, when=lambda _i: True, insolvent_candidate=True)
    _forbid_fit(monkeypatch)
    real_endpoint = labels._ineligible_arm_endpoint

    def nan_remainder(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        record, problems = real_endpoint(*args, **kwargs)
        record["pending_order_remainders"] = [["a" * 64, float("nan")]]
        record["terminal_cash"] = "inf"
        return record, problems

    monkeypatch.setattr(labels, "_ineligible_arm_endpoint", nan_remainder)
    destination = tmp_path / "attempt"
    # The intended failure is raised, not a serialisation ValueError.
    with pytest.raises(study.IncompletePairedLabelsError) as caught:
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
            output_dir=destination,
        )
    digest = caught.value.receipt_digest
    assert digest is not None
    receipt = _assert_locked_and_reloadable(destination, digest)
    pair = receipt["pairs"][0]  # type: ignore[index]
    assert pair["candidate"]["pending_order_remainders"] == [
        ["a" * 64, "nonfinite:nan"]
    ]
    assert pair["candidate"]["terminal_cash"] == "nonfinite:inf"
    # A second run into the same root is refused before any fit.
    monkeypatch.setattr(study, "_train_block", lambda *a, **k: pytest.fail("retry"))
    with pytest.raises(FileExistsError):
        study.run_decision_value_dev_study(
            setup.dataset,
            setup.protocol,
            actual_artifact_digest=setup.artifact.artifact_digest,
            artifact_root=setup.artifact.root,
            output_dir=destination,
        )


def test_serialisation_cannot_fail_between_abort_decision_and_root_claim(
    tmp_path: Path,
) -> None:
    """Values that used to raise inside canonical encoding now claim the root."""
    root = tmp_path / "attempt"
    digest = _persist(
        root,
        _row(
            candidate=_endpoint(
                pending_order_remainders=[["a" * 64, float("inf")]],
                termination_reason="\ud800",
            )
        ),
    )
    _assert_locked_and_reloadable(root, digest)


# --------------------------------------------------------------------- Low 3
def _numeric_id(value: str) -> str:
    """Short, environment-safe test id (long values must not enter the env)."""
    return f"{value[:12]!r}-len{len(value)}"


_REJECTED_NUMERIC = (
    "1e99999999",
    "1e-99999999",
    "1e4097",
    "1e-4097",
    "1" * 4001,
    "1" * 20_000,
    "1" * 2_000 + "." + "1" * 2_001,
    "9" * 4001 + "/3",
    "1/" + "9" * 4001,
    "1/0",
    "1e",
    "e5",
    ".",
    "+",
    "1e+-1",
    "1e--1",
    "1_000",
    "0x10",
    "١٢",
    "nan",
    "inf",
    "-inf",
    "Infinity",
    "",
    " 1",
    "1 ",
    "1\n",
)
_ACCEPTED_NUMERIC = (
    "0",
    "-1/3",
    "3.5",
    "+2.5",
    ".5",
    "5.",
    "1e-308",
    "1e308",
    "1e4096",
    "1e-4096",
    "-0.0",
    "1E5",
    "1" * 4000,
    "1" * 4000 + "/" + "7" * 4000,
)


@pytest.mark.parametrize("value", _REJECTED_NUMERIC, ids=_numeric_id)
def test_numeric_parser_rejects_hostile_strings_quickly(value: str) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="finite numeric string"):
        study._require_finite_decimal(value, field="terminal_cash")
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("value", _ACCEPTED_NUMERIC, ids=_numeric_id)
def test_numeric_parser_keeps_bounded_exact_forms(value: str) -> None:
    started = time.perf_counter()
    study._require_finite_decimal(value, field="terminal_cash")
    assert time.perf_counter() - started < 1.0


def test_very_long_numeric_string_is_rejected_without_scanning_it_all() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="finite numeric string"):
        study._require_finite_decimal("1" * 5_000_000, field="terminal_cash")
    assert time.perf_counter() - started < 1.0


def test_sentinels_are_not_numeric_for_the_plain_parser() -> None:
    with pytest.raises(ValueError, match="finite numeric string"):
        study._require_finite_decimal("nonfinite:nan", field="terminal_cash")
    study._require_finite_decimal(
        "nonfinite:nan", field="terminal_cash", allow_sentinel=True
    )
    with pytest.raises(ValueError, match="finite numeric string"):
        study._require_finite_decimal(
            "nonfinite:bogus", field="terminal_cash", allow_sentinel=True
        )


@pytest.fixture(scope="module")
def sealed_receipt(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("sealed") / "attempt"
    return root, _persist(root, _row())


@pytest.mark.parametrize(
    "value",
    ("1e99999999", "1e-99999999", "1" * 1_000_000, "9" * 5000 + "/3"),
    ids=_numeric_id,
)
@pytest.mark.parametrize("pinned", (False, True))
def test_resealed_numeric_bomb_is_rejected_fast_with_or_without_pin(
    sealed_receipt: tuple[Path, str], tmp_path: Path, value: str, pinned: bool
) -> None:
    root, digest = sealed_receipt
    resealed = _copy_receipt(root, tmp_path / "bomb")
    _rewrite(
        resealed,
        lambda payload: payload["pairs"][0]["candidate"].update(terminal_cash=value),  # type: ignore[index]
        redigest=True,
    )
    started = time.perf_counter()
    with pytest.raises(ValueError):
        study.load_incomplete_pair_attempt(
            resealed, expected_receipt_digest=digest if pinned else None
        )
    assert time.perf_counter() - started < 1.0


# ------------------------------------------------------- Low 5 / versioning
def _digest_variants(base: Any) -> dict[str, Any]:
    timestamps = base.timestamps + np.timedelta64(1, "ns")
    tradable = base.tradable.copy()
    tradable[3, 0] = not tradable[3, 0]
    available = base.feature_available.copy()
    available[3, 0, 0] = not available[3, 0, 0]
    staleness = base.feature_staleness.copy()
    staleness[3, 0, 0] = 1.0 if not available[3, 0, 0] else 0.0
    global_features = base.global_features.copy()
    global_features[3, 0] += 1.0
    return {
        "timestamps": replace(
            base,
            timestamps=timestamps,
            available_at=base.available_at + np.timedelta64(1, "ns"),
        ),
        "tradable": replace(base, tradable=tradable),
        "feature_available": replace(
            base, feature_available=available, feature_staleness=staleness
        ),
        "global_features": replace(base, global_features=global_features),
    }


def test_observed_view_digest_covers_timestamps_masks_and_global_features() -> None:
    base = _compact_market(30)
    reference = study._dataset_view_digest(base)
    assert study._dataset_view_digest(base) == reference
    for name, variant in _digest_variants(base).items():
        assert study._dataset_view_digest(variant) != reference, name


def test_observed_view_digest_still_tracks_the_original_seven_arrays() -> None:
    base = _compact_market(30)
    reference = study._dataset_view_digest(base)
    for name in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "features",
    ):
        array = getattr(base, name).copy()
        if name in ("open", "high", "low", "close"):
            array[3, 0] = array[3, 0] * (1.0 + 1e-6)
        else:
            array[3, 0] = array[3, 0] + 1.0 if name != "features" else 5.0
        changed = replace(base, **{name: array})
        assert study._dataset_view_digest(changed) != reference, name


def _as_v2(
    root: Path, mutate: Callable[[dict[str, object]], None] | None = None
) -> None:
    def to_v2(payload: dict[str, object]) -> None:
        payload["schema_version"] = _V2
        if mutate is not None:
            mutate(payload)

    _rewrite(root, to_v2, redigest=True)


def test_existing_v2_receipts_remain_readable(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    _persist(root, _row())
    _as_v2(root)
    receipt = study.load_incomplete_pair_attempt(root)
    assert receipt["schema_version"] == _V2
    pinned = study.load_incomplete_pair_attempt(
        root, expected_receipt_digest=str(receipt["receipt_digest"])
    )
    assert pinned["receipt_digest"] == receipt["receipt_digest"]


def test_v2_readable_with_probe_phase_digest(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    _persist(root, _row(), phase="causality_probe", observed_view_digest="5" * 64)
    _as_v2(root)
    assert study.load_incomplete_pair_attempt(root)["phase"] == "causality_probe"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p["pairs"][0]["candidate"].update(terminal_cash="nonfinite:nan"),
        lambda p: p["pairs"][0]["candidate"].update(
            pending_order_remainders=[["a" * 64, "nonfinite:inf"]]
        ),
        lambda p: p["pairs"][0].update(
            unvalidated_source_record={
                "encoding": "repr",
                "sha256": "0" * 64,
                "text": "x",
                "truncated": False,
                "validation_error": "x",
            }
        ),
    ),
    ids=("sentinel-decimal", "sentinel-pending", "quarantine-record"),
)
def test_v2_receipts_do_not_accept_the_v3_only_encodings(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    root = tmp_path / "attempt"
    _persist(root, _row())
    _as_v2(root, mutate)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        study.load_incomplete_pair_attempt(root)


@pytest.mark.parametrize("schema", ("decision_value_incomplete_pair_attempt_v1", "x"))
def test_unknown_receipt_schema_is_rejected(tmp_path: Path, schema: str) -> None:
    root = tmp_path / "attempt"
    _persist(root, _row())
    _rewrite(root, lambda p: p.update(schema_version=schema), redigest=True)
    with pytest.raises(ValueError, match="contract"):
        study.load_incomplete_pair_attempt(root)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p["pairs"][0]["candidate"].update(terminal_cash="nonfinite:bogus"),
        lambda p: p["pairs"][0]["candidate"].update(terminal_cash=None),
        lambda p: p["pairs"][0]["candidate"].update(terminal_cash="nonfinite:"),
        lambda p: p["pairs"][0]["candidate"].update(requested_stop_index=2**70),
        lambda p: p["pairs"][0].update(candidate_action=[2**70]),
        lambda p: p["pairs"][0].update(unvalidated_source_record={"text": "x"}),
        lambda p: p["pairs"][0].update(reference_vintage={"x": 2**70}),
    ),
    ids=(
        "bad-kind",
        "null-value",
        "empty-kind",
        "huge-int",
        "huge-action",
        "bad-quarantine-shape",
        "bad-vintage",
    ),
)
def test_v3_reader_still_rejects_malformed_values(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    root = tmp_path / "attempt"
    _persist(root, _row())
    _rewrite(root, mutate, redigest=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        study.load_incomplete_pair_attempt(root)


def test_receipt_digest_binds_the_encoded_form(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    digest = _persist(root, _row(candidate=_endpoint(terminal_cash="nan")))
    loaded = json.loads((root / _RECEIPT).read_bytes())
    body = {k: v for k, v in loaded.items() if k != "receipt_digest"}
    assert content_digest(body) == digest
    assert loaded["pairs_digest"] == content_digest(loaded["pairs"])
