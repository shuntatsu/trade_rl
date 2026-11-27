from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from trade_rl.evaluation.comparison.multiple_testing import SharpeMoments
from trade_rl.evaluation.comparison.trial_ledger import (
    TRIAL_RECORD_SCHEMA,
    TrialLedger,
    TrialLedgerError,
    TrialRecord,
    TrialStatus,
    decode_trial_ledger_jsonl,
    encode_trial_ledger_jsonl,
)

DIGEST_A = "a" * 64
DIGEST_B = "0123456789abcdef" * 4


def _completed(
    trial_id: str = "t1",
    *,
    family: str = "fam",
    sharpe: float = 0.1,
    digest: str = DIGEST_A,
    label: str | None = "dev window",
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        family=family,
        method="rule-x",
        config_digest=digest,
        status=TrialStatus.COMPLETED,
        performance=SharpeMoments(sharpe=sharpe, n_obs=250, skew=-0.5, kurtosis=4.0),
        evaluation_data_label=label,
    )


def _failed(trial_id: str = "f1", *, family: str = "fam") -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        family=family,
        method="rule-x",
        config_digest=DIGEST_B,
        status=TrialStatus.FAILED,
        reason="diverged",
    )


def _abandoned(trial_id: str = "a1", *, family: str = "fam") -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        family=family,
        method="rule-y",
        config_digest=DIGEST_B,
        status="ABANDONED",  # type: ignore[arg-type]
        reason="out of budget",
    )


def _with_returns() -> TrialRecord:
    return TrialRecord(
        trial_id="r1",
        family="fam",
        method="rule-z",
        config_digest=DIGEST_A,
        status=TrialStatus.COMPLETED,
        returns=(0.01, -0.02, 0.015, 0.0, 0.03),
    )


def _ledger() -> TrialLedger:
    return TrialLedger(
        [_completed("t1"), _failed(), _abandoned(), _with_returns(), _completed("t2")]
    )


# --- record contract -------------------------------------------------------


def test_record_is_immutable_and_normalises_values() -> None:
    record = _completed()
    with pytest.raises(FrozenInstanceError):
        record.family = "other"  # type: ignore[misc]
    assert record.status is TrialStatus.COMPLETED
    with_returns = _with_returns()
    assert isinstance(with_returns.returns, tuple)
    assert with_returns.moments().n_obs == 5
    assert _abandoned().status is TrialStatus.ABANDONED


def test_completed_record_moments_return_supplied_statistics() -> None:
    assert _completed(sharpe=0.25).moments().sharpe == 0.25


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"trial_id": ""}, "trial_id"),
        ({"trial_id": " padded"}, "trial_id"),
        ({"trial_id": "line\nbreak"}, "trial_id"),
        ({"trial_id": 5}, "trial_id"),
        ({"trial_id": "x" * 257}, "trial_id"),
        ({"family": ""}, "family"),
        ({"method": None}, "method"),
        ({"config_digest": "A" * 64}, "config_digest"),
        ({"config_digest": "a" * 63}, "config_digest"),
        ({"config_digest": "g" * 64}, "config_digest"),
        ({"config_digest": None}, "config_digest"),
        ({"status": "DONE"}, "status"),
        ({"reason": ""}, "reason"),
        ({"reason": "bad\tchar"}, "reason"),
        ({"evaluation_data_label": " "}, "evaluation_data_label"),
        ({"performance": None}, "exactly one"),
        ({"returns": (0.01, 0.02, 0.04)}, "exactly one"),
        ({"performance": object()}, "SharpeMoments"),
    ],
)
def test_record_refuses_invalid_fields(
    changes: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "trial_id": "t1",
        "family": "fam",
        "method": "m",
        "config_digest": DIGEST_A,
        "status": TrialStatus.COMPLETED,
        "performance": SharpeMoments(0.1, 100, 0.0, 3.0),
    }
    base.update(changes)
    with pytest.raises(TrialLedgerError, match=message):
        TrialRecord(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", [TrialStatus.FAILED, TrialStatus.ABANDONED])
def test_non_completed_trials_require_reason_and_no_results(
    status: TrialStatus,
) -> None:
    with pytest.raises(TrialLedgerError, match="require a reason"):
        TrialRecord("t", "f", "m", DIGEST_A, status)
    with pytest.raises(TrialLedgerError, match="must not carry"):
        TrialRecord(
            "t",
            "f",
            "m",
            DIGEST_A,
            status,
            reason="why",
            performance=SharpeMoments(0.1, 10, 0.0, 3.0),
        )
    with pytest.raises(TrialLedgerError, match="must not carry"):
        TrialRecord("t", "f", "m", DIGEST_A, status, reason="why", returns=(0.1, 0.2))


@pytest.mark.parametrize(
    "returns",
    [
        (0.01,),
        (0.0, 0.0, 0.0),
        (0.01, math.nan, 0.02),
        (0.01, math.inf, 0.02),
        (0.01, True, 0.03),
        (0.01, "x", 0.03),
        "abc",
    ],
)
def test_completed_record_refuses_unusable_returns(returns: object) -> None:
    with pytest.raises(TrialLedgerError):
        TrialRecord(
            "t",
            "f",
            "m",
            DIGEST_A,
            TrialStatus.COMPLETED,
            returns=returns,  # type: ignore[arg-type]
        )


def test_moments_refused_for_non_completed_trial() -> None:
    with pytest.raises(TrialLedgerError, match="only COMPLETED"):
        _failed().moments()


def test_completed_trial_may_carry_an_optional_reason() -> None:
    record = TrialRecord(
        "t",
        "f",
        "m",
        DIGEST_A,
        TrialStatus.COMPLETED,
        reason="note",
        performance=SharpeMoments(0.1, 10, 0.0, 3.0),
    )
    assert record.reason == "note"


# --- ledger ----------------------------------------------------------------


def test_ledger_is_append_only_and_idempotent() -> None:
    ledger = TrialLedger()
    assert ledger.append(_completed("t1")) is True
    assert ledger.append(_completed("t1")) is False
    assert len(ledger) == 1
    with pytest.raises(TrialLedgerError, match="different content"):
        ledger.append(_completed("t1", sharpe=0.2))
    with pytest.raises(TrialLedgerError, match="different content"):
        ledger.append(_failed("t1"))
    assert ledger.get("t1") == _completed("t1")
    assert "t1" in ledger and "missing" not in ledger
    assert len(ledger) == 1


def test_ledger_rejects_non_record_and_exposes_only_copies() -> None:
    ledger = _ledger()
    with pytest.raises(TrialLedgerError, match="only TrialRecord"):
        ledger.append({"trial_id": "x"})  # type: ignore[arg-type]
    assert isinstance(ledger.records, tuple)
    counts = ledger.family_counts()
    counts["fam"] = 999
    assert ledger.family_counts() == {"fam": 5}
    assert [item.trial_id for item in ledger] == ["t1", "f1", "a1", "r1", "t2"]
    assert not hasattr(ledger, "remove") and not hasattr(ledger, "pop")


def test_family_counts_include_failed_and_abandoned() -> None:
    ledger = _ledger()
    ledger.append(_failed("g1", family="other"))
    assert ledger.family_counts() == {"fam": 5, "other": 1}
    assert [item.trial_id for item in ledger.family_records("other")] == ["g1"]
    assert ledger.family_records("none") == ()


# --- codec -----------------------------------------------------------------


def test_codec_round_trip_and_determinism() -> None:
    ledger = _ledger()
    data = encode_trial_ledger_jsonl(ledger)
    assert data.endswith(b"\n") and b"\r" not in data
    assert data == encode_trial_ledger_jsonl(_ledger())
    restored = decode_trial_ledger_jsonl(data, expected_digest=ledger.digest())
    assert restored.records == ledger.records
    assert encode_trial_ledger_jsonl(restored) == data
    assert ledger.digest() == sha256(data).hexdigest()


def test_codec_line_is_sorted_compact_and_schema_tagged() -> None:
    line = encode_trial_ledger_jsonl(TrialLedger([_failed()])).decode("utf-8")
    payload = json.loads(line)
    assert payload["schema"] == TRIAL_RECORD_SCHEMA
    assert payload["status"] == "FAILED"
    assert payload["performance"] is None and payload["returns"] is None
    assert line == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def test_empty_ledger_round_trips_to_empty_bytes() -> None:
    empty = TrialLedger()
    assert encode_trial_ledger_jsonl(empty) == b""
    assert decode_trial_ledger_jsonl(b"").records == ()
    assert empty.digest() == sha256(b"").hexdigest()


def test_digest_depends_on_content_and_order() -> None:
    first = TrialLedger([_completed("t1"), _completed("t2", sharpe=0.2)])
    second = TrialLedger([_completed("t2", sharpe=0.2), _completed("t1")])
    assert first.digest() != second.digest()
    assert first.digest() != TrialLedger([_completed("t1")]).digest()


def _line(**overrides: object) -> str:
    payload: dict[str, object] = json.loads(
        encode_trial_ledger_jsonl(TrialLedger([_completed()])).decode("utf-8")
    )
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def test_decode_rejects_duplicate_keys_at_every_level() -> None:
    top = _line().rstrip("\n")
    duplicated_top = top[:-1] + ',"family":"fam"}\n'
    with pytest.raises(TrialLedgerError, match="duplicate JSON member"):
        decode_trial_ledger_jsonl(duplicated_top.encode())
    nested = top.replace('"n_obs":250', '"n_obs":250,"n_obs":251')
    assert nested != top
    with pytest.raises(TrialLedgerError, match="duplicate JSON member"):
        decode_trial_ledger_jsonl((nested + "\n").encode())
    escaped = top[:-1] + ',"fam\\u0069ly":"x"}\n'
    with pytest.raises(TrialLedgerError, match="duplicate JSON member"):
        decode_trial_ledger_jsonl(escaped.encode())


def test_decode_rejects_unknown_and_missing_fields() -> None:
    with pytest.raises(TrialLedgerError, match="unknown=\\['extra'\\]"):
        decode_trial_ledger_jsonl(_line(extra=1).encode())
    payload = json.loads(_line())
    del payload["reason"]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(TrialLedgerError, match="missing=\\['reason'\\]"):
        decode_trial_ledger_jsonl(text.encode())
    nested = json.loads(_line())
    nested["performance"]["extra"] = 1.0
    text = json.dumps(nested, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(TrialLedgerError, match="performance fields"):
        decode_trial_ledger_jsonl(text.encode())


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_decode_rejects_non_finite_numbers(constant: str) -> None:
    text = _line().replace('"sharpe":0.1', f'"sharpe":{constant}')
    assert constant in text
    with pytest.raises(TrialLedgerError, match="finite|invalid JSON"):
        decode_trial_ledger_jsonl(text.encode())


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_digest": "A" * 64},
        {"config_digest": "a" * 10},
        {"config_digest": 12},
        {"schema": "trial_record_v0"},
        {"status": "DONE"},
        {"status": 3},
        {"status": "FAILED"},
        {"trial_id": ""},
        {"returns": [0.1, 0.2]},
        {"returns": "oops"},
        {"performance": 5},
        {"reason": 3},
    ],
)
def test_decode_rejects_malformed_values(overrides: dict[str, object]) -> None:
    with pytest.raises(TrialLedgerError):
        decode_trial_ledger_jsonl(_line(**overrides).encode())


def test_decode_rejects_non_canonical_and_structural_variants() -> None:
    canonical = _line()
    spaced = canonical.replace(",", ", ", 1)
    with pytest.raises(TrialLedgerError, match="not canonical"):
        decode_trial_ledger_jsonl(spaced.encode())
    reordered = json.dumps(json.loads(canonical), separators=(",", ":")) + "\n"
    if reordered != canonical:
        with pytest.raises(TrialLedgerError, match="not canonical"):
            decode_trial_ledger_jsonl(reordered.encode())
    integer_sharpe = canonical.replace('"sharpe":0.1', '"sharpe":1')
    with pytest.raises(TrialLedgerError, match="finite JSON float"):
        decode_trial_ledger_jsonl(integer_sharpe.encode())
    float_n_obs = canonical.replace('"n_obs":250', '"n_obs":250.0')
    with pytest.raises(TrialLedgerError, match="n_obs"):
        decode_trial_ledger_jsonl(float_n_obs.encode())
    with pytest.raises(TrialLedgerError, match="newline"):
        decode_trial_ledger_jsonl(canonical.rstrip("\n").encode())
    with pytest.raises(TrialLedgerError, match="blank line"):
        decode_trial_ledger_jsonl((canonical + "\n").encode())
    with pytest.raises(TrialLedgerError, match="not canonical|invalid JSON"):
        decode_trial_ledger_jsonl(canonical.replace("\n", "\r\n").encode())
    with pytest.raises(TrialLedgerError, match="UTF-8"):
        decode_trial_ledger_jsonl(b"\xff\xfe\n")
    with pytest.raises(TrialLedgerError, match="JSON object"):
        decode_trial_ledger_jsonl(b"[1]\n")
    with pytest.raises(TrialLedgerError, match="invalid JSON"):
        decode_trial_ledger_jsonl(b"{not json}\n")
    with pytest.raises(TrialLedgerError, match="bytes"):
        decode_trial_ledger_jsonl("text")  # type: ignore[arg-type]


def test_decode_rejects_duplicate_trial_id_even_when_identical() -> None:
    line = _line()
    with pytest.raises(TrialLedgerError, match="duplicate trial_id"):
        decode_trial_ledger_jsonl((line + line).encode())
    conflicting = _line(family="other")
    with pytest.raises(TrialLedgerError, match="different content"):
        decode_trial_ledger_jsonl((line + conflicting).encode())


def test_decode_detects_tampering_through_the_expected_digest() -> None:
    ledger = _ledger()
    data = encode_trial_ledger_jsonl(ledger)
    expected = ledger.digest()
    tampered = data.replace(b'"sharpe":0.1,', b'"sharpe":0.9,', 1)
    assert tampered != data
    # A self-consistent edit still decodes alone but fails the digest binding.
    decode_trial_ledger_jsonl(tampered)
    with pytest.raises(TrialLedgerError, match="digest does not match"):
        decode_trial_ledger_jsonl(tampered, expected_digest=expected)
    dropped = b"".join(data.splitlines(keepends=True)[:-1])
    with pytest.raises(TrialLedgerError, match="digest does not match"):
        decode_trial_ledger_jsonl(dropped, expected_digest=expected)
    with pytest.raises(TrialLedgerError, match="config_digest"):
        decode_trial_ledger_jsonl(data, expected_digest="XYZ")


def test_decode_rejects_oversized_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import trade_rl.evaluation.comparison.trial_ledger as module

    monkeypatch.setattr(module, "MAX_LEDGER_BYTES", 10)
    with pytest.raises(TrialLedgerError, match="size limit"):
        decode_trial_ledger_jsonl(b"x" * 11)
