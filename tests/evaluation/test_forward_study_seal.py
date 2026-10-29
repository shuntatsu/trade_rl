"""FS1 seal: every refusal path, with fakes only (no dataset, no network, no P&L)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from trade_rl.evaluation.forward_study import seal
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    MANIFEST_SCHEMA,
)
from trade_rl.evaluation.forward_study.seal import (
    CodeIdentity,
    Fs1Refusal,
    ScoringPermit,
    authorize_scoring,
    parse_manifest,
    require_permit,
)

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-123456789"
FETCH_AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-111111111"
SHA = "a" * 40
AFTER_WINDOW = datetime(2027, 5, 1, 0, 0, tzinfo=UTC)


def manifest_dict(**overrides: Any) -> dict[str, Any]:
    first = np.datetime64(FS1_REGISTRY.dataset_first_row_latest)
    last = np.datetime64(FS1_REGISTRY.scoring_end)
    value: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": "b" * 64,
        "symbols": list(FS1_REGISTRY.symbols),
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "n_bars": int((last - first) / np.timedelta64(1, "h")) + 1,
        "build_policy_digest": "c" * 64,
        "source_manifest": {"BTCUSDT-1h-2026-11.zip": "d" * 64},
        "authorization_reference": FETCH_AUTH,
        "sealed_utc": "2027-05-02T00:00:00",
    }
    value.update(overrides)
    return value


def manifest_raw(**overrides: Any) -> bytes:
    return (
        json.dumps(manifest_dict(**overrides), sort_keys=True, indent=2) + "\n"
    ).encode()


def good(**overrides: Any) -> dict[str, Any]:
    raw = overrides.pop("manifest_bytes", manifest_raw())
    kwargs: dict[str, Any] = {
        "now": AFTER_WINDOW,
        "authorization": AUTH,
        "expected_params_digest": FROZEN_PARAMS_DIGEST,
        "sealed_code_sha": SHA,
        "code_identity": CodeIdentity(SHA, True),
        "manifest_bytes": raw,
        "expected_manifest_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
    }
    kwargs.update(overrides)
    return kwargs


def reasons_of(**overrides: Any) -> tuple[str, ...]:
    with pytest.raises(Fs1Refusal) as caught:
        authorize_scoring(**good(**overrides))
    return caught.value.reasons


def test_every_precondition_met_returns_a_permit() -> None:
    permit = authorize_scoring(**good())
    assert isinstance(permit, ScoringPermit)
    assert permit.params_digest == FROZEN_PARAMS_DIGEST
    assert permit.authorization == AUTH and permit.code_sha == SHA
    assert permit.manifest.dataset_id == "b" * 64
    assert permit.authorized_at_utc == "2027-05-01T00:00:00Z"
    assert require_permit(permit) is permit


def test_refuses_before_the_scoring_end_by_wall_clock() -> None:
    just_before = AFTER_WINDOW - timedelta(seconds=1)
    assert reasons_of(now=just_before) == ("scoring_window_not_closed",)
    assert reasons_of(now=datetime(2026, 10, 30, tzinfo=UTC)) == (
        "scoring_window_not_closed",
    )


def test_the_clock_is_compared_in_utc() -> None:
    plus_nine = timezone(timedelta(hours=9))
    # 2027-05-01T08:59 in +09:00 is still 2027-04-30T23:59 UTC.
    early = datetime(2027, 5, 1, 8, 59, tzinfo=plus_nine)
    assert reasons_of(now=early) == ("scoring_window_not_closed",)
    authorize_scoring(**good(now=datetime(2027, 5, 1, 9, 0, tzinfo=plus_nine)))
    with pytest.raises(ValueError):
        authorize_scoring(**good(now=datetime(2027, 6, 1)))


@pytest.mark.parametrize("missing", [None, ""])
def test_refuses_without_an_authorization_reference(missing: str | None) -> None:
    assert reasons_of(authorization=missing) == ("authorization_missing",)


@pytest.mark.parametrize(
    "bad",
    [
        "yes",
        "https://github.com/shuntatsu/trade_rl/issues/810",
        "https://github.com/shuntatsu/trade_rl/pull/810#issuecomment-1",
        "https://github.com/other/trade_rl/issues/810#issuecomment-1",
        "http://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1 ",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-abc",
    ],
)
def test_refuses_a_malformed_authorization_reference(bad: str) -> None:
    assert reasons_of(authorization=bad) == ("authorization_malformed",)


def test_refuses_when_the_params_digest_differs_from_the_seal_record() -> None:
    assert reasons_of(expected_params_digest="0" * 64) == ("params_digest_mismatch",)
    assert reasons_of(expected_params_digest=None) == ("params_digest_mismatch",)


def test_refuses_when_the_registry_is_not_the_frozen_one() -> None:
    altered = replace(FS1_REGISTRY, max_drawdown_limit_pct=30.0)
    reasons = reasons_of(registry=altered, expected_params_digest=altered.digest())
    assert reasons == ("registry_digest_differs_from_frozen",)
    # An operator pasting the frozen digest does not hide the alteration either.
    both = reasons_of(registry=altered)
    assert both == ("registry_digest_differs_from_frozen", "params_digest_mismatch")


def test_refuses_when_the_code_is_not_the_sealed_commit() -> None:
    assert reasons_of(code_identity=CodeIdentity("b" * 40, True)) == (
        "code_sha_mismatch",
    )
    assert reasons_of(code_identity=CodeIdentity(SHA, False)) == ("code_tree_dirty",)
    assert reasons_of(code_identity=CodeIdentity(None, False)) == (
        "code_identity_unavailable",
    )


@pytest.mark.parametrize("bad", [None, "", "abc", "A" * 40, "a" * 39])
def test_refuses_without_a_full_sealed_code_sha(bad: str | None) -> None:
    assert "sealed_code_sha_missing" in reasons_of(sealed_code_sha=bad)


def test_refuses_without_a_manifest() -> None:
    assert reasons_of(manifest_bytes=None, expected_manifest_sha256=None) == (
        "manifest_missing",
    )


def test_refuses_when_the_manifest_digest_is_not_the_recorded_one() -> None:
    assert reasons_of(expected_manifest_sha256="0" * 64) == (
        "manifest_sha256_mismatch",
    )
    assert reasons_of(expected_manifest_sha256=None) == ("manifest_sha256_mismatch",)
    tampered = manifest_raw(dataset_id="e" * 64)
    assert reasons_of(
        manifest_bytes=tampered,
        expected_manifest_sha256=hashlib.sha256(manifest_raw()).hexdigest(),
    ) == ("manifest_sha256_mismatch",)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        json.dumps({**manifest_dict(), "extra": 1}).encode(),
        json.dumps(manifest_dict(schema_version="other")).encode(),
        json.dumps(manifest_dict(dataset_id="XYZ")).encode(),
        json.dumps(manifest_dict(source_manifest={})).encode(),
        json.dumps(manifest_dict(source_manifest={"a": "short"})).encode(),
        json.dumps(manifest_dict(authorization_reference="yes")).encode(),
        json.dumps(manifest_dict(n_bars=1.5)).encode(),
        json.dumps(manifest_dict(symbols=[])).encode(),
    ],
)
def test_refuses_an_invalid_manifest(raw: bytes) -> None:
    assert reasons_of(manifest_bytes=raw) == ("manifest_invalid",)
    with pytest.raises(ValueError):
        parse_manifest(raw)


def test_refuses_a_manifest_for_another_window_or_roster() -> None:
    wrong_last = manifest_raw(last_timestamp="2027-04-30T00:00:00")
    assert reasons_of(manifest_bytes=wrong_last) == ("manifest_window_mismatch",)
    wrong_symbols = manifest_raw(symbols=["BTCUSDT", "ETHUSDT"])
    assert reasons_of(manifest_bytes=wrong_symbols) == ("manifest_window_mismatch",)
    late_first = manifest_raw(first_timestamp="2021-03-01T00:00:00")
    assert reasons_of(manifest_bytes=late_first) == ("manifest_window_mismatch",)


def test_all_unmet_preconditions_are_reported_together() -> None:
    reasons = reasons_of(
        now=datetime(2026, 10, 30, tzinfo=UTC),
        authorization=None,
        expected_params_digest="0" * 64,
        code_identity=CodeIdentity("b" * 40, False),
        manifest_bytes=None,
        expected_manifest_sha256=None,
    )
    assert set(reasons) == {
        "scoring_window_not_closed",
        "authorization_missing",
        "params_digest_mismatch",
        "code_sha_mismatch",
        "code_tree_dirty",
        "manifest_missing",
    }


def test_a_permit_cannot_be_forged_by_construction() -> None:
    real = authorize_scoring(**good())
    forged = ScoringPermit(
        registry=real.registry,
        manifest=real.manifest,
        manifest_sha256=real.manifest_sha256,
        authorization=real.authorization,
        params_digest=real.params_digest,
        code_sha=real.code_sha,
        authorized_at_utc=real.authorized_at_utc,
    )
    with pytest.raises(TypeError):
        require_permit(forged)
    with pytest.raises(TypeError):
        require_permit(object())


def test_current_code_identity_reports_a_checkout_and_degrades_without_git(
    tmp_path: Any,
) -> None:
    identity = seal.current_code_identity()
    assert identity.git_head is None or len(identity.git_head) == 40
    bare = seal.current_code_identity(tmp_path)
    assert bare == CodeIdentity(None, False)


def test_manifest_round_trip_is_canonical_and_strict() -> None:
    raw = manifest_raw()
    parsed = parse_manifest(raw)
    assert parsed.symbols == FS1_REGISTRY.symbols
    assert parsed.authorization_reference == FETCH_AUTH
    assert np.datetime64(parsed.last_timestamp) == np.datetime64(
        FS1_REGISTRY.scoring_end
    )
