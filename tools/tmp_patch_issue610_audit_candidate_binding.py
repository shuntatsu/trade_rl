from __future__ import annotations

from pathlib import Path


HELPER = Path("tools/tmp_issue610_real_interpretation_audit_v1.py")
TEST = Path("tests/tools/test_tmp_issue610_real_interpretation_audit_symbol_roster.py")

helper = HELPER.read_text(encoding="utf-8")

insert = r'''


def _is_sha256_api_digest(value: str) -> bool:
    prefix = "sha256:"
    if not value.startswith(prefix):
        return False
    digest = value[len(prefix) :]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _validate_candidate_authority(
    candidate: Mapping[str, object],
    *,
    precompute: Mapping[str, object],
    candidate_execution_run_id: int,
    candidate_artifact_id: int,
    candidate_artifact_digest: str,
    precompute_artifact_id: int,
    precompute_artifact_digest: str,
    producer_result_artifact_id: int,
    producer_result_artifact_digest: str,
    fresh_result_artifact_id: int,
    fresh_result_artifact_digest: str,
) -> None:
    """Fail closed on drift in the immutable candidate/reveal authority chain."""

    for label, artifact_id in (
        ("candidate artifact", candidate_artifact_id),
        ("precompute artifact", precompute_artifact_id),
        ("producer result artifact", producer_result_artifact_id),
        ("fresh result artifact", fresh_result_artifact_id),
    ):
        _require(artifact_id > 0, f"{label} id invalid")
    for label, digest in (
        ("candidate artifact", candidate_artifact_digest),
        ("precompute artifact", precompute_artifact_digest),
        ("producer result artifact", producer_result_artifact_digest),
        ("fresh result artifact", fresh_result_artifact_digest),
    ):
        _require(_is_sha256_api_digest(digest), f"{label} API digest malformed")

    _require(candidate.get("issue_number") == ISSUE_NUMBER, "candidate issue drift")
    _require(
        candidate.get("execution_run_id") == candidate_execution_run_id,
        "candidate execution run mismatch",
    )
    _require(
        candidate.get("precompute_run_id") == PRECOMPUTE_RUN_ID,
        "candidate precompute run mismatch",
    )
    _require(
        candidate.get("precompute_artifact_id") == precompute_artifact_id,
        "candidate precompute artifact id mismatch",
    )
    _require(
        candidate.get("precompute_artifact_api_digest") == precompute_artifact_digest,
        "candidate precompute artifact digest mismatch",
    )
    _require(
        candidate.get("precompute_authority_content_digest")
        == precompute.get("content_digest"),
        "candidate/precompute content binding mismatch",
    )
    _require(candidate.get("ppo_seeds") == list(SEEDS), "candidate seed roster drift")
    _require(candidate.get("symbols") == list(SYMBOLS), "candidate symbol roster drift")
    for field, expected in (
        ("baseline_retrained", False),
        ("candidate_training_performed", True),
        ("economic_values_interpreted", False),
        ("final_test_accessed", False),
        ("operational_eligibility_established", False),
        ("production_eligible", False),
        ("live_trading_authorized", False),
        ("merge_authorized", False),
    ):
        _require(
            candidate.get(field) is expected,
            f"candidate boundary mismatch: {field}",
        )
'''

marker = "\ndef audit_interpretation(\n"
if "def _validate_candidate_authority(" not in helper:
    if helper.count(marker) != 1:
        raise SystemExit("audit insertion marker mismatch")
    helper = helper.replace(marker, insert + marker, 1)

old = '''    precompute = _load_canonical(precompute_path)\n\n    _require(\n        candidate_execution_run_id == 35103004952,\n'''
new = '''    precompute = _load_canonical(precompute_path)\n\n    _validate_candidate_authority(\n        candidate,\n        precompute=precompute,\n        candidate_execution_run_id=candidate_execution_run_id,\n        candidate_artifact_id=candidate_artifact_id,\n        candidate_artifact_digest=candidate_artifact_digest,\n        precompute_artifact_id=precompute_artifact_id,\n        precompute_artifact_digest=precompute_artifact_digest,\n        producer_result_artifact_id=producer_result_artifact_id,\n        producer_result_artifact_digest=producer_result_artifact_digest,\n        fresh_result_artifact_id=fresh_result_artifact_id,\n        fresh_result_artifact_digest=fresh_result_artifact_digest,\n    )\n\n    _require(\n        candidate_execution_run_id == 35103004952,\n'''
if "_validate_candidate_authority(\n        candidate," not in helper:
    if helper.count(old) != 1:
        raise SystemExit("candidate validation call marker mismatch")
    helper = helper.replace(old, new, 1)

HELPER.write_text(helper, encoding="utf-8")

tests = TEST.read_text(encoding="utf-8")
tests = tests.replace(
    "    SYMBOLS,\n    _validate_absolute_diagnostic,\n",
    "    SEEDS,\n    SYMBOLS,\n    _validate_absolute_diagnostic,\n    _validate_candidate_authority,\n",
    1,
)

extra = r'''


_VALID_API_DIGEST = "sha256:" + "a" * 64
_PRECOMPUTE_CONTENT_DIGEST = "b" * 64


def _candidate_authority() -> dict[str, object]:
    return {
        "issue_number": 610,
        "execution_run_id": 35103004952,
        "precompute_run_id": 35098793118,
        "precompute_artifact_id": 10447243668,
        "precompute_artifact_api_digest": _VALID_API_DIGEST,
        "precompute_authority_content_digest": _PRECOMPUTE_CONTENT_DIGEST,
        "ppo_seeds": list(SEEDS),
        "symbols": list(SYMBOLS),
        "baseline_retrained": False,
        "candidate_training_performed": True,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }


def _validate_candidate(candidate: dict[str, object]) -> None:
    _validate_candidate_authority(
        candidate,
        precompute={"content_digest": _PRECOMPUTE_CONTENT_DIGEST},
        candidate_execution_run_id=35103004952,
        candidate_artifact_id=10453782249,
        candidate_artifact_digest=_VALID_API_DIGEST,
        precompute_artifact_id=10447243668,
        precompute_artifact_digest=_VALID_API_DIGEST,
        producer_result_artifact_id=10455728296,
        producer_result_artifact_digest=_VALID_API_DIGEST,
        fresh_result_artifact_id=10454484504,
        fresh_result_artifact_digest=_VALID_API_DIGEST,
    )


def test_candidate_authority_contract_is_accepted() -> None:
    _validate_candidate(_candidate_authority())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("issue_number", 611, "candidate issue drift"),
        ("execution_run_id", 1, "candidate execution run mismatch"),
        ("precompute_run_id", 1, "candidate precompute run mismatch"),
        ("precompute_artifact_id", 1, "candidate precompute artifact id mismatch"),
        (
            "precompute_artifact_api_digest",
            "sha256:" + "c" * 64,
            "candidate precompute artifact digest mismatch",
        ),
        (
            "precompute_authority_content_digest",
            "c" * 64,
            "candidate/precompute content binding mismatch",
        ),
        ("ppo_seeds", [0, 1, 2, 3], "candidate seed roster drift"),
        ("symbols", list(reversed(SYMBOLS)), "candidate symbol roster drift"),
        ("baseline_retrained", True, "candidate boundary mismatch: baseline_retrained"),
        (
            "candidate_training_performed",
            False,
            "candidate boundary mismatch: candidate_training_performed",
        ),
        (
            "economic_values_interpreted",
            True,
            "candidate boundary mismatch: economic_values_interpreted",
        ),
        ("final_test_accessed", True, "candidate boundary mismatch: final_test_accessed"),
        (
            "operational_eligibility_established",
            True,
            "candidate boundary mismatch: operational_eligibility_established",
        ),
        ("production_eligible", True, "candidate boundary mismatch: production_eligible"),
        (
            "live_trading_authorized",
            True,
            "candidate boundary mismatch: live_trading_authorized",
        ),
        ("merge_authorized", True, "candidate boundary mismatch: merge_authorized"),
    ),
)
def test_candidate_authority_drift_is_rejected(
    field: str, replacement: object, message: str
) -> None:
    candidate = _candidate_authority()
    candidate[field] = replacement
    with pytest.raises(RuntimeError, match=message):
        _validate_candidate(candidate)


def test_noncanonical_api_digest_is_rejected() -> None:
    candidate = _candidate_authority()
    with pytest.raises(RuntimeError, match="candidate artifact API digest malformed"):
        _validate_candidate_authority(
            candidate,
            precompute={"content_digest": _PRECOMPUTE_CONTENT_DIGEST},
            candidate_execution_run_id=35103004952,
            candidate_artifact_id=10453782249,
            candidate_artifact_digest="a" * 64,
            precompute_artifact_id=10447243668,
            precompute_artifact_digest=_VALID_API_DIGEST,
            producer_result_artifact_id=10455728296,
            producer_result_artifact_digest=_VALID_API_DIGEST,
            fresh_result_artifact_id=10454484504,
            fresh_result_artifact_digest=_VALID_API_DIGEST,
        )
'''
if "def test_candidate_authority_contract_is_accepted" not in tests:
    tests += extra
TEST.write_text(tests, encoding="utf-8")
