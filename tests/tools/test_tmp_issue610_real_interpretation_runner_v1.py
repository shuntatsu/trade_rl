from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
import tools.tmp_issue610_real_interpretation_runner_v1 as runner
from tools.tmp_issue610_real_interpretation_runner_v1 import (
    build_decision_envelope,
    build_strict_precheck_report,
)


def _strict(
    *,
    new_termination: tuple[str, ...] = (),
    invalid: tuple[str, ...] = (),
) -> dict[str, object]:
    return build_strict_precheck_report(
        interpretation_run_id=123,
        precompute_artifact_id=10,
        precompute_artifact_digest="sha256:" + "a" * 64,
        candidate_artifact_id=20,
        candidate_artifact_digest="sha256:" + "b" * 64,
        new_termination=new_termination,
        invalid_termination_evidence=invalid,
    )


def _v2(
    *,
    decision: str = "KEEP_BASELINE",
    new_termination: tuple[str, ...] = (),
    termination_validity: tuple[str, ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "decision": decision,
        "new_termination_violations": list(new_termination),
        "termination_validity_violations": list(termination_validity),
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def test_invalid_strict_evidence_short_circuits_economic_interpretation() -> None:
    strict = _strict(invalid=("candidate PPO termination evidence malformed",))
    envelope = build_decision_envelope(strict_precheck=strict, v2_result=None)

    assert envelope["decision"] == "INVALID"
    assert envelope["v2_result_content_digest"] is None
    assert envelope["strict_termination_evidence_valid"] is False
    assert envelope["economic_values_interpreted"] is False
    assert envelope["candidate_retrained_during_interpretation"] is False
    assert envelope["final_test_accessed"] is False
    assert envelope["production_eligible"] is False
    assert envelope["live_trading_authorized"] is False
    assert envelope["merge_authorized"] is False


def test_invalid_precheck_never_invokes_v2_economic_interpretation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "precheck_candidate_artifact",
        lambda **_kwargs: ((), ("candidate termination evidence malformed",)),
    )

    def forbidden_v2(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("v2 economic interpretation must not run")

    monkeypatch.setattr(runner, "interpret_candidate_v2", forbidden_v2)
    output_root = tmp_path / "result"
    envelope = runner.interpret_real_candidate_v1(
        source_root=tmp_path / "source",
        candidate_root=tmp_path / "candidate",
        precompute_path=tmp_path / "precompute.json",
        output_root=output_root,
        artifact_module_path=tmp_path / "artifact.py",
        interpretation_run_id=123,
        precompute_artifact_id=10,
        precompute_artifact_digest="sha256:" + "a" * 64,
        candidate_artifact_id=20,
        candidate_artifact_digest="sha256:" + "b" * 64,
    )

    assert envelope["decision"] == "INVALID"
    assert (output_root / "strict-precheck.json").is_file()
    assert (output_root / "decision.json").is_file()
    assert not (output_root / "v2").exists()


def test_valid_strict_evidence_requires_frozen_v2_interpretation() -> None:
    with pytest.raises(
        RuntimeError,
        match="valid termination evidence requires frozen v2 interpretation",
    ):
        build_decision_envelope(strict_precheck=_strict(), v2_result=None)


def test_valid_accept_candidate_is_preserved_without_changing_v2_decision() -> None:
    strict = _strict()
    v2_result = _v2(decision="ACCEPT_CANDIDATE")
    envelope = build_decision_envelope(
        strict_precheck=strict,
        v2_result=v2_result,
    )

    assert envelope["decision"] == "ACCEPT_CANDIDATE"
    assert envelope["v2_result_content_digest"] == v2_result["content_digest"]
    assert envelope["strict_termination_evidence_valid"] is True
    assert envelope["economic_values_interpreted"] is True


def test_valid_keep_baseline_with_genuine_new_termination_is_preserved() -> None:
    strict_new = ("new PPO termination: seed=0 BTCUSDT reason=economic_floor",)
    strict = _strict(new_termination=strict_new)
    v2_result = _v2(
        decision="KEEP_BASELINE",
        new_termination=("new PPO termination reason: seed=0 symbol=BTCUSDT",),
    )
    envelope = build_decision_envelope(
        strict_precheck=strict,
        v2_result=v2_result,
    )

    assert envelope["decision"] == "KEEP_BASELINE"
    assert envelope["strict_new_termination_present"] is True
    assert envelope["economic_values_interpreted"] is True


def test_strict_and_v2_new_termination_disagreement_fails_closed() -> None:
    strict = _strict(
        new_termination=("new PPO termination: seed=0 BTCUSDT reason=economic_floor",)
    )
    with pytest.raises(
        RuntimeError,
        match="strict/v2 new-termination classification disagreement",
    ):
        build_decision_envelope(strict_precheck=strict, v2_result=_v2())


def test_v2_termination_validity_disagreement_fails_closed() -> None:
    with pytest.raises(
        RuntimeError,
        match="strict/v2 termination validity disagreement",
    ):
        build_decision_envelope(
            strict_precheck=_strict(),
            v2_result=_v2(termination_validity=("old classifier rejected evidence",)),
        )


def test_invalid_strict_evidence_forbids_v2_result() -> None:
    with pytest.raises(
        RuntimeError,
        match="v2 interpretation forbidden for invalid termination evidence",
    ):
        build_decision_envelope(
            strict_precheck=_strict(invalid=("malformed",)),
            v2_result=_v2(),
        )


def test_tampered_v2_digest_fails_closed() -> None:
    v2_result = _v2()
    v2_result["content_digest"] = "0" * 64
    with pytest.raises(RuntimeError, match="v2 result content digest mismatch"):
        build_decision_envelope(
            strict_precheck=_strict(),
            v2_result=v2_result,
        )


def test_tampered_strict_digest_fails_closed() -> None:
    strict = _strict()
    strict["content_digest"] = "0" * 64
    with pytest.raises(RuntimeError, match="strict precheck content digest mismatch"):
        build_decision_envelope(
            strict_precheck=strict,
            v2_result=_v2(),
        )
