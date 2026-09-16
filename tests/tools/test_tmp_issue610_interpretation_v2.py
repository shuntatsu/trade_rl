from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tools.tmp_issue610_interpretation_v2 import (
    ABSOLUTE_DIAGNOSTIC_SCHEMA,
    RECOVERY_BINDING_SCHEMA,
    SEEDS,
    SYMBOLS,
    absolute_candidate_ppo_diagnostic,
    classify_termination_evidence,
    validate_recovery_binding,
)
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest


def _loaded(
    *,
    termination_count: int = 0,
    termination_reasons: list[str] | None = None,
    malformed: bool = False,
) -> SimpleNamespace:
    runs: dict[int, SimpleNamespace] = {}
    reasons = [] if termination_reasons is None else termination_reasons
    for seed in SEEDS:
        by_symbol: list[dict[str, object]] = []
        for symbol_index, symbol in enumerate(SYMBOLS):
            strategy: dict[str, object] = {
                "name": "ppo",
                "metrics": {
                    "termination_count": termination_count,
                    "total_return": 0.01 * (symbol_index + 1) + 0.001 * seed,
                },
                "diagnostics": {"termination_reasons": list(reasons)},
            }
            if malformed and seed == SEEDS[0] and symbol == SYMBOLS[0]:
                strategy["metrics"] = {
                    "termination_count": "bad",
                    "total_return": 0.01,
                }
            by_symbol.append({"symbol": symbol, "strategies": [strategy]})
        runs[seed] = SimpleNamespace(summary={"by_symbol": by_symbol})
    return SimpleNamespace(runs=runs)


def _write_canonical(path: Path, payload: dict[str, object]) -> dict[str, object]:
    resolved = dict(payload)
    resolved["content_digest"] = content_digest(resolved)
    path.write_bytes(canonical_json_bytes(resolved))
    return resolved


def test_malformed_termination_is_validity_violation_not_gate_failure() -> None:
    baseline = _loaded()
    malformed = _loaded(malformed=True)
    new_termination, invalid = classify_termination_evidence(baseline, malformed)
    assert new_termination == ()
    assert len(invalid) == 1
    assert invalid[0].startswith("termination evidence malformed:")


def test_new_termination_is_gate_failure_not_evidence_invalidity() -> None:
    baseline = _loaded(termination_count=1, termination_reasons=["risk_limit"])
    candidate = _loaded(
        termination_count=2,
        termination_reasons=["risk_limit", "economic_floor"],
    )
    new_termination, invalid = classify_termination_evidence(baseline, candidate)
    assert invalid == ()
    assert any("new PPO termination count" in item for item in new_termination)
    assert any("new PPO termination reason" in item for item in new_termination)


def test_absolute_candidate_diagnostic_publishes_each_symbol_seed_median() -> None:
    candidate = _loaded()
    symbol_medians = [0.01 * (index + 1) + 0.002 for index in range(len(SYMBOLS))]
    comparison: dict[str, object] = {
        "cross_symbol": {
            "ppo": {
                "median_candidate_total_return": symbol_medians[2],
            }
        }
    }
    diagnostic = absolute_candidate_ppo_diagnostic(candidate, comparison)
    assert diagnostic["schema_version"] == ABSOLUTE_DIAGNOSTIC_SCHEMA
    assert diagnostic["gates_development_decision"] is False
    by_symbol = diagnostic["by_symbol"]
    assert isinstance(by_symbol, dict)
    for index, symbol in enumerate(SYMBOLS):
        entry = by_symbol[symbol]
        assert isinstance(entry, dict)
        assert entry["median_candidate_total_return"] == symbol_medians[index]
        by_seed = entry["by_seed_total_return"]
        assert isinstance(by_seed, dict)
        assert tuple(by_seed) == tuple(str(seed) for seed in SEEDS)
    cross = diagnostic["cross_symbol"]
    assert isinstance(cross, dict)
    assert cross["symbol_count"] == 5
    assert cross["positive_symbol_count"] == 5
    assert cross["median_candidate_total_return"] == symbol_medians[2]


def test_absolute_candidate_diagnostic_rejects_comparison_mismatch() -> None:
    candidate = _loaded()
    comparison: dict[str, object] = {
        "cross_symbol": {"ppo": {"median_candidate_total_return": -999.0}}
    }
    try:
        absolute_candidate_ppo_diagnostic(candidate, comparison)
    except ValueError as error:
        assert "absolute candidate median mismatch" in str(error)
    else:
        raise AssertionError("mismatched absolute diagnostic was accepted")


def test_recovery_binding_requires_result_blind_consistency(tmp_path: Path) -> None:
    candidate_authority = _write_canonical(
        tmp_path / "candidate-authority.json",
        {
            "schema_version": "issue610_candidate_execution_v1",
            "candidate_evidence_fingerprint": "a" * 64,
            "candidate_training_performed": True,
            "economic_values_interpreted": False,
            "final_test_accessed": False,
        },
    )
    recovery = _write_canonical(
        tmp_path / "recovery-authority.json",
        {
            "schema_version": "issue610_preexecution_recovery_authority_v1",
            "recovery_candidate_execution_authorized": True,
            "failed_run_candidate_training_performed": False,
            "failed_run_candidate_artifact_published": False,
        },
    )
    binding = _write_canonical(
        tmp_path / "recovery-binding.json",
        {
            "schema_version": RECOVERY_BINDING_SCHEMA,
            "issue_number": 610,
            "candidate_evidence_fingerprint": candidate_authority[
                "candidate_evidence_fingerprint"
            ],
            "candidate_authority_content_digest": candidate_authority["content_digest"],
            "recovery_authority_content_digest": recovery["content_digest"],
            "failed_run_candidate_training_performed": False,
            "recovery_candidate_training_performed": True,
            "economic_values_interpreted": False,
            "final_test_accessed": False,
            "production_eligible": False,
            "live_trading_authorized": False,
            "merge_authorized": False,
        },
    )
    assert binding["content_digest"]
    assert validate_recovery_binding(tmp_path, candidate_authority) == ()

    forged = dict(binding)
    forged.pop("content_digest")
    forged["economic_values_interpreted"] = True
    forged["content_digest"] = content_digest(forged)
    (tmp_path / "recovery-binding.json").write_bytes(canonical_json_bytes(forged))
    violations = validate_recovery_binding(tmp_path, candidate_authority)
    assert (
        "candidate recovery binding mismatch: economic_values_interpreted" in violations
    )
