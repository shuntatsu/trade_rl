from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from tests.evaluation.test_directional_selection import _result, _roster
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation import ppo_risk_study as study
from trade_rl.evaluation.ppo_risk_study import compare_paired_ppo


def test_lower_loss_is_relative_improvement_and_never_profitability() -> None:
    baseline = _roster()
    candidate = {f"ppo{i}": _result(-0.05) for i in range(5)}
    report = compare_paired_ppo(baseline, candidate)
    assert report["decision"] == "RELATIVE_IMPROVEMENT_ONLY"
    assert report["candidate_family"]["qualified"] is False
    assert report["production_eligible"] is False


def test_four_paired_wins_are_required_and_missing_seed_is_rejected() -> None:
    baseline = _roster()
    candidate = {f"ppo{i}": _result(-0.05 if i < 3 else -0.2) for i in range(5)}
    assert compare_paired_ppo(baseline, candidate)["decision"] == "KEEP_BASELINE"
    del candidate["ppo4"]
    with pytest.raises(ValueError, match="five"):
        compare_paired_ppo(baseline, candidate)


def test_risk_violation_or_missing_period_prevents_relative_pass() -> None:
    baseline = _roster()
    candidate = {f"ppo{i}": _result(0.1) for i in range(5)}
    broken = deepcopy(candidate)
    broken["ppo0"]["ledger_max_drawdown"] = 0.21
    assert compare_paired_ppo(baseline, broken)["decision"] == "KEEP_BASELINE"
    broken = deepcopy(candidate)
    broken["ppo0"]["returns"] = [0.0]
    assert compare_paired_ppo(baseline, broken)["decision"] == "KEEP_BASELINE"
    assert (
        compare_paired_ppo(baseline, candidate)["decision"]
        == "PROSPECTIVE_PAPER_REQUIRED"
    )


def _artifact(root: Path) -> dict:
    protocol = {
        "provenance": {"implementation_digest": "frozen"},
        "training_risk": asdict(study.MATCHED_RISK),
    }
    directory = root / "ppo0"
    directory.mkdir()
    (directory / "model.zip").write_bytes(b"frozen model")
    row = {
        "arm": "ppo0",
        "protocol_digest": content_digest(protocol),
        "provenance": protocol["provenance"],
        "training_risk": protocol["training_risk"],
        "model_sha256": {"model.zip": sha256(b"frozen model").hexdigest()},
    }
    _publish(directory, row)
    return protocol


def _publish(directory: Path, row: dict) -> None:
    raw = canonical_json_bytes(row)
    (directory / "result.json").write_bytes(raw)
    (directory / "result.sha256.json").write_bytes(
        canonical_json_bytes({"sha256": sha256(raw).hexdigest()})
    )


def test_candidate_artifact_rejects_wrong_training_risk_after_rehash(
    tmp_path: Path,
) -> None:
    protocol = _artifact(tmp_path)
    row = study._read_arm(tmp_path, "ppo0", protocol)
    row["training_risk"]["max_gross"] = 1.0
    _publish(tmp_path / "ppo0", row)
    with pytest.raises(ValueError, match="training risk"):
        study._read_arm(tmp_path, "ppo0", protocol)


@pytest.mark.parametrize("damage", ["result", "model", "failed", "roster"])
def test_candidate_evidence_rejects_corruption(tmp_path: Path, damage: str) -> None:
    protocol = _artifact(tmp_path)
    directory = tmp_path / "ppo0"
    if damage == "result":
        (directory / "result.json").write_bytes(b"{}")
    elif damage == "model":
        (directory / "model.zip").write_bytes(b"replacement")
    elif damage == "failed":
        (directory / "failed.json").write_bytes(b"{}")
    else:
        row = study._read_arm(tmp_path, "ppo0", protocol)
        row["model_sha256"] = {}
        _publish(directory, row)
    with pytest.raises(ValueError):
        study._read_arm(tmp_path, "ppo0", protocol)


def test_factor_boundary_rejects_nonfactor_change_and_bad_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "trade_rl"
    (package / "evaluation").mkdir(parents=True)
    current = package / "evaluation" / "metrics.py"
    current.write_bytes(b"VALUE = 1\n")
    monkeypatch.setattr(
        study, "__file__", str(package / "evaluation" / "ppo_risk_study.py")
    )
    protocol = {
        "provenance": {
            "implementation": {
                "files": [
                    {
                        "path": "evaluation/metrics.py",
                        "sha256": sha256(b"VALUE = 1\n").hexdigest(),
                    }
                ]
            }
        }
    }
    with ZipFile(tmp_path / "source-snapshot.zip", "w") as archive:
        archive.writestr("trade_rl/evaluation/metrics.py", b"VALUE = 1\n")
    study._verify_factor_boundary(tmp_path, protocol)
    current.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="non-factor"):
        study._verify_factor_boundary(tmp_path, protocol)
    with ZipFile(tmp_path / "source-snapshot.zip", "w") as archive:
        archive.writestr("trade_rl/evaluation/metrics.py", b"VALUE = 2\n")
    with pytest.raises(ValueError, match="snapshot hash"):
        study._verify_factor_boundary(tmp_path, protocol)


def test_runtime_drift_is_not_an_allowed_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = {"provenance": {"runtime_environment": {"torch": "old"}}}
    monkeypatch.setattr(study, "BASELINE_PROTOCOL", content_digest(protocol))
    study.reserve_study(tmp_path / "baseline", protocol)
    monkeypatch.setattr(
        study,
        "expected_protocol",
        lambda source: {"provenance": {"runtime_environment": {"torch": "new"}}},
    )
    with pytest.raises(ValueError, match="runtime"):
        study._baseline(tmp_path / "source", tmp_path / "baseline")
