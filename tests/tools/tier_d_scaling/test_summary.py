"""Incomplete measurements cannot become a scaling curve or a byte oracle."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.tier_d_nonrl.tier_d_records import output_manifest, save_gzip, write_json
from tools.tier_d_scaling.supervision import LIMITS
from trade_rl.artifacts import canonical_json_bytes


def _case(
    root: Path,
    name: str,
    target: int,
    profile: bool,
    status: str = "PREFIX_OBSERVED",
) -> None:
    root.mkdir(parents=True)
    raw = {"status": "failed", "frontier": 2903 + target}
    save_gzip(root, "receipt.partial.json.gz", raw)
    digest = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    write_json(
        root / "measurement.json",
        {
            "case": {"name": name, "target": target, "profile": profile},
            "protocol": {
                "scope": "ARTIFICIAL_SOFTWARE_PROFILE_ONLY",
                "measurement": {"limits": dict(LIMITS)},
            },
            "plan_digest": "b" * 64,
            "provenance": {"source_head": "c" * 40},
            "prepared_root": "same-prepared",
            "status": status,
            "expected_stop": True,
            "count": target,
            "elapsed_seconds": float(target) / 10,
            "readback_verified": True,
            "receipt_canonical_sha256": digest,
            "financial_authority": False,
        },
    )
    write_json(root / "output-manifest.json", output_manifest(root))


def _supervision(root: Path, **overrides: Any) -> None:
    wrapper = root.parent
    name = json.loads((root / "measurement.json").read_text(encoding="utf-8"))["case"][
        "name"
    ]
    record = {
        "status": "COMPLETED",
        "command": [
            "python",
            "-B",
            "-m",
            "tools.tier_d_scaling",
            "--worker",
            "measure",
            "--root",
            str(root.resolve()),
            "--prepared-root",
            str(wrapper.parent / "prepared"),
            "--case",
            name,
        ],
        "returncode": 0,
        "worker_reaped": True,
        "owned_members_remaining": 0,
        "failure": None,
        "cleanup_error": None,
        "final_violation": [],
        "final_sample_error": None,
        "limits": dict(LIMITS),
    }
    record.update(overrides)
    write_json(wrapper / "supervision.json", record)


def _four_valid_cases(tmp_path: Path) -> list[Path]:
    roots = []
    for name, target, profile in (
        ("n100", 100, False),
        ("n400", 400, False),
        ("n1600", 1600, False),
        ("n1600-profile", 1600, True),
    ):
        root = tmp_path / name / "attempt"
        _case(root, name, target, profile)
        _supervision(root)
        roots.append(root)
    return roots


def test_partial_curve_and_missing_profile_remain_unestablished(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.tier_d_scaling.summary")
    root = tmp_path / "a" / "attempt"
    _case(root, "n100", 100, False)
    _supervision(root)
    value = module.summarize([root])
    assert value["status"] == "INCOMPLETE"
    assert value["receipt_byte_oracle"] == "NOT_ESTABLISHED"
    assert value["time_ratios"] == []


def test_identical_same_plan_profile_bytes_and_curve(tmp_path: Path) -> None:
    module = importlib.import_module("tools.tier_d_scaling.summary")
    roots = _four_valid_cases(tmp_path)
    value = module.summarize(roots)
    assert value["status"] == "MEASUREMENTS_COMPLETE"
    assert value["receipt_byte_oracle"] == "MATCH"
    assert value["time_ratios"] == [4.0, 4.0]
    assert value["historical_release_equivalence"] == "NOT_ESTABLISHED"
    assert value["financial_authority"] is False


@pytest.mark.parametrize(
    "invalid_record",
    [
        "missing",
        "failed",
        "foreign_root",
        "returncode",
        "unreaped",
        "remaining",
        "failure",
        "cleanup_error",
        "final_violation",
        "final_sample_error",
        "limits",
        "protocol_limits",
    ],
)
def test_invalid_or_unbound_supervision_cannot_complete_measurements(
    tmp_path: Path, invalid_record: str
) -> None:
    module = importlib.import_module("tools.tier_d_scaling.summary")
    roots = _four_valid_cases(tmp_path)
    supervision_path = roots[0].parent / "supervision.json"
    if invalid_record == "missing":
        supervision_path.unlink()
    else:
        record = json.loads(supervision_path.read_text(encoding="utf-8"))
        if invalid_record == "failed":
            record["status"] = "FAILED"
        elif invalid_record == "foreign_root":
            root_flag = record["command"].index("--root")
            record["command"][root_flag + 1] = str(tmp_path / "somewhere-else")
        elif invalid_record == "returncode":
            record["returncode"] = 1
        elif invalid_record == "unreaped":
            record["worker_reaped"] = False
        elif invalid_record == "remaining":
            record["owned_members_remaining"] = 1
        elif invalid_record == "failure":
            record["failure"] = "worker failed"
        elif invalid_record == "cleanup_error":
            record["cleanup_error"] = "cleanup failed"
        elif invalid_record == "final_violation":
            record["final_violation"] = ["rss_bytes"]
        elif invalid_record == "final_sample_error":
            record["final_sample_error"] = "final sample failed"
        elif invalid_record == "limits":
            record["limits"]["rss_bytes"] += 1
        elif invalid_record == "protocol_limits":
            for root in roots:
                measurement = json.loads(
                    (root / "measurement.json").read_text(encoding="utf-8")
                )
                measurement["protocol"]["measurement"]["limits"]["rss_bytes"] += 1
                (root / "measurement.json").write_text(
                    json.dumps(measurement), encoding="utf-8"
                )
                (root / "output-manifest.json").write_text(
                    json.dumps(output_manifest(root)), encoding="utf-8"
                )
        supervision_path.write_text(json.dumps(record), encoding="utf-8")

    value = module.summarize(roots)
    assert value["status"] == "INCOMPLETE"
    assert value["financial_authority"] is False


def test_modified_case_artifact_is_rejected(tmp_path: Path) -> None:
    module = importlib.import_module("tools.tier_d_scaling.summary")
    root = tmp_path / "a" / "attempt"
    _case(root, "n100", 100, False)
    _supervision(root)
    (root / "measurement.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        module.summarize([root])
