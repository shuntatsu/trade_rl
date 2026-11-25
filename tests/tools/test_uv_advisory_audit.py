"""Network-free contracts for the locked dependency advisory monitor."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

LOCK = """
[[package]]
name = "trade-rl"
source = { editable = "." }

[[package]]
name = "example-package"
version = "1.2.3"
source = { registry = "https://pypi.org/simple" }
"""
PROJECT = '[tool.uv]\nrequired-version = "==0.13.10"\n'


def _report(findings: bool = False) -> dict:
    entries = (
        [
            {
                "dependency": {"name": "example-package", "version": "1.2.3"},
                "id": "GHSA-example-1234",
                "description": "untrusted advisory prose",
            }
        ]
        if findings
        else []
    )
    return {
        "schema": {"version": "preview"},
        "summary": {
            "audited_packages": 1,
            "vulnerabilities": len(entries),
            "adverse_statuses": 0,
        },
        "vulnerabilities": entries,
        "adverse_statuses": [],
    }


def test_report_distinguishes_complete_scan_from_findings() -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    packages = locked_packages(LOCK)
    assert packages == {("example-package", "1.2.3")}
    assert not parse_report(json.dumps(_report()), 0, packages).has_findings
    result = parse_report(json.dumps(_report(True)), 1, packages)
    assert result.has_findings
    assert result.advisories == (("example-package", "1.2.3", "GHSA-example-1234"),)


@pytest.mark.parametrize("code", [1, 2, 127, -9])
def test_failed_scanner_not_clean(code: int) -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    with pytest.raises(ValueError):
        parse_report(json.dumps(_report()), code, locked_packages(LOCK))


@pytest.mark.parametrize(
    "field,value",
    [
        ("audited_packages", 0),
        ("audited_packages", 2),
        ("audited_packages", True),
        ("vulnerabilities", 0),
        ("adverse_statuses", 1),
    ],
)
def test_incomplete_report_refused(field: str, value: object) -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    report = _report(True)
    report["summary"][field] = value
    with pytest.raises(ValueError):
        parse_report(json.dumps(report), 1, locked_packages(LOCK))


@pytest.mark.parametrize("raw", ["", "not JSON", '{"summary":{},"summary":{}}'])
def test_malformed_json_refused(raw: str) -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    with pytest.raises(ValueError):
        parse_report(raw, 0, locked_packages(LOCK))


def test_unknown_source_refused() -> None:
    from tools.uv_advisory_audit import locked_packages

    with pytest.raises(ValueError):
        locked_packages(
            LOCK.replace(
                'registry = "https://pypi.org/simple"',
                'git = "https://example.com/repo"',
            )
        )


@pytest.mark.parametrize("report_only,expected", [(False, 1), (True, 0)])
def test_cli_findings_visible_without_install_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    report_only: bool,
    expected: int,
) -> None:
    from tools import uv_advisory_audit as audit

    monkeypatch.chdir(tmp_path)
    Path("pyproject.toml").write_text(PROJECT)
    Path("uv.lock").write_text(LOCK)
    calls = []
    monkeypatch.setenv("UV_NO_DEV", "true")
    monkeypatch.setenv("UV_AUDIT_IGNORE", "GHSA-example-1234")

    def run(argv, **kwargs):
        calls.append(argv)
        assert "UV_NO_DEV" not in kwargs["env"]
        assert "UV_AUDIT_IGNORE" not in kwargs["env"]
        assert kwargs["env"]["UV_PYTHON_DOWNLOADS"] == "never"
        if argv == ["uv", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "uv 0.13.10 (build)\n", "")
        assert argv[:2] == ["uv", "--config-file"]
        assert Path(argv[2]).read_text() == 'required-version = "==0.13.10"\n'
        return subprocess.CompletedProcess(argv, 1, json.dumps(_report(True)), "")

    monkeypatch.setattr(audit.subprocess, "run", run)
    assert audit.main(["--report-only"] if report_only else []) == expected
    assert calls[0] == ["uv", "--version"]
    assert calls[1][:2] == ["uv", "--config-file"]
    assert calls[1][3:] == [
        "audit",
        "--frozen",
        "--no-build",
        "--output-format",
        "json",
        "--service-url",
        "https://api.osv.dev/",
    ]
    assert not Path(calls[1][2]).exists()
    assert Path("uv.lock").read_text() == LOCK
    assert Path("pyproject.toml").read_text() == PROJECT
    output = capsys.readouterr().out
    assert "FINDINGS_PRESENT" in output
    assert "GHSA-example-1234" in output
    assert "untrusted advisory prose" not in output


def test_cli_wrong_version_or_mutation_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import uv_advisory_audit as audit

    monkeypatch.chdir(tmp_path)
    Path("pyproject.toml").write_text(PROJECT)
    Path("uv.lock").write_text(LOCK)
    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "uv 0.13.7\n", ""),
    )
    assert audit.main(["--report-only"]) == 2

    def mutate(argv, **kwargs):
        if argv == ["uv", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "uv 0.13.10\n", "")
        Path("uv.lock").write_text(LOCK + "\n# changed")
        return subprocess.CompletedProcess(argv, 0, json.dumps(_report()), "")

    monkeypatch.setattr(audit.subprocess, "run", mutate)
    assert audit.main(["--report-only"]) == 2


@pytest.mark.parametrize("field,value", [("id", "bad\nannotation"), ("id", None)])
def test_invalid_advisory_identity_refused(field: str, value: object) -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    report = _report(True)
    report["vulnerabilities"][0][field] = value
    with pytest.raises(ValueError):
        parse_report(json.dumps(report), 1, locked_packages(LOCK))


def test_advisory_outside_lock_and_inconsistent_success_refused() -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    report = _report(True)
    with pytest.raises(ValueError):
        parse_report(json.dumps(report), 0, locked_packages(LOCK))
    report["vulnerabilities"][0]["dependency"]["version"] = "2.0.0"
    with pytest.raises(ValueError):
        parse_report(json.dumps(report), 1, locked_packages(LOCK))


def test_adverse_status_is_still_a_finding() -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    report = _report()
    report["summary"]["adverse_statuses"] = 1
    report["adverse_statuses"] = [{"status": "deprecated"}]
    assert parse_report(json.dumps(report), 0, locked_packages(LOCK)).has_findings


def test_advisory_suppression_refused_before_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import uv_advisory_audit as audit

    monkeypatch.chdir(tmp_path)
    Path("uv.lock").write_text(LOCK)
    Path("pyproject.toml").write_text(
        PROJECT + '\n[tool.uv.audit]\nignore = ["GHSA-example"]\n'
    )

    def must_not_run(*args, **kwargs):
        pytest.fail("scanner must not run when findings are suppressed")

    monkeypatch.setattr(audit.subprocess, "run", must_not_run)
    assert audit.main(["--report-only"]) == 2


def test_duplicate_package_or_advisory_refused() -> None:
    from tools.uv_advisory_audit import locked_packages, parse_report

    with pytest.raises(ValueError):
        locked_packages(LOCK + "[[package]]" + LOCK.split("[[package]]")[-1])
    report = _report(True)
    report["vulnerabilities"].append(report["vulnerabilities"][0])
    report["summary"]["vulnerabilities"] = 2
    with pytest.raises(ValueError):
        parse_report(json.dumps(report), 1, locked_packages(LOCK))


def test_external_config_override_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import uv_advisory_audit as audit

    monkeypatch.chdir(tmp_path)
    Path("uv.lock").write_text(LOCK)
    Path("pyproject.toml").write_text(PROJECT)
    monkeypatch.setenv("UV_NO_CONFIG", "1")

    def must_not_run(*args, **kwargs):
        pytest.fail("configuration override must not run")

    monkeypatch.setattr(audit.subprocess, "run", must_not_run)
    assert audit.main(["--report-only"]) == 2
