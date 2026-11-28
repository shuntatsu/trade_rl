"""Read-only advisory inventory of the exact committed PyPI lock (#1441)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}")
_PIN = re.compile(r"==([0-9]+\.[0-9]+\.[0-9]+)")
_AUDIT_ARGUMENTS = [
    "audit",
    "--frozen",
    "--no-build",
    "--output-format",
    "json",
    "--service-url",
    "https://api.osv.dev/",
]


@dataclass(frozen=True)
class AuditReport:
    audited_packages: int
    advisories: tuple[tuple[str, str, str], ...]
    adverse_statuses: int

    @property
    def has_findings(self) -> bool:
        return bool(self.advisories or self.adverse_statuses)


def _token(value: object) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError("invalid package identity or advisory identifier")
    return value


def locked_packages(raw: str) -> frozenset[tuple[str, str]]:
    entries = tomllib.loads(raw).get("package")
    if not isinstance(entries, list):
        raise ValueError("missing lock package roster")
    result: set[tuple[str, str]] = set()
    local_roots = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid lock entry")
        source = entry.get("source")
        if source == {"editable": "."} and entry.get("name") == "trade-rl":
            local_roots += 1
            continue
        if source != {"registry": "https://pypi.org/simple"}:
            raise ValueError("unsupported lock source; advisory coverage unknown")
        package = (_token(entry.get("name")), _token(entry.get("version")))
        if package in result:
            raise ValueError("duplicate locked package identity")
        result.add(package)
    if local_roots != 1 or not result:
        raise ValueError("incomplete lock package roster")
    return frozenset(result)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def parse_report(
    raw: str, exit_code: int, packages: frozenset[tuple[str, str]]
) -> AuditReport:
    if exit_code not in (0, 1):
        raise ValueError("scanner failed")
    report = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(report, dict) or report.get("schema") != {"version": "preview"}:
        raise ValueError("unrecognized audit report schema")
    summary = report.get("summary")
    advisories = report.get("vulnerabilities")
    adverse = report.get("adverse_statuses")
    if (
        not isinstance(summary, dict)
        or not isinstance(advisories, list)
        or not isinstance(adverse, list)
        or any(not isinstance(entry, dict) or not entry for entry in adverse)
    ):
        raise ValueError("incomplete audit report")
    for key, expected in (
        ("audited_packages", len(packages)),
        ("vulnerabilities", len(advisories)),
        ("adverse_statuses", len(adverse)),
    ):
        count = summary.get(key)
        if type(count) is not int or count != expected:
            raise ValueError("audit coverage or finding count mismatch")
    if not packages or (exit_code == 1 and not (advisories or adverse)):
        raise ValueError("empty audit or inconsistent scanner status")
    if exit_code == 0 and advisories:
        raise ValueError("scanner reported success despite vulnerabilities")

    found: set[tuple[str, str, str]] = set()
    for item in advisories:
        if not isinstance(item, dict) or not isinstance(item.get("dependency"), dict):
            raise ValueError("invalid advisory record")
        dep = item["dependency"]
        package = (_token(dep.get("name")), _token(dep.get("version")))
        if package not in packages:
            raise ValueError("advisory does not match the lock")
        record = (*package, _token(item.get("id")))
        if record in found:
            raise ValueError("duplicate advisory record")
        found.add(record)
    return AuditReport(len(packages), tuple(sorted(found)), len(adverse))


def required_uv_version(project: bytes) -> str:
    uv = tomllib.loads(project.decode("utf-8"))["tool"]["uv"]
    if not isinstance(uv, dict):
        raise ValueError("invalid uv configuration")
    pin = uv.get("required-version")
    match = _PIN.fullmatch(pin) if isinstance(pin, str) else None
    if not match:
        raise ValueError("an exact uv pin is required")
    # This monitor supports the project's current pin-only uv configuration.
    # New project settings must be explicitly reviewed for audit coverage.
    if (
        set(uv) != {"required-version"}
        or Path("uv.toml").exists()
        or Path("uv.toml").is_symlink()
        or any(os.environ.get(key) for key in ("UV_NO_CONFIG", "UV_CONFIG_FILE"))
    ):
        raise ValueError("unsupported audit configuration or override")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="PR reporting only; findings remain visible",
    )
    args = parser.parse_args(argv)
    manifest = Path("pyproject.toml")
    lock = Path("uv.lock")
    try:
        before = (manifest.read_bytes(), lock.read_bytes())
        expected_uv = required_uv_version(before[0])
        packages = locked_packages(before[1].decode("utf-8"))
        # The explicit, pin-enforcing config prevents host/user uv settings
        # from silently filtering packages or suppressing known advisories.
        # Neither --no-config nor a relaxed project version is used.
        env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
        env.update(UV_PYTHON_PREFERENCE="only-system", UV_PYTHON_DOWNLOADS="never")
        version = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=env,
        )
        match = re.match(r"^uv ([0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)", version.stdout)
        if version.returncode or not match or match.group(1) != expected_uv:
            raise ValueError("actual uv version differs from project pin")
        with tempfile.TemporaryDirectory(prefix="locked-uv-audit-") as directory:
            config = Path(directory) / "uv.toml"
            config.write_text(
                f'required-version = "=={expected_uv}"\n', encoding="utf-8"
            )
            result = subprocess.run(
                ["uv", "--config-file", str(config), *_AUDIT_ARGUMENTS],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=env,
            )
        if (manifest.read_bytes(), lock.read_bytes()) != before:
            raise ValueError("manifest or lock changed during scan")
        report = parse_report(result.stdout, result.returncode, packages)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError):
        # Do not echo untrusted advisory descriptions or raw command errors.
        print("SCAN_FAILED: incomplete, invalid, or unavailable advisory inventory")
        return 2

    print("## Locked Python dependency advisory inventory")
    print()
    print(
        "SCAN_STATUS: " + ("FINDINGS_PRESENT" if report.has_findings else "NO_FINDINGS")
    )
    print(f"- Audited registry packages: {report.audited_packages}")
    print(f"- Advisory entries: {len(report.advisories)}")
    print(f"- Adverse statuses: {report.adverse_statuses}")
    print("- PR report-only mode: " + ("yes" if args.report_only else "no"))
    print("- This inventory is not security clearance or Dependabot recovery.")
    if report.advisories:
        print()
        print("| Package | Locked version | Advisory |")
        print("|---|---|---|")
        for name, version_text, advisory_id in report.advisories:
            print(f"| {name} | {version_text} | {advisory_id} |")
    return 1 if report.has_findings and not args.report_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
