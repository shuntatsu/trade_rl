"""Bulk retirement preserves exact historical source and current shared contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tools.agent_repo.research_assurance import evaluate_assurance
from tools.agent_repo.source_index import ImportCollector, within_module
from tools.docs_source_consistency import check_retired_studies

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "report/legacy-research-retirement-20261126.json"


def test_inventory_and_archive_absence():
    check_retired_studies(ROOT)


def test_archive_sources_match_exact_git_bytes():
    manifest = json.loads(MANIFEST.read_text())
    for record in manifest["records"]:
        ref = f"{record['source_commit']}:{record['path']}"
        raw = subprocess.check_output(["git", "show", ref], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == record["sha256"], ref
        assert (
            subprocess.check_output(
                ["git", "rev-parse", ref], cwd=ROOT, text=True
            ).strip()
            == record["git_blob"]
        ), ref


def test_retired_modules_are_not_core_import_dependencies():
    manifest = json.loads(MANIFEST.read_text())
    modules = {
        row["path"].removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        for row in manifest["records"]
        if row["path"].startswith(("trade_rl/", "tools/"))
        and row["path"].endswith(".py")
    }
    for name in ("trade_rl", "tools"):
        package = ROOT / name
        collector = ImportCollector(package)
        for path in package.rglob("*.py"):
            assert not any(
                within_module(imported, module)
                for imported in collector.collect(path)
                for module in modules
            ), path.relative_to(ROOT)


# Source at the final pre-retirement synchronization, not a movable branch.
RETIREMENT_MAIN = "c9347f52b62ac8107db06c65675f1708423349e1"
PREVIOUS_MANIFEST = "1db92680cfe20f872073e71bd9df427411eb8f77"
PREVIOUS_AGGREGATE = "14432c4f78247d7c2ef3438eacae262f3551a9a0"
PACKET_BASE = "640a9bf3c6b6888d5e65a66d65188c5523342529"
UPDATED_ARCHIVE_PATHS = {
    "docs/specs/core7-mtf-shadow-anchor-ppo.md",
    "tests/evaluation/test_core7_mtf_oracle.py",
    "tests/evaluation/test_signal_diagnosis_run_packet.py",
    "trade_rl/evaluation/learning_lab/runner.py",
}


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", "--no-replace-objects", *args], cwd=ROOT)


def test_retirement_records_latest_source_and_preserves_previous_manifest() -> None:
    manifest = json.loads(MANIFEST.read_bytes())
    assert manifest["retirement_main"] == RETIREMENT_MAIN
    assert manifest["previous_manifest_commit"] == PREVIOUS_MANIFEST
    assert manifest["previous_aggregate"] == PREVIOUS_AGGREGATE
    previous = json.loads(
        _git_bytes(
            "show", f"{PREVIOUS_MANIFEST}:{MANIFEST.relative_to(ROOT).as_posix()}"
        )
    )
    assert manifest["inventory"] == previous["inventory"]
    before = {row["path"]: row for row in previous["records"]}
    after = {row["path"]: row for row in manifest["records"]}
    assert before.keys() == after.keys()
    assert {
        path for path in before if before[path] != after[path]
    } == UPDATED_ARCHIVE_PATHS
    for path in UPDATED_ARCHIVE_PATHS:
        row = after[path]
        assert row["source_commit"] == RETIREMENT_MAIN
        assert row["kind"] == before[path]["kind"]
        original = before[path]
        assert (
            hashlib.sha256(
                _git_bytes("show", f"{original['source_commit']}:{path}")
            ).hexdigest()
            == original["sha256"]
        )
        assert (
            _git_bytes("rev-parse", f"{original['source_commit']}:{path}")
            .decode()
            .strip()
            == original["git_blob"]
        )
    tracked = {
        entry.split("\t", 1)[1]: entry.split()[2]
        for entry in _git_bytes("ls-tree", "-r", RETIREMENT_MAIN).decode().splitlines()
    }
    for path, row in after.items():
        if path in tracked:
            assert row["git_blob"] == tracked[path], path


def test_pre_retirement_reports_are_unchanged_and_new_packets_unreviewed() -> None:
    # New reports may be added; the old reports must not be rewritten.
    report_entries = (
        _git_bytes("ls-tree", "-r", RETIREMENT_MAIN, "--", "report")
        .decode()
        .splitlines()
    )
    assert len(report_entries) == 161
    for entry in report_entries:
        meta, path = entry.split("\t", 1)
        expected = meta.split()[2]
        actual = (ROOT / path).read_bytes()
        git_blob = hashlib.sha1(
            f"blob {len(actual)}".encode() + bytes(1) + actual, usedforsecurity=False
        ).hexdigest()
        assert git_blob == expected, path
    added = (
        _git_bytes(
            "diff",
            "--name-only",
            "--diff-filter=A",
            PACKET_BASE,
            RETIREMENT_MAIN,
            "--",
            "report",
        )
        .decode()
        .splitlines()
    )
    assert len(added) == 12
    for path in added:
        record = json.loads((ROOT / path).read_bytes())
        result = evaluate_assurance(record)
        assert result.status == "UNREVIEWED" and not result.errors, path
        assert not result.economic_execution_authorized, path
