"""Bulk retirement preserves exact historical source and current shared contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

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
