from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_root_readme_documents_complete_candidate_run_artifact() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required_file in ("summary.json", "returns.npz", "provenance.json"):
        assert required_file in readme
