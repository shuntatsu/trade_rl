from __future__ import annotations

from pathlib import Path

from scripts.docs._core import (
    build_manifest,
    context_for_path,
    stage_site,
    validate_repository,
)

ROOT = Path(__file__).resolve().parents[2]


def test_repository_documentation_governance_is_valid() -> None:
    assert validate_repository(ROOT) == ()


def test_manifest_separates_current_authority_from_history() -> None:
    manifest = build_manifest(ROOT)
    current = manifest["current"]
    history = manifest["history"]

    assert current
    assert history
    assert all(record["lifecycle"] in {"current", "deprecated"} for record in current)
    assert all(record["lifecycle"] == "historical" for record in history)
    assert all(record["authority"] == "none" for record in history)
    assert any(
        record["path"] == "docs/ARCHITECTURE.md"
        and record["authority"] == "canonical"
        for record in current
    )
    assert not any(
        record["path"].startswith("docs/history/")
        and record["authority"] == "canonical"
        for record in current + history
    )


def test_reward_source_routes_agent_to_reward_authority() -> None:
    records = context_for_path(ROOT, "trade_rl/rl/rewards.py")
    assert records
    assert records[0]["path"] == "docs/REWARD_OBJECTIVE.md"
    assert records[0]["authority"] == "canonical"
    assert "reward-objective" in records[0]["source_of_truth_for"]


def test_generated_site_contains_current_reference_and_history_index(
    tmp_path: Path,
) -> None:
    result = stage_site(ROOT, tmp_path)
    staged_docs = result / "site-src"
    generated_config = result / "mkdocs.generated.yml"
    generated_manifest = result / "manifest.json"

    assert generated_config.is_file()
    assert generated_manifest.is_file()
    assert (staged_docs / "index.md").is_file()
    assert (staged_docs / "reference" / "architecture.md").is_file()
    assert (staged_docs / "history" / "index.md").is_file()
    assert not (staged_docs / "history" / "raw").exists()
