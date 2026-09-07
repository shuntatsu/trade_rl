from __future__ import annotations

from pathlib import Path


def test_rewrite_text_rebases_quickstart_links_after_move() -> None:
    from scripts.docs.migrate_layout_references import rewrite_text

    source = Path("docs/getting-started/quickstart.md")
    text = (
        "[config](docs/CONFIGURATION.md)\n"
        "[universal](docs/UNIVERSAL_TRAINING.md)\n"
        "[binance](docs/BINANCE.md)\n"
        "[status](docs/RESEARCH_STATUS.md)\n"
        "[gpu](docs/operations/docker-gpu-full-training.md)\n"
    )
    rewritten = rewrite_text(source, text)
    assert "../reference/configuration.md" in rewritten
    assert "../reference/universal-training.md" in rewritten
    assert "../guides/binance-data.md" in rewritten
    assert "../research/status.md" in rewritten
    assert "../operations/docker-gpu-full-training.md" in rewritten


def test_rewrite_text_rebases_reference_and_research_links() -> None:
    from scripts.docs.migrate_layout_references import rewrite_text

    universal = rewrite_text(
        Path("docs/reference/universal-training.md"),
        "[start](../START.md) [arch](ARCHITECTURE.md) "
        "[config](CONFIGURATION.md) [research](RESEARCH_STATUS.md) "
        "[reward](REWARD_OBJECTIVE.md) [gpu](operations/docker-gpu-full-training.md)",
    )
    assert "../getting-started/quickstart.md" in universal
    assert "architecture.md" in universal
    assert "configuration.md" in universal
    assert "../research/status.md" in universal
    assert "reward-objective.md" in universal
    assert "../operations/docker-gpu-full-training.md" in universal

    research = rewrite_text(
        Path("docs/research/multi-timeframe.md"),
        "[arch](ARCHITECTURE.md) [config](CONFIGURATION.md) "
        "[gpu](operations/docker-gpu-full-training.md)",
    )
    assert "../reference/architecture.md" in research
    assert "../reference/configuration.md" in research
    assert "../operations/docker-gpu-full-training.md" in research


def test_rewrite_text_rebases_example_paths_from_deeper_docs() -> None:
    from scripts.docs.migrate_layout_references import rewrite_text

    binance = rewrite_text(
        Path("docs/guides/binance-data.md"),
        "[example](../examples/binance-multitimeframe/training-full.json) "
        "[reward](REWARD_OBJECTIVE.md)",
    )
    assert "../../examples/binance-multitimeframe/training-full.json" in binance
    assert "../reference/reward-objective.md" in binance

    configuration = rewrite_text(
        Path("docs/reference/configuration.md"),
        "[example](../examples/quickstart/training.json) [single](SINGLE_SYMBOL.md)",
    )
    assert "../../examples/quickstart/training.json" in configuration
    assert "single-symbol.md" in configuration


def test_rewrite_text_updates_repo_relative_canonical_paths_outside_docs() -> None:
    from scripts.docs.migrate_layout_references import rewrite_text

    text = "docs/ARCHITECTURE.md docs/CONFIGURATION.md docs/RESEARCH_STATUS.md"
    rewritten = rewrite_text(Path("tests/test_contract.py"), text)
    assert rewritten == (
        "docs/reference/architecture.md "
        "docs/reference/configuration.md "
        "docs/research/status.md"
    )


def test_rewrite_repository_never_mutates_history(tmp_path: Path) -> None:
    from scripts.docs.migrate_layout_references import rewrite_repository

    current = tmp_path / "README.md"
    current.write_text("docs/ARCHITECTURE.md\n", encoding="utf-8")
    history = tmp_path / "docs" / "history" / "plan.md"
    history.parent.mkdir(parents=True)
    history.write_text("docs/ARCHITECTURE.md\n", encoding="utf-8")

    changed = rewrite_repository(tmp_path)

    assert Path("README.md") in changed
    assert current.read_text(encoding="utf-8") == "docs/reference/architecture.md\n"
    assert history.read_text(encoding="utf-8") == "docs/ARCHITECTURE.md\n"
