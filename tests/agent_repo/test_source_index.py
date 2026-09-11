from __future__ import annotations

from pathlib import Path

from tools.agent_repo.source_index import PathContext, SourceIndex


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _tree(root: Path) -> None:
    _write(root, "trade_rl/__init__.py", "")
    _write(root, "trade_rl/evaluation/__init__.py", "")
    _write(
        root,
        "trade_rl/evaluation/runs/__init__.py",
        "from .config import CandidateRunConfig\n__all__ = ['CandidateRunConfig']\n",
    )
    _write(
        root,
        "trade_rl/evaluation/runs/config.py",
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n\n"
        "RUN_SCHEMA = 'candidate_run_v1'\n"
        "DOCUMENTATION_URL = 'https://example.invalid/spec'\n\n"
        "@dataclass(frozen=True)\n"
        "class CandidateRunConfig:\n"
        "    value: int\n\n"
        "class _PrivateHelper:\n"
        "    pass\n\n"
        "def persist(path: Path) -> None:\n"
        "    path.write_text('x', encoding='utf-8')\n",
    )
    _write(root, "trade_rl/evaluation/experiments/__init__.py", "")
    _write(
        root,
        "trade_rl/evaluation/experiments/workflow.py",
        "from trade_rl.evaluation.runs import CandidateRunConfig\n",
    )
    _write(
        root,
        "tests/evaluation/test_candidate_run_config.py",
        "from trade_rl.evaluation.runs import CandidateRunConfig\n",
    )
    _write(
        root,
        "tests/data/test_config.py",
        "def test_unrelated_config_name():\n    assert True\n",
    )
    _write(
        root,
        "docs/architecture/package-boundaries.md",
        "Run config owner: `trade_rl.evaluation.runs.config`.\n",
    )


def test_context_reports_source_derived_capability_and_consumers(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    index = SourceIndex.build(tmp_path)

    context = index.context("trade_rl/evaluation/runs/config.py")

    assert isinstance(context, PathContext)
    assert context.path == "trade_rl/evaluation/runs/config.py"
    assert context.module == "trade_rl.evaluation.runs.config"
    assert context.domain == "evaluation"
    assert context.capability == "runs"
    assert context.facade_module == "trade_rl.evaluation.runs"
    assert context.direct_consumers == ()
    assert context.semantic_consumers == (
        "trade_rl/evaluation/experiments/workflow.py",
    )
    assert context.public_exports == ("CandidateRunConfig",)
    assert context.schema_constants == ("RUN_SCHEMA",)
    assert context.test_candidates == ("tests/evaluation/test_candidate_run_config.py",)
    assert context.doc_references == ("docs/architecture/package-boundaries.md",)
    assert context.effect_signals == ("filesystem:write_text",)


def test_context_does_not_invent_public_or_network_semantics(tmp_path: Path) -> None:
    _tree(tmp_path)

    context = SourceIndex.build(tmp_path).context("trade_rl/evaluation/runs/config.py")

    assert "_PrivateHelper" not in context.public_exports
    assert all("network:" not in value for value in context.effect_signals)
    assert "tests/data/test_config.py" not in context.test_candidates


def test_impact_is_deterministic_by_path(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = SourceIndex.build(tmp_path)

    contexts = index.impact(
        (
            "trade_rl/evaluation/experiments/workflow.py",
            "trade_rl/evaluation/runs/config.py",
        )
    )

    assert tuple(context.path for context in contexts) == (
        "trade_rl/evaluation/experiments/workflow.py",
        "trade_rl/evaluation/runs/config.py",
    )
