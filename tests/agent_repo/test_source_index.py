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
        "tests/architecture/test_runs_boundary.py",
        'RUNS_CAPABILITY = "trade_rl.evaluation.runs"\n'
        'RUNS_CONFIG = "trade_rl/evaluation/runs/config.py"\n',
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
    assert context.architecture_tests == ("tests/architecture/test_runs_boundary.py",)
    assert context.project_references == ()
    assert context.doc_references == ("docs/architecture/package-boundaries.md",)
    assert context.effect_signals == ("filesystem:write_text",)


def test_context_does_not_invent_public_or_network_semantics(tmp_path: Path) -> None:
    _tree(tmp_path)

    context = SourceIndex.build(tmp_path).context("trade_rl/evaluation/runs/config.py")

    assert "_PrivateHelper" not in context.public_exports
    assert all("network:" not in value for value in context.effect_signals)
    assert "tests/data/test_config.py" not in context.test_candidates


def test_context_does_not_claim_facade_when_nearest_all_is_mutated(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    _write(
        tmp_path,
        "trade_rl/evaluation/__init__.py",
        "ParentSurface = object()\n__all__ = ['ParentSurface']\n",
    )
    _write(
        tmp_path,
        "trade_rl/evaluation/runs/__init__.py",
        "from .config import CandidateRunConfig\n"
        "__all__ = ['CandidateRunConfig']\n"
        "__all__.append('ShadowExport')\n",
    )

    context = SourceIndex.build(tmp_path).context("trade_rl/evaluation/runs/config.py")

    assert context.facade_module is None
    assert context.public_exports == ()


def test_context_reports_optional_capability_extra_from_dynamic_import(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "trade_rl/__init__.py", "")
    _write(tmp_path, "trade_rl/strategies/__init__.py", "")
    _write(tmp_path, "trade_rl/strategies/forecasts/__init__.py", "")
    _write(
        tmp_path,
        "trade_rl/strategies/forecasts/lightgbm.py",
        "import importlib\n\n"
        "def load_backend():\n"
        "    return importlib.import_module('lightgbm')\n",
    )
    _write(
        tmp_path,
        "pyproject.toml",
        "[project]\n"
        "name = 'example'\n\n"
        "[project.optional-dependencies]\n"
        "forecast-gbm = ['lightgbm==4.7.0']\n"
        "dev = ['lightgbm==4.7.0', 'pytest>=8']\n",
    )

    context = SourceIndex.build(tmp_path).context(
        "trade_rl/strategies/forecasts/lightgbm.py"
    )

    assert context.project_references == ("project.optional-dependencies.forecast-gbm",)


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
