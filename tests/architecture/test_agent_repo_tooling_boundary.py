from __future__ import annotations

from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_production_cannot_depend_on_repository_agent_tooling(tmp_path: Path) -> None:
    package = tmp_path / "trade_rl"
    _write(tmp_path, "trade_rl/__init__.py", "")
    offender = _write(
        tmp_path,
        "trade_rl/example.py",
        "from tools.agent_repo import source_index\n",
    )
    assert any(
        within_module(name, "tools")
        for name in ImportCollector(package).collect(offender)
    )

    collector = ImportCollector(PACKAGE)
    real_offenders = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(PACKAGE.rglob("*.py"))
        if any(within_module(name, "tools") for name in collector.collect(path))
    ]
    assert real_offenders == []


def test_permanent_ci_lints_formats_and_types_repository_tooling() -> None:
    workflow = CI.read_text(encoding="utf-8")

    assert "uv run ruff check trade_rl tests tools" in workflow
    assert "uv run ruff format --check trade_rl tests tools" in workflow
    assert "uv run mypy tools/agent_repo tests/architecture/distribution.py" in workflow
