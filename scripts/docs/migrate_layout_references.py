from __future__ import annotations

import argparse
from pathlib import Path

_REPO_RELATIVE_REPLACEMENTS = (
    ("docs/README.md", "docs/index.md"),
    ("docs/ARCHITECTURE.md", "docs/reference/architecture.md"),
    ("docs/BINANCE.md", "docs/guides/binance-data.md"),
    ("docs/CONFIGURATION.md", "docs/reference/configuration.md"),
    ("docs/EXECUTION_ROBUSTNESS.md", "docs/reference/execution-robustness.md"),
    ("docs/LICENSING.md", "docs/legal/licensing.md"),
    ("docs/LICENSING_PROVENANCE.md", "docs/legal/licensing-provenance.md"),
    ("docs/MULTITIMEFRAME_RESEARCH.md", "docs/research/multi-timeframe.md"),
    ("docs/NAUTILUS_MIGRATION.md", "docs/reference/nautilus-migration.md"),
    ("docs/RESEARCH_STATUS.md", "docs/research/status.md"),
    ("docs/REWARD_OBJECTIVE.md", "docs/reference/reward-objective.md"),
    ("docs/RUN_REPORTING.md", "docs/reference/run-reporting.md"),
    ("docs/SINGLE_SYMBOL.md", "docs/reference/single-symbol.md"),
    ("docs/UNIVERSAL_TRADE_RL.md", "docs/reference/universal-trade-rl.md"),
    ("docs/UNIVERSAL_TRAINING.md", "docs/reference/universal-training.md"),
)

_CURRENT_DOC_REPLACEMENTS: dict[Path, tuple[tuple[str, str], ...]] = {
    Path("docs/getting-started/quickstart.md"): (
        ("docs/BINANCE.md", "../guides/binance-data.md"),
        ("docs/CONFIGURATION.md", "../reference/configuration.md"),
        ("docs/RESEARCH_STATUS.md", "../research/status.md"),
        ("docs/UNIVERSAL_TRAINING.md", "../reference/universal-training.md"),
        (
            "docs/operations/docker-gpu-full-training.md",
            "../operations/docker-gpu-full-training.md",
        ),
    ),
    Path("docs/guides/binance-data.md"): (
        ("../examples/", "../../examples/"),
        ("REWARD_OBJECTIVE.md", "../reference/reward-objective.md"),
    ),
    Path("docs/reference/architecture.md"): (
        ("SINGLE_SYMBOL.md", "single-symbol.md"),
    ),
    Path("docs/reference/configuration.md"): (
        ("../examples/", "../../examples/"),
        ("SINGLE_SYMBOL.md", "single-symbol.md"),
    ),
    Path("docs/reference/universal-training.md"): (
        ("../START.md", "../getting-started/quickstart.md"),
        ("ARCHITECTURE.md", "architecture.md"),
        ("CONFIGURATION.md", "configuration.md"),
        ("MULTITIMEFRAME_RESEARCH.md", "../research/multi-timeframe.md"),
        ("RESEARCH_STATUS.md", "../research/status.md"),
        ("REWARD_OBJECTIVE.md", "reward-objective.md"),
        (
            "operations/docker-gpu-full-training.md",
            "../operations/docker-gpu-full-training.md",
        ),
    ),
    Path("docs/research/multi-timeframe.md"): (
        ("ARCHITECTURE.md", "../reference/architecture.md"),
        ("CONFIGURATION.md", "../reference/configuration.md"),
        (
            "operations/docker-gpu-full-training.md",
            "../operations/docker-gpu-full-training.md",
        ),
    ),
    Path("docs/research/status.md"): (
        ("UNIVERSAL_TRAINING.md", "../reference/universal-training.md"),
    ),
}

_GENERIC_ROOTS = (
    Path("tests"),
    Path(".github"),
    Path("scripts"),
    Path("frontend"),
    Path("examples"),
)
_GENERIC_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".txt"}


def _replace_all(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def rewrite_text(relative_path: Path, text: str) -> str:
    """Rewrite only references whose authority/path changed in this migration."""

    relative_path = Path(relative_path.as_posix())
    if relative_path.parts[:2] == ("docs", "history"):
        return text

    if relative_path.parts and relative_path.parts[0] == "docs":
        replacements = _CURRENT_DOC_REPLACEMENTS.get(relative_path, ())
        return _replace_all(text, replacements)

    return _replace_all(text, _REPO_RELATIVE_REPLACEMENTS)


def _candidate_files(root: Path) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    for filename in ("README.md", "START.md"):
        path = root / filename
        if path.is_file():
            candidates.add(path)

    for relative in _CURRENT_DOC_REPLACEMENTS:
        path = root / relative
        if path.is_file():
            candidates.add(path)

    for relative_root in _GENERIC_ROOTS:
        base = root / relative_root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in _GENERIC_SUFFIXES:
                candidates.add(path)

    return tuple(sorted(candidates, key=lambda path: path.relative_to(root).as_posix()))


def rewrite_repository(root: Path) -> tuple[Path, ...]:
    """Apply the approved path migration without touching raw history or runtime code."""

    root = root.resolve()
    changed: list[Path] = []
    for path in _candidate_files(root):
        relative = path.relative_to(root)
        if relative.parts[:2] == ("docs", "history"):
            continue
        original = path.read_text(encoding="utf-8")
        rewritten = rewrite_text(relative, original)
        if rewritten == original:
            continue
        path.write_text(rewritten, encoding="utf-8")
        changed.append(relative)
    return tuple(changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the governed docs path migration")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    changed = rewrite_repository(Path(args.root))
    for path in changed:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
