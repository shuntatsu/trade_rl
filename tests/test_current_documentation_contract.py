from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAINTAINED_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "README.ja.md",
    ROOT / "START.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "RESEARCH_STATUS.md",
    ROOT / "docs" / "BINANCE.md",
    ROOT / "docs" / "operations" / "docker-gpu-full-training.md",
    ROOT / "studio" / "README.md",
)

EXPECTED_LAYERS = (
    "trade_rl.cli",
    "trade_rl.studio",
    "trade_rl.workflows",
    "trade_rl.integrations",
    "trade_rl.serving",
    "trade_rl.learning",
    "trade_rl.rl",
    "trade_rl.risk",
    "trade_rl.simulation",
    "trade_rl.strategies",
    "trade_rl.data",
    "trade_rl.catalog",
    "trade_rl.evaluation",
    "trade_rl.release",
    "trade_rl.artifacts",
    "trade_rl.domain",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_maintained_documents_exist() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in MAINTAINED_DOCUMENTS
        if not path.is_file()
    ]
    assert missing == []


def test_current_schema_contracts_are_documented() -> None:
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    readme_ja = _text(ROOT / "README.ja.md")
    assert "baseline_residual_observation_v5" in architecture
    assert "pending-order" in architecture.lower() or "pending order" in architecture.lower()
    assert "bundle v5" in architecture.lower()
    assert "bundle v5" in readme_ja.lower()
    assert "bundle v4" not in readme_ja.lower()


def test_architecture_layer_order_matches_import_linter() -> None:
    import_linter = _text(ROOT / ".importlinter")
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    configured = tuple(
        line.strip()
        for line in import_linter.splitlines()
        if line.startswith("    trade_rl.")
    )[: len(EXPECTED_LAYERS)]
    assert configured == EXPECTED_LAYERS
    positions = tuple(architecture.index(layer) for layer in EXPECTED_LAYERS)
    assert positions == tuple(sorted(positions))
    assert "trade_rl.telemetry" in architecture
    telemetry_context = architecture[
        architecture.index("trade_rl.telemetry") : architecture.index("trade_rl.telemetry")
        + 240
    ].lower()
    assert "not" in telemetry_context or "not listed" in telemetry_context


def test_obsolete_universal_capacity_statement_is_absent() -> None:
    obsolete_patterns = (
        r"last completed bar(?:'s)? volume as its capacity proxy",
        r"previous completed bar(?:'s)? volume as (?:the|its) capacity proxy",
        r"前バー.*volume.*capacity proxy",
    )
    for path in MAINTAINED_DOCUMENTS:
        text = _text(path)
        for pattern in obsolete_patterns:
            assert re.search(pattern, text, flags=re.IGNORECASE) is None, path


def test_postgres_is_not_described_as_payload_storage() -> None:
    readme_ja = _text(ROOT / "README.ja.md")
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    combined = f"{readme_ja}\n{architecture}".lower()
    for phrase in ("metadata catalog", "filesystem artifact"):
        assert phrase in combined
    assert "model blob" not in combined
    assert "checkpoint blob" not in combined


def test_live_training_boundary_is_explicit() -> None:
    studio = _text(ROOT / "studio" / "README.md")
    readme = _text(ROOT / "README.md")
    combined = f"{studio}\n{readme}".lower()
    for phrase in (
        "not exchange",
        "not selection",
        "not sealed",
        "no-go",
    ):
        assert phrase in combined


def test_internal_markdown_links_resolve() -> None:
    link_pattern = re.compile(
        r"\[[^\]]+\]\((?!https?://|#|mailto:)([^)#]+)(?:#[^)]+)?\)"
    )
    broken: list[str] = []
    for document in MAINTAINED_DOCUMENTS:
        text = _text(document)
        for target in link_pattern.findall(text):
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert broken == []
