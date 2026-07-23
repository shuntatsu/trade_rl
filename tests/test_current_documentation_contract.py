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


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _constant(path: Path, name: str) -> str:
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', _text(path), flags=re.MULTILINE)
    assert match is not None, f"missing {name} in {path.relative_to(ROOT)}"
    return match.group(1)


def _configured_layers() -> tuple[str, ...]:
    text = _text(ROOT / ".importlinter")
    pattern = (
        r"\[importlinter:contract:layers\].*?^layers\s*=\s*\n"
        r"(?P<body>(?:    trade_rl\.[^\n]+\n)+)"
    )
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None
    return tuple(line.strip() for line in match.group("body").splitlines())


def test_maintained_documents_exist() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in MAINTAINED_DOCUMENTS
        if not path.is_file()
    ]
    assert missing == []


def test_current_schema_contracts_are_documented() -> None:
    observation_schema = _constant(
        ROOT / "trade_rl" / "rl" / "observations.py", "OBSERVATION_SCHEMA"
    )
    bundle_schema = _constant(
        ROOT / "trade_rl" / "serving" / "bundle.py", "SERVING_BUNDLE_SCHEMA"
    )
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    readme = _text(ROOT / "README.md")
    readme_ja = _text(ROOT / "README.ja.md")
    for document in (architecture, readme, readme_ja):
        assert observation_schema in document
        assert bundle_schema in document
    assert (
        "pending-order" in architecture.lower()
        or "pending order" in architecture.lower()
    )
    assert "observation schema v3" not in readme.lower()
    assert "observation schema v3" not in readme_ja.lower()
    assert "observation schema v3" not in architecture.lower()


def test_architecture_layer_order_matches_import_linter() -> None:
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    configured = tuple(
        layer.removeprefix("trade_rl.") for layer in _configured_layers()
    )
    assert "telemetry" in configured
    marker = "The enforced Import Linter layer order is exactly:"
    documented_section = architecture.split(marker, maxsplit=1)[1]
    documented_block = documented_section.split("```", maxsplit=2)[1]
    documented = tuple(
        line.strip().removeprefix("-> ")
        for line in documented_block.splitlines()
        if line.strip() and line.strip() != "text"
    )
    assert documented == configured
    telemetry_start = architecture.index("`telemetry` is explicitly placed")
    telemetry_context = architecture[telemetry_start : telemetry_start + 360].lower()
    for stale in (
        "outside the enforced",
        "not listed in the enforced",
        "not currently governed",
        "missing enforcement",
    ):
        assert stale not in telemetry_context


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
    combined = "\n".join(
        _text(path)
        for path in (
            ROOT / "README.md",
            ROOT / "README.ja.md",
            ROOT / "docs" / "ARCHITECTURE.md",
        )
    ).lower()
    for phrase in ("metadata catalog", "filesystem artifact"):
        assert phrase in combined
    forbidden_phrases = (
        "model blob",
        "checkpoint blob",
        "postgresql is the numerical source",
    )
    for forbidden in forbidden_phrases:
        assert forbidden not in combined


def test_live_training_boundary_is_explicit() -> None:
    studio = _text(ROOT / "studio" / "README.md")
    readme = _text(ROOT / "README.md")
    assert "not exchange activity" in readme.lower()
    assert "not model-selection evidence" in readme.lower()
    assert "not sealed evaluation" in readme.lower()
    assert "not profitability evidence" in readme.lower()
    assert "取引所注文ではありません" in studio
    assert "モデル選択" in studio
    assert "Sealed" in studio
    assert "収益性" in studio
    assert "NO-GO" in studio


def test_remediated_findings_are_not_described_as_current() -> None:
    current_documents = "\n".join(
        _text(path)
        for path in (
            ROOT / "README.md",
            ROOT / "README.ja.md",
            ROOT / "docs" / "ARCHITECTURE.md",
            ROOT / "docs" / "RESEARCH_STATUS.md",
            ROOT / "studio" / "README.md",
        )
    ).lower()
    for stale in (
        "telemetry is not yet placed",
        "telemetry` is not yet placed",
        "outside the enforced layer stack",
        "scan the jsonl file from the beginning",
        "coerced with python truthiness",
        "discovery order instead of being rejected",
        "execute_interval` remains a separate compatibility",
        "baseline reward pre-roll currently uses the compatibility execution path",
    ):
        assert stale not in current_documents


def test_resolved_live_training_isolation_is_not_described_as_open() -> None:
    research_status = _text(ROOT / "docs" / "RESEARCH_STATUS.md").lower()
    for stale in (
        "available_for_diagnostic_replay_with_stream_isolation_gap",
        "one confirmed diagnostic limitation remains",
        "without an environment or episode selector",
    ):
        assert stale not in research_status
    for current in (
        "available_for_diagnostic_replay_with_episode_isolation",
        "producer-issued `episode_id`",
        "selected vector environment",
        "current episode",
        "historical records with `null` identity",
    ):
        assert current in research_status


def test_environment_facade_audit_is_resolved_with_protected_state_boundary() -> None:
    closeout = _text(
        ROOT / "docs" / "verification" / "2026-07-23-architecture-audit-closeout.md"
    )
    summary = closeout.split("## 2. Closeout summary", maxsplit=1)[1].split(
        "## 3. Finding-by-finding disposition", maxsplit=1
    )[0]
    assert "| `AUD-RL-001` | P2 risk | RESOLVED |" in summary

    section = closeout.split("### AUD-RL-001 — RESOLVED", maxsplit=1)[1].split(
        "## 4. Current architecture judgment", maxsplit=1
    )[0]
    for required in (
        "150-line architecture limit",
        "typed",
        "`step()` remains an orchestration facade",
        "`reset()` retains mutable Gymnasium state",
        "architecture tests",
        "100.0% critical coverage ratchets",
        "no further mechanical split",
    ):
        assert required in section
    assert "OPEN RISK" not in section
    assert "Production remains `NO-GO`" in closeout


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
