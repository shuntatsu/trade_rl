from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAINTAINED_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "START.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "CONFIGURATION.md",
    ROOT / "docs" / "RESEARCH_STATUS.md",
    ROOT / "docs" / "MULTITIMEFRAME_RESEARCH.md",
    ROOT / "docs" / "BINANCE.md",
    ROOT / "docs" / "operations" / "docker-gpu-full-training.md",
    ROOT / "docs" / "operations" / "causal-scenario-c3-execution.md",
    ROOT / "docs" / "performance" / "4070ti-super-full-training.md",
    ROOT / "studio" / "README.md",
)

REMOVED_HISTORY_PATHS = (
    ROOT / "README.ja.md",
    ROOT / ".superpowers" / "sdd" / "task-8-report.md",
    ROOT / "docs" / "audits",
    ROOT / "docs" / "reviews",
    ROOT / "docs" / "plans",
    ROOT / "docs" / "verification",
    ROOT / "docs" / "superpowers",
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


def _all_markdown() -> tuple[Path, ...]:
    paths = {
        ROOT / "README.md",
        ROOT / "START.md",
        ROOT / "studio" / "README.md",
        *ROOT.joinpath("docs").rglob("*.md"),
    }
    return tuple(sorted((path for path in paths if path.is_file()), key=str))


def test_maintained_documents_exist() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in MAINTAINED_DOCUMENTS
        if not path.is_file()
    ]
    assert missing == []


def test_historical_documentation_clutter_is_removed() -> None:
    remaining = [
        path.relative_to(ROOT).as_posix()
        for path in REMOVED_HISTORY_PATHS
        if path.exists()
    ]
    assert remaining == []


def test_current_schema_contracts_are_documented() -> None:
    observation_schema = _constant(
        ROOT / "trade_rl" / "rl" / "observations.py", "OBSERVATION_SCHEMA"
    )
    bundle_schema = _constant(
        ROOT / "trade_rl" / "serving" / "bundle.py", "SERVING_BUNDLE_SCHEMA"
    )
    readme = _text(ROOT / "README.md")
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    configuration = _text(ROOT / "docs" / "CONFIGURATION.md")
    for value in (observation_schema, bundle_schema):
        assert value in readme
        assert value in architecture
    for value in (
        "training_run_config_v4",
        "flat_mlp",
        "asset_set",
        "hierarchical_sequence_v2",
        "sequence_timeframe_attention_heads",
        "sequence_timeframe_attention_layers",
        "sequence_asset_attention_heads",
        "sequence_asset_attention_layers",
        "sb3_policy_identity_v1",
        "structured_policy_export_v2",
        "CanonicalStructuredPolicyLoader",
    ):
        assert value in configuration
    assert "Gated Cross-Timeframe Attention" in architecture
    assert "Gated Cross-Asset Attention" in architecture


def test_operator_runbooks_use_current_training_schema() -> None:
    for path in (
        ROOT / "START.md",
        ROOT / "docs" / "operations" / "docker-gpu-full-training.md",
    ):
        text = _text(path)
        assert "training_run_config_v3" in text
        assert "training_run_config_v2" not in text


def test_research_status_has_timeless_heading_and_explicit_stage_boundaries() -> None:
    research_status = _text(ROOT / "docs" / "RESEARCH_STATUS.md")
    assert "## Current status\n" in research_status
    assert "## Current status —" not in research_status
    for boundary in (
        "StageAZeroShotSoftware: IMPLEMENTED_AND_CI_VERIFIED",
        "StageAEmpiricalEvaluation: NOT_COMPLETED",
        "StageBSpotFuturesGeneralization: NOT_IMPLEMENTED",
    ):
        assert boundary in research_status


def test_legacy_settings_are_only_documented_as_rejected_inputs() -> None:
    configuration = _text(ROOT / "docs" / "CONFIGURATION.md")
    for legacy in (
        "training_run_config_v1",
        "sequence_encoder",
        "asset_set_encoder",
        "sequence_attention_heads",
        "sequence_attention_layers",
    ):
        assert legacy in configuration
    assert "自動変換しません" in configuration
    for path in MAINTAINED_DOCUMENTS:
        if path.name == "CONFIGURATION.md":
            continue
        text = _text(path)
        assert "sequence_encoder" not in text
        assert "asset_set_encoder" not in text
        assert "sequence_attention_heads" not in text
        assert "sequence_attention_layers" not in text


def test_architecture_layer_order_matches_import_linter() -> None:
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    configured = tuple(
        layer.removeprefix("trade_rl.") for layer in _configured_layers()
    )
    marker = "Import Linterの強制順序は次のとおりです:"
    documented_section = architecture.split(marker, maxsplit=1)[1]
    documented_block = documented_section.split("```", maxsplit=2)[1]
    documented = tuple(
        line.strip()
        for line in documented_block.splitlines()
        if line.strip() and line.strip() != "text"
    )
    assert documented == configured


def test_live_training_boundary_is_explicit() -> None:
    readme = _text(ROOT / "README.md").lower()
    studio = _text(ROOT / "studio" / "README.md")
    for phrase in (
        "not exchange activity",
        "not model-selection evidence",
        "not sealed evaluation",
        "not profitability evidence",
    ):
        assert phrase in readme
    for phrase in (
        "取引所注文ではありません",
        "モデル選択",
        "Sealed",
        "収益性",
        "NO-GO",
    ):
        assert phrase in studio


def test_postgres_is_described_as_metadata_catalog() -> None:
    combined = "\n".join(
        _text(path)
        for path in (
            ROOT / "README.md",
            ROOT / "docs" / "ARCHITECTURE.md",
        )
    ).lower()
    assert "metadata catalog" in combined
    assert "filesystem artifact" in combined
    for forbidden in (
        "model blob",
        "checkpoint blob",
        "postgresql is the numerical source",
    ):
        assert forbidden not in combined


def test_readme_is_a_bounded_entry_point() -> None:
    readme = _text(ROOT / "README.md")
    lines = readme.splitlines()
    level_two_headings = [line for line in lines if line.startswith("## ")]
    assert len(lines) <= 230
    assert len(level_two_headings) <= 12
    assert "[START.md](START.md)" in readme
    assert "[ドキュメント一覧](docs/README.md)" in readme


def test_internal_markdown_links_resolve() -> None:
    link_pattern = re.compile(
        r"\[[^\]]+\]\((?!https?://|#|mailto:)([^)#]+)(?:#[^)]+)?\)"
    )
    broken: list[str] = []
    for document in _all_markdown():
        text = _text(document)
        for target in link_pattern.findall(text):
            resolved = (document.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                broken.append(f"{document.relative_to(ROOT)} -> {target} (escape)")
                continue
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert broken == []
