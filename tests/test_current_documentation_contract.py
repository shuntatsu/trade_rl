from __future__ import annotations

import re
from pathlib import Path

from tests.architecture.import_linter_config import (
    configured_layers as import_linter_layers,
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE_ROOT = ROOT / "trade_rl"
FRONTEND_ROOT = ROOT / "frontend"
DOCS_ROOT = ROOT / "docs"

ARCHITECTURE_DOCUMENT = DOCS_ROOT / "reference" / "architecture.md"
CONFIGURATION_DOCUMENT = DOCS_ROOT / "reference" / "configuration.md"
SINGLE_SYMBOL_DOCUMENT = DOCS_ROOT / "reference" / "single-symbol.md"
UNIVERSAL_TRAINING_DOCUMENT = DOCS_ROOT / "reference" / "universal-training.md"
RESEARCH_STATUS_DOCUMENT = DOCS_ROOT / "research" / "status.md"

CURRENT_OPERATION_RUNBOOKS = (
    DOCS_ROOT / "operations" / "causal-scenario-c3-execution.md",
    DOCS_ROOT / "operations" / "docker-gpu-full-training.md",
)
OPERATIONS_INDEX = DOCS_ROOT / "operations" / "index.md"

MAINTAINED_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "START.md",
    DOCS_ROOT / "index.md",
    SINGLE_SYMBOL_DOCUMENT,
    UNIVERSAL_TRAINING_DOCUMENT,
    ARCHITECTURE_DOCUMENT,
    CONFIGURATION_DOCUMENT,
    RESEARCH_STATUS_DOCUMENT,
    DOCS_ROOT / "reference" / "reward-objective.md",
    DOCS_ROOT / "reference" / "execution-robustness.md",
    DOCS_ROOT / "research" / "multi-timeframe.md",
    DOCS_ROOT / "guides" / "binance-data.md",
    DOCS_ROOT / "reference" / "nautilus-migration.md",
    DOCS_ROOT / "legal" / "licensing.md",
    DOCS_ROOT / "legal" / "licensing-provenance.md",
    *CURRENT_OPERATION_RUNBOOKS,
    DOCS_ROOT / "performance" / "4070ti-super-full-training.md",
    FRONTEND_ROOT / "README.md",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _constant(path: Path, name: str) -> str:
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', _text(path), flags=re.MULTILINE)
    assert match is not None, f"missing {name} in {path.relative_to(ROOT)}"
    return match.group(1)


def _configured_layers() -> tuple[str, ...]:
    return import_linter_layers()


def _all_current_markdown() -> tuple[Path, ...]:
    current_docs = {
        path
        for path in DOCS_ROOT.rglob("*.md")
        if "history" not in path.relative_to(DOCS_ROOT).parts
    }
    paths = {
        ROOT / "README.md",
        ROOT / "START.md",
        FRONTEND_ROOT / "README.md",
        *current_docs,
    }
    return tuple(sorted((path for path in paths if path.is_file()), key=str))


def test_maintained_documents_exist() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in MAINTAINED_DOCUMENTS
        if not path.is_file()
    ]
    assert missing == []


def test_operations_directory_contains_index_and_only_current_runbooks() -> None:
    actual = tuple(sorted((DOCS_ROOT / "operations").glob("*.md"), key=str))
    expected = (*CURRENT_OPERATION_RUNBOOKS, OPERATIONS_INDEX)
    assert actual == tuple(sorted(expected, key=str))


def test_current_schema_contracts_are_documented() -> None:
    observation_schema = _constant(
        PYTHON_SOURCE_ROOT / "rl" / "observations.py", "OBSERVATION_SCHEMA"
    )
    bundle_schema = _constant(
        PYTHON_SOURCE_ROOT / "serving" / "bundle.py", "SERVING_BUNDLE_SCHEMA"
    )
    architecture = _text(ARCHITECTURE_DOCUMENT)
    configuration = _text(CONFIGURATION_DOCUMENT)
    single_symbol = _text(SINGLE_SYMBOL_DOCUMENT)
    for value in (observation_schema, bundle_schema):
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
        "sb3_policy_identity_v4",
        "structured_policy_export_v2",
        "CanonicalStructuredPolicyLoader",
    ):
        assert value in configuration
    assert "Gated Cross-Timeframe Attention" in architecture
    assert "Gated Cross-Asset Attention" in architecture
    assert "single_symbol_bypass_v1" in architecture
    assert "single_symbol_bypass_v1" in configuration
    assert "one run" in single_symbol.lower()
    assert "target_weight:BTCUSDT" in single_symbol
    assert "NO-GO" in single_symbol


def test_readme_does_not_duplicate_reference_internals() -> None:
    readme = _text(ROOT / "README.md")
    for reference_only_term in (
        "Gated Cross-Timeframe Attention",
        "CanonicalStructuredPolicyLoader",
        "sb3_policy_identity_v4",
        "structured_policy_export_v2",
    ):
        assert reference_only_term not in readme


def test_current_documentation_targets_exist() -> None:
    required = (
        ROOT / "docker" / "Dockerfile.training",
        ROOT / "docker" / "compose.training.yaml",
        ROOT / "docker" / "compose.universal-training.yaml",
        ROOT / "LICENSE",
        ROOT / "LICENSES" / "THIRD_PARTY_NOTICES.md",
    )
    missing = [
        path.relative_to(ROOT).as_posix() for path in required if not path.is_file()
    ]
    assert missing == []


def test_readme_uses_current_third_party_notice_path() -> None:
    readme = _text(ROOT / "README.md")
    assert "LICENSES/THIRD_PARTY_NOTICES.md" in readme
    assert "`THIRD_PARTY_NOTICES.md`" not in readme


def test_maintained_reference_docs_do_not_depend_on_transient_pr_numbers() -> None:
    transient = re.compile(r"PR\s*#\d+")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in MAINTAINED_DOCUMENTS
        if transient.search(_text(path))
    ]
    assert offenders == []


def test_universal_training_documents_checkpoint_generator_identity() -> None:
    universal = _text(UNIVERSAL_TRAINING_DOCUMENT)
    for phrase in (
        "generator_code_digest",
        "grid_digest",
        "causal_alpha_selection_checkpoint_metric_v2",
        "Fail closed",
    ):
        assert phrase.lower() in universal.lower()


def test_operator_runbooks_use_current_training_schema() -> None:
    for path in (
        ROOT / "START.md",
        DOCS_ROOT / "operations" / "docker-gpu-full-training.md",
    ):
        text = _text(path)
        assert "training_run_config_v4" in text
        for legacy in (
            "training_run_config_v1",
            "training_run_config_v2",
            "training_run_config_v3",
        ):
            assert legacy not in text


def test_research_status_has_timeless_heading_and_explicit_stage_boundaries() -> None:
    research_status = _text(RESEARCH_STATUS_DOCUMENT)
    assert "## Current status\n" in research_status
    assert "## Current status —" not in research_status
    for boundary in (
        "StageAZeroShotSoftware: IMPLEMENTED_AND_CI_VERIFIED",
        "StageAEmpiricalEvaluation: NOT_COMPLETED",
        "StageBSpotFuturesGeneralization: NOT_IMPLEMENTED",
    ):
        assert boundary in research_status


def test_legacy_settings_are_only_documented_as_rejected_inputs() -> None:
    configuration = _text(CONFIGURATION_DOCUMENT)
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
        if path == CONFIGURATION_DOCUMENT:
            continue
        text = _text(path)
        assert "sequence_encoder" not in text
        assert "asset_set_encoder" not in text
        assert "sequence_attention_heads" not in text
        assert "sequence_attention_layers" not in text


def test_architecture_layer_order_matches_import_linter() -> None:
    architecture = _text(ARCHITECTURE_DOCUMENT)
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
    frontend = _text(FRONTEND_ROOT / "README.md")
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
        assert phrase in frontend


def test_postgres_is_described_as_metadata_catalog() -> None:
    combined = "\n".join(
        _text(path)
        for path in (
            ROOT / "README.md",
            ARCHITECTURE_DOCUMENT,
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
    assert "[ドキュメント一覧](docs/index.md)" in readme


def test_current_internal_markdown_links_resolve() -> None:
    link_pattern = re.compile(
        r"\[[^\]]+\]\((?!https?://|#|mailto:)([^)#]+)(?:#[^)]+)?\)"
    )
    broken: list[str] = []
    for document in _all_current_markdown():
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
