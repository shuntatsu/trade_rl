from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_training_run_config_contract_lives_below_workflows() -> None:
    contract = ROOT / "trade_rl/rl/training_run_config.py"
    workflow = ROOT / "trade_rl/workflows/training_run.py"
    studio = ROOT / "trade_rl/studio/config_catalog.py"

    assert contract.is_file()
    contract_source = contract.read_text(encoding="utf-8")
    workflow_source = workflow.read_text(encoding="utf-8")
    studio_source = studio.read_text(encoding="utf-8")

    assert "class TrainingRunConfig" in contract_source
    assert "from trade_rl.rl.training_run_config import" in workflow_source
    assert (
        "from trade_rl.rl.training_run_config import TrainingRunConfig" in studio_source
    )
    assert "trade_rl.workflows.training_run" not in studio_source


def test_generic_config_field_validation_lives_in_domain() -> None:
    domain_helper = ROOT / "trade_rl/domain/config_fields.py"
    compatibility = ROOT / "trade_rl/workflows/config_fields.py"

    assert domain_helper.is_file()
    domain_source = domain_helper.read_text(encoding="utf-8")
    compatibility_source = compatibility.read_text(encoding="utf-8")

    assert "def require_exact_fields(" in domain_source
    assert "def require_dataclass_fields(" in domain_source
    assert "from trade_rl.domain.config_fields import" in compatibility_source


def test_structured_policy_contract_is_neutral_and_serving_owned() -> None:
    contract = ROOT / "trade_rl/artifacts/structured_policy_contract.py"
    exporter = ROOT / "trade_rl/rl/structured_export.py"
    serving_paths = (
        ROOT / "trade_rl/serving/policy_loader.py",
        ROOT / "trade_rl/serving/structured_policy.py",
    )

    assert contract.is_file()
    contract_source = contract.read_text(encoding="utf-8")
    exporter_source = exporter.read_text(encoding="utf-8")

    assert "class StructuredInputSpec" in contract_source
    assert "class StructuredExportManifest" in contract_source
    assert "import torch" not in contract_source
    assert "from torch" not in contract_source
    assert "trade_rl.rl" not in contract_source
    assert (
        "from trade_rl.artifacts.structured_policy_contract import" in exporter_source
    )

    for path in serving_paths:
        source = path.read_text(encoding="utf-8")
        assert "from trade_rl.artifacts.structured_policy_contract import" in source
        assert "trade_rl.rl.structured_export" not in source


def test_future_stage_b_market_roles_are_explicit_but_not_claimed_complete() -> None:
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    status = (ROOT / "docs/RESEARCH_STATUS.md").read_text(encoding="utf-8")

    for text in (architecture, status):
        assert "SpotLongBook: FUTURE_LONG_ONLY_ROLE" in text
        assert "USDSMShortBook: FUTURE_SHORT_ONLY_ROLE" in text
        assert "StageBSpotFuturesGeneralization: NOT_IMPLEMENTED" in text
