from pathlib import Path


def test_start_documents_executable_universal_full_research_training_command() -> None:
    start = Path("START.md").read_text(encoding="utf-8")

    command = "uv run python scripts/run_universal_full_research.py"
    assert command in start
    for option in (
        "--selected-architecture",
        "--ppo-config",
        "--lagrangian-config",
        "--discounted-config",
        "--runtime-factory",
        "--instrument-artifact-root",
        "--postgres-url",
        "--dataset-artifact-root",
        "--fold-train-start",
        "--fold-train-stop",
        "--normalizer-digest",
        "--feature-schema-digest",
        "--baseline",
        "--fold",
        "--output-root",
    ):
        assert option in start

    assert "research_success=false" in start
    assert "sealed" in start.lower()


def test_universal_full_research_training_script_exists() -> None:
    script = Path("scripts/run_universal_full_research.py")
    assert script.is_file()
    source = script.read_text(encoding="utf-8")
    assert "run_universal_full_research_training" in source
    assert "TrainingRunConfig.from_json" in source
    assert "UniversalRuntimeFactoryContext" in source
