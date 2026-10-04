import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_transitional_domain_package_is_absent() -> None:
    assert not (PACKAGE / "domain").exists()


def test_validation_and_canonical_owners_exist() -> None:
    assert (PACKAGE / "_validation.py").is_file()
    assert (PACKAGE / "artifacts" / "canonical.py").is_file()


def test_evaluation_gate_package_exists() -> None:
    gates = PACKAGE / "evaluation" / "gates"
    assert (gates / "__init__.py").is_file()
    assert (gates / "models.py").is_file()
    assert (gates / "resolve.py").is_file()


def test_forwarding_artifact_codec_is_absent() -> None:
    assert not (PACKAGE / "artifacts" / "codec.py").exists()


def test_candidate_and_study_publishers_use_the_shared_directory_primitive() -> None:
    modules = (
        PACKAGE / "evaluation" / "runs" / "artifact.py",
        PACKAGE / "evaluation" / "experiments" / "store.py",
    )
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "atomic_rename_directory"
        ]
        assert calls, path.relative_to(ROOT)
