from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase4b_one_shot_workflows_are_absent() -> None:
    for relative_path in (
        ".github/workflows/phase4b-format-red-contract.yml",
        ".github/workflows/phase4b-documentation-falsification.yml",
    ):
        assert not (ROOT / relative_path).exists(), relative_path
