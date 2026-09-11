from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from trade_rl.evaluation.robustness.perfect_information import (
    solver as _perfect_information_solver,
)
from trade_rl.evaluation.robustness.perfect_information.bound import (
    PerfectInformationBoundConfig,
    solve_perfect_information_bound,
)

ROOT = Path(__file__).resolve().parents[2]
SCIPY_REQUIREMENT = "scipy>=1.14,<1.18"


def _project_metadata() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def test_oracle_extra_declares_scipy_without_making_it_core() -> None:
    project = _project_metadata()
    dependencies = project["dependencies"]
    optional = project["optional-dependencies"]
    assert isinstance(dependencies, list)
    assert isinstance(optional, dict)
    assert SCIPY_REQUIREMENT in optional["oracle"]
    assert SCIPY_REQUIREMENT in optional["dev"]
    assert not any(
        str(requirement).lower().startswith("scipy") for requirement in dependencies
    )


def test_readme_documents_oracle_extra() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "perfect-information" in readme
    assert "--extra oracle" in readme


def test_missing_scipy_error_points_to_oracle_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = _perfect_information_solver.importlib.import_module

    def missing_scipy(name: str) -> object:
        if name.startswith("scipy"):
            raise ImportError("scipy intentionally unavailable")
        return real_import(name)

    monkeypatch.setattr(
        _perfect_information_solver.importlib,
        "import_module",
        missing_scipy,
    )
    with pytest.raises(RuntimeError, match=r"uv sync --extra oracle"):
        solve_perfect_information_bound(
            np.asarray([[0.01]], dtype=np.float64),
            PerfectInformationBoundConfig(n_assets=1),
        )
