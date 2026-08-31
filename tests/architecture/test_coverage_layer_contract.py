from __future__ import annotations

import configparser
import tomllib

from tests.architecture.repository_paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT


def test_core_and_nautilus_coverage_have_separate_fail_closed_scopes() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    coverage = project["tool"]["coverage"]

    assert coverage["report"]["fail_under"] == 80
    assert coverage["run"]["branch"] is True
    assert coverage["run"]["source"] == ["trade_rl"]
    assert coverage["run"]["omit"] == ["trade_rl/integrations/nautilus/*"]

    dedicated_path = ROOT / ".coveragerc.nautilus"
    assert dedicated_path.is_file()
    dedicated = configparser.ConfigParser()
    dedicated.read(dedicated_path, encoding="utf-8")

    run = dedicated["run"]
    assert run.getboolean("branch") is True
    assert run.getboolean("parallel") is True
    assert run.getboolean("sigterm") is True
    assert run["source"].split() == ["trade_rl/integrations/nautilus"]
    assert {value.strip() for value in run["concurrency"].split(",")} == {
        "multiprocessing",
        "thread",
    }
    assert set(run["patch"].split()) == {"subprocess", "_exit"}


def test_nautilus_capability_keeps_native_probes_isolated_under_coverage() -> None:
    workflow = (ROOT / ".github" / "workflows" / "nautilus-capability.yml").read_text(
        encoding="utf-8"
    )

    coverage_prefix = "coverage run --rcfile=.coveragerc.nautilus -m pytest"
    assert workflow.count(coverage_prefix) >= 10
    assert "coverage combine --rcfile=.coveragerc.nautilus" in workflow
    assert "coverage json --rcfile=.coveragerc.nautilus" in workflow
    assert "coverage report --rcfile=.coveragerc.nautilus --fail-under=81.3" in workflow
    assert "nautilus-coverage.json" in workflow

    for path in (
        "tests/integrations/test_nautilus_dual_shadow.py",
        "tests/integrations/test_nautilus_sign_flip_conformance.py",
        "tests/integrations/test_nautilus_target_change_conformance.py",
    ):
        command = (
            "coverage run --rcfile=.coveragerc.nautilus -m pytest -q -m nautilus "
            + path
        )
        assert command in workflow
