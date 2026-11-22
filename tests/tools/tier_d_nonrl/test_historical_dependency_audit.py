"""Synthetic static source-closure falsifiers for #907/#888, no economic data."""

from __future__ import annotations

import json

from tools.tier_d_nonrl import historical_dependency_audit as audit
from tools.tier_d_nonrl.historical_dependency_audit import analyze_sources


def test_missing_transitive_modules_and_supervisor_api_drift_are_reported() -> None:
    historical = {
        "tools/tier_d_nonrl/__init__.py": "",
        "tools/tier_d_nonrl/tier_d_runner.py": (
            "from .resume import reload_plan\n"
            "from .tier_d_supervisor import run_supervised\n"
        ),
        "tools/tier_d_nonrl/resume.py": (
            "from trade_rl.evaluation.allocation_terminal_execution import replay\n"
        ),
        "tools/tier_d_nonrl/tier_d_supervisor.py": (
            "from .tier_d_process_tree import ProcessTree, terminate_owned_tree\n"
        ),
        "tools/tier_d_nonrl/tier_d_process_tree.py": (
            "class ProcessTree: pass\ndef terminate_owned_tree(): pass\n"
        ),
        "trade_rl/evaluation/allocation_terminal_execution.py": (
            "from trade_rl.evaluation.allocation_market_inputs import load\n"
        ),
        "trade_rl/evaluation/allocation_market_inputs.py": "def load(): pass\n",
    }
    maintained = {
        "tools/tier_d_nonrl/__init__.py": "",
        "tools/tier_d_nonrl/tier_d_process_tree.py": (
            "class ProcessIdentity: pass\ndef capture_process_snapshot(): pass\n"
        ),
    }
    report = analyze_sources(
        historical,
        maintained,
        seeds=("tools.tier_d_nonrl.tier_d_runner",),
    )
    assert report["analysis_complete"] is True
    assert report["recovery_ready"] is False
    assert report["reachable_module_count"] == 6
    assert report["missing_module_count"] == 5
    assert "tools/tier_d_nonrl/resume.py" in report["missing_files"]
    assert "trade_rl/evaluation/allocation_market_inputs.py" in report["missing_files"]
    assert {x["symbol"] for x in report["missing_static_exports"]} == {
        "ProcessTree",
        "terminate_owned_tree",
    }


def test_relative_submodule_import_and_false_missing_symbols_are_not_confused() -> None:
    historical = {
        "tools/tier_d_nonrl/runner.py": "from . import progress\nimport trade_rl.data\n",
        "tools/tier_d_nonrl/progress.py": "def done(): pass\n",
        "trade_rl/data/__init__.py": "from .market import X\n",
        "trade_rl/data/market.py": "X = 1\n",
    }
    report = analyze_sources(
        historical, historical, seeds=("tools.tier_d_nonrl.runner",)
    )
    assert report["missing_files"] == []
    assert report["missing_static_exports"] == []
    assert report["reachable_module_count"] == 4
    assert (
        report["recovery_ready"] is False
    )  # static imports are never an execution grant


def test_unresolved_seed_cannot_be_misrepresented_as_empty_success() -> None:
    report = analyze_sources({}, {}, seeds=("tools.tier_d_nonrl.tier_d_runner",))
    assert report["analysis_complete"] is False
    assert report["recovery_ready"] is False
    assert report["missing_seed_modules"] == ["tools.tier_d_nonrl.tier_d_runner"]


def test_historical_source_syntax_error_fails_closed() -> None:
    report = analyze_sources(
        {"tools/tier_d_nonrl/tier_d_runner.py": "def f(:\n"},
        {},
        seeds=("tools.tier_d_nonrl.tier_d_runner",),
    )
    assert report["analysis_complete"] is False
    assert report["recovery_ready"] is False
    assert report["parse_errors"]


def test_cli_fails_closed_when_static_owner_is_missing(monkeypatch, capsys) -> None:
    def fake_tree(repo, sha):
        return (
            {
                f"tools/tier_d_nonrl/{name}.py": "pass\n"
                for name in ("tier_d_runner", "resume", "tier_d_supervisor")
            }
            if sha == "a" * 40
            else {}
        )

    monkeypatch.setattr(audit, "_tree", fake_tree)
    assert (
        audit.main(
            ["--repo", ".", "--historical-sha", "a" * 40, "--maintained-sha", "b" * 40]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["analysis_complete"] is True
    assert report["missing_module_count"] == 3
    assert report["recovery_ready"] is False


def test_tree_reads_only_the_selected_immutable_commit(tmp_path) -> None:
    import subprocess

    def git(*args: str) -> str:
        return (
            subprocess.check_output(
                ["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    git("init", "-q")
    root = tmp_path / "tools" / "tier_d_nonrl"
    root.mkdir(parents=True)
    source = root / "tier_d_runner.py"
    source.write_text("DANGER = False\n", encoding="utf-8")
    (tmp_path / "trade_rl").mkdir()
    (tmp_path / "trade_rl" / "__init__.py").write_text("", encoding="utf-8")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "original",
    )
    original = git("rev-parse", "HEAD")
    source.write_text("DANGER = True\n", encoding="utf-8")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "revised",
    )
    revised = git("rev-parse", "HEAD")
    (root / "untracked.py").write_text("raise AssertionError('not read')\n")
    before = git("status", "--porcelain")
    first = audit._tree(tmp_path, original)
    second = audit._tree(tmp_path, revised)
    assert first["tools/tier_d_nonrl/tier_d_runner.py"] == "DANGER = False\n"
    assert second["tools/tier_d_nonrl/tier_d_runner.py"] == "DANGER = True\n"
    assert "tools/tier_d_nonrl/untracked.py" not in second
    assert git("status", "--porcelain") == before


def test_tree_refuses_untrusted_revision_and_noncommit_object(tmp_path) -> None:
    import subprocess

    import pytest

    with pytest.raises(ValueError, match="literal 40-digit"):
        audit._tree(tmp_path, "HEAD")
    with pytest.raises(ValueError, match="literal 40-digit"):
        audit._tree(tmp_path, "; rm -rf .")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    blob = (
        subprocess.check_output(
            ["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"],
            input=b"fake commit",
        )
        .decode()
        .strip()
    )
    with pytest.raises(ValueError, match="not a commit"):
        audit._tree(tmp_path, blob)


def test_cli_code_two_for_unresolved_seed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        audit,
        "_tree",
        lambda repo, sha: {
            "tools/tier_d_nonrl/tier_d_runner.py": "pass\n",
        },
    )
    assert (
        audit.main(
            [
                "--repo",
                ".",
                "--historical-sha",
                "a" * 40,
                "--maintained-sha",
                "b" * 40,
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["analysis_complete"] is False
    assert result["recovery_ready"] is False


def test_cli_refuses_unavailable_source(monkeypatch, capsys) -> None:
    import pytest

    def fail(repo, sha):
        raise OSError("redacted private reason")

    monkeypatch.setattr(audit, "_tree", fail)
    with pytest.raises(SystemExit) as error:
        audit.main(
            ["--repo", ".", "--historical-sha", "a" * 40, "--maintained-sha", "b" * 40]
        )
    assert error.value.code == 2
    result = capsys.readouterr()
    assert not result.out and "redacted private reason" not in result.err
