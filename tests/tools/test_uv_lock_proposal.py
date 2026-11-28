"""Fail-closed, review-only lockfile proposals."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import uv_lock_proposal as proposal

ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def _lock(version: str = "1.2.3", *, source: str = "https://pypi.org/simple") -> str:
    return f'''version = 1
revision = 5
requires-python = "==3.12.*"

[[package]]
name = "example"
version = "{version}"
source = {{ registry = "{source}" }}
sdist = {{ url = "https://files.pythonhosted.org/packages/example-{version}.tar.gz", hash = "sha256:{HASH}", size = 1 }}

[[package]]
name = "trade-rl"
source = {{ editable = "." }}
dependencies = [{{ name = "example" }}]
'''


def test_current_lock_supports_bounded_release_proposals() -> None:
    snapshot = proposal.read_lock((ROOT / "uv.lock").read_text(encoding="utf-8"))
    targets = proposal.upgrade_targets(snapshot, ())
    assert targets
    assert "numpy>=1.26.4,<2" in targets
    assert any(target.startswith("python-dateutil>=") for target in targets)
    assert len(targets) == len(snapshot.packages)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("1.2.3", "1.2.4"),
        ("1.2.3", "1.3.0"),
        ("1.2.3", "1.2.3.post1"),
        ("13.0.3.0", "13.0.3.1"),
        ("2026.3", "2026.4"),
    ],
)
def test_same_major_forward_releases(before: str, after: str) -> None:
    assert proposal.compare_locks(_lock(before), _lock(after), ("example",)) == (
        ("example", before, after),
    )


@pytest.mark.parametrize(
    "after", ["2.0.0", "1.2.2", "1.2.3rc1", "1!1.2.4", "1.2.4+local", "1.2.3.0"]
)
def test_major_downgrade_and_non_supported_release_are_refused(after: str) -> None:
    with pytest.raises(ValueError):
        proposal.compare_locks(_lock(), _lock(after), ("example",))


@pytest.mark.parametrize(
    "change",
    [
        lambda raw: raw.replace("revision = 5", "revision = 6"),
        lambda raw: raw.replace(
            'requires-python = "==3.12.*"', 'requires-python = ">=3.12"'
        ),
        lambda raw: raw.replace(
            "https://pypi.org/simple", "https://mirror.example/simple"
        ),
        lambda raw: raw.replace("files.pythonhosted.org", "mirror.example"),
        lambda raw: raw.replace("sha256:" + HASH, "sha256:" + "b" * 64),
        lambda raw: raw.replace(
            'dependencies = [{ name = "example" }]', "dependencies = []"
        ),
        lambda raw: raw.replace('name = "example"', 'name = "different"'),
        lambda raw: raw + "\n",
        lambda raw: (
            raw
            + raw[
                raw.index("[[package]]") : raw.index('[[package]]\nname = "trade-rl"')
            ]
        ),
    ],
)
def test_no_schema_source_roster_or_same_release_rewrite(change) -> None:
    with pytest.raises(ValueError):
        proposal.compare_locks(_lock(), change(_lock()), ("example",))


def test_only_selected_packages_are_allowed() -> None:
    with pytest.raises(ValueError):
        proposal.compare_locks(_lock(), _lock("1.2.4"), ("other",))
    assert proposal.compare_locks(_lock(), _lock(), ("example",)) == ()


@pytest.mark.parametrize(
    "names",
    [
        ("unknown",),
        ("--upgrade",),
        ("example", "example"),
        ("example>=1",),
        ("Example",),
    ],
)
def test_target_selection_rejects_unknown_or_invalid_names(
    names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        proposal.upgrade_targets(proposal.read_lock(_lock()), names)


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    for args in (
        ["init", "-q"],
        ["config", "user.name", "Synthetic"],
        ["config", "user.email", "synthetic@example.invalid"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / "pyproject.toml").write_text(
        '[tool.uv]\nrequired-version = "==0.13.10"\n', encoding="utf-8"
    )
    (root / "uv.lock").write_text(_lock(), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "synthetic"], cwd=root, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    monkeypatch.chdir(root)
    monkeypatch.delenv("UV_NO_CONFIG", raising=False)
    monkeypatch.delenv("UV_CONFIG_FILE", raising=False)
    return root, head


def test_candidate_never_changes_source_and_binds_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, head = _repo(tmp_path, monkeypatch)
    output = tmp_path / "proposal"

    def resolve(project: Path, pin: str, targets: tuple[str, ...]) -> None:
        assert project != root
        assert pin == "0.13.10"
        assert targets == ("example>=1.2.3,<2",)
        assert (project / "uv.lock").read_text(encoding="utf-8") == _lock()
        (project / "uv.lock").write_text(_lock("1.2.4"), encoding="utf-8")

    monkeypatch.setattr(proposal, "resolve_candidate", resolve)
    proposal.generate(output, ("example",), head)
    assert (root / "uv.lock").read_text(encoding="utf-8") == _lock()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["source_head"] == head
    assert report["status"] == "CANDIDATE"
    assert report["changes"] == [["example", "1.2.3", "1.2.4"]]
    assert (output / "uv.lock").read_text(encoding="utf-8") == _lock("1.2.4")
    assert (
        (output / "uv-lock.patch")
        .read_text(encoding="utf-8")
        .startswith("--- a/uv.lock\n")
    )


def test_failed_or_disallowed_resolution_has_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, head = _repo(tmp_path, monkeypatch)
    output = tmp_path / "proposal"

    def resolve(project: Path, pin: str, targets: tuple[str, ...]) -> None:
        (project / "uv.lock").write_text(_lock("2.0.0"), encoding="utf-8")

    monkeypatch.setattr(proposal, "resolve_candidate", resolve)
    with pytest.raises(ValueError):
        proposal.generate(output, ("example",), head)
    assert not output.exists()
    assert (root / "uv.lock").read_text(encoding="utf-8") == _lock()


def test_no_change_and_stale_source_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, head = _repo(tmp_path, monkeypatch)
    output = tmp_path / "proposal"
    monkeypatch.setattr(proposal, "resolve_candidate", lambda *args: None)
    with pytest.raises(ValueError):
        proposal.generate(output, (), "f" * 40)
    assert not output.exists()
    proposal.generate(output, (), head)
    assert (
        json.loads((output / "report.json").read_text(encoding="utf-8"))["status"]
        == "NO_CHANGES"
    )
    assert not (output / "uv.lock").exists()
    assert not (output / "uv-lock.patch").exists()
    with pytest.raises(ValueError):
        proposal.generate(output, (), head)
    (root / "uv.lock").write_text(_lock("1.2.4"), encoding="utf-8")
    with pytest.raises(ValueError):
        proposal.generate(tmp_path / "other", (), head)


def test_solver_checks_actual_uv_version_before_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(proposal.shutil, "which", lambda _: "/fake/uv")

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "uv 0.13.7\n", "")

    monkeypatch.setattr(proposal.subprocess, "run", run)
    with pytest.raises(ValueError, match="version"):
        proposal.resolve_candidate(tmp_path, "0.13.10", ("example>=1.2.3,<2",))
    assert calls == [["/fake/uv", "--version"]]


def test_source_change_or_resolver_failure_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, head = _repo(tmp_path, monkeypatch)

    def fail(project: Path, pin: str, targets: tuple[str, ...]) -> None:
        raise subprocess.CalledProcessError(1, ["uv", "lock"])

    monkeypatch.setattr(proposal, "resolve_candidate", fail)
    assert proposal.main(["--head", head, "--output", str(tmp_path / "failure")]) == 2
    assert not (tmp_path / "failure").exists()
    assert (root / "uv.lock").read_text(encoding="utf-8") == _lock()

    def change_manifest(project: Path, pin: str, targets: tuple[str, ...]) -> None:
        (project / "pyproject.toml").write_text("changed", encoding="utf-8")

    monkeypatch.setattr(proposal, "resolve_candidate", change_manifest)
    with pytest.raises(ValueError, match="source or manifest changed"):
        proposal.generate(tmp_path / "changed", (), head)
    assert not (tmp_path / "changed").exists()


def test_solver_preserves_exact_config_and_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_INDEX_URL", "https://invalid.example/simple")
    monkeypatch.setenv("PIP_INDEX_URL", "https://invalid.example/simple")
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-unused-token")
    monkeypatch.setattr(proposal.shutil, "which", lambda _: "/fake/uv")
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(argv)
        env = kwargs["env"]
        assert "UV_INDEX_URL" not in env
        assert "PIP_INDEX_URL" not in env
        assert "GITHUB_TOKEN" not in env
        assert env["UV_PYTHON_DOWNLOADS"] == "never"
        assert kwargs["cwd"] == tmp_path
        if argv[-1] == "--version":
            return subprocess.CompletedProcess(argv, 0, "uv 0.13.10\n", "")
        assert kwargs["check"] is True
        config = Path(argv[argv.index("--config-file") + 1])
        assert config.read_text(encoding="utf-8") == 'required-version = "==0.13.10"\n'
        assert "--no-build" in argv
        assert "--no-config" not in argv
        assert argv[argv.index("--default-index") + 1] == "https://pypi.org/simple"
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(proposal.subprocess, "run", run)
    proposal.resolve_candidate(tmp_path, "0.13.10", ("example>=1.2.3,<2",))
    assert len(calls) == 4
    assert calls[0][-1] == "--version"
    assert calls[1][-1] == calls[3][-1] == "--check"
    assert calls[2][-2:] == ["--upgrade-package", "example>=1.2.3,<2"]
