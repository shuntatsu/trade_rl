"""Synthetic host preflight oracles; no external account or market inputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import claude_host_preflight as probe

ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    root = tmp_path / "trusted repository with spaces"
    root.mkdir()
    _git(root, "init", "--quiet", "--initial-branch=main")
    for item in (
        "AGENTS.md",
        "CLAUDE.md",
        ".claude/settings.json",
        ".claude/skills/research-challenge/SKILL.md",
        "tools/claude_session_context.py",
    ):
        target = root / item
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / item, target)
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test Agent",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
        "--quiet",
    )
    sha = _git(root, "rev-parse", "HEAD")
    _git(root, "remote", "add", "origin", "git@github.com:shuntatsu/trade_rl.git")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    claude = fake_bin / "claude"
    claude.write_text(
        '#!/usr/bin/env python3\nprint("2.1.300 (Claude Code) - private-path-redacted")\n',
        encoding="utf-8",
    )
    claude.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    return root, sha


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("git@github.com:shuntatsu/trade_rl.git", True),
        ("https://github.com/shuntatsu/trade_rl", True),
        ("ssh://git@github.com/shuntatsu/trade_rl.git", True),
        ("https://github.com/shuntatsu/trade_rl.git/other", False),
        ("https://github.com/shuntatsu/trade_rl.evil", False),
        ("https://github.com.evil/shuntatsu/trade_rl", False),
        ("https://private-token@github.com/shuntatsu/trade_rl", False),
        ("file:///somewhere/shuntatsu/trade_rl", False),
        ("https://github.com/shuntatsu/trade_rl?x=1", False),
        ("https://github.com:invalid/shuntatsu/trade_rl", False),
        ("https://[broken", False),
        ("ssh://[broken", False),
    ],
)
def test_remote_identity_is_exact(remote: str, expected: bool) -> None:
    assert probe._known_repo(remote) is expected


def test_clean_trusted_fixture_produces_bounded_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    before = probe._index_fingerprint(root)
    got = probe.preflight(root, required_merge=sha)
    assert got == {
        "schema": "trade_rl_claude_host_preflight_v1",
        "claim": "local-preflight-only; Claude Code host activation NOT verified",
        "repository_root": True,
        "expected_upstream": True,
        "head": sha,
        "merged_base_ancestor": True,
        "working_tree_clean": True,
        "shared_instructions": True,
        "manual_research_skill": True,
        "configured_session_hook": True,
        "claude_cli_version": "2.1.300",
        "claude_cli_present": True,
        "hook_direct_exec_passed": True,
        "git_index_unchanged": True,
        "preflight_ok": True,
    }
    assert probe._index_fingerprint(root) == before
    serialized = json.dumps(got)
    assert "trusted repository" not in serialized
    assert "private-path-redacted" not in serialized
    assert "git@github.com:" not in serialized


def test_dirty_worktree_fails_closed_and_does_not_execute_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    (root / "API_SECRET_token.txt").write_text("password=test\n", encoding="utf-8")
    result = probe.preflight(root, required_merge=sha)
    assert result["working_tree_clean"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False
    assert "API_SECRET" not in json.dumps(result)
    assert "password" not in json.dumps(result)


def test_wrong_upstream_never_runs_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    _git(root, "remote", "set-url", "origin", "git@github.com:someone/other.git")
    result = probe.preflight(root, required_merge=sha)
    assert result["repository_root"] is True
    assert result["expected_upstream"] is False
    assert result["claude_cli_present"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False


@pytest.mark.parametrize("remote", ["https://[broken", "ssh://[broken"])
def test_malformed_origin_url_fails_closed_without_hook_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: str
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    _git(root, "remote", "set-url", "origin", remote)
    result = probe.preflight(root, required_merge=sha)
    assert result["repository_root"] is True
    assert result["expected_upstream"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False
    assert "broken" not in json.dumps(result)


def test_wrong_merge_commit_never_runs_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    result = probe.preflight(root, required_merge="a" * 40)
    assert result["merged_base_ancestor"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False


def test_missing_claude_cli_does_not_claim_host_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    original = probe._exec

    def no_cli(
        args: list[str], *, cwd: Path, timeout: int = 5, input_data: str | None = None
    ) -> tuple[int, str]:
        if args[0] == "claude":
            return -1, ""
        return original(args, cwd=cwd, timeout=timeout, input_data=input_data)

    monkeypatch.setattr(probe, "_exec", no_cli)
    result = probe.preflight(root, required_merge=sha)
    assert result["claude_cli_present"] is False
    assert result["claude_cli_version"] is None
    assert result["hook_direct_exec_passed"] is True
    assert result["preflight_ok"] is False


def test_manual_skill_and_settings_are_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    settings = root / ".claude/settings.json"
    original = settings.read_text(encoding="utf-8")
    settings.write_text("{}\n", encoding="utf-8")
    result = probe.preflight(root, required_merge=sha)
    assert result["configured_session_hook"] is False
    assert result["hook_direct_exec_passed"] is False
    settings.write_text(original, encoding="utf-8")
    skill = root / ".claude/skills/research-challenge/SKILL.md"
    skill.write_text(skill.read_text(encoding="utf-8").replace("true", "false"))
    result = probe.preflight(root, required_merge=sha)
    assert result["manual_research_skill"] is False
    assert result["preflight_ok"] is False


def test_hook_index_mutation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    hook = root / "tools/claude_session_context.py"
    hook.write_text(
        """from pathlib import Path
import os
import subprocess
root = Path(os.environ['CLAUDE_PROJECT_DIR'])
p = root / '.git/index'
s = p.stat()
os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns + 1_000_000_000))
head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
print('[trade_rl / Claude Code | local snapshot only]')
print('HEAD: ' + head)
print('this hook cannot authorize economic runs')
""",
        encoding="utf-8",
    )
    _git(root, "add", "tools/claude_session_context.py")
    _git(
        root,
        "-c",
        "user.name=Test Agent",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "tampered hook",
        "--quiet",
    )
    sha = _git(root, "rev-parse", "HEAD")
    result = probe.preflight(root, required_merge=sha)
    assert result["git_index_unchanged"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False


def test_linked_detached_worktree_with_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    dest = tmp_path / "detached worktree with spaces"
    _git(root, "worktree", "add", "--detach", str(dest), sha)
    assert probe.preflight(dest, required_merge=sha)["preflight_ok"] is True


def test_nonrepo_does_not_print_sensitive_paths(tmp_path: Path) -> None:
    root = tmp_path / "secret-cwd"
    root.mkdir()
    result = probe.preflight(root)
    assert result["repository_root"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False
    assert "secret-cwd" not in json.dumps(result)


def test_unexpected_hook_output_is_not_accepted_or_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    hook = root / "tools/claude_session_context.py"
    hook.write_text(hook.read_text(encoding="utf-8") + "\nprint('SECRET_PATH_HERE')\n")
    _git(root, "add", "tools/claude_session_context.py")
    _git(
        root,
        "-c",
        "user.name=Test Agent",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "unexpected output",
        "--quiet",
    )
    sha = _git(root, "rev-parse", "HEAD")
    report = probe.preflight(root, required_merge=sha)
    assert report["git_index_unchanged"] is True
    assert report["hook_direct_exec_passed"] is False
    assert report["preflight_ok"] is False
    assert "SECRET_PATH_HERE" not in json.dumps(report)


def test_no_cli_version_ignores_unexpected_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha = _fixture(tmp_path, monkeypatch)
    fake_cli = tmp_path / "bin/claude"
    fake_cli.write_text(
        "#!/usr/bin/env python3\nprint('CREDENTIAL=not-a-version')\n",
        encoding="utf-8",
    )
    report = probe.preflight(root, required_merge=sha)
    assert report["claude_cli_present"] is False
    assert report["preflight_ok"] is False
    assert "CREDENTIAL" not in json.dumps(report)


def test_claude_version_cannot_change_index_before_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Version probes are subprocesses and must be covered by the index fence."""
    root, sha = _fixture(tmp_path, monkeypatch)
    before = probe._index_fingerprint(root)
    fake_cli = tmp_path / "bin" / "claude"
    fake_cli.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import subprocess\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['CLAUDE_PROJECT_DIR'])\n"
        "index = subprocess.check_output(\n"
        "    ['git', '-C', str(root), 'rev-parse', '--git-path', 'index'],\n"
        "    text=True,\n"
        ").strip()\n"
        "path = Path(index)\n"
        "if not path.is_absolute():\n"
        "    path = root / path\n"
        "stat = path.stat()\n"
        "os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))\n"
        "print('2.1.300 (Claude Code)')\n",
        encoding="utf-8",
    )
    result = probe.preflight(root, required_merge=sha)
    assert probe._index_fingerprint(root) != before
    assert result["git_index_unchanged"] is False
    assert result["preflight_ok"] is False


def test_cli_version_untracked_workspace_write_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version subprocess cannot leave a dirty tree and still be approved."""
    root, sha = _fixture(tmp_path, monkeypatch)
    before = probe._index_fingerprint(root)
    fake_cli = tmp_path / "bin" / "claude"
    fake_cli.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['CLAUDE_PROJECT_DIR'])\n"
        "(root / 'UNTRACKED_VERSION_OUTPUT').write_text('private-content')\n"
        "print('2.1.300 (Claude Code)')\n",
        encoding="utf-8",
    )
    result = probe.preflight(root, required_merge=sha)
    assert probe._index_fingerprint(root) == before
    assert (root / "UNTRACKED_VERSION_OUTPUT").exists()
    assert result["working_tree_clean"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False
    assert "UNTRACKED_VERSION_OUTPUT" not in json.dumps(result)


def test_hook_untracked_workspace_write_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid hook stdout cannot hide new files written by the hook process."""
    root, _ = _fixture(tmp_path, monkeypatch)
    hook = root / "tools/claude_session_context.py"
    hook.write_text(
        hook.read_text(encoding="utf-8")
        + "\nfrom pathlib import Path\n"
        + "Path(__file__).resolve().parents[1].joinpath("
        + "'UNTRACKED_HOOK_OUTPUT').write_text('private-content')\n",
        encoding="utf-8",
    )
    _git(root, "add", "tools/claude_session_context.py")
    _git(
        root,
        "-c",
        "user.name=Test Agent",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture hook side effect",
        "--quiet",
    )
    sha = _git(root, "rev-parse", "HEAD")
    before = probe._index_fingerprint(root)
    result = probe.preflight(root, required_merge=sha)
    assert probe._index_fingerprint(root) == before
    assert (root / "UNTRACKED_HOOK_OUTPUT").exists()
    assert result["working_tree_clean"] is False
    assert result["hook_direct_exec_passed"] is False
    assert result["preflight_ok"] is False
    assert "UNTRACKED_HOOK_OUTPUT" not in json.dumps(result)
