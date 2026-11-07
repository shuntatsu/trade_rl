"""Redacted, non-authorizing Claude Code host preflight for trade_rl.

This exercises the project hook as a subprocess, *not* the Claude Code host.
Real SessionStart/skill discovery still require manual Claude Code verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

MERGED_BASE = "47c17fe429f48716a6ce013814380528e6f42b1d"
HOOK_ARGUMENT = "${CLAUDE_PROJECT_DIR}/tools/claude_session_context.py"
SOURCES = {"startup", "resume", "compact", "clear", "fork"}
_VERSION = re.compile(r"\b\d+\.\d+\.\d+\b")
_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _exec(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 5,
    input_data: str | None = None,
) -> tuple[int, str]:
    """Return exit status and stdout, never stderr (may contain private paths)."""
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            input=input_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(cwd)},
        )
    except (OSError, subprocess.TimeoutExpired):
        return -1, ""
    return result.returncode, result.stdout.strip()


def _git(root: Path, *args: str) -> tuple[int, str]:
    return _exec(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", *args],
        cwd=root,
    )


def _known_repo(remote: str) -> bool:
    """Check only the literal upstream identity, without logging remote URLs."""
    if remote.startswith("git@github.com:"):
        path = remote[len("git@github.com:") :]
    else:
        parsed = urlparse(remote)
        if parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com":
            return False
        if parsed.username not in {None, "git"} or parsed.password is not None:
            return False
        try:
            port = parsed.port
        except ValueError:
            return False
        if parsed.query or parsed.fragment or parsed.params or port is not None:
            return False
        path = parsed.path.lstrip("/")
    return path.removesuffix(".git").rstrip("/") == "shuntatsu/trade_rl"


def _index_fingerprint(root: Path) -> tuple[str, int] | None:
    code, index = _git(root, "rev-parse", "--git-path", "index")
    if code != 0 or not index:
        return None
    path = Path(index)
    if not path.is_absolute():
        path = root / path
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None
    return digest, stat.st_mtime_ns


def _settings_hook(root: Path) -> dict[str, object] | None:
    try:
        data = json.loads((root / ".claude/settings.json").read_text(encoding="utf-8"))
        hooks = data["hooks"]["SessionStart"]
        if not isinstance(hooks, list) or len(hooks) != 1:
            return None
        group = hooks[0]
        entries = group["hooks"]
        if not isinstance(entries, list) or len(entries) != 1:
            return None
        hook = entries[0]
        if (
            set(group["matcher"].split("|")) != SOURCES
            or hook.get("type") != "command"
            or hook.get("command") != "python"
            or hook.get("args") != [HOOK_ARGUMENT]
            or hook.get("timeout") != 15
        ):
            return None
        return hook
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _skill_available(root: Path) -> bool:
    try:
        skill = (root / ".claude/skills/research-challenge/SKILL.md").read_text(
            encoding="utf-8"
        )
    except OSError:
        return False
    header = skill.split("---", 2)
    return (
        len(header) == 3
        and header[0] == ""
        and "name: research-challenge" in header[1].splitlines()
        and "disable-model-invocation: true" in header[1].splitlines()
    )


def preflight(root: Path, *, required_merge: str = MERGED_BASE) -> dict[str, object]:
    """Collect bounded booleans, without credentials, paths or market data."""
    root = root.resolve()
    report: dict[str, object] = {
        "schema": "trade_rl_claude_host_preflight_v1",
        "claim": "local-preflight-only; Claude Code host activation NOT verified",
    }
    code, inside = _git(root, "rev-parse", "--is-inside-work-tree")
    code_top, top = _git(root, "rev-parse", "--show-toplevel")
    root_valid = (
        code == code_top == 0 and inside == "true" and Path(top).resolve() == root
    )
    report["repository_root"] = root_valid
    code, remote = _git(root, "remote", "get-url", "origin")
    report["expected_upstream"] = root_valid and code == 0 and _known_repo(remote)
    code, current_head = _git(root, "rev-parse", "--verify", "HEAD")
    report["head"] = (
        current_head if code == 0 and _SHA.fullmatch(current_head) else None
    )
    ancestor_code, _ = _git(root, "merge-base", "--is-ancestor", required_merge, "HEAD")
    report["merged_base_ancestor"] = root_valid and ancestor_code == 0
    status_code, dirty = _git(
        root, "status", "--porcelain=v1", "--untracked-files=normal"
    )
    report["working_tree_clean"] = root_valid and status_code == 0 and dirty == ""
    try:
        entry = (root / "CLAUDE.md").read_text(encoding="utf-8")
        report["shared_instructions"] = (
            entry.startswith("@AGENTS.md\n") and (root / "AGENTS.md").is_file()
        )
    except OSError:
        report["shared_instructions"] = False
    report["manual_research_skill"] = _skill_available(root)
    hook = _settings_hook(root)
    report["configured_session_hook"] = hook is not None
    # Do not execute either Claude CLI or the hook in an untrusted/dirty checkout.
    safe = all(
        report[name] is True
        for name in (
            "repository_root",
            "expected_upstream",
            "merged_base_ancestor",
            "working_tree_clean",
            "shared_instructions",
            "configured_session_hook",
            "manual_research_skill",
        )
    )
    report["claude_cli_version"] = None
    report["claude_cli_present"] = False
    report["hook_direct_exec_passed"] = False
    report["git_index_unchanged"] = False
    if safe and hook is not None:
        # Include the external version subprocess in the index mutation fence.
        before = _index_fingerprint(root)
        version_code, version_output = _exec(["claude", "--version"], cwd=root)
        version_match = _VERSION.search(version_output) if version_code == 0 else None
        report["claude_cli_version"] = version_match.group(0) if version_match else None
        report["claude_cli_present"] = version_match is not None
        if before is not None:
            script = str(hook["args"][0]).replace("${CLAUDE_PROJECT_DIR}", str(root))
            result_code, stdout = _exec(
                [str(hook["command"]), script],
                cwd=root,
                timeout=15,
                input_data='{"hook_event_name":"SessionStart","source":"startup"}',
            )
            after = _index_fingerprint(root)
            report["git_index_unchanged"] = before == after
            lines = stdout.splitlines()
            expected_tail = [
                "Cached refs may be stale. Check current GitHub main, open PRs.",
                "Check research authority and exact-head CI/reviews separately.",
                "Follow AGENTS.md and docs/AGENTS.md; this hook cannot authorize economic runs.",
                "This hook does not authorize sealed-data access, live orders, or merges.",
            ]
            report["hook_direct_exec_passed"] = (
                result_code == 0
                and len(lines) == 10
                and lines[0] == "[trade_rl / Claude Code | local snapshot only]"
                and lines[1].startswith("branch: ")
                and lines[2] == f"HEAD: {current_head}"
                and lines[3] == "working tree: clean"
                and lines[4].startswith("cached origin/main: ")
                and lines[5]
                in {
                    "On main: inspect only; use a dedicated branch/worktree.",
                    "Never modify main directly; check worktree ownership first.",
                }
                and lines[6:] == expected_tail
                and report["git_index_unchanged"] is True
            )
    report["preflight_ok"] = all(
        report[name] is True
        for name in (
            "repository_root",
            "expected_upstream",
            "merged_base_ancestor",
            "working_tree_clean",
            "shared_instructions",
            "manual_research_skill",
            "configured_session_hook",
            "claude_cli_present",
            "hook_direct_exec_passed",
            "git_index_unchanged",
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    report = preflight(args.root)
    print(json.dumps(report, sort_keys=True, ensure_ascii=True))
    return 0 if report["preflight_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
