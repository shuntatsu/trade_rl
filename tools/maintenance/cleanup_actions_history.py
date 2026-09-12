from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

HIGH_VALUE_RE = re.compile(
    r"(?:research|experiment|evidence|baseline|pre[-_ ]?reg|prereg|canonical|"
    r"repro|benchmark|publication|artifact|audit|falsif|mutation|release|provenance)",
    re.IGNORECASE,
)
DISPOSABLE_RE = re.compile(
    r"(?:helper|cleanup|temporary|(?:^|[/ _-])temp(?:[/ _-]|$)|merge[-_ ]?helper|"
    r"branch[-_ ]?retire|debug|probe)",
    re.IGNORECASE,
)
TMP_PREFIXES = ("tmp/", "temp/", "cleanup/", "ops/tmp-", "chore/tmp")
NUISANCE_CONCLUSIONS = {"cancelled", "skipped", "stale", "neutral", "action_required"}
RUN_ID_RE = re.compile(r"(?<!\d)(\d{10,12})(?!\d)")


def _parse_github_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def classify_run(
    run: dict[str, Any],
    *,
    now: datetime,
    protected_shas: set[str],
    protected_run_ids: set[int],
    artifact_run_ids: set[int],
    existing_branches: set[str],
    current_run_id: int,
    current_workflow_path: str,
    recent_hours: int,
) -> tuple[str, str]:
    run_id = int(run["id"])
    if run_id == current_run_id:
        return "KEEP", "current cleanup run"
    if run.get("status") != "completed":
        return "KEEP", "run is not completed"
    if run_id in protected_run_ids:
        return "KEEP", "run id is referenced in repository/issue/PR text"
    if run_id in artifact_run_ids:
        return "KEEP", "run has an Actions artifact"
    if run.get("head_sha") in protected_shas:
        return "KEEP", "run belongs to current main or an open PR HEAD"

    haystack = " ".join(
        str(run.get(key) or "")
        for key in ("name", "path", "head_branch", "display_title")
    )
    if HIGH_VALUE_RE.search(haystack):
        return "KEEP", "research/evidence/high-value keyword"
    if run.get("path") == current_workflow_path:
        return "DELETE", "superseded cleanup workflow run"

    created_at = _parse_github_time(str(run["created_at"]))
    if created_at >= now - timedelta(hours=recent_hours):
        return "KEEP", f"created within last {recent_hours} hours"

    branch = str(run.get("head_branch") or "")
    conclusion = str(run.get("conclusion") or "")
    branch_exists = bool(branch) and branch in existing_branches
    is_tmp = any(branch.startswith(prefix) for prefix in TMP_PREFIXES)
    is_disposable = bool(DISPOSABLE_RE.search(haystack))

    if is_tmp and (is_disposable or conclusion in NUISANCE_CONCLUSIONS):
        return "DELETE", "old disposable tmp/helper run"
    if not branch_exists and is_disposable:
        return "DELETE", "old disposable workflow/branch with deleted branch"

    is_ci = (
        run.get("name") == "CI"
        or str(run.get("path") or "").endswith("/.github/workflows/ci.yml")
        or str(run.get("path") or "") == ".github/workflows/ci.yml"
    )
    if not branch_exists and is_ci and conclusion in NUISANCE_CONCLUSIONS:
        return "DELETE", "old cancelled/skipped/stale CI on deleted branch"
    if branch_exists:
        return "KEEP", "branch still exists"
    return "REVIEW", "old completed run is not safely disposable under strict policy"


class GitHubApi:
    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token
        self.base = "https://api.github.com"
        self.rate_remaining: int | None = None

    def _request(self, method: str, path: str) -> tuple[Any, dict[str, str]]:
        url = path if path.startswith("https://") else f"{self.base}{path}"
        req = urllib.request.Request(
            url,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "trade-rl-actions-history-cleanup",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                if "x-ratelimit-remaining" in headers:
                    self.rate_remaining = int(headers["x-ratelimit-remaining"])
                body = resp.read()
                if not body:
                    return None, headers
                return json.loads(body.decode("utf-8")), headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub API {method} {url} failed: {exc.code} {detail}"
            ) from exc

    def get(self, path: str) -> Any:
        data, _ = self._request("GET", path)
        return data

    def get_paginated(self, path: str, item_key: str | None = None) -> list[Any]:
        sep = "&" if "?" in path else "?"
        next_url = f"{path}{sep}per_page=100"
        out: list[Any] = []
        while next_url:
            data, headers = self._request("GET", next_url)
            items = data[item_key] if item_key else data
            if not isinstance(items, list):
                raise RuntimeError(f"Expected list from {next_url}")
            out.extend(items)
            next_url = _next_link(headers.get("link", ""))
        return out

    def delete_run(self, run_id: int) -> None:
        self._request("DELETE", f"/repos/{self.repo}/actions/runs/{run_id}")


def _next_link(link_header: str) -> str | None:
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' in section:
            return section.split(";", 1)[0].strip()[1:-1]
    return None


def _extract_run_ids(texts: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for text in texts:
        for match in RUN_ID_RE.findall(text or ""):
            ids.add(int(match))
    return ids


def _repository_numeric_refs(repo_root: Path) -> set[int]:
    try:
        proc = subprocess.run(
            ["git", "grep", "-I", "-h", "-Eo", r"[0-9]{10,12}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return set()
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed: {proc.stderr}")
    return {int(x) for x in proc.stdout.splitlines() if x.isdigit()}


def _collect_text_refs(api: GitHubApi, repo_root: Path) -> set[int]:
    texts: list[str] = []
    issues = api.get_paginated(f"/repos/{api.repo}/issues?state=all")
    texts.extend(str(item.get("body") or "") for item in issues)
    issue_comments = api.get_paginated(f"/repos/{api.repo}/issues/comments")
    texts.extend(str(item.get("body") or "") for item in issue_comments)
    review_comments = api.get_paginated(f"/repos/{api.repo}/pulls/comments")
    texts.extend(str(item.get("body") or "") for item in review_comments)
    commit_comments = api.get_paginated(f"/repos/{api.repo}/comments")
    texts.extend(str(item.get("body") or "") for item in commit_comments)
    return _extract_run_ids(texts) | _repository_numeric_refs(repo_root)


def dedupe_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for run in runs:
        run_id = int(run["id"])
        if run_id in seen:
            continue
        seen.add(run_id)
        out.append(run)
    return out


def collect_context(api: GitHubApi, repo_root: Path) -> dict[str, Any]:
    repo = api.get(f"/repos/{api.repo}")
    default_branch = repo["default_branch"]
    default_branch_data = api.get(
        f"/repos/{api.repo}/branches/{urllib.parse.quote(default_branch, safe='')}"
    )
    protected_shas = {default_branch_data["commit"]["sha"]}

    open_prs = api.get_paginated(f"/repos/{api.repo}/pulls?state=open")
    protected_shas.update(pr["head"]["sha"] for pr in open_prs)

    branches = api.get_paginated(f"/repos/{api.repo}/branches")
    existing_branches = {branch["name"] for branch in branches}

    artifacts = api.get_paginated(
        f"/repos/{api.repo}/actions/artifacts", item_key="artifacts"
    )
    artifact_run_ids = {
        int(artifact["workflow_run"]["id"])
        for artifact in artifacts
        if artifact.get("workflow_run") and artifact["workflow_run"].get("id")
    }

    run_meta = api.get(f"/repos/{api.repo}/actions/runs?per_page=1")
    repo_total_runs = int(run_meta["total_count"])

    candidate_runs: list[dict[str, Any]] = []
    for status in ("cancelled", "skipped", "stale"):
        candidate_runs.extend(
            api.get_paginated(
                f"/repos/{api.repo}/actions/runs?status={status}",
                item_key="workflow_runs",
            )
        )

    cleanup_branch = os.environ.get("CLEANUP_BRANCH", "")
    if cleanup_branch:
        candidate_runs.extend(
            api.get_paginated(
                f"/repos/{api.repo}/actions/runs?branch={urllib.parse.quote(cleanup_branch, safe='')}",
                item_key="workflow_runs",
            )
        )
    runs = dedupe_runs(candidate_runs)

    referenced_run_ids = _collect_text_refs(api, repo_root)
    actual_run_ids = {int(run["id"]) for run in runs}
    referenced_run_ids &= actual_run_ids

    return {
        "default_branch": default_branch,
        "protected_shas": protected_shas,
        "existing_branches": existing_branches,
        "artifact_run_ids": artifact_run_ids,
        "protected_run_ids": referenced_run_ids,
        "runs": runs,
        "repo_total_runs": repo_total_runs,
        "open_prs": [
            {"number": pr["number"], "head_sha": pr["head"]["sha"]}
            for pr in open_prs
        ],
    }


def write_summary(report: dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    lines = [
        "# Actions history cleanup",
        "",
        f"- Mode: **{'DRY RUN' if report['dry_run'] else 'EXECUTE'}**",
        f"- Repository total runs: **{report['total_runs']}**",
        f"- Strict candidate pool scanned: **{report['scanned_candidate_pool']}**",
        f"- Safe delete candidates: **{report['candidate_count']}**",
        f"- Deleted: **{report['deleted_count']}**",
        f"- REVIEW (not deleted): **{report['review_count']}**",
        f"- Protected explicit references: **{report['protected_reference_count']}**",
        f"- Runs with artifacts protected: **{report['artifact_run_count']}**",
        f"- Open PRs protected: **{len(report['open_prs'])}**",
        "",
        "## Decision counts",
        "",
    ]
    for key, value in sorted(report["decision_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Delete reason counts", ""])
    for key, value in sorted(report["delete_reason_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Sample safe-delete candidates", ""])
    for sample in report["candidate_samples"][:20]:
        lines.append(
            f"- `{sample['id']}` — {sample['name']} — `{sample['head_branch']}` — "
            f"{sample['conclusion']} — {sample['reason']}"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    print("SUMMARY_JSON=" + json.dumps(report, sort_keys=True, separators=(",", ":")))
    if summary_path:
        Path(summary_path).write_text(text, encoding="utf-8")


def run_cleanup() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    current_run_id = int(os.environ["GITHUB_RUN_ID"])
    current_workflow_path = os.environ.get(
        "CURRENT_WORKFLOW_PATH", ".github/workflows/actions-history-cleanup.yml"
    )
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    max_delete = int(os.environ.get("MAX_DELETE", "3000"))
    recent_hours = int(os.environ.get("RECENT_HOURS", "72"))
    api = GitHubApi(repo, token)
    context = collect_context(api, Path.cwd())
    now = datetime.now(timezone.utc)

    classified: list[tuple[dict[str, Any], str, str]] = []
    for item in context["runs"]:
        decision, reason = classify_run(
            item,
            now=now,
            protected_shas=context["protected_shas"],
            protected_run_ids=context["protected_run_ids"],
            artifact_run_ids=context["artifact_run_ids"],
            existing_branches=context["existing_branches"],
            current_run_id=current_run_id,
            current_workflow_path=current_workflow_path,
            recent_hours=recent_hours,
        )
        classified.append((item, decision, reason))

    candidates = [
        (item, reason)
        for item, decision, reason in classified
        if decision == "DELETE"
    ]
    candidates.sort(key=lambda pair: pair[0]["created_at"])
    selected = candidates[:max_delete]
    deleted: list[int] = []
    if not dry_run:
        for item, _ in selected:
            if api.rate_remaining is not None and api.rate_remaining < 250:
                raise RuntimeError(
                    "Stopping before rate-limit exhaustion; "
                    f"remaining={api.rate_remaining}, deleted={len(deleted)}"
                )
            api.delete_run(int(item["id"]))
            deleted.append(int(item["id"]))
            if len(deleted) % 250 == 0:
                print(f"Deleted {len(deleted)} runs...")

    decision_counts = Counter(decision for _, decision, _ in classified)
    delete_reason_counts = Counter(
        reason for _, decision, reason in classified if decision == "DELETE"
    )
    report = {
        "dry_run": dry_run,
        "total_runs": context["repo_total_runs"],
        "scanned_candidate_pool": len(context["runs"]),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "deleted_count": len(deleted),
        "review_count": decision_counts.get("REVIEW", 0),
        "decision_counts": dict(decision_counts),
        "delete_reason_counts": dict(delete_reason_counts),
        "protected_reference_count": len(context["protected_run_ids"]),
        "artifact_run_count": len(context["artifact_run_ids"]),
        "open_prs": context["open_prs"],
        "rate_remaining": api.rate_remaining,
        "candidate_samples": [
            {
                "id": int(item["id"]),
                "name": str(item.get("name") or ""),
                "head_branch": str(item.get("head_branch") or ""),
                "conclusion": str(item.get("conclusion") or ""),
                "created_at": str(item.get("created_at") or ""),
                "reason": reason,
            }
            for item, reason in candidates[:50]
        ],
    }
    write_summary(report)
    if not dry_run and len(deleted) != len(selected):
        raise RuntimeError("Not all selected safe-delete candidates were deleted")
    return 0


def self_test() -> int:
    now = datetime(2026, 9, 12, 4, 0, tzinfo=timezone.utc)
    old = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    recent = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    def item(**overrides: Any) -> dict[str, Any]:
        base = {
            "id": 10000000001,
            "name": "CI",
            "path": ".github/workflows/ci.yml",
            "head_branch": "deleted/branch",
            "head_sha": "deadbeef",
            "status": "completed",
            "conclusion": "cancelled",
            "event": "pull_request",
            "created_at": old,
        }
        base.update(overrides)
        return base

    common = dict(
        now=now,
        protected_shas={"mainsha", "prsha"},
        protected_run_ids={22222222222},
        artifact_run_ids={33333333333},
        existing_branches={"main", "live/branch"},
        current_run_id=99999999999,
        current_workflow_path=".github/workflows/actions-history-cleanup.yml",
        recent_hours=72,
    )
    checks = [
        (item(status="in_progress", conclusion=None), "KEEP"),
        (item(head_sha="mainsha"), "KEEP"),
        (item(head_sha="prsha"), "KEEP"),
        (item(id=22222222222), "KEEP"),
        (item(id=33333333333), "KEEP"),
        (item(created_at=recent), "KEEP"),
        (
            item(
                name="Canonical M2 execution economics audit",
                head_branch="tmp/m2-audit",
            ),
            "KEEP",
        ),
        (
            item(
                name="Merge helper",
                head_branch="tmp/merge-helper",
                conclusion="failure",
            ),
            "DELETE",
        ),
        (item(), "DELETE"),
        (item(conclusion="failure"), "REVIEW"),
        (item(head_branch="live/branch"), "KEEP"),
        (
            item(
                id=88888888888,
                path=".github/workflows/actions-history-cleanup.yml",
                name="Actions history cleanup",
            ),
            "DELETE",
        ),
        (
            item(
                id=99999999999,
                path=".github/workflows/actions-history-cleanup.yml",
                name="Actions history cleanup",
                created_at=recent,
            ),
            "KEEP",
        ),
    ]
    for idx, (case, expected) in enumerate(checks, 1):
        actual, reason = classify_run(case, **common)
        if actual != expected:
            raise AssertionError(
                f"case {idx}: expected {expected}, got {actual}: {reason}"
            )
    print(f"self-test: {len(checks)} classification contracts passed")
    return 0


if __name__ == "__main__":
    sys.exit(run_cleanup())
