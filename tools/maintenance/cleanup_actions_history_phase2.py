from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

HIGH_VALUE_RE = re.compile(
    r"(?:research|experiment|evidence|baseline|pre[-_ ]?reg|prereg|canonical|"
    r"repro|benchmark|publication|artifact|audit|falsif|mutation|release|provenance|"
    r"golden|verification|verify|\bRED\b|\bGREEN\b|primary[-_ ]?gate)",
    re.IGNORECASE,
)
RUN_ID_RE = re.compile(r"(?<!\d)(\d{10,12})(?!\d)")
CLEANUP_BRANCH_PREFIX = "ops/actions-history-cleanup"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_ci(run: dict[str, Any]) -> bool:
    return run.get("name") == "CI" or str(run.get("path") or "") == ".github/workflows/ci.yml"


def _success_key(run: dict[str, Any]) -> tuple[str, int]:
    return (str(run.get("head_branch") or ""), int(run.get("workflow_id") or 0))


def _has_high_value_keyword(run: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(run.get(key) or "")
        for key in ("name", "path", "head_branch", "display_title")
    )
    return HIGH_VALUE_RE.search(haystack) is not None


def classify_failure(
    run: dict[str, Any],
    *,
    now: datetime,
    protected_shas: set[str],
    protected_run_ids: set[int],
    artifact_run_ids: set[int],
    existing_branches: set[str],
    latest_success: dict[tuple[str, int], dict[str, Any]],
    current_run_id: int,
    recent_hours: int,
) -> tuple[str, str]:
    run_id = int(run["id"])
    if run_id == current_run_id:
        return "KEEP", "current cleanup run"
    if run.get("status") != "completed" or run.get("conclusion") != "failure":
        return "KEEP", "not a completed failure"
    if run_id in protected_run_ids:
        return "KEEP", "run id is referenced in repository/issue/PR text"
    if run_id in artifact_run_ids:
        return "KEEP", "run has an Actions artifact"
    if str(run.get("head_sha") or "") in protected_shas:
        return "KEEP", "run belongs to current main or an open PR HEAD"
    if _has_high_value_keyword(run):
        return "KEEP", "research/evidence/high-value keyword"

    branch = str(run.get("head_branch") or "")
    if branch.startswith(CLEANUP_BRANCH_PREFIX):
        return "DELETE", "obsolete cleanup-helper failure"
    if not _is_ci(run):
        return "REVIEW", "failed non-CI workflow is not deleted in phase 2"
    if branch and branch in existing_branches:
        return "KEEP", "branch still exists"

    created_at = _parse_time(str(run["created_at"]))
    if created_at >= now - timedelta(hours=recent_hours):
        return "KEEP", f"created within last {recent_hours} hours"

    success = latest_success.get(_success_key(run))
    if success is None:
        return "REVIEW", "deleted branch has no successful run for the same workflow"
    if _parse_time(str(success["created_at"])) <= created_at:
        return "REVIEW", "same-workflow success is not later than the failure"
    return "DELETE", "superseded failed CI on deleted branch with later same-workflow success"


def classify_cleanup_run(
    run: dict[str, Any],
    *,
    protected_run_ids: set[int],
    artifact_run_ids: set[int],
    current_run_id: int,
) -> tuple[str, str]:
    run_id = int(run["id"])
    if run_id == current_run_id:
        return "KEEP", "current cleanup run"
    if run.get("status") != "completed":
        return "KEEP", "cleanup run is not completed"
    if run_id in protected_run_ids:
        return "KEEP", "cleanup run id is explicitly referenced"
    if run_id in artifact_run_ids:
        return "KEEP", "cleanup run has an artifact"
    return "DELETE", "superseded temporary cleanup-helper run"


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
                "User-Agent": "trade-rl-actions-history-cleanup-phase2",
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
        next_url: str | None = f"{path}{sep}per_page=100"
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
    proc = subprocess.run(
        ["git", "grep", "-I", "-h", "-Eo", r"[0-9]{10,12}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed: {proc.stderr}")
    return {int(x) for x in proc.stdout.splitlines() if x.isdigit()}


def _collect_text_refs(api: GitHubApi, repo_root: Path) -> set[int]:
    texts: list[str] = []
    for path in (
        f"/repos/{api.repo}/issues?state=all",
        f"/repos/{api.repo}/issues/comments",
        f"/repos/{api.repo}/pulls/comments",
        f"/repos/{api.repo}/comments",
    ):
        items = api.get_paginated(path)
        texts.extend(str(item.get("body") or "") for item in items)
    return _extract_run_ids(texts) | _repository_numeric_refs(repo_root)


def _failure_query(api: GitHubApi, start: date, end: date) -> str:
    created = urllib.parse.quote(f"{start.isoformat()}..{end.isoformat()}", safe=".")
    return f"/repos/{api.repo}/actions/runs?status=failure&created={created}"


def _collect_failure_range(api: GitHubApi, start: date, end: date) -> list[dict[str, Any]]:
    if end < start:
        return []
    path = _failure_query(api, start, end)
    meta = api.get(path + "&per_page=1")
    count = int(meta.get("total_count") or 0)
    if count == 0:
        return []
    if count <= 1000:
        runs = api.get_paginated(path, item_key="workflow_runs")
        if len(runs) != count:
            raise RuntimeError(
                f"Failure partition count mismatch for {start}..{end}: expected {count}, got {len(runs)}"
            )
        return runs
    if start == end:
        raise RuntimeError(
            f"Single-day failure partition exceeds GitHub's 1000-run search cap: {start} count={count}"
        )
    span_days = (end - start).days
    midpoint = start + timedelta(days=span_days // 2)
    left = _collect_failure_range(api, start, midpoint)
    right = _collect_failure_range(api, midpoint + timedelta(days=1), end)
    return left + right


def _latest_success_for_keys(
    api: GitHubApi, keys: set[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for branch, workflow_id in sorted(keys):
        if not branch or workflow_id <= 0:
            continue
        encoded_branch = urllib.parse.quote(branch, safe="")
        data = api.get(
            f"/repos/{api.repo}/actions/workflows/{workflow_id}/runs"
            f"?branch={encoded_branch}&status=success&per_page=1"
        )
        runs = data.get("workflow_runs") or []
        if runs:
            out[(branch, workflow_id)] = runs[0]
    return out


def collect_context(api: GitHubApi, repo_root: Path, now: datetime, recent_hours: int) -> dict[str, Any]:
    repo = api.get(f"/repos/{api.repo}")
    default_branch = str(repo["default_branch"])
    default_branch_data = api.get(
        f"/repos/{api.repo}/branches/{urllib.parse.quote(default_branch, safe='')}"
    )
    protected_shas = {str(default_branch_data["commit"]["sha"])}

    open_prs = api.get_paginated(f"/repos/{api.repo}/pulls?state=open")
    protected_shas.update(str(pr["head"]["sha"]) for pr in open_prs)

    branches = api.get_paginated(f"/repos/{api.repo}/branches")
    existing_branches = {str(branch["name"]) for branch in branches}

    artifacts = api.get_paginated(
        f"/repos/{api.repo}/actions/artifacts", item_key="artifacts"
    )
    artifact_run_ids = {
        int(artifact["workflow_run"]["id"])
        for artifact in artifacts
        if artifact.get("workflow_run") and artifact["workflow_run"].get("id")
    }

    repo_created = date.fromisoformat(str(repo["created_at"])[:10])
    cutoff = now - timedelta(hours=recent_hours)
    failure_runs = _collect_failure_range(api, repo_created, cutoff.date())

    protected_run_ids = _collect_text_refs(api, repo_root)
    failure_ids = {int(run["id"]) for run in failure_runs}
    protected_run_ids &= failure_ids

    success_keys: set[tuple[str, int]] = set()
    for run in failure_runs:
        branch = str(run.get("head_branch") or "")
        if (
            run.get("status") == "completed"
            and run.get("conclusion") == "failure"
            and _is_ci(run)
            and branch
            and branch not in existing_branches
            and not branch.startswith(CLEANUP_BRANCH_PREFIX)
            and str(run.get("head_sha") or "") not in protected_shas
            and int(run["id"]) not in protected_run_ids
            and int(run["id"]) not in artifact_run_ids
            and not _has_high_value_keyword(run)
            and _parse_time(str(run["created_at"])) < cutoff
        ):
            success_keys.add(_success_key(run))

    latest_success = _latest_success_for_keys(api, success_keys)

    cleanup_runs: list[dict[str, Any]] = []
    cleanup_branches = {
        os.environ.get("LEGACY_CLEANUP_BRANCH", ""),
        os.environ.get("CLEANUP_HEAD_BRANCH", ""),
    }
    for branch in sorted(b for b in cleanup_branches if b):
        data = api.get(
            f"/repos/{api.repo}/actions/runs?branch={urllib.parse.quote(branch, safe='')}&per_page=100"
        )
        cleanup_runs.extend(data.get("workflow_runs") or [])

    return {
        "protected_shas": protected_shas,
        "existing_branches": existing_branches,
        "artifact_run_ids": artifact_run_ids,
        "protected_run_ids": protected_run_ids,
        "failure_runs": failure_runs,
        "latest_success": latest_success,
        "success_key_count": len(success_keys),
        "cleanup_runs": cleanup_runs,
        "open_prs": [
            {"number": int(pr["number"]), "head_sha": str(pr["head"]["sha"])}
            for pr in open_prs
        ],
    }


def write_summary(report: dict[str, Any]) -> None:
    lines = [
        "# Actions history cleanup phase 2",
        "",
        f"- Mode: **{'DRY RUN' if report['dry_run'] else 'EXECUTE'}**",
        f"- Old failed runs scanned completely: **{report['failure_runs_scanned']}**",
        f"- Deleted-branch CI workflow keys checked for later success: **{report['success_key_count']}**",
        f"- Safe delete candidates: **{report['candidate_count']}**",
        f"- Deleted: **{report['deleted_count']}**",
        f"- REVIEW (not deleted): **{report['review_count']}**",
        f"- KEEP: **{report['keep_count']}**",
        f"- Protected explicit failure-run references: **{report['protected_reference_count']}**",
        f"- Artifact-bearing runs protected: **{report['artifact_run_count']}**",
        f"- Open PRs protected: **{len(report['open_prs'])}**",
        "",
        "## Delete reason counts",
        "",
    ]
    for key, value in sorted(report["delete_reason_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate samples", ""])
    for sample in report["candidate_samples"]:
        lines.append(
            f"- `{sample['id']}` — {sample['name']} — `{sample['head_branch']}` — "
            f"{sample['conclusion']} — {sample['reason']}"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    print("SUMMARY_JSON=" + json.dumps(report, sort_keys=True, separators=(",", ":")))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
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
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    max_delete = int(os.environ.get("MAX_DELETE", "1500"))
    recent_hours = int(os.environ.get("RECENT_HOURS", "168"))
    api = GitHubApi(repo, token)
    now = datetime.now(timezone.utc)
    context = collect_context(api, Path.cwd(), now, recent_hours)

    classified: list[tuple[dict[str, Any], str, str]] = []
    for run in context["failure_runs"]:
        decision, reason = classify_failure(
            run,
            now=now,
            protected_shas=context["protected_shas"],
            protected_run_ids=context["protected_run_ids"],
            artifact_run_ids=context["artifact_run_ids"],
            existing_branches=context["existing_branches"],
            latest_success=context["latest_success"],
            current_run_id=current_run_id,
            recent_hours=recent_hours,
        )
        classified.append((run, decision, reason))

    seen = {int(run["id"]) for run, _, _ in classified}
    for run in context["cleanup_runs"]:
        if int(run["id"]) in seen:
            continue
        decision, reason = classify_cleanup_run(
            run,
            protected_run_ids=context["protected_run_ids"],
            artifact_run_ids=context["artifact_run_ids"],
            current_run_id=current_run_id,
        )
        classified.append((run, decision, reason))
        seen.add(int(run["id"]))

    candidates = [(run, reason) for run, decision, reason in classified if decision == "DELETE"]
    candidates.sort(key=lambda pair: pair[0]["created_at"])
    selected = candidates[:max_delete]
    deleted: list[int] = []
    if not dry_run:
        for run, _ in selected:
            if api.rate_remaining is not None and api.rate_remaining < 250:
                raise RuntimeError(
                    "Stopping before rate-limit exhaustion; "
                    f"remaining={api.rate_remaining}, deleted={len(deleted)}"
                )
            api.delete_run(int(run["id"]))
            deleted.append(int(run["id"]))
            if len(deleted) % 250 == 0:
                print(f"Deleted {len(deleted)} runs...")

    counts = Counter(decision for _, decision, _ in classified)
    reasons = Counter(reason for _, decision, reason in classified if decision == "DELETE")
    report = {
        "dry_run": dry_run,
        "failure_runs_scanned": len(context["failure_runs"]),
        "success_key_count": context["success_key_count"],
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "deleted_count": len(deleted),
        "review_count": counts.get("REVIEW", 0),
        "keep_count": counts.get("KEEP", 0),
        "delete_reason_counts": dict(reasons),
        "protected_reference_count": len(context["protected_run_ids"]),
        "artifact_run_count": len(context["artifact_run_ids"]),
        "open_prs": context["open_prs"],
        "rate_remaining": api.rate_remaining,
        "candidate_samples": [
            {
                "id": int(run["id"]),
                "name": str(run.get("name") or ""),
                "head_branch": str(run.get("head_branch") or ""),
                "conclusion": str(run.get("conclusion") or ""),
                "created_at": str(run.get("created_at") or ""),
                "reason": reason,
            }
            for run, reason in selected[:50]
        ],
    }
    write_summary(report)
    return 0


def self_test() -> int:
    now = datetime(2026, 9, 12, tzinfo=timezone.utc)
    old = "2026-08-20T00:00:00Z"
    recent = "2026-09-11T00:00:00Z"

    def run(
        run_id: int,
        *,
        branch: str = "old/branch",
        name: str = "CI",
        path: str = ".github/workflows/ci.yml",
        title: str = "fix: routine change",
        created_at: str = old,
        sha: str = "oldsha",
        workflow_id: int = 123,
        conclusion: str = "failure",
    ) -> dict[str, Any]:
        return {
            "id": run_id,
            "status": "completed",
            "conclusion": conclusion,
            "head_branch": branch,
            "head_sha": sha,
            "path": path,
            "name": name,
            "display_title": title,
            "created_at": created_at,
            "workflow_id": workflow_id,
        }

    success = run(
        90000000002,
        created_at="2026-08-21T00:00:00Z",
        conclusion="success",
    )
    common = dict(
        now=now,
        protected_shas=set(),
        protected_run_ids=set(),
        artifact_run_ids=set(),
        existing_branches=set(),
        latest_success={_success_key(success): success},
        current_run_id=99999999999,
        recent_hours=168,
    )

    cases: list[tuple[str, dict[str, Any], dict[str, Any], str]] = [
        ("superseded failed CI", run(90000000001), {}, "DELETE"),
        ("no later success", run(90000000003, workflow_id=999), {}, "REVIEW"),
        ("recent failure", run(90000000004, created_at=recent), {}, "KEEP"),
        ("artifact failure", run(90000000005), {"artifact_run_ids": {90000000005}}, "KEEP"),
        ("referenced failure", run(90000000006), {"protected_run_ids": {90000000006}}, "KEEP"),
        ("protected SHA", run(90000000007, sha="protected"), {"protected_shas": {"protected"}}, "KEEP"),
        ("existing branch", run(90000000008), {"existing_branches": {"old/branch"}}, "KEEP"),
        ("non-CI failure", run(90000000009, name="Helper", path=".github/workflows/tmp.yml"), {}, "REVIEW"),
        ("research failure", run(90000000010, title="research: experiment red"), {}, "KEEP"),
        ("cleanup failure", run(90000000011, branch="ops/actions-history-cleanup-old", workflow_id=999), {}, "DELETE"),
    ]

    for label, item, overrides, expected in cases:
        kwargs = dict(common)
        kwargs.update(overrides)
        actual, reason = classify_failure(item, **kwargs)
        if actual != expected:
            raise AssertionError(f"{label}: expected {expected}, got {actual}: {reason}")

    cleanup = run(90000000012, branch="ops/actions-history-cleanup-old", conclusion="success")
    actual, _ = classify_cleanup_run(
        cleanup,
        protected_run_ids=set(),
        artifact_run_ids=set(),
        current_run_id=99999999999,
    )
    if actual != "DELETE":
        raise AssertionError(f"cleanup supersession: expected DELETE, got {actual}")

    print(f"self-test: {len(cases) + 1} selective phase-2 contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cleanup())
