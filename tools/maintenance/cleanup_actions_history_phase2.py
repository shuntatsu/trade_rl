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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

HIGH_VALUE_RE = re.compile(
    r"(?:research|experiment|evidence|baseline|pre[-_ ]?reg|prereg|canonical|"
    r"repro|benchmark|publication|artifact|audit|falsif|mutation|release|provenance|"
    r"golden|verification|verify|\bRED\b|\bGREEN\b|primary[-_ ]?gate)",
    re.IGNORECASE,
)
RUN_ID_RE = re.compile(r"(?<!\d)(\d{10,12})(?!\d)")
TEMP_WORKFLOW_RE = re.compile(
    r"(?:^|/)(?:tmp[-_]|temporary[-_]|scratch[-_]|one[-_]?shot[-_]|oneoff[-_])",
    re.IGNORECASE,
)
CLEANUP_BRANCH_PREFIX = "ops/actions-history-cleanup"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_ci(run: dict[str, Any]) -> bool:
    path = str(run.get("path") or "")
    return run.get("name") == "CI" or path == ".github/workflows/ci.yml"


def _run_key(run: dict[str, Any]) -> tuple[str, str]:
    return (str(run.get("head_branch") or ""), str(run.get("path") or ""))


def _latest_by_key(runs: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        key = _run_key(run)
        previous = latest.get(key)
        if previous is None or _parse_time(str(run["created_at"])) > _parse_time(
            str(previous["created_at"])
        ):
            latest[key] = run
    return latest


def classify_run(
    run: dict[str, Any],
    *,
    now: datetime,
    protected_shas: set[str],
    protected_run_ids: set[int],
    artifact_run_ids: set[int],
    existing_branches: set[str],
    latest_by_key: dict[tuple[str, str], dict[str, Any]],
    current_run_id: int,
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
    if str(run.get("head_sha") or "") in protected_shas:
        return "KEEP", "run belongs to current main or an open PR HEAD"

    haystack = " ".join(
        str(run.get(key) or "")
        for key in ("name", "path", "head_branch", "display_title")
    )
    if HIGH_VALUE_RE.search(haystack):
        return "KEEP", "research/evidence/high-value keyword"

    branch = str(run.get("head_branch") or "")
    if branch and branch in existing_branches:
        return "KEEP", "branch still exists"

    created_at = _parse_time(str(run["created_at"]))
    if created_at >= now - timedelta(hours=recent_hours):
        return "KEEP", f"created within last {recent_hours} hours"

    if branch.startswith(CLEANUP_BRANCH_PREFIX):
        return "DELETE", "obsolete cleanup-helper run on deleted branch"

    conclusion = str(run.get("conclusion") or "")
    path = str(run.get("path") or "")

    if conclusion == "failure" and _is_ci(run):
        latest = latest_by_key.get(_run_key(run))
        if (
            latest is not None
            and int(latest["id"]) != run_id
            and latest.get("status") == "completed"
            and latest.get("conclusion") == "success"
            and _parse_time(str(latest["created_at"])) > created_at
        ):
            return "DELETE", "superseded failed CI on deleted branch with later success"
        return "REVIEW", "failed CI lacks a later successful terminal run on the same branch/workflow"

    if conclusion in {"cancelled", "stale", "skipped"} and TEMP_WORKFLOW_RE.search(path):
        return "DELETE", "old disposable temporary-workflow run on deleted branch"

    return "REVIEW", "old run is not provably disposable under phase-2 policy"


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


def collect_context(api: GitHubApi, repo_root: Path) -> dict[str, Any]:
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

    all_runs = api.get_paginated(
        f"/repos/{api.repo}/actions/runs", item_key="workflow_runs"
    )
    referenced_run_ids = _collect_text_refs(api, repo_root)
    actual_run_ids = {int(run["id"]) for run in all_runs}
    referenced_run_ids &= actual_run_ids

    return {
        "default_branch": default_branch,
        "protected_shas": protected_shas,
        "existing_branches": existing_branches,
        "artifact_run_ids": artifact_run_ids,
        "protected_run_ids": referenced_run_ids,
        "all_runs": all_runs,
        "latest_by_key": _latest_by_key(all_runs),
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
        f"- Repository total runs scanned: **{report['total_runs']}**",
        f"- Safe delete candidates: **{report['candidate_count']}**",
        f"- Deleted: **{report['deleted_count']}**",
        f"- REVIEW (not deleted): **{report['review_count']}**",
        f"- KEEP: **{report['keep_count']}**",
        f"- Protected explicit references: **{report['protected_reference_count']}**",
        f"- Runs with artifacts protected: **{report['artifact_run_count']}**",
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
    context = collect_context(api, Path.cwd())
    now = datetime.now(timezone.utc)

    classified: list[tuple[dict[str, Any], str, str]] = []
    for run in context["all_runs"]:
        decision, reason = classify_run(
            run,
            now=now,
            protected_shas=context["protected_shas"],
            protected_run_ids=context["protected_run_ids"],
            artifact_run_ids=context["artifact_run_ids"],
            existing_branches=context["existing_branches"],
            latest_by_key=context["latest_by_key"],
            current_run_id=current_run_id,
            recent_hours=recent_hours,
        )
        classified.append((run, decision, reason))

    candidates = [
        (run, reason)
        for run, decision, reason in classified
        if decision == "DELETE"
    ]
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
        "total_runs": len(context["all_runs"]),
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
        path: str = ".github/workflows/ci.yml",
        name: str = "CI",
        conclusion: str = "failure",
        created_at: str = old,
        sha: str = "oldsha",
        title: str = "change",
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
        }

    later_success = run(
        90000000002,
        conclusion="success",
        created_at="2026-08-21T00:00:00Z",
    )
    latest = _latest_by_key([run(90000000001), later_success])

    common = dict(
        now=now,
        protected_shas=set(),
        protected_run_ids=set(),
        artifact_run_ids=set(),
        existing_branches=set(),
        latest_by_key=latest,
        current_run_id=99999999999,
        recent_hours=168,
    )

    cases: list[tuple[str, dict[str, Any], dict[str, Any], str]] = [
        ("superseded failed CI", run(90000000001), {}, "DELETE"),
        ("failure without later success", run(90000000003), {"latest_by_key": {}}, "REVIEW"),
        ("recent failure", run(90000000004, created_at=recent), {"latest_by_key": {}}, "KEEP"),
        ("artifact failure", run(90000000005), {"artifact_run_ids": {90000000005}}, "KEEP"),
        ("referenced failure", run(90000000006), {"protected_run_ids": {90000000006}}, "KEEP"),
        ("protected SHA", run(90000000007, sha="protected"), {"protected_shas": {"protected"}}, "KEEP"),
        ("existing branch", run(90000000008), {"existing_branches": {"old/branch"}}, "KEEP"),
        (
            "old tmp skipped",
            run(90000000009, path=".github/workflows/tmp-helper.yml", name="Helper", conclusion="skipped"),
            {"latest_by_key": {}},
            "DELETE",
        ),
        (
            "named skipped non-temp",
            run(90000000010, path=".github/workflows/u2-primary.yml", name="U2 Primary Gate", conclusion="skipped"),
            {"latest_by_key": {}},
            "KEEP",
        ),
        (
            "high-value tmp skipped",
            run(90000000011, path=".github/workflows/tmp-helper.yml", name="Canonical Evidence Helper", conclusion="skipped"),
            {"latest_by_key": {}},
            "KEEP",
        ),
        (
            "old cleanup helper",
            run(90000000012, branch="ops/actions-history-cleanup-old", conclusion="failure"),
            {"latest_by_key": {}},
            "DELETE",
        ),
        (
            "research failure",
            run(90000000013, name="Research experiment CI"),
            {"latest_by_key": {}},
            "KEEP",
        ),
        (
            "old tmp cancelled",
            run(90000000014, path=".github/workflows/temporary-helper.yml", name="Helper", conclusion="cancelled"),
            {"latest_by_key": {}},
            "DELETE",
        ),
    ]

    for label, item, overrides, expected in cases:
        kwargs = dict(common)
        kwargs.update(overrides)
        actual, reason = classify_run(item, **kwargs)
        if actual != expected:
            raise AssertionError(f"{label}: expected {expected}, got {actual}: {reason}")

    print(f"self-test: {len(cases)} phase-2 classification contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cleanup())
