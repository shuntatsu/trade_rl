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
TEMPORARY_RE = re.compile(r"(?:^|[/_. -])(?:tmp|temp|temporary)(?:[/_. -]|$)", re.IGNORECASE)
RUN_ID_RE = re.compile(r"(?<!\d)(\d{10,12})(?!\d)")
CLEANUP_BRANCH_PREFIX = "ops/actions-history-cleanup"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_ci(run: dict[str, Any]) -> bool:
    return run.get("name") == "CI" or str(run.get("path") or "") == ".github/workflows/ci.yml"


def _haystack(run: dict[str, Any]) -> str:
    return " ".join(
        str(run.get(key) or "")
        for key in ("name", "path", "head_branch", "display_title")
    )


def _has_high_value_keyword(run: dict[str, Any]) -> bool:
    return HIGH_VALUE_RE.search(_haystack(run)) is not None


def _is_explicit_temporary_helper(run: dict[str, Any]) -> bool:
    name = str(run.get("name") or "")
    path = str(run.get("path") or "")
    title = str(run.get("display_title") or "")
    basename = path.rsplit("/", 1)[-1]
    return (
        basename.startswith(("tmp-", "temp-", "temporary-"))
        or name.lower().startswith(("tmp ", "temp ", "temporary "))
        or TEMPORARY_RE.search(name) is not None
        or TEMPORARY_RE.search(title) is not None
    )


def classify_run(
    run: dict[str, Any],
    *,
    now: datetime,
    protected_shas: set[str],
    protected_run_ids: set[int],
    artifact_run_ids: set[int],
    existing_branches: set[str],
    current_run_id: int,
    recent_hours: int,
) -> tuple[str, str]:
    run_id = int(run["id"])
    if run_id == current_run_id:
        return "KEEP", "current cleanup run"
    if run.get("status") != "completed":
        return "KEEP", "run is not completed"
    conclusion = str(run.get("conclusion") or "")
    if conclusion not in {"cancelled", "stale", "skipped"}:
        return "KEEP", "conclusion is outside phase-3 scope"
    if run_id in protected_run_ids:
        return "KEEP", "run id is referenced in repository/issue/PR text"
    if run_id in artifact_run_ids:
        return "KEEP", "run has an Actions artifact"
    if str(run.get("head_sha") or "") in protected_shas:
        return "KEEP", "run belongs to current main or an open PR HEAD"
    if _has_high_value_keyword(run):
        return "KEEP", "research/evidence/TDD/high-value keyword"

    branch = str(run.get("head_branch") or "")
    if branch and branch in existing_branches:
        return "KEEP", "branch still exists"

    created_at = _parse_time(str(run["created_at"]))
    if created_at >= now - timedelta(hours=recent_hours):
        return "KEEP", f"created within last {recent_hours} hours"

    if branch.startswith(CLEANUP_BRANCH_PREFIX):
        return "DELETE", "obsolete cleanup-helper run"
    if conclusion in {"cancelled", "stale"} and _is_ci(run):
        return "DELETE", "old cancelled/stale CI on deleted branch"
    if _is_explicit_temporary_helper(run):
        return "DELETE", "old cancelled/stale/skipped explicit temporary helper on deleted branch"
    return "REVIEW", "old nuisance-conclusion run is not explicitly disposable"


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
                "User-Agent": "trade-rl-actions-history-cleanup-phase3",
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


def _status_query(api: GitHubApi, status: str, start: date, end: date) -> str:
    created = urllib.parse.quote(f"{start.isoformat()}..{end.isoformat()}", safe=".")
    return f"/repos/{api.repo}/actions/runs?status={status}&created={created}"


def _collect_status_range(
    api: GitHubApi, status: str, start: date, end: date
) -> list[dict[str, Any]]:
    if end < start:
        return []
    path = _status_query(api, status, start, end)
    meta = api.get(path + "&per_page=1")
    count = int(meta.get("total_count") or 0)
    if count == 0:
        return []
    if count <= 1000:
        runs = api.get_paginated(path, item_key="workflow_runs")
        if len(runs) != count:
            raise RuntimeError(
                f"{status} partition count mismatch for {start}..{end}: expected {count}, got {len(runs)}"
            )
        return runs
    if start == end:
        raise RuntimeError(
            f"Single-day {status} partition exceeds GitHub's 1000-run search cap: {start} count={count}"
        )
    span_days = (end - start).days
    midpoint = start + timedelta(days=span_days // 2)
    return _collect_status_range(api, status, start, midpoint) + _collect_status_range(
        api, status, midpoint + timedelta(days=1), end
    )


def _dedupe_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for run in runs:
        run_id = int(run["id"])
        if run_id not in seen:
            seen.add(run_id)
            out.append(run)
    return out


def collect_context(
    api: GitHubApi, repo_root: Path, now: datetime, recent_hours: int
) -> dict[str, Any]:
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

    cutoff = now - timedelta(hours=recent_hours)
    repo_created = date.fromisoformat(str(repo["created_at"])[:10])
    runs: list[dict[str, Any]] = []
    for status in ("cancelled", "stale", "skipped"):
        runs.extend(_collect_status_range(api, status, repo_created, cutoff.date()))
    runs = _dedupe_runs(runs)

    protected_run_ids = _collect_text_refs(api, repo_root)
    actual_ids = {int(run["id"]) for run in runs}
    protected_run_ids &= actual_ids

    return {
        "runs": runs,
        "protected_shas": protected_shas,
        "existing_branches": existing_branches,
        "artifact_run_ids": artifact_run_ids,
        "protected_run_ids": protected_run_ids,
        "open_prs": [
            {"number": int(pr["number"]), "head_sha": str(pr["head"]["sha"])}
            for pr in open_prs
        ],
    }


def write_summary(report: dict[str, Any]) -> None:
    lines = [
        "# Actions history cleanup phase 3",
        "",
        f"- Mode: **{'DRY RUN' if report['dry_run'] else 'EXECUTE'}**",
        f"- Old nuisance-conclusion runs scanned completely: **{report['runs_scanned']}**",
        f"- Safe delete candidates: **{report['candidate_count']}**",
        f"- Deleted: **{report['deleted_count']}**",
        f"- REVIEW (not deleted): **{report['review_count']}**",
        f"- KEEP: **{report['keep_count']}**",
        f"- Protected explicit references in scan: **{report['protected_reference_count']}**",
        f"- Artifact-bearing runs protected: **{report['artifact_run_count']}**",
        f"- Open PRs protected: **{len(report['open_prs'])}**",
        "",
        "## Delete reason counts",
        "",
    ]
    for key, value in sorted(report["delete_reason_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate samples", ""])
    for sample in report["candidate_samples"][:30]:
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


def self_test() -> int:
    now = datetime(2026, 9, 12, 5, 30, tzinfo=timezone.utc)
    old = "2026-09-01T00:00:00Z"
    recent = "2026-09-12T04:00:00Z"
    base = {
        "id": 12345678901,
        "name": "CI",
        "path": ".github/workflows/ci.yml",
        "head_branch": "deleted/branch",
        "head_sha": "old-sha",
        "display_title": "normal change",
        "status": "completed",
        "conclusion": "cancelled",
        "created_at": old,
    }
    common = dict(
        now=now,
        protected_shas={"main-sha", "pr-sha"},
        protected_run_ids=set(),
        artifact_run_ids=set(),
        existing_branches={"main", "live/branch"},
        current_run_id=99999999999,
        recent_hours=168,
    )

    cases: list[tuple[dict[str, Any], str]] = [
        (dict(base), "REVIEW"),
        ({**base, "conclusion": "failure"}, "KEEP"),
        ({**base, "conclusion": "success"}, "KEEP"),
        ({**base, "status": "in_progress"}, "KEEP"),
        ({**base, "head_branch": "live/branch"}, "KEEP"),
        ({**base, "head_sha": "main-sha"}, "KEEP"),
        ({**base, "created_at": recent}, "KEEP"),
        ({**base, "id": 22222222222}, "KEEP"),
        ({**base, "id": 33333333333}, "KEEP"),
        ({**base, "name": "Canonical research CI"}, "KEEP"),
        ({**base, "display_title": "TDD RED expected failure"}, "KEEP"),
        ({**base, "conclusion": "skipped"}, "REVIEW"),
        (
            {
                **base,
                "name": "Temporary apply helper",
                "path": ".github/workflows/tmp-apply.yml",
                "conclusion": "skipped",
            },
            "DELETE",
        ),
        (
            {
                **base,
                "name": "Temporary GREEN helper",
                "path": ".github/workflows/tmp-green.yml",
                "conclusion": "skipped",
            },
            "KEEP",
        ),
        (
            {**base, "head_branch": "ops/actions-history-cleanup-old", "name": "cleanup"},
            "DELETE",
        ),
    ]
    for index, (run, expected) in enumerate(cases):
        kwargs = dict(common)
        if index == 7:
            kwargs["protected_run_ids"] = {22222222222}
        if index == 8:
            kwargs["artifact_run_ids"] = {33333333333}
        decision, _ = classify_run(run, **kwargs)
        assert decision == expected, (index, decision, expected)
    assert _is_explicit_temporary_helper(
        {
            "name": "worker",
            "path": ".github/workflows/tmp-worker.yml",
            "display_title": "x",
        }
    )
    assert not _is_explicit_temporary_helper(
        {"name": "CI", "path": ".github/workflows/ci.yml", "display_title": "x"}
    )
    print("self-test: 17 phase-3 classification contracts passed")
    return 0


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
    max_delete = int(os.environ.get("MAX_DELETE", "3000"))
    recent_hours = int(os.environ.get("RECENT_HOURS", "168"))
    api = GitHubApi(repo, token)
    now = datetime.now(timezone.utc)
    context = collect_context(api, Path.cwd(), now, recent_hours)

    classified: list[tuple[dict[str, Any], str, str]] = []
    for run in context["runs"]:
        decision, reason = classify_run(
            run,
            now=now,
            protected_shas=context["protected_shas"],
            protected_run_ids=context["protected_run_ids"],
            artifact_run_ids=context["artifact_run_ids"],
            existing_branches=context["existing_branches"],
            current_run_id=current_run_id,
            recent_hours=recent_hours,
        )
        classified.append((run, decision, reason))

    candidates = [(run, reason) for run, decision, reason in classified if decision == "DELETE"]
    candidates.sort(key=lambda pair: pair[0]["created_at"])
    selected = candidates[:max_delete]
    deleted: list[int] = []
    if not dry_run:
        for run, _ in selected:
            if api.rate_remaining is not None and api.rate_remaining < 300:
                raise RuntimeError(
                    "Stopping before rate-limit exhaustion; "
                    f"remaining={api.rate_remaining}, deleted={len(deleted)}"
                )
            api.delete_run(int(run["id"]))
            deleted.append(int(run["id"]))
            if len(deleted) % 250 == 0:
                print(f"Deleted {len(deleted)} runs...")

    decision_counts = Counter(decision for _, decision, _ in classified)
    delete_reason_counts = Counter(
        reason for _, decision, reason in classified if decision == "DELETE"
    )
    report = {
        "dry_run": dry_run,
        "runs_scanned": len(context["runs"]),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "deleted_count": len(deleted),
        "review_count": decision_counts.get("REVIEW", 0),
        "keep_count": decision_counts.get("KEEP", 0),
        "protected_reference_count": len(context["protected_run_ids"]),
        "artifact_run_count": len(context["artifact_run_ids"]),
        "open_prs": context["open_prs"],
        "delete_reason_counts": dict(delete_reason_counts),
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


if __name__ == "__main__":
    raise SystemExit(run_cleanup())
