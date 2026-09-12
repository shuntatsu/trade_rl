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


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def is_ci(run: dict[str, Any]) -> bool:
    return run.get("name") == "CI" or str(run.get("path") or "") == ".github/workflows/ci.yml"


def key(run: dict[str, Any]) -> tuple[str, int]:
    return str(run.get("head_branch") or ""), int(run.get("workflow_id") or 0)


def high_value(run: dict[str, Any]) -> bool:
    text = " ".join(str(run.get(k) or "") for k in ("name", "path", "head_branch", "display_title"))
    return HIGH_VALUE_RE.search(text) is not None


def classify(
    run: dict[str, Any],
    *,
    now: datetime,
    protected_shas: set[str],
    protected_run_ids: set[int],
    artifact_run_ids: set[int],
    existing_branches: set[str],
    later_success: dict[tuple[str, int], dict[str, Any]],
    current_run_id: int,
    recent_hours: int,
) -> tuple[str, str]:
    run_id = int(run["id"])
    if run_id == current_run_id:
        return "KEEP", "current cleanup run"
    if run.get("status") != "completed" or run.get("conclusion") != "cancelled":
        return "KEEP", "not a completed cancelled run"
    if run_id in protected_run_ids:
        return "KEEP", "run id is explicitly referenced"
    if run_id in artifact_run_ids:
        return "KEEP", "run has an Actions artifact"
    if str(run.get("head_sha") or "") in protected_shas:
        return "KEEP", "run belongs to current main or an open PR HEAD"
    if high_value(run):
        return "KEEP", "research/evidence/high-value keyword"
    if not is_ci(run):
        return "REVIEW", "cancelled non-CI workflow is not deleted in phase 3"

    branch = str(run.get("head_branch") or "")
    if branch and branch in existing_branches:
        return "KEEP", "branch still exists"
    created_at = parse_time(str(run["created_at"]))
    if created_at >= now - timedelta(hours=recent_hours):
        return "KEEP", f"created within last {recent_hours} hours"

    success = later_success.get(key(run))
    if success is None:
        return "REVIEW", "no later success for same branch/workflow"
    if parse_time(str(success["created_at"])) <= created_at:
        return "REVIEW", "same-workflow success is not later than cancellation"
    return "DELETE", "superseded cancelled CI on deleted branch with later same-workflow success"


class GitHubApi:
    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token
        self.base = "https://api.github.com"
        self.rate_remaining: int | None = None

    def request(self, method: str, path: str) -> tuple[Any, dict[str, str]]:
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
                return (json.loads(body.decode("utf-8")) if body else None), headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {url} failed: {exc.code} {detail}") from exc

    def get(self, path: str) -> Any:
        data, _ = self.request("GET", path)
        return data

    def paginated(self, path: str, item_key: str | None = None) -> list[Any]:
        sep = "&" if "?" in path else "?"
        url: str | None = f"{path}{sep}per_page=100"
        out: list[Any] = []
        while url:
            data, headers = self.request("GET", url)
            items = data[item_key] if item_key else data
            if not isinstance(items, list):
                raise RuntimeError(f"Expected list from {url}")
            out.extend(items)
            url = next_link(headers.get("link", ""))
        return out

    def delete_run(self, run_id: int) -> None:
        self.request("DELETE", f"/repos/{self.repo}/actions/runs/{run_id}")


def next_link(header: str) -> str | None:
    for part in header.split(","):
        section = part.strip()
        if 'rel="next"' in section:
            return section.split(";", 1)[0].strip()[1:-1]
    return None


def extract_ids(texts: Iterable[str]) -> set[int]:
    out: set[int] = set()
    for text in texts:
        out.update(int(x) for x in RUN_ID_RE.findall(text or ""))
    return out


def repository_refs(root: Path) -> set[int]:
    proc = subprocess.run(
        ["git", "grep", "-I", "-h", "-Eo", r"[0-9]{10,12}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr)
    return {int(x) for x in proc.stdout.splitlines() if x.isdigit()}


def collect_refs(api: GitHubApi, root: Path) -> set[int]:
    texts: list[str] = []
    for path in (
        f"/repos/{api.repo}/issues?state=all",
        f"/repos/{api.repo}/issues/comments",
        f"/repos/{api.repo}/pulls/comments",
        f"/repos/{api.repo}/comments",
    ):
        for item in api.paginated(path):
            texts.append(str(item.get("body") or ""))
    return extract_ids(texts) | repository_refs(root)


def status_query(api: GitHubApi, start: date, end: date) -> str:
    created = urllib.parse.quote(f"{start.isoformat()}..{end.isoformat()}", safe=".")
    return f"/repos/{api.repo}/actions/runs?status=cancelled&created={created}"


def collect_cancelled_range(api: GitHubApi, start: date, end: date) -> list[dict[str, Any]]:
    if end < start:
        return []
    path = status_query(api, start, end)
    meta = api.get(path + "&per_page=1")
    count = int(meta.get("total_count") or 0)
    if count == 0:
        return []
    if count <= 1000:
        runs = api.paginated(path, item_key="workflow_runs")
        if len(runs) != count:
            raise RuntimeError(
                f"Cancelled partition mismatch {start}..{end}: expected {count}, got {len(runs)}"
            )
        return runs
    if start == end:
        raise RuntimeError(f"Single-day cancelled partition exceeds 1000: {start} count={count}")
    midpoint = start + timedelta(days=(end - start).days // 2)
    return collect_cancelled_range(api, start, midpoint) + collect_cancelled_range(
        api, midpoint + timedelta(days=1), end
    )


def latest_successes(api: GitHubApi, keys: set[tuple[str, int]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for branch, workflow_id in sorted(keys):
        if not branch or workflow_id <= 0:
            continue
        encoded = urllib.parse.quote(branch, safe="")
        data = api.get(
            f"/repos/{api.repo}/actions/workflows/{workflow_id}/runs"
            f"?branch={encoded}&status=success&per_page=1"
        )
        runs = data.get("workflow_runs") or []
        if runs:
            out[(branch, workflow_id)] = runs[0]
    return out


def context(api: GitHubApi, root: Path, now: datetime, recent_hours: int) -> dict[str, Any]:
    repo = api.get(f"/repos/{api.repo}")
    default_branch = str(repo["default_branch"])
    main = api.get(f"/repos/{api.repo}/branches/{urllib.parse.quote(default_branch, safe='')}")
    protected_shas = {str(main["commit"]["sha"])}
    open_prs = api.paginated(f"/repos/{api.repo}/pulls?state=open")
    protected_shas.update(str(pr["head"]["sha"]) for pr in open_prs)
    branches = api.paginated(f"/repos/{api.repo}/branches")
    existing_branches = {str(b["name"]) for b in branches}
    artifacts = api.paginated(f"/repos/{api.repo}/actions/artifacts", item_key="artifacts")
    artifact_run_ids = {
        int(a["workflow_run"]["id"])
        for a in artifacts
        if a.get("workflow_run") and a["workflow_run"].get("id")
    }

    cutoff = now - timedelta(hours=recent_hours)
    repo_created = date.fromisoformat(str(repo["created_at"])[:10])
    cancelled = collect_cancelled_range(api, repo_created, cutoff.date())
    cancelled_ids = {int(r["id"]) for r in cancelled}
    protected_run_ids = collect_refs(api, root) & cancelled_ids

    candidate_keys: set[tuple[str, int]] = set()
    for run in cancelled:
        branch = str(run.get("head_branch") or "")
        if (
            run.get("status") == "completed"
            and run.get("conclusion") == "cancelled"
            and is_ci(run)
            and branch
            and branch not in existing_branches
            and str(run.get("head_sha") or "") not in protected_shas
            and int(run["id"]) not in protected_run_ids
            and int(run["id"]) not in artifact_run_ids
            and not high_value(run)
            and parse_time(str(run["created_at"])) < cutoff
        ):
            candidate_keys.add(key(run))

    return {
        "cancelled": cancelled,
        "later_success": latest_successes(api, candidate_keys),
        "candidate_key_count": len(candidate_keys),
        "protected_shas": protected_shas,
        "protected_run_ids": protected_run_ids,
        "artifact_run_ids": artifact_run_ids,
        "existing_branches": existing_branches,
        "open_prs": [{"number": int(p["number"]), "head_sha": str(p["head"]["sha"])} for p in open_prs],
    }


def report(data: dict[str, Any]) -> None:
    lines = [
        "# Actions history cleanup phase 3",
        "",
        f"- Mode: **{'DRY RUN' if data['dry_run'] else 'EXECUTE'}**",
        f"- Old cancelled runs scanned completely: **{data['scanned']}**",
        f"- Deleted-branch CI keys checked for later success: **{data['candidate_key_count']}**",
        f"- Safe delete candidates: **{data['candidate_count']}**",
        f"- Deleted: **{data['deleted_count']}**",
        f"- REVIEW: **{data['review_count']}**",
        f"- KEEP: **{data['keep_count']}**",
        f"- Explicit references protected: **{data['protected_reference_count']}**",
        f"- Artifact-bearing runs protected: **{data['artifact_run_count']}**",
        f"- Open PRs protected: **{len(data['open_prs'])}**",
        "",
        "## Candidate samples",
        "",
    ]
    for s in data["candidate_samples"]:
        lines.append(
            f"- `{s['id']}` — {s['name']} — `{s['head_branch']}` — {s['reason']}"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    print("SUMMARY_JSON=" + json.dumps(data, sort_keys=True, separators=(",", ":")))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(text, encoding="utf-8")


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    api = GitHubApi(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    now = datetime.now(timezone.utc)
    recent_hours = int(os.environ.get("RECENT_HOURS", "168"))
    current_run_id = int(os.environ["GITHUB_RUN_ID"])
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    max_delete = int(os.environ.get("MAX_DELETE", "1500"))
    ctx = context(api, Path.cwd(), now, recent_hours)

    classified: list[tuple[dict[str, Any], str, str]] = []
    for item in ctx["cancelled"]:
        decision, reason = classify(
            item,
            now=now,
            protected_shas=ctx["protected_shas"],
            protected_run_ids=ctx["protected_run_ids"],
            artifact_run_ids=ctx["artifact_run_ids"],
            existing_branches=ctx["existing_branches"],
            later_success=ctx["later_success"],
            current_run_id=current_run_id,
            recent_hours=recent_hours,
        )
        classified.append((item, decision, reason))

    candidates = [(r, reason) for r, d, reason in classified if d == "DELETE"]
    candidates.sort(key=lambda x: x[0]["created_at"])
    selected = candidates[:max_delete]
    deleted: list[int] = []
    if not dry_run:
        for item, _ in selected:
            if api.rate_remaining is not None and api.rate_remaining < 250:
                raise RuntimeError(
                    f"Stopping before rate-limit exhaustion: remaining={api.rate_remaining}, deleted={len(deleted)}"
                )
            api.delete_run(int(item["id"]))
            deleted.append(int(item["id"]))
            if len(deleted) % 250 == 0:
                print(f"Deleted {len(deleted)} runs...")

    counts = Counter(d for _, d, _ in classified)
    data = {
        "dry_run": dry_run,
        "scanned": len(ctx["cancelled"]),
        "candidate_key_count": ctx["candidate_key_count"],
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "deleted_count": len(deleted),
        "review_count": counts.get("REVIEW", 0),
        "keep_count": counts.get("KEEP", 0),
        "protected_reference_count": len(ctx["protected_run_ids"]),
        "artifact_run_count": len(ctx["artifact_run_ids"]),
        "open_prs": ctx["open_prs"],
        "rate_remaining": api.rate_remaining,
        "candidate_samples": [
            {
                "id": int(r["id"]),
                "name": str(r.get("name") or ""),
                "head_branch": str(r.get("head_branch") or ""),
                "created_at": str(r.get("created_at") or ""),
                "reason": reason,
            }
            for r, reason in selected[:50]
        ],
    }
    report(data)
    return 0


def self_test() -> int:
    now = datetime(2026, 9, 12, tzinfo=timezone.utc)
    old = "2026-08-20T00:00:00Z"
    recent = "2026-09-11T00:00:00Z"

    def item(run_id: int, **kw: Any) -> dict[str, Any]:
        base = {
            "id": run_id,
            "status": "completed",
            "conclusion": "cancelled",
            "head_branch": "old/branch",
            "head_sha": "oldsha",
            "path": ".github/workflows/ci.yml",
            "name": "CI",
            "display_title": "fix: routine change",
            "created_at": old,
            "workflow_id": 123,
        }
        base.update(kw)
        return base

    success = item(90000000002, conclusion="success", created_at="2026-08-21T00:00:00Z")
    common = {
        "now": now,
        "protected_shas": set(),
        "protected_run_ids": set(),
        "artifact_run_ids": set(),
        "existing_branches": set(),
        "later_success": {key(success): success},
        "current_run_id": 99999999999,
        "recent_hours": 168,
    }
    cases = [
        ("superseded cancellation", item(90000000001), {}, "DELETE"),
        ("no success", item(90000000003, workflow_id=999), {}, "REVIEW"),
        ("recent", item(90000000004, created_at=recent), {}, "KEEP"),
        ("artifact", item(90000000005), {"artifact_run_ids": {90000000005}}, "KEEP"),
        ("reference", item(90000000006), {"protected_run_ids": {90000000006}}, "KEEP"),
        ("protected sha", item(90000000007, head_sha="protected"), {"protected_shas": {"protected"}}, "KEEP"),
        ("existing branch", item(90000000008), {"existing_branches": {"old/branch"}}, "KEEP"),
        ("non CI", item(90000000009, name="Helper", path=".github/workflows/helper.yml"), {}, "REVIEW"),
        ("high value", item(90000000010, display_title="research: canonical experiment"), {}, "KEEP"),
    ]
    for label, run_item, overrides, expected in cases:
        kwargs = dict(common)
        kwargs.update(overrides)
        actual, reason = classify(run_item, **kwargs)
        if actual != expected:
            raise AssertionError(f"{label}: expected {expected}, got {actual}: {reason}")
    print(f"self-test: {len(cases)} cancelled-CI classification contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
