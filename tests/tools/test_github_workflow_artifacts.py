"""Refusal oracles for GET-only run-scoped Actions artifact observations."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.github_workflow_artifacts import observe_workflow_artifacts

RUN = 41427295732
REPO_ID = 1103009698
HEAD_ID = 1103009698
SHA = "a" * 40
BRANCH = "feature/a"
REPOSITORY = "shuntatsu/trade_rl"


def _artifact(artifact_id: int) -> dict[str, object]:
    return {
        "id": artifact_id,
        "node_id": f"A_{artifact_id}",
        "name": "repeated-name",  # legitimate repeated names across attempts
        "size_in_bytes": 128,
        "expired": False,
        "digest": "sha256:" + "c" * 64,
        "workflow_run": {
            "id": RUN,
            "repository_id": REPO_ID,
            "head_repository_id": HEAD_ID,
            "head_branch": BRANCH,
            "head_sha": SHA,
        },
    }


class FakeGet:
    def __init__(self, count: int = 2) -> None:
        self.repo = {"id": REPO_ID, "node_id": "R_123", "full_name": REPOSITORY}
        self.run = {
            "id": RUN,
            "run_attempt": 3,
            "head_branch": BRANCH,
            "head_sha": SHA,
            "status": "completed",
            "conclusion": "success",
            "updated_at": "2026-11-02T10:30:39Z",
            "repository": {"id": REPO_ID},
            "head_repository": {"id": HEAD_ID},
        }
        self.artifacts = [_artifact(i) for i in range(1, count + 1)]
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []
        self.repo_reads = 0
        self.run_reads = 0
        self.scans = 0
        self.next_artifacts: list[dict[str, object]] | None = None
        self.next_run: dict[str, object] | None = None
        self.next_repo: dict[str, object] | None = None
        self.fail_scan: int | None = None
        self.fake_total: int | None = None
        self.second_page_total: int | None = None

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        assert method == "GET"
        self.calls.append((method, suffix, query))
        if suffix == "":
            self.repo_reads += 1
            return deepcopy(
                self.next_repo
                if self.repo_reads > 1 and self.next_repo is not None
                else self.repo
            )
        if suffix == f"actions/runs/{RUN}":
            self.run_reads += 1
            return deepcopy(
                self.next_run
                if self.run_reads > 1 and self.next_run is not None
                else self.run
            )
        if suffix == f"actions/runs/{RUN}/artifacts":
            assert query is not None and query["per_page"] == "100"
            page = int(query["page"])
            if page == 1:
                self.scans += 1
            if self.fail_scan == self.scans:
                raise RuntimeError("GitHub artifacts API unavailable")
            artifacts = (
                self.next_artifacts
                if self.scans > 1 and self.next_artifacts is not None
                else self.artifacts
            )
            total = len(artifacts) if self.fake_total is None else self.fake_total
            if page > 1 and self.second_page_total is not None:
                total = self.second_page_total
            return {
                "total_count": total,
                "artifacts": deepcopy(artifacts[(page - 1) * 100 : page * 100]),
            }
        raise AssertionError(f"unexpected GET endpoint {suffix}")


def _observe(api: FakeGet) -> dict[str, object]:
    return observe_workflow_artifacts(api, expected_repository=REPOSITORY, run_id=RUN)


def test_stable_artifact_roster_is_only_run_scoped_and_never_authoritative() -> None:
    api = FakeGet()
    report = _observe(api)
    assert report["schema"] == "github_workflow_artifact_observation_v1"
    assert report["run_id"] == RUN
    assert report["run_attempt_observed"] == 3
    assert report["head_branch"] == BRANCH
    assert report["head_sha"] == SHA
    assert report["artifact_count"] == 2
    assert [a["id"] for a in report["artifacts"]] == [1, 2]
    assert [a["name"] for a in report["artifacts"]] == ["repeated-name"] * 2
    assert report["artifact_metadata_double_read"] is True
    for flag in (
        "artifact_attempt_binding_verified",
        "artifact_bytes_verified",
        "artifact_history_complete",
        "workflow_activity_checked",
        "task_registry_complete",
        "lease_reassignment_authorized",
        "cleanup_authorized",
    ):
        assert report[flag] is False
    assert api.repo_reads == 2
    assert api.run_reads == 2
    assert api.scans == 2
    assert len(api.calls) == 6


def test_zero_artifacts_is_an_observation_not_proof_of_historical_absence() -> None:
    api = FakeGet(0)
    result = _observe(api)
    assert result["artifact_count"] == 0
    assert result["artifacts"] == []
    assert result["artifact_history_complete"] is False
    assert api.scans == 2


def test_complete_multi_page_enumeration_and_sorted_ids() -> None:
    api = FakeGet(101)
    result = _observe(api)
    assert result["artifact_count"] == 101
    assert result["artifacts"][100]["id"] == 101
    assert api.scans == 2
    assert len(api.calls) == 8


@pytest.mark.parametrize("bad", [0, True, -1, "1", 1.0])
def test_invalid_run_id_never_starts_a_read(bad: object) -> None:
    api = FakeGet()
    invalid_run: int = bad  # type: ignore[assignment]
    with pytest.raises(ValueError, match="run_id"):
        observe_workflow_artifacts(
            api, expected_repository=REPOSITORY, run_id=invalid_run
        )
    assert api.calls == []


@pytest.mark.parametrize("bad", [None, "bad-repo", "owner/", "owner/a/b"])
def test_invalid_repository_never_starts_a_read(bad: object) -> None:
    api = FakeGet()
    invalid_repo: str = bad  # type: ignore[assignment]
    with pytest.raises(ValueError, match="repository"):
        observe_workflow_artifacts(
            api, expected_repository=invalid_repo, run_id=RUN
        )
    assert api.calls == []


def test_wrong_repository_identity_and_drift_refuse() -> None:
    api = FakeGet()
    api.repo["full_name"] = "other/repository"
    with pytest.raises(ValueError, match="repository identity"):
        _observe(api)

    api = FakeGet()
    api.next_repo = {**api.repo, "node_id": "R_456"}
    with pytest.raises(ValueError, match="repository changed"):
        _observe(api)


def test_run_attempt_or_status_change_during_scan_refuses() -> None:
    api = FakeGet()
    api.next_run = {**api.run, "run_attempt": 4}
    with pytest.raises(ValueError, match="run changed"):
        _observe(api)
    api = FakeGet()
    api.next_run = {**api.run, "status": "in_progress", "conclusion": None}
    with pytest.raises(ValueError, match="run changed"):
        _observe(api)


def test_wrong_run_repository_or_head_repo_identity_refuses() -> None:
    api = FakeGet()
    api.run["repository"] = {"id": 999}
    with pytest.raises(ValueError, match="run repository"):
        _observe(api)
    api = FakeGet()
    api.artifacts[0]["workflow_run"]["head_repository_id"] = 999  # type: ignore[index]
    with pytest.raises(ValueError, match="artifact workflow identity"):
        _observe(api)


def test_mismatched_artifact_run_or_head_or_branch_refuses() -> None:
    for key, bad in (("id", RUN + 1), ("head_sha", "b" * 40), ("head_branch", "wrong")):
        api = FakeGet()
        api.artifacts[0]["workflow_run"][key] = bad  # type: ignore[index]
        with pytest.raises(ValueError, match="artifact workflow identity"):
            _observe(api)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("id", True),
        ("id", -1),
        ("node_id", " "),
        ("name", ""),
        ("size_in_bytes", -1),
        ("size_in_bytes", False),
        ("expired", "false"),
        ("digest", "sha256:not-a-digest"),
        ("workflow_run", None),
    ],
)
def test_malformed_artifact_metadata_refuses(key: str, bad: object) -> None:
    api = FakeGet()
    api.artifacts[0][key] = bad
    with pytest.raises(ValueError):
        _observe(api)


def test_missing_digest_is_explicitly_not_verified_and_expired_is_retained() -> None:
    api = FakeGet()
    api.artifacts[0]["digest"] = None
    api.artifacts[1]["expired"] = True
    result = _observe(api)
    assert result["artifacts"][0]["digest"] is None
    assert result["artifacts"][1]["expired"] is True
    assert result["artifact_bytes_verified"] is False


def test_duplicate_artifact_ids_refuse_instead_of_deduplicating() -> None:
    api = FakeGet()
    api.artifacts[1]["id"] = api.artifacts[0]["id"]
    with pytest.raises(ValueError, match="duplicate artifact id"):
        _observe(api)


def test_empty_or_partial_page_and_total_drift_refuse() -> None:
    api = FakeGet()
    api.fake_total = 3
    with pytest.raises(ValueError, match="pagination"):
        _observe(api)
    api = FakeGet(0)
    api.fake_total = 1
    with pytest.raises(ValueError, match="pagination"):
        _observe(api)
    api = FakeGet(101)
    api.second_page_total = 100
    with pytest.raises(ValueError, match="total_count changed"):
        _observe(api)


def test_overlimit_total_count_refuses_before_pagination() -> None:
    api = FakeGet()
    api.fake_total = 10_001
    with pytest.raises(ValueError, match="total_count"):
        _observe(api)


def test_artifact_added_removed_or_expired_midscan_refuses() -> None:
    api = FakeGet()
    api.next_artifacts = [*deepcopy(api.artifacts), _artifact(3)]
    with pytest.raises(ValueError, match="artifact roster changed"):
        _observe(api)
    api = FakeGet()
    api.next_artifacts = deepcopy(api.artifacts[:1])
    with pytest.raises(ValueError, match="artifact roster changed"):
        _observe(api)
    api = FakeGet()
    api.next_artifacts = deepcopy(api.artifacts)
    api.next_artifacts[0]["expired"] = True
    with pytest.raises(ValueError, match="artifact roster changed"):
        _observe(api)


def test_second_scan_failure_never_uses_first_scan_as_success() -> None:
    api = FakeGet()
    api.fail_scan = 2
    with pytest.raises(RuntimeError, match="unavailable"):
        _observe(api)


def test_invalid_run_status_and_nonexistent_head_are_not_absence() -> None:
    api = FakeGet()
    api.run["head_sha"] = "invalid"
    with pytest.raises(ValueError, match="head SHA"):
        _observe(api)
    api = FakeGet()
    api.run["status"] = "unknown"
    with pytest.raises(ValueError, match="status"):
        _observe(api)


def test_hidden_older_artifacts_cannot_be_detected_from_misreported_count() -> None:
    api = FakeGet(101)
    api.fake_total = 100
    report = _observe(api)
    assert report["artifact_count"] == 100
    assert report["artifact_history_complete"] is False
    assert report["lease_reassignment_authorized"] is False
