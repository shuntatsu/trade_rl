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
        observe_workflow_artifacts(api, expected_repository=invalid_repo, run_id=RUN)
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


@pytest.mark.parametrize("count", [100, 200, 10_000])
def test_full_final_page_requires_empty_terminal_page(count: int) -> None:
    api = FakeGet(count)
    report = _observe(api)
    assert report["artifact_count"] == count
    pages = [
        int(query["page"])
        for _, suffix, query in api.calls
        if suffix.endswith("/artifacts") and query is not None
    ]
    assert pages == list(range(1, count // 100 + 2)) * 2
    assert report["artifact_history_complete"] is False
    assert report["lease_reassignment_authorized"] is False


@pytest.mark.parametrize("second_scan_only", [False, True])
def test_extra_artifact_after_full_page_refuses(second_scan_only: bool) -> None:
    api = FakeGet(100 if second_scan_only else 101)
    api.fake_total = 100
    if second_scan_only:
        api.next_artifacts = [*api.artifacts, _artifact(101)]
    with pytest.raises(ValueError, match="pagination"):
        _observe(api)


def test_terminal_page_total_drift_refuses() -> None:
    api = FakeGet(100)
    api.second_page_total = 101
    with pytest.raises(ValueError, match="total_count changed"):
        _observe(api)


@pytest.mark.parametrize("count", [2, 101])
@pytest.mark.parametrize("second_scan_only", [False, True])
def test_duplicate_artifact_node_ids_refuse(count: int, second_scan_only: bool) -> None:
    api = FakeGet(count)
    rows = deepcopy(api.artifacts)
    rows[-1]["node_id"] = rows[0]["node_id"]
    if second_scan_only:
        api.next_artifacts = rows
    else:
        api.artifacts = rows
    with pytest.raises(ValueError, match="duplicate artifact node_id"):
        _observe(api)


@pytest.mark.parametrize("readback", [False, True])
def test_unfinished_artifact_run_requires_explicit_null_conclusion(
    readback: bool,
) -> None:
    api = FakeGet()
    api.run.update(status="in_progress", conclusion=None)
    if readback:
        api.next_run = dict(api.run)
        del api.next_run["conclusion"]
    else:
        del api.run["conclusion"]
    with pytest.raises(ValueError, match="conclusion is missing"):
        _observe(api)


def test_unhashable_artifact_run_conclusion_refuses_with_value_error() -> None:
    api = FakeGet()
    api.run["conclusion"] = []
    with pytest.raises(ValueError, match="conclusion"):
        _observe(api)


def test_unfinished_artifact_run_accepts_explicit_null_but_not_authority() -> None:
    api = FakeGet()
    api.run.update(status="in_progress", conclusion=None)
    report = _observe(api)
    assert report["conclusion"] is None
    assert report["cleanup_authorized"] is False
    assert report["lease_reassignment_authorized"] is False


def test_matching_null_run_and_artifact_heads_are_not_exact_commits() -> None:
    api = FakeGet()
    api.run["head_sha"] = "0" * 40
    for artifact in api.artifacts:
        artifact["workflow_run"]["head_sha"] = "0" * 40
    with pytest.raises(ValueError, match="SHA"):
        _observe(api)


class _ForgedText(str):
    def __eq__(self, other: object) -> bool:
        # Preserve the existing null-SHA rejection check so the real equality
        # defect, rather than that unrelated guard, is the failure oracle.
        return not (type(other) is str and other == "0" * 40)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


class _CasefoldTrap(str):
    def casefold(self) -> str:
        raise AssertionError("foreign repository casefold called")


@pytest.mark.parametrize(
    ("key", "forged"),
    [
        ("head_sha", _ForgedText("b" * 40)),
        ("head_branch", _ForgedText("wrong/branch")),
    ],
)
def test_artifact_origin_rejects_foreign_identity(key: str, forged: str) -> None:
    api = FakeGet()
    api.artifacts[0]["workflow_run"][key] = forged  # type: ignore[index]
    with pytest.raises(ValueError, match="artifact workflow identity"):
        _observe(api)


def test_artifact_run_rejects_foreign_sha_even_when_it_forges_readback() -> None:
    api = FakeGet()
    api.run["head_sha"] = _ForgedText("b" * 40)
    api.next_run = {**api.run, "head_sha": SHA}
    with pytest.raises(ValueError, match="run head SHA"):
        _observe(api)


def test_artifact_repo_argument_rejects_foreign_casefold_before_reads() -> None:
    api = FakeGet()
    with pytest.raises(ValueError, match="expected_repository"):
        observe_workflow_artifacts(
            api, expected_repository=_CasefoldTrap(REPOSITORY), run_id=RUN
        )
    assert api.calls == []


def test_artifact_repo_response_rejects_foreign_casefold() -> None:
    api = FakeGet()
    api.repo["full_name"] = _CasefoldTrap(REPOSITORY)
    with pytest.raises(ValueError, match="repository identity"):
        _observe(api)


@pytest.mark.parametrize(
    ("key", "foreign"),
    [
        ("node_id", _ForgedText("A_1")),
        ("name", _ForgedText("repeated-name")),
        ("digest", _ForgedText("sha256:" + "c" * 64)),
    ],
)
def test_artifact_metadata_rejects_foreign_text(key: str, foreign: str) -> None:
    api = FakeGet()
    api.artifacts[0][key] = foreign
    message = (
        "artifact digest is malformed"
        if key == "digest"
        else f"artifact {key} must be non-empty text"
    )
    with pytest.raises(ValueError, match=message):
        _observe(api)


def test_artifact_native_repository_case_insensitive_control() -> None:
    api = FakeGet()
    result = observe_workflow_artifacts(
        api, expected_repository=REPOSITORY.upper(), run_id=RUN
    )
    assert result["artifact_count"] == 2
    assert result["cleanup_authorized"] is False


class _CompareTrap(str):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("foreign equality called")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("foreign inequality called")


@pytest.mark.parametrize("key", ["head_branch", "head_sha"])
def test_artifact_origin_refuses_before_foreign_comparison(key: str) -> None:
    api = FakeGet()
    api.artifacts[0]["workflow_run"][key] = _CompareTrap(  # type: ignore[index]
        BRANCH if key == "head_branch" else SHA
    )
    with pytest.raises(ValueError, match="artifact workflow identity"):
        _observe(api)


def test_artifact_run_refuses_before_foreign_sha_comparison() -> None:
    api = FakeGet()
    api.run["head_sha"] = _CompareTrap(SHA)
    with pytest.raises(ValueError, match="run head SHA"):
        _observe(api)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("head_sha", _CompareTrap(SHA), "run head SHA"),
        ("head_branch", _CompareTrap(BRANCH), "run head branch"),
        ("updated_at", _CompareTrap("2026-11-02T10:30:39Z"), "run updated_at"),
    ],
)
def test_run_readback_refuses_foreign_text(
    field: str, value: str, message: str
) -> None:
    api = FakeGet()
    api.next_run = {**api.run, field: value}
    with pytest.raises(ValueError, match=message):
        _observe(api)
    assert api.run_reads == api.scans == 2


def test_repository_readback_refuses_foreign_node_id() -> None:
    api = FakeGet()
    api.next_repo = {**api.repo, "node_id": _CompareTrap("R_123")}
    with pytest.raises(ValueError, match="repository node id"):
        _observe(api)
    assert api.repo_reads == api.run_reads == api.scans == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("node_id", _CompareTrap("A_1"), "artifact node_id"),
        ("name", _CompareTrap("repeated-name"), "artifact name"),
        ("digest", _CompareTrap("sha256:" + "c" * 64), "artifact digest"),
    ],
)
def test_artifact_metadata_refuses_before_foreign_comparison(
    field: str, value: str, message: str
) -> None:
    api = FakeGet()
    api.artifacts[0][field] = value
    with pytest.raises(ValueError, match=message):
        _observe(api)


class _MappingTrap(dict[str, object]):
    def get(self, *args, **kwargs):
        raise AssertionError("foreign object lookup called")


class _RowsTrap(list[object]):
    def __len__(self):
        raise AssertionError("foreign list length called")

    def __iter__(self):
        raise AssertionError("foreign list iteration called")


class _ForeignKey(str):
    pass


class _ForeignResponse(FakeGet):
    def __init__(self, location: str):
        super().__init__()
        self.location = location

    def _request_json(self, method, suffix, *, query=None):
        raw = super()._request_json(method, suffix, query=query)
        location = self.location
        if suffix == "":
            if location == "repo":
                return _MappingTrap(raw)
            if location == "repo_key":
                return {
                    _ForeignKey(key) if key == "full_name" else key: value
                    for key, value in raw.items()
                }
        elif suffix == f"actions/runs/{RUN}":
            if location == "run":
                return _MappingTrap(raw)
            if location in ("repository", "head_repository"):
                raw[location] = _MappingTrap(raw[location])
        elif suffix == f"actions/runs/{RUN}/artifacts":
            if location == "page":
                return _MappingTrap(raw)
            if location == "rows":
                raw["artifacts"] = _RowsTrap(raw["artifacts"])
            elif location == "artifact":
                raw["artifacts"][0] = _MappingTrap(raw["artifacts"][0])
            elif location == "origin":
                row = raw["artifacts"][0]
                row["workflow_run"] = _MappingTrap(row["workflow_run"])
        return raw


@pytest.mark.parametrize(
    "location",
    [
        "repo",
        "repo_key",
        "run",
        "repository",
        "head_repository",
        "page",
        "rows",
        "artifact",
        "origin",
    ],
)
def test_observation_refuses_foreign_json_containers(location: str) -> None:
    with pytest.raises(ValueError, match="native JSON|artifact pagination"):
        _observe(_ForeignResponse(location))


@pytest.mark.parametrize(
    ("expected", "alias"),
    [
        ("shuntatsu/trade_rl", "\u017fhuntatsu/trade_rl"),
        ("owner/k", "owner/\u212a"),
        ("owner/ffi", "owner/\ufb03"),
    ],
)
@pytest.mark.parametrize("at_readback", [False, True])
def test_artifact_repo_rejects_unicode_casefold_alias_at_both_reads(
    expected: str, alias: str, at_readback: bool
) -> None:
    api = FakeGet()
    if at_readback:
        api.repo["full_name"] = expected
        api.next_repo = {**api.repo, "full_name": alias}
    else:
        api.repo["full_name"] = alias

    with pytest.raises(ValueError, match="repository identity differs"):
        observe_workflow_artifacts(api, expected_repository=expected, run_id=RUN)

    # Initial refusal must precede all run/artifact reads. Drift refusal
    # happens only at the final repository readback.
    assert api.repo_reads == (2 if at_readback else 1)
    assert api.run_reads == (2 if at_readback else 0)
    assert api.scans == (2 if at_readback else 0)


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        ("shuntatsu/trade_rl", "SHUNTATSU/TRADE_RL"),
        ("owner/k", "Owner/K"),
        ("owner/ffi", "OWNER/FFI"),
    ],
)
def test_artifact_repo_preserves_ascii_case_insensitive_identity(
    expected: str, observed: str
) -> None:
    api = FakeGet()
    api.repo["full_name"] = observed
    report = observe_workflow_artifacts(api, expected_repository=expected, run_id=RUN)
    assert report["repository_id"] == REPO_ID
    assert report["artifact_count"] == 2
    assert report["artifact_history_complete"] is False
    assert report["cleanup_authorized"] is False
    assert api.repo_reads == 2


@pytest.mark.parametrize(
    "branch", ["bad\0ref", "bad\nref", "bad\ud800ref", "bad//ref", "bad..ref", "bad.lock"]
)
@pytest.mark.parametrize("readback", [False, True])
def test_artifact_observation_rejects_malformed_head_ref(
    branch: str, readback: bool
) -> None:
    api = FakeGet()
    if readback:
        api.next_run = {**api.run, "head_branch": branch}
    else:
        api.run["head_branch"] = branch
        for artifact in api.artifacts:
            origin = artifact["workflow_run"]
            assert type(origin) is dict
            origin["head_branch"] = branch
    with pytest.raises(ValueError, match="run head branch is malformed"):
        _observe(api)
    if not readback:
        assert api.scans == 0
        assert api.run_reads == 1


@pytest.mark.parametrize("branch", ["main", "feature/日本語", "v1.2.3", "a" * 40])
def test_artifact_observation_accepts_valid_head_ref_without_authority(branch: str) -> None:
    api = FakeGet()
    api.run["head_branch"] = branch
    for artifact in api.artifacts:
        origin = artifact["workflow_run"]
        assert type(origin) is dict
        origin["head_branch"] = branch
    report = _observe(api)
    assert report["head_branch"] == branch
    assert report["artifact_attempt_binding_verified"] is False
    assert report["workflow_activity_checked"] is False
    assert report["task_registry_complete"] is False
    assert report["lease_reassignment_authorized"] is False
    assert report["cleanup_authorized"] is False
