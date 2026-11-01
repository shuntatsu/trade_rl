"""Fail-closed Git commit/tree/blob registry reader oracles (read-only)."""

from __future__ import annotations

import base64
import hashlib
from copy import deepcopy

import pytest

from tools.agent_repo.coordination.registry import CoordinatorRegistry
from tools.github_coordination_registry_reader import GitHubRegistryObjectReader

REPOSITORY = "shuntatsu/trade-rl-coordination-disposable"
PATH = "coordination/registry.json"
COMMIT = "a" * 40
TREE = "b" * 40


def _blob_sha(raw: bytes) -> str:
    data = b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw
    return hashlib.sha1(data).hexdigest()


class FakeGitApi:
    def __init__(self, raw: bytes | None = None) -> None:
        self.raw = (
            CoordinatorRegistry(REPOSITORY).canonical_bytes() if raw is None else raw
        )
        self.blob_sha = _blob_sha(self.raw)
        self.commit: dict[str, object] = {"sha": COMMIT, "tree": {"sha": TREE}}
        self.tree: dict[str, object] = {
            "sha": TREE,
            "truncated": False,
            "tree": [
                {
                    "path": PATH,
                    "mode": "100644",
                    "type": "blob",
                    "sha": self.blob_sha,
                    "size": len(self.raw),
                }
            ],
        }
        self.blob: dict[str, object] = {
            "sha": self.blob_sha,
            "encoding": "base64",
            "content": base64.b64encode(self.raw).decode("ascii"),
            "size": len(self.raw),
        }
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []
        self.fail = False

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        self.calls.append((method, suffix, deepcopy(query)))
        if self.fail:
            raise RuntimeError("GitHub unavailable")
        assert method == "GET"
        if suffix == f"git/commits/{COMMIT}":
            return deepcopy(self.commit)
        if suffix == f"git/trees/{TREE}":
            assert query == {"recursive": "1"}
            return deepcopy(self.tree)
        if suffix == f"git/blobs/{self.blob_sha}":
            return deepcopy(self.blob)
        raise AssertionError(f"unexpected request: {suffix}")


def _reader(api: FakeGitApi, *, repository: str = REPOSITORY, path: str = PATH):
    return GitHubRegistryObjectReader(
        api, expected_repository=repository, registry_path=path
    )


def test_reads_only_exact_commit_tree_blob_and_preserves_canonical_bytes() -> None:
    api = FakeGitApi()
    raw = _reader(api).read_registry_bytes(REPOSITORY, COMMIT)
    assert raw == api.raw
    assert api.calls == [
        ("GET", f"git/commits/{COMMIT}", None),
        ("GET", f"git/trees/{TREE}", {"recursive": "1"}),
        ("GET", f"git/blobs/{api.blob_sha}", None),
    ]


@pytest.mark.parametrize("requested", ["other/repo", "shuntatsu/trade_rl", ""])
def test_foreign_repository_is_rejected_without_io(requested: str) -> None:
    api = FakeGitApi()
    with pytest.raises(ValueError, match="repository"):
        _reader(api).read_registry_bytes(requested, COMMIT)
    assert api.calls == []


@pytest.mark.parametrize("commit", ["a" * 39, "A" * 40, "branch/main", ""])
def test_commit_must_be_exact_lowercase_oid(commit: str) -> None:
    api = FakeGitApi()
    with pytest.raises(ValueError, match="commit"):
        _reader(api).read_registry_bytes(REPOSITORY, commit)
    assert api.calls == []


@pytest.mark.parametrize(
    "path",
    ["/coordination/registry.json", "../registry.json", "a//b", "a/./b", "a\\b", ""],
)
def test_registry_path_must_be_relative_and_unambiguous(path: str) -> None:
    api = FakeGitApi()
    with pytest.raises(ValueError, match="registry path"):
        _reader(api, path=path)
    assert api.calls == []


def test_commit_identity_and_tree_identity_drift_fail_closed() -> None:
    api = FakeGitApi()
    api.commit["sha"] = "c" * 40
    with pytest.raises(ValueError, match="commit SHA"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)
    assert len(api.calls) == 1

    api = FakeGitApi()
    api.tree["sha"] = "c" * 40
    with pytest.raises(ValueError, match="tree SHA"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)
    assert len(api.calls) == 2


def test_missing_or_malformed_commit_tree_fails_closed() -> None:
    api = FakeGitApi()
    api.commit["tree"] = None
    with pytest.raises(ValueError, match="commit tree"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)

    api = FakeGitApi()
    api.tree["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)

    api = FakeGitApi()
    api.tree["truncated"] = None
    with pytest.raises(ValueError, match="truncated"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)

    api = FakeGitApi()
    api.tree["tree"] = None
    with pytest.raises(ValueError, match="tree entries"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)


def test_registry_path_missing_or_duplicated_fails_closed() -> None:
    api = FakeGitApi()
    api.tree["tree"] = []
    with pytest.raises(ValueError, match="registry path"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)

    api = FakeGitApi()
    api.tree["tree"] = api.tree["tree"] * 2
    with pytest.raises(ValueError, match="registry path"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)


@pytest.mark.parametrize(
    "mode,kind",
    [("120000", "blob"), ("100755", "blob"), ("040000", "tree"), ("160000", "commit")],
)
def test_symlinks_executables_submodules_and_trees_are_not_payload_files(
    mode: str, kind: str
) -> None:
    api = FakeGitApi()
    entries = api.tree["tree"]
    assert isinstance(entries, list)
    entries[0]["mode"] = mode
    entries[0]["type"] = kind
    with pytest.raises(ValueError, match="ordinary blob"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)
    assert len(api.calls) == 2


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("sha", "c" * 40, "blob SHA"),
        ("size", True, "blob size"),
        ("size", 999, "blob size"),
        ("encoding", "utf-8", "base64"),
        ("content", "%%%", "base64"),
    ],
)
def test_blob_identity_encoding_size_and_content_are_strict(
    field: str, value: object, error: str
) -> None:
    api = FakeGitApi()
    api.blob[field] = value
    with pytest.raises(ValueError, match=error):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)


def test_blob_bytes_must_rehash_to_tree_oid() -> None:
    api = FakeGitApi()
    api.blob["content"] = base64.b64encode(b"x" * len(api.raw)).decode("ascii")
    with pytest.raises(ValueError, match="Git blob SHA"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)


def test_tree_blob_size_must_match_decoded_payload() -> None:
    api = FakeGitApi()
    entries = api.tree["tree"]
    assert isinstance(entries, list)
    entries[0]["size"] = len(api.raw) + 1
    with pytest.raises(ValueError, match="tree blob size"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)


@pytest.mark.parametrize("raw", [b"not JSON", b"{}", b'{"schema":"x"}'])
def test_noncanonical_registry_bytes_are_rejected(raw: bytes) -> None:
    api = FakeGitApi(raw)
    with pytest.raises(ValueError):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)


def test_canonical_other_repository_registry_is_rejected() -> None:
    api = FakeGitApi(CoordinatorRegistry("other/repo").canonical_bytes())
    with pytest.raises(ValueError, match="registry repository"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)


def test_api_errors_propagate_and_do_not_fallback_to_unverified_content() -> None:
    api = FakeGitApi()
    api.fail = True
    with pytest.raises(RuntimeError, match="unavailable"):
        _reader(api).read_registry_bytes(REPOSITORY, COMMIT)
    assert len(api.calls) == 1
