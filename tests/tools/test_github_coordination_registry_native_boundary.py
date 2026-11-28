"""Direct API boundaries must reject foreign values before executing hooks."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import pytest

from tools.agent_repo.coordination.registry import CoordinatorRegistry
from tools.github_coordination_registry_reader import GitHubRegistryObjectReader
from tools.github_coordination_registry_snapshot import observe_registry_snapshot

REPOSITORY = "shuntatsu/trade-rl-coordination-disposable"
REF = "refs/heads/coordination/task-registry"
PATH = "coordination/registry.json"
COMMIT = "a" * 40
TREE = "b" * 40


class _TextTrap(str):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("foreign equality invoked")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("foreign inequality invoked")

    __hash__ = str.__hash__

    def __len__(self) -> int:
        raise AssertionError("foreign length invoked")

    def __bool__(self) -> bool:
        raise AssertionError("foreign truthiness invoked")

    def strip(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("foreign strip invoked")

    def replace(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("foreign replace invoked")


class _PlainText(str):
    pass


class _ObjectTrap(dict):
    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("foreign mapping lookup invoked")

    def __iter__(self):
        raise AssertionError("foreign mapping iteration invoked")


class _RowsTrap(list):
    def __iter__(self):
        raise AssertionError("foreign row iteration invoked")


class _KeyTrap(str):
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        raise AssertionError("foreign key comparison invoked")


class _Api:
    """Exact GET sequence with real Git blob and canonical registry bytes."""

    def __init__(self, raw: bytes | None = None, *, snapshot: bool = True) -> None:
        self.raw = (
            CoordinatorRegistry(REPOSITORY).canonical_bytes() if raw is None else raw
        )
        oid = hashlib.sha1(
            b"blob " + str(len(self.raw)).encode("ascii") + b"\x00" + self.raw
        ).hexdigest()
        repository = {"full_name": REPOSITORY, "node_id": "R_123", "id": 123}
        ref = {"ref": REF, "object": {"type": "commit", "sha": COMMIT}}
        self.rows: dict[str, Any] = {
            "repository": repository.copy(),
            "ref": {"ref": REF, "object": ref["object"].copy()},
            "commit": {"sha": COMMIT, "tree": {"sha": TREE}},
            "tree": {
                "sha": TREE,
                "truncated": False,
                "tree": [
                    {
                        "path": PATH,
                        "type": "blob",
                        "mode": "100644",
                        "sha": oid,
                        "size": len(self.raw),
                    }
                ],
            },
            "blob": {
                "sha": oid,
                "encoding": "base64",
                "content": base64.b64encode(self.raw).decode("ascii"),
                "size": len(self.raw),
            },
            "ref_after": {"ref": REF, "object": ref["object"].copy()},
            "repository_after": repository.copy(),
        }
        ref_suffix = "git/ref/" + REF.removeprefix("refs/")
        self.steps = [
            ("repository", "", None),
            ("ref", ref_suffix, None),
            ("commit", f"git/commits/{COMMIT}", None),
            ("tree", f"git/trees/{TREE}", {"recursive": "1"}),
            ("blob", f"git/blobs/{oid}", None),
            ("ref_after", ref_suffix, None),
            ("repository_after", "", None),
        ]
        if not snapshot:
            self.steps = self.steps[2:5]
        self.calls: list[str] = []

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        assert method == "GET"
        stage, expected_suffix, expected_query = self.steps[len(self.calls)]
        assert (suffix, query) == (expected_suffix, expected_query)
        self.calls.append(stage)
        return self.rows[stage]


def _observe(api: _Api, *, repository: str = REPOSITORY, path: str = PATH):
    return observe_registry_snapshot(
        api, expected_repository=repository, ref_name=REF, registry_path=path
    )


def _reader(api: _Api, *, repository: str = REPOSITORY, path: str = PATH):
    return GitHubRegistryObjectReader(
        api, expected_repository=repository, registry_path=path
    )


def _replace(api: _Api, stage: str, path: tuple[str | int, ...], factory) -> None:
    keys = (stage, *path)
    container: Any = api.rows
    for key in keys[:-1]:
        container = container[key]
    last = keys[-1]
    container[last] = factory(container[last])


@pytest.mark.parametrize(
    ("stage", "path"),
    [
        ("repository", ()),
        ("ref", ()),
        ("ref", ("object",)),
        ("commit", ()),
        ("commit", ("tree",)),
        ("tree", ()),
        ("tree", ("tree", 0)),
        ("blob", ()),
        ("ref_after", ()),
        ("repository_after", ()),
    ],
)
def test_foreign_response_object_is_rejected_before_lookup(stage, path) -> None:
    api = _Api()
    _replace(api, stage, path, _ObjectTrap)
    with pytest.raises(ValueError):
        _observe(api)
    assert api.calls[-1] == stage


@pytest.mark.parametrize("stage", ["repository", "commit", "tree", "blob", "ref_after"])
def test_unread_foreign_extension_key_cannot_hide_in_native_dict(stage) -> None:
    api = _Api()
    api.rows[stage][_KeyTrap("foreign_extension")] = None
    with pytest.raises(ValueError):
        _observe(api)
    assert api.calls[-1] == stage


def test_foreign_key_at_actual_lookup_is_rejected_without_comparison() -> None:
    api = _Api()
    row = api.rows["commit"]
    row[_KeyTrap("sha")] = row.pop("sha")
    with pytest.raises(ValueError):
        _observe(api)
    assert api.calls[-1] == "commit"


def test_foreign_tree_roster_is_rejected_before_iteration() -> None:
    api = _Api()
    _replace(api, "tree", ("tree",), _RowsTrap)
    with pytest.raises(ValueError):
        _observe(api)
    assert api.calls[-1] == "tree"


@pytest.mark.parametrize(
    ("stage", "path"),
    [
        ("repository", ("full_name",)),
        ("repository", ("node_id",)),
        ("ref", ("ref",)),
        ("ref", ("object", "type")),
        ("ref", ("object", "sha")),
        ("commit", ("sha",)),
        ("commit", ("tree", "sha")),
        ("tree", ("sha",)),
        ("tree", ("tree", 0, "path")),
        ("tree", ("tree", 0, "mode")),
        ("tree", ("tree", 0, "type")),
        ("tree", ("tree", 0, "sha")),
        ("blob", ("sha",)),
        ("blob", ("encoding",)),
        ("blob", ("content",)),
        ("ref_after", ("object", "sha")),
        ("repository_after", ("node_id",)),
    ],
)
def test_foreign_identity_text_is_rejected_before_hooks(stage, path) -> None:
    api = _Api()
    _replace(api, stage, path, _TextTrap)
    with pytest.raises(ValueError):
        _observe(api)
    assert api.calls[-1] == stage


@pytest.mark.parametrize(
    "boundary", ["reader", "snapshot", "requested", "commit", "path"]
)
def test_foreign_caller_inputs_fail_before_io(boundary) -> None:
    api = _Api(snapshot=boundary == "snapshot")
    with pytest.raises(ValueError):
        if boundary == "reader":
            _reader(api, repository=_PlainText(REPOSITORY))
        elif boundary == "snapshot":
            _observe(api, repository=_PlainText(REPOSITORY))
        elif boundary == "requested":
            _reader(api).read_registry_bytes(_TextTrap(REPOSITORY), COMMIT)
        elif boundary == "commit":
            _reader(api).read_registry_bytes(REPOSITORY, _TextTrap(COMMIT))
        else:
            _reader(api, path=_PlainText(PATH))
    assert api.calls == []


@pytest.mark.parametrize("boundary", ["reader", "snapshot"])
def test_nul_registry_path_is_rejected_before_io(boundary) -> None:
    api = _Api(snapshot=boundary == "snapshot")
    path = "coordination/registry\x00.json"
    api.rows["tree"]["tree"][0]["path"] = path
    with pytest.raises(ValueError, match="registry path"):
        if boundary == "reader":
            _reader(api, path=path).read_registry_bytes(REPOSITORY, COMMIT)
        else:
            _observe(api, path=path)
    assert api.calls == []


def test_impossible_observed_tree_entry_path_rejected() -> None:
    api = _Api()
    api.rows["tree"]["tree"].append({"path": "invalid\x00name"})
    with pytest.raises(ValueError, match="tree entry path"):
        _observe(api)
    assert api.calls[-1] == "tree"


def test_native_snapshot_preserves_bytes_and_has_no_authority() -> None:
    api = _Api()
    api.rows["tree"]["tree"].append({"path": "unrelated file.txt"})
    content = api.rows["blob"]["content"]
    lines = [content[i : i + 76] for i in range(0, len(content), 76)]
    api.rows["blob"]["content"] = "\n".join(lines) + "\n"
    result = _observe(api)
    assert result.registry.canonical_bytes() == api.raw
    assert result.registry_digest == hashlib.sha256(api.raw).hexdigest()
    assert result.ref_oid == COMMIT
    assert result.ref_readback_verified is True
    assert result.authoritative_registry_established is False
    assert result.task_registry_complete is False
    assert result.cleanup_authorized is False
    assert api.calls == [stage for stage, _, _ in api.steps]


@pytest.mark.parametrize("raw", [b"{}", b"not JSON", b'{"schema":"wrong"}'])
def test_hash_valid_malformed_registry_still_refuses(raw: bytes) -> None:
    api = _Api(raw)
    with pytest.raises(ValueError):
        _observe(api)
    assert api.calls[-1] == "blob"
