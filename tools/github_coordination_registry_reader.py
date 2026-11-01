"""Read-only GitHub Git-object registry reader; never Task or ref authority.

The caller supplies a repository-bound GET API. This verifies coherent Git
commit/tree/blob references and Git blob byte identity, but does not itself
authenticate that API, a Coordinator writer, ref protection or roster coverage.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from typing import Any, Protocol

from tools.agent_repo.coordination.registry import parse_registry_bytes

_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MAX_PAYLOAD_BYTES = 1024 * 1024


class GitObjectReadApi(Protocol):
    """Existing repository-bound GitHubApi _request_json GET-only surface."""

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> Any: ...


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _oid(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _OID_RE.fullmatch(value):
        raise ValueError(f"{name} must be an exact lowercase Git commit SHA")
    return value


def _size(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a native nonnegative integer")
    return value


def _path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(component in ("", ".", "..") for component in value.split("/"))
        or any(character.isspace() for character in value)
    ):
        raise ValueError("registry path must be a safe relative Git tree path")
    return value


class GitHubRegistryObjectReader:
    """Resolve one pinned commit to canonical registry bytes via GET only."""

    def __init__(
        self,
        api: GitObjectReadApi,
        *,
        expected_repository: str,
        registry_path: str,
    ) -> None:
        if not isinstance(expected_repository, str) or not _REPOSITORY_RE.fullmatch(
            expected_repository
        ):
            raise ValueError("expected repository must use owner/name form")
        self._api = api
        self._repository = expected_repository
        self._registry_path = _path(registry_path)

    def read_registry_bytes(self, repository: str, commit_oid: str) -> bytes:
        """Fail closed on absent/drifting objects; never update a GitHub ref."""

        if repository != self._repository:
            raise ValueError("requested repository differs from reader repository")
        commit_oid = _oid(commit_oid, "commit SHA")

        commit = _mapping(
            self._api._request_json("GET", f"git/commits/{commit_oid}"),
            "commit",
        )
        if commit.get("sha") != commit_oid:
            raise ValueError("commit SHA changed during object read")
        tree_node = _mapping(commit.get("tree"), "commit tree")
        tree_sha = _oid(tree_node.get("sha"), "commit tree SHA")
        tree = _mapping(
            self._api._request_json(
                "GET", f"git/trees/{tree_sha}", query={"recursive": "1"}
            ),
            "tree",
        )
        if tree.get("sha") != tree_sha:
            raise ValueError("tree SHA changed during object read")
        if tree.get("truncated") is not False:
            raise ValueError("tree truncated state is missing or true")
        entries = tree.get("tree")
        if not isinstance(entries, list):
            raise ValueError("tree entries must be a list")
        matches = [
            entry
            for item in entries
            if (entry := _mapping(item, "tree entry")).get("path")
            == self._registry_path
        ]
        if len(matches) != 1:
            raise ValueError("registry path must exist exactly once in Git tree")
        target = matches[0]
        if target.get("type") != "blob" or target.get("mode") != "100644":
            raise ValueError("registry path must be an ordinary blob (100644)")
        blob_sha = _oid(target.get("sha"), "tree blob SHA")
        tree_size = _size(target.get("size"), "tree blob size")
        if tree_size > _MAX_PAYLOAD_BYTES:
            raise ValueError("tree blob size exceeds registry payload limit")

        blob = _mapping(self._api._request_json("GET", f"git/blobs/{blob_sha}"), "blob")
        if blob.get("sha") != blob_sha:
            raise ValueError("blob SHA differs from tree reference")
        blob_size = _size(blob.get("size"), "blob size")
        if blob_size > _MAX_PAYLOAD_BYTES:
            raise ValueError("blob size exceeds registry payload limit")
        if blob.get("encoding") != "base64":
            raise ValueError("blob encoding must be base64")
        content = blob.get("content")
        if not isinstance(content, str) or len(content) > 2 * _MAX_PAYLOAD_BYTES:
            raise ValueError("blob content must be bounded base64")
        encoded = content.replace("\r", "").replace("\n", "")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("blob content must be strict base64") from error
        if base64.b64encode(raw).decode("ascii") != encoded:
            raise ValueError("blob content must be canonical base64")
        if len(raw) != blob_size:
            raise ValueError("blob size differs from decoded content")
        if len(raw) != tree_size:
            raise ValueError("tree blob size differs from decoded content")
        git_blob = b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw
        if hashlib.sha1(git_blob).hexdigest() != blob_sha:
            raise ValueError("Git blob SHA differs from decoded content")

        registry = parse_registry_bytes(raw)
        if registry.repository != repository:
            raise ValueError("registry repository differs from requested repository")
        return raw


__all__ = ["GitHubRegistryObjectReader"]
