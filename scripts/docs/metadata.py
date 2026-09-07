from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from scripts.docs.model import DocumentMetadata

_AGENT_INSTRUCTION_NAME = "AGENTS.md"


def _mapping(value: object, *, source: Path) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"metadata must be a mapping: {source}")
    return {str(key): item for key, item in value.items()}


def _front_matter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError(f"unterminated YAML front matter: {path}") from exc
    payload = yaml.safe_load("\n".join(lines[1:end]))
    return _mapping(payload, source=path)


def _folder_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, source=path)


def _page_metadata(
    folder_metadata: dict[str, object],
    *,
    metadata_path: Path,
    page_path: Path,
) -> dict[str, object]:
    pages_raw = folder_metadata.get("pages")
    if pages_raw is None:
        return {}
    pages = _mapping(pages_raw, source=metadata_path)
    page_raw = pages.get(page_path.name)
    if page_raw is None:
        return {}
    return _mapping(page_raw, source=metadata_path)


def _as_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item))


def _derived_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return path.stem.replace("-", " ").replace("_", " ").strip().title()


def _metadata_chain(path: Path, docs_root: Path) -> tuple[Path, ...]:
    parent = path.parent.resolve()
    root = docs_root.resolve()
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"document is outside docs root: {path}") from exc

    directories: list[Path] = []
    current = root
    directories.append(current)
    relative_parent = parent.relative_to(root)
    for part in relative_parent.parts:
        current = current / part
        directories.append(current)
    return tuple(directory / ".meta.yml" for directory in directories)


def load_effective_metadata(path: Path, docs_root: Path) -> DocumentMetadata:
    """Load folder defaults, page-map metadata, then page front matter."""

    path = path.resolve()
    docs_root = docs_root.resolve()
    merged: dict[str, object] = {}
    for metadata_path in _metadata_chain(path, docs_root):
        folder = _folder_metadata(metadata_path)
        merged.update({key: value for key, value in folder.items() if key != "pages"})
        if metadata_path.parent.resolve() == path.parent.resolve():
            merged.update(
                _page_metadata(
                    folder,
                    metadata_path=metadata_path,
                    page_path=path,
                )
            )
    merged.update(_front_matter(path))

    nav_order_raw = merged.get("nav_order", 100)
    nav_order = nav_order_raw if isinstance(nav_order_raw, int) else 100
    description_raw = merged.get("description")
    description = str(description_raw) if description_raw is not None else None

    return DocumentMetadata(
        path=path.relative_to(docs_root),
        title=str(merged.get("title") or _derived_title(path)),
        doc_type=str(merged.get("doc_type") or ""),
        lifecycle=str(merged.get("lifecycle") or ""),
        authority=str(merged.get("authority") or ""),
        topics=_as_strings(merged.get("topics")),
        source_of_truth_for=_as_strings(merged.get("source_of_truth_for")),
        related_code=_as_strings(merged.get("related_code")),
        agent_read_when=_as_strings(merged.get("agent_read_when")),
        nav_order=nav_order,
        description=description,
    )


def discover_documents(docs_root: Path) -> tuple[DocumentMetadata, ...]:
    docs_root = docs_root.resolve()
    paths = sorted(
        (
            path
            for path in docs_root.rglob("*.md")
            if path.is_file()
            and path.name != _AGENT_INSTRUCTION_NAME
            and ".docs-build" not in path.parts
        ),
        key=lambda path: path.relative_to(docs_root).as_posix(),
    )
    return tuple(load_effective_metadata(path, docs_root) for path in paths)


def read_front_matter(path: Path) -> dict[str, Any]:
    """Expose page front matter for diagnostics without inherited defaults."""

    return dict(_front_matter(path))
