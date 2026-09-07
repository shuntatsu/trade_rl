from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.docs.metadata import load_effective_metadata
from scripts.docs.model import DocumentMetadata

DOC_TYPES = {
    "tutorial",
    "guide",
    "reference",
    "research",
    "runbook",
    "performance",
    "legal",
    "index",
    "history",
}
LIFECYCLES = {"current", "historical", "deprecated"}
AUTHORITIES = {"canonical", "supporting", "evidence", "none"}
FOLDER_TYPES = {
    "getting-started": "tutorial",
    "guides": "guide",
    "reference": "reference",
    "research": "research",
    "operations": "runbook",
    "performance": "performance",
    "legal": "legal",
    "history": "history",
}
_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)")


def _load_documents(docs_root: Path) -> tuple[tuple[DocumentMetadata, ...], tuple[str, ...]]:
    documents: list[DocumentMetadata] = []
    errors: list[str] = []
    for path in sorted(docs_root.rglob("*.md"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        try:
            documents.append(load_effective_metadata(path, docs_root))
        except (OSError, ValueError) as exc:
            errors.append(f"metadata parse error: {path.relative_to(docs_root)}: {exc}")
    return tuple(documents), tuple(errors)


def _expected_type(document: DocumentMetadata) -> str | None:
    if document.path == Path("index.md"):
        return "index"
    if not document.path.parts:
        return None
    first = document.path.parts[0]
    expected = FOLDER_TYPES.get(first)
    if expected is None:
        return None
    if document.path.name == "index.md" and first != "history":
        return "index"
    return expected


def _internal_targets(path: Path, root: Path) -> tuple[Path, ...]:
    targets: list[Path] = []
    text = path.read_text(encoding="utf-8")
    for raw in _LINK_PATTERN.findall(text):
        target = raw.split("#", maxsplit=1)[0].strip()
        if not target:
            continue
        targets.append((path.parent / target).resolve())
    return tuple(targets)


def _related_code_exists(root: Path, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/").lstrip("./")
    if any(character in normalized for character in "*?["):
        return any(root.glob(normalized))
    return (root / normalized.rstrip("/")).exists()


def _validate_orphans(
    root: Path,
    docs_root: Path,
    current: tuple[DocumentMetadata, ...],
) -> list[str]:
    errors: list[str] = []
    root_index = docs_root / "index.md"
    if not root_index.is_file():
        return ["missing documentation portal: docs/index.md"]
    root_targets = set(_internal_targets(root_index, root))

    for document in current:
        source = docs_root / document.path
        if document.path == Path("index.md"):
            continue
        if len(document.path.parts) < 2:
            errors.append(f"orphan current document: {document.path.as_posix()}")
            continue
        section = document.path.parts[0]
        section_index = docs_root / section / "index.md"
        if document.path.name == "index.md":
            if section_index.resolve() not in root_targets:
                errors.append(
                    f"orphan current document: {document.path.as_posix()} not linked from docs/index.md"
                )
            continue
        if not section_index.is_file():
            errors.append(
                f"orphan current document: {document.path.as_posix()} has no section index"
            )
            continue
        section_targets = set(_internal_targets(section_index, root))
        if source.resolve() not in section_targets:
            errors.append(
                f"orphan current document: {document.path.as_posix()} not linked from {section}/index.md"
            )
    return errors


def validate_documents(root: Path) -> tuple[str, ...]:
    """Return deterministic validation errors; an empty tuple means valid."""

    root = root.resolve()
    docs_root = root / "docs"
    if not docs_root.is_dir():
        return ("missing docs directory",)

    documents, parse_errors = _load_documents(docs_root)
    errors = list(parse_errors)

    for document in documents:
        if document.doc_type not in DOC_TYPES:
            errors.append(
                f"invalid doc_type for {document.path.as_posix()}: {document.doc_type!r}"
            )
        if document.lifecycle not in LIFECYCLES:
            errors.append(
                f"invalid lifecycle for {document.path.as_posix()}: {document.lifecycle!r}"
            )
        if document.authority not in AUTHORITIES:
            errors.append(
                f"invalid authority for {document.path.as_posix()}: {document.authority!r}"
            )

        expected = _expected_type(document)
        if expected is not None and document.doc_type != expected:
            errors.append(
                f"folder/type mismatch for {document.path.as_posix()}: expected {expected!r}, got {document.doc_type!r}"
            )

        if document.path.parts and document.path.parts[0] == "history":
            if (
                document.doc_type != "history"
                or document.lifecycle != "historical"
                or document.authority != "none"
            ):
                errors.append(
                    f"history authority violation: {document.path.as_posix()} must be history/historical/none"
                )
            continue

        if document.lifecycle == "current":
            missing: list[str] = []
            if not document.title:
                missing.append("title")
            if not document.doc_type:
                missing.append("doc_type")
            if not document.authority:
                missing.append("authority")
            if not document.topics:
                missing.append("topics")
            if missing:
                errors.append(
                    f"missing required metadata for {document.path.as_posix()}: {', '.join(missing)}"
                )

            for related in document.related_code:
                if not _related_code_exists(root, related):
                    errors.append(
                        f"invalid related_code for {document.path.as_posix()}: {related}"
                    )

    current = tuple(document for document in documents if document.lifecycle == "current")
    owners: dict[str, list[str]] = defaultdict(list)
    for document in current:
        if document.authority != "canonical":
            continue
        for key in document.source_of_truth_for:
            owners[key].append(document.path.as_posix())
    for key, paths in sorted(owners.items()):
        if len(paths) > 1:
            errors.append(
                f"duplicate canonical ownership for {key}: {', '.join(sorted(paths))}"
            )

    for document in current:
        source = docs_root / document.path
        for target in _internal_targets(source, root):
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(
                    f"broken current link: {document.path.as_posix()} escapes repository"
                )
                continue
            if not target.exists():
                errors.append(
                    f"broken current link: {document.path.as_posix()} -> {target.relative_to(root).as_posix()}"
                )

    errors.extend(_validate_orphans(root, docs_root, current))
    return tuple(sorted(set(errors)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Trade RL documentation governance")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    errors = validate_documents(Path(args.root))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Documentation governance validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
