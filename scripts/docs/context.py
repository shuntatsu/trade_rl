from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.docs.metadata import discover_documents
from scripts.docs.model import DocumentMetadata

_AUTHORITY_RANK = {"canonical": 0, "supporting": 1, "evidence": 2, "none": 3}
_LIFECYCLE_RANK = {"current": 0, "deprecated": 1, "historical": 2}


def _sort_key(document: DocumentMetadata) -> tuple[int, int, int, str]:
    return (
        _LIFECYCLE_RANK.get(document.lifecycle, 99),
        _AUTHORITY_RANK.get(document.authority, 99),
        document.nav_order,
        document.path.as_posix(),
    )


def _matches_path(pattern: str, changed_path: str) -> bool:
    pattern = pattern.replace("\\", "/").lstrip("./")
    changed_path = changed_path.replace("\\", "/").lstrip("./")
    if any(character in pattern for character in "*?["):
        return fnmatch.fnmatch(changed_path, pattern)
    normalized = pattern.rstrip("/")
    return changed_path == normalized or changed_path.startswith(f"{normalized}/")


def documents_for_path(
    root: Path,
    changed_path: str,
    *,
    include_history: bool = False,
) -> tuple[DocumentMetadata, ...]:
    documents = discover_documents(root.resolve() / "docs")
    matches = [
        document
        for document in documents
        if (include_history or document.lifecycle != "historical")
        and any(_matches_path(pattern, changed_path) for pattern in document.related_code)
    ]
    return tuple(sorted(matches, key=_sort_key))


def documents_for_topic(
    root: Path,
    topic: str,
    *,
    include_history: bool = False,
) -> tuple[DocumentMetadata, ...]:
    documents = discover_documents(root.resolve() / "docs")
    normalized = topic.strip().lower()
    matches = [
        document
        for document in documents
        if (include_history or document.lifecycle != "historical")
        and normalized in {candidate.lower() for candidate in document.topics}
    ]
    return tuple(sorted(matches, key=_sort_key))


def _print_documents(documents: tuple[DocumentMetadata, ...]) -> None:
    if not documents:
        print("No governed documentation context matched.")
        return
    for document in documents:
        print(
            f"{document.authority.upper():10} "
            f"{document.lifecycle.upper():10} docs/{document.path.as_posix()}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve Trade RL documentation context")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--path", dest="changed_path")
    selector.add_argument("--topic")
    parser.add_argument("--root", default=".")
    parser.add_argument("--include-history", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if args.changed_path:
        documents = documents_for_path(
            root,
            args.changed_path,
            include_history=args.include_history,
        )
    else:
        documents = documents_for_topic(
            root,
            args.topic,
            include_history=args.include_history,
        )
    _print_documents(documents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
