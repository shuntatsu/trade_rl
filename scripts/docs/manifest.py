from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from scripts.docs.metadata import discover_documents


def build_manifest(root: Path) -> dict[str, object]:
    """Build a deterministic machine-readable documentation manifest."""

    root = root.resolve()
    documents = discover_documents(root / "docs")
    records = [document.to_manifest_record() for document in documents]

    canonical_owners: dict[str, str] = {}
    topics: dict[str, list[str]] = defaultdict(list)
    related_code: dict[str, list[str]] = defaultdict(list)

    for document in documents:
        if document.lifecycle == "current" and document.authority == "canonical":
            for key in document.source_of_truth_for:
                canonical_owners[key] = document.path.as_posix()
        for topic in document.topics:
            topics[topic].append(document.path.as_posix())
        if document.lifecycle == "current":
            for pattern in document.related_code:
                related_code[pattern].append(document.path.as_posix())

    return {
        "schema_version": "trade_rl_docs_manifest_v1",
        "documents": records,
        "canonical_owners": dict(sorted(canonical_owners.items())),
        "topics": {topic: sorted(paths) for topic, paths in sorted(topics.items())},
        "related_code": {
            pattern: sorted(paths) for pattern, paths in sorted(related_code.items())
        },
    }
