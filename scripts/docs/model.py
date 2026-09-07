from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Normalized documentation metadata with a path relative to docs/."""

    path: Path
    title: str
    doc_type: str
    lifecycle: str
    authority: str
    topics: tuple[str, ...]
    source_of_truth_for: tuple[str, ...] = ()
    related_code: tuple[str, ...] = ()
    agent_read_when: tuple[str, ...] = ()
    nav_order: int = 100
    description: str | None = None

    @property
    def is_current(self) -> bool:
        return self.lifecycle == "current"

    @property
    def is_historical(self) -> bool:
        return self.lifecycle == "historical"

    def to_manifest_record(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "title": self.title,
            "doc_type": self.doc_type,
            "lifecycle": self.lifecycle,
            "authority": self.authority,
            "topics": list(self.topics),
            "source_of_truth_for": list(self.source_of_truth_for),
            "related_code": list(self.related_code),
            "agent_read_when": list(self.agent_read_when),
            "nav_order": self.nav_order,
            "description": self.description,
        }
