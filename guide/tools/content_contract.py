from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GUIDE_CONTENT = ROOT / "guide" / "content"
TOPICS_DIR = GUIDE_CONTENT / "topics"
MANIFEST_PATH = GUIDE_CONTENT / "manifest.json"

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_VISUALIZATIONS = {
    "architecture",
    "data-flow",
    "economics",
    "observation",
    "experiment-loop",
    "research-status",
}


class GuideContractError(ValueError):
    """Raised when the human guide no longer matches its source contract."""


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _headings(text: str) -> tuple[list[str], list[tuple[int, int, str]]]:
    lines = _normalize_newlines(text).splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    fence_character: str | None = None
    fence_length = 0

    for index, line in enumerate(lines):
        fence = _FENCE_RE.match(line)
        if fence is not None:
            marker = fence.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            continue
        if fence_character is not None:
            continue

        match = _HEADING_RE.match(line.rstrip("\n"))
        if match is None:
            continue
        raw_title = match.group(2).strip()
        title = re.sub(r"[ \t]+#+[ \t]*$", "", raw_title).strip()
        headings.append((index, len(match.group(1)), title))

    return lines, headings


def extract_markdown_section(text: str, heading: str) -> str:
    lines, headings = _headings(text)
    matches = [entry for entry in headings if entry[2] == heading]
    if not matches:
        raise GuideContractError(f"missing heading: {heading}")
    if len(matches) > 1:
        raise GuideContractError(f"duplicate heading: {heading}")

    start_index, level, _ = matches[0]
    end_index = len(lines)
    for index, candidate_level, _candidate_title in headings:
        if index > start_index and candidate_level <= level:
            end_index = index
            break

    section = "".join(lines[start_index:end_index]).rstrip()
    return f"{section}\n"


def section_sha256(text: str) -> str:
    normalized = _normalize_newlines(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _authoritative_source_path(path_text: str) -> PurePosixPath:
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise GuideContractError(f"non-authoritative source path: {path_text}")
    architecture = len(path.parts) >= 3 and path.parts[:2] == ("docs", "architecture")
    research = path == PurePosixPath("docs/research/current-status.md")
    if not (architecture or research):
        raise GuideContractError(f"non-authoritative source path: {path_text}")
    return path


def validate_source_sections(root: Path, source_sections: list[dict[str, str]]) -> None:
    if not source_sections:
        raise GuideContractError("source_sections must not be empty")

    resolved_root = root.resolve()
    for source in source_sections:
        path_text = source.get("path", "")
        heading = source.get("heading", "")
        expected = source.get("sha256", "")
        path = _authoritative_source_path(path_text)
        if not heading:
            raise GuideContractError(f"missing source heading for {path_text}")
        if not _HEX_64_RE.fullmatch(expected):
            raise GuideContractError(f"invalid source fingerprint for {path_text}#{heading}")

        source_path = (root / Path(*path.parts)).resolve()
        try:
            source_path.relative_to(resolved_root)
        except ValueError as exc:
            raise GuideContractError(f"source escapes repository: {path_text}") from exc
        if not source_path.is_file():
            raise GuideContractError(f"missing source file: {path_text}")

        section = extract_markdown_section(
            source_path.read_text(encoding="utf-8"), heading
        )
        actual = section_sha256(section)
        if actual != expected:
            raise GuideContractError(
                "stale source fingerprint: "
                f"{path_text}#{heading}: expected {expected}, actual {actual}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuideContractError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GuideContractError(f"JSON root must be an object: {path}")
    return value


def _manifest_topic_ids(manifest: dict[str, Any]) -> list[str]:
    topic_ids = manifest.get("topics")
    if not isinstance(topic_ids, list) or not topic_ids:
        raise GuideContractError("manifest topics must be a non-empty list")
    if not all(isinstance(topic_id, str) and topic_id for topic_id in topic_ids):
        raise GuideContractError("manifest topic ids must be non-empty strings")
    if len(set(topic_ids)) != len(topic_ids):
        raise GuideContractError("manifest contains duplicate topic ids")
    return topic_ids


def _topic_source_sections(topic: dict[str, Any], *, topic_id: str) -> list[dict[str, str]]:
    raw_sections = topic.get("source_sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise GuideContractError(f"topic {topic_id} has no source_sections")
    sections: list[dict[str, str]] = []
    for raw in raw_sections:
        if not isinstance(raw, dict):
            raise GuideContractError(f"topic {topic_id} has malformed source section")
        path = raw.get("path")
        heading = raw.get("heading")
        sha256 = raw.get("sha256")
        if not all(isinstance(value, str) for value in (path, heading, sha256)):
            raise GuideContractError(f"topic {topic_id} has malformed source section")
        sections.append({"path": path, "heading": heading, "sha256": sha256})
    return sections


def check_content(root: Path = ROOT) -> None:
    manifest_path = root / "guide" / "content" / "manifest.json"
    topics_dir = root / "guide" / "content" / "topics"
    manifest = _read_json(manifest_path)
    topic_ids = _manifest_topic_ids(manifest)

    actual_ids = sorted(path.stem for path in topics_dir.glob("*.json"))
    if sorted(topic_ids) != actual_ids:
        raise GuideContractError(
            f"manifest/topic mismatch: manifest={sorted(topic_ids)}, files={actual_ids}"
        )

    for topic_id in topic_ids:
        topic_path = topics_dir / f"{topic_id}.json"
        topic = _read_json(topic_path)
        if topic.get("id") != topic_id:
            raise GuideContractError(f"topic id mismatch: {topic_path}")
        visualization = topic.get("visualization")
        if not isinstance(visualization, dict):
            raise GuideContractError(f"topic {topic_id} has no visualization object")
        kind = visualization.get("kind")
        if kind not in _ALLOWED_VISUALIZATIONS:
            raise GuideContractError(
                f"topic {topic_id} has unknown visualization kind: {kind!r}"
            )
        validate_source_sections(
            root,
            _topic_source_sections(topic, topic_id=topic_id),
        )


def refresh_sources(topic_ids: list[str], root: Path = ROOT) -> None:
    if not topic_ids:
        raise GuideContractError("refresh requires at least one explicit topic id")

    topics_dir = root / "guide" / "content" / "topics"
    for topic_id in topic_ids:
        topic_path = topics_dir / f"{topic_id}.json"
        topic = _read_json(topic_path)
        sections = _topic_source_sections(topic, topic_id=topic_id)
        refreshed: list[dict[str, str]] = []
        for source in sections:
            path = _authoritative_source_path(source["path"])
            source_path = root / Path(*path.parts)
            if not source_path.is_file():
                raise GuideContractError(f"missing source file: {source['path']}")
            section = extract_markdown_section(
                source_path.read_text(encoding="utf-8"), source["heading"]
            )
            refreshed.append(
                {
                    "path": source["path"],
                    "heading": source["heading"],
                    "sha256": section_sha256(section),
                }
            )
        topic["source_sections"] = refreshed
        topic_path.write_text(
            json.dumps(topic, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Interactive Guide sources.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--refresh", nargs="+", metavar="TOPIC_ID")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.refresh is not None:
            refresh_sources(args.refresh)
        else:
            check_content()
    except GuideContractError as exc:
        print(f"guide content contract: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
