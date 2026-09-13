from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import unicodedata
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GUIDE_CONTENT = ROOT / "guide" / "content"
META_DIR = GUIDE_CONTENT / "meta"
PAGES_DIR = GUIDE_CONTENT / "pages"
MANIFEST_PATH = GUIDE_CONTENT / "manifest.json"
LEGACY_TOPICS_DIR = GUIDE_CONTENT / "topics"

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_CODE_KINDS = {"function", "class", "method"}
_ALLOWED_ROLES = {"overview", "detail", "status", "reference"}
_REQUIRED_GROUP_IDS = ("overview", "mechanics", "status", "reference")


class GuideContractError(ValueError):
    """Raised when the human guide no longer matches its source contract."""


def _code_symbols_module() -> ModuleType:
    module_name = "guide.tools.code_symbols" if __package__ else "code_symbols"
    return importlib.import_module(module_name)


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


def _heading_slug(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    parts: list[str] = []
    pending_separator = False
    for character in normalized:
        if character.isalnum():
            if pending_separator and parts:
                parts.append("-")
            parts.append(character)
            pending_separator = False
        else:
            pending_separator = True
    return "".join(parts).strip("-")


def validate_markdown_page(path: Path) -> None:
    if not path.is_file():
        raise GuideContractError(f"missing Markdown page: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise GuideContractError(f"empty Markdown page: {path}")
    _, headings = _headings(text)
    slugs: list[str] = []
    for _line, _level, title in headings:
        slug = _heading_slug(title)
        if not slug:
            raise GuideContractError(f"Markdown heading has no stable slug: {path}: {title}")
        slugs.append(slug)
    if len(set(slugs)) != len(slugs):
        raise GuideContractError(f"duplicate normalized Markdown heading slug: {path}")


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuideContractError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GuideContractError(f"JSON root must be an object: {path}")
    return value


def _required_string(record: dict[str, Any], field: str, *, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GuideContractError(f"{label} requires non-empty {field}")
    return value.strip()


def _manifest_topic_ids(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    if manifest.get("schema_version") != "document-guide-v2":
        raise GuideContractError("manifest schema must be document-guide-v2")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        raise GuideContractError("manifest groups must be a non-empty list")
    group_ids: list[str] = []
    topic_ids: list[str] = []
    for index, raw in enumerate(groups):
        if not isinstance(raw, dict):
            raise GuideContractError(f"manifest group {index} is malformed")
        group_id = _required_string(raw, "id", label=f"manifest group {index}")
        _required_string(raw, "label", label=f"manifest group {group_id}")
        topics = raw.get("topics")
        if not isinstance(topics, list) or not topics:
            raise GuideContractError(f"manifest group {group_id} topics must be non-empty")
        if not all(isinstance(topic, str) and topic for topic in topics):
            raise GuideContractError(f"manifest group {group_id} topic ids are malformed")
        if len(set(topics)) != len(topics):
            raise GuideContractError(f"manifest group {group_id} contains duplicate topics")
        group_ids.append(group_id)
        topic_ids.extend(topics)
    if tuple(group_ids) != _REQUIRED_GROUP_IDS:
        raise GuideContractError(
            f"manifest groups must be {_REQUIRED_GROUP_IDS!r}, got {tuple(group_ids)!r}"
        )
    if len(set(topic_ids)) != len(topic_ids):
        raise GuideContractError("manifest topic ids must be globally unique")

    home = manifest.get("home")
    if not isinstance(home, str) or home not in topic_ids:
        raise GuideContractError("manifest home must reference a topic")
    reading_order = manifest.get("reading_order")
    if not isinstance(reading_order, list) or not reading_order:
        raise GuideContractError("manifest reading_order must be non-empty")
    if not all(isinstance(topic, str) and topic for topic in reading_order):
        raise GuideContractError("manifest reading_order contains invalid topic ids")
    if len(set(reading_order)) != len(reading_order):
        raise GuideContractError("manifest reading_order contains duplicate topic ids")
    if reading_order[0] != home:
        raise GuideContractError("manifest reading_order must start at home")
    if "code-map" in reading_order:
        raise GuideContractError("reference code-map must not be in default reading_order")
    unknown = set(reading_order) - set(topic_ids)
    if unknown:
        raise GuideContractError(f"reading_order references unknown topics: {sorted(unknown)}")
    return topic_ids, reading_order


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
        digest = raw.get("sha256")
        if not all(isinstance(value, str) for value in (path, heading, digest)):
            raise GuideContractError(f"topic {topic_id} has malformed source section")
        sections.append({"path": path, "heading": heading, "sha256": digest})
    return sections


def validate_source_sections(root: Path, source_sections: list[dict[str, str]]) -> None:
    resolved_root = root.resolve()
    for source in source_sections:
        path_text = source["path"]
        heading = source["heading"]
        expected = source["sha256"]
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
        section = extract_markdown_section(source_path.read_text(encoding="utf-8"), heading)
        actual = section_sha256(section)
        if actual != expected:
            raise GuideContractError(
                "stale source fingerprint: "
                f"{path_text}#{heading}: expected {expected}, actual {actual}"
            )


def _topic_code_references(topic: dict[str, Any], *, topic_id: str) -> list[dict[str, Any]]:
    raw_references = topic.get("code_references", [])
    if not isinstance(raw_references, list):
        raise GuideContractError(f"topic {topic_id} code_references must be a list")
    references: list[dict[str, Any]] = []
    ids: list[str] = []
    for raw in raw_references:
        if not isinstance(raw, dict):
            raise GuideContractError(f"topic {topic_id} has malformed code reference")
        reference_id = _required_string(raw, "id", label=f"topic {topic_id} code reference")
        ids.append(reference_id)
        references.append(raw)
    if len(set(ids)) != len(ids):
        raise GuideContractError(f"topic {topic_id} code_references contains duplicate ids")
    return references


def _validated_test_path(path_text: str) -> Path:
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "tests":
        raise GuideContractError(f"invalid Guide test path: {path_text}")
    resolved = (ROOT / Path(*path.parts)).resolve()
    try:
        resolved.relative_to((ROOT / "tests").resolve())
    except ValueError as exc:
        raise GuideContractError(f"Guide test path escapes tests/: {path_text}") from exc
    if not resolved.is_file():
        raise GuideContractError(f"missing Guide referenced test: {path_text}")
    return resolved


def _symbol_map() -> dict[str, dict[str, object]]:
    module = _code_symbols_module()
    index = module.build_symbol_index(
        ROOT / "trade_rl",
        revision=module.resolve_revision(),
    )
    raw_symbols = index.get("symbols")
    if not isinstance(raw_symbols, list):
        raise GuideContractError("generated code symbol index is malformed")
    result: dict[str, dict[str, object]] = {}
    for raw in raw_symbols:
        if isinstance(raw, dict) and isinstance(raw.get("qualified_name"), str):
            result[str(raw["qualified_name"])] = raw
    return result


def validate_code_references(topic: dict[str, Any], *, topic_id: str) -> None:
    references = _topic_code_references(topic, topic_id=topic_id)
    symbols = _symbol_map() if references else {}
    for reference in references:
        symbol_name = _required_string(reference, "symbol", label=f"topic {topic_id} reference")
        kind = _required_string(reference, "kind", label=f"topic {topic_id} reference {symbol_name}")
        expected_digest = _required_string(
            reference,
            "source_sha256",
            label=f"topic {topic_id} reference {symbol_name}",
        )
        if kind not in _ALLOWED_CODE_KINDS:
            raise GuideContractError(f"unsupported code kind in topic {topic_id}: {kind}")
        if not _HEX_64_RE.fullmatch(expected_digest):
            raise GuideContractError(f"invalid code source digest in topic {topic_id}: {symbol_name}")
        _required_string(reference, "label_ja", label=f"topic {topic_id} reference {symbol_name}")
        _required_string(
            reference,
            "description_ja",
            label=f"topic {topic_id} reference {symbol_name}",
        )

        symbol = symbols.get(symbol_name)
        if symbol is None:
            raise GuideContractError(f"missing code symbol in topic {topic_id}: {symbol_name}")
        if symbol.get("kind") != kind:
            raise GuideContractError(f"wrong code kind in topic {topic_id}: {symbol_name}")
        if symbol.get("source_sha256") != expected_digest:
            raise GuideContractError(f"stale code source digest in topic {topic_id}: {symbol_name}")

        raw_variables = reference.get("variables", [])
        if not isinstance(raw_variables, list):
            raise GuideContractError(f"variables must be a list in topic {topic_id}: {symbol_name}")
        local_names = symbol.get("local_names")
        valid_names = set(local_names) if isinstance(local_names, list) else set()
        variable_names: list[str] = []
        for raw_variable in raw_variables:
            if not isinstance(raw_variable, dict):
                raise GuideContractError(f"malformed variable in topic {topic_id}: {symbol_name}")
            name = _required_string(
                raw_variable,
                "name",
                label=f"topic {topic_id} variable for {symbol_name}",
            )
            _required_string(
                raw_variable,
                "label_ja",
                label=f"topic {topic_id} variable {name}",
            )
            _required_string(
                raw_variable,
                "description_ja",
                label=f"topic {topic_id} variable {name}",
            )
            if name not in valid_names:
                raise GuideContractError(
                    f"unknown local variable in topic {topic_id}: {symbol_name}.{name}"
                )
            variable_names.append(name)
        if len(set(variable_names)) != len(variable_names):
            raise GuideContractError(f"duplicate variables in topic {topic_id}: {symbol_name}")

        tests = reference.get("tests", [])
        if not isinstance(tests, list) or not all(isinstance(path, str) for path in tests):
            raise GuideContractError(f"tests must be string paths in topic {topic_id}: {symbol_name}")
        for test_path in tests:
            _validated_test_path(test_path)


def _validate_topic_metadata(topic: dict[str, Any], *, topic_id: str) -> None:
    if topic.get("id") != topic_id:
        raise GuideContractError(f"topic id/path mismatch: {topic_id}")
    for field in ("title", "nav_label", "summary"):
        _required_string(topic, field, label=f"topic {topic_id}")
    keywords = topic.get("keywords")
    if not isinstance(keywords, list) or not all(isinstance(value, str) for value in keywords):
        raise GuideContractError(f"topic {topic_id} keywords must be a string list")
    role = topic.get("role")
    if role not in _ALLOWED_ROLES:
        raise GuideContractError(f"topic {topic_id} has invalid role: {role!r}")
    references = _topic_code_references(topic, topic_id=topic_id)
    if role in {"overview", "status"} and references:
        raise GuideContractError(f"topic {topic_id} role {role} forbids code_references")
    if role == "reference" and not references:
        raise GuideContractError(f"topic {topic_id} reference role requires code_references")
    validate_source_sections(ROOT, _topic_source_sections(topic, topic_id=topic_id))
    validate_code_references(topic, topic_id=topic_id)


def _load_document_topics() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _read_json(MANIFEST_PATH)
    topic_ids, _reading_order = _manifest_topic_ids(manifest)
    if LEGACY_TOPICS_DIR.exists():
        raise GuideContractError("legacy guide/content/topics must be removed")
    meta_ids = {path.stem for path in META_DIR.glob("*.json")} if META_DIR.is_dir() else set()
    page_ids = {path.stem for path in PAGES_DIR.glob("*.md")} if PAGES_DIR.is_dir() else set()
    expected = set(topic_ids)
    if meta_ids != expected:
        raise GuideContractError(
            f"manifest/meta mismatch: expected {sorted(expected)}, got {sorted(meta_ids)}"
        )
    if page_ids != expected:
        raise GuideContractError(
            f"manifest/Markdown mismatch: expected {sorted(expected)}, got {sorted(page_ids)}"
        )

    topics: dict[str, dict[str, Any]] = {}
    for topic_id in topic_ids:
        topic = _read_json(META_DIR / f"{topic_id}.json")
        validate_markdown_page(PAGES_DIR / f"{topic_id}.md")
        _validate_topic_metadata(topic, topic_id=topic_id)
        topics[topic_id] = topic
    return manifest, topics


def check_contract() -> None:
    _load_document_topics()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def refresh_topic(topic_id: str) -> None:
    path = META_DIR / f"{topic_id}.json"
    topic = _read_json(path)
    if topic.get("id") != topic_id:
        raise GuideContractError(f"topic id/path mismatch: {topic_id}")
    raw_sections = topic.get("source_sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise GuideContractError(f"topic {topic_id} has no source_sections")
    for raw in raw_sections:
        if not isinstance(raw, dict):
            raise GuideContractError(f"topic {topic_id} has malformed source section")
        path_text = _required_string(raw, "path", label=f"topic {topic_id} source")
        heading = _required_string(raw, "heading", label=f"topic {topic_id} source")
        source = _authoritative_source_path(path_text)
        source_path = ROOT / Path(*source.parts)
        section = extract_markdown_section(source_path.read_text(encoding="utf-8"), heading)
        raw["sha256"] = section_sha256(section)
    _write_json(path, topic)


def refresh_code_topic(topic_id: str) -> None:
    path = META_DIR / f"{topic_id}.json"
    topic = _read_json(path)
    references = _topic_code_references(topic, topic_id=topic_id)
    symbols = _symbol_map()
    for reference in references:
        symbol_name = _required_string(reference, "symbol", label=f"topic {topic_id} reference")
        symbol = symbols.get(symbol_name)
        if symbol is None:
            raise GuideContractError(f"missing code symbol in topic {topic_id}: {symbol_name}")
        digest = symbol.get("source_sha256")
        if not isinstance(digest, str) or not _HEX_64_RE.fullmatch(digest):
            raise GuideContractError(f"malformed generated source digest: {symbol_name}")
        reference["source_sha256"] = digest
    _write_json(path, topic)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Human Guide source/code freshness.")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--refresh", metavar="TOPIC_ID")
    actions.add_argument("--refresh-code", metavar="TOPIC_ID")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.refresh:
            refresh_topic(args.refresh)
            check_contract()
        elif args.refresh_code:
            refresh_code_topic(args.refresh_code)
            check_contract()
        else:
            check_contract()
    except (GuideContractError, ValueError) as exc:
        print(f"guide content contract: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
