from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GUIDE_CONTENT = ROOT / "guide" / "content"
TOPICS_DIR = GUIDE_CONTENT / "topics"
MANIFEST_PATH = GUIDE_CONTENT / "manifest.json"

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_CODE_KINDS = {"function", "class", "method"}
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


def _topic_code_references(topic: dict[str, Any], *, topic_id: str) -> list[dict[str, Any]]:
    raw_references = topic.get("code_references", [])
    if not isinstance(raw_references, list):
        raise GuideContractError(f"topic {topic_id} code_references must be a list")
    references: list[dict[str, Any]] = []
    ids: list[str] = []
    for raw in raw_references:
        if not isinstance(raw, dict):
            raise GuideContractError(f"topic {topic_id} has malformed code reference")
        reference_id = raw.get("id")
        if not isinstance(reference_id, str) or not reference_id:
            raise GuideContractError(f"topic {topic_id} has malformed code reference id")
        ids.append(reference_id)
        references.append(raw)
    if len(set(ids)) != len(ids):
        raise GuideContractError(f"topic {topic_id} code_references contains duplicate ids")
    return references


def _required_string(record: dict[str, Any], field: str, *, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise GuideContractError(f"{label} requires non-empty {field}")
    return value


def _code_reference_test_path(root: Path, path_text: str) -> Path:
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "tests"
        or path.suffix != ".py"
    ):
        raise GuideContractError(f"invalid code reference test path: {path_text}")
    resolved_root = root.resolve()
    resolved = (root / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise GuideContractError(f"invalid code reference test path: {path_text}") from exc
    if not resolved.is_file():
        raise GuideContractError(f"invalid code reference test path: {path_text}")
    return resolved


def validate_code_reference(
    root: Path,
    reference: dict[str, object],
    symbols: dict[str, dict[str, object]],
    *,
    check_digest: bool = True,
) -> None:
    raw = dict(reference)
    reference_id = _required_string(raw, "id", label="code reference")
    symbol_name = _required_string(raw, "symbol", label=f"code reference {reference_id}")
    expected_kind = _required_string(raw, "kind", label=f"code reference {reference_id}")
    if expected_kind not in _ALLOWED_CODE_KINDS:
        raise GuideContractError(f"invalid code symbol kind: {expected_kind}")
    _required_string(raw, "label_ja", label=f"code reference {reference_id}")
    _required_string(raw, "description_ja", label=f"code reference {reference_id}")

    symbol = symbols.get(symbol_name)
    if symbol is None:
        raise GuideContractError(f"missing code symbol: {symbol_name}")
    actual_kind = symbol.get("kind")
    if actual_kind != expected_kind:
        raise GuideContractError(
            f"code symbol kind mismatch: {symbol_name}: expected {expected_kind}, actual {actual_kind}"
        )

    expected_digest = raw.get("source_sha256")
    if not isinstance(expected_digest, str) or not _HEX_64_RE.fullmatch(expected_digest):
        raise GuideContractError(f"invalid code symbol fingerprint: {symbol_name}")
    actual_digest = symbol.get("source_sha256")
    if check_digest and actual_digest != expected_digest:
        raise GuideContractError(
            f"stale code symbol fingerprint: {symbol_name}: "
            f"expected {expected_digest}, actual {actual_digest}"
        )

    local_names = symbol.get("local_names")
    if not isinstance(local_names, list) or not all(
        isinstance(name, str) for name in local_names
    ):
        raise GuideContractError(f"malformed code symbol local names: {symbol_name}")
    allowed_names = set(local_names)
    raw_variables = raw.get("variables", [])
    if not isinstance(raw_variables, list):
        raise GuideContractError(f"code reference {reference_id} variables must be a list")
    variable_names: list[str] = []
    for variable in raw_variables:
        if not isinstance(variable, dict):
            raise GuideContractError(f"code reference {reference_id} has malformed variable")
        name = _required_string(variable, "name", label=f"code reference {reference_id} variable")
        _required_string(variable, "label_ja", label=f"code reference {reference_id} variable")
        _required_string(
            variable,
            "description_ja",
            label=f"code reference {reference_id} variable",
        )
        variable_names.append(name)
        if name not in allowed_names:
            raise GuideContractError(f"unknown code variable: {symbol_name}.{name}")
    if len(set(variable_names)) != len(variable_names):
        raise GuideContractError(f"code reference {reference_id} has duplicate variables")

    raw_tests = raw.get("tests", [])
    if not isinstance(raw_tests, list) or not all(
        isinstance(path, str) and path for path in raw_tests
    ):
        raise GuideContractError(f"code reference {reference_id} tests must be string paths")
    for test_path in raw_tests:
        _code_reference_test_path(root, test_path)


def _symbol_map(root: Path) -> dict[str, dict[str, object]]:
    code_symbols = _code_symbols_module()
    index = code_symbols.build_symbol_index(root / "trade_rl", revision="0" * 40)
    raw_symbols = index.get("symbols")
    if not isinstance(raw_symbols, list):
        raise GuideContractError("code symbol index has malformed symbols")
    symbols: dict[str, dict[str, object]] = {}
    for raw in raw_symbols:
        if not isinstance(raw, dict):
            raise GuideContractError("code symbol index has malformed entry")
        name = raw.get("qualified_name")
        if not isinstance(name, str) or not name:
            raise GuideContractError("code symbol index has malformed qualified name")
        symbols[name] = raw
    return symbols


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

    symbols = _symbol_map(root)
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
        for reference in _topic_code_references(topic, topic_id=topic_id):
            validate_code_reference(root, reference, symbols)


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


def refresh_code_references(topic_ids: list[str], root: Path = ROOT) -> None:
    if not topic_ids:
        raise GuideContractError("refresh-code requires at least one explicit topic id")

    topics_dir = root / "guide" / "content" / "topics"
    symbols = _symbol_map(root)
    for topic_id in topic_ids:
        topic_path = topics_dir / f"{topic_id}.json"
        if not topic_path.is_file():
            raise GuideContractError(f"missing topic for refresh-code: {topic_id}")
        topic = _read_json(topic_path)
        if topic.get("id") != topic_id:
            raise GuideContractError(f"topic id mismatch: {topic_path}")
        references = _topic_code_references(topic, topic_id=topic_id)
        for reference in references:
            validate_code_reference(
                root,
                reference,
                symbols,
                check_digest=False,
            )
            symbol_name = str(reference["symbol"])
            reference["source_sha256"] = symbols[symbol_name]["source_sha256"]
        topic["code_references"] = references
        topic_path.write_text(
            json.dumps(topic, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Interactive Guide sources.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--refresh", nargs="+", metavar="TOPIC_ID")
    mode.add_argument("--refresh-code", nargs="+", metavar="TOPIC_ID")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.refresh is not None:
            refresh_sources(args.refresh)
        elif args.refresh_code is not None:
            refresh_code_references(args.refresh_code)
        else:
            check_content()
    except GuideContractError as exc:
        print(f"guide content contract: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
