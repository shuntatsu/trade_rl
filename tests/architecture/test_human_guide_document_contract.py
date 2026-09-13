from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "guide" / "content"

EXPECTED_TOPICS = {
    "overview",
    "implementation-replay",
    "implementation-ppo",
    "code-map",
    "data-flow",
    "execution-economics",
    "experiment-loop",
    "research-status",
}
EXPECTED_GROUPS = ("overview", "mechanics", "status", "reference")
VALID_ROLES = {"overview", "detail", "status", "reference"}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_guide_uses_document_manifest_with_grouped_reading_order() -> None:
    manifest = _read_json(CONTENT / "manifest.json")

    assert manifest["schema_version"] == "document-guide-v2"
    assert manifest["home"] == "overview"
    groups = manifest["groups"]
    assert isinstance(groups, list)
    assert tuple(group["id"] for group in groups if isinstance(group, dict)) == EXPECTED_GROUPS

    grouped_topics = {
        topic
        for group in groups
        if isinstance(group, dict)
        for topic in group.get("topics", [])
        if isinstance(topic, str)
    }
    assert grouped_topics == EXPECTED_TOPICS

    reading_order = manifest["reading_order"]
    assert isinstance(reading_order, list)
    assert reading_order[0] == "overview"
    assert set(reading_order) == EXPECTED_TOPICS - {"code-map"}
    assert "code-map" not in reading_order


def test_every_guide_topic_is_one_markdown_meta_pair_with_valid_role() -> None:
    pages = CONTENT / "pages"
    meta = CONTENT / "meta"
    assert pages.is_dir()
    assert meta.is_dir()

    page_ids = {path.stem for path in pages.glob("*.md")}
    meta_ids = {path.stem for path in meta.glob("*.json")}
    assert page_ids == EXPECTED_TOPICS
    assert meta_ids == EXPECTED_TOPICS

    for topic_id in sorted(EXPECTED_TOPICS):
        document = (pages / f"{topic_id}.md").read_text(encoding="utf-8")
        metadata = _read_json(meta / f"{topic_id}.json")
        assert document.strip()
        assert metadata["id"] == topic_id
        assert metadata["role"] in VALID_ROLES
        references = metadata.get("code_references", [])
        assert isinstance(references, list)
        if metadata["role"] in {"overview", "status"}:
            assert references == []
        if metadata["role"] == "reference":
            assert references


def test_markdown_first_topics_do_not_keep_legacy_primary_topic_json() -> None:
    legacy = CONTENT / "topics"
    assert not legacy.exists()
