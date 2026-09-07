from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.docs.manifest import build_manifest
from scripts.docs.metadata import discover_documents
from scripts.docs.validate import validate_documents

_REPOSITORY_URL = "https://github.com/shuntatsu/trade_rl"
_LINK_PATTERN = re.compile(r"(\[[^\]]+\]\()([^)]+)(\))")


def _github_repository_url(root: Path, resolved: Path) -> str:
    relative = resolved.relative_to(root).as_posix()
    view = "tree" if resolved.is_dir() else "blob"
    return f"{_REPOSITORY_URL}/{view}/main/{relative}"


def _rewrite_site_links(
    text: str,
    *,
    source: Path,
    docs_root: Path,
    root: Path,
) -> str:
    """Keep current-doc links local and route non-site repository links to GitHub."""

    root = root.resolve()
    docs_root = docs_root.resolve()
    history_root = (docs_root / "history").resolve()

    def replace(match: re.Match[str]) -> str:
        raw_target = match.group(2)
        if raw_target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        path_part, separator, anchor = raw_target.partition("#")
        if not path_part:
            return match.group(0)
        resolved = (source.parent / path_part).resolve()

        try:
            resolved.relative_to(root)
        except ValueError:
            return match.group(0)

        try:
            resolved.relative_to(history_root)
        except ValueError:
            pass
        else:
            url = _github_repository_url(root, resolved)
            if separator:
                url = f"{url}#{anchor}"
            return f"{match.group(1)}{url}{match.group(3)}"

        try:
            resolved.relative_to(docs_root)
        except ValueError:
            url = _github_repository_url(root, resolved)
            if separator:
                url = f"{url}#{anchor}"
            return f"{match.group(1)}{url}{match.group(3)}"

        return match.group(0)

    return _LINK_PATTERN.sub(replace, text)


def _copy_current_source(root: Path, site_src: Path) -> None:
    docs_root = root / "docs"
    documents = discover_documents(docs_root)
    current_paths = {
        document.path for document in documents if document.lifecycle == "current"
    }

    for relative in sorted(current_paths, key=lambda path: path.as_posix()):
        source = docs_root / relative
        target = site_src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = _rewrite_site_links(
            source.read_text(encoding="utf-8"),
            source=source,
            docs_root=docs_root,
            root=root,
        )
        target.write_text(text, encoding="utf-8")

    for source in sorted(docs_root.rglob("*"), key=lambda path: path.as_posix()):
        if not source.is_file() or source.suffix == ".md":
            continue
        relative = source.relative_to(docs_root)
        if relative.parts and relative.parts[0] == "history":
            continue
        target = site_src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _history_catalog(root: Path) -> str:
    documents = discover_documents(root / "docs")
    historical = sorted(
        (document for document in documents if document.lifecycle == "historical"),
        key=lambda document: document.path.as_posix(),
    )
    lines = [
        "---",
        "title: History catalog",
        "description: Historical design and verification archive; not current runtime authority.",
        "---",
        "",
        "# History catalog",
        "",
        "> **Historical / non-authoritative.** These files preserve past design, plans, and evidence. Use current reference documents for runtime behavior.",
        "",
    ]
    for document in historical:
        path = document.path.as_posix()
        url = f"{_REPOSITORY_URL}/blob/main/docs/{path}"
        lines.append(f"- [{document.title}]({url}) — `{path}`")
    lines.append("")
    return "\n".join(lines)


def _topic_index(manifest: dict[str, object]) -> str:
    topics = manifest.get("topics", {})
    assert isinstance(topics, dict)
    lines = ["# Topic index", ""]
    for topic, paths in sorted(topics.items()):
        lines.extend((f"## {topic}", ""))
        assert isinstance(paths, list)
        for path in paths:
            path_text = str(path)
            if path_text.startswith("history/"):
                url = f"{_REPOSITORY_URL}/blob/main/docs/{path_text}"
                lines.append(f"- [{path_text}]({url}) — historical")
            else:
                relative_from_generated = f"../{path_text}"
                lines.append(f"- [{path_text}]({relative_from_generated})")
        lines.append("")
    return "\n".join(lines)


def _authority_index(manifest: dict[str, object]) -> str:
    owners = manifest.get("canonical_owners", {})
    assert isinstance(owners, dict)
    lines = ["# Canonical authority index", ""]
    for key, path in sorted(owners.items()):
        lines.append(f"- `{key}` → [{path}](../{path})")
    lines.append("")
    return "\n".join(lines)


def write_build_outputs(root: Path, output_root: Path) -> None:
    root = root.resolve()
    output_root = output_root.resolve()
    errors = validate_documents(root)
    if errors:
        raise ValueError("documentation validation failed:\n" + "\n".join(errors))

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    manifest = build_manifest(root)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    site_src = output_root / "site-src"
    site_src.mkdir()
    _copy_current_source(root, site_src)

    history_index = site_src / "history" / "index.md"
    history_index.parent.mkdir(parents=True, exist_ok=True)
    history_index.write_text(_history_catalog(root), encoding="utf-8")

    generated = site_src / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    topic_text = _topic_index(manifest)
    authority_text = _authority_index(manifest)
    (generated / "topics.md").write_text(topic_text, encoding="utf-8")
    (generated / "authority.md").write_text(authority_text, encoding="utf-8")

    generated_indexes = output_root / "generated-indexes"
    generated_indexes.mkdir()
    (generated_indexes / "topics.md").write_text(topic_text, encoding="utf-8")
    (generated_indexes / "authority.md").write_text(authority_text, encoding="utf-8")
    (generated_indexes / "history.md").write_text(
        _history_catalog(root),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Trade RL documentation build inputs"
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output", default=".docs-build")
    args = parser.parse_args(argv)
    try:
        write_build_outputs(Path(args.root), Path(args.output))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Documentation build inputs written to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
