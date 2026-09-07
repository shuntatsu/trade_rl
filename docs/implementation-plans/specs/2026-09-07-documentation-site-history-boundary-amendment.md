# Documentation Site History Boundary Amendment

## Status

Approved implementation-time amendment to `2026-09-07-documentation-governance-site-design.md`.

## Reason for amendment

The original design required both current documentation and the raw historical archive to be direct inputs to the strict Zensical site build while also requiring historical Markdown to remain unchanged, including stale historical paths and links. Those requirements conflict: a strict current-site link oracle would incorrectly turn intentionally preserved historical references into current documentation failures.

This is an authority-boundary defect, not a reason to weaken strict validation.

## Revised boundary

The repository keeps all raw historical Markdown under `docs/history/` and includes it in the documentation manifest as `lifecycle: historical`, `authority: none`.

The strict site source is generated into `.docs-build/site-src/` and contains:

1. every current documentation page;
2. generated current section/topic/authority indexes;
3. a generated History catalog page that lists archived documents and links to their repository source.

Raw historical page bodies are **not copied into the strict site source**. They remain preserved and browsable in GitHub and are available to agents only when history is explicitly requested.

## Invariants preserved

- No historical file is deleted merely because it contains stale paths.
- Historical documents remain discoverable through the manifest and generated catalog.
- Historical documents remain excluded from current runtime authority and default agent routing.
- Strict site build remains a real oracle for all published current documentation links.
- Current documentation may link to historical material only as provenance, never as normative current behavior.

## Acceptance criteria amendment

The original criterion "include current docs and history" is replaced with:

> Publish all current documentation under strict validation and publish an automatically generated history catalog that provides access to the preserved repository archive without treating raw historical Markdown as current site input.

All other acceptance criteria and quality gates remain unchanged.
