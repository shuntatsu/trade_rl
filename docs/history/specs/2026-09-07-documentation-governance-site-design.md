# Documentation Governance and Site Design

## Status

Approved by the repository owner for implementation on `docs/full-governance-site`.

This design replaces the repository's ad-hoc documentation layout with one machine-readable documentation contract shared by humans, coding agents, CI, and the generated documentation site.

## Objective

Build a complete documentation management system in which:

- current documentation has explicit ownership and authority;
- historical design material cannot be mistaken for current runtime truth;
- agents have a deterministic entry point and topic/source-path routing;
- every current document carries validated YAML metadata;
- indexes and an agent-readable manifest are generated from that metadata rather than maintained independently;
- CI detects orphan documents, duplicate canonical ownership, folder/metadata mismatches, stale current links, and invalid source-to-doc routing;
- Zensical builds the same Markdown into a searchable static site;
- GitHub Pages deployment is isolated from software runtime behavior.

## Non-goals

- Do not alter RL, execution, evaluation, serving, research-gate, or production behavior to make documentation easier to organize.
- Do not change the meaning of maintained schemas or research evidence.
- Do not reinterpret historical plans as current truth.
- Do not delete unique historical design, evidence, licensing, provenance, security, migration, or falsification records.
- Do not make successful site generation evidence of model correctness, profitability, or production readiness.

## Boundary re-evaluation

The previous layout mixed three dimensions: subject (`architecture`), lifecycle (`implementation-plans`), and use (`operations`). The new layout makes the top-level directory represent the reader's intent and authority class:

```text
docs/
├── index.md
├── getting-started/
├── guides/
├── reference/
├── research/
├── operations/
├── performance/
├── legal/
└── history/
```

### Current authority boundaries

- `getting-started/`: first successful maintained workflow. Tutorial semantics, not exhaustive reference.
- `guides/`: task-oriented current guidance that is broader than a single operational runbook.
- `reference/`: canonical current behavior, schemas, contracts, architecture, semantics, compatibility boundaries.
- `research/`: current empirical/research state and methodology boundaries. Software implementation and empirical success remain distinct.
- `operations/`: currently executable operator runbooks only.
- `performance/`: hardware- or benchmark-specific evidence; never a general runtime contract.
- `legal/`: current licensing and provenance authority.
- `history/`: immutable or provenance-oriented specs, plans, ADRs, verification notes, old evidence, and legacy implementation records. Historical files may contain stale paths and are not current authority.

This is intentionally different from the previous `docs/architecture/`, `docs/implementation/`, and `docs/implementation-plans/` split because those names encode chronology or subject without reliably encoding authority.

## Target physical layout

```text
AGENTS.md
README.md
START.md
mkdocs.yml
scripts/docs/
  __init__.py
  model.py
  metadata.py
  manifest.py
  generate.py
  validate.py
  context.py

docs/
  index.md
  .meta.yml
  getting-started/
    .meta.yml
    index.md
    quickstart.md
  guides/
    .meta.yml
    index.md
    binance-data.md
  reference/
    .meta.yml
    index.md
    architecture.md
    configuration.md
    single-symbol.md
    universal-trade-rl.md
    universal-training.md
    reward-objective.md
    execution-robustness.md
    run-reporting.md
    nautilus-migration.md
  research/
    .meta.yml
    index.md
    status.md
    multi-timeframe.md
  operations/
    .meta.yml
    index.md
    docker-gpu-full-training.md
    causal-scenario-c3-execution.md
  performance/
    .meta.yml
    index.md
    4070ti-super-full-training.md
  legal/
    .meta.yml
    index.md
    licensing.md
    licensing-provenance.md
  history/
    .meta.yml
    index.md
    architecture/
    specs/
    plans/
    evidence/
    legacy/
```

`START.md` remains a compatibility entry point and links to the canonical `docs/getting-started/quickstart.md`. `README.md` remains the repository landing page. `frontend/README.md` remains code-local documentation and is linked from the docs portal rather than duplicated.

## Metadata contract

Current site documents use YAML front matter plus inherited `.meta.yml` defaults. Effective metadata is the deterministic merge of ancestor `.meta.yml` files followed by page front matter.

Required effective fields for every current page:

```yaml
title: string
doc_type: tutorial | guide | reference | research | runbook | performance | legal | index
lifecycle: current | historical | deprecated
authority: canonical | supporting | evidence | none
topics: [non-empty strings]
```

Optional governance fields:

```yaml
source_of_truth_for: [unique canonical keys]
related_code: [repository path prefixes or glob-like prefixes]
agent_read_when: [human-readable triggers]
nav_order: integer
description: string
```

History inherits:

```yaml
doc_type: history
lifecycle: historical
authority: none
```

Historical pages are exempt from current-source path/link freshness requirements because preserving historical state is the purpose of the archive. They remain subject to containment, UTF-8 readability, and site classification rules.

## Canonical ownership

A `source_of_truth_for` key may be owned by exactly one `authority: canonical`, `lifecycle: current` document. Duplicate canonical keys fail CI.

Initial keys include:

- `repository-documentation-governance`
- `runtime-architecture`
- `training-configuration`
- `single-symbol-contract`
- `universal-trade-rl-contract`
- `universal-training-contract`
- `reward-semantics`
- `execution-model`
- `run-reporting-contract`
- `nautilus-compatibility`
- `research-status`
- `multi-timeframe-research-boundary`
- `binance-public-data`
- `operations-docker-gpu-training`
- `operations-causal-scenario-c3`
- `licensing`
- `licensing-provenance`

## Agent contract

Root `AGENTS.md` is the bootstrap document. Agents must:

1. read `AGENTS.md` and `docs/index.md` before material repository changes;
2. call `python scripts/docs/context.py --path <changed-path>` or topic mode when documentation context is not obvious;
3. read returned canonical current docs before implementation;
4. never use `docs/history/` as current runtime authority;
5. update the owning canonical document when a public/current contract changes;
6. run documentation validation and site build before completion when docs or governed source areas change.

Source-of-truth priority is:

1. executable source/schema and contract tests;
2. current canonical reference/legal/research documents for semantics not directly executable;
3. current guides/runbooks;
4. performance/evidence documents;
5. historical material.

A contradiction between source/tests and current docs is a documentation defect; agents must not silently choose whichever statement is convenient.

## Manifest and indexes

`scripts/docs/generate.py` produces deterministic build artifacts under `.docs-build/`:

```text
.docs-build/
  manifest.json
  generated-indexes/
```

The manifest contains normalized effective metadata, canonical ownership, topics, related-code routing, and relative site paths. Generated content is build output and is not committed.

Human-facing section `index.md` files are maintained as concise explanatory portals. Machine-generated topic/authority indexes are injected into a temporary docs staging tree for site builds so generated files do not become a second source of truth.

## Source-to-doc routing

`related_code` entries use repository-relative prefixes. The context tool returns documents ordered by:

1. exact/current canonical matches;
2. current supporting documents;
3. research/operations context;
4. history only when explicitly requested.

A current canonical document with an invalid `related_code` repository root fails validation. The tool does not claim that every code change requires a docs edit; it identifies required reading and candidate documentation impact.

## Site architecture

Use Zensical `0.0.59`, pinned in commands/workflow rather than added to the application dependency lock. The site is configured through `mkdocs.yml`, which Zensical officially supports. Required capabilities are limited to functionality supported by Zensical 0.0.59: Material-compatible theme, search, meta, tags, navigation indexes, front matter, strict validation, and static build.

The site must:

- use `docs/` as source;
- expose search;
- expose section index pages;
- include current docs and history, with history visibly labeled non-authoritative;
- use relative Markdown links;
- build in strict mode on CI;
- contain repository view/edit links;
- not require JavaScript for core content readability.

## GitHub Pages boundary

A dedicated workflow builds on pull requests and pushes affecting documentation infrastructure. Deployment occurs only from `main` after a successful build.

The deployment job uses the GitHub Pages environment with minimum permissions (`contents: read`, `pages: write`, `id-token: write`). All third-party GitHub Actions are pinned to immutable commit SHAs to satisfy repository workflow-security policy.

Pages deployment is a documentation publishing capability only. It is not part of training, serving, exchange routing, empirical research gates, or production authorization.

## Migration rules

### Current documents

Current documents are moved and links are updated. Their normative meaning is preserved unless a repository-wide boundary review finds the old placement/statement itself misleading.

### History

- `docs/implementation-plans/specs/` -> `docs/history/specs/`
- `docs/implementation-plans/plans/` -> `docs/history/plans/`
- `docs/implementation-plans/evidence/` -> `docs/history/evidence/`
- root historical files under `docs/implementation-plans/` -> `docs/history/legacy/implementation-plans/`
- `docs/implementation/` -> `docs/history/legacy/implementation/`
- `docs/architecture/` -> `docs/history/architecture/`

Historical contents are not rewritten merely to modernize paths or terminology. Their old links may be stale by design and are excluded from current-link freshness validation and strict site navigation when necessary. The archive index states this explicitly.

## Failure modes

- A current document is added without metadata or without being reachable from a portal.
- Two canonical documents claim the same contract.
- A current page is placed in a folder whose inherited type contradicts page metadata.
- An agent reads a historical spec as current runtime authority.
- A source path is changed but the agent routing manifest still points to a removed path.
- Current Markdown links break after migration.
- Root README/START links keep using pre-migration paths.
- Zensical succeeds while the agent manifest is invalid, or manifest validation succeeds while the site is broken.
- Pages deployment grants unnecessary permissions or uses mutable action tags.
- Generated output is committed and drifts from source metadata.
- Documentation refactoring accidentally changes runtime code or research claims.

## Risk

Overall risk is medium because the change moves many documentation paths and changes CI/workflow surface. Runtime execution risk is low if the diff remains documentation/tooling/workflow-only. The highest risks are navigation/link breakage, silent loss of current authority, and historical/current contamination.

## Test oracle

Correctness requires observing all of the following, not just a successful site build:

- metadata parser produces deterministic effective metadata;
- malformed/unknown enum/missing required metadata fails;
- duplicate canonical ownership fails;
- folder/type mismatch fails;
- orphan current documents fail;
- current relative links resolve;
- current path claims used by routing exist;
- history is excluded from current authority and routing by default;
- context CLI returns the expected canonical docs for representative source paths/topics;
- generated manifest is deterministic;
- Zensical strict build succeeds from the generated/staged documentation source;
- workflow security checker accepts the new Pages workflow;
- existing documentation/schema/architecture contracts are updated rather than weakened;
- full repository CI remains green on the final exact HEAD.

## Required test layers

- Unit: metadata merge, schema validation, canonical uniqueness, routing, manifest determinism.
- Contract: folder classification, orphan detection, current-link resolution, AGENTS bootstrap requirements, legacy/current authority separation.
- Integration: generator + validator + context CLI against the real repository tree.
- Site build: `uvx --from zensical==0.0.59 zensical build --strict`.
- Static: Ruff/format/Mypy for new Python tooling, existing workflow security validation.
- Repository: existing architecture/documentation tests and full pytest.
- CI: same-head pull-request checks plus documentation-site workflow.

## Acceptance criteria

1. Root `AGENTS.md` exists and deterministically routes agents into current docs.
2. Current documentation is physically organized by reader intent/authority; obsolete top-level current-doc paths are removed.
3. `docs/implementation/`, `docs/implementation-plans/`, and historical `docs/architecture/` no longer compete with current docs and are consolidated under `docs/history/`.
4. Every current site page has valid effective metadata.
5. Every canonical ownership key is unique.
6. Current docs are reachable from section portals and are not orphaned.
7. Agent context lookup works by source path and topic and excludes history by default.
8. A deterministic manifest and generated indexes can be built without committing generated output.
9. Zensical 0.0.59 strict build succeeds and provides search/navigation.
10. A pinned-action GitHub Pages workflow validates PRs and deploys only from `main`.
11. Existing production `NO-GO`, profitability, causal, sealed-evaluation, licensing, and direct-exchange-routing boundaries are preserved.
12. Documentation contract tests are strengthened; no existing safety-oriented assertion is weakened merely to make migration pass.
13. Final diff contains no runtime/RL behavior change.
14. Final exact HEAD passes required targeted tests, static checks, site build, full pytest, and applicable GitHub CI.

## Invariants

- Production status stays `NO-GO` unless separately changed by real authorization/evidence.
- No profitability claim is introduced.
- Direct exchange routing remains unimplemented unless runtime code independently changes.
- Maintained single-symbol and Universal shared-policy contracts remain distinct.
- Causal and sealed evaluation boundaries are not weakened.
- Historical provenance remains recoverable by Git history and the new `docs/history/` layout.
- Generated documentation output is derivative, never authority.
- Application dependency/runtime surface does not depend on Zensical.

## Quality gate

Do not mark complete unless all acceptance criteria are mapped to a verified change; targeted and full required tests pass; workflow security passes; strict site build passes; final diff and repository status are reviewed; same-head CI is inspected; and a falsification review searches for orphan current docs, duplicate authority, stale current links, accidental runtime changes, mutable Actions, history contamination, and unverified status claims.
