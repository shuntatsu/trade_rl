# Full Documentation Governance and Site Design

## Status

Approved design for repository-wide documentation governance, agent routing, automated indexes, and static documentation publishing.

This file is a historical design artifact. After implementation, current documentation governance is authoritative in root `AGENTS.md`, `docs/AGENTS.md`, `docs/README.md`, documentation metadata, tooling, and tests.

## Objective

Turn Trade RL documentation into one typed, machine-verifiable system shared by humans, agents, CI, and the generated documentation site.

The system must make it mechanically clear which documents describe current runtime contracts, current research state, executable operations, benchmark evidence, legal/provenance requirements, or historical design material. It must route agents from changed source paths to the canonical documents they need to read and review.

## Non-goals

- Do not change RL, market, reward, execution, evaluation, serving, or research semantics merely to simplify documentation.
- Do not promote empirical status, profitability, or production authorization.
- Do not make raw historical plans/specifications current authority.
- Do not require generated site output to be committed.
- Do not introduce an application server or CMS; the published site remains static.

## Boundary model

Filesystem layout encodes the strongest lifecycle boundary, while YAML metadata encodes finer document responsibility.

- `docs/` top-level maintained documents: current canonical/supporting reference and research documentation.
- `docs/operations/`: executable current operator runbooks only.
- `docs/performance/`: hardware/benchmark-specific evidence only.
- `docs/history/`: ADRs, designs, plans, verification records, and legacy material; always historical and non-authoritative for current runtime behavior.
- `docs/AGENTS.md`: local instructions; excluded from published content and the documentation manifest.

Fine-grained website sections such as Getting Started, Reference, Research, Operations, Performance, Legal, and History are generated from metadata rather than forcing every current source document through path churn. This avoids making physical layout duplicate the metadata model while still enforcing the critical current/history boundary physically.

## Metadata contract

Current effective metadata is loaded from ancestor `.meta.yml` files and optional page front matter. The effective fields are:

- `title`
- `doc_type`: `tutorial`, `reference`, `research`, `runbook`, `performance`, `legal`, or `index`
- `lifecycle`: `current` or `deprecated`
- `authority`: `canonical`, `supporting`, `evidence`, or `none`
- `topics`: non-empty list
- `source_of_truth_for`: canonical ownership keys
- `related_code`: repository-relative file paths or glob patterns used for agent routing
- `agent_read_when`: human-readable routing triggers
- `site.section` and `site.order`: generated navigation placement

Historical documents are normalized by location to `doc_type: history`, `lifecycle: historical`, and `authority: none`, regardless of legacy text inside the document.

## Authority rules

- A `source_of_truth_for` key has at most one current canonical owner.
- Historical material can explain provenance but cannot own current contract keys.
- A source/current-doc conflict is a repository inconsistency to resolve; agents must not silently prefer history.
- Current secondary documents link to the canonical owner instead of duplicating long normative definitions.

## Agent entry points

Root `AGENTS.md` is the repository bootstrap contract. It directs agents to:

1. read `docs/README.md`;
2. run `python scripts/docs/context.py --path <changed-path>` before changing a subsystem;
3. read returned canonical documents before implementation;
4. update the owning document when an observable current contract changes;
5. run documentation validation and generation before completion.

`docs/AGENTS.md` defines documentation-specific placement and metadata rules.

## Generated outputs

`python scripts/docs/generate.py` creates `.docs-build/` from source documentation and metadata:

- `manifest.json`: deterministic agent/CI document catalog;
- `site-src/`: current pages rewritten into logical site sections;
- generated home, section, topic, authority, and history indexes;
- `mkdocs.generated.yml`: deterministic Zensical-compatible site configuration.

Raw `docs/history/**/*.md` is indexed but is not copied wholesale into strict site input. Historical files can preserve old paths and chronology without weakening current-site link validation.

Generated outputs are derived and uncommitted.

## Site and publishing

The site is built with pinned Zensical `0.0.59` in strict mode. This release is selected because it is the current immutable upstream release at implementation time and includes the completed core metadata/navigation plugin replacements and CJK search support from the current Zensical line.

Pull requests build the site as a quality gate. `main` builds and deploys the exact verified site artifact to GitHub Pages through immutable-pinned official GitHub Actions.

Publishing is a presentation concern only. Site build/deploy success does not imply software, research, profitability, or production readiness.

## Documentation contract tests

Required oracles:

- every current Markdown content page has valid effective metadata;
- every historical page is normalized non-authoritative;
- canonical ownership keys are unique;
- `related_code` patterns used for agent routing resolve to repository content;
- changed source paths deterministically route to relevant current docs;
- current relative Markdown links resolve;
- obsolete parallel history roots (`docs/implementation/`, `docs/implementation-plans/`, current-looking `docs/architecture/`) do not remain after migration;
- generated site contains all current content and generated indexes but not raw history;
- generated manifest/configuration is deterministic;
- strict Zensical build succeeds.

## Failure modes

- A historical design is mistaken for current behavior.
- Two documents claim the same canonical contract key.
- A new current document lacks metadata and is silently omitted.
- A code change has no discoverable documentation owner.
- Physical moves create broken current links.
- Raw history breaks the strict site build.
- Site build succeeds while the agent manifest is incomplete.
- GitHub Pages workflow introduces mutable action references or unsafe write permissions.
- Generated files are edited or committed as authority.

## Invariants

- Production remains `NO-GO` unless independently changed by actual evidence and authorization.
- No profitability claim is introduced.
- Direct exchange routing remains unimplemented unless runtime implementation changes independently.
- Current one-run/one-instrument and Universal shared-policy research contracts remain distinct.
- Causal, sealed-evaluation, licensing, security, and provenance boundaries are preserved.
- Historical content is retained, not rewritten to imply current semantics.

## Acceptance criteria

1. Root and docs-scoped Agent instructions exist and agree on authority/routing rules.
2. Current docs are represented by valid YAML metadata and one deterministic manifest.
3. Old history roots are consolidated under `docs/history/` without converting history into current authority.
4. Agent path routing identifies canonical docs for important source areas including rewards, configuration, architecture, Universal workflows, execution, evaluation, serving, licensing, and operations.
5. Current indexes/site navigation are generated from metadata rather than manually duplicating exhaustive lists.
6. Zensical strict build succeeds from generated site source.
7. Pull-request CI validates metadata, links, manifest generation, and site build.
8. `main` has an immutable-action GitHub Pages deployment workflow whose site artifact is built from the exact checked-out SHA.
9. Existing documentation/schema/architecture/license safety assertions are updated to the new lifecycle boundary rather than weakened or skipped.
10. Final verification includes targeted docs tests, workflow security, static checks, full tests, site build, diff review, independent falsification review, and same-head CI.

## Quality gate

Do not mark this work complete unless all acceptance criteria are mapped to concrete changes, the final diff contains no unintended runtime changes, targeted and full required checks pass, Zensical strict build passes, workflow security passes, the PR's final exact HEAD has successful required CI, and remaining unverified external Pages-enablement state is reported separately if repository settings prevent deployment.
