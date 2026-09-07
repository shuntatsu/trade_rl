# Documentation Agent Instructions

このFileは`docs/`配下を作成・移動・更新するAgent向けの局所ルールです。Root [`AGENTS.md`](../AGENTS.md)の規則も同時に適用されます。このFile自体はdocumentation site / manifestのcontentではありません。

## Choose the directory by responsibility

新しい文書は「内容の話題」ではなく**読者の目的とauthority**で配置します。

| Directory | Place here | Do not place here |
|---|---|---|
| `getting-started/` | 初回成功までのmaintained tutorial | exhaustive schema reference |
| `guides/` | current task-oriented procedure | operator-only runbook / historical plan |
| `reference/` | current canonical behavior, schema, architecture, compatibility | chronology / experiment result |
| `research/` | current research state, empirical boundary, methodology | implementation plan / hardware benchmark |
| `operations/` | current executable operator runbook only | design, completed hardening record |
| `performance/` | hardware/benchmark-specific evidence | universal runtime contract |
| `legal/` | license and provenance authority | design rationale |
| `history/` | ADR, spec, plan, verification/evidence history | current runtime authority |

分類に迷う場合は新Directoryを増やす前に既存boundaryを再評価し、同じ責務を別名で複製しないでください。

## Metadata

Current documentのeffective metadataはancestor `.meta.yml` defaults、同Directory `.meta.yml`の`pages:` entry、page front matterの順にmergeされます。後ろほど優先です。

Current pageに必要なeffective fields:

```yaml
title: string
doc_type: tutorial | guide | reference | research | runbook | performance | legal | index
lifecycle: current | deprecated
authority: canonical | supporting | evidence | none
topics: [non-empty]
```

Canonical pageでは必要に応じて次も定義します。

```yaml
source_of_truth_for: [unique-key]
related_code: [repository-relative path or glob]
agent_read_when: [human-readable trigger]
```

Page固有のgovernance metadataは、内容を散らさず管理できる場合はsection `.meta.yml`の`pages:` mapを優先します。本文固有の表示metadataが必要ならfront matterを使えます。

## Authority rules

- 一つのcurrent contract keyには`authority: canonical` ownerを一つだけ置く。
- Secondary documentはcanonical ownerへlinkし、長いnormative sectionを複製しない。
- `history/`は常に`doc_type: history`, `lifecycle: historical`, `authority: none`。
- Current documentからhistoryを参照するときはprovenanceとして参照し、current behaviorの根拠にしない。
- Raw historyは古いpath/linkを保存し得るため、strict current-site buildの直接入力にはしない。

## Updating existing docs

実装変更時は`uv run --extra dev python scripts/docs/context.py --path ...`でcandidate ownerを確認します。Document移動時はcurrent relative links、root `README.md` / `START.md`、tests、workflows、Agent instructionsも検索してください。

`docs/index.md`は全current sectionへのportalです。各section `index.md`はそのsectionの全current content pageへlinkします。Current pageを追加してindexへ載せ忘れるとorphan contractがfailします。

## Generated outputs

`.docs-build/`と`site/`はderived outputです。Authorityではなくcommitもしません。

- `.docs-build/manifest.json`: Agent/CI用manifest
- `.docs-build/site-src/`: strict site staging
- `.docs-build/generated-indexes/`: generated topic/authority/history indexes
- `site/`: Zensical build output

Generated outputを手編集してSource of Truthにしないでください。

## Required checks

```bash
uv run --extra dev pytest -q tests/docs
uv run --extra dev python scripts/docs/validate.py
uv run --extra dev python scripts/docs/generate.py
uvx --from zensical==0.0.59 zensical build --strict
```

文書移動またはworkflow変更では、既存documentation/schema/architecture/license contract testsとworkflow-security checkも必須です。
