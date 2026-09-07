# Trade RL Agent Instructions

このRepositoryで変更を行うAgentは、実装前にこのFileと[`docs/index.md`](docs/index.md)を読み、必要なcurrent canonical documentationを特定してください。

## Documentation bootstrap

1. 変更対象のcurrent contractが明らかな場合は、そのcanonical documentを先に読む。
2. 明らかでない場合は次を実行する。Docs toolingは`dev` extraのPyYAMLを使うため、fresh cloneでも再現できるよう`--extra dev`を明示する。

```bash
uv run --extra dev python scripts/docs/context.py --path trade_rl/path/to/changed_file.py
uv run --extra dev python scripts/docs/context.py --topic universal-rl
```

3. 出力された`CANONICAL CURRENT`相当の文書を実装前に確認する。
4. `docs/history/`は設計経緯・過去Evidenceの調査にだけ使用する。**current runtime authorityではない**。
5. Historyが必要な場合だけ`--include-history`を明示する。

## Source-of-truth priority

矛盾時の優先順位は次です。

1. executable source / schema / contract tests
2. current `authority: canonical` documentation
3. current guides / runbooks
4. performance / evidence documentation
5. `docs/history/`

Source/Testとcurrent canonical documentationが矛盾する場合、どちらかを都合よく採用して進めずdocumentation defectとして確認・修正してください。

## Documentation update triggers

Public/current contractを変えた場合は、対応する`source_of_truth_for` ownerを更新します。

- Package responsibility / artifact / serving boundary → `runtime-architecture`
- Training config / schema / maintained examples → `training-configuration`
- Single-instrument action/environment contract → `single-symbol-contract`
- Universal U0-U2 contract → `universal-trade-rl-contract`
- Universal training / causal teacher / BC / warm start → `universal-training-contract`
- Reward / constraint-cost semantics → `reward-semantics`
- Execution / fill / simulation semantics → `execution-model`
- Run report artifact → `run-reporting-contract`
- Nautilus integration / compatibility → `nautilus-compatibility`
- Research stage / production gate → `research-status`
- Multi-timeframe / sealed-evaluation boundary → `multi-timeframe-research-boundary`
- Binance public data workflow → `binance-public-data`
- License / provenance → `licensing` / `licensing-provenance`

`source_of_truth_for`はcurrent canonical ownerを一意にするためのmachine-readable keyです。新しいcanonical contractを作る場合は重複keyを作らないでください。

## Documentation folder routing

詳細規則は[`docs/AGENTS.md`](docs/AGENTS.md)を参照してください。概要:

- `docs/getting-started/`: 初回成功までのTutorial
- `docs/guides/`: current task-oriented Guide
- `docs/reference/`: current canonical Reference
- `docs/research/`: current empirical/research state and methodology boundaries
- `docs/operations/`: current executable Runbook only
- `docs/performance/`: Hardware/benchmark Evidence
- `docs/legal/`: License / provenance authority
- `docs/history/`: historical Spec / Plan / ADR / Evidence; non-authoritative

## Before completion

Documentationまたはgoverned sourceを変更した場合、少なくとも次を実行し、実結果を確認します。

```bash
uv run --extra dev pytest -q tests/docs
uv run --extra dev python scripts/docs/validate.py
uv run --extra dev python scripts/docs/generate.py
uvx --from zensical==0.0.59 zensical build --strict
```

Repository全体のRequired checksはこれとは別に必要です。Documentation/site buildの成功はRL correctness、empirical success、profitability、Production authorizationを意味しません。
