# テスト

[English](../TESTING.md) | **日本語**

## 必須CIゲート

すべてのmerge候補は、正確なPR headで次を合格しなければなりません。

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy mars_lite
uv run pytest --cov=mars_lite --cov-fail-under=70 tests/
```

以前のcommitで成功したrunは、新しいcommitの検証にはなりません。

## テスト層

### Unit

- ServingBundleのdigest、path、file set、schema検証
- 不変Registry登録と原子的active pointer
- 実際の現在weightを使うobservation構築
- 口座状態とpending orderの検証
- guardrailのturnover、損失、drawdown計算
- pre-trade worst-case exposure
- Bearer認証とreplay store

### Integration

- 候補構築、登録、activation、served identity
- 破損した候補が以前の健全なruntimeを維持すること
- policy prediction前に現在positionが含まれること
- feature正規化、feature mask、symbol順序、global feature順序がBundleと一致すること
- 読み取り専用Serving routeがdestructive operationを公開しないこと
- Registry activationより先にdeployment gateが実行されること
- rollbackによりserved identityがknown-good versionへ戻ること

### Adversarial

- manifest改ざんとpath traversal
- non-finite metricとthreshold上書き試行
- concurrent Registry operation
- 部分的disk copyとactive pointer失敗
- pending orderの片側約定scenario
- request replayと競合payload
- stale、NaN、all-zero市場データ

### 研究・strategy regression

P0、Walk-Forward、replay simulation、baseline、synthetic data、training testはalgorithm regressionの検出に有用です。ただし、成功してもlive profitabilityを証明しません。

## テスト置換方針

基礎となる契約が削除された場合、そのテストを削除できます。ただし、新しい契約に対して同等のsafety coverageを追加しなければなりません。古い実装を存続させるためだけにobsolete behaviorをテストへ残してはいけません。

## Slow test

Full suiteにはCPU負荷の高い学習・評価caseが含まれます。開発中はfocused testを使用できますが、merge可否はcomplete CI suiteとcoverage gateだけで判断します。

## PRに記載する証拠

PR descriptionには次を記載してください。

- 正確なhead SHA
- CI run ID
- lint、format、mypy、pytest、coverage結果
- 未テストの外部integration
- 残っているProduction blocker