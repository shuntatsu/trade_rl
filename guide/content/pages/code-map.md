## このページの役割

ここは初見向けの学習順ではなく、概念を理解した後に**どのpackage / class / functionが何を所有するかを引く参照ページ**です。

## Ownership

| 領域 | 所有する責務 | 所有しないもの |
| --- | --- | --- |
| integrations | provider固有レスポンスを内部source evidenceへ変換 | 売買判断 |
| data | causalな`MarketDataset`、availability、staleness、dataset identity | strategy固有の意思決定 |
| strategies | 共通観測から論理的な売買意図を返す | fill、fee、accounting |
| risk | turnover、exposure、drawdown等のhard limitを適用 | entry/exitの経済判断 |
| simulation | fill、cost、funding、borrow、margin、`BookState` | winner判断 |
| evaluation | Dataset / Strategy / Risk / Executionを共通評価経路へ結ぶ | Dataset構築 |
| artifacts / research evidence | runや研究判断を再現可能な証拠として固定 | strategy runtime |

## Dependency direction

```text
integrations
    ↓
data
    ↓
strategies
    ↓
risk
    ↓
simulation
    ↓
evaluation
    ↓
artifacts / evidence
```

この矢印は説明上の責務・データの方向です。見た目の近さや配列順から存在しない依存を追加しません。

## 主要な実装入口

| 日本語での役割 | 実identifier |
| --- | --- |
| 市場データの共通契約 | `trade_rl.data.market.MarketDataset` |
| 単銘柄戦略の共通interface | `trade_rl.strategies.interface.SingleSymbolStrategy` |
| hard riskの所有者 | `trade_rl.risk.pretrade.PreTradeRisk` |
| 約定・会計の所有者 | `trade_rl.simulation.execution.MarketExecutor` |
| 共通評価リプレイ | `trade_rl.evaluation.replay.run_single_symbol_replay` |

詳細なpath、line、signature、関連testは、このページ末尾の**実装を確認する**から開けます。

## 境界を見るときのチェック

- data層へstrategy判断を入れていないか。
- strategy層へfee/accountingを重複実装していないか。
- risk層が経済的なentry/exit判断まで所有していないか。
- evaluationが別のP&L authorityを作っていないか。
- artifactがsource/runtime provenanceから切り離されていないか。

責務が二重化すると、同じ候補でも経路によって結果が変わり、比較可能性が失われます。
