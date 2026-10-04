## このページの役割

ここは初見向けの学習順ではなく、概念を理解した後に**どのpackage / class / functionが何を所有するかを引く参照ページ**です。

## Ownership

| 領域 | 所有する責務 | 所有しないもの |
| --- | --- | --- |
| integrations | provider固有レスポンスを内部source evidenceへ変換 | 売買判断 |
| data | causalな`MarketDataset`、availability、staleness、dataset identity | strategy固有の意思決定 |
| strategies | 共通観測から論理的な売買意図を返す。保有経過時間とminimum-hold制約も実約定数量を基準に共有し、AdaptiveはReplay由来のgross returnと実約定quantityでprotective exitを判断する | fill、fee、accounting |
| risk | turnover、exposure、drawdown等のhard limitを適用 | entry/exitの経済判断 |
| simulation | fill、cost、funding、borrow、margin、`BookState` | winner判断 |
| evaluation | Dataset / Strategy / Risk / Executionを共通評価経路へ結ぶ。研究protocolごとの適格性・勝者選択を明示する | Dataset構築、strategy runtime |
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
| 単銘柄の共通評価リプレイ | `trade_rl.evaluation.replay.run_single_symbol_replay` |
| 共通資金の複数銘柄リプレイ | `trade_rl.evaluation.replay.run_shared_cash_replay` |
| 約定数量から保有期間を進める共通規則 | `trade_rl.strategies.position_duration.next_position_age_bars` |
| 保有期間Studyの固定protocol・risk条件 | `trade_rl.evaluation.experiments.contracts.study.StudyPlan` |
| 独立口座v1の適格性・選択 | `trade_rl.evaluation.experiments.protocols.ppo_holding_metrics` |
| 共通資金v3の適格性・選択 | `trade_rl.evaluation.experiments.protocols.ppo_shared_cash_holding_metrics` |

詳細なpath、line、signature、関連testは、このページ末尾の**実装を確認する**から開けます。

共有資金Candidate Run v11は順序付き会計遷移と、約定で受理された正確な数量およびBookStateに適用された数量差分を保存し、loaderが保存入力から残高・約定・carryなどを再計算します。loader自体はDataset digestから元のsource rowsを再読込しません。一方、現行の共通資金v3 EvidenceSetの生成経路は、open / mark価格、fundingの発生・rate・timestamp・multiplier、split、delisting、dividend、cash / borrow rate、経過時間を元Datasetの該当行へ照合します。約定価格・数量・notional・流動性上限・fee・spread・impactも、元Datasetと固定済みexecution configから再計算します。さらにcanonical Dataset identityを全identity arrayから再検証します。source行だけを変えるmutation testと、volume・参加上限・fee・closeなどのDataset identity arrayを変えるtestは、保存ledgerを固定したまま拒否されます。loaderはsource整合性を検証しますが、利益性を証明しません。Run Core外から固定済みexecution overlayを解決する場合は、公開facadeの`trade_rl.evaluation.runs.execution_cost_for_overlay`を使います。独立G0-G2レビューが済むまでは経済実験を許可しません。

決済専用注文の識別・受付は注文モデルと受付処理が担当します。流動性の配分処理は実際の約定順で残高を制限し、口座への約定反映と注文残量の失効はstateful実行処理が担当します。取引所固有の最小発注額の例外は、この能力とは別に検証する必要があります。

PPO環境と各replayは、分割後の数量単位に希望数量を換算します。目標を注文数量へ変換するsimulationの処理はBookStateのマーク価格を使い、注文の参照価格・limit・stopは取引価格を使います。評価と注文の価格基準を分けても、約定・会計の所有者は共通のMarketExecutorです。

保有数量がまだない口座は、同じsimulationの処理が現在の市場マーク価格を解決してからエントリー数量を求めます。初期価格を省略したBookStateも、この経路で処理できます。

`position_duration` は学習とreplayで共通の保有age規則を定義し、`contracts.study` は結果前にprotocolとriskを固定します。v1 selectorは独立口座、現行v3 selectorは`run_shared_cash_replay`のportfolio-level return/DD/excess/terminal stateから適格性を再計算します。historical v2は旧realized-DD条件を保ったread-only protocolです。bootstrapはDatasetとStudyPlanのみを準備します。

## 境界を見るときのチェック

- data層へstrategy判断を入れていないか。
- strategy層へfee/accountingを重複実装していないか。
- risk層が経済的なentry/exit判断まで所有していないか。
- evaluationが別のP&L authorityを作っていないか。
- artifactがsource/runtime provenanceから切り離されていないか。

責務が二重化すると、同じ候補でも経路によって結果が変わり、比較可能性が失われます。
