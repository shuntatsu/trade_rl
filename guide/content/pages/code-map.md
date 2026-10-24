## このページの役割

ここは初見向けの学習順ではなく、概念を理解した後に**どのpackage / class / functionが何を所有するかを引く参照ページ**です。

## Ownership

| 領域 | 所有する責務 | 所有しないもの |
| --- | --- | --- |
| integrations | provider固有レスポンスを内部source evidenceへ変換 | 売買判断 |
| data | causalな`MarketDataset`、availability、staleness、dataset identity、検証済みartifactのidentity/data読込 | strategy固有の意思決定 |
| strategies | 共通観測から論理的な売買意図を返す。forecast familyは`supervised.py`でcausal training rowsとlosslessなfit cutoffを共有し、保有経過時間とminimum-hold制約も実約定数量を基準に共有する。cost-aware forecastはReplayのbudget・actual weight・filled sideを使って離散targetを選び、AdaptiveはReplay由来のgross returnと実約定quantityでprotective exitを判断する | fill、fee、accounting |
| risk | turnover、exposure、drawdown等のhard limitを適用 | entry/exitの経済判断 |
| simulation | fill、cost、funding、borrow、margin、`BookState`、新Study用の純粋なfinancial objective/clock宣言 | winner判断、Study実行 |
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
| 検証済みDataset identityとデータの同時読み込み | `trade_rl.data.load_published_market_dataset_artifact` |
| 単銘柄戦略の共通interface | `trade_rl.strategies.interface.SingleSymbolStrategy` |
| hard riskの所有者 | `trade_rl.risk.pretrade.PreTradeRisk` |
| 新しいStudyのfinancial objective / capital / clock contract | `trade_rl.simulation.objectives.ObjectiveContract` / `CapitalContract` / `BoundObjectiveClock` |
| 約定・会計の所有者 | `trade_rl.simulation.execution.MarketExecutor` |
| 単銘柄の共通評価リプレイ | `trade_rl.evaluation.replay.run_single_symbol_replay` |
| 共通資金の複数銘柄リプレイ | `trade_rl.evaluation.replay.run_shared_cash_replay` |
| 固定action sequenceによる共通資金replay | `trade_rl.evaluation.replay.run_scripted_shared_cash_replay` |
| 共通資金でのstrategy factory比較 | `trade_rl.evaluation.comparison.strategies.compare_shared_cash_strategy_factories` |
| 約定数量から保有期間を進める共通規則 | `trade_rl.strategies.position_duration.next_position_age_bars` |
| filled exposureを使うforecast intent utility | `trade_rl.strategies.forecasts.controller.CostAwareForecastIntentController.decide_for_position` |
| Ridge forecast modelのimmutable artifact | `trade_rl.strategies.forecasts.ridge_artifact.publish_ridge_forecast_artifact` |
| frozen Ridgeを使う4-arm shared-cash比較 | `trade_rl.evaluation.runs.compare_ridge_artifact_shared_cash` |
| 4-arm比較結果のwrite-once保存とstrict reload | `trade_rl.evaluation.runs.publish_ridge_comparison_artifact` / `load_ridge_comparison_artifact` |
| 4-arm比較を予約・provenance採取・失敗保持する | `trade_rl.evaluation.runs.run_ridge_comparison_attempt` |
| 保有期間Studyの固定protocol・risk条件 | `trade_rl.evaluation.experiments.contracts.study.StudyPlan` |
| 独立口座v1の適格性・選択 | `trade_rl.evaluation.experiments.protocols.ppo_holding_metrics` |
| 共通資金v2の適格性・選択 | `trade_rl.evaluation.experiments.protocols.ppo_shared_cash_holding_metrics` |

`simulation.objectives`はafter-cost ledgerから渡されたequity値を固定初期資本で
正規化するpure declaration ownerです。`BoundObjectiveClock`はUTC evaluation windowと
finite economic horizonの秒数一致を検証します。ここからcurrent PPOのreward、gamma、
terminal settlementは変更されず、ledgerやwinner selectionも新設されません。

詳細なpath、line、signature、関連testは、このページ末尾の**実装を確認する**から開けます。

共通資金の比較はstrategy armごとに全symbolを一つのaccountでreplayします。armとsymbolごとにfresh strategy adapterを使い、既存replayのcost・funding・borrow・turnover diagnosticsからmetricsを作ります。これはmodel fittingやwinner selectionを行うAPIではありません。

固定action sequence用のopt-in replayでは、`None`がactorの新しい判断なしを表し、既存targetとpending orderを進めます。`HOLD_QUANTITY`は実bookのsigned quantityを保ち、未約定remainderをcancelします。hard riskは引き続き縮小・flattenでき、既存`SingleSymbolStrategy` interfaceは変わりません。

`decide_for_position` はStrategy packageのutilityで、直接呼び出しに加えて、cost-aware設定かつbudget付きobservationを受け取るRidge / LightGBM adapterから使われます。filled quantity/weight、forecast、同じlogical intentのoptionalなcached targetから`PositionIntent`を返します。simple switch-cost proxyは帳簿feeではなく、canonical replayはその後に既存のhold制約、risk、execution、会計を適用します。

Ridge artifactは既存のfrozen model parametersと宣言されたfit contextを一つのcanonical JSONへ保存します。load時にはartifact digestと呼出し元が期待するbindingを検証します。これはfitの独立証明や評価結果ではなく、既存の`RidgeForecastStrategy`へ渡せるmodelを安全に保存・復元する契約です。

`compare_ridge_artifact_shared_cash` はpaired loaderが一回の検証で返すpublished Datasetとcaller-bound Ridge artifactを使い、cash・legacy hysteresis・position-aware zero-cost・position-aware costの4 armを同じexplicit replay条件で実行します。各armは独立したshared-cash bookを持ち、terminal settlementとledger evidenceは既存replayが担当します。返り値は使用した正確なconfigを保持します。opt-inでは、同じfrozen modelの予測とzero forecastを、同一の成熟済みlog-return pairで比較するdiagnosticsも返します。このAPIはfitやwinner選択を行わず、宣言されたbindingを過去のfit手順の独立証明として扱いません。

`publish_ridge_comparison_artifact` はその完全な4-arm返り値を一つのcanonical JSONへwrite-once保存します。必須の外部digestを使うloaderは、full ledger、returns、metrics、config、ordered symbol、Dataset/model境界、caller-supplied provenanceを検証してdeep read-only payloadを返します。returnsは`BASE_BAR`・elapsed-years未指定・共通annualizationに固定し、flat no-order cash/equity flowと利用可能なfunding event合計を照合します。cash armはtrade/rebalance、closed/open-position、realized net/gross trading P&Lがゼロで、FLATのまま正負のcash interestを保持できます。returnから再計算したinterval-end drawdownとledger intrainterval maximum drawdownは別々に扱い、前者が後者以下であることだけを検証します。ledger v2が独立fieldを持たないdelisting/split settlement proceedsを含む全fill-to-cash再構築は主張しません。保存profileは宣言データのままで、replay前後のsource/runtime採取やfit証明を追加しません。opt-in diagnosticsはartifact v2へ保存し、default v1 artifactの読込互換を保ちます。hourly horizon labelsは重なるためpair countは独立sample数ではなく、metricはprequential predictionやP&Lを示しません。このstorage capabilityだけではStudy protocol、winner選択、研究結果、profitabilityを示しません。

`run_ridge_comparison_attempt` は新しいattempt rootを入力読込前に予約し、exact config/binding/pathとcaller context digestを記録します。比較を一度実行し、その前後に既存provenance ownerを呼びます。変化がなければ結果をpublish・strict reloadし、最後にcompletionを記録します。失敗も記録し、比較が返した後なら各armのledger mappingをpartial evidenceとして残します。これはlocal software attempt boundaryであり、global retry budget、approval、fit、研究結果を追加しません。

決済専用注文の識別・受付は注文モデルと受付処理が担当します。流動性の配分処理は実際の約定順で残高を制限し、口座への約定反映と注文残量の失効はstateful実行処理が担当します。取引所固有の最小発注額の例外は、この能力とは別に検証する必要があります。

PPO環境と各replayは、分割後の数量単位に希望数量を換算します。目標を注文数量へ変換するsimulationの処理はBookStateのマーク価格を使い、注文の参照価格・limit・stopは取引価格を使います。評価と注文の価格基準を分けても、約定・会計の所有者は共通のMarketExecutorです。

保有数量がまだない口座は、同じsimulationの処理が現在の市場マーク価格を解決してからエントリー数量を求めます。初期価格を省略したBookStateも、この経路で処理できます。

`position_duration` は学習とreplayで共通の保有age規則を定義し、`contracts.study` は結果前にprotocolとriskを固定します。v1 selectorは独立口座、v2は`run_shared_cash_replay`のportfolio-level return/DD/excess/terminal stateから適格性を再計算します。bootstrapはDatasetとStudyPlanのみを準備します。

## 境界を見るときのチェック

- data層へstrategy判断を入れていないか。
- strategy層へfee/accountingを重複実装していないか。
- risk層が経済的なentry/exit判断まで所有していないか。
- evaluationが別のP&L authorityを作っていないか。
- artifactがsource/runtime provenanceから切り離されていないか。

責務が二重化すると、同じ候補でも経路によって結果が変わり、比較可能性が失われます。
