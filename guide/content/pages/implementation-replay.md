## このページで答える問い

1本の市場バーを処理するとき、**現在時点の観測 → 戦略判断 → hard risk → 約定・会計 → 次のBookState** はどの順番で進むのか。

重要なのは、戦略の出力をそのまま注文にしないことと、戦略ごとに別会計を持たせないことです。

## 処理順

```text
MarketDataset
    │
    │ 1. 現在時点の観測を組み立てる
    │    PPO v3では実約定後の保有年齢も含む
    ▼
Replay
    │
    │ 2. strategy.decide(observation)
    ▼
Strategy
    │
    │ 3. SHORT / FLAT / LONG を返す
    ▼
Replay
    │
    │ minimum-hold中は自発的な変更を抑制
    │ hard riskはその後に適用され、常に優先
    │ desired_quantity → proposal_weight
    │
    │ 4. hard riskを適用
    ▼
PreTradeRisk.constrain
    │
    │ target_weight
    │
    │ 5. 制約済み目標を約定・会計へ渡す
    ▼
MarketExecutor.execute_interval
    │
    │ fill / cost / interval net return
    ▼
BookState
    │
    │ 6. 帳簿と処理位置を次のバーへ引き継ぐ
    └───────────────────────────────→ Replay

最後に 7. returns / decisions / diagnostics を結果へまとめる
```

この順序はクリックしなくても全体を読めるように固定しています。

## 1. 観測を作る

現在バーのfeature、availability、staleness、global feature情報に、**現在の売買意図・現在ウェイト・実約定数量・銘柄ごとのgross budget**を加えて`StrategyObservation`を作ります。budgetはcanonical replayでは常に渡し、直接構築する従来callerでは省略できます。cached desired quantityを現在のmark/equityでweightにした`same_intent_target_weight`もoptionalで渡します。Cost-aware forecast adapterはbudgetがある場合、signed filled positionからactual sideを決め、actual side・そこからのbaseline hysteresis proposal・FLATを比較します。候補がlogical current intentと一致するときだけcached targetを使い、それ以外は`-budget` / `0` / `+budget`です。actual weightからcached targetまでのpartial-fill residualはcostに残り、same-intentのquantity holdは価格driftだけを理由に再約定しません。minimum-hold中とunlock decisionはhintを省き、既存のquantity overrideとnominal target resetを使います。これは離散intent候補のproxyで、risk projection前にpartial quantityを連続最適化するものではありません。PPO Observation v3ではさらに実際の約定数量から数えた`position_age_bars`を渡します。最後のactionではなくfillが年齢の起点です。

未来のバーを見てから観測を作ることはありません。判断に入る情報は現在時点までに利用可能なものに限定します。

Ridge / LightGBM strategy adapterは、保存済みobservation timestampが`NaT`、またはmodelの`fit_cutoff`がobservation timestampより後なら、feature-unavailable fallbackやpredictionの前に拒否します。同時刻は許します。このas-of floorはmodelの生成・配布時刻や過去のvintage availabilityを証明せず、`StrategyObservation`構築前に既にwrapされたtimestampも復元しません。

Replayは開始行と、約定またはterminal settlementが処理する各target行について、raw rowの `available_at` がその行のtimestampより後なら、その行を帳簿・risk・strategy・executionへ渡す前に停止します。single-symbolでは選択銘柄と実保有銘柄、shared-cashでは全strategy rosterが対象です。直接`MarketExecutor.execute_interval`も、有効targetのnonzero weightと帳簿のnonzero exact holdingを対象にdecision rowを同じ時計で確認し、遅着ならvaluation・reconciliation前に停止します。flat accountのzero targetでpending orderだけをcancelする場合は許します。未選択flat銘柄と処理しないtailは対象外で、`information_available=False`だけでは拒否しません。判定は前のdecision時刻ではなくtarget行自身のtimestampと比べます。明示`mark_price`・外部帳簿markの独立availability、明示`OrderIntent`実行、直接close liquidation、venue arrivalまで保証するものではありません。

## 2. 戦略がintentを返す

戦略へ現在の`StrategyObservation`を渡し、`SHORT` / `FLAT` / `LONG`の論理的なポジション意図を受け取ります。

ここではまだfillも手数料も発生していません。**戦略の責務は経済判断であって、約定や会計ではありません。**

Cost-aware設定のRidge / LightGBM adapterは、replayがper-symbol `intent_gross_budget`を渡すと、filled quantity/weightとoptionalなcached targetを使う`decide_for_position`を呼びます。same-intent holdはcached desired quantityを維持し、partial fillやpending residualもactual weightから評価します。utilityは`PositionIntent`だけを返し、fillや帳簿feeは作りません。通常のminimum-hold、hard-risk、execution、会計は共通の後段で処理されます。

PPOに最低保有期間を設定した場合、まだ期間内の実保有数量を維持するよう、FLAT・反対方向などの自発的なintent変更を抑えます。未約定の目標残量も取り消し、partial fillを実保有状態へ反映します。これは約定やriskを飛ばす仕組みではありません。

## 3. 希望数量とproposalへ変換する

intentが変わった場合、gross budgetから目標ウェイトを計算し、現在価格と現在の資産額を使って希望保有数量へ変換します。

希望数量と目標ウェイトの相互変換には、BookStateの評価用マーク価格を使います。注文の参照価格は取引価格のままです。評価価格だけが変わっても、同じintentの保有数量を買い増しません。株式分割を処理した区間では、保有数量と同じ倍率で希望数量も換算し、未約定の目標残量も新しい数量単位へ引き継ぎます。

```text
PositionIntent
    ↓
desired_quantity
    ↓
proposal_weight
```

`proposal_weight`はまだ実行可能性を保証しない、risk適用前の提案値です。

## 4. hard riskを適用する

`PreTradeRisk`はproposalへ、turnover、単一ウェイト上限、gross上限、drawdown縮小、emergency flatten、reduce-onlyなどの制約を適用します。

minimum-holdによる抑制の後にもhard riskを適用するため、drawdown stopやemergency flattenは保有期間中でも縮小・全決済できます。保有期間は決められた日数まで必ず持つ保証ではありません。

明示されたshared-cash terminal settlementでは、strategy roster内の実保有だけをterminal `emergency_flatten_mask`へ渡し、通常の`max_turnover`で最後のclose requestを縮小しません。通常のstrategy decisionsは引き続き同じturnover上限に従います。closeは設定済みのrisk・execution経路を通り、venue rule、liquidity、fill capacity、cost、pending orderの扱いもそのままです。容量や制約が足りなければ残数量・注文残を保持し、exact-flat validationは失敗します。

```text
proposal_weight
      ↓
PreTradeRisk.constrain
      ↓
target_weight
```

約定器へ渡るのは、戦略が最初に希望した値ではなく**制約後の`target_weight`**です。

ここでproposal stateとrisk projectionを混同しません。`max_turnover` が理由に含まれるtargetは、`max_abs_weight` / `max_gross` や `hard_risk_turnover_override` が同時に記録されても収束途中のstepとして扱い、縮小後quantityを次バーの `desired_quantity` へ書き戻しません。したがって価格drift後のLONG→SHORT反転でも、最初のhard-deleverage targetへproposalが凍結せず、元のSHORT proposalから次のturnover stepを計算できます。`drawdown_deleveraging` も現在のdrawdownからそのバーのtargetへ掛ける一時的な縮小であり、同様に書き戻しません。transient理由がない静的hard capだけは従来どおりbounded proposalとして再束縛します。

## 5. 約定と会計を行う

`MarketExecutor`が次の1区間を共通ルールで処理します。

- 注文とrealized fillを区別する
- fee / spread / funding / borrow等を共通経路で計上する
- partial fillならrealized quantityだけpositionへ反映する
- 区間のnet returnを作る
- 新しい`BookState`を返す

戦略ごとに別の損益計算を実装しません。

## 6. BookStateを引き継ぐ

発注単位の小数を何度も加減して誤差がたまらないよう、約定した数量と注文の残量は正確な分数として保持します。画面や学習に渡す小数表示は保有量を過大に見せない方向へ変換します。取引所の最小発注額を満たさない残高を、丸めて消す処理は行いません。

約定後の`BookState`を次のバーの共通状態として採用します。同時に今回のintentと次の処理index、累積cost・turnover等を引き継ぎます。

このため、次の観測も**実際に約定・会計された状態**から始まります。

## 7. 結果を確定する

全区間を処理した後、raw interval returns、decisions、execution diagnosticsに加えて、最後のstateful executionが持つactive order remainderとterminal order reasonを`SingleSymbolReplayResult`へまとめます。

保有期間比較のRun artifactはPPOの抑制・解除、年齢、fill前後quantity、risk理由を記録します。共通terminal settlementの後にもpositionや注文残があるseed-symbol accountは未決済として識別され、PPO期間選定のeligible armにはできません。

このためposition数量が0でも未約定orderが残っている状態を「完全にflat」と誤認せず、research hard guard側でterminal execution stateを検査できます。order evidenceは同じ`MarketExecutor`のobserverから取得し、別のexecutionを再計算しません。

比較に使うのはこの共通経路を通った結果です。

## 同時portfolio policyのopt-in経路

`run_shared_portfolio_replay`はsimulation-ownedな
`SharedPortfolioSession`を呼び出すopt-in batch consumerです。eligibleなnative
barごとにpolicyを一度だけ呼びます。sessionは同じ`MarketExecutor`・実際に返された
`BookState`・native indexを次の呼び出しへ引き継ぎ、各numeric targetを共通risk・
execution・accounting経路へ渡します。公開book viewはdetached cloneです。
`strategies/portfolio_observation.py`が現在の`MarketDataset` row、`BookState`、
active-order tuple、risk config、decision windowからsnapshotを作り、evaluation replay
はsource-owned builderへこれらの入力を渡します。strategyへexecutorやevaluation
configを渡さず、snapshotにはcurrent-row feature・availability・staleness、accountと
active-orderのprimitive factsだけをtupleとscalarでコピーします。active orderの
limit/stop/submission-reference priceもsource intentからコピーします。直接構築した
legacy snapshotは3価格fieldをすべて`None`にできますが、価格情報としては不完全です。
future rowやopaque order IDはpolicy入力に含めません。これは完全なMarkov状態を
提供するものではなく、free/reserved cashも合成しません。

policyは各symbolのtarget weightをすべて返します。明示的なexact-quantity HOLDも指定できますが、そのweightがhard-risk適用後も変わらない場合だけ既存数量を維持します。riskがtargetを縮小した場合は通常どおり実行され、latency-aware actor boundary後のterminal intervalはsessionのrisk-owned emergency flattenが処理します。observer/transition failure後はexecution stateが進んでいる可能性があるため、sessionを再試行できません。このsnapshotは観測したaccountとpending-order factsであり、free/reserved cashや完全なMarkov stateを合成しません。

この数値portfolio経路は既存の`run_shared_cash_replay` intent APIとは別のopt-in結果型・ledger schemaを使います。既存intent result、schema、default behaviorは変更しません。これは将来のactor接続点であり、PPO学習、fit、研究結果、収益性の証拠を追加するものではありません。

## 不変条件

- 判断前に未来バーを読まない。
- strategy intentを直接fill扱いしない。
- hard riskを通す前のproposalを約定しない。
- fee / spread等をstrategy側で別途控除しない。
- 次のバーは約定後の共通`BookState`から始める。
- 比較対象strategyは同じexecution/accountingを使う。
- 保有年齢はpolicy actionでなく実際の約定後数量から数える。
- minimum-holdの抑制よりhard riskを優先する。

## 失敗すると何が壊れるか

| 破る契約 | 起こる問題 |
| --- | --- |
| 未来情報を観測へ入れる | leakageで見かけの利益が増える |
| intentを直接約定する | riskやfill制約を無視した非現実的P&Lになる |
| strategyごとに会計を変える | 戦略比較が同一条件でなくなる |
| costを複数箇所で引く | 二重課金になる |
| 約定前stateを次バーへ持つ | portfolio状態とP&Lが食い違う |
