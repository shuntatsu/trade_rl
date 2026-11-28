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

globalのreturn平均・標準偏差に値がない行は、`global_feature_available=False`と欠損理由で区別します。`global_feature_staleness_hours=0.0`はこのとき単なる数値placeholderで、最新値の存在や0時間前の予測を保証しません。経過時間は値が実際に存在する場合だけ解釈します。

クロスアセット特徴量に15分足のような短いnative sourceを使う場合も、内部returnはそのsourceの実際の利用可能時刻に合わせて基準足へ整列します。replay時点より後に公開されたsourceを観測へ混ぜず、追加の公開特徴量列は作りません。

未来のバーを見てから観測を作ることはありません。判断に入る情報は現在時点までに利用可能なものに限定します。

Ridge / LightGBM strategy adapterは、保存済みobservation timestampが`NaT`、またはmodelの`fit_cutoff`がobservation timestampより後なら、feature-unavailable fallbackやpredictionの前に拒否します。同時刻は許します。このas-of floorはmodelの生成・配布時刻や過去のvintage availabilityを証明せず、`StrategyObservation`構築前に既にwrapされたtimestampも復元しません。

native calendarの集約は上流の完全性検査です。直接構築するMarketDatasetも、価格・利用可能時刻・特徴量・取引費用等の配列に残る欠損マスクや不明な所有元を、正規化の前に拒否します。これにより既存の明示availability契約を維持し、正常な入力のidentityを変更しません。暦月・UTC週・固定間隔のsourceを明示し、欠損マスクや検証不能な配列所有元を上流で拒否して全memberの最終publicationを保持しても、粗いバーから細かい約定経路を推定したり、既存replayに新しい時間足を自動認可したりはしません。 Spotの収集済み確定streamを受け入れる場合も、実受信時刻を使う上流の検査であり、取引所接続の真正性や先物会計・replayの認可を与えません。 上位足の価格・出来高特徴量だけを利用する場合は、source pinと銘柄契約を照合する明示的なclocked overlayで既存の基準足Datasetへ追加できます。strategyが受け取るのは公開時刻・鮮度を満たした特徴量であり、約定・費用・資金調達・年率は元の基準足契約のままです。

native特徴量の時点整列を固定サイズのchunkで処理しても、strategyへ渡る公開済みの最新close、鮮度およびavailabilityの意味は変わりません。

一目均衡表の明示期間設定では、変位なしの雲と可視雲を区別します。可視雲は過去のspanを現在行の価格と比較し、必要な全依存行が公開済みのときだけ新しい特徴量として扱います。

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

`PreTradeRisk`はproposalへ、turnover、単一ウェイト上限、gross上限、drawdown縮小、emergency flatten、reduce-onlyなどの制約を適用します。turnoverや最終投影距離が有限値に収まらない入力は結果を発行せず拒否し、緊急マスクは真偽値配列だけを受理します。 目標値や制御マスクが欠損元を保持する配列の場合も、risk結果や注文を作る前に拒否します。

minimum-holdによる抑制の後にもhard riskを適用するため、drawdown stopやemergency flattenは保有期間中でも縮小・全決済できます。保有期間は決められた日数まで必ず持つ保証ではありません。

明示されたshared-cash terminal settlementでは、strategy roster内の実保有だけをterminal `emergency_flatten_mask`へ渡し、通常の`max_turnover`で最後のclose requestを縮小しません。通常のstrategy decisionsは引き続き同じturnover上限に従います。closeは設定済みのrisk・execution経路を通り、venue rule、liquidity、fill capacity、cost、pending orderの扱いもそのままです。容量や制約が足りなければ残数量・注文残を保持し、exact-flat validationは失敗します。venueの最小notionalもこのcloseを免除しないため、最小額を下回るまで縮んだ保有は残ります。開発用study（Phase 2とtrend-stop）のledger照合は、そのような残りのうち、engine自身が最終区間でそのcloseを `below_minimum_notional` で拒否したものだけをterminal dust（銘柄・正確な数量・notional・理由 `below_minimum_notional`）として記録し（流動性不足や参加率上限など他の理由で残った保有は、markでの価値にかかわらずdustではありません）、markから想定exit cost（fee・spread・impact）を引いて評価し、equity pathの最終点もその分だけ下げます。`terminal_flat` はfalseのまま、残りがすべてdustなら `terminal_dust_only` がtrueで、cashとdust調整後equityが一致すれば合格します。dust以外の残り、またはcost modelなしで評価できないdustは従来どおり不合格です。

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

Task18の人工native経路は、凍結した単純リターン予測・費用・実snapshotを照合して各独立口座の数値targetを作ります。明示的な72区間の終端予約（latency+1以上）を指定し、risk適用後の全重みが現在と完全一致する場合の数量HOLDと債務保持をopt-inにできます。既存の既定値は変更しません。完了は全6,599 native区間、厳密flat、未約定なし、および台帳・実費用・Funding・判断の照合を要求します。保存済みの人工完了armは外部pinから読み直し、再実行せず同じsource oracleで確認できます。停止後に別runでpending armを実行する場合は、先行handoff digestを保存し、RUNNINGからartifact再検証後にのみCOMPLETEDへ進めます。共通replayの任意通知は実native advanceの観測を診断処理前に返します。後継runは初回・64区間ごと・最終・非正equity時に保存し、例外時は最新実観測をFAILEDへ、初回観測前のみunknownにします。強制停止は最後の保存地点だけを証明します。完成receiptの照合時には途中観測のequity/drawdownが同じindexのledgerと一致することも確認します。複数回の再回復では、全祖先handoffとlineage v2を外部pinで検証し、各世代の完成済みarmを累積してから次のpendingを決めます。別世代のreceiptを現在のrunへ書き換えません。Task18のscalar allocatorはexact Fractionの傾き閾値で旧endpointの最適解と同値規則を維持し、独立oracleは元の候補比較を残します。全48-armの人工検証では単一親が明示batchを登録し、最大4 workerが重複しないarmを実native実行・保存・読み戻しします。48-armの市場実行や認可を示すものではありません。

`run_shared_portfolio_replay`はsimulation-ownedな
`SharedPortfolioSession`を呼び出すopt-in batch consumerです。eligibleなnative
barごとに既定ではpolicyを一度だけ呼びます。sessionは同じ`MarketExecutor`・実際に返された
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

明示的に判断するfine barを指定すると、それ以外の区間は新しいactor判断なしで進みます。決定論的な非レバレッジの成行注文に限り、現在の保有・残注文の全約定後・銘柄ごとの部分約定による最大露出をリスク検査します。許容される残注文は元のID・数量・約定可能時刻を維持し、許容されなければ通常のHOLD取消・リスク縮小を行います。新しいv3台帳で予定・実際のactor indexと判断のない区間を区別し、既定のv1/v2台帳はデータクラスのフィールド構成も含めて維持します。現在markによる検査は将来の約定価格でのリスクを保証せず、終端決済の残高検査も必要です。

現物のcash-only評価では `run_shared_portfolio_replay(..., strict_spot_cash=True)` を明示できます。long-only・非レバレッジ・決定論的な費用設定が前提で、処理行のFundingや負の非取引cash flowを拒否します。実際の約定数量・価格・費用を反映するとcashまたはexact inventoryが負になるfillは、同じBookStateのcommit前に拒否し、no-fillとして記録します。別会計を作らず、未約定・再試行・終端決済もnative経路に従います。注文全体を拒否する初期仕様であり、取引所の現物認証や混合spot/perpの証拠金契約ではありません。既定モードと過去の実行識別子は変更しません。

特徴量を新規構成するときは、ADXのseedが確定した時点で有効になる `adx_causal_v1` を明示できます。旧 `adx` の追加1本を要求する互換挙動と別identityであり、過去のDatasetや学習済みモデルを自動変更しません。

UTC暦月の現物リプレイは、専用Datasetを作り `native_calendar_month=True` を明示した共有ポートフォリオ経路だけで受理します。実際の月の長さを保持して約定・現金利息・年率を計算し、通常の固定時間足へ流しません。

契約数量単位の暦月リプレイは、別途 `native_calendar_month_linear_carry=True` を併用します。検証済みv9 Datasetの明示イベント価格率、前足容量と次の始値での約定、終端の最大1件のfunding、実際の月日数による借入・現金利息を既存の会計で処理します。全windowの借入可否、利用可能性、費用と終端予約を先に検査します。複数・月内イベント、遅延、企業行動、負の現金金利、market profile、rule stress、疎なactorスケジュールは拒否します。固定足・Spot既定モードとは別の実行ポリシーdigestを使用し、Dataset・月足return形式は維持します。デリバティブ取得builder・実取引所認証は対象外です。

任意のfill_cash_observerは成功したnative invocationごとに、各実約定直前・直後のcash・exact quantity・実費用を順序付きの別レコードで通知します。終端の通常約定も対象ですが、marginによる非注文清算やcarryは含みません。旧ledger形式・既定の会計はそのままです。callbackの失敗は実行を中断し、すでに外部へ書いた観測は自動的に取り消しません。

この数値portfolio経路は既存の`run_shared_cash_replay` intent APIとは別のopt-in結果型・ledger schemaを使います。既存intent result、schema、default behaviorは変更しません。これは将来のactor接続点であり、PPO学習、fit、研究結果、収益性の証拠を追加するものではありません。

`strategies/portfolio_policy_codec.py`はGym adapterとfrozen-policy inferenceが共有する純粋なstrategy-layer codecです。raw-v1 observationのkey・shape・dtype・値を同じDTOから作り、strictなjoint action codeをnumeric target proposalへ変換します。exact HOLDは現在の実数量を維持するproposalとして表し、codecはGym/SB3、risk、execution、evaluationをimportしません。`PortfolioInferencePolicy.decide_portfolio`はloaded bundleのsymbol/feature roster、decision window、native timestampを確認してから同じcodecを使います。

`SingleSymbolPortfolioPolicyAdapter`は、固定roster上のexact built-in constant・trend・Ridge strategyをportfolio targetへ変換するopt-in adapterです。filled sideと明示budgetを使いますが、legacy cached-target、minimum-hold、pending-aware intent、exact-quantity HOLDや任意strategyの互換性は主張しません。より豊かなsemanticsにはdirect `PortfolioPolicy`を使います。

`run_four_family_shared_portfolio_replay`は、callerがpinしたDataset、Ridge artifact、joint PPO bundleと共通設定を全て検証してから、cash・fixed Trend・Ridge・joint PPOを既存native replayへ順番に渡すTier-2 in-memory coordinatorです。各armはfresh policy/risk/sessionを持ち、同じDatasetとimmutableなexecution/risk/profile設定を使います。terminal settlementとledger evidenceを常に有効にし、早期終了やinsolvencyはraw outcomeとflagsに残します。任意の`on_family_completed` observerは完了armを次armの前に受け取り、observerまたは後続armの失敗はそこで伝播します。この入口はattempt予約・永続化・fit・winner selectionを行わず、研究上の合格や収益性を確立しません。

`run_four_family_shared_portfolio_scenario_replay_v2`は別moduleの明示opt-in入口です。外部pinされたscenarioとcurrent scenario-inference sourceをartifact読込前に照合し、既存のDataset/Ridge preflightとprivate native replay coreを使います。scenarioのruntime `ExecutionCostConfig`が4 armすべての唯一のexecution configになり、同じDataset・capital・risk・profile・clock・Trend/Ridge入力・frozen PPO checkpoint/statisticsのもとでarmごとのfresh policy/risk/sessionを作ります。返却値はnative four-arm outcomeに実際のscenario inference bindingを組み合わせたin-memory resultです。v1 signatureとstrict matching、`trade_rl.evaluation.runs` facade、v1 persisted schemaは変わりません。この入口はattempt/resultを保存せず、scenarioを選ばず、M34/economic evidence・winner・profitabilityを示しません。
各armとcompletion observerの直前にもscenario/source pinとloaded policyのruntime environment bindingを再検証し、不一致なら次armまたはobserverの前に停止します。

`run_four_family_portfolio_scenario_attempt_v2`は、このin-memory replay用の別opt-in one-shot wrapperです。v2専用identityとexpected result digestを要求し、started record、before/after provenanceとscenario-source recordsを保存します。各armの既存native rowは次armの前に保存され、strict result reloadの後だけcomplete markerを書きます。`load_four_family_portfolio_scenario_attempt_v2`はcompletion、failureの不存在、外部identity/digest、resultとnative rowsを検証します。failure時は先行arm rowsと元例外を保持します。この保存形式はv1と別schemaで、scenarioの選択、M34 screenの更新、収益性・研究承認・live eligibilityを示しません。
`run_four_family_portfolio_attempt`は、このin-memory coordinatorを一回限りのlocal filesystem attemptへ接続する別のopt-in wrapperです。開始identityとbefore/after provenanceを記録し、各armのraw outcomeを次armの前に保存してから、four-family resultをstrict reloadし、最後にcomplete markerを書きます。public loaderにはcompletion recordとは独立したexpected result digestが必要です。failure journalが書ければ失敗phaseと先に完了したarmのrecordを残し、journal write自体が失敗した場合も元例外を維持します。native comparatorがarm内でcallback前に失敗した場合、公開callbackはその途中のnative ledgerを返さないため、failure recordが保持するのは先行armのrowsだけです。途中armの実行数を0とは推定しません。failure/completionが併存するrootは読み込みを拒否します。既存native ledgerをそのまま保存するため、二つ目の帳簿や損益再計算を作らず、global retry budget、fit、winner selection、研究承認も追加しません。

`evaluation/portfolio_inference.py`はcallerが指定したDataset artifactとbundleを接続するopt-in filesystem helperです。paired Dataset loaderを一度だけ使い、resetしないGym shellでbundleとのcontextを照合した後、同じDataset objectと同じrisk/execution設定を`run_shared_portfolio_replay`へ一度渡します。Gym shellはfinancial sessionを作らず、replayが唯一のsessionとterminal settlementを所有します。これはsoftware integration contractであり、モデル性能・収益性・研究gate・deployment readinessを示しません。

`SharedPortfolioGymEnv`はこのnumeric sessionを使うopt-in Gym adapterです。各symbolの`MultiDiscrete` actionは`HOLD=0`、`SHORT=1`、`FLAT=2`、`LONG=3`で、HOLDはrisk projectionが許す範囲で現在の実数量を保ちます。固定shapeのDict observationはsnapshot builderのcurrent-row/account/order/risk factsを符号化しますが、partial observationでありfree/reserved cashを作りません。actor decision後に予約されたterminal settlementも同じsessionが処理し、rewardはそのstepで進んだ全native intervalのnet-equity差を固定初期資本で割った値です。`objective_endpoint_valid`はsettled・flat・注文残なし・非insolvent・termination reasonなしという valuation 条件だけを表し、objective/risk/digest全体の合格やstudy admissionを意味しません。価格gapによる実現drawdownがrisk stopを超えてもその値を保持します。既存native insolvencyはterminal cash/equityをzeroへfloorするため、Gymは追加のreward clippingをしませんが、固定初期資本の-100%を超えるdebt-loss semanticsも確立しません。insolvent episodeはinvalidのまま保持します。Gym adapter自体はtrainerではなく、既存PPO defaultsを変えません。

明示v8の3d・固定7d・UTC週足の既定Spot経路では、Gym構築時とreset前、PPO学習前、v1/v2 bundleのpolicy読込前に、Dataset identity・long-only native執行設定（isolated marginの銘柄数制約とrisk/executionレバレッジ整合を含む）・終端予約を含む宣言windowの利用可能性を共通の執行条件で検査します。Datasetの開始・終了行時刻とobjective、3つのinterval clock、経済horizonも再照合するため、利用可能な別windowへの移動だけでは実行を受理しません。処理行のfunding・企業行動・負のcash carryを拒否します。元のimmutable native Datasetを保持し、別Datasetへ切り替える場合は新しい環境を構築します。actorへ未来の観測を渡す検査ではなく、実データの承認や終端決済の保証でもありません。この固定時計の契約は維持し、月足学習には次の別時計を使います。旧checkpointの自動移行は対象外です。

明示 `native_fixed_linear_carry=True` の共有口座replayは、検証済みv8の3d・固定7d・UTC週足の契約数量単位Datasetに限ります。次の始値で実約定した後の数量へ、処理足終端に割り当てた最大1件の資金調達を適用し、借入費用は始値評価のshort notionalと実経過年数から共通会計で計算します。全windowをactor呼出前に検査し、複数イベント・借入不可・欠損・企業行動・負のcash carryを拒否します。cross margin、最大レバレッジ1、決定論的MARKET・前足容量・追加latencyなし・partial-fill carryなし・終端決済指定を要求します。Spot既定経路・旧identityは変更せず、別execution-policy digestを使います。実取引所のイベント時刻・市場内経路を復元する機能ではありません。

共有口座PPOにも同じ明示 `native_fixed_linear_carry=True` を指定できます。上記のDataset・費用・全windowの制約を再利用し、LONG/SHORT/HOLD/FLAT、gamma=1の固定初期資本equity delta、終端決済を共通sessionで処理します。観測変換・training contextにはキャリー専用schemaを使い、modeの違うモデルは同じデータ・費用でも読込前に拒否します。固定/月足の既存payloadは変更せず、外部pin付きのDatasetとbundleのpaired replayにも明示flagを引き継ぎます。固定scenario-v2・four-family評価・月足先物・複数funding eventや不規則sessionは対象外です。

UTC暦月Spotの共有口座PPOは、検証済みv9 Datasetと明示 `CalendarMonthFinancialClockContract` の組合せだけを受理します。1ステップは実際の暦月、期間は月数と実経過秒を検査し、固定30日へ換算しません。gamma=1、固定初期資本のequity delta、GAE lambdaとrollout・latency・終端予約はnative step単位です。観測変換とforward bindingは別schemaで、実際のUTC境界を各indexへ照合し、従来の統計fit範囲と同じ帳簿を維持します。学習済みbundleは月足のclockとtransformを別payloadで固定し、同一Datasetの互換forward windowと外部pinを検証してから読込みます。既定Spotモードは月足先物・資金調達を拒否します。固定scenario-v2・four-family manifest、任意不規則session、月内の約定推定も対象外です。

契約数量単位の暦月PPOは `SharedPortfolioGymEnv(..., native_calendar_month_linear_carry=True)` の明示モードでのみ受理します。暦月キャリーの執行・borrow/funding・終端決済をそのまま使い、同じ月足clock、actor行のみのfit統計とgamma=1 equity deltaを維持します。観測変換とtraining contextは月足signed専用schema、forward bindingは変換digestとmodeを照合します。読込前に月足Spotや固定足キャリーとの取り違えを拒否し、paired replayにも明示flagを渡します。既存payload・既存checkpoint readerは変更しません。実取引所の月内event、複数funding、研究や経済結果の証明ではありません。

このraw Dictを学習前に整えるopt-in経路が`PortfolioObservationTransform`です。payloadはversioned field layoutとDataset・symbol/feature order・objective/clock・risk/execution・fit scopeをbindします。local featureは既存PPO normalizerを使い、global featureと銘柄別geometric mark referenceは`[start_index, agent_stop_index)`のactor-fit rowsだけから求めます。availability maskを残し、account・active-order notionalはfixed initial capital基準、maintenance-margin rateはdimensionlessのまま保ち、UTC nanosecondsはfloat変換前に整数差分を取ります。serialized payloadの読込には別途保持したtrusted expected digestが必要です。これはmetadata改変を検出しますが、署名やfit実施の証明ではありません。raw validationはrisk flagと注文indexの型も検査します。`PortfolioGymObservationWrapper`はnative reward・termination・infoを変えず、terminal時は固定paddingを返します。raw stepの後に変換が失敗した場合は、そのtransitionを取り消したと主張せず、wrapperを記録付きでpoisonし、reset成功まで次のstepを拒否します。

同じwindow以外へ適用する場合は、`PortfolioObservationRuntimeBinding`とtypedなtraining `BoundObjectiveClock`を明示します。Dataset、symbol/feature roster、資本、risk、execution/profile、objectiveとclockの非時間semanticsを維持し、evaluation endpointsとeconomic horizonだけの差を許し、runtime開始をexclusive training stop以後に制約します。変換時のclock/order offsetsはruntime windowを使い、actor-fit統計はそのまま再利用します。このforward bindingもfeature availability/vintageやtrained policyの妥当性を証明せず、trainer・学習結果・収益性の証拠を追加しません。

`transform_observation(..., runtime_binding=...)`を直接呼ぶ場合は、先に`create`または`validate_environment`でtypedなtraining boundと実際のruntime environmentへbindingを照合します。`PortfolioGymObservationWrapper`は使用前にこの照合を行います。`from_payload`単独ではschemaとdigestだけを検証し、live environment contextは検証しません。

`PortfolioRuntimeExecutionScenarioV2`は、baseとruntimeのtyped execution policy、training context、transform、checkpoint、caller宣言planを結ぶversioned declarationです。opt-in v2 loaderはexternal scenario digest、current v2 source digest、既存bundleのpinを要求し、base policyをfit contextへ、runtime policyを実際のforward `SharedPortfolioGymEnv`へ照合してから同じPPO checkpointを読み込みます。base latencyで元のfit actor stopを確認し、fit統計を変更せず、runtime latencyでruntime actor windowとnative terminal reservationを確認します。v1 loader、payload、strict cost matchingはそのままです。v2 bindingとobservation wrapperは両方のexecution identityを保持し、native resetと各stepの前に実環境を再照合してからraw observationを同じcodecで変換します。scenario loaderとreplay entry自体はfour-family resultを保存しません。比較結果をpersistする別の`run_four_family_portfolio_scenario_attempt_v2` wrapperとv2 result artifact publisher/loaderは上記に記載した別schemaのAPIです。このruntime binding/replay pathはscenario recipe選択、経済性能・M34結果・profitabilityを追加しません。

loaderはtyped inputからexecution policyとruntime/windowの条件を再構成しますが、保持するtraining-bound digestだけではfitの実施やそのprovenanceを独立に証明しません。

`PortfolioEpisodeJournalWrapper`はこのGym環境またはobservation-only Gym wrapperの外側に置くopt-in event journalです。underlying環境が`SharedPortfolioGymEnv`であることを検証し、action objectを変換せずに渡します。manifestにはconfigured target weightsを記録し、各stepにはsourceのrequested target weightsとeffective HOLD symbolsを記録します。これらはfillsではありません。observation、policy log-probability、policy versionは保存せず、PPO trainerも追加しません。

`PortfolioPolicySampleJournal`は別のopt-in SB3 forward-hook recorderです。SB3がenvironmentを進める前に、policyへ実際に渡ったtensor入力、sampled action code、joint log-probability、critic valueをbatch rowごとのcanonical JSONへ保存します。sample schema v2ではforward batch sizeと、action/value/log-probability出力tensorの実dtype・batch全体shapeも保存しますが、既存のaction・scalar valuesは変換しません。attach時はpolicy-local/globalのforward hookがない状態を要求し、その後は各hook layoutとdispatch flagを固定します。後から追加・削除・置換されたhookを検出したら、次のsampleを書き込む前にfail closedします。constructorが登録後の検証に失敗した場合は、自分のhookだけを除去しpolicy予約を解放します。context、transform、policy versionの文字列はcaller宣言のlabelで、Dataset・model weights・checkpoint lineageを証明しません。optimizer updateやnative financial eventとのjoinは記録せず、legacy PPO defaultsやpolicy serializationも変更しません。PyTorch/SB3はrecorder構築時だけimportします。

`fit_shared_portfolio_ppo`はこれらを組み合わせる別の明示的なsoftware training entrypointです。1つの`SharedPortfolioGymEnv`でCPU PPOを実行し、policy sample、native episode event、submitted actionを照合してからtransition joinを記録します。試行rootを確保する前に`n_steps`と`gae_lambda`を環境のbound financial clockに一致させます。rollout前後のcheckpoint hash、sample/native/join件数、optimizer step呼出数、SB3 epoch counterを保存し、exact budget完了かfailure/incompleteかをwrite-once attemptへ記録します。failure receiptはhook登録後に観測したpost-step呼出数とSB3の報告するepoch counterを保持し、未初期化なら明示的にnullを記録します。SB3のtarget-KLはepoch途中で停止し得るため、epoch counterは部分epochを含む場合があります。optimizer step中の失敗ではpost-step hookの観測数が適用済み更新数を下回る可能性があります。trainer自体はbundleを作りません。別の`portfolio_inference_bundle.py`が、strictly verified completed attemptを独立したversioned bundleへpublishし、trusted external digestsと実際のforward environment/action-space bindingをpolicy deserialization前に確認してloadします。digest検証後にcheckpointを読み込んだ後、実policyのspace、MultiInput policy class、actor/critic MLP width、activation、declared default features extractorも確認します。どちらのopt-in software pathも利益、model selection、research gate completion、deployment readinessの証拠ではありません。legacy per-symbol PPOやdefaultは変更しません。

`publish_portfolio_inference_bundle`と`load_portfolio_inference_bundle`は、明示的に選んだtraining attemptだけを扱うstrict v1 opt-in APIです。bundleはcontext、checkpoint bytes、selected source/runtime owner manifest、Gym observation/action spacesをbindします。別ownerのscenario-v2 loaderは同じverified bundle/checkpoint routinesを使いますが、v1の全financial-context一致を緩めません。選択されたsource ownerにはDataset module `data.market` と共有timestamp normalizer `data._timestamps` が含まれますが、manifestはcomplete transitive source closureやfit実施の独立証明ではありません。両loaderは外部expected digestとforward runtime bindingをpolicy deserialization前に検証し、checkpoint読込後は実policy spaceと宣言architectureも照合します。software integrityは実データ性能や将来利益、research gateまたはlive deploymentの承認を意味しません。

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


明示sessionの実経過時間評価は `run_session_elapsed_portfolio_replay` を使います。呼出元が独立に確定した初期markと全processing closeのUTCナノ秒列、nominal barの秒数を、検証済みSESSION Datasetへ照合してから既存の共有Spot replayを一度実行します。欠損を休場とみなして補完せず、venue calendarの真正性は別の責務です。休場時間と取引足のcash carryは既存会計に残し、全区間完了・未処理注文なし・exact flatを要求します。専用 `SessionElapsedReturnSeries` は休場も含めた実経過年数と各intervalのdurationで年率化し、別payloadにDatasetと境界列を束縛します。名目年率・旧paired/seed統計・OOS結合への暗黙変換は拒否します。既存session replayの既定値は不変です。

同じ明示sessionの共有Spot PPOには `SessionFinancialClockContract` を使います。独立に宣言した秒単位のUTC境界列とnominal processing bar秒数、実経過horizonを金融objectiveに束縛し、構築・reset・fit・推論読込で同じ完全windowとSpot会計条件を再検査します。観測の経過比率は実時刻で計算し、GAEとrollout/latency/終端予約は観測step単位、gamma=1のequity deltaは既存sessionを使います。clock・transform・forward binding・training contextには専用schemaを使い、同じDatasetでも固定足モデルとの取り違えや途中時刻の改変を拒否します。固定/月足の保存形式は変更せず、既存16MiBのmetadata上限を超えるsession contextはattempt確保前に拒否します。paired artifact replayは同じsession elapsed経路で終端完了を要求します。固定scenario-v2・four-family、session short/先物、実venue calendar取得・任意のsession内約定経路は未対応です。
