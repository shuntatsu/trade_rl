# Current research status

更新基準: 2026-10-05 (JST)

## Issue #810: signed insolvency software candidate

An opt-in canonical execution policy retains negative marked terminal equity
instead of flooring it at zero. Historical defaults, policy digests and bounded
interval/log diagnostics remain unchanged. Synthetic short-gap, repeated-margin,
cache and order-identity tests use an independent cash/liability oracle. This is
a software prerequisite for a new fixed-capital reward, not market profitability
or a liquidation-cost calibration. Final-head full CI and formal independent
research approval remain integration requirements. No economic result, sealed
rerun, unused-future opening or live-order eligibility is established.

## 結論

Trade RLの現在地は、**lean core、5候補+3 controlsの共通比較基盤、provenance-bound candidate Run Core、Controlled Experiment Loop v1、Canonical M2 bootstrap toolingを実装し、`market_build_v3` / `portable_feature_numerics_v1`、real-cost-assumption Dataset、fit-scope-safe PPO Observation v2を固定したportable Canonical real-data baselineを、結果前のplan-only preregistrationからfresh post-Artifact verificationまで完了した**段階である。

一方、**Portable Controlled Experiment 0001は独立再検証まで完了し、formal decisionはKEEP_BASELINE**である。事前登録した唯一のcandidateはmean-reversionで5 / 5 symbolsをbaseline比改善し、median turnoverも低下したが、candidate total returnが正だったのは1 / 5 symbolsだけだった。結果前に固定したformal ruleではこの条件がKEEP_BASELINEに該当する。したがって現在も次を主張しない。

- profitabilityは未証明。
- winnerは未選定。
- Production/live order routingは未認可。
- PPOやforecastがruleを上回るという結論はない。

次の研究上の本質的作業は、新しいmodel familyやbootstrap toolingを増やすことではない。Experiment 0001のKEEP_BASELINEをcurrent development authorityとして維持し、次に検証するControlled Factorを結果を見る前にpreregisterしたうえで、同じfactor-isolation・raw-return・cost/cash・fresh post-Artifact verification契約でdevelopment Experimentを積み上げることである。

## Net-profit redesign: prequential forecast capability (2026-10-05)

Issue #810 P2 now has a software path from the existing causal row selector and
Ridge solver to frozen future-block packets and the existing cost-aware intent
controller. The selector records its actual pooled symbol/price/publication
endpoints. Each fit uses only strictly matured prefix labels; later blocks can
consume newly matured labels but cannot rewrite earlier vintages or packets.
Stored model/scaler and selected prediction inputs survive a digest-checked JSON
round trip. Future-only Dataset suffix mutation must leave causal identities
unchanged, while the whole-Dataset lineage ID may change.

This is a partial P2 implementation. The default horizon remains 24h; no 72h
economic comparison, parameter selection or real-data fit/replay has been
performed for this capability. Forecast completion/latency are declared
simulation assumptions, not measured receipts. The source only admits inputs
available on their Dataset row; it does not recover delayed feature histories.
Log-return forecasts and the current cost proxy do not establish expected simple
return or an optimal shared-cash allocator.

The non-RL portfolio allocator, downstream RL packet observation, common
DecisionContext, continuous-account walk-forward recipe, complete source/runtime
bundle and economic diagnostics remain separate unfinished work. G0/G1 for an
exact economic Study and G3-G5 are NOT ESTABLISHED. Synthetic timing, algebra,
tamper and causality tests are software evidence only. The research drawdown
guardrail remains 20%; there is no new winner, final-data opening or live-order
authorization. See the result-blind mechanism contract in
`architecture/research-assurance.md`.

## Trading-bot named signal and fixed-configuration diagnostic (2026-10-04)

利用者がBTC/ETHのdevelopment、after-cost profit、observed drawdown20%目標、prefix-only selection、fixed-parameter cost/latency stress、required CIと独立review後の通常PR統合を指定した。live発注はこの作業の対象に含めない。既存CLI/tuningは常にsignal index0を使い、canonical multi-timeframe Datasetの24bar signalを名前で固定できなかった。`--signal-feature`とtuning/comparison APIの明示indexを追加し、baseline/cash/candidate/foldへ同じindexを渡す。未指定の意味は変えず、無効indexや存在しない名前はreplay前に拒否する。異なる符号の先頭列への並べ替えでも、名前で選んだsignalの実order/returnが変わらないsoftware oracleを使う。

結果前の追加診断は既存verified Dataset `6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518`からBTCUSDT/ETHUSDTと2024Q1のhourly closeを抽出し、source ID、元のrow/symbol indices、content identity、availability/economicsを固定する。local `1h__log_return_24bar`だけをdecision signalにし、adaptive/balanced/8 grid候補/3fold/100,000 USDT、既存portfolio-global riskとquantity holdを固定する。各foldはcapital/stateをresetし、tuningの最後のbarでsettleして次foldへ情報を戻さない。cash carryは実Datasetのまま比較する。全windowのafter-cost returnが正、ledger最大drawdown20%以下、exact flat、active orderなし、economic terminationなしをdevelopment eligibilityの必要条件とし、一つでも欠ければNOT_ESTABLISHEDとする。主評価は各windowのafter-cost returnとreset-window仮想積であり、continuous wealth、independent sample、sealed final、live edgeを証明しない。adaptive regime仮説自体は未確立であり、この診断はimplementation/robustness確認に限定する。

prefixで選定したconfigを凍結し、同じevaluation windowにbase、`ExecutionCostConfig.multiplier=2`、`order_latency_bars=2`を適用する。multiplier stressは既存executorのfee/spread/impact chargeを2倍し、funding/borrow経済やstrategy/risk parameterは変えない。latencyはbase0から絶対値2への変更であり、最初のprocessing waitとterminal予約区間も変わる。選択し直さず全stressを同じ必要条件で検査し、cash/no-tradeまたは任意cell失敗からprofitable trading candidateを作らない。Q1は既にconsumed developmentであり、unusedへ戻さない。protocol、source hashes、raw returns、ledger、config、receiptはignored `output/bot-completion-20261004/`へ保存する。fresh result-blind AI G0-G2確認とmachine verificationを実行前に要求する。

初回protocolのresult-blind点検でcash検算がsimulator return/bookの自己一致に留まることと、G0 premise記述の不足を指摘されたため、未実行の初回rootを保存し、新しい`output/bot-completion-20261004-r2/`へprotocol v2を固定した。digestは`1e330adb4916eb08de5e04133c9dbba59b5d8859a764dcece4efa0356da91192`、subset Dataset IDは`35b6b086b5a205ebff7b60215da127ba1140dc28740c2474b5d66a2f8a33bd5f`、economic execution sourceは`66b08491`である。fresh read-only AIによるG0-G2承認後に一度実行し、両foldのprefix選定はcashになった。全selected base/cost2/latency2はreturn0%・drawdown0%・terminal flat/order remainderなしで、profitable windowは0、software PASS / economic NOT_ESTABLISHEDだった。untuned adaptiveのbase evaluation reportは−5.856874%と−9.311650%であり、後続損失を見てcashへ切り替えたものではない。source cash rate・timestampから独立計算するcarry oracle、全interval coverage、raw-return/final-equity整合性、selected config凍結を確認した。最大利益、winner、unused final、live eligibilityは未確立で、結果後のparameter再選択は行っていない。

fresh post-Artifact検算でsource/runtime、subsetの42 arrays、元executionの37 artifact hashesを独立確認した。初回出力にはprefix候補とbaseline evaluationのraw evidenceが不足していたため、既存8候補・baseline・cashと元の区間だけを固定して事後再構成した。新しい候補の追加やparameter再選択は行わず、元executionを変更していない。`selection-reconstruction-v2/receipt.json`は67 artifact hashes、22 replay、15,976 intervalsをbindし、別のread-only AIが6,454 fills、1,982 funding events、inventory/cash/cost/funding、順位と元baseline reportの完全一致を確認した。両prefixの全trading scoreは負でcashの0を下回り、経済判定はNOT_ESTABLISHEDのままである。この事後検算はconsumed developmentの証拠補完であり、別GitHub principalによる正式PR reviewを代替しない。

## Quantity-preserving hold repair (2026-10-05)

Synthetic counterexamples exposed two quantity-hold mismatches in PPO training
and canonical replay. A split changed the filled book but left the desired
quantity in old units; a distinct mark price was used for weights but trading
close for inverse quantity sizing. Unchanged intents consequently created extra
fills, fees, and unintended exposure. Repair rebases cached proposals, including
unfilled entries, by processed split factors and separates mark-based sizing from
trading-reference order prices. Signed quantity, cash, fee, and order-bound
oracles cover training, single-symbol replay, and shared-cash replay. Cash-book
entry also resolves current market marks when initial prices were omitted. Historical
artifacts are unchanged. This is a software repair; it supplies no new PPO
profitability or winner evidence and does not authorize a sealed-run retry.

## Trading-bot validation repair (2026-10-03)

継続監査では、価格が一定でsignalだけがentryを要求するsynthetic marketにおいて、non-zero costで全候補が損失でもtunerが取引candidateを選ぶ反例を確認した。prefix-onlyのcash controlを常に比較する修復と、実accountのcost/funding/borrow/turnover/fill diagnosticsを追加する。後続windowを見てcashへ変更する処理は導入しない。過去Q1結果は既にconsumed development evidenceであり、修復後の再確認やcost/latency stressも未閲覧finalとして再分類しない。

追加のfirst-holdout-open shockで、終端決済のfillがreplay `stop_index`行のopenを参照し、evaluation開始と同じstopを渡すとfirst evaluation barがtuning scoreとdrawdown eligibilityへ漏れることを再現した。selection replayを最後のtuning bar内で終端決済し、walk-forwardも同じhelperで境界を分離する。修正前source `a526cc18` のQ1 development smokeは探索的な診断として保持し、first evaluation barから独立したholdout証拠とは扱わない。この修正ではreal-market replayを実行せず、新たなprofitability evidenceも作らない。

adaptive設定でNaN/Infinityや不正なholding期間が受理され、exit比較を無効にする入口もRED contract testから修復する。これは設定のfail-closed化であり、既存finite候補のstrategy economicsを変更しない。

実SB3 integrationでPPO inference bundleの単発directory renameがWindows permission failureで停止したため、既存のbounded atomic-publication primitiveをこの経路にも使う。transient lockとretry exhaustionをfake policyの回帰テストで再現し、staging cleanupを確認する。training objectiveやhistorical model bytesの意味は変更しない。

cash追加の開発診断契約は同じBTC/ETH 2024Q1、adaptive/ balanced/8 grid候補/3fold/capital100,000/costを固定し、cash追加後も正のprefixで選定した既存configと後続結果が変わらないことを確認する。実book・raw return・ledgerを保持し、base-prefixで選んだconfigを再選定せずexecution multiplier2とorder_latency_bars=1のstressへ適用する。software修復のPASSはprefix選択とcash/ledger契約の一致であり、経済screenは各base/stressでafter-cost returnが正、observed drawdown20%以下、終端flat/order remainderなしを要求する。一つでも満たさなければprofitabilityは未確立とし、winner/final/liveへ昇格しない。protocol・review・実行receiptは新しいignored `data/bot-cash-control-2024q1/`へ保存する。

source `8ff87cbf` のcash追加診断は、fresh result-blind G0-G2レビューと170件のmachine testを経て一度だけ実行し、software PASS / economic NOT_ESTABLISHEDとなった。正のprefixで選ばれたconfigと元のcore結果は完全に保持された。selectedのafter-cost returnは−1.941582%と+0.579034%、実行costはaccount currencyで1,998.597217と1,509.094741、funding PnLは−78.761112と−36.671020だった。設定を選び直さずcost multiplierを2にした結果は−3.902920%と−0.934769%である。18 replayのraw return・ledger・cost/funding/fill計算と21 artifact hashを独立事後確認し、全cash対照はzero-return / zero-cost / zero-fill、全終端はflatだった。

order eligibilityは`submit_index + order_latency_bars`だが、最初のprocessingは`submit_index + 1`である。したがって設定0と1は最初の約定可能足が同じであり、当初の0→1 armは追加のprocessing waitを検証していなかった。ただし`agent_stop = stop_index - order_latency_bars - 1`も変わるため、全replayの一般的なno-opとは扱わない。

補足のignored `data/bot-effective-latency-2024q1/` は、同じsource・Dataset・既存selected configをfreezeし、latencyだけ2へ変更する。別のfresh result-blindレビューで189件のmachine testを通し、実行前にprotocol `be216073f8298654daabe4e1d34c52ca738f80ffbd39ec280e995805638ae2ba`を固定した。この設定は最初のprocessingを1本待つ一方、terminal予約区間も3本へ変えるexecution-setting treatmentであり、pure-delay effectや同一trade pathは主張しない。after-cost returnは+1.846582%と−2.026866%、agent期間の同一orderでwait→eligible→fillを確認した件数は76と58、双方のterminal settlementは完了した。software PASSだがeffective-latency screenはNOT_ESTABLISHEDであり、winner / final / liveへ昇格しない。

これらは再利用Q1のdevelopment診断であり、候補・threshold・seedを結果後に再選択していない。最大利益や将来のrobustnessは未確立である。独立raw-return検算のclose-only drawdownとledgerのintrainterval最大drawdownは別の証拠であり、後者の全経路をraw returnsだけから再構築したとは主張しない。

Botの`balanced` scoreは損失の符号を反転していた。channel戦略はchannelでない列を固定位置で読み、synthetic channelは現在足のextremaを含んでいた。これらを独立計算とmocked reportのRED testで再現し、符号付きscore、名前で解決する既存prior-candle channel、終端決済完了を要求する選択条件へ修復する。replay/accounting ownerやPPO研究の機構は変更しない。

未統合のwalk-forward実装は既存tunerと異なる探索を重複して持ち、少数candidateで探索axisを落とし、最後の残余barを捨てていた。同じ探索実装へ統一し、CLIでdevelopment diagnosticとして実行できる契約を追加する。windowごとのcapital/state resetと仮想return積を明示し、continuous wealthやsealed final profitabilityとは扱わない。scopeはソフトウェア修復と開発実行の確認であり、live注文接続や新しいPPO実験の認可ではない。

追加の契約テストでは、未決済reportのpositive marked P&Lが`profitable_windows`へ入る集計漏れと、walk-forward内のwindow reportが`holdout` scopeのまま出力される不整合を確認した。集計をpositive P&Lかつterminal settlement確認済みの場合だけ数えるよう修復し、手動reportで未提供の決済状態はunknown (`None`) としてfail-closedに扱う。各nested windowにも`development_walk_forward` scopeを付ける。これは評価報告の正確性を直すものであり、以前の開発結果を書き換えず、profitability evidenceも追加しない。

実データの開発確認はBTCUSDT/ETHUSDT USD-M、1h、2024-01-01から2024-04-01 UTCへ結果前に固定する。公式Vision archiveのchecksumとraw hash、現行exchange-info snapshot、明示cost、Dataset identityを保持し、offline再build一致を確認する。adaptive family、3fold、8candidate/window、balanced、initial capital100,000、既存non-zero execution overlayを固定する。現在metadataのhistorical適用、close mark proxy、bar capacityは仮定であり、point-in-time venue rulesやlive fillを証明しない。生成Dataset、protocol、結果はignored `data/bot-development-2024q1/` に置く。これはdevelopment smokeであり、winnerや利益の証明にはしない。

このsmokeはsource `a526cc18` のfresh read-only AIによるresult-blind G0-G2確認とmachine verificationの後に実行した。Dataset IDは`7cff150e0f4d18dcc457009232f53cc7ca3f350db4893c433db6494887979616`、protocol digestは`085fc5242aa05c77ae4cd99184cb9551dbd558e35256eb7a22ca46bad0809475`である。2184本のhourly close、12 archive checksum、offline Dataset再buildの一致を確認した。次foldのselected candidateのafter-cost returnは−1.941582%と+0.579034%、observed maximum drawdownは3.899723%と2.296434%だった。双方のterminal settlementは完了したが、reset-windowの仮想積は−1.373790%であり、全体のprofitabilityは成立しない。この結果を見てparameterを選び直しておらず、sealed final、future-data、live suitabilityは引き続き未確立である。実行receiptと元のJSONは同じignored directoryの`execution/`へ保存した。session内の独立AI確認はGitHubの別principalによる必須PR approvalの代わりではない。

## Trading-bot tuning contract correction (2026-10-02)

GitHub `main` の `a696d5c` では、明示的な `--dataset` がないまま `--mode optimize --strategy all` を実行すると、500-barのgenerated demo Datasetへ暗黙にfallbackし、shared-cash replayにはzero execution costを渡していた。したがってそのCLI経路は実market evidenceではなく、profitabilityの根拠にもならない。tuningは同一full Datasetで選択・報告しており、出力文言もmaximum profitを示唆していた。

`codex/profit-engine-hardening` の修正では、optimize / compareに明示的なDatasetまたは明示的な `--demo` を要求し、canonical non-zero `ExecutionCostConfig()` を既定のreplay costにした。単一strategyのparameter選択はchronological tuning prefixだけで行い、baseline / candidate reportは後続holdoutのfresh replayから計算する。tuning-windowの最大drawdownが20%を超えるcandidateは選択対象外だが、これはeligibility vetoであり、gapやexecution timingを越えたdrawdown上限の保証ではない。`compare` はfull-rangeのin-sample診断である。`tune_all_strategies` の複数family報告windowはfamily間で比較した時点でdevelopment evidenceとして扱い、最終評価にはさらに後の未閲覧windowを使う。

明示Datasetにprice-channel featuresがない場合、botの`channel_breakout`実行と全戦略のcompare / optimizeは、直前480本・240本のcandleから因果的なchannelを導出する。導出後はsource Dataset IDとwindow定義を含む新しいcontent identityへbindする。4列の一部だけがあるDatasetは拒否し、導出時は最低481 barsを要求する。

walk-forward結果もchannel派生後のDataset identityへbindする。terminal settlement状態がunknownの場合は残ポジションがあると断定せず、明示的な未決済と区別して表示する。この報告精度の修正ではreal-market trainingやeconomic tuning runを行わない。

Bot reportはbar-return intervalのcount / positive rate / profit factorとDataset period metadataに基づくSharpeを明示し、closed-trade metricsとは呼ばない。adaptive protective exitsはactual fillからbar-closeまでのgross price returnでthresholdを判定し、直近のeffective intentではなく実約定quantityが0になるまでflat intentをlatchedして最低保有期間をbypassする。missed / partial fill後に価格がtrigger未満へ回復してもexit requestを維持するが、entry後fee・funding・borrowを含まず、fillはtrigger後のeligible execution stepで行われる。gap、latency、liquidity、costによりthresholdを越える結果があり得るため、これもprofit protectionの保証ではない。このrepairではreal-market trainingやeconomic tuning runを行っておらず、新たなprofitability resultは確立していない。

## Active PPO medium-term holding-duration design

The user's direction is to keep PPO as the main learner and compare multi-day
to multi-week holding treatments under a 20% maximum-drawdown guardrail. The
current frozen `ppo_holding_duration_v1` design compares a freshly trained,
Observation-v3 H=0 PPO with minimum-hold horizons of 72, 168, 336, and 504
hourly bars (3, 7, 14, and 21 days). This estimates the effect of assigning a PPO system a minimum-dwell
rule; separate PPO training means the trades and later actions may also change.

All arms must bind the same Dataset/time scope, selected features and fit
symbols, PPO seed roster and realized training budget, initial capital,
execution costs/funding/borrow, execution overlay, risk config, and terminal
settlement. The protocol fixes max gross 0.5, max absolute weight 0.1, no
turnover cap, drawdown deleveraging at 10%, and hard stop at 20%, with other
`PreTradeRiskConfig` fields at their defaults. This exact profile is enforced
identically in PPO training and every strategy replay. In v1 each symbol remains
an independent account; the 20% stop cannot guarantee the realized drawdown
stays below 20% after a price gap. The new v2 protocol described below measures
the same PPO horizon question on one 100,000 USDT account shared across the
five symbols.

The age-aware run resolver requires a continuous, exactly regular one-hour
clock, and low-level PPO APIs reject positive minimum-hold durations with the
age-blind Observation v2 schema. Candidate artifacts preserve the suppressed
and unlocked replay events, actual age and quantities, post-risk target, risk
reasons, final inventory, and active/terminal order state. A terminal-settlement
flag does not by itself prove that the account finished flat.

A separate result-blind shared-cash replay capability accepts age-aware
minimum-hold decisions and reserved terminal settlement for multiple symbols in
one account, with a versioned per-decision ledger. The immutable v5 StudyPlan
above still means five independent 100,000 USDT accounts and retains that
historical selection semantics. A new `ppo_shared_cash_holding_duration_v2`
protocol now has a separate `canonical_m2_bootstrap_config_v6` /
`controlled_study_plan_v6` identity. Its
Candidate Run schema v7 persists the combined portfolio return series,
terminal account state, and shared-ledger identity; comparison schema v4
recomputes each seed's combined return / drawdown and selects on shared-cash
results rather than averaging symbol accounts. Mocked bootstrap and full
Study-lifecycle tests exercise this path. The local implementation is not yet
cleared by the required fresh result-blind G0-G2 review, and no v2 fit or
economic replay has been run. The previous sealed one-shot normalization run
36356182462 completed execution but its independent verification failed, so it
published no verified comparison. No verified PPO profitability result exists.

The result-blind `ppo_holding_duration_v1` code path and focused contract tests
are implemented on the `codex/ppo-holding-duration` work branch. On 2026-10-01, the new
`canonical_m2_bootstrap_config_v5` / `controlled_study_plan_v5` path completed a
fresh Binance source freeze and published an immutable Dataset and StudyPlan.
The bootstrap manifest, Dataset manifest, exact input config, and exact StudyPlan
are preserved in a workspace-only result-blind v5 packet at
`report/ppo-hold-duration-v5-20261001/`. It is not part of the versioned GitHub
tree because it contains the exact source roster and Dataset metadata.
Its config digest is
`f31fd955c50dd69d68ae78db4b756a1bc43a4cb425ab7d0f4045327395a83044`, Dataset
ID is `c489ed47a55f2013fcd4f8c1bf560997b8ba41c4ec6dff0d16d0dd95d516a72b`,
Dataset artifact digest is
`27aba635367cc79d2086fd28709c8565684e6ef2c00af2bfad8202ce479661ab`, StudyPlan
digest is `f7e0098a658952e3d3359f6079aa07e97cf6792e3e6b447b95b1ebc75aed78a6`,
and outer bootstrap digest is
`af0ba9ab3f3a88a665a8e6be29072ed21ae3772e6eedcc3733315a9111bcd3bd`.

The older `controlled_study_plan_v4` remains development-only. The frozen
Dataset covers 2021-01 through 2026-08 for BTCUSDT, ETHUSDT,
BNBUSDT, XRPUSDT, and ADAUSDT, with a 1h decision clock plus 4h and 1d feature
streams. The baseline uses the existing 12-feature roster, fit cutoff
2023-01-01, and development evaluation from 2023-01-01 through 2026-08-01.
Every fit requests 262,144 PPO steps across the five fixed seeds. The common
research assumption is 0.05% fee, 0.02% spread, 5% participation capacity,
100,000 USDT initial capital per independent symbol account, and the fixed
20% drawdown stop. The unused final window is preregistered as
2026-11-01 through 2027-11-01 and has not been fetched or opened.

The network-free independent inspection matched the recorded bootstrap, Dataset,
and StudyPlan identities; confirmed `controlled_study_plan_v5`, Observation v3,
H=0, the five seeds, all four horizons, and the sole `PPO_MINIMUM_HOLD` factor;
and confirmed that no baseline or candidate run exists. The initially requested
September 2026 monthly Vision archive returned 404, so the frozen range ends at
the latest complete month available during this bootstrap rather than filling or
substituting missing data.

The protocol fixes H=0 with Observation v3, five ordered PPO seeds, only
`PPO_MINIMUM_HOLD`, four experiments, and the shared risk config. It fixes the
ordered candidates at 72 / 168 / 336 / 504 one-hour bars and refuses to run any
arm until all four have been preregistered.

The duration protocol stores its pre-registered selection rule in the immutable
StudyPlan before outcomes. Eligibility requires every H=0 and candidate
seed-symbol account to complete terminal settlement flat with no active order
remainder, every account's realized maximum drawdown to stay at or below 20%,
and positive median paired excess return versus H=0. Both the primary score and
paired excess first take an equal-weight mean across symbols within each seed,
then the median across the five seeds. The eligible arm with the highest
primary score wins; exact ties go to the shorter hold. Absolute return is not
an extra development-screen eligibility gate. If no arm qualifies, the Study
freezes as NO_WINNER. `controlled_evidence_comparison_v3` stores the per-seed
cells and aggregate needed for independent reconstruction; protocol inspection
rejects a downgraded v1/v2 comparison. This remains a development screen, not a
profitability claim; a frozen winner still needs separate one-shot sealed
unused-future evaluation.

G0 is now bound to the exact Dataset, development/final windows, and immutable
StudyPlan before outcomes exist, but G0 has not passed fresh independent
result-blind review. G1 is fixed in the same plan and also awaits that review.
G2 remains NOT ESTABLISHED until the exact implementation receives fresh
independent result-blind review and all required contract checks pass. Human
review of the updated Guide description is a separate documentation gate
required before its source fingerprints are refreshed; it is not a G2 oracle.
No PPO training or economic replay has started for this duration study. G4
remains blocked; existing M2 results are not evidence for this duration
question. The next sequence is to close G0-G2 on this exact packet, then freshly
train H=0 under Observation v3 before any candidate result is generated.
The local Study workflow does not authenticate an external G0-G2 review; this
remains an operator release prerequisite, and `run_baseline` / `run_experiment`
must not be called until it is closed. Caller-written `assurance-review.json`
is rejected as an unexpected Study artifact and cannot establish approval.

## 研究目的

### Separate paired funding-carry development capability

A separate BTC/ETH spot-long/perpetual-short carry comparison is complete.
It has no forecasting or learned selector.
The source roster is fixed to official hourly/funding archives covering reused
2023-2024 development time; 150 raw archives were acquired and hashed before any
carry economic replay. Software review covers equal-quantity holding, actual
exits, signed funding, fee conservation and futures collateral excluding spot.
The strict-clock source attempt stopped before economic replay: monthly and
independent daily spot archives both lack the 2023-03-24 13:00 UTC open for BTC
and ETH. Binance's official incident report documents a spot halt from 11:27 to
14:00 UTC. Preserve that failed attempt; a separately identified source revision
explicitly masks intersecting halt bins and uses a preceding-close stale mark
only for a missing whole halted bin. Unknown gaps still fail. Software validation
and source acquisition alone are not profit evidence. The subsequently frozen
halt-aware protocol
`aa67889f4facdbad927437cbc65d6ef1f9b6f3e8e6ab2deb3c446570a2183aef`
completed all twelve predeclared replays. The decision is
`PROSPECTIVE_PAPER_DESIGN_REQUIRED`, with all twelve development screens passing.
Shared-account net return over the full two years was +5.3031%, with +2.0269% in
2023, +3.2111% in 2024 and ledger maximum drawdown 0.6145%. Double fees/spread
returned +4.9830%, one-hour initial-entry delay +5.3028%, and 10x lower
participation +5.3031%. Independent BTC and ETH base returns were +5.0491% and
+5.5622%; every base year and every stress full-period return was positive.
All replays completed 17544 intervals, actually closed all positions and had no
stop, unmatched hedge, collateral breach or canonical termination.

The fixed research contract used 10000 USDT shared capital (5000 for each
independent pair), gross 0.5, monthly equal-base quantity targets, common lot
0.001 and minimum notional 10. Fees were spot 10bp/perpetual 5bp plus 5bp adverse
execution per leg; participation was 1% of preceding hourly quote volume.
The irreversible carry drawdown stop was 10%, below the user's 20% research
tolerance. Base qualification required both years and full-period net profit;
all nine stresses required full-period profit. Every arm additionally required
drawdown below 10%, complete execution, actual flatness and no stop/invalidity.
The tests do not prove profitability at larger capital or under venue-specific
account rules. Hourly marks and coarse halt masks do not establish intrabar
drawdown. The apparent equity spike at the March spot halt reflects asynchronous
stale spot versus current perpetual valuation; it is not a realized windfall.

An independent auditor, without importing the simulator, reconstructed all twelve
cash/quantity/equity paths from recorded fills and the frozen market arrays,
including costs, signed funding, capacity, actual flatness and collateral checks.
Shared base final cash was 10530.3127 USDT, funding income 564.1770 and execution
costs 31.1492. The final selection SHA-256 is
`8196ceebe637d39d906a8f1559f37a7f7783bef8eee8de24d3f00c8f3b1f3776`.
This is positive reused-development evidence, not unused-future validation or
permission for live orders. Perpetual-close mark proxies, assumed fees/rules,
instant wallet transfers and unobserved order-book execution remain limitations.
The next active design is `docs/specs/funding-carry-paper.md`. Completed PPO and
carry studies retain their original frozen source and evidence.

Forward source capture has passed permanent CI and real public-feed acquisition.
The separate recorded-depth paper matcher models partial bid/ask fills, declared
fees and adverse pricing on the canonical account. These are software components;
current public-rule capture and verified snapshot readers are available, while
an append-only protocol-bound journal has been implemented and tested with
concurrent writes and interrupted transactions. The deterministic paper engine
now composes saved decisions, verified later quotes, partial fills and settled
funding on the existing account, and verifies every committed result on restart.
Funding uses exact quantities held before the settlement timestamp, including
when payment evidence arrives after an exit. Revisions, late/missing funding and
observation gaps remain visible permanent quality failures. A standalone
source/runtime-bound collection component now seals its identity before start,
refreshes public rules, supervises pending decisions and durably halts on failures
or interrupted cycles. The independent real-public-data supervisor smoke executed
four entry and four exit fills, reopened identically and reconciled to exact zero
quantities. Final cash was 9987.8435571 from 10000 virtual USDT, fees 7.4208051,
and funding income zero. This brief software check demonstrates execution costs,
not profitability. Its earlier helper-error attempt is preserved as failed.

The cadence CLI and fixed prospective screen now support seal/run/status/evaluate.
The screen lasts ninety UTC days with three thirty-day blocks and unchanged
10000 virtual USDT/default cost assumptions. It requires positive full-period
and each-block returns, net profit above recorded fees, drawdown below 10%,
funding receipts in every block, no unpaid announced settlement, complete
source/control evidence and actual terminal flatness. Fee headroom is assessed
on the realized trajectory, not a separate doubled-fee strategy replay. Minute
observations cannot guarantee intraminute risk. Status output is operational;
only full offline replay at the fixed deadline may decide the paper screen.
The first formal prospective screen is sealed under protocol
`843467870d85d0b085e65cf904e9d458287c14fc1e31615b9aaf77029f0669b7`.
Its fixed UTC start is 2026-09-17 20:05, close is 2026-12-16 20:05, and terminal
observation deadline is 20:08 that day. The dedicated detached worktree
`funding-carry-forward-20260918` and its private locked environment remain frozen;
the collector was launched before start. Its source implementation digest is
`5e093425ee775639b2ac840831e8a71f30c34966b6150367e40bdc0b7f6ed349`
and runtime environment digest is
`0c56fc67d16f583d49b5dea7a3878e3fcba0f58e6abd0e9dd66e51393c994cc9`.
The collector produced 188 observations; its last committed observation was
`2026-09-17T23:11:01Z`. A 2026-09-27 audit found the recorded PID 23548
absent and no `collection-failure.json`. Journal-only status verification
confirmed 188 events through tip
`1a84746aa875ee5a21b72944a5101c2ffb5b00860c9d085a1ccb1fe780d5836b`.
The last stored status retains quantities `0.016`, `-0.016`, `0.509`, and
`-0.509`, marked equity 9993.9707, zero funding income, and `terminal_flat=false`.
No shutdown reason was recorded; operational status is not an economic replay.

The observation gap now exceeds the frozen 180-second limit, so this protocol
cannot qualify. Preserve its incomplete evidence without resuming or backfilling
it; this operational journal does not establish whether an economic replay ran.

The paper screen contract is now version 2. It declares one required fill in
each of BTC spot, BTC perpetual, ETH spot and ETH perpetual, and assigns funding
block coverage by the published settlement time inside the fixed ninety-day
window. Receipt during terminal grace cannot make a post-close settlement count
for block 3; any nonzero post-close settlement now rejects the screen even when
the credited cash appears in final account equity. A new sealed run is attempt 2
and must bind attempt 1's protocol
SHA-256 `843467870d85d0b085e65cf904e9d458287c14fc1e31615b9aaf77029f0669b7`,
final journal tip `1a84746aa875ee5a21b72944a5101c2ffb5b00860c9d085a1ccb1fe780d5836b`,
and exact last observation time `2026-09-17T23:11:01.134389+00:00` with disposition
`invalidated` and reason `observation_gap`. The protocol digest and event tip
were re-read from the preserved root on 2026-09-28. The old chain itself has no
terminal gap event; the invalidation follows from the frozen gap limit and the
verified time since its final event. A new attempt must never resume or rewrite
that root.

Attempt-lineage v2 accepts only operational `invalidated` / `incomplete`
dispositions and fixed operational reason codes; prior screen pass/reject labels
and economic metrics are excluded. These lineage values are still assertions,
not root authentication. A fresh result-blind reviewer must resolve attempt 1
with its preserved v1 reader, verify the protocol/event chain and gap, and check
for an earlier unreported attempt before any successor is sealed. The schema by
itself cannot authenticate hashes or prevent a new root from claiming attempt 1.
Any successor needs a separately sealed
protocol with a new future start and current source/runtime identity. Prospective
profitability remains unproven.

Fresh result-blind review independently verified the known attempt-1 v1
protocol digest, all 188 canonical journal events and parent links, final tip,
and last observation `2026-09-17T23:11:01.134389+00:00`. At the minimum reviewed
seal time `2026-09-27T16:37:40Z`, the observation gap was 840398.865611 seconds.
The review found no earlier fixed carry-screen root among 18 protocol files in
the inspected `C:\dev\trade_rl` workspace. G0 for this predecessor is verified
within that scope; global attempt uniqueness outside the workspace is not
established. G2 review confirms that the four-instrument fill roster is bound to
the sealed plan and funding blocks use settlement time in `[start, close)`;
their targeted synthetic tests pass. Overall G2 and an independent full
source-to-ledger replay remain unestablished. No successor screen has been
sealed.

A subsequent real one-minute CLI software probe exposed partial ETH spot depth:
the 20-level capture filled 0.3821 ETH against a 0.51 ETH perpetual short. The
engine rejected the unmatched hedge, actually flattened all four legs on the
next observation, and exited with code 1. Final cash was 9988.8165415 USDT,
fees 6.7969229 and funding zero. Preserve that rejected software attempt.
New capture uses an explicitly versioned 100-level profile; the old v1 reader
remains available for evidence verification. This expands observed price levels,
while retaining 10% participation, costs, risk and qualification conditions.
It does not establish that later quotes would have filled the earlier orders.
The separate 100-level software probe then completed its whole minute cadence
and terminal grace, with four completed cycles, six journal events and eight
fills. It reopened identically, ended actually flat and had no quality failure.
Final cash was 9987.8380018 USDT, fees 7.4237759 and funding zero. This remains
excluded from the formal economic screen. The dedicated study environment also
passed all 155 paper and forward-source tests before sealing.

### Directional development study under a 20% drawdown budget

The complete 13-arm study returned `NO_QUALIFIED_CANDIDATE`; all five PPO seeds
lost money. Their net returns were -17.1134%, -15.4319%, -13.3033%, -13.7667%,
and -15.0823%, with median -15.0823%. Channel breakout returned +1.4456% but
failed the 2024-positive and terminal-flat gates; the constant-long control
returned +14.8059% and also retained terminal holdings. No stress replay was
reached because no candidate passed the base screen. Independent raw-return
arithmetic and file/model hashes passed; this is not an independent order-ledger
reconstruction. The original immutable protocol is
`aa9bf63e43668a2edbc0c8a54a5c6e689a277de4fb9e2af7862892914e783000`.

The subsequent five-seed risk-only comparison completed under separately frozen
protocol `eafa0327189021986116aefce33936a34e56409de7b3b53e8ba337c969e149c4`,
bound to all baseline result hashes and the completed selection. Its frozen
source passed both permanent CI jobs; exact source revision, workflow identity,
and raw snapshot hashes are retained with the study's run-context evidence.
This software verification is not economic evidence.
Its decision is `KEEP_BASELINE`: paired net-return improvements occurred for only
three of five seeds, below the registered four-seed requirement. Candidate net
returns were +12.2341%, +4.3244%, -12.5830%, -19.5275%, and -17.5039%; the median
was -12.5830%. The median paired improvement was +0.7203 percentage points;
this differs from subtracting the two family medians. No seed passed all base
gates, so stress replays were not reached. Seed 0 had positive full and both-year
returns but retained terminal holdings. All five completed 262144 training steps
and all 17544 evaluation intervals. The independent audit verified saved model
parameters, raw-return arithmetic, hashes, and the aggregate decision, but did
not independently reconstruct an order ledger. Positive individual seeds do not
qualify this PPO family. The default training risk remains unchanged.

The risk-only relative gate requires at least four paired net-return wins,
a positive five-seed median paired delta, complete baseline and candidate
replays, candidate ledger drawdowns at most 20%, and no increased termination
count per seed. Cost and turnover are diagnostics. The absolute profitability,
terminal-flat, and stress gates remain separate; relative improvement alone
never authorizes deployment. Baseline model/result/selection hashes, dataset,
source snapshot, and identical runtime are frozen before fitting candidates.

Source and synthetic execution diagnostics also identified an admission/ledger
quantity inconsistency: accumulated floating-point lot additions/subtractions
can leave approximately one lot that admission accepts but fill allocation
rejects. Genuine below-minimum-notional residuals are a separate issue under the
current execution contract. Neither terminal holdings nor thresholds are altered
in the risk comparison. A seed that did end flat still lost 13.7667%, so terminal
flatness alone does not explain the observed lack of profitability. A future
execution fix requires its own reviewed contract and fresh evidence; historical
results remain unchanged.

The user subsequently explicitly prioritized PPO and data/training improvements.
A train-only feature audit and source audit found that the default training risk
does not match the directional evaluator: training allows gross/per-symbol 1.0
and drawdown start/stop 1.0, while evaluation uses gross 0.5, per-symbol 0.1,
drawdown start 0.1 and stop 0.2. The completed comparison changed only these
training risk settings. Training still has one active symbol per account while
evaluation shares cash across five, and policy inputs lack account drawdown.
The optional risk configuration does not resolve these remaining mismatches.
This is not a reward
cost omission: the current PPO reward already uses log net return after costs.
Fill counts combine policy and risk actions and cannot alone diagnose churning.
Feature-scale normalization and the sealed interleaved experiment remain separate.

An opt-in fit-only PPO feature standardizer and bound model-bundle capability
are implemented separately from the completed, immutable risk comparison.
The durable preprocessing/model contract is in `architecture/lean-core.md`.
Synthetic scope, masking, default compatibility, balanced statistics and
model/transform reload checks and full software CI pass. The five-seed normalized
real-data comparison completed under frozen protocol
`464278c3390abd20e6741c7176dcc7382a3679f5351de8aa245f734fc971ff0c`.
The decision is `RELATIVE_IMPROVEMENT_ONLY`: four of five paired returns improved,
and the median paired delta was +3.8399 percentage points. Net returns were
-8.2816%, -19.0971%, -9.4633%, -9.1378%, and -12.0839%; median -9.4633% versus
the original -15.0823%. All five remained unprofitable and retained terminal
holdings; no seed passed the base screen and no stress replay was reached.
Median turnover increased from 48.5957 to 364.0627, a diagnostic to investigate,
not proof of policy-driven churn. All five completed 262144 training steps and
17544 evaluation intervals. An independent arithmetic/hash/model audit also
reconstructed fit-only balanced normalizer moments from raw source arrays and
confirmed the aggregate gates. It did not independently reconstruct order ledgers.
The final comparison SHA-256 is
`fab5e1e13c35263533f3f36f62ce7a7017fe92e5413b58a56f61f84ba910df33`.
The isolated comparison retained original default risk, source, budget and gates;
it does not establish operational profit or reopen the sealed interleaved study.

That historical normalization result remains bound to the economic implementation
that produced it and is not a current-economics control. A separate corrected-
economics normalization replication protocol is sealed result-blind. Its current
software boundary requires ten fresh matched fits (`control_raw_seed0..4` and
`candidate_normalized_seed0..4`), delegates both arms to the maintained
`fit_ppo_strategy` / `DIRECTIONAL_BASE_EXECUTION_COST` / PPO inference bundle /
shared-cash directional evaluator, and differs only by fit-only normalization. The
sealed protocol keeps its preregistration-time source-blob provenance unchanged; the
post-prereg feature-schema correctness fixes are part of the separately activation-bound
current implementation identity rather than a rewrite of the protocol bytes.

The hardened execution boundary makes `prepare_replication_execution` the only
root-creation transition and publishes a completely validated sibling staging tree
atomically. Slot claim/failure transitions are private and derive activation and
implementation identity from the prepared root; pre-fit and consumed failures are
schema/slot/arm/seed/normalization/chronology checked. Source bytes and runtime
identity are rechecked after long fit/replay before durable publication, bundle
parents and manifest are validated before SB3 deserialization, and both fitted and
reloaded policies must report exactly 262,144 timesteps.

The implementation deliberately commits `ppo_normalization_activation.json` with
`activation_sha256=null`. This non-Python authority file is outside the Python-only
candidate implementation digest, so a result-blind activation can bind the exact
reviewed implementation without creating a self-referential implementation hash.
The activation must bind the immutable implementation-seal, fresh-reconstruction,
and assurance-review digests. A local `verified.json` does not count as independent
verification by itself: comparison publication additionally requires all ten
verification identities to be bound by a fresh verifier artifact authority carrying
repository/run/artifact identity, raw SHA-256 and matching API digest. The verifier
runtime contract matches Python implementation/version, machine architecture, OS
family, and the complete bound package map while treating kernel release as recorded
provenance rather than an equality gate.

The repository now has a separate authenticated one-shot transport capability in
`tools/ppo_normalization_actions.py` and
`.github/workflows/ppo-normalization-execution.yml`. It accepts only an open Draft
execution-request PR whose sole delta is the canonical request record, requires that
exact HEAD to contain current `main` and pass Core / real-PPO / Guide / generic
independent-review gates, revalidates the merged implementation seal/review tags and
the frozen source Artifact, and requires the static repository activation tag to be
absent. The request HEAD is not the economic implementation authority: the canonical
request separately binds the reviewed source SHA from #770, and execution/verifier jobs
checkout that sealed source even when current `main` has moved. The execution job builds
the activation from that sealed source plus its own runtime provenance, creates the
repository-global activation tag before any slot is consumed, and keeps all ten fits on
that exact source/runtime. Complete execution evidence is uploaded only
after all ten slots finish; a failed activated run can expose only a non-economic
failure receipt. A separate no-refit verifier re-downloads the complete execution
artifact by id/run/raw digest, and the finalizer reveals the comparison only after a
fresh verification artifact is itself API/digest-bound.

The training mechanism still uses one-active-symbol episodes with
`risk_config=None`, whereas evaluation uses the maintained shared-cash directional
account and 10%/20% drawdown hard-risk semantics. That mismatch is common to both
arms and therefore does not change the normalization-only factor, but it limits the
absolute claim.

The one-shot authorization was consumed by GitHub Actions run `36356182462` on
2026-09-27/28. It used request HEAD `ee8dde846286` and sealed execution source
`72ec5a082a1e`. The run created the repository tag
`activation/ppo-normalization-corrected-v1` (annotated tag object `b2120e442833`, activation digest
`985d34eabba5629fb934203450697a8691bea8819dfc9ac3bc7fc43b7e0652a3`), completed
all ten fit slots, and uploaded execution artifact `10946283192` with raw SHA-256
`0c0d5335c389f13fe7b1ff22bd53d97025cc4b39253f280843baeba06024a81f`. The fresh
no-refit verifier job `108745022839` failed with
`fresh bundle replay differs from published result`; the finalizer was skipped.
Static source tracing found that the verifier compared incompatible envelope fields:
slot publication replaces the directional evaluator's top-level `schema` with the
slot-result schema, while fresh replay retains the evaluator schema. The verifier
then compared the fresh schema against a stored payload from which `schema` had
already been excluded, so every otherwise-matching slot failed. This PR adds a
regression test and changes the comparison to ignore only that unpersisted replay
schema; all other replay fields remain strict. The original execution artifact and
failed verification remain unverified: no no-refit verification has been rerun, and
no economic result was inspected. Do not re-trigger this one-shot request or refit
these slots.

The source authority file still has `activation_sha256=null`; the separate
repository activation tag records the consumed run. No verified comparison or
economic disposition was finalized, and no unused-future evaluation was accessed.
The activation metadata records `economic_result_inspected=false`,
`final_test_accessed=false`, `production_eligible=false`, and
`live_trading_authorized=false`. The run therefore establishes neither a profitable
normalization candidate nor production/live eligibility.

The user subsequently broadened the search to other RL algorithms, ensembles
and additional data. These are permitted future candidates, subject to the same
cost, drawdown and out-of-sample evidence requirements. The completed
standardization study changed only its registered preprocessing factor.
Exact fill-quantity accounting now preserves accepted lot counts and genuine
remainders across book/order updates and resume; capacity allocation searches
integer lots against the actual monetary bound. This separate implementation
does not alter any active frozen study or its historical results.

### PPO/A2C update-family comparison: draft contract and synthetic oracle only

The result-blind PPO/A2C code-contract draft and pure cell-decision oracle are
implemented in `trade_rl.evaluation.rl_family_comparison`. The proposed study holds the existing
12-feature task and execution contract fixed, fits the full five-symbol roster
before 2023, and screens the already reused 2023–2024 development period under
base, doubled-cost, and one-bar-latency scenarios. PPO and A2C each have five
matched seeds and a 256,000-transition budget. Absolute qualification is
independent for each family; paired A2C uplift can select between them only if
both independently qualify.

Each family needs at least four of five seeds with positive full-period and
annual returns in every scenario on at least four symbols, plus positive
five-seed medians. A paired symbol votes for A2C only with four of five
positive same-seed deltas across all years and scenarios and positive median
deltas; A2C can be selected by uplift only when both families qualify.

Mocked PPO/A2C constructor tests bind the registered hyperparameters to the
actual fitter arguments, and synthetic-cell tests exercise complete coverage,
profit screens, paired decisions, the 20% per-symbol drawdown boundary, and
execution-risk vetoes. No Dataset, model, or replay artifact was opened, and
no new economic result was generated. No model fit or replay has been run, and this contract
does not provide an execution workflow or independent G0–G3 reviews. The 20%
per-symbol limit does not satisfy the requested portfolio-level drawdown target;
a future candidate still needs a separate shared-capital screen. Profitability,
future-data validity, paper eligibility, and live readiness remain
`NOT ESTABLISHED`.

### PPO BTC-relative feature ablation: G4 completed, KEEP_BASELINE

PPO BTC-relative feature ablationはG3/G4まで独立監査済み。
absolute base profitabilityは0 / 5 symbols。doubled-cost / one-bar-latency stressは0 / 5 symbolsでfailし、
development decisionは`KEEP_BASELINE`。G5とProduction/live eligibilityは未判定である。

A dedicated private runner, `trade_rl.evaluation.ppo_feature_study`, and its
result-blind protocol are implemented. The first preregistered attempt was
started on 2026-09-20, then stopped during the fifth baseline arm's development
replay when the host approached its memory-commit limit. Four baseline arms had
completed; the fifth model was saved, but its ledger replay and the paired study
were incomplete. The attempt was not finalized, and its economic outputs were
not inspected. Its partial write-once artifacts remain preserved under
`output/ppo-btc-relative-feature-ablation-20260920` and must not be combined with
a later protocol.

They test one narrow question: whether adding three existing BTC-relative
return features helps the current PPO candidate under its fixed development
screens. The exact frozen successor Dataset/Study artifact is preserved in the
companion `directional-profit-bot/output/source-successor` worktree. Its
Dataset, artifact, StudyPlan, evaluation-Dataset, and protocol identities were
re-resolved and matched the fixed values. The replacement run uses the same
fixed source and economic comparison, with a memory-bounded ledger observer.
Do not substitute the separate 2024–2026 realdata generation. The
baseline feature roster is compared with the same roster plus
`1h__relative_return_to_btc_1bar`, `4h__relative_return_to_btc_1bar`, and
`1d__relative_return_to_btc_1bar`. This is a PPO feature ablation, not a
comparison of PPO against other RL families.

The protocol pairs two arms over the same five seeds (0–4), sequential PPO
layout, 262,144 requested training steps per seed, full fit-symbol roster,
pre-2023 fit cutoff, `max_gross=0.5`, `max_abs_weight=0.1`, no turnover cap,
drawdown start 0.1 / stop 0.2, and the same execution/accounting implementation.
That is ten fits in total. Evaluation uses five independent 10,000 USDT accounts
(one per BTC/ETH/BNB/XRP/ADA symbol), each with gross budget 0.1, over the same
17,544 development intervals from 2023-01-01 00:00 to 2025-01-01 00:00. Return
intervals are grouped by their start timestamp: 2023 has 8,760 intervals and
2024 has 8,784; the final 2025 timestamp closes the last 2024 interval.

Candidate base and stress cells all carry a global hard veto: any seed-symbol
cell with ledger drawdown above 20%, termination, a non-flat terminal position,
or an active order remainder blocks admission. A seed votes for the absolute
return screen only when full and every-year returns are positive in base and
both stresses. Four of five same-seed full-return deltas must be strictly
positive for the paired vote, with positive five-seed median full and each-year
deltas. These economic votes never relax the hard veto; all candidate medians
use all five seeds. The candidate alone is replayed under doubled execution
cost and one-bar latency, across every seed-symbol cell. A pass can request a
prospective paper study only.

The independent result-blind G0–G2 review passed on 2026-09-20 for the original
protocol
digest `09ec5e9e051c7867686dcac9290f6d6a32120c8db459069439386c286f8bbf44`,
implementation digest `58f1e0e801b094df5fc5b8dfe683f8f55edcc5955dc5251e77244497f56dbb62`,
and source snapshot `e04210707a33d812bd3e41b7907528658d17f94869d1950772d48389d3d4bce2`.
The reviewer independently rehashed the implementation and source snapshot;
the protocol digest was supplied as the fixed binding. That review authorized
only the original exact source; it does not carry over to the memory fix. The
replacement protocol was prepared on 2026-09-21 at
`output/ppo-btc-relative-feature-ablation-20260921-r1`, with protocol digest
`a25aa21fcfb2b4c17c83f7fc465a49b1e08171704742563511a24e9933b07fb3`,
implementation digest `f3db8d4f070f3d3bd21c73cd35462c5f87405c79774140ff3e7e4c00162313e4`,
and source snapshot `8a994f7e3d7f6961edff9363f8c65b52e534a391970bde43d3f6f4281b27dc16`.
Its independent result-blind review completed on 2026-09-21: G0 PASS, G1 PASS,
and G2 PASS for this exact binding. The baseline seed-0 fit started on
2026-09-21 and was safely interrupted before fit completion after available
physical memory fell to 1.18 GB on a 15.75 GB host (92% load). Its arm directory
contains only `started.json`; no model, result, ledger, or economic output was
published or inspected. The partial start marker is preserved. A retry needs a
new write-once output root, and must not combine with the 2026-09-20 attempt.
The reviewers confirmed G0 and G1 for the fixed paired mechanism. G2 PASS is
bound to the exact implementation
digest and source snapshot above; its ledger validator does not independently
recompute P&L from persisted order/fill events, and its ledger schema does not
carry an expected symbol index. The current generator passes the symbol index
through single-symbol replay, keeps other symbols flat, and validates row
identity, so reviewers found no current-generation mismatch. These are limits
of the evidence verifier, not a claim that event-level P&L has been independently
reconstructed. No economic outputs from either attempt were inspected when the
reviews were performed. These reviews do not establish completion of repository
quality gates or any economic result. The Dataset and 2023–2024
development interval have already been used by prior research, so this
experiment is not confirmatory, is not an unused-data validation, and cannot
establish general profitability. No production or live-trading claim follows
from its outcome.

After the r1 interruption, source review found `run_arm` retained both the raw
Dataset and the immutable price-channel-augmented Dataset for the full fit and
replay. The code now releases the unused raw reference immediately after
augmentation in both `expected_protocol` and `run_arm`; a weak-reference test
was RED before this change and passes at fit entry. The 61 feature-study tests,
Ruff, and package mypy passed. New exact protocol
`output/ppo-btc-relative-feature-ablation-20260921-r2` binds protocol digest
`e5e11eda3c3206087752e184381931eedc8d94efd6fc21679457b6cfb71633b0`,
implementation digest `7d685cd2f83e59c149171f7c367a95332faff69f56a66542a7124a1faa5a908d`,
and source snapshot `cb55da0e9d53a2e4af53aed0ab8f5fc25e98238ff73292681b9ec71c108f7b37`.
Its independent result-blind review completed on 2026-09-21 with G0 PASS, G1
PASS, and G2 PASS for this exact binding. The G2 reviewer confirmed the
lifetime-only change does not alter Dataset values or training/evaluation
semantics; the memory reduction itself has not yet been measured in a full fit.
The reviewer did not reload the companion Dataset/Study artifacts; local
`reserve_study()` re-resolved their identities while preparing r2. G2 retains
the earlier ledger-verifier limitations: no event-level independent P&L
reconstruction and no expected symbol index in the ledger schema. Baseline
seed 0 fit completed and entered replay on 2026-09-21. Replay was safely
interrupted when host available physical memory reached 1.479 GB of 15.75 GB
(90% load), below the 1.5 GB stop line. Preserve r2 as incomplete: it contains
the fitted `model.zip`, `started.json`, and base ledgers for symbols 0, 1, and
2 of 5; symbols 3 and 4 and the arm result were not published. No economic
output was read. Do not finalize this root or combine it with the original or
r1 partial attempts. Any retry needs a new output root and at least 4.0 GB of
available physical memory at preflight; interrupt again if availability falls
below 1.5 GB.

The checkpoint runner is implemented at
`trade_rl/evaluation/ppo_feature_checkpoint.py`. It uses a separate protocol
identity and atomic completion boundaries for fits, individual
seed-symbol-scenario replays, arm assembly, and comparison. A retry verifies
completed evidence before reuse; missing work may be rerun, while tampered
completed evidence fails closed. An interrupted fit restarts rather than
resuming partial optimizer or rollout state. The original, r1, and r2 roots
remain preserved as incomplete evidence and are not inputs to this runner.

The exact Linux protocol was prepared on 2026-09-21 under immutable tag
`seal/ppo-btc-relative-checkpoint-20260921-v1`. Hosted preparation run
[35537964829](https://github.com/shuntatsu/trade_rl/actions/runs/35537964829)
succeeded and published artifact `10612544268` (raw ZIP SHA-256
`7f776884512bab19f32b50b13e2f47bb7bd3928ef8c2b562c469cb1d6dbbc7b3`). The
outer protocol digest is
`e04d0146fea39bdbb95e1b78ed5b94b2fead296f67774f1007f0496236e9b5d2`; the
source snapshot digest is
`7fb2b501b79dda20bc32ddf69b0e5b56d881f1d2bffce68649a7020ec085eb97`, with
all 170 snapshot files matching Git. Fresh independent result-blind review
passed G0, G1, and G2 with disposition
`G0_G1_CLEAR_G2_EVIDENCE_BOUND`; its record digest is
`2579a7214cf74a394821d80568be7864a963a9b5fa1a0e9c7197e1585591fe9b` ([review
record](https://github.com/shuntatsu/trade_rl/pull/744#issuecomment-5752780311)).

All ten fixed arms have been dispatched under that same frozen protocol: baseline
seeds 0–4 are runs 35538782635, 35538813730, 35538815588, 35538817265, and
35538818919; candidate seeds 0–4 are runs 35538820441, 35538822037,
35538823806, 35538825367, and 35538826669. The unchanged 12-feature baseline
and 15-feature candidate use five seeds and 262,144 requested steps per fit;
their preregistered roster contains 100 replay cells. Each replay starts with
an independent 10,000 USDT per-symbol account, and every candidate cell remains
subject to the 20% drawdown hard veto. The shared-portfolio, stress, and
development-only limits above remain in force.

The exact hosted run completed on 2026-09-21 as run `35542548393`, attempt 1,
and published artifact `10614569766`. Its ZIP is 589,319,909 bytes with
SHA-256 `3997d64142c9143085ea954ef340905c4764ce58d264e21ade791b8d41f7120f`.
The frozen source head is `b876f1c5b5a7`; checkpoint protocol digest is
`e04d0146fea39bdbb95e1b78ed5b94b2fead296f67774f1007f0496236e9b5d2`, and the
canonical core-protocol digest is
`e37701b92ddfb93f3bc6d528e1affebcd692db929415fa1b27ff65ea94dbd475`. The
independent G3 audit passed: its record SHA-256 is
`3a2b8d353aa4e176e0dacb1bf9963cfb2995e905d51611b37641f96d5fbf8e5a`. It
validated the exact archive roster, source provenance, ten completed 262,144-
step fits, 100 replay ledgers, and their checkpoint-to-arm identity before any
economic payload was opened.

An independent G4 audit then reaggregated all 100 cells and 17,544 intervals
per cell from the exact artifact. It checked ledger equity/return chains,
preregistered start-year slices, recorded drawdown traces, terminal quantities,
active orders, and arm-cell equality without importing the frozen evaluator or
simulator. Its source-comparison recomputation is `MATCH`; both report
`KEEP_BASELINE`. The committed [G4 audit record](../../report/ppo-btc-relative-feature-ablation-g4-20260922.json)
has SHA-256 `cd01a193ca2a6fa34355873fdf21b62f5f425977b2d05a388259f69cf753f481`;
the [exact audit script](../../report/independent_ppo_feature_g4_audit.py) has
SHA-256 `1dc742dd479617e1aacca5251fd3c9e5a46951e8e284c562e877b52f40d86e9e`.

| G4 gate | Result |
|---|---|
| Candidate hard guards: drawdown at most 20%, no termination, terminal flat, no active remainder | PASS; zero violations; maximum across candidate base and stress cells 19.91485% |
| Absolute base profitability | FAIL; 0 / 5 symbols qualify |
| Paired relative screen | FAIL overall; ETHUSDT alone qualifies |
| Doubled-cost and one-bar-latency stresses | FAIL; 0 / 5 symbols qualify |
| At least four common symbols | FAIL; 0 qualify |
| G5 unused-future and production/live eligibility | NOT ESTABLISHED |

Median full-period returns (baseline → candidate) were BTC −14.11% → −14.48%,
ETH −13.18% → −6.66%, BNB −12.06% → −17.82%, XRP −5.09% → −17.38%, and ADA
−15.29% → −15.34%. Every candidate median and every baseline median was
negative. ETH showed relative improvement, but not positive absolute returns
or passing stress results. `KEEP_BASELINE` therefore means no feature-augmented
candidate qualified; it does **not** establish that the baseline is profitable.
The study reuses 2023–2024 development data and does not qualify a prospective
paper stage, a winner, or live trading.

The audit reaggregated P&L from portfolio-value snapshots persisted in these
ledgers. It did not fetch raw market bars or independently replay source fills,
so it does not establish a second market-data-to-execution oracle. The audit
used the persisted full-ledger maximum-drawdown trace and cross-checked it
against independently recomputed interval-end equity drawdown; it cannot
rebuild every intrainterval mark without the frozen Dataset. Annual returns
were grouped by registered interval-start year; the artifact's interval-end
year diagnostic is not the admission oracle.

The checkpoint implementation passed the full Linux repository, PPO runtime,
distribution, clean-install, and Guide browser checks at the implementation
head; its independent code review found no blocking integration issue. Hosted
execution uses the manual `ppo-feature-checkpoint.yml` workflow on standard
Ubuntu runners because the local host did not retain the required free memory.
It retrieves the frozen source from artifact `10331899302`, run `34803217815`,
with outer SHA-256
`89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce`.
The publisher's verification stopped at a SHA-prefix comparison. Its existing
verification-only recovery, run `34803432434`, subsequently passed independent
source reconstruction and the verification audit for this same bundle.
Recovery artifact `10332575500` has outer SHA-256
`2067c38da3c1f45927748e2cc7481f11268710b2b386a5bd3b9a289394937589`;
its downloaded bytes, bundle binding, Dataset/Study identities, and explicit
no-P&L assertions have been checked. It does not replace the new study review.
The preparation artifact and independent review are bound to the same frozen
source, workflow revision, Linux runtime, and outer protocol digest. Each arm
dispatch reuses that binding, saves completed checkpoint evidence, and
preserves failed attempts. Run dispatch alone does not establish G3 or G4.
This transport does not import the original, r1, or r2 partial roots.
After the strategy-owned feature schema was integrated, the checkpoint replay
wrapper was updated to retain the loaded policy's feature names. A focused
counterexample failed before the forwarding fix; the real SB3 roundtrip also
checks that loaded and replayed policies retain the fitted schema. This is an
identity-preservation repair and does not change the registered feature roster.

### PPO 4h indicator smoke: preregistered, not yet executed

A new development-only smoke is preregistered for the single-symbol PPO path.
The user's shorthand is fixed before results as **MACD + ATR + DI± + Ichimoku
on the native 4h feature clock**. The selected observation roster is exactly
ten local features: three 4h MACD values, 4h ATR%, 4h +DI/-DI, and four 4h
Ichimoku distances/cloud descriptors. No symbol ID, cross-sectional feature,
normalization change, shared-cash training, reward change, threshold search, or
PPO hyperparameter change is part of this smoke.

The decision clock remains the maintained 1h clock. Only the selected input
features come from the causal 4h-native feature stream; changing the trading
clock to one action every four hours would be a separate factor and is not
mixed into this test. One common policy is fit across the full five-symbol
pre-2023 roster, while every 2023-2024 replay uses an independent 10,000 USDT
single-symbol account. The fixed seed is 0 and the requested training budget is
100,000 PPO steps. Base execution is accompanied by doubled-cost and one-bar
latency stresses under the maintained hard-risk contract.

This is intentionally a one-seed screening experiment on already-used
development data. It can only promote the hypothesis to a new preregistered
five-seed study. Promotion requires all hard guards in all 15 replay cells,
positive base return on at least four of five symbols, positive cross-symbol
base median, positive 2023 and 2024 base medians, and positive medians under
both fixed stresses. Otherwise the lineage stops after the smoke. Even a pass
does not establish general profitability, unused-data evidence, production
eligibility, or live-trading readiness. Economic execution remains blocked
until exact-head Full CI and a fresh result-blind G0-G2 review are bound to the
fixed contract. This smoke does not accept a same-author review as that fresh review. The
trigger gate requires a formal GitHub PR review on the smoke's owning pull
request from a distinct GitHub principal, exact review `commit_id`, exact
review-body hash, and a
canonical `ppo_4h_indicator_source_review_v2` payload whose code/contract
identity, reviewer-independence status, G0/G1/G2 outcomes, blocking findings and
development-only authorization all agree with the trigger record before source
download or PPO fitting may start. GitHub now exposes the waiting state through
an `Independent Research Review` check that runs only after Lean Core, real-SB3
PPO Runtime, and Human Guide verification succeed. Review submit/edit/dismiss
events repeat those checks and then refetch the current formal-review inventory;
only reviews passing the execution gate's canonical validator can satisfy the check.
`PENDING` therefore means software verification completed but the exact-head
independent result-blind review is still absent or invalid; `READY` means only
that the authenticated review-evidence transition may proceed, not that PPO
economics may run directly.

A separate trusted reviewer transport is being integrated in `tools/ppo_4h_gemini_review.py` and `.github/workflows/ppo-4h-gemini-review.yml`. Its authority model separates canonical request/packet generation, secret-free exact-SHA Core / PPO Runtime / Human Guide verification, and a fresh secret-bearing Google Gemini review job. The reviewer attestation binds exact code/tag identity, frozen trusted workflow/runner identity, request authority, canonical result-blind packet digest, trusted verification job identities, system-instruction/Gemini-request digests, raw Gemini-response digest, and returned Gemini model/response identity. The review job does not execute target code or consume artifacts produced by target-executing jobs. This capability is **not yet an authorization for this smoke by itself**: after the transport is independently reviewed and merged, its exact implementation identity must be frozen, and the active execution gate must consume and independently verify reviewer-run evidence from that frozen authority before same-GitHub-principal posting can replace its current distinct-principal requirement. No Gemini reviewer artifact, PPO fit/replay, or economic result is claimed by merely integrating the transport capability.

The repository now has an A2C intent adapter, sequential CPU fitter, and
algorithm-specific inference bundle over the same `Discrete(3)` environment.
Synthetic CPU fit/save/load tests check the explicit config, rollout rounding,
nominal fit-budget metadata, feature binding, and A2C policy family. This is
software validation only: no research A2C candidate fit, replay,
PPO-versus-A2C comparison, or economic evidence exists. The next research step
remains an algorithm comparison before ensembling, using fixed data, features,
fit scope, account, and execution contract. Test DQN only as a later, separate
factor because it adds replay-buffer and exploration settings.
Do not combine policies unless independent candidates first pass the same
out-of-sample gates and their errors show useful complementarity.

Data improvement should also be isolated from the learner comparison. The
Binance aggTrades parser is not connected to the canonical Dataset feature
builder, and those records do not provide exchange publication or client
receipt times. Any signed-volume-flow feature therefore needs a documented
availability lag and verified raw coverage before it enters a study. Broader
regime and symbol coverage is preferable to adding many unverified indicators.

The separately preregistered corrected-accounting replication in Issue #645 then
ran the unchanged 13-arm screen from frozen source `c80652a12678` plus only
the accounting-correction source `95a4831bfbd5`. Official run `35287338444` on
workflow HEAD `6dcbde815de7` completed exactly once with all 13 immutable arm
slots and an independent no-refit publication audit. Its canonical selection is
`NO_QUALIFIED_CANDIDATE`, `winner=null`, and `production_eligible=false`; no
candidate qualified. Run-summary Artifact `10528400988` has API digest
`sha256:8d07b6b6119bf71416742c33ef3d2c0eb4fe14c8c6ce329208daa281f8950bc9`,
and independent audit Artifact `10527862342` has API digest
`sha256:2d6c88ee811a795543e1d772d3b5f4c7162661da38a706a6ee7ce3324913587b`.
This corrected-accounting development evidence does not alter the earlier pre-fix
Issue #640 result and does not validate the later #657/#658/#659
reduce-only/profile semantics now present on `main`. It accessed no unused/final
data and authorizes neither production nor live trading. Existing profitable
individual seeds do not authorize selecting a model or changing its qualification
gate.

### Directional exit diagnostics

A separate unchanged constant-long diagnostic has now completed all 17544
development intervals on the corrected accepted-lot ledger. Protocol
`666c864868f09dce66e3c1ae87dea934756ed00380177135aaeda08404ccee22`
binds source/runtime and an observer-only raw order/account trace. Net return
was +14.02045%, with ledger maximum drawdown 11.78340%, but exact final holdings
were BTC 0, ETH 0.004, BNB 0, XRP 0.1 and ADA 1. All three nonzero final exits
were rejected under the unchanged minimum-notional contract. This diagnostic
does not qualify the control or revise the original study.

An independent Fraction/Decimal audit reconciled all intervals, 193 fills and
2192 funding boundaries against the frozen dataset arrays, including exact
inventories, cash, costs, capacity and ledger drawdown. It verified source/helper
pins and the genuine final residuals. It did not reconstruct original exchange
archives, every admission decision, or tick-level funding eligibility. The result
SHA-256 is
`a9aaa8c028edaf808752160ef34f509867b6ee6d6b83666b16c2e50e345670d5`.

Its 2024 return was only +0.02473%: material positions had already been reduced
to residual amounts during 2023. Source inspection and a fixed-price projection
example show that feeding each constrained target into the next proposal applies
the historical drawdown scale repeatedly. At a fixed 15% drawdown, a 50% gross
proposal becomes 25%, 12.5%, 6.25%, and so on. This risk-contract investigation
is separate from the explicit reduce-only execution design in
`docs/specs/reduce-only-exits.md`. No risk change or minimum-notional exception
has yet been applied to a new economic comparison.

The common simulator provides explicit MARKET reduce-only orders, exact fill-time
inventory bounds and persistence/event evidence. An opt-in source-bound USD-M
one-way MARKET profile enables same-side reduction reconciliation and its
per-order venue-notional exception, retaining runtime minima and source quantity
bounds. False/true profiles support a matched comparison; omission retains default
behavior. Current exchange filters on historical bars are a declared assumption,
not point-in-time source recovery. The shared-cash directional evaluator accepts
the profile explicitly, records actual execution-policy identity and exact terminal
inventory, and now has observer-only interval ledger evidence for independent
reconciliation without changing replay economics. These are software contracts,
not economic improvement evidence.

Issue #661 preregistered the matched Stage C economic comparison but could not
establish its required current-rule source authority. Official source run
`35337609697` on Ubuntu and recovery run `35343362615` on macOS both stopped
at the first Binance USD-M `exchangeInfo` request with HTTP 451 and published no
source artifact. Two separately frozen transport recoveries also produced no valid
`exchangeInfo` bytes: a single direct GET failed before validation, and a single
live raw-body fetch returned Binance's HTTP-451 restricted-location error JSON.
No successful response was substituted, no alternate host/mirror was shopped, and
no model fit or ordinary/reduce-only economic replay occurred. The terminal Stage C
status is therefore `SOURCE_AUTHORITY_UNAVAILABLE`: the reduce-only economic
hypothesis is **untested, not rejected**. No unused/final data was accessed and
nothing from this stopped lineage authorizes production or live trading.

On 2026-09-17 the user selected 20% as a research drawdown tolerance. A separate
directional study uses the frozen successor Dataset from run 34803217815,
artifact 10331899302, fits before 2023, and screens 2023-2024 only. Its immutable
protocol is assembled by `directional_study.expected_protocol` and is retained
with the raw study evidence; the maintained contract is described here.
This study uses 10000 USDT simulated capital and one shared account; it does
not replace the earlier independent-symbol Study or reopen sealed experiments.
It compares the five maintained families, a fixed 20/10-day channel breakout,
and three controls. Sequential PPO uses five seeds and 262144 steps per seed;
the opt-in interleaved capability is outside this experiment's treatment.

Qualification requires positive full and both-year returns, ledger drawdown
at most 20%, complete replay, no termination, and actual terminal flatness.
Double-cost and extra-bar-delay stresses plus per-symbol diagnostics precede
prospective paper eligibility. PPO needs four of five seeds passing both base
and stress gates; medians always include all five. There is no profitable
candidate claim before the write-once selection artifact completes, and even
a qualified development candidate requires prospective paper evidence.

The channel entry uses the prior 480 hourly bars and exit the prior 240 bars,
excluding the decision bar from both extrema. Qualification ranks full net
return, then turnover, then fixed complexity order (trend, mean reversion,
channel breakout, ridge24, lightgbm24, PPO); controls cannot win. Risk reduction
starts at 10% historical maximum drawdown and requests flat at 20%, but price
gaps can exceed the bound and therefore fail the gate. A terminal next-open
mark/close at 2025-01-01 00:00 is the endpoint of the last 2024 interval; no later
2025 or 2026 evaluation is performed by these studies.

The write-once CLI entrypoints are
`python -m trade_rl.evaluation.directional_study` and
`python -m trade_rl.evaluation.ppo_risk_study`. Both expose `prepare`, `run`,
and `finalize` with required `--source` and `--output`; `run` additionally takes
one frozen `--arm`. The risk-only CLI also requires `--baseline` pointing to the
completed original study with its exact source snapshot and model/result hashes.
It accepts only that registered baseline, not an arbitrary profitable rerun.
An output directory or arm cannot be overwritten. Production source/runtime
must remain byte-identical between prepare and final publication; raw source
snapshots permit later independent inspection after main advances.

一つの銘柄ID非依存strategy/model/policyを学習・凍結し、各銘柄へ独立に適用する。その結果がpoint-in-time data、同一execution/accounting、hard risk、明示的なexecution-cost assumptionsの下でcontrolsを超え、unused dataでも再現するかを検証する。

Aggregate P&Lだけで成功を判定せず、各symbolの結果とraw interval returnsを保持する。

## 初期比較: 5 candidates + 3 controls

| 名前 | 系統 | 現行役割 |
|---|---|---|
| `trend` | rule | 単一signalのtrend + hysteresis |
| `mean_reversion` | rule | 単一signalのmean reversion + hysteresis |
| `ridge24` | forecast | universal Ridge 24h forecast + 共通controller |
| `lightgbm24` | forecast | universal shallow LightGBM 24h forecast + 共通controller |
| `ppo` | RL | universal teacher-free PPO |
| `cash` | control | 常時FLAT |
| `constant_long` | control | 常時LONG |
| `constant_short` | control | 常時SHORT |

初回比較へTransformer、SAC、TD3、TQC、複数horizon ensemble、大規模hyperparameter gridを追加しない。

判断原則:

- ruleが同等以上ならruleを優先する。
- forecastが明確に優位でPPOが上乗せしないならforecastを優先する。
- PPOはunused dataでも単純候補へ安定して上乗せする場合だけ残す。
- 差が不明ならより単純な候補を残す。
- 全候補が弱ければ **no winner** を正しい結論とする。
- controlsはbenchmarkであり、Study winnerとは呼ばない。controlが最良ならno winnerである。

## Universal fit contract

Universal model/policyにsymbol ID、symbol-specific embedding、symbol-specific coefficientを初期状態では入れない。

共通条件:

- 同じfeature schemaを使う。
- `fit_symbol_names` で事前登録した銘柄だけをfitへ使う。
- fit scope外の銘柄をtraining row/episodeへ混ぜない。
- content-verified Datasetでfit scopeを全symbolより狭める場合は、selected featureの情報依存もfit scope内へ閉じる。cross-sectional rank / dispersionのようなuniverse-dependent featureはsubset fitでrejectし、reference-relative / correlation / betaはreference symbolがfit scope内にある場合だけ許す。FeatureKind provenanceを復元できないverified subset fitはfail closedとし、identity provenanceのないlegacy/synthetic経路だけをunseen-symbol isolationの証拠には使わない。
- 同じfit cutoffを使う。
- 同じfrozen strategy/model/policyを評価対象の各銘柄へ適用する。
- 評価symbolごとにfresh strategy/controller wrapperを生成し、学習済みmodel/policy weightだけを共有する。前symbolのwrapper内部状態を次symbolへ持ち越さない。
- 評価は各銘柄を独立portfolioとしてReplayする。

### Ridge / LightGBM

Eligible row数の多い銘柄がtrainingを支配しないよう、各fit symbolの総sample weightを等しくする。Ridgeはweighted statistics/normal equationを使い、LightGBMは同じweightを`sample_weight`へ渡す。

### PPO

初回real-data M2のPPO Observation v2は、selected local feature values、availability / finite mask、normalized local staleness、current intent、current weightだけをこの順序で使う。symbol IDは含めない。Training episodeはfit symbolをround-robinし、各episodeは一つのactive symbolだけを扱う。

現行datasetのglobal regimeは全dataset symbolから集計されるため、fit-symbol subset外の情報がtrainingへ混入しないよう初回M2のpolicy inputから除外した。Observation v2は空のglobal rosterをsemantic identityへ明示bindする。global contextは、fit-scope-safeなreference universeを事前固定できる場合にだけ別Controlled Factorとして検証する。

PPO fitの既定layoutは既存互換の`sequential`である。複数fit symbolを宣言するsequential fitでは、SB3の2048-step rollout丸め後の実効budgetが全fit symbolへ最低1 full agent episodeずつ届くことをfit前に要求し、後半symbolが0 transitionになる設定をfail closedにする。これは最低coverage保証であり、完全なsample均等化や性能改善を意味しない。実装上はopt-inの`interleaved`も選べ、fit symbolごとのfixed-symbol `PPOTradingEnv`を`DummyVecEnv`へ束ね、明示した`rollout_steps_per_env`ごとに全envからrolloutを集める。 `PPOTradingEnv`は実際にepisodeで売買するactive symbol scopeと、selected featureが参照してよいinformation symbol scopeを分離する。direct constructionではinformation scopeを省略するとactive scopeと同一として検証し、fitter経由ではsequential/interleavedとも全fit symbol rosterをinformation scopeとして明示する。したがってinterleavedの各slotが1銘柄activeでも、fit内reference-relative featureを誤ってrejectせず、fit外symbol依存は#707の共有validatorで拒否する。これは学習sample schedulingだけを変えるunevaluated capabilityであり、Observation v2、reward、hard risk、network、entropy係数を変更しない。Directional PPOではfitとdevelopment replayが同じbase execution economicsを共有し、Dataset由来のborrowを両方で課す。さらにcurrent directional fitはfinite-horizon末尾を無料resetにせず、latencyを考慮して最後のagent decision後にcanonical FLAT settlement区間を予約する。settlementはagent actionではなくenvironment terminal transitionとしてrisk/executionを通し、内部settlement barへ追加discountを掛けず、その実現log wealth changeをterminal rewardへ加算する。capacity等で残余が残ればflatと偽装しない。これはper-symbol training endpointの補正であり、shared-cash evaluationとのcross-symbol accounting差は別途残る。旧interleaved prereg/evaluatorはこのborrow修正前のimplementation authorityへbindされているため、current economicsでの実行authorityとしてはobsoleteであり、結果を見ずにfresh protocolを作り直す必要がある。`expected_ppo_realized_timesteps`はlayoutごとのrollout丸めを定義し、fit後の`model.num_timesteps`とcandidate artifactのrequested/realized transition evidenceを照合する。Controlled Factor `PPO_TRAINING_LAYOUT`はlayoutとrollout長を同時に変更し、paired comparisonでは両条件のrealized transition数も同じにする。結果盲検のCPU synthetic fit（2銘柄、各513 bars、8 features、2,048 requested transitions、seed 11、warm-up後3回交互測定）では、sequential中央値2.9377秒（2,048 realized transitions）に対しinterleaved/512中央値2.5308秒（同2,048）で約13.9%短かった。layout間でpolicy hashも変わるため、これは速度と利益品質のどちらの実データ証拠でもない。新factorのeconomic comparisonは未実行である。また、vector envのreset seed差がexecution randomnessへ混入しないよう、interleavedは`slippage_std > 0`を拒否する。

PPOのconstructor/policy constructionについて、current implementationが実際に依存する主要defaultはsourceへ明示bindする。対象はlearning rate、rollout長、batch、epoch、discount/GAE、clip、advantage normalization、entropy/value係数、gradient clip、gSDE/target-KL、およびMlpPolicyのTanh・orthogonal init・FlattenExtractor・shared extractor・Adam epsである。これは値を変更する探索ではなく、pinned runtimeで既に有効だった値をsource contractへ昇格する変更である。一方、SB3/PyTorch内部algorithm implementationまでrepositoryへ複製したわけではないため、library versionとruntime provenanceは引き続きtraining implementation identityの一部であり、dependency変更時のsemantic equivalenceを自動仮定しない。

## Causality and evaluation rules

- `feature_available_time <= decision_time` を守る。
- supervised labelは `label_end_time < fit_cutoff` で完結し、始点と終点の価格行がともに観測済みで、両価格の `available_at < fit_cutoff` を満たす場合だけ採用する。欠損barのforward-fill価格とcutoff時点で未公開の価格を学習labelへ使わない。
- future由来のscaler/normalization/imputation/feature selectionを禁止する。
- fit symbol subsetを使うtrainingでは、そのsubsetを情報scopeとして扱う。content-verified Datasetのselected cross-asset featureがholdout symbol universeへ依存する場合はfit前にrejectし、reference-dependent featureはreference symbolがfit scope内にある場合だけ許す。verified transformで元build configを追跡できないstrict subsetもfail closedにする。
- development/final期間をfitやthreshold調整へ戻さない。
- 同calendar shockを受ける複数銘柄を完全独立標本とみなさない。
- 同じfrozen strategyを各symbolへ独立Replayし、`UniversalStrategyComparison.by_symbol`を主要結果として扱う。

各symbol × strategyでは少なくともtotal return、Sharpe、Sortino、maximum drawdown、turnover、total execution cost、funding P&L、borrow cost、trade/rebalance/termination diagnostics、raw interval returnsを保持する。

## Execution economics contract

Execution economicsはfeature configurationではなくDataset environment semanticsである。

- build-level authorityは`trade_rl.data.build.ExecutionEconomicsProfile`が持つ。
- `MarketBuildConfig`はfeature/build authorityのまま維持する。
- profile省略時はlegacy build behavior/content identityを維持する。
- 明示profileは既存economic-semantics経路を通り、immutable Dataset economic arrays/content identityへbindする。
- Canonical M2 bootstrap v1のreader/payload/digest互換は維持する。
- Canonical M2 bootstrap v2は明示的なexecution economicsを必須とし、zero economicsへのsilent fallbackを禁止する。
- runtimeは`zero_overlay_dataset_fields_authoritative`を維持し、Dataset economicsへ第二のcost overlayを重ねない。

初回real-data M2で採用したeconomicsは再現可能なresearch assumptionであり、historical/account-specific venue truthではない。static spread、zero borrow cost、Dataset-authoritative market-impact/slippage不在は残存realism limitationである。

## M1 / M2 / M3

### M1 — Lean core: complete

実装済み:

- causal/point-in-time `MarketDataset`
- deterministic filesystem dataset artifact
- canonical `MarketExecutor + BookState` accounting
- hard-risk projection
- quantity-preserving independent symbol replay
- DB/UI/teacher pipelineなしで成立するcore CI

### M2 — Canonical real-data baseline and Controlled Experiment 0001 verified

実装・検証済み:

- 5 candidates + 3 controls
- universal Ridge / LightGBM / teacher-free PPO fit
- symbol-balanced supervised fit
- fit-symbol scopeの明示
- symbol-ID-free PPO
- fit-scope-safe PPO Observation v2（local values + availability/finite mask + normalized staleness + portfolio state、global policy rosterは空）
- `lean_candidate_result_v3`によるObservation / requested-vs-realized PPO transition bindingとhistorical v1/v2 reader互換
- `resolved_run_config_v3`によるObservation / PPO layout Study identity binding、historical v1/v2 read互換、legacy schema mutation拒否
- 全symbol独立comparison
- shared candidate config resolution
- in-memory candidate execution seam
- implementation/runtime/research-context provenance生成
- immutable 3-file candidate-run artifact
- semantic candidate-artifact identityとraw file integrity evidence
- append-only Study/Experiment state machineとprocess-safe mutation lock
- Study-owned multi-seed EvidenceSetとdeterministic seed invariance
- one-factor resolved delta verificationとunaffected-strategy raw-return invariance
- paired/bootstrap/seed analysis、ACCEPT-only lineage、FAILED/INVALID terminal、WINNER/NO_WINNER freeze
- strict `CanonicalM2BootstrapConfig` とsingle `ppo_seeds` authority
- exact Binance exchange-info / Vision plan / raw archive rosterのfreeze
- source同期後のcache-only network cut
- canonical dataset artifact + immutable StudyPlanのwhole-root atomic publication
- bootstrap manifestによるsource / dataset / Study / provenance identity binding
- `inspect_canonical_m2_bootstrap` によるnetwork-free再検証
- build-level execution economicsのDataset identity bindingとbootstrap v2 closure
- `market_build_v3` / `portable_feature_numerics_v1` によるCPU-portableなidentity-bound feature numerics
- persisted `market_build_v2` artifactのhistorical reader互換
- heterogeneous AMD / Intel hosted runnerでのfull priced Dataset byte identity一致
- portable Dataset / StudyPlanの結果前plan-only preregistrationと独立再構築
- real-cost-assumption portable Datasetを使ったbaseline-only Studyの実行
- baseline Artifactを別runnerで再取得し、Dataset / StudyPlan / EvidenceSet / raw Candidate Runsをpublication indexに依存せず独立再構築・再計算するpost-Artifact verification

現在のcanonical portable baselineで確認したこと:

- build semanticsは `market_build_v3` / `portable_feature_numerics_v1` で、hash-only roundingやtolerance弱体化は使わない。
- same sealed sourceからのfull priced Dataset identityはheterogeneous AMD / Intel hosted runner間でbyte-identicalに再現した。
- 結果を見る前のplan-only preregistrationはrun `34702660287`、Artifact ID `10300479733`、outer digest `sha256:60127cc2f24c7e8b3dcb5c6157ca49a5dd2f5b60c44f1020e4d540405e34e1f4` としてbaseline結果より先に封印した。
- portable Dataset IDは `d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f`、Dataset artifact digestは `77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8`、Study digestは `3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79` である。
- portable baselineはrun `34700123151`、Artifact ID `10301701698`、outer digest `sha256:f814fe4e205f8714c4344238911aae16e89ce0279908265feb1fdc85069b2a0a`、EvidenceSet fingerprint `526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29` である。
- preregistrationとbaselineは同じDataset artifactと**完全に同一のStudyPlan**へbindされ、preregistrationはplan-only、baselineはbaseline-onlyのままでExperiment countは0、未freezeである。
- preregisteredな5 PPO seedsと5 symbols × 8 strategiesの完全なbaseline evidenceが存在する。
- tradeが発生した175 observationsすべてで`total_cost > 0`、cash 25 observationsはzero-trade / zero-cost / zero-return、aggregate realized trading costは正である。
- fresh post-Artifact verifier run `34704606059` はsealed source、portable preregistration、baseline Artifactを再取得し、Dataset / StudyPlanを独立再構築したうえでraw Candidate Runsからreturn・cost・seed invarianceを再検証した。verifier Artifact IDは `10300932825`、outer digestは `sha256:77c26e8cdeec023a750582ec5aebe3870ca84728e580c9e27d1b2e420368a9` である。
- Portable Controlled Experiment 0001 (#511) は結果前preregistrationを封印・fresh verificationした唯一のcandidate EvidenceSetを再実行せずに完遂し、fresh runnerで公開result Artifactを再取得して独立再検証した。formal decisionは`KEEP_BASELINE`。mean-reversionのfactor effectは5 / 5 symbolsで正、median excess total returnは`+0.16996869069426646`、candidate positive-total-return symbolsは1 / 5、candidate median turnoverは`467.48617120292243`（baseline `858.3114067468092`）だった。unaffected raw-return equality 150 checks、deterministic metric seed invariance 1120 checks、tradeあり175 / 175 positive-cost、cash 25 zero-trade / zero-cost / zero-returnもGreenである。
- publication indexは独立再構築後のcross-checkにだけ使い、結果のoracleにはしていない。
- `research/m2-canonical-study-004` はpre-portable `market_build_v2` numericsで生成されたimmutable historical evidenceとしてhead/treeを維持するが、current canonical inputとしてはportable successorにsupersedeされた。旧Studyを書き換えたり削除したりしない。
- pre-portable Study 004 Experiment 0001 (#498) はfail-closedし、result Artifactも結果解釈も存在しないhistorical lineとして保持する。portable lineageのExperiment 0001は結果前preregistrationからfresh result re-verificationまで完了し、`KEEP_BASELINE`をdevelopment decisionとして固定した。
- PPO cross-seed candidate-metric aggregation defect (#476) はcurrent mainで修正済みだが、Issue #511は修正前にfreeze/startしたimplementation provenanceを維持する。PPO aggregate candidate metricsはformal decisionのoracleに使わず、raw-return equality controlとしてのみ扱う。
- baseline成立はresearch environment / identity / evidence pathの検証であり、profitability、winner、Production readinessを意味しない。

未完了:

1. Experiment 0001の`KEEP_BASELINE`を維持し、次のControlled Factorを結果を見る前にpreregisterする。
2. 次のExperimentでもone-factor delta、unaffected-strategy raw-return invariance、deterministic metric seed invariance、cost/cash semantics、fresh post-Artifact verificationを必須にする。
3. development Experimentを事前登録して継続し、最終的にwinnerをfreezeするかno-winnerと判断する。
4. winner候補が成立した場合だけsealed unused-future / final-testへ進み、Production/live tradingとは引き続き分離する。

**Portable Canonical real-data baselineは結果前preregistrationからpost-Artifact独立検証まで完了し、Portable Controlled Experiment 0001もfresh result re-verificationまで完了してformal decisionはKEEP_BASELINEである。** このExperimentは相対改善とturnover低下を示したが、candidate profitabilityやwinnerを成立させなかった。Baseline成立やKEEP_BASELINE decisionはProduction readinessの証拠ではない。

### M3 — Finalize and delete: not started

M2で候補をfreezeした後だけ進む。

1. 未使用future / zero-shot評価を一度だけ開く。
2. pre-registered stressを実行する。
3. 支持されなかったstrategy familyと専用test/extra/dead adapterを削除する。
4. README/config/CI/testsを採用構成へさらに縮約する。
5. Production認可は研究結果とは別に扱う。

## Canonical M2 bootstrap

Canonical M2 bootstrapはresearch runそのものではなく、real development Studyの入力を固定するpreparation stepである。

入力JSONは少なくとも次を事前登録する。

- Binance USD-M market
- ordered symbol roster
- base timeframe / feature timeframes
- exact data start / exclusive stop
- baseline signal/features/fit symbols/fit cutoff/development window
- rule / forecast thresholds
- PPO training budget
- ordered `ppo_seeds`
- gross budget / initial capital
- allowed controlled factors / experiment budget
- bootstrap count / seed
- bootstrap v2では明示的なexecution economics profile
- 一般のfinal-eligibleな新規Studyを作るbootstrap v4では、unused `final_evaluation_start` / `final_evaluation_stop_exclusive` と `StudyResearchContext` を固定する。PPO保有期間protocolのbootstrap v5は、これらに加えてprotocolとObservation / minimum-hold / terminal-settlement / riskを含むbaselineを固定する。

PPO保有期間のbootstrap v5ではDatasetの最終timestampが`data_stop_exclusive`と一致するため、`final_evaluation_start`は`data_stop_exclusive`より厳密に後でなければならない。v5 config readerがsource取得前にこの境界を拒否する。bootstrap v1-v4 configの境界semanticsは維持する。

baseline JSONに`ppo_seed`は持たず、`ppo_seeds[0]`だけがbaseline seed authorityである。

実行入口:

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.experiments.bootstrap.cli \
  --config <canonical-bootstrap-config.json> \
  --output <new-bootstrap-dir>
```

このCLIは `bootstrap_canonical_m2_study` を呼ぶ薄いfilesystem adapterである。`--output`は存在していてはならない。

成功したbootstrapは概ね次を持つ。

```text
<new-bootstrap-dir>/
  bootstrap.json
  bootstrap-manifest.json
  source/
    exchange-info/{exchange-info.raw.json,manifest.json}
    vision-plan.json
    vision-cache/**
  dataset/{manifest.json,arrays.npz}
  study/{plan.json,.mutation.lock}
```

source同期後のdataset buildはcache-onlyで、missing cacheをnetwork fallbackで補わない。whole rootはstaging内で完成・検証してから最後にrenameする。成功直後のStudyはPlanだけで、baselineはまだ実行されていない。

Inspection:

```python
from trade_rl.evaluation.experiments import inspect_canonical_m2_bootstrap

result = inspect_canonical_m2_bootstrap("<new-bootstrap-dir>")
```

Inspectionは保存されたconfig、source roster、dataset artifact、StudyPlan、bootstrap manifestをnetwork-freeで相互検証する。bootstrap v2では、configured economics、生成/reloadしたDataset economic arrays、identity-bound profileの一致もfail-closedで確認する。

## Development Run Core

必要な入力は次の3つである。

1. canonical filesystem dataset artifact
2. JSON run config
3. 存在していない新しいoutput directory

現行runnerが受理するconfig keyは次である。

```json
{
  "signal_name": "<rule signal feature name>",
  "feature_names": ["<feature A>", "<feature B>"],
  "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
  "fit_cutoff": "2026-01-01T00:00:00",
  "evaluation_start": "2026-01-01T00:00:00",
  "evaluation_stop_exclusive": "2026-02-01T00:00:00",
  "rule_entry_threshold": 0.10,
  "rule_exit_threshold": 0.02,
  "forecast_entry_threshold": 0.01,
  "forecast_exit_threshold": 0.002,
  "ppo_total_timesteps": 100000,
  "ppo_seed": 0,
  "gross_budget": 0.5,
  "initial_capital": 100000.0
}
```

`evaluation_start` と `evaluation_stop_exclusive` はdataset timestampへexact matchする必要がある。Evaluation startはfit cutoffより前にできない。

Standalone実行:

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.runs.candidate \
  --dataset <dataset-artifact-dir> \
  --config <run-config.json> \
  --output <new-result-dir>
```

Controlled StudyでRunを事前登録contextへbindする場合は、higher-level workflowから同じRun CoreへSHA-256 `research_context_digest`を渡す。CLIにも `--research-context-digest <sha256>` がある。Standalone Runでは省略できる。

既存output directoryへの上書きは拒否する。実行前後でimplementation/runtime provenance digestが変化した場合もpublishしない。

出力:

```text
<new-result-dir>/
  summary.json
  returns.npz
  provenance.json
```

- `summary.json`: dataset artifact identity、resolved run config/scope、各symbol × strategy metrics/diagnostics。
- `returns.npz`: paired comparison/block-bootstrap等に使うraw interval-return series。`allow_pickle=False`で検証可能なnumeric 1D arraysだけを正当なevidenceとする。
- `provenance.json`: exact Python source-byte manifest、runtime/dependency roster、optional research-context digest。

3ファイルを一つのimmutable Run evidenceとして扱う。Semantic artifact identityはNPZ ZIP compressionの違いでは変えず、検証済みarray content、summary、provenanceへbindする。raw file SHA-256/sizeはtamper検出用evidenceとして別に保持できる。

## Development後の判断順序

結果を見たら、architecture/model sizeを増やす前に次を確認する。

1. controlsより本当に上か。
2. 全symbolでreturn符号、drawdown、costが許容可能か。
3. 特定一銘柄だけが利益を作っていないか。
4. turnover/costでgross edgeが消えていないか。
5. ruleとforecastの差は十分か。
6. PPOが単純候補へ本当に上乗せしているか。
7. raw returnsが特定time block/regimeだけに依存していないか。
8. 差が弱ければ単純側を残す。
9. 全て弱ければno-winnerとする。

No-winnerの後に最初に疑う順序は、model sizeではなく **information source → feature causality/quality → horizon → regime hypothesis → period → model complexity** とする。

PPO controlled comparisonのcross-symbol candidate metricsは、各symbol内でfrozen seedを先に集約してからsymbol間summaryへ進む明示契約へ更新した。total return / turnover / total costはseed中央値、maximum drawdownはseed内worstを使い、factor effectは従来どおりseedごとのpaired excessの中央値をsymbol代表値とする。新規factor-effectは`controlled_evidence_comparison_v2`、persisted v1はhistorical first-seed semanticsでinspection互換を維持する。

## Final unused-data / stress protocol

Developmentで繰り返し見た期間をfinal testと呼ばない。Candidate、feature、threshold、seed policy、cost/riskをfreezeした後、未使用futureを一度だけ開く。

採用候補には事前固定したstressを適用する。

- fee adverse
- spread adverse
- impact adverse
- +1 decision latency
- capacity reduction
- initial capital sensitivity
- funding / borrow coverage check

Stress結果を見てから合格thresholdを変更しない。

Controlled Experiment Loop自体からsealed unused-futureを開かない。Development StudyをWINNER/NO_WINNERへfreezeした後、別subsystemでのみfinal authorizationを扱う。

research-governance側では `RESEARCH-001` の機械可読化として、`StudyResearchContext` / `ConsumedEvidence` と `controlled_study_plan_v3` を追加した。新規Studyは、仮説・observation/model/hyperparameter/evaluation design/result interpretationへ使った既知development evidenceのdigest、canonical time scope、利用目的とparent context digestをStudy identityへbindできる。新規final-eligible research lineは `canonical_m2_bootstrap_config_v4` でfinal windowとresearch contextをresult前に固定し、final startがdevelopment Datasetまたは申告済みconsumed-evidence scopeの終了以前にある場合はfail closedにする。historical `canonical_m2_bootstrap_config_v3` / `controlled_study_plan_v2` を含む既存artifactは当時の意味を維持し、contextを後付けして再分類しない。この機構は申告済みconsumptionを固定するもので、研究者やAIが閲覧した全情報の完全な申告を自動証明するものではない。

現行codeには、final境界として `trade_rl.evaluation.final_test` の**authorization capabilityだけ**がある。frozen `WINNER` のStudyPlan/StudyFreeze/winner evidence/winner strategyと未使用windowをcanonical one-shot artifactへbindするが、final Datasetを読まず、P&L/stressを実行しない。したがってM3 final economic evaluation自体は未実行であり、authorization capabilityやresearch-context infrastructureのGreenをfinal evidenceとして数えない。

## Superseded evidenceの扱い

旧zero-cost canonical Studyは削除・再解釈せず、diagnostic evidenceとして保持する。新しいreal-cost Dataset / Study / EvidenceSetは別identityであり、旧Studyをin-place mutationしていない。

PPO Observation v2確定前に生成されたeconomics-only baselineもdiagnostic evidenceとしてのみ扱う。Current canonical M2のStudy semantic identityを満たさないため、Controlled Experimentのbaselineとして採用しない。

## 現在の次アクション

現時点の次アクションは、Run Coreやbootstrap toolingをさらに拡張することではない。

> 現在のcanonical baselineをimmutable inputとして、一つのControlled Factorを結果を見る前に事前登録し、最初のControlled Experimentを実行・独立検証する。

旧teacher-selection runのrejectは旧mandatory teacher経路を再採用する根拠でも、現候補のprofitabilityを示す証拠でもない。現在の候補は現在のlean contract上で改めて評価する。

## Issue #810: opt-in after-cost allocation foundation

The software now includes an independent-account scalar allocator and an actual
execution consumer through `evaluation.allocation`. Expected-simple return,
horizon variance, asymmetric transaction costs, future exit/funding/borrow/cash
estimates and a variance preference determine a bounded surrogate target.
Actual BookState and pending orders bind each proposal; final risk and canonical
execution produce a separate approved/filled trace. Exact quantity HOLD cancels
pending MARKET remainders while preserving canonical carry/split processing.

This is an opt-in software foundation with synthetic decision, independent
numerical and cash/quantity oracles. The separate direct-simple connection below
adds synthetic model fits; no expected-simple market model has been fitted,
costs calibrated, economic comparison run or improved profit established.
The current-close signal is a declared proxy for next-processing-bar execution;
risk and venue rules can alter the scalar optimum. A mean-log prediction is not
an expected-simple prediction. Existing replay/PPO/artifact semantics and the
20% drawdown research guardrail remain unchanged; gaps or missed fills can still
exceed that guardrail. No sealed run is reopened.

Issue #810 remains open. Outstanding work includes empirical expected-simple
calibration, common nonRL/residual/direct RL context and execution,
objective/financial-clock integration, continuous-account walk-forward,
calibrated cost/capacity stress, a finite preregistered Study and independently
authorized economic evidence. This slice supplies neither a new WINNER nor
unused-data, paper/live-order or deployment eligibility. Final exact-head CI
and formal independent research approval remain integration requirements.

## Net-profit redesign groundwork (Issue #810)

新規の非LLM trader再設計は、事業目的・資本・金融時計の宣言から開始する。`evaluation/objectives` は期待終端純利益を目的名にし、canonical after-cost ledgerの実現endpointを固定初期資本と符号付き入出金で評価する。複数独立口座と共有口座1つの資本分母、利益/log目的の順位差、負の終端equity、regular clockとdiscountの意味をsynthetic contract testsで検査する。研究DD基準は最大20%を維持する。

`BoundObjectiveClock` により、新規bindingではUTC評価期間とeconomic horizonの一致を整数秒で要求し、個別宣言のdigestを同時に固定する。小数秒を丸めた期間や不一致の時計は拒否する。profile内容やruntimeの検証へは接続されていない。

これはP0の初期software capabilityであり、P0全体完了や経済仮説の成立ではない。現在のPPO log報酬・独立銘柄訓練・Run/Study reader・歴史的選定基準は維持する。PR #794のtraining discount objectiveとこのbusiness objectiveは別の責務で、gamma/GAEの既定値を重複変更しない。PR #790のshared-cash replayもjoint learning済みとは扱わない。

次段階は参照したrisk/economics/運用recipeの検証と、既存executorを使ったtraining/replay parity、上流のprequential packet、金融clock/終端のadapter整合、合成市場での学習可能性の検査である。これらの不足を閉じ、fresh result-blind G0-G2 reviewと正式な有限予算研究契約を結果前に固定するまで、新規G4実行へ進まない。実データfit/replay、未使用future開封、封印run再試行、live発注はこの変更では実施しない。利益性、WINNER/NO_WINNER/INVALIDの経済判断はまだ未確立である。

## Issue #810: direct-simple forecast connection

A separate versioned producer derives direct close-ratio labels from the existing
selector's exact mature rows and freezes next-block Ridge projections. Packets
bind selected snapshot, decision price, availability, horizon and uncalibrated
pooled marginal label variance. The consumer connects these to actual BookState
allocation, separately declared horizon costs, hard risk and canonical execution.
Current/forecast close valuations must match; stale/delayed packets and changed
proposal inputs fail.

The new consumer requires a declared Dataset/index account clock. Canonical
execution preserves and advances it, rejecting an old index that would repeat
dividend/carry despite unchanged marks. Continuation carries book, order book and
next index together. Synthetic tests cover that failure and equivalent valid
timestamp storage units. Bootstrap clocks remain caller declarations; this
bounded consistency check does not close Issue #810's full financial runtime or
continuous-account research requirements. Legacy unmarked books remain unchanged.

This is software tested on synthetic inputs, not measured market profit. Raw
close labels are price-return surrogates for next-open execution and exclude
held-quantity corporate-action wealth. Costs, carry, variance and conditional
expectations are uncalibrated. Future actions never filter present predictions.
Old log/PPO/Run/Study semantics and the 20% research DD guardrail are retained.

The candidate preserves exact P0/P2/allocation histories for review. Remaining
Issue #810 requirements include full order/account state for common RL consumers,
the full financial runtime and continuous-account walk-forward, cost/capacity
evidence, complete finite trial
registration and independently authorized economic comparison. No new WINNER,
G4/G5 result, final-data or live-order eligibility is established. Integration
still requires final-head full CI and formal independent research approval.

## Issue #810: common allocation PPO candidate

An unintegrated opt-in software candidate now connects bounded direct/residual
actions to the same final risk and canonical independent-account execution.
Fixed-capital after-cost reward and gamma1 are bound to actual regular clocks.
The first real SB3 signal protocol passed all three fixed seeds and complete
save/load execution-trace parity. Subsequent strict receipt/coverage repairs and
the separate fee-only protocol are verified as software tests, not market
selection. None of these synthetic schedules establish market profitability.

The candidate rejects stale recipe/capital/profile declarations and UTC overflow,
preserves signed terminal debt through a distinct canonical execution policy,
and records actual action counts plus policy/critic-bootstrap observation rows.
Calendar kind and effective bar duration are bound to deployment semantics.
Full order/account observations,
fit-only preprocessing, continuous account updates/walk-forward, calibration,
capacity sensitivity, complete market trial registration, formal independent
approval and authorized economic comparison remain open. Existing log/PPO/A2C,
Run/Study and sealed outcomes are unchanged. No new WINNER/NO_WINNER/INVALID
economic result, unused-future opening or live eligibility is established.

## Issue #810: allocation snapshot contract stage

現在の候補には、liveな独立口座のaccount/order factsを受け取るimmutable下位DTOを追加した。
宣言mappingのsoftware testでclock、active-order status、exact quantity、
deep immutabilityとdigest scopeを検証する。既存PPO v1はこのDTOを消費しない。
source factsの読み出しは下記observer、numeric projectionは下記pure encoderが担当する。
明示v2 runtime接続は下記候補に追加した。予約cash・position ageのモデルは未実装である。
continuous walk-forward、正式な経済比較は後続作業である。
実市場fit / replay、経済比較、formal independent approvalやprofitabilityを達成した状態ではない。

## Issue #810: live independent-account observer candidate

A separate opt-in software observer now freezes live canonical account facts and
active MARKET-order evidence. Selected-only vectors preserve the global symbol
slot; a source digest retains full account/history identity. It requires a known
matching clock, currently available selected market source, native order
validation and already-refreshed canonical margin, including bootstrap setup.
Impossible waiting clocks and zero-fill/nonzero-notional records are rejected;
completed active remainders are checked with native completion tolerance.
Active filled progress also respects the native absolute minimum fill quantity;
legal small fills on large requests remain observable.
Its cloned checks preserve source book/cache, orders and execution randomness.

Pure no-fit tests cover exact fractions, direction/clock/schema counterexamples,
detachment and unavailable-price exposure. This DTO is not a numeric RL
observation, terminal reader, holding-age/reservation model or provenance proof.
Opt-in policy-state connection is below; train-only preprocessing, sampling, continuous
walk-forward and complete trial registration remain separate work. No market
result, G4/G5 approval, unused-future opening or live eligibility is established;
integration still requires final-head full CI and formal independent approval.

## Issue #810: allocation observation v2 declaration candidate

The lower immutable schema now declares ordered `F+9+13+24K` field names and
binds feature names, bounded order slots, fixed capital and episode normalization
length. Software tests cover layout/identity, detachment and constructor failures.
The peer pure encoder supplies numeric generation; the opt-in runtime below
consumes it. Allocation v1 recipes/tensors and default consumers remain unchanged.
There is no new fit/replay, market result, research approval or live eligibility.

## Issue #810: pure allocation observation v2 encoder candidate

An opt-in lower encoder now checks matching snapshot/decision declarations and
returns detached bounded float32 account/order values. Signed exact quantities,
current/historical drawdown, causal clock masks and fixed capital retain explicit
meaning; ID-based execution priority, ages/reservations and source authenticity
are outside its scope. No-fit arithmetic/failure oracles and v1 behavior checks
cover this software stage. Opt-in train/load/input receipts are below; train-only
preprocessing, continuous walk-forward and economic comparison remain open.

## Issue #810: allocation observation v2 runtime candidate

The explicit v2 environment now reads admitted live account/order state, retains
the same native transitions/rewards and uses distinct strict recipes/bundles.
Software oracles verify default v1 compatibility, actual partial-order columns,
terminal sentinels, real SB3 masks/bootstrap, consumed actor/boundary inputs and
save/load trace parity. This is G0-G2 synthetic software evidence only.
Sources/receipts do not prove authenticity, full Markov state or profitability.
Explicit training settings/returned-call telemetry are covered below; fit-only
preprocessing and sampling, continuous-account handover/walk-forward, calibrated finite market
Study and formal G4/G5 approval remain open. Existing Run/Study/sealed outcomes
and deployment eligibility remain unchanged.

## Issue #810: explicit allocation training protocol declaration candidate

The candidate adds a pure immutable declaration with 19 required learning values
and closed fixed SB3/Torch categorical CPU construction identifiers. A supplied
positive Adam epsilon is always recorded; the literal initial fixture uses 1e-5.
No-fit literal/digest and rejection oracles validate the declaration, not training.
Explicit runtime construction and returned-call receipts are covered below;
v1/v2 recipes, receipts and defaults remain fixed. Durable raw-action evidence
and delayed-payoff learnability remain next stages. Preprocessing/sampling,
actual continuous-account handover, formal review and economic comparison remain open.
No new market fit/replay, profitability, G4/G5 approval or live eligibility is established.

## Issue #810: explicit allocation PPO protocol runtime candidate

Explicit protocol fits now bind the financial clock and known installed
PPO/policy/Adam configuration. Bundle/training v3 retains recipe_v2 and adds a
closed protocol/digest and bounded completed-Adam-call/entered-epoch receipt.
Fixed two/four-step synthetic oracles verify final capture, minibatch counts,
native KL no-step behavior, hook cleanup, strict pre-loader tampering and actual
post-load settings. None v1/v2 bytes/source/action behavior stays unchanged.
This is G0-G2 software evidence, not parameter-improvement or fit authenticity.
Raw-action evidence, delayed-payoff learning, fit-only preprocessing/sampling,
continuous-account handover/walk-forward, market calibration/Study and formal
G4/G5 approval remain open; existing sealed outcomes and eligibility are unchanged.
