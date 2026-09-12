# Current research status

更新基準: 2026-09-12 (JST)

## 結論

Canonical M2は、**real-cost-assumption Datasetとfit-scope-safe PPO Observation v2を同時に固定したStudy 004のbaselineまで実行・独立検証済み**である。

現在確定している研究上の事実は次のとおり。

- Study 004 baseline EvidenceSetはimmutable Artifactとしてpublish済みで、別runnerによるpost-Artifact再構築・raw evidence再計算も成功した。
- zero runtime overlayのままDataset-authoritativeなfee/spread/participation semanticsが実際のReplayへ入り、tradeが発生した175/175 observationsで`total_cost > 0`を観測した。
- superseded Study 001は変更されておらず、zero-trading-cost diagnostic evidenceとして保持する。
- Study 004は**baseline-only**であり、Controlled Experimentはまだ開始していない。
- profitability、winner、PPO/forecastの優位性、Production/live trading authorizationは**未確定**である。

Repository統合状態は研究Artifactとは別に扱う。Study 004はPR #484と#491を結合したfrozen integration commitで実行したが、`main`へのmergeはまだ行っていない。

## 研究目的

一つのsymbol-ID-independent strategy/model/policyを学習・凍結し、各symbolへ独立に適用する。その結果がpoint-in-time data、共通execution/accounting、hard risk、明示的なexecution-cost assumptionsの下で単純候補/controlsを超え、unused futureでも再現するかを検証する。

Aggregate P&Lだけで成功判定せず、各symbolの結果とraw interval returnsを保持する。

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

初回M2へTransformer、SAC、TD3、TQC、複数horizon ensemble、大規模hyperparameter gridを追加しない。差が弱ければ単純側を残し、全候補が弱ければ`no winner`を正しい結論とする。

## Universal fit / information boundary

共通契約:

- `fit_symbol_names`で事前登録したsymbolだけをfitへ使う。
- fit scope外のsymbolをtraining row/episodeへ混ぜない。
- 同じfit cutoffとfrozen strategy/model/policyを使う。
- 評価は各symbolを独立portfolioとしてReplayする。
- symbol ID、symbol-specific embedding、symbol-specific coefficientを初期contractへ入れない。
- future由来のnormalization/imputation/feature selectionを禁止する。

PPO Observation v2は、selected local feature values → availability/finite mask → normalized local feature staleness → current intent → current weightの順で構成する。初回M2ではfull-universe aggregateによるfit-scope leakageを避けるためdataset-global policy channelを持たず、空のglobal rosterをsemantic identityへbindする。

## Execution economics contract

Execution economicsはfeature configurationではなく**Dataset environment semantics**である。

- build-level authorityは`trade_rl.data.build.ExecutionEconomicsProfile`。
- `MarketBuildConfig`はfeature/build authorityのまま維持する。
- profile省略時はlegacy behavior/content identityを維持する。
- 明示profileは既存`build_market_economic_semantics()`経路を通り、immutable Dataset economic arrays/content identityへbindされる。
- canonical bootstrap v1 reader互換を維持し、v2は明示profile必須でzero economicsへのsilent fallbackを禁止する。
- runtime側は`zero_overlay_dataset_fields_authoritative`を維持し、Dataset semanticsへ第二のcost overlayを重ねない。

Study 004で使用した`canonical_m2_research_assumption_v1`は再現可能な**research assumption**であり、historical/account-specific Binance fee truthではない。現在もstatic spread、zero spot-style borrow cost、Dataset-authoritative market-impact/slippage不在というrealism limitationがある。

## M1 / M2 / M3

### M1 — Lean core: complete

causal/point-in-time Dataset、deterministic filesystem artifact、canonical execution/accounting、hard risk、independent-symbol replay、5 candidates + 3 controls、Candidate Run provenance/artifact、Controlled Experiment state machineは実装済み。

### M2 — Canonical baseline verified; controlled experiments not started

Study 004は、execution economics repair (#484) と PPO Observation v2 (#491) を結合したfrozen codeで実行した。

Frozen implementation:

- integration SHA: `135cbdaad1a40cd064b8b08492d591d385a2c89e`
- integration tree: `afe75e0d45fd401e5e6819c7b78a1d099666cc53`
- #484 parent: `558919861f1fa86956664405b8946ab923b2d863`
- #491 parent: `b3c20f01966c8f4e944255ee7df5cd4258684a0f`

Real-cost Dataset/bootstrap:

- source run: `34679031604`
- Artifact ID: `10293251579`
- Dataset ID: `13334380e5e52fae6b69688a4270de736958c384a7991234ea5635887b06adc0`
- Dataset artifact digest: `c1fb38f2805f452c423a648b7daf14ea1e910aae63c8bb5e24429323b14f02c2`
- raw-source roster digest: `0293e7575533bb3bd8876f5b130495491b2496144bff2ac9a5d0ba78553fe6d0`
- Vision-resolution digest: `ba0a247f5b32d82493e6ad67d106ccb60b1dedafb7948870179afbc904d15fcf`

Study 004:

- Study digest: `9138a5a99ad517440e3a95a1651387a0a66d2f5341d1cb0a695f921675ce6a78`
- schema: `resolved_run_config_v2`
- PPO schema: `ppo_observation_v2`
- PPO global roster: `[]`
- execution overlay: `zero_overlay_dataset_fields_authoritative`
- PPO seeds: `0,1,2,3,4`

Baseline Artifact:

- run: `34687675504`
- Artifact ID: `10296984864`
- Artifact digest: `sha256:377cf2ed5fc987d1bfd41bef108778e468843053d2c2be46e80a1436eceebf58`
- EvidenceSet fingerprint: `926d31bb7265319e1c52d543caef4c8a5f10bd7b4a31f0638a277c3d6a7e7a27`
- aggregate realized `total_cost`: `3878704.5834642216`
- trading observations: `175`
- positive-cost trading observations: `175`
- cash observations: `25`, all zero-trade / zero-cost / zero-return
- Experiment count: `0`
- frozen: `false`

Independent post-Artifact verifier:

- run: `34691921737`
- Artifact ID: `10297845358`
- Artifact digest: `sha256:5e8acbb712944b95860cbbcb1b1e876bccc59807202c6496cb9d3e0eacd9648b`
- raw runsからDataset/Study/EvidenceSet、raw-return compounding、execution diagnostics、cost、seed invariance、within-EvidenceSet analysisを独立再計算して一致を確認した。
- baseline indexはoracleとして使用していない。

このbaselineはreal-cost-assumption環境が実際に有効であることを示すが、候補の優位性やprofitabilityを示すものではない。

### M3 — Final unused-data / stress: not started

Developmentでcandidate/feature/threshold/seed policy/cost/riskをfreezeした後にだけ進む。未使用futureを一度だけ開き、事前固定したfee/spread/impact/latency/capacity/capital/funding等のstressを適用する。結果を見て合格thresholdを変更しない。

## Superseded Study 001

旧Canonical M2 Study 001は削除・再解釈しない。zero-trading-cost diagnostic Studyとして保持する。

- branch SHA: `aebc35cb6422726d19d8d8cf545729c25bb9ae0e`
- tree: `edb2c00f70a0235108406063ff4e53cac7353923`
- Dataset ID: `8a41ad3c26289836d756de76ce541280dbcaaf80f26e2d8ecab055d6d0e749ab`
- Study digest: `580a745c5013e742fa3339728204a8c0bd6e1e7552d39ffac0de7f855bb933f3`
- baseline fingerprint: `641653fac2e40a53f66f430beb21ec5a95036b1f3a5feeb1f78f5ed31c708432`

Study 004 verifierは旧branch SHA/treeと旧Artifact metadataも再確認しており、旧evidenceは変更されていない。

## Development後の判断原則

Controlled Experimentの結果を見たら、model sizeを増やす前に次を確認する。

1. controlsより本当に上か。
2. 各symbolでreturn符号、drawdown、costが許容可能か。
3. 特定symbolだけがaggregate edgeを作っていないか。
4. turnover/costでgross edgeが消えていないか。
5. rule/forecast/PPO間の差が実質的か。
6. raw returnsが特定time block/regimeだけへ依存していないか。
7. 差が弱ければ単純側を残す。
8. 全て弱ければ`no winner`とする。

Known issue #476では、controlled comparisonのPPO cross-symbol candidate metricsにfirst-seed-only aggregationが混在する点を追跡している。これはStudy 004 baselineのcost/evidence validityを無効化しないが、将来PPOのcontrolled factor effectを解釈する際は、#476を解消するか、明示的なmulti-seed独立集計をoracleとして用いる。

## 現在の次アクション

1. #481/#484のdurable-doc cleanupとfinal exact-head gateを完了する。
2. #489/#491を含むrepository統合状態を整理する。明示的な許可なしに`main`へmergeしない。
3. Study 004で最初のControlled Factorを**結果を見る前に**preregisterし、baseline Artifactをimmutable inputとして使う。
4. Experimentごとにfactor isolation、raw-return再計算、全strategy side effect、multi-seed contractを検証する。
5. Development StudyをWINNER/NO_WINNERへfreezeするまでsealed futureを開かない。

Study 004 baselineの成立は「研究環境が修復された」ことの証拠であり、「儲かるstrategyが見つかった」ことの証拠ではない。
