# Current research status

更新基準: 2026-09-12 (JST)

## 結論

Trade RLの現在地は、**lean core、5候補+3 controlsの共通比較基盤、provenance-bound candidate Run Core、Controlled Experiment Loop v1、Canonical M2 bootstrap toolingを実装し、real-cost-assumption Datasetとfit-scope-safe PPO Observation v2を固定したCanonical real-data baselineは実行・独立検証済み**の段階である。

一方、**Controlled Experimentはまだ開始していない**。したがって現在は次を主張しない。

- profitabilityは未証明。
- winnerは未選定。
- Production/live order routingは未認可。
- PPOやforecastがruleを上回るという結論はない。

次の研究上の本質的作業は、新しいmodel familyやbootstrap toolingを増やすことではない。現在のbaselineをimmutable inputとして、一つのControlled Factorを結果を見る前に事前登録し、最初のControlled Experimentを開始することである。

## 研究目的

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
- 同じfit cutoffを使う。
- 同じfrozen strategy/model/policyを評価対象の各銘柄へ適用する。
- 評価は各銘柄を独立portfolioとしてReplayする。

### Ridge / LightGBM

Eligible row数の多い銘柄がtrainingを支配しないよう、各fit symbolの総sample weightを等しくする。Ridgeはweighted statistics/normal equationを使い、LightGBMは同じweightを`sample_weight`へ渡す。

### PPO

初回real-data M2のPPO Observation v2は、selected local feature values、availability / finite mask、normalized local staleness、current intent、current weightだけをこの順序で使う。symbol IDは含めない。Training episodeはfit symbolをround-robinし、各episodeは一つのactive symbolだけを扱う。

現行datasetのglobal regimeは全dataset symbolから集計されるため、fit-symbol subset外の情報がtrainingへ混入しないよう初回M2のpolicy inputから除外した。Observation v2は空のglobal rosterをsemantic identityへ明示bindする。global contextは、fit-scope-safeなreference universeを事前固定できる場合にだけ別Controlled Factorとして検証する。

## Causality and evaluation rules

- `feature_available_time <= decision_time` を守る。
- supervised labelは `label_end_time < fit_cutoff` で完結する。
- future由来のscaler/normalization/imputation/feature selectionを禁止する。
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

### M2 — Canonical real-data baseline verified; controlled comparison not started

実装・検証済み:

- 5 candidates + 3 controls
- universal Ridge / LightGBM / teacher-free PPO fit
- symbol-balanced supervised fit
- fit-symbol scopeの明示
- symbol-ID-free PPO
- fit-scope-safe PPO Observation v2（local values + availability/finite mask + normalized staleness + portfolio state、global policy rosterは空）
- Observation contractのRun evidence bindingとhistorical reader互換
- resolved Study identityへのObservation contract bindingと旧Studyへの新semantic mutation拒否
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
- real-cost-assumption Datasetを使ったbaseline-only Studyの実行
- baseline Artifactを別runnerで再取得し、raw Candidate Runsからanalysisまで独立再計算するpost-Artifact verification

現在のcanonical baselineで確認したこと:

- preregisteredな5 PPO seedsが存在する。
- 5 symbols × 8 strategiesの完全なbaseline evidenceが存在する。
- tradeが発生した175 observationsすべてで`total_cost > 0`を観測した。
- cash 25 observationsはzero-trade / zero-cost / zero-returnである。
- aggregate realized trading costは正である。
- Dataset / Dataset Artifact / Study / EvidenceSetのidentityはsuperseded zero-cost lineageからすべて変化した。
- raw returns、summary、execution diagnostics、deterministic seed invariance、within-EvidenceSet analysisをpublished indexに依存せず独立再計算した。
- Studyはbaseline-onlyのままで、Experiment countは0、未freezeである。

未完了:

1. 最初のControlled Factorを結果を見る前に事前登録する。
2. baseline Artifactをimmutable inputとして最初のControlled Experimentを実行する。
3. factor isolation、unaffected-strategy raw-return invariance、multi-seed analysisを独立検証する。
4. 5 candidates + 3 controlsのdevelopment結果を全symbolで完全報告する。
5. winnerをfreezeするか、no-winnerと判断する。

**Canonical real-data baselineは実行・独立検証済みだが、Controlled Experimentはまだ開始していない。** Baseline成立はresearch environmentとevidence pathの検証であり、profitabilityやwinnerの証拠ではない。

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

PPOについては、controlled comparisonのcross-symbol candidate metricsでmulti-seed factor effectとfirst-seed-only candidate summaryが混在する既知のaggregation asymmetryがある。Baseline evidenceのcost/identity validityとは別問題だが、PPOをControlled Factorとして解釈する前に、明示的なcross-seed aggregation contractを確立するか、独立したmulti-seed oracleで補う。

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

## Superseded evidenceの扱い

旧zero-cost canonical Studyは削除・再解釈せず、diagnostic evidenceとして保持する。新しいreal-cost Dataset / Study / EvidenceSetは別identityであり、旧Studyをin-place mutationしていない。

PPO Observation v2確定前に生成されたeconomics-only baselineもdiagnostic evidenceとしてのみ扱う。Current canonical M2のStudy semantic identityを満たさないため、Controlled Experimentのbaselineとして採用しない。

## 現在の次アクション

現時点の次アクションは、Run Coreやbootstrap toolingをさらに拡張することではない。

> 現在のcanonical baselineをimmutable inputとして、一つのControlled Factorを結果を見る前に事前登録し、最初のControlled Experimentを実行・独立検証する。

旧teacher-selection runのrejectは旧mandatory teacher経路を再採用する根拠でも、現候補のprofitabilityを示す証拠でもない。現在の候補は現在のlean contract上で改めて評価する。