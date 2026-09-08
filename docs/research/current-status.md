# Current research status

更新基準: 2026-09-09 (JST)

## 結論

Trade RLの現在地は、**lean coreと5候補+3 controlsの共通比較基盤、およびprovenance-bound candidate Run Coreまでは実装済みだが、canonical実データを使ったM2 development比較はまだ実施していない**段階である。

したがって現在は次を主張しない。

- profitabilityは未証明。
- winnerは未選定。
- Production/live order routingは未認可。
- PPOやforecastがruleを上回るという結論はない。

次の研究上の本質的作業は、新しいmodel familyを増やすことではなく、**一つのcanonical実データartifactとdevelopment windowをfreezeし、Controlled Experiment Loopで5候補+3 controlsを同じ条件にbindして比較すること**である。

## 研究目的

一つの銘柄ID非依存strategy/model/policyを学習・凍結し、各銘柄へ独立に適用する。その結果がpoint-in-time data、同一execution/accounting、hard risk、実コストの下でcontrolsを超え、unused dataでも再現するかを検証する。

Aggregate P&Lだけで成功を判定せず、各symbolの結果を保持する。

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

PPO観測はselected feature、feature availability、current intent、current weightを中心とし、symbol IDを含めない。Training episodeはfit symbolをround-robinし、各episodeは一つのactive symbolだけを扱う。

## Causality and evaluation rules

- `feature_available_time <= decision_time` を守る。
- supervised labelは `label_end_time < fit_cutoff` で完結する。
- future由来のscaler/normalization/imputation/feature selectionを禁止する。
- development/final期間をfitやthreshold調整へ戻さない。
- 同calendar shockを受ける複数銘柄を完全独立標本とみなさない。
- 同じfrozen strategyを各symbolへ独立Replayし、`UniversalStrategyComparison.by_symbol`を主要結果として扱う。

各symbol × strategyでは少なくともtotal return、Sharpe、Sortino、maximum drawdown、turnover、total execution cost、funding P&L、borrow cost、trade/rebalance/termination diagnostics、raw interval returnsを保持する。

## M1 / M2 / M3

### M1 — Lean core: complete

実装済み:

- causal/point-in-time `MarketDataset`
- deterministic filesystem dataset artifact
- canonical `MarketExecutor + BookState` accounting
- hard-risk projection
- quantity-preserving independent symbol replay
- DB/UI/teacher pipelineなしで成立するcore CI

### M2 — Common comparison: Run Core complete, Controlled Experiment Loop implementation in progress, real-data development comparison not run

実装済み:

- 5 candidates + 3 controls
- universal Ridge / LightGBM / teacher-free PPO fit
- symbol-balanced supervised fit
- fit-symbol scopeの明示
- symbol-ID-free PPO
- 全symbol独立comparison
- shared candidate config resolution
- in-memory candidate execution seam
- implementation/runtime/research-context provenance生成
- immutable 3-file candidate-run artifact
- semantic candidate-artifact identityとraw file integrity evidence

未完了:

1. Controlled Experiment LoopのStudy/Experiment lifecycle実装を完了する。
2. canonical実データdevelopment artifactを選ぶ。
3. signal/feature、`fit_symbol_names`、fit cutoff、development window、seed policyをfreezeする。
4. 5 candidates + 3 controlsを同じStudy条件で実行する。
5. 全symbolの結果を完全報告する。
6. winnerをfreezeするか、no-winnerと判断する。

### M3 — Finalize and delete: not started

M2で候補をfreezeした後だけ進む。

1. 未使用future / zero-shot評価を一度だけ開く。
2. pre-registered stressを実行する。
3. 支持されなかったstrategy familyと専用test/extra/dead adapterを削除する。
4. README/config/CI/testsを採用構成へさらに縮約する。
5. Production認可は研究結果とは別に扱う。

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

## 現在の次アクション

現時点の次アクションは、Run Coreをさらに拡張することではない。

> Active Controlled Experiment Loop v1 designを実装し、同じcanonical development条件・provenance・semantic factor・seed policyへcandidate evidenceをbindしたうえで比較を開始する。

旧teacher-selection runのrejectは旧mandatory teacher経路を再採用する根拠でも、現候補のprofitabilityを示す証拠でもない。現在の候補は現在のlean contract上で改めて評価する。
