# Controlled Experiment Loop

## 結論

`trade_rl.evaluation.experiments` は、developmentデータ上の反復研究を **Study → Experiment → EvidenceSet → verification → comparison → decision → freeze** のappend-only state machineとして管理する。単一Runの計算は `evaluation/runs` が担当し、StudyはそのRun Coreを再利用してevidenceを自ら生成する。

このsubsystemはdevelopment専用であり、sealed unused-future / final authorizationを開かない。Studyを `WINNER` または `NO_WINNER` にfreezeした後のfinal評価は別subsystemの責務である。

## Responsibility boundary

```text
evaluation/runs
  config resolution
  one candidate-suite execution
  implementation/runtime/context provenance
  immutable Run artifact
        ↑
evaluation/experiments
  Study contract and seed policy
  multi-seed EvidenceSet
  one-factor delta verification
  paired/seed analysis
  lineage, budget, FAILED/INVALID terminals
  WINNER/NO_WINNER freeze
        ↑
evaluation/experiments/bootstrap
  canonical development source preparation
  canonical dataset publication
  immutable StudyPlan creation
```

`evaluation/runs -> evaluation/experiments` の逆依存は禁止する。`evaluation/experiments` からsealed final-test authorizationへ依存してはならない。

`evaluation/experiments` 内部では、`codec.py` がpersisted payloadのdecode/semantic identity、`inspection.py` がread-only filesystem reconstructionとtamper validation、`workflow.py` がmutation commandを所有する。read sideからmutation workflowへの逆依存は作らない。既存の `workflow.StudySnapshot` / `workflow.inspect_study` は明示export契約としてinspection ownerの同一objectをre-exportする。

## Canonical M2 bootstrap preparation

Canonical M2 bootstrapは、real-data development Studyを開始できる状態までを一度だけ構築するpreparation-only boundaryである。公開入口は `CanonicalM2BootstrapConfig`、`CanonicalM2BootstrapResult`、`bootstrap_canonical_m2_study`、`inspect_canonical_m2_bootstrap` の4つに限定する。

`bootstrap_canonical_m2_study` は次の順序を固定する。

1. strict JSON configを読み、Binance USD-M、symbol roster、base/feature timeframe、data range、baseline config、ordered `ppo_seeds`、allowed factor、experiment budget、bootstrap seed/countを事前登録する。baseline側に別の`ppo_seed` authorityは持たず、`ppo_seeds[0]`だけをbaseline seedへ注入する。
2. Binance exchange-infoのraw bytesと、そのsource URI・retrieval time・SHA-256をfreezeする。
3. pre-registrationから決まる`vision-plan.json`をprimary source planとして固定する。primary monthly kline archiveのtimestamp coverageに欠損がある場合だけ、欠損UTC dayのofficial daily archiveをdeterministic repairとして取得し、`vision-resolution.json`へsymbol/timeframe・missing open timestamp・repair URLを記録する。repair判断にprice/return/P&Lを使わず、補間・synthetic OHLC・REST kline repairは禁止する。primary + repairの実使用raw archiveをURL・SHA-256・sizeのordered rosterとしてfreezeする。
4. source同期後はmarket-data transportを`allow_network=False`で再構成し、dataset buildをcache-onlyへ切る。cache missやREST fallbackによるnetwork accessは失敗とする。
5. frozen metadataとVision evidenceだけからcanonical `MarketDataset`をbuildし、filesystem dataset artifactとしてpublish・reload・identity検証する。
6. 既存Run resolverでbaseline configをdatasetへ解決し、既存`create_study()`だけをStudyPlan authorityとして使う。
7. implementation/runtime provenanceをbootstrap開始時とStudyPlan作成後・終了時に照合し、driftがあればpublishしない。
8. config、primary Vision plan、resolved Vision repair、raw source roster、dataset id/schema/digest、StudyPlan digest、implementation/runtime provenanceを`bootstrap-manifest.json`へbindする。新規writerは`canonical_m2_bootstrap_manifest_v2`を出力し`vision_resolution_digest`を含める。legacy v1 readerはprimary-only artifactだけを明示的に維持し、v1へresolution fileを後付けしたschema混同は拒否する。
9. complete rootをstaging directory内で検証した後、最後のrename一回だけでfinal outputをpublishする。

公開`CanonicalM2BootstrapConfig`はJSON loader経由だけでなく直接constructorでも同じcanonical validationを強制する。output parentはsymlink経由を拒否する。dataset publication後は、事前登録したsource rangeに対するexact timestamp coverageと、登録済みfit symbol scopeに実際のcausal training rowが存在することを確認してからStudyPlanを作る。final staging graphを完全検証した**後**に終了時provenanceを取得・開始時と照合し、その後はResultを事前構築してrenameだけを行う。rename後にnetwork、inspection、hash再計算など失敗し得る処理を置かない。

このbootstrapは**baselineは実行しない**。`run_baseline`、Experiment定義/実行、winner選択、`freeze_study`、sealed unused-future / final authorizationはbootstrap責務ではない。成功直後の`study/`は`plan.json`と同期primitiveだけを持ち、baseline evidenceは存在しない。

```text
<bootstrap>/
  bootstrap.json
  bootstrap-manifest.json
  source/
    vision-plan.json
    vision-resolution.json
    vision-cache/**
    exchange-info/{exchange-info.raw.json,manifest.json}
  dataset/{manifest.json,arrays.npz}
  study/{plan.json,.mutation.lock}
```

`inspect_canonical_m2_bootstrap` はnetworkを使わず、保存済みconfig、primary plan、primary raw bytesから再導出したtimestamp gap、resolved daily repair、primary + repair raw-source roster、dataset artifact、StudyPlan、bootstrap manifestの相互参照を再構築して検証する。保存済み`vision-resolution.json`を盲信せずprimary bytesからrepair URLを再計算する。repair overlapはnormalized open timestamp・OHLC・quote volumeがprimaryと一致する場合だけ許し、primaryをauthorityとして1 rowへcollapseする。不一致、repair後のmissing/duplicate/irregular clock、repair raw bytes/sidecar tamperはfail-closedにする。raw archiveとsidecarを同時に書き換えた場合でも、bootstrap manifestにfreezeされたraw-source rosterとの不一致で拒否する。

## Frozen Study contract

`StudyPlan` は少なくとも次をdigestへbindする。

- canonical dataset identity / artifact schema / artifact digest / symbol roster
- baseline resolved semantic config
- 2個以上のunique non-negative `ppo_seeds`
- allowed controlled factors
- `max_experiments`
- bootstrap count/seed
- implementation/runtime provenance digest

Study作成時にRun Coreの共通resolverでbaseline configを事前解決する。独自のfeature/symbol/timestamp resolverをexperiments層に作らない。

現在の新規Studyでは`ResolvedRunConfig.from_candidate_spec()`が`resolved_run_config_v2`を生成し、PPO Observation schemaとglobal policy rosterをbaseline semantic configへbindする。初回M2のglobal rosterは意図的に空である。`schema_version`、`ppo_observation_schema`、`ppo_global_feature_names`はStudy-fixed resolved fieldであり、Controlled Factorとして変更できない。同一Studyの途中でObservation contractを変えない。

historical `resolved_run_config_v1` / Study artifactはread/inspection互換のため維持するが、current v2 Runをv1 Studyへ継ぎ足すことは許さない。EvidenceSet生成は実行前のfixed-field照合でv1/v2混在をfail-closedにする。旧Studyを新Observationへ暗黙migrationせず、新しいObservation contractで研究を続ける場合は新Studyを作る。

## Study-owned EvidenceSet

Controlled Studyは外部で生成済みの好都合なRunを後付け登録しない。`execute_evidence_set` がStudy seedごとに1 Runを生成し、Run Coreのconfig resolution、execution、provenance、artifact publication/load/inspectionを通す。

EvidenceSet内で変えてよいのは `ppo_seed` だけである。EvidenceSetのsemantic configとそのdigestからは `ppo_seed` を除外し、seed policyはordered `ppo_seeds`として別にidentityへbindする。各Run artifactには実際の `ppo_seed` を保持する。

Studyで維持するordered strategy rosterは `StudyPlan.STRATEGY_NAMES`、そのうちPPO seedを変えてもraw returnsが完全一致しなければならないrosterは `StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES` をsource contractの正本とする。現行契約では `ppo` だけがPPO seed依存を許される。rosterをarchitecture docs側で別途列挙・管理しない。

1 seedでも失敗した場合、complete EvidenceSetはpublishしない。EvidenceSetとanalysisはbundle staging内で完成させてから一度だけrenameするため、partial candidate/baselineを有効evidenceとして残さない。

## Controlled Experiment

`ExperimentDefinition` はcandidate実行前に存在し、sequenceを即時に消費する。sequenceは1始まりで連続し、`max_experiments`を越えない。

各Experimentは1つの `ControlledFactor` だけを宣言する。verificationはbaseline/candidateのresolved semantic configを比較し、宣言factorに対応するpath以外の差分を拒否する。no-opもINVALIDである。

現行factor:

- `FEATURE_SET`
- `RULE_SIGNAL`
- `RULE_THRESHOLDS`
- `FORECAST_THRESHOLDS`
- `FIT_SYMBOL_SCOPE`
- `PPO_TRAINING_BUDGET`
- `GROSS_BUDGET`

factorごとに影響しないstrategyのraw returnsを完全一致で検証する。dataset identity、symbol roster、Study seed policy、implementation/runtime provenance、Study-fixed fieldが変わってもINVALIDである。

## Analysis semantics

既存のpaired-return / block-bootstrap / seed robustness primitiveを再利用する。

- deterministic strategyはseed複製を独立sampleとして数えない。
- PPOはStudyで登録した同一seedをbaseline/candidateでpaired比較する。
- seedごとの結果を保持し、複数seedから単一の架空p-valueを合成しない。
- PPOのcross-symbol candidate metricsは各symbol内でfrozen seedを先に集約する。total return / turnover / total costはseed中央値、maximum drawdownはseed内worstを使い、その後にsymbol間のdescriptive summaryを計算する。これによりseed数をsymbol weightへ変換しない。
- 新規factor-effect payloadは`controlled_evidence_comparison_v2`を使う。persisted `controlled_evidence_comparison_v1`はhistorical first-seed candidate-metric semanticsでread/inspection再計算し、immutable comparisonを暗黙migrationしない。
- symbolは同calendar shockを共有し得るため、cross-symbol significance claimを作らない。
- symbol別結果、正負count、median、worst/best、drawdown/cost/turnover等はdescriptive evidenceとして保持する。

## State machine and lineage

Study mutationは`StudyStore.mutation_lock()`でprocess間serializeし、各mutation前にdisk artifactからstateを再構築する。別DBやin-memory pointerをstateの正本にしない。 read-only `inspect_study` も同じmutation lock内でdisk graphを再構築するため、inspectionとmutationが別moduleでもstate authorityはfilesystemのままである。

```text
create_study
  → run_baseline
  → define_experiment
  → run_experiment
  → verify_experiment
    ├─ INVALID: terminal / budgeted / lineage不変
    └─ CONTROLLED
         → compare_experiment
         → decide_experiment
           ├─ ACCEPT_CANDIDATE: candidate EvidenceSetをlineageへ追加
           ├─ KEEP_BASELINE: lineage不変
           └─ INCONCLUSIVE: lineage不変
```

candidate EvidenceSet完成前の実行失敗は `record_experiment_failure` で `FAILED` terminalとして記録できる。FAILEDはbudgetを消費し、lineageへ入らず、complete candidateを後からFAILEDへ書き換えることはできない。

後続Experimentのbaselineは初期baselineまたは過去の`ACCEPT_CANDIDATE`で到達可能なEvidenceSetだけである。KEEP/INCONCLUSIVE/INVALID/FAILED candidateを復活させない。

## Freeze

`freeze_study` は全Experimentがterminalである場合だけ一度だけ`freeze.json`をpublishする。

- `NO_WINNER`: selected evidence/strategyを持たない。
- `WINNER`: 過去の`ACCEPT_CANDIDATE`でlineageへ入ったEvidenceSetと、5 candidate strategyの1つだけを選べる。controlsはwinnerにできない。

freeze artifactはStudy digestと全decision digestをbindする。再構築時にもdecision lineage、terminal状態、WINNERのaccepted lineageを再検証する。freeze後はbaseline/definition/run/verification/comparison/decision/failure/freezeを含む全mutationを`StudyFrozenError`で拒否する。

## Artifact layout

```text
<study>/
  plan.json
  baseline/
    evidence/
      manifest.json
      runs/seed-*/{summary.json,returns.npz,provenance.json}
    analysis.json
  experiments/
    0001/
      definition.json
      candidate/        # complete EvidenceSetがある場合
      verification.json # candidate完成後
      comparison.json   # CONTROLLEDのみ
      decision.json     # comparison後
      failure.json      # candidate完成前FAILEDの場合
  freeze.json            # terminal Studyのみ
```

`.mutation.lock` は同期primitiveでありresearch evidenceではない。canonical evidenceはappend-onlyで既存targetを上書きしない。

## Failure / tamper contract

loader/inspectionはsymlink、path traversal、malformed JSON、digest mismatch、Run artifact tamper、analysis mismatch、seed roster mismatch、context/provenance drift、stale reference、non-contiguous sequence、illegal terminal transitionをfail-closedに扱う。

Process raceでもlock取得後に必ずdisk stateを再構築するため、例えばfreezeと新Experiment定義が同時に来ても、どちらか一方だけが先行状態へcommitできる。

## What this does not prove

このworkflowがGreenでも、strategyのprofitability、実データ上のwinner、unused-futureでの再現性、production/live routingの安全性は証明されない。Canonical M2 bootstrap toolingの実装完了も同様に研究結果ではない。次の研究作業はcanonical real development bootstrapを実際に作成し、そのStudyでbaselineとControlled Experimentを実行して、`WINNER`または`NO_WINNER`へfreezeすることである。
