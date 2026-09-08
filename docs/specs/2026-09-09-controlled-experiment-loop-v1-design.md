Status: Active

# Controlled Experiment Loop v1 Design

## 1. 結論

Controlled Experiment Loop (CEL) v1 は、現在の M2 development research を **Evidence-first / append-only / fail-closed** な Study として実行する。

最大効果を得るため、実装は二段階に分ける。

1. **PR1 — Candidate Run Core Refactor**
   - 現在の `evaluation/runs/candidate.py` が抱える config resolution / execution / artifact / CLI を分離する。
   - candidate computation semantics は変えない。
   - CEL が CLI/filesystem 都合ではなく in-memory Run Core API を再利用できるようにする。
   - candidate artifact に provenance と stable semantic identity を追加する。

2. **PR2 — Controlled Experiment Loop v1**
   - `evaluation/experiments/` を新設する。
   - canonical dataset と Study 条件を freeze する。
   - baseline と各 Experiment を Study-owned evidence として CEL 自身が実行する。
   - PPO は事前登録した複数 seed を毎回実行する。
   - baseline/candidate の実測 resolved delta が「宣言した1 semantic factorだけ」かを検証する。
   - paired evidence、seed robustness、cross-symbol descriptive evidence を保存する。
   - ACCEPT / KEEP / INCONCLUSIVE を append-only に記録し、最終的に WINNER / NO_WINNER を freeze する。

v1 は development Study の freeze で終了する。unused future / M3 final / stress は開かない。

## 2. Primary correctness property

> dataset、code/runtime、evaluation scope、seed policy、baseline lineage、declared factor、candidate evidence のどれかが統制されていない・改変されている・混在している場合、そのattemptは valid controlled Experiment や Study winner として記録できない。

## 3. 現行 authority と維持する不変条件

次は変更後も維持する。

- 5 candidates: `trend`, `mean_reversion`, `ridge24`, `lightgbm24`, `ppo`
- 3 controls: `cash`, `constant_long`, `constant_short`
- `feature_available_time <= decision_time`
- supervised label は `label_end_time < fit_cutoff`
- universal model/policy に symbol ID を入れない
- `fit_symbol_names` 外の symbol を training row / PPO episode に混ぜない
- evaluation は同じ frozen strategy/model/policy を各 symbol へ独立 replay する
- `MarketExecutor + BookState` が economic/accounting authority
- candidate output overwrite を拒否する
- current candidate CLI を維持する
- no-winner は正常な研究結論である
- control strategy は benchmark であり Study winner ではない

## 4. Non-goals

v1 では次を実装しない。

- final unused-future execution
- final stress orchestration
- fitted Ridge / LightGBM / PPO serialization
- frozen trained-strategy artifact
- new strategy family
- hyperparameter grid/search/optimizer
- automatic threshold tuning
- automatic winner selection
- generic DAG/workflow engine
- experiment DB / registry service
- distributed execution
- web UI/dashboard
- Production/live order routing
- code changeをExperiment factorにすること
- dependency/runtime changeをExperiment factorにすること
- external pre-existing candidate Run を Study に import/register する機能

外部Run importをv1から外す理由は、CEL自身がRun CoreでStudy-owned evidenceを生成する方が、symlink/snapshot/manual registration等の攻撃面と手動ミスを減らし、同じ研究目的をより小さいsurfaceで達成できるためである。

---

# Part A — PR1 Candidate Run Core Refactor

## 5. Objective

現在の `trade_rl/evaluation/runs/candidate.py` は次を同一fileで所有している。

- raw JSON validation
- feature/symbol/timestamp resolution
- dataset artifact loading
- candidate suite execution
- summary/returns conversion
- filesystem publication
- CLI

CELから安全に再利用できるよう、責務を分離する。

これは semantic-preserving refactor であり、`run_lean_candidate_suite()` の fit/replay semantics は原則変更しない。

## 6. Target package

```text
trade_rl/evaluation/runs/
├── __init__.py
├── candidate.py         # thin CLI + compatibility facade
├── candidate_suite.py   # existing 5 candidates + 3 controls execution authority
├── config.py            # raw config + resolution
├── execute.py           # in-memory candidate execution
├── provenance.py        # implementation/runtime provenance
└── artifact.py          # publish/load/inspect/digest
```

## 7. CandidateRunConfig

`config.py` に immutable raw semantic config を置く。

```python
@dataclass(frozen=True, slots=True)
class CandidateRunConfig:
    signal_name: str
    feature_names: tuple[str, ...]
    fit_symbol_names: tuple[str, ...]
    fit_cutoff: np.datetime64
    evaluation_start: np.datetime64
    evaluation_stop_exclusive: np.datetime64
    rule_entry_threshold: float
    rule_exit_threshold: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    ppo_total_timesteps: int
    ppo_seed: int
    gross_budget: float
    initial_capital: float
```

Public Run Core API:

```python
def parse_candidate_run_config(raw: Mapping[str, object]) -> CandidateRunConfig: ...
def load_candidate_run_config(path: str | Path) -> CandidateRunConfig: ...
```

Validation rulesは現行candidate runnerの意味を維持し、unknown key / invalid type / NaT / non-finite / empty duplicate names / threshold relation / invalid PPO budget/seed をfail-closedにする。

## 8. ResolvedCandidateRunSpec

```python
@dataclass(frozen=True, slots=True)
class ResolvedCandidateRunSpec:
    dataset_id: str
    dataset_artifact_schema: str
    dataset_artifact_digest: str
    config: CandidateRunConfig
    lean_config: LeanCandidateConfig
    evaluation_start_index: int
    evaluation_stop_index: int
```

API:

```python
def resolve_candidate_run_spec(
    dataset: MarketDataset,
    *,
    dataset_artifact_schema: str,
    dataset_artifact_digest: str,
    config: CandidateRunConfig,
) -> ResolvedCandidateRunSpec: ...
```

Resolution authorityは一つだけにする。CEL StudyPlanもこの関数を利用し、独自のfeature/symbol/timestamp resolutionを再実装しない。

Resolutionで確定するもの:

- feature name -> index
- fit symbol name -> index
- exact evaluation start/stop indices
- fit/evaluation ordering
- `LeanCandidateConfig`

filesystem pathはidentityに含めない。

## 9. CandidateRunResult / execution

```python
@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    spec: ResolvedCandidateRunSpec
    symbols: tuple[str, ...]
    comparison: UniversalStrategyComparison
```

API:

```python
def execute_candidate_run(
    dataset: MarketDataset,
    spec: ResolvedCandidateRunSpec,
) -> CandidateRunResult: ...
```

`execute_candidate_run` は `run_lean_candidate_suite()` を唯一のfit/replay authorityとして利用する。

実行前に少なくとも `dataset.dataset_id == spec.dataset_id` を再確認する。

## 10. Run provenance

candidate artifact は既存2file semanticsを維持しつつ、evidence-only companionを追加する。

```text
candidate-run/
├── summary.json
├── returns.npz
└── provenance.json
```

`summary.json` の result schema は引き続き `lean_candidate_result_v1` とし、候補計算の意味を変えない。

`provenance.json` schema:

```text
candidate_run_provenance_v1
```

少なくとも次を含む。

### implementation manifest/digest

importされた `trade_rl` package root 以下の runtime `.py` を対象に:

1. normalized relative POSIX path
2. exact source byte SHA-256
3. path順sort
4. versioned canonical manifest
5. `content_digest()`

を生成する。

`__pycache__` / bytecode / tests / docs / absolute path / generated debris は含めない。

### runtime environment manifest/digest

固定されたpackage listをcode/testで管理し、少なくとも次を記録する。

- Python implementation + full version
- OS family/release
- machine architecture
- `trade-rl`
- NumPy
- Gymnasium
- LightGBM
- Stable-Baselines3
- Torch

optional packageが未installなら `null` としてcanonical manifestへ記録する。実candidate suiteを実行できる環境では必要extrasがinstallされていることは別契約である。

## 11. Candidate artifact identity

raw `.npz` file byte hashだけをresearch identityにしない。ZIP metadataによりsemantic-equivalent artifactのcontainer bytesが異なり得るためである。

`candidate_run_artifact_identity_v1` は次をsemantic digestする。

1. canonical parsed `summary.json`
2. sorted returns array keys
3. 各arrayの normalized dtype descriptor
4. shape
5. contiguous C-order numeric bytes SHA-256
6. canonical parsed `provenance.json`

`returns.npz` は:

- `allow_pickle=False`
- numeric 1-D arrays only
- finite values only

を要求する。

同時にtamper/debug用file evidenceとして各fileの raw SHA-256 + byte sizeもinspection resultに保持するが、filesystem path/mtimeはsemantic identityに含めない。

## 12. Candidate artifact API

```python
@dataclass(frozen=True, slots=True)
class PublishedCandidateRun:
    root: Path
    summary_path: Path
    returns_path: Path
    provenance_path: Path

@dataclass(frozen=True, slots=True)
class LoadedCandidateRun: ...

@dataclass(frozen=True, slots=True)
class CandidateRunArtifactIdentity:
    schema_version: str
    result_schema_version: str
    artifact_digest: str
    summary_file_sha256: str
    returns_file_sha256: str
    provenance_file_sha256: str
```

API:

```python
def publish_candidate_run(... ) -> PublishedCandidateRun: ...
def load_candidate_run_artifact(root: str | Path) -> LoadedCandidateRun: ...
def inspect_candidate_run_artifact(root: str | Path) -> CandidateRunArtifactIdentity: ...
```

publicationはstaging directory -> final renameでatomicにし、existing destinationを拒否する。

## 13. CLI compatibility

現行入口は維持する。

```bash
python -m trade_rl.evaluation.runs.candidate \
  --dataset <dataset-artifact-dir> \
  --config <run-config.json> \
  --output <new-result-dir>
```

`candidate.py` は次のthin orchestrationだけを行う。

```text
inspect/load dataset artifact
-> load CandidateRunConfig
-> resolve CandidateRunSpec
-> execute CandidateRun
-> build provenance
-> publish CandidateRun
-> print output root
```

existing `run_candidate_artifact()` はcompatibility facadeとして同じobservable behaviorを維持し、新Core APIへ委譲する。

## 14. PR1 Quality Contract

### Objective

Run computationとfilesystem/CLIを分離し、CELが同じresolution/execution/artifact authorityを再利用できるようにする。

### Non-goals

candidate roster、fit semantics、threshold semantics、risk/execution/accounting、evaluation scope、return metricsは変えない。

### Invariants

- `LeanCandidateConfig` semantics不変
- exact candidate-suite call arguments不変
- `lean_candidate_result_v1` summary semantic payload不変
- raw interval returns不変
- candidate CLI不変
- top-level `trade_rl.evaluation.__all__`不変

### Failure Modes

- duplicated resolution implementation
- dataset/spec identity mismatch
- CLI drift
- result payload drift
- partial publication
- malformed/extra/missing artifact file
- object/pickle/non-finite return array
- provenance nondeterminism from absolute paths
- semantic digest dependence on mtime/zip metadata

### Test Oracle

- exact config parse/resolve
- exact suite call args
- existing candidate summary/returns regression
- publish/load roundtrip
- semantic digest stability under repacked NPZ
- implementation/runtime provenance stability
- overwrite and cleanup filesystem state
- CLI help/exit/output

### Required Test Layers

Unit / Contract / Filesystem Integration / Architecture / Static / Full tests / exact-head CI / source-level falsification.

---

# Part B — PR2 Controlled Experiment Loop v1

## 15. Core terminology

### Candidate Run

一つの `CandidateRunConfig` と一つの PPO seed で実行される既存8-strategy suite。

### EvidenceSet

同じ semantic config を、Study-wide `ppo_seeds` 全てで実行した ordered Candidate Run集合。

PPO seed以外のconfig差分は禁止する。

### Experiment

```text
one explicit hypothesis
+ one lineage-eligible baseline EvidenceSet
+ one candidate EvidenceSet
+ one declared semantic factor
+ actual resolved delta verification
+ comparison evidence
+ one immutable decision
```

### Study

同じ development dataset/evaluation/provenance/seed policy 下で行う bounded sequence of Experiments。

## 16. Package boundary

```text
trade_rl/evaluation/experiments/
├── __init__.py
├── errors.py
├── contracts/
│   ├── __init__.py
│   ├── study.py
│   ├── experiment.py
│   └── decision.py
├── store.py
├── evidence.py
├── delta.py
├── analysis.py
└── workflow.py
```

責務:

- `contracts`: immutable domain contracts only
- `store`: filesystem lock / atomic append-only publication
- `evidence`: multi-seed Run Core execution + EvidenceSet inspection
- `delta`: declared factor / fixed field / unaffected-strategy verification
- `analysis`: existing paired/seed primitivesによるevidence
- `workflow`: state transition orchestration

`experiments` は final sealed-test authorizationをimportしない。

## 17. Study filesystem artifact

```text
<study-root>/
├── plan.json
├── baseline/
│   ├── evidence.json
│   ├── analysis.json
│   └── runs/
│       ├── seed-0/{summary.json,returns.npz,provenance.json}
│       ├── seed-1/{summary.json,returns.npz,provenance.json}
│       └── ...
├── experiments/
│   ├── 0001/
│   │   ├── definition.json
│   │   ├── candidate/
│   │   │   ├── evidence.json
│   │   │   ├── analysis.json
│   │   │   └── runs/seed-*/...
│   │   ├── verification.json
│   │   ├── comparison.json
│   │   └── decision.json
│   └── ...
├── freeze.json
└── .mutation.lock
```

`.mutation.lock` はfilesystem coordinationのみでresearch identityではない。

published research artifactsはoverwriteしない。

## 18. StudyPlan

Study creation input:

- research question
- canonical dataset artifact
- baseline `CandidateRunConfig`
- `ppo_seeds`
- allowed controlled factors
- `max_experiments`
- analysis policy

Study creationはRun Coreの `resolve_candidate_run_spec()` を利用してbaseline configをfitなしでpre-resolveする。

`StudyPlan` は少なくとも次をfreezeする。

```text
schema_version = controlled_experiment_study_v1
research_question
dataset_id
dataset_artifact_schema
dataset_artifact_digest
evaluation symbol roster
fit_cutoff
evaluation_start/evaluation_stop_exclusive
initial_capital
execution contract identifier
ppo_seeds
implementation_digest
runtime_environment_digest
max_experiments
allowed_factor_types
analysis policy
initial_baseline_resolved_config
candidate roster
control roster
```

Rules:

- `ppo_seeds` は最低2個、unique、non-negative int
- `baseline_config.ppo_seed == ppo_seeds[0]`
- `1 <= max_experiments <= 9999`
- research_question non-empty
- unknown factor rejected

implementation/runtime digestsはStudy creation時にfreezeし、以後の全Run provenanceと一致必須。

## 19. Study-wide fixed controls

次はExperiment factorではなくStudy固定。

- dataset identity/artifact digest
- evaluation symbol roster/window
- fit cutoff
- initial capital
- execution contract
- PPO seed policy
- implementation digest
- runtime environment digest
- 5 candidate + 3 control roster

変更する場合は新Study。

## 20. Controlled factors

v1で許可できるsemantic factorは次だけ。

```text
FEATURE_SET
RULE_SIGNAL
RULE_THRESHOLDS
FORECAST_THRESHOLDS
FIT_SYMBOL_SCOPE
PPO_TRAINING_BUDGET
GROSS_BUDGET
```

各factorは「変更を許可するresolved fields」と「raw returnsが不変でなければならないstrategy集合」を固定する。

| Factor | Allowed resolved delta | Required unaffected strategies |
|---|---|---|
| `FEATURE_SET` | feature names/indices | cash, constant_long, constant_short, trend, mean_reversion |
| `RULE_SIGNAL` | rule signal name/index | cash, constant_long, constant_short, ridge24, lightgbm24, ppo |
| `RULE_THRESHOLDS` | rule entry/exit thresholds | cash, constant_long, constant_short, ridge24, lightgbm24, ppo |
| `FORECAST_THRESHOLDS` | forecast entry/exit thresholds | cash, constant_long, constant_short, trend, mean_reversion, ppo |
| `FIT_SYMBOL_SCOPE` | fit symbol names/indices | cash, constant_long, constant_short, trend, mean_reversion |
| `PPO_TRAINING_BUDGET` | PPO total timesteps | cash, constant_long, constant_short, trend, mean_reversion, ridge24, lightgbm24 |
| `GROSS_BUDGET` | evaluation gross budget | cash |

factorはnon-empty actual deltaを要求する。declared factorがno-opならCONTROLLEDにはしない。

上表のunaffected setはcurrent code dataflowをarchitecture/contract testで独立検証する。実装中に反証された場合は、理由とsource evidenceをspecへ反映してから変更する。

## 21. Multi-seed EvidenceSet execution

CELは外部Runをimportせず、Run CoreでStudy-owned evidenceを直接生成する。

EvidenceSet生成時:

1. dataset artifact digestをStudyPlanと再照合
2. current implementation/runtime provenanceをStudyPlanと再照合
3. base configからseedごとにconfigをclone
4. 変更を許可するfieldは `ppo_seed` のみ
5. 各seedで Run Core resolve -> execute -> publish
6. 各candidate artifactをinspectしてdigest検証
7. deterministic candidate invarianceを検証
8. analysisを生成
9. staging rootをEvidenceSet final directoryへatomic rename

途中failureではfinal EvidenceSet directoryを残さない。

### deterministic candidate invariant across PPO seeds

次はseed間でraw returns、metrics、diagnosticsがsemantic equalでなければならない。

- cash
- constant_long
- constant_short
- trend
- mean_reversion
- ridge24
- lightgbm24

PPOのみseed variationを許可する。

## 22. EvidenceSet identity

`candidate_evidence_set_v1` fingerprint はcanonical contentから計算する。

```text
dataset identity
resolved semantic config excluding ppo_seed
ordered ppo_seeds
ordered seed -> candidate artifact semantic digest
implementation/runtime digests
analysis policy identity
```

filesystem path/mtimeを含めない。

## 23. Within-EvidenceSet analysis

analysisはpublished candidate artifactsをloadして行い、strategy fit/replayを再実行しない。

### deterministic candidates vs controls

各symbolで:

- trend
- mean_reversion
- ridge24
- lightgbm24

を cash / constant_long / constant_short と `compare_paired_returns()` で比較する。

### deterministic candidate-vs-candidate

trend / mean_reversion / ridge24 / lightgbm24 の全unordered pairを secondary evidenceとして比較する。

### PPO seed robustness

PPOの全seed seriesを各symbolで次のbaselineそれぞれに対して `summarize_seed_robustness()` で要約する。

- cash
- constant_long
- constant_short
- trend
- mean_reversion
- ridge24
- lightgbm24

単一のprimary PPO seedをwinner evidenceとして利用しない。

### cross-symbol descriptive evidence

完全独立標本とみなすaggregate significance testは作らない。

candidateごとに少なくとも:

- positive/negative excess symbol count vs controls
- median excess total return
- worst/best symbol excess
- median total return
- worst max drawdown
- median turnover
- median total cost

を保存する。

raw per-symbol evidenceが常に正本。

## 24. Study state machine

### Baseline

```text
plan.json
  -> run_baseline()
  -> baseline/ EvidenceSet
```

baselineは1回だけ。

### Experiment

```text
definition.json                         -> DEFINED
+ candidate/ EvidenceSet                -> EVIDENCE_BOUND
+ verification.json CONTROLLED          -> VERIFIED
+ comparison.json                       -> COMPARED
+ decision.json                         -> DECIDED

verification.json INVALID               -> INVALID terminal
```

workflow stageをskipしない。

## 25. ExperimentDefinition

definitionはcandidate execution前にpublishする。

```text
schema_version
study_digest
sequence
hypothesis
baseline_evidence_set_fingerprint
controlled_factor
candidate requested config
```

Requirements:

- exact next contiguous sequence
- 4-digit directory name
- non-empty hypothesis
- Study-permitted factor
- lineage-eligible baseline
- Study not frozen
- budget remains
- candidate config parses successfully
- Study-fixed fields requested config上でも変更されていない

Experiment definition publication時点で1attemptを消費する。

## 26. Controlled-delta verification

`verify_experiment()` はbaseline/candidateの**resolved Study-owned evidence**を比較する。

CONTROLLED条件:

- all Run artifacts integrity valid
- Study/dataset/provenance/seed policy identity一致
- evaluation roster/window一致
- fit cutoff / initial capital / execution contract一致
- strategy/control roster一致
- candidate configのactual resolved deltaがdeclared factor fieldだけ
- actual delta non-empty
- required unaffected strategiesのraw returnsがbaseline/candidateでexact equal

### INVALID

trustworthy evidenceが上のcontract violationを証明した場合、terminal `INVALID` verificationをimmutable publishする。

INVALIDはbudgetから消さない。

### Operational failureとの区別

missing/malformed/tampered artifact等でtrustworthy comparison自体が成立しない場合はINVALIDを書かず、integrity/state errorをraiseする。

## 27. Experiment comparison

CONTROLLEDだけ比較可能。

同じsymbol × same strategyについてbaseline/candidate factor effectを保存する。

- baseline metrics
- candidate metrics
- total-return delta
- Sharpe delta
- max-drawdown delta
- turnover delta
- total-cost delta
- funding delta
- borrow delta
- paired excess total/log return
- moving-block bootstrap evidence
- baseline/candidate EvidenceSet fingerprint
- verification digest

candidate EvidenceSet自身のwithin-suite analysis digestもcomparisonへbindする。

missing / duplicate / extra symbol×strategy cellはfail-closed。

p-valueはdecisionを自動化しない。

## 28. Experiment decision

Exactly:

```text
ACCEPT_CANDIDATE
KEEP_BASELINE
INCONCLUSIVE
```

Decision contract:

```text
schema_version
study_digest
experiment_digest
verification_digest
comparison_digest
decision
rationale
decided_by
decided_at
```

- rationale / decided_by non-empty
- decided_at timezone-aware and caller-supplied
- immutable one-shot publication
- retestは新Experiment

## 29. Accepted-baseline lineage

later Experiment baselineとして利用可能なのは:

1. initial baseline EvidenceSet
2. prior `ACCEPT_CANDIDATE` candidate EvidenceSetでinitial baselineからreachableなもの

KEEP_BASELINE / INCONCLUSIVE / INVALID candidateはlineageに入らない。

## 30. Budget and sequence

- `1 <= max_experiments <= 9999`
- Experiment definitionごとに1attempt
- INVALIDもattempt消費
- contiguous sequence starting 0001
- overwrite / reuse / renumber禁止
- gapがあるStudyはfreeze不可
- budget超過は `ExperimentBudgetExceededError`

これはdevelopment windowへのadaptive overfittingを完全には消さないが、無制限反復を防ぐ最低限のresearch controlである。

## 31. Concurrency

v1はsingle-writer / multi-reader。

mutation operationは `StudyStore` のprocess-safe advisory lockでserializeする。

実装は追加dependencyを導入せず:

- POSIX: `fcntl.flock`
- Windows: `msvcrt.locking`

をprivate abstractionで使う。

process terminationでOS lockがreleaseされる方式とし、stale sentinelだけに依存しない。

lock fileは`.mutation.lock`でresearch identity外。

race testで少なくとも同sequence二重defineとfreeze/mutation競合を検証する。

## 32. Study freeze

Terminal outcomes:

```text
WINNER
NO_WINNER
```

### WINNER

requires:

- selected EvidenceSet fingerprintがACCEPT-lineage reachable
- selected_strategyが `trend|mean_reversion|ridge24|lightgbm24|ppo`
- control不可
- non-empty rationale

### NO_WINNER

selected EvidenceSet / strategyを持たない。

freeze前にdiskから再構築して:

- plan integrity
- baseline integrity
- contiguous experiment sequence
- all attempts terminal DECIDED or INVALID
- all digest references
- accepted lineage
- selected candidate validity
- no existing freeze

を検証する。

freeze後の全mutationは `StudyFrozenError`。

PPO等のexact fitted model serializationはv1のfreeze対象外であり、M3へ進む前に別sub-project `Frozen Strategy Artifact` が必要。

## 33. Public API

`trade_rl.evaluation.experiments` package boundaryとしてのみ公開し、top-level `trade_rl.evaluation.__all__` へは再exportしない。

High-level API:

```text
create_study
run_baseline
define_experiment
run_experiment
verify_experiment
compare_experiment
decide_experiment
freeze_study
inspect_study
```

Public immutable contracts:

```text
StudyPlan
ControlledFactor
ExperimentDefinition
EvidenceSet
ControlledVerification
ExperimentComparison
ExperimentDecision
StudyFreeze
StudySnapshot
```

## 34. Error model

```text
ControlledExperimentError
├── ContractViolationError
├── ArtifactIntegrityError
├── InvalidExperimentStateError
├── UncontrolledDeltaError
├── ExperimentBudgetExceededError
└── StudyFrozenError
```

`UncontrolledDeltaError` はdomain verification内部で利用できるが、workflowはtrustworthy violationを通常terminal INVALIDとしてmaterializeする。

## 35. PR2 Quality Contract

### Objective

同じStudy固定条件下で「何を変えたか」をresolved evidenceで証明し、multiple PPO seedsを含むcomplete evidenceをappend-onlyに保存し、bounded accepted lineageからwinner/no-winnerをfreezeする。

### Invariants

1. lower fit/replay/accounting semanticsを再実装しない
2. Study fixed controls/provenanceはCONTROLLED experimentで変わらない
3. 一Experiment一semantic factor
4. actual delta non-empty
5. EvidenceSetは全pre-registered PPO seedsを含む
6. PPO以外のdeterministic candidatesはseed間で不変
7. unaffected strategy raw returnsはfactor変更前後で不変
8. INVALIDはvisible/budgeted
9. append-only immutable publication
10. rejected/inconclusive/invalid candidateはlineage不可
11. controlsはwinner不可
12. freeze terminal
13. final unused dataに依存しない

### Failure Modes

- malformed StudyPlan/config
- dataset artifact drift
- implementation/runtime provenance drift
- seed missing/duplicate/extra
- seed以外のrun config drift
- deterministic candidate seed drift
- unsupported factor
- no-op factor
- undeclared resolved delta
- unaffected strategy drift
- sequence race/gap/reuse
- budget exhaustion
- broken baseline lineage
- partial EvidenceSet publication
- tampered candidate artifact/digest mismatch
- compare before CONTROLLED
- decision before comparison
- second decision
- freeze with open Experiment
- rejected/invalid/control winner resurrection
- mutation after freeze

### Test Oracle

- canonical/content digest equality
- exact filesystem state
- exact resolved configs
- seed set/order
- candidate artifact semantic identity
- deterministic raw return equality
- controlled delta partition
- paired evidence equality to existing primitives
- seed robustness equality to existing primitive
- complete symbol×strategy matrix
- lineage graph reconstruction
- sequence/budget accounting
- lock/race behavior
- freeze reconstruction

### Required Test Layers

- contract unit tests
- property tests for factor delta partition / identity
- filesystem integration
- candidate Run Core integration via established execution seam
- paired/seed analysis tests
- state/lineage/freeze tests
- race tests
- tamper/digest regression tests
- architecture dependency tests
- Ruff / Format / Mypy
- full pytest
- package identity
- exact-head GitHub CI
- requirements-first falsification review

## 36. TDD order

### PR1 RED

1. new runs package responsibilities/API absent
2. config resolution extraction contract
3. candidate artifact provenance/digest roundtrip
4. semantic digest repack stability
5. existing CLI/result compatibility

### PR2 RED

1. `evaluation/experiments` boundary absent
2. final sealed-test dependency forbidden
3. StudyPlan/seed/budget/factor contracts
4. EvidenceSet multi-seed execution
5. deterministic candidate invariance
6. delta factor partition
7. append-only state ordering
8. paired/seed analysis
9. lineage/freeze
10. concurrency/tamper/partial cleanup

No assertion weakening、skip、forwarding shim、test-only hardcodeでGreenにしない。

## 37. Acceptance Criteria

### PR1

1. CandidateRunConfig / ResolvedCandidateRunSpec / CandidateRunResultが分離される
2. CELがJSON/CLIを介さずRun Coreを利用できる
3. shared config resolution authorityは一つ
4. candidate suite fit/replay semantics不変
5. `lean_candidate_result_v1` summary semantics不変
6. provenance companionが生成される
7. semantic artifact digestがpath/mtime/NPZ zip metadataに依存しない
8. candidate artifact load/inspectがtamper/malformed arrayをfail-closed
9. current CLIと`run_candidate_artifact()` facade維持
10. targeted/full/static/package identity/exact-head CI Green
11. falsification reviewでsemantic driftなし

### PR2

1. `evaluation/experiments`がdistinct boundary
2. Study creationがRun Core resolutionを再利用
3. Studyはdataset/evaluation/provenance/PPO seed policyをfreeze
4. baseline EvidenceSetは全seedをStudy-ownedに生成
5. each Experiment definitionはexecution前にimmutable publish
6. candidate EvidenceSetはPPO seedだけをrun間で変更
7. deterministic candidatesはseed間でexact invariant
8. declared factorだけがresolved baseline/candidate delta
9. unaffected strategy drift検出
10. trustworthy uncontrolled attemptはINVALIDとして残りbudget消費
11. CONTROLLEDだけcomparison可能
12. within-suite paired evidence + PPO seed robustnessを保存
13. cross-symbol evidenceをfalse-independent significanceへ変換しない
14. decisionはACCEPT/KEEP/INCONCLUSIVEのみ
15. accepted lineageだけがlater baseline/winner source
16. budget/sequence append-only
17. mutation concurrencyでsequence/freeze破壊なし
18. WINNERはcandidateのみ、NO_WINNERを正常terminal stateとして扱う
19. freeze後mutation不可
20. final unused-data/stressへ依存しない
21. architecture/research docs更新
22. targeted/full/static/package identity/exact-head CI Green
23. falsification reviewでunresolved Critical/Highなし

## 38. Documentation lifecycle

実装中、このspecとimplementation planは `Status: Active` で `docs/specs/` / `docs/plans/` に保持する。

PR2完了時、durable contractsを:

- `docs/architecture/package-boundaries.md`
- `docs/research/current-status.md`
- 必要な場合のみ `docs/architecture/lean-core.md`

へ反映した後、completed spec/planをcurrent treeから削除する。Git historyがarchive。

## 39. Implementation / merge order

1. PR1 Candidate Run Core Refactor
2. PR1 exact-head verification
3. PR2 Controlled Experiment Loop v1 based on PR1
4. PR2 exact-head verification
5. main mergeは別途explicit authorization
6. M3へ進む前にFrozen Strategy Artifactを別設計する
