# Experiment Strategy Roster Authority

Status: Active

## Objective

Controlled Experiment が使用する strategy roster と PPO-seed invariance policy の意味上の正本を `StudyPlan` に集約し、`analysis.py` / `evidence.py` / `delta.py` が同じ8戦略を独自に再定義しない構造にする。

同時に、Run Core は実際のstrategy構築・実行authorityとして独立維持し、Controlled Experiment から Run Core へ研究分類を逆流させない。

## Non-goals

- strategy実装、学習、simulation、risk、returns、research decisionを変更しない。
- `CANDIDATE_STRATEGY_NAMES` / `CONTROL_STRATEGY_NAMES` の値・順序・persisted Study payloadを変更しない。
- Candidate Run artifact schema、Study schema version、EvidenceSet schema、analysis schemaを変更しない。
- `candidate_suite.py` を Controlled Experiment contractへ依存させない。
- Factorごとの `unaffected_strategies` policyを共通rosterへ置換しない。
- 将来の未知のstochastic strategyを先回りした汎用StrategyRoster型・registry・plugin systemを導入しない。
- `trade_rl.evaluation.experiments.__all__` や `contracts.__all__` を拡張しない。

## Current problem

Study contract は既に以下を正本としてpersistしている。

- `CANDIDATE_STRATEGY_NAMES = (trend, mean_reversion, ridge24, lightgbm24, ppo)`
- `CONTROL_STRATEGY_NAMES = (cash, constant_long, constant_short)`

しかし同じStudy rosterが以下で独立再定義されている。

- `experiments/analysis.py::_STRATEGIES`
- `experiments/evidence.py::_DETERMINISTIC_STRATEGIES` + `_EXPECTED_STRATEGIES`
- `experiments/delta.py::_STRATEGIES`

Run Core の `candidate_suite.py` にあるstrategy mappingも同じ8名称・同じ順序だが、これは実際のstrategy objectを構築する実行authorityであり、研究契約rosterとは責務が異なる。ここを1つのproduction定数へ統合すると `evaluation/runs -> evaluation/experiments` 依存を作るか、逆にStudyのcandidate/control分類をRun Coreへ移してしまう。

## Chosen design

### Study-owned semantic roster

`StudyPlan` に `ClassVar` として次を追加する。

```python
STRATEGY_NAMES: ClassVar[tuple[str, ...]]
PPO_SEED_INVARIANT_STRATEGY_NAMES: ClassVar[tuple[str, ...]]
```

`STRATEGY_NAMES` は既存module-level authorityから機械的に構成する。

```python
CONTROL_STRATEGY_NAMES + CANDIDATE_STRATEGY_NAMES
```

`PPO_SEED_INVARIANT_STRATEGY_NAMES` は、現行Study契約で唯一PPO seedに依存してよい `ppo` を除いたrosterとして同じownerで導出する。

これらは `ClassVar` なのでdataclass field・Study payload・digestには入らない。既存のcandidate/control module constantsとStudy payloadがpersisted contractの正本であることも維持する。

### Consumer behavior

- `analysis.py` は matrixの期待roster・順序とseed-invariant strategy loopを `StudyPlan` から参照する。
- `evidence.py` はEvidenceSet内の期待rosterとseed invariance検証を `StudyPlan` から参照する。
- `delta.py` はRun evidenceの期待rosterを `StudyPlan` から参照する。
- `FACTOR_RULES[*].unaffected_strategies` はFactor固有policyとして維持する。ただし全要素が `StudyPlan.STRATEGY_NAMES` の部分集合であることをcontract testで保証する。

### Run Core boundary

`evaluation/runs/candidate_suite.py` のstrategy mappingは変更しない。Run Coreは実行authorityのまま独立する。

代わりにtestで、実際にCandidate Suiteへ渡されるstrategy mappingのordered keysが `StudyPlan.STRATEGY_NAMES` と一致することを検証する。これにより、Study契約とRun実装のどちらか一方だけが変更されたdriftを検出する。

既存Study contract testではcandidate/controlの具体的なliteral rosterを引き続きassertし、cross-layer consistency testだけをproduction定数の相互照合にする。したがって両production側を同時に誤変更しても、exact semantic contractの独立oracleが残る。

## Data flow

```text
CANDIDATE_STRATEGY_NAMES ─┐
                          ├─> StudyPlan.STRATEGY_NAMES
CONTROL_STRATEGY_NAMES ───┘            │
                                       ├─> analysis expected matrix/order
                                       ├─> evidence expected roster
                                       ├─> delta expected roster
                                       └─> cross-layer test
                                                │
Run Core candidate_suite strategy mapping ──────┘
```

PPO seed policy:

```text
StudyPlan.STRATEGY_NAMES
        │
        └─ remove only `ppo`
              ↓
StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES
              ├─ evidence raw-return equality across PPO seeds
              └─ analysis deterministic single-seed comparisons
```

## Acceptance Criteria

1. `StudyPlan.STRATEGY_NAMES` is exactly controls followed by candidates and equals the maintained eight-strategy Study roster.
2. `StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES` contains exactly the seven current non-PPO strategies, preserving order.
3. `analysis.py`, `evidence.py`, and `delta.py` no longer contain independent full Study roster tuples.
4. `analysis.py` uses the Study-owned roster for matrix length/order and the Study-owned PPO-seed-invariant roster for deterministic comparisons.
5. `evidence.py` uses the Study-owned roster for expected names and the Study-owned PPO-seed-invariant roster for cross-seed raw-return equality.
6. `delta.py` uses the Study-owned roster when validating frozen Study evidence.
7. Every `FACTOR_RULES[*].unaffected_strategies` remains a subset of the Study roster; its factor-specific contents remain unchanged.
8. Candidate Run suite execution produces the exact same ordered strategy names as `StudyPlan.STRATEGY_NAMES` without adding a production dependency from Run Core to Experiments.
9. Existing candidate/control Study payload fields and Study digest semantics are unchanged.
10. Existing Candidate Run artifact/evidence/analysis/delta outputs remain behaviorally unchanged.
11. No new public facade export, registry, generic roster class, or permanent generated metadata is introduced.
12. The durable architecture document stops hand-maintaining the seven-item deterministic roster and instead names the Study contract authority.

## Invariants

- Study candidate/control names and ordering stay unchanged.
- `ppo` remains the only maintained strategy allowed to vary with `ppo_seed`.
- Run Core remains independent of Controlled Experiment production modules.
- `StudyPlan.to_payload()` remains byte-semantically equivalent for identical inputs.
- `StudyPlan.digest` remains unchanged for identical inputs.
- Existing schema versions remain unchanged.
- Factor-specific unaffected-strategy policy remains independently owned by `FACTOR_RULES`.

## Failure Modes

- `StudyPlan` ClassVars accidentally become dataclass fields and alter serialization/digest.
- Analysis/evidence/delta retain a hidden duplicate roster under a different local name.
- Strategy ordering changes while set equality still passes.
- `ppo` accidentally enters the seed-invariant roster.
- A future strategy is added to StudyPlan but not Candidate Run execution, or vice versa.
- Factor-specific unaffected sets contain a typo/non-Study strategy.
- Run Core gains an import on Controlled Experiment contracts merely to obtain names.
- Docs retain a second manually maintained exhaustive deterministic roster.

## Test Oracle

### Architecture/contract RED

Add a focused architecture contract that requires:

- exact `StudyPlan.STRATEGY_NAMES` and `PPO_SEED_INVARIANT_STRATEGY_NAMES`;
- both names are `ClassVar` rather than dataclass fields;
- no local full-roster declarations remain in `analysis.py`, `evidence.py`, `delta.py`;
- factor-specific unaffected sets are subsets of the Study roster;
- no `evaluation.runs` production import of `evaluation.experiments` is introduced.

The pre-change stack head must fail only these new authority conditions after Ruff/Format/Mypy reach pytest.

### Behavior/consistency

- keep the existing exact candidate/control literal assertions in Experiment contract tests;
- change/add Candidate Suite test so observed strategy mapping order equals `StudyPlan.STRATEGY_NAMES`;
- existing analysis/evidence/delta tests must remain green;
- existing Study payload/digest tests remain green.

### Final gate

On the exact final stacked head:

- Ruff;
- Format;
- production Mypy;
- architecture-tooling Mypy;
- full pytest;
- `uv build`;
- tracked source closure for sdist/direct wheel/sdist-rebuilt wheel;
- clean non-editable installed package smoke;
- Candidate Run and canonical bootstrap CLI smoke;
- package identity;
- final diff review and current parent/base relationship.

After parent #469 merges, this PR must be retargeted/rebased or merged-forward to current `main` without force-push and the final gate rerun before it can be Ready for main integration.

## Docs lifecycle

This spec is Active only while the change is under review/implementation. A matching Active plan is allowed during implementation. After the implementation is verified, durable rationale is compressed into the existing `docs/architecture/controlled-experiment-loop.md`; this spec and plan are removed from the current tree and remain available only through Git history.
