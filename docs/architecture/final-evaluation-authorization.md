# Final evaluation authorization

## 結論

`trade_rl.evaluation.final_test` は、development Studyが `StudyOutcome.WINNER` でfreezeされた後にだけ、**unused-future / final evaluationを開く資格**を一度だけ封印する境界である。

このsubsystem自身はfinal Datasetを取得・構築せず、strategyを実行せず、P&Lを計算しない。役割は「developmentで選ばれたwinnerと、まだ開いていないfinal windowをimmutable authorizationへbindする」ことだけである。

したがって、authorization artifactが存在することはprofitability、final-test成功、Production readiness、live order routing認可を意味しない。

## Objective

Developmentとfinal evaluationの間に、コードで強制されるhard gateを置く。

満たすべき性質は次である。

- development Studyはauthorization前にterminal `WINNER` でfreezeされている。
- `NO_WINNER`、未freeze、途中Studyからはfinal evaluationを認可しない。
- winner EvidenceSet digestとwinner strategyは`StudyFreeze`と完全一致する。
- final evaluation windowはdevelopment evaluation windowと重ならない。
- authorization発行はfrozen Studyを書き換えない。
- authorizationは別rootへone-shotでpublishし、再発行・上書きを許さない。
- read-back時にStudyPlan、StudyFreeze、winner evidence、winner strategy、window、authorization digestを再検証する。

## Non-goals

このpackageは次を所有しない。

- final Datasetのdownload/build/cache。
- exchange/provider access。
- strategy fitまたはinference。
- replay、execution、accounting、P&L。
- robustness/stress/capacity評価。
- account-specific execution economics calibration。
- Production/live authorization。

そのため`trade_rl.evaluation.final_test`から`data`、`integrations`、`strategies`、`evaluation.replay`、`evaluation.runs`、`evaluation.robustness`への直接依存を禁止する。

## Authorization contract

`FinalEvaluationAuthorization` は少なくとも次をbindする。

| Field | Authority |
|---|---|
| `study_digest` | frozen development `StudyPlan.digest` |
| `study_freeze_digest` | terminal `StudyFreeze.digest` |
| `winner_evidence_digest` | `StudyFreeze.selected_evidence_digest` |
| `winner_strategy` | `StudyFreeze.selected_strategy` |
| `development_evaluation_stop_exclusive` | frozen baseline config |
| `final_evaluation_start` | preregistered final window |
| `final_evaluation_stop_exclusive` | preregistered final window |
| `authorized_by` | authorization actor identity |
| `authorized_at` | aware authorization timestamp |
| `schema_version` | persisted contract version |

Timestampはcanonical nanosecond表現を使う。`final_evaluation_start` はdevelopment stopより前であってはならず、`final_evaluation_stop_exclusive` はstartより厳密に後でなければならない。`authorized_at` はbound Studyの`frozen_at`より前であってはならない。

## State transition

```text
Development Study
    │
    ├─ unfrozen --------------------------X final authorization
    │
    ├─ freeze(NO_WINNER) -----------------X final authorization
    │
    └─ freeze(WINNER)
           │
           │ inspect_study() / binding verification
           ▼
       FinalEvaluationAuthorization
           │
           │ atomic one-shot publication
           ▼
       authorization.json
           │
           │ independent read-back + Study re-binding
           ▼
       verified authorization
           │
           └─ later subsystem may consume it
              to open exactly the preregistered final window
```

重要なのは、最後のconsumerはこのpackageには存在しないことである。authorizationとfinal-data materialization / final executionを別実装に保つことで、development中にunused dataへ到達する便利な経路を作らない。

## Publication

Authorization artifact rootは作成前に存在してはならない。publicationはsibling staging directoryへcanonical JSONを書き、fileをflush/fsyncした後にroot renameで一度だけ公開する。

Artifact rootは`authorization.json`一つだけを許可する。root/fileのsymlink、非regular file、余分なfile、malformed JSON、schema mismatch、authorization digest mismatchはfail closedする。

同じsemantic Studyを別filesystem pathへcopyしただけの場合、pathはidentityではない。binding authorityはStudyPlan / StudyFreeze / selected evidenceのsemantic digestである。

## Study immutability

Final authorizationをStudy rootへ追記しない。

`evaluation/experiments`はdevelopment Study lifecycleのauthorityであり、freeze後はimmutableである。final authorizationは別rootへ発行するread-only consumerとすることで、development evidence graphのdigestと履歴を変えない。

Authorization前後でStudy treeのfile contentが変わらないことをtest oracleに含める。

## Failure modes

次はすべてauthorization拒否またはinspection失敗にする。

- Studyが未freeze。
- `StudyOutcome.NO_WINNER`。
- WINNER freezeにselected evidence/strategyがない。
- selected evidenceがaccepted lineageに存在しない。
- final windowがdevelopment windowへ重なる。
- final windowが空または逆転している。
- authorization timestampがStudy freezeより前。
- output rootが既存、symlink、またはsymlink parentを経由する。
- artifactのJSON/schema/digestが壊れている。
- 別StudyPlan、別StudyFreeze、別winner evidence、別winner strategyへartifactを差し替える。

## Test Oracle

最低限、次を観測する。

1. WINNER Studyからだけauthorizationが発行できる。
2. publication後のread-backが同じcontractを再構築する。
3. Study rootの全file digestがauthorization前後で同一である。
4. 同じoutput rootへの二回目のpublicationが失敗する。
5. overlap / empty final windowが失敗する。
6. pre-freeze authorization timestampが失敗する。
7. JSON/body tamper、別freezeへのsubstitution、symlink outputが失敗する。
8. architecture testがfinal-test packageからdata/execution pathへの直接importを拒否する。
9. full repository CIで既存development experiment semanticsがGreenのままである。

## 次段階

将来のfinal evaluation実行は別のpreregistered subsystemとして実装する。

その段階でも、verified authorizationを入力にし、final Dataset identity、execution economics、winner implementation/runtime provenanceを固定し、winnerを一度だけ評価する。final結果をdevelopment Studyへ戻してthreshold・feature・strategyを再調整してはならない。
