# Final evaluation authorization

## 結論

`trade_rl.evaluation.final_test` は、development Studyが `StudyOutcome.WINNER` でfreezeされた後にだけ、**unused-future / final evaluationを開く資格**を一度だけ封印する境界である。

このpackage自身はfinal Datasetを取得・構築せず、strategyをfit/inferenceせず、Replay/P&L/stressを実行せず、Production/liveを認可しない。役割は「developmentで確定したwinnerと、まだ開いていないfinal windowをimmutable authorizationへbindする」ことだけである。

したがってauthorization artifactの存在は、profitability、final-test成功、Production readiness、live order routing認可を意味しない。

## Objective

Developmentとunused/final evaluationの間にhard gateを置く。

- Studyはauthorization前にterminal `WINNER` でfreezeされている。
- `NO_WINNER`、未freeze、途中Studyはauthorizationできない。
- winner EvidenceSet digestとwinner strategyは`StudyFreeze`と一致し、accepted lineageに存在する。
- final windowはdevelopment windowと重ならず、空でない。
- authorization発行はfrozen Studyを一切変更しない。
- authorizationはStudy外の別rootへone-shotでpublishする。
- read-back時にStudyPlan、StudyFreeze、winner evidence、winner strategy、window、authorization digest、canonical artifact bytesを再検証する。

## Preregistration chain

final-eligibleな新規research lineは、unused windowをdevelopment結果の後に選ばない。authority chainを次の順序で固定する。

```text
CanonicalM2BootstrapConfig v3
  final_evaluation_start / final_evaluation_stop_exclusive
        ↓ bootstrap config digest
StudyPlan v2
  canonical nanosecond final window
        ↓ StudyPlan digest
StudyFreeze(WINNER)
        ↓ freeze digest + accepted winner lineage
FinalEvaluationAuthorization
```

`canonical_m2_bootstrap_config_v3` はexecution economicsに加えてfinal windowを結果前configへbindする。bootstrapはそのwindowを `controlled_study_plan_v2` へcanonical nanosecond表現で移し、inspection時にもconfigとPlanの一致を再計算する。 `controlled_study_plan_v2` を作る時点でも、`final_evaluation_start` はdevelopment Datasetに含まれる全timestampより厳密に後でなければならない。development replayで未使用でもDataset内に既に存在する期間はunused/finalとは扱わない。

authorization時のcallerはwindowを選択できない。public authorization APIは `final_evaluation_start` / `final_evaluation_stop_exclusive` を引数に持たず、StudyPlan v2にpreregisterされたwindowだけからartifactを構築する。

historical bootstrap v1/v2 と `controlled_study_plan_v1` はread/inspection互換のまま保持するが、final windowを後付けしない。final windowを持たないlegacy Studyが後からWINNERになってもfinal authorization対象にはならない。final-eligibleな研究を行う場合は、結果前に新しいv3 bootstrap / v2 StudyPlanを作る。

## Non-goals / dependency boundary

このpackageは次を所有しない。

- final Datasetのdownload/build/cache。
- provider/exchange access。
- strategy fit / inference。
- Replay、execution、accounting、P&L。
- robustness / stress / capacity評価。
- account-specific execution economics calibration。
- Production/live authorization。

そのためarchitecture testは `trade_rl.evaluation.final_test` から `data`、`integrations`、`strategies`、`evaluation.replay`、`evaluation.runs`、`evaluation.robustness` への直接importを拒否する。

## Authorization contract

`FinalEvaluationAuthorization` は少なくとも次をbindする。

| Field | Authority |
|---|---|
| `study_digest` | frozen development `StudyPlan.digest` |
| `study_freeze_digest` | terminal `StudyFreeze.digest` |
| `winner_evidence_digest` | `StudyFreeze.selected_evidence_digest` |
| `winner_strategy` | `StudyFreeze.selected_strategy` |
| `development_evaluation_stop_exclusive` | frozen baseline config |
| `final_evaluation_start` | preregistered unused window |
| `final_evaluation_stop_exclusive` | preregistered unused window |
| `authorized_by` | authorization actor |
| `authorized_at` | aware authorization timestamp |
| `schema_version` | persisted contract version |

window timestampはcanonical nanosecond表現を使う。低層authorization contractではfinal startはdevelopment replay stopより前であってはならず、final stopはstartより厳密に後でなければならない。さらにfinal-eligibleなStudyPlan v2の作成時にはfinal startがdevelopment Datasetの最後のtimestampより厳密に後であることを要求する。authorization timeはbound Studyのfreeze時刻より前であってはならない。

## State transition

```text
bootstrap v3 / StudyPlan v2
    |
    | preregistered unused window
    v
Development Study
    |
    +-- unfrozen --------------------------X authorization
    |
    +-- freeze(NO_WINNER) -----------------X authorization
    |
    +-- freeze(WINNER)
           |
           | inspect_study / accepted-lineage verification
           v
       FinalEvaluationAuthorization
           |
           | canonical bytes + one-shot publication claim
           v
       authorization.json
           |
           | independent read-back + Study re-binding
           v
       verified authorization
           |
           +-- later subsystem may consume it
               to open exactly the authorized unused window
```

最後のconsumerはこのpackageには存在しない。authorizationとfinal-data materialization / economic executionを別実装へ分離し、development中にunused dataへ到達する便利な経路を作らない。

## Publication / concurrency

output rootは発行前に存在してはならず、Study rootの外側に置く。root、ancestor、artifact fileでsymlink traversalを許さない。

publicationはprivate sibling staging directoryへcanonical JSONを書き、fileをflush/fsyncした後、final root名に対するsibling claimをexclusive createで取得してからrenameする。同じsubsystemを使う同時publisherのうちclaimを取得できるのは一つだけである。成功したpublisherはclaimをbest-effortで除去する。

publication途中に失敗し、authorityが曖昧になり得る場合は安全側に倒す。stale claimが残ると再発行を止めることがあるが、unused futureを二重に開くよりfail closedを優先する。claimは同期primitiveでありresearch evidenceではない。

rename競合・既存targetは生のfilesystem例外として外へ漏らさず、one-shot conflictとして扱う。final rootは`authorization.json`一つだけを許す。

## Artifact immutability

artifactはsemantic JSONが同じだけでは足りない。writerが生成したcanonical JSON bytesそのものをimmutable authorityとする。

inspectionは次を拒否する。

- malformed JSON。
- schema/key mismatch。
- authorization digest mismatch。
- semantic内容が同じでもwhitespace/key order等を変えたnon-canonical byte rewrite。
- root/file symlinkおよびsymlink ancestor経由。
- extra file。
- 別StudyPlan / StudyFreeze / winner evidence / winner strategyへのsubstitution。

同じsemantic Studyを別filesystem pathへcopyしただけの場合、path自体はidentityではない。binding authorityはStudyPlan / StudyFreeze / selected evidenceのdigestである。

## Study immutability

Final authorizationをStudy rootへ追記しない。

`evaluation/experiments` はdevelopment lifecycleのauthorityであり、freeze後はimmutableである。final authorizationは別rootのread-only consumerとしてStudyをinspectionするだけである。

test oracleはauthorization前後でStudy treeの全file bytes/digestが同一であることを確認する。

## Failure modes / Test Oracle

少なくとも次をpermanent testで反証する。

1. WINNER Studyだけがauthorizationできる。
2. NO_WINNER / unfrozenを拒否する。
3. legacy StudyPlan v1などpreregistered final windowを持たないStudyを拒否する。
4. public authorization APIがfinal-window override引数を持たないことを固定する。
5. final startがdevelopment Dataset内に残るStudyPlan v2を作成段階で拒否する。
6. development overlap / empty final windowを低層contractでも拒否する。
7. pre-freeze authorization timestampを拒否する。
8. publication後のread-backが同じcontractを再構築する。
9. Study bytesがauthorization前後で不変である。
10. 二回目の同一root publicationを拒否する。
11. concurrent publicationで成功者を一つに限定する。
12. rename直前に別publisherがtargetをclaimしたraceをone-shot conflictとして拒否する。
13. JSON/body tamper、non-canonical rewrite、別Study substitution、symlink root/ancestorを拒否する。
14. packageがdata/execution pathをimportしない。
15. full repository CIでdevelopment Study semanticsがGreenのままである。

## Research assuranceとの関係

これはG4のwinner選択を行わず、G5へ進む**data-use authorization capability**である。development P&Lの良さを理由にWINNER/freeze/lineage条件をskipしない。

final execution protocol、stress、Dataset identity、execution economics、winnerの使用方法を将来変更する場合は、そのconsumer側でresult-blindなG0/G1 semantic reviewとG2 machine evidenceを新たに満たす。AI reviewだけでauthorization artifactやmachine bindingを代替しない。

## 次段階

final evaluationの実行は別のpreregistered subsystemとして実装する。

そのconsumerはverified authorizationを入力にし、authorized window以外を読まず、final Dataset identity、winner implementation/runtime provenance、execution economics、stress contractを固定して一度だけ評価する。final結果をdevelopment Studyへ戻してthreshold・feature・strategy・seed policyを再調整してはならない。
