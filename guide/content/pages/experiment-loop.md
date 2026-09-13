## このページで答える問い

候補戦略を結果に合わせて何度も調整するのではなく、**一度に一つの変更要因だけを変え、結果を見る前に判断規則を固定する**にはどう進めるのか。

## 実験の状態遷移

```text
Studyを作る
    ↓
Baseline EvidenceSetを固定
    ↓
変更要因・仮説・判断ruleを事前登録
    ↓
Candidate EvidenceSetを1回生成
    ↓
CONTROLLED / INVALIDを検証
    ↓
raw evidenceから比較を再計算
    ↓
ACCEPT / KEEP / INCONCLUSIVEを決定
    ↓
研究終了時にWINNER / NO_WINNERとしてfreeze
```

途中でoperational failureが起きた場合は`FAILED`、一因子契約を破った場合は`INVALID`としてfail closedにします。

## 1. Studyを作る

Studyは、Dataset、baseline config、PPO seed方針、変更を許すControlled Factor、実験budgetなどの研究authorityを固定します。

後続Experimentが勝手に別Datasetや別execution条件へ移動できないようにします。

## 2. Baseline EvidenceSetを固定する

複数seedのCandidate Runをまとめ、baseline fingerprintを変更不能なevidenceとして残します。

candidateの比較対象は、このreachable lineage上のbaselineです。

## 3. 結果を見る前にExperimentを事前登録する

少なくとも次をcandidate実行より前に固定します。

- 仮説
- Controlled Factor
- candidate config
- baseline evidence binding
- formal target
- ACCEPT / KEEP / INCONCLUSIVEの判断rule

この段階ではcandidate returnを見ません。

## 4. Candidate EvidenceSetを生成する

同じStudy authorityの下でcandidateを実行します。候補だけ別のexecution accountingや別の評価期間へ切り替えません。

## 5. 一因子だけ変わったか検証する

`verify_controlled_delta`等の検証で、宣言したfactor以外の差分が混入していないことを確認します。

影響しないはずのstrategyについてraw returnsが変わっていないかも重要な反証です。

## 6. 生のreturnsから比較する

persist済みsummaryだけを信用せず、必要なfactor effect、total return、turnover、cost等をraw evidenceから独立再計算できるようにします。

## 7. 事前登録ruleで判断する

結果を見た後に「今回はこの指標なら勝ち」とruleを変えません。

```text
pre-registered evidence
        ↓
formal decision rule
        ↓
ACCEPT_CANDIDATE / KEEP_BASELINE / INCONCLUSIVE
```

side effectは開示しますが、formal targetを書き換える理由にはしません。

## 8. Studyをfreezeする

実験budgetを使い終え、未完了Experimentがなくなった段階でStudyをfreezeします。WINNERを選ぶ場合は、ACCEPT_CANDIDATEとして正当に到達したevidenceだけが候補です。

## FAILUREとINVALID

| 状態 | 意味 |
| --- | --- |
| `FAILED` | candidate evidence完成前のoperational failure。都合の良いrerunへ silently置き換えない |
| `INVALID` | 宣言したControlled Factor以外が変わる等、比較契約そのものが壊れた |

どちらも利益結果として解釈しません。

## Overfittingを防ぐ不変条件

- 一度に変えるfactorは一つ。
- candidate result前に仮説とdecision ruleを固定する。
- candidateだけDataset / scope / economicsを変えない。
- failureを別runで都合よく置換しない。
- raw evidenceとprovenanceを保持する。
- winner判断後までsealed final-testを開けない。

## 実験がGreenでもproduction認可ではない

Controlled Experimentで候補をacceptできても、それはdevelopment条件内での研究判断です。unused future data、execution stress、account-specific economics、capacity、live authorizationは別ゲートです。
