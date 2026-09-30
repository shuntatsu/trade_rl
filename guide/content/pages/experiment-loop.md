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

新しい経済仮説やobservation、risk、execution semanticsを変えるStudyでは、結果を作る前にG0-G2の問い・機構・反証条件を固定し、freshなresult-blind AI reviewと独立oracleを通します。G3のevidence検証はこの前提審査の代わりになりません。条件が未確立の間はStudyPlanの準備までに留め、baselineやcandidateの経済結果は生成しません。

現行のStudy CLIは、外部reviewやoracleの完了を認証する自動gateを持ちません。G0-G2が閉じる前に経済実験を始めないことは、実行者が守るrelease prerequisiteです。

## 1. Studyを作る

Studyは、Dataset、baseline config、PPO seed方針、変更を許すControlled Factor、実験budgetなどの研究authorityを固定します。

final evaluationへ進める一般の新規Studyでは、execution economics、unused window、`StudyResearchContext`をbootstrap config v4へ事前登録し、context-bound StudyPlan v3へ固定します。PPO保有期間protocolはbootstrap config v5とStudyPlan v5を使い、protocolと明示baseline/riskも同じ事前登録へbindします。historical StudyPlan v1/v2へ後からfinal windowやcontextを追加することはしません。final startはdevelopment Datasetに含まれる最後のtimestampと申告済みconsumed-evidence scopeの両方より後でなければならず、Datasetには既に存在するがreplayでは未使用だった期間をfinalへ読み替えません。

後続Experimentが勝手に別Datasetや別execution条件へ移動できないようにします。

### PPOの中期保有期間を比較する場合

`ppo_holding_duration_v1`は、同じDataset・評価期間・費用・資金・リスク条件で、Observation v3を使うH=0 PPOと72 / 168 / 336 / 504本の1時間bar（3 / 7 / 14 / 21日）を比較します。5 seedと4つの保有期間を先にStudyPlanへ固定し、4期間すべて登録するまでcandidateを実行しません。各symbolは独立口座として採点し、StudyPlanには次のselection rule全体を結果前に保存します。

- H=0と全candidateのseed × symbol口座が終端決済後にフラットで、未約定注文がなく、各口座の実現最大DDが20%以下。
- seedごとにsymbolのafter-cost total returnを等重み平均し、その5 seedの中央値をprimary scoreとする。
- 同じseed内でH=0との差をsymbol平均してから5 seedの中央値を取り、正の場合にeligibleとする。
- eligibleの中でprimary score最大を選び、同点は短い期間。eligibleなしは`NO_WINNER`。

絶対returnが正かどうかはこのrelative development screenの追加条件にしません。4候補の選定は利益証明ではなく、winner候補にも別のone-shot sealed unused-future評価が必要です。現在の実装はこの事前設計とテストを整備中で、これらの期間のPPO学習・経済replay結果はまだありません。

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

PPO training layoutの比較では、`ppo_training_layout`と`ppo_rollout_steps_per_env`の組だけをControlled Factorとして変えます。SB3はwhole rollout単位で学習するため、requested timestep数だけでは公平な比較になりません。candidate evidenceに実際のtransition数を保存し、両条件で揃えて比べます。

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

freeze後もControlled Experiment Loop自身はunused futureへ触れません。結果前にunused windowをbindしたStudyPlan v3またはPPO保有期間用v5がWINNERになった場合だけ、別の `evaluation.final_test` 境界がStudyPlan・StudyFreeze・winner evidence・winner strategy・その事前登録windowをone-shot authorizationへbindします。authorization時に別windowへ差し替えることはできません。このauthorizationはfinal Datasetを取得せず、P&Lも計算しません。実際にunused futureを開く処理はさらに別のfuture consumerの責務です。

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
- final-eligibleなStudyではunused window自体をdevelopment resultより前に固定する。
- winner判断後までsealed unused-futureを開けない。
- WINNER後もauthorization発行とfinal Dataset/P&L実行を同じ機能へまとめない。

## 実験がGreenでもproduction認可ではない

Controlled Experimentで候補をacceptできても、それはdevelopment条件内での研究判断です。unused future data、execution stress、account-specific economics、capacity、live authorizationは別ゲートです。
