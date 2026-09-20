# Research assurance architecture

## 結論

Trade RLの研究変更は、結果の良し悪しを見る前に、**正しい問いを解いているか、意図した経済・市場メカニズムを定義できているか、そのメカニズムを実装が本当に満たしているか**を独立に確認する。

保証を次の6段階へ分離する。

1. **G0 Research Question Validity** — 研究目的、運用上の対象、経済仮説、反証条件が最終目的に整合しているか。
2. **G1 Mechanism Validity** — 資本、観測、行動、報酬、risk、execution、accounting、clockの意味が研究仮説を表現しているか。
3. **G2 Implementation Conformance** — sourceがG1の意味を実装し、独立oracle・反例・保存則に耐えるか。
4. **G3 Evidence Validity** — causal data、identity、会計、比較、provenance、result blindnessが有効か。
5. **G4 Development Economic Evidence** — development evidence上で事前定義した経済指標が改善したか。
6. **G5 Unused / Deployment Eligibility** — unused data、stress、hard risk、実運用制約で次段階へ進めるか。

**G0-G3はresult-blindで判定する。** P&L、return、Sharpe、winner判定その他の経済結果が良いことを、G0-G3の不備を救済する証拠にしてはならない。利益が出てもcausal invariantやmechanism contractへ違反すれば、その理由で失格である。逆に赤字でもG0-G3を満たすなら、失敗した仮説として有効な研究証拠になり得る。

この文書は「正しい思想」を自動的に証明する仕組みではない。誤った問い、意味の不一致、実装上の抜け道を**結果を見る前に反証可能にするためのcurrent architecture contract**である。

## Scope / non-goals

このcontractは、新しいproduction runtime、Study artifact schema、経済モデル、winner ruleを追加しない。既存のimmutable Study/Run/EvidenceSetを遡及的に書き換えず、過去の結果を別の判定へ再分類しない。

また、G0-G2の記録形式をただちに一つのmachine schemaへ固定しない。既存のpreregistration、StudyPlan、Issue/PRのresult-blind design record、architecture/contract testを使ってよい。ただし、経済結果を見た後に不足項目を都合よく追加して「事前に満たしていた」と扱ってはならない。将来machine-readable fieldへ昇格する場合はversioned schemaとして導入し、既存artifactを暗黙migrationしない。

## G0: Research Question Validity

研究を実行する前に、最低限次のResearch Question Contractを明示する。

- `objective` — 何を改善・検証する研究か。
- `operational_target` — 実運用上のどの意思決定・制約・収益源へ対応するか。
- `economic_hypothesis` — なぜedgeまたはrisk改善が存在し得るのか。
- `causal_story` — 誰の行動・制約・情報・需給からその機会が生じる想定か。
- `assumptions` — 成立に必要な前提。
- `known_limitations` — 最初から分かっている非現実性・代理変数・未観測要因。
- `falsifiers` — どの観測なら仮説を棄却・降格するか。
- `counterfactuals` — 仮説が正しい場合と誤っている場合を区別する比較。
- `primary_metric` — 事前に定義する主評価。
- `guardrail_metrics` — 主評価の改善だけでは許さないrisk/cost/quality条件。
- `development_use` — development dataを何の選択へ使うか。
- `unused_data_policy` — 何をunusedとして残し、何が一度見たらunusedへ戻らないか。
- `what_this_can_prove` — この研究から正当に主張できること。
- `what_this_cannot_prove` — Greenでも主張できないこと。

G0では「モデルを複雑にすれば利益が出る」「RLだから逐次意思決定に向く」のような手段を目的やedgeの根拠にしない。予測、execution、capital allocation、hedging、risk reductionのどこへ価値を出す仮説なのかを分ける。

## G1: Mechanism Validity

研究上の世界をMechanism Contractとして明示する。最低限、変更に関係する次のdimensionを固定する。

- `capital_model` — shared cashかindependent accountか、collateralをどう共有するか。
- `observation_scope` — per-symbolかjoint portfolioか、どの時点の状態を見られるか。
- `action_scope` — per-symbol actionかjoint portfolio actionか。
- `reward_scope` — per-symbol rewardかportfolio rewardか、cost/riskをどこへ含めるか。
- `risk_scope` — symbol localかportfolio globalか、hard riskの優先順位。
- `execution_scope` — 独立capacityか共有capacityか、注文競合をどう解くか。
- `accounting_scope` — independent cash booksかsingle shared ledgerか。
- `decision_clock` — decisionへ使える情報の締切。
- `execution_clock` — 注文がeligibleになる最初の時刻と遅延。
- `terminal_semantics` — terminal valuation、forced close、未約定残をどう扱うか。
- training/evaluation/deploymentでmechanismが一致する部分と、意図的に異なる部分。

**train/evaluation account semantics** が一致しないのに、同じ経済問題を学習・評価したと扱わない。例えばportfolio-wide shared-cash allocationを主張するなら、trainingが互いに独立なcash bookしか観測・制御しないことを無視しない。差を研究対象にするなら、その差自体をresult-blindに明示し、何を証明できないかを残す。

data availabilityでも意味を分離する。少なくとも、market eventの発生時刻、sourceが利用可能になった時刻、collectorが受信した時刻を同一視しない。event timestampがdecision以前であることはmarket-event causalityの一部を示すが、そのsourceが当時実際に取得可能だったhistorical point-in-time availabilityを単独では証明しない。

## G2: Implementation Conformance

G2は「実装を読んだ限り正しそう」ではなく、G1の意味を独立に反証する。

重要境界ではproduction functionをそのままtest oracleへ再利用しない。同じ誤解をproductionとtestが共有すると、自己一致だけでGreenになるためである。必要に応じて小さく遅い**independent oracle**、reference ledger、metamorphic test、conservation property、failure injection、differential replayを使う。

各semantic invariantは最低限次の形で記録する。

- **Statement** — 常に成立すべき意味。
- Scope — どのcomponent・状態遷移へ適用するか。
- **Counterexample** — 破る最小入力または状態。
- **Oracle** — どの独立観測でpass/failを決めるか。
- **Known limitations** — Greenでも証明しないこと。

example-based unit testだけで重要なmechanismを保証済みとしない。境界値、入力変換、時刻変更、複数注文、異常終了などを通じて、同じinvariantを別の形でも壊せないか確認する。

## Initial semantic invariant catalog

### CAUSAL-001 — temporal non-interference

**Statement:** 時刻 `t` までの利用可能情報が同一なら、`t` より後のmarket data、labels、depth、funding、他symbolのfuture pathを変更しても、`t` までのfeature、observation、action、submitted orderは変化してはならない。

**Counterexample:** future price/depthを極端値へpoisonしただけで過去のfeatureまたはdecisionが変化する。

**Oracle:** 同じprefixを持つ二つの入力を作り、futureだけを改変してprefix出力のbit/semantic equalityを比較するmetamorphic test。

**Known limitations:** market-event causalityを確認しても、source publication timingやhistorical point-in-time availabilityまでは証明しない。

### SOURCE-001 — availability authority is explicit

**Statement:** market `event_time`、historical `source_available_time`、live/forwardの`received_time`を、provider evidenceなしに同じ時刻として扱わない。decisionへ使用可能と主張する時刻authorityをdata sourceごとに明示する。

**Counterexample:** historical archive rowのmarket timestampを、そのarchiveまたはfieldが当時traderへ公開済みだった時刻として流用する。

**Oracle:** provider publication evidence、raw acquisition sidecar、request/receipt clock、または同等のpoint-in-time provenanceからavailability authorityを独立に再構築する。十分な証拠がなければhistorical point-in-time source availabilityを `NOT ESTABLISHED` とする。

**Known limitations:** sourceが当時取得可能だったことを確認しても、latency、queue、private account restrictions、実際のfill可能性までは証明しない。

### EXEC-001 — native capacity conservation

**Statement:** BASE_ASSETやCONTRACTSのnative liquidity capacityは、reference priceからquote notionalへ変換した後のfill price差だけを理由に増えてはならない。QUOTE_NOTIONAL capacityとは別のauthorityとして保持する。

**Counterexample:** 10 base unitsのavailable volumeをreference price 100で1000 quoteへ変換し、fill price 90で割ることで11.11 base unitsをfillできる。

**Oracle:** native quantity poolとquote poolを独立に計算し、複数orderを含む全fill合計が両方の適用poolを超えないことを確認するproperty test。これは**native capacity conservation**のoracleである。

**Known limitations:** displayed/recorded volume capacityを守ることはqueue position、hidden liquidity、実際のlive fillを証明しない。

### ACCOUNT-001 — economic flow exactly once

**Statement:** fill、fee、funding、borrow、cash interest、realized/unrealized P&L、terminal settlementは、それぞれ定義されたchannelと時刻で一度だけbookへ反映される。

**Counterexample:** execution priceに含まれたcostを再度feeとして控除する、または同じfunding settlementを再play時に二重計上する。

**Oracle:** production executorとは独立した最小reference ledgerへ同一event列を与え、eventごとのquantity、cash、equity、cost channelを比較する。

**Known limitations:** 会計恒等式の一致は、入力price/fill自体が市場で実現可能だったことを証明しない。

### PORTFOLIO-001 — coherent shared state

**Statement:** portfolio-wide制約を主張する経路は、一つの整合したpre-execution account stateに対して全proposalを評価し、shared cash/gross/margin/capacityを競合込みで適用する。

**Counterexample:** 二つのsymbolが同じcashをそれぞれ独立に全額使用した後、結果だけをportfolioとして合算する。

**Oracle:** 単独では通る複数proposalを同時投入するsynthetic counterexampleで、aggregate allocationがglobal boundを超えないことを確認する。

**Known limitations:** constraint整合性はpolicyが良いallocationを学習することを証明しない。

### TRAIN-EVAL-001 — train/evaluation account semantics

**Statement:** trainingとevaluationが同じ経済能力を比較すると主張する場合、資本・観測・action・reward・risk・execution/accounting scopeは同じ意味を持つか、差分が事前に研究factorとして明示されていなければならない。

**Counterexample:** trainingはper-symbol independent cashで各環境が同じbudgetを使える一方、evaluationだけsingle shared-cash gross limitを適用する。

**Oracle:** 同一synthetic proposal/observation sequenceをtraining adapterとevaluation referenceへ与え、Mechanism Contract上のstate transition・constraint semanticsを比較する。

**Known limitations:** semantic parityはRLのcredit assignment、optimization安定性、収益性を保証しない。

### RISK-001 — hard risk priority

**Statement:** hard riskによる縮小・拒否・終了はstrategyの利益期待より優先され、economic controllerがriskを迂回してはならない。

**Counterexample:** strategyが高い期待returnを出したため、既にbreachしたhard limitを無視してpositionを維持・増加する。

**Oracle:** breach状態へ利益方向のactionを注入し、risk transitionがstrategy preferenceと独立に適用されるfailure-injection test。

**Known limitations:** rule enforcementはrisk threshold自体の妥当性や未知のtail riskを証明しない。

### EVIDENCE-001 — immutable reconstructible evidence

**Statement:** 有効な研究evidenceはexact source/runtime/config/data identityへbindされ、publication後の書換え、都合の良いpartial selection、別implementationへのすり替えを許さない。

**Counterexample:** resultを見た後にartifactの一部を置換しても同じStudy/EvidenceSetとして読める。

**Oracle:** digest、schema、lineage、fresh reconstruction、tamper injectionでfail-closedを確認する。

**Known limitations:** artifactが真正であることは、その研究仮説や経済edgeが正しいことを証明しない。

### RESEARCH-001 — one-way evidence use

**Statement:** 一度development上の選択、threshold調整、候補選択、結果解釈へ使った市場期間は、Study名、seed、model、branchを変えてもunused dataへ戻らない。

**Counterexample:** 既知の期間で複数候補を試した後、名前だけ変えた最終候補を同じ期間で「初見」と扱う。

**Oracle:** Experiment Ledger / Study lineage / data-scope recordから、各期間がどの意思決定に使われたかを追跡し、unused authorization前に照合する。

**Known limitations:** 未使用期間を守っても単一final windowだけで将来の普遍的収益性は証明できない。

## G3: Evidence Validity

G3は既存のcausal data、common accounting、immutable artifact、provenance、controlled factor、lineage、result-blind preregistrationをまとめて「その結果を研究判断へ使ってよいか」を問う。

G2との違いは、G2が**実装したmechanismそのものの意味**を検証するのに対し、G3はそのmechanismを使って得た**具体的evidenceの由来・比較・保存**を検証する点にある。

Study/ExperimentのverificationがGreenでも、G0-G2を自動的に証明したことにはならない。

## G4: Development Economic Evidence

G4で初めてreturn、drawdown、turnover、cost、profit concentration等の経済結果を主要な判断材料としてよい。

developmentで改善しても、それは「次の研究参照候補」または事前定義したdevelopment decisionの意味しか持たない。G0-G3のfailureを利益で上書きしない。またdevelopment winnerをproduction/live suitabilityへ昇格しない。

## G5: Unused / Deployment Eligibility

G5はunused data、複数期間・複数symbol、stress、hard risk、execution realism、shared-capital feasibility、運用上のobservabilityとfailure handlingを確認する。

G5のpass条件は対象研究で事前定義する。development上の高い利益だけでG5をskipしない。paper/live forward evidenceが必要な場合、historical replayと別のauthorityとして扱う。

## Assurance Matrix

研究・実装・reviewの最終報告では、該当gateを少なくとも `PASS` / `FAIL` / `NOT ESTABLISHED` / `NOT APPLICABLE` で分離する。総合的な「検証完了」だけで未確認領域を隠さない。

例:

```text
Research question defined: PASS
Economic mechanism stated: PASS
Falsifier defined before result: PASS
Market-event causality: PASS
Historical point-in-time source availability: NOT ESTABLISHED
Train/eval mechanism parity: PASS
Accounting independent oracle: PASS
Execution conservation: PASS
Hard-risk semantics: PASS
Economic edge: NOT ESTABLISHED
Unused-data robustness: NOT ESTABLISHED
Production eligibility: NOT ESTABLISHED
```

このmatrixはscoreではない。上位gateのFAILを下位gateの利益で相殺せず、未検証事項を0点のように平均もしない。

## Controlled Experimentとの関係

`architecture/controlled-experiment-loop.md` のStudy state machineは主にG3-G4を厳密に扱う。新しい経済仮説、data availability semantics、account model、observation/action/reward contract、execution/accounting semanticsを変更する場合、Studyを回す前に本contractのG0-G2を確認する。

`verify_experiment` がcontrolled factor以外の差を拒否することは重要だが、研究問題そのものが妥当か、training/evaluation mechanismが一致するか、production implementationが意味どおりかを単独では証明しない。

既存のimmutable Study/EvidenceSetはこの文書の追加によって改変しない。新しいassurance requirementで過去evidenceの意味を拡大せず、必要なら新しいStudy/verification lineageを作る。

## Change routing

- economic hypothesis、causal story、primary metric、unused policyを変える → G0を再確認する。
- capital/observation/action/reward/risk/execution/accounting/clock semanticsを変える → G1とG2を再確認し、既存Study fixed semanticsと両立しなければ新Studyにする。
- production implementationだけを変更する → 対応するG2 invariantとfailure modeをREDから再検証する。historical evidenceはexact旧implementationの証拠として保持する。
- data source / availability / timestamp meaningを変える → G0-G3を再確認し、market-event causalityとhistorical source availabilityを区別する。
- comparison threshold / winner ruleを変える → resultを見た後の救済にせず、G4 protocolとして新しい事前定義を作る。
- final/unused/live eligibilityを変える → G5 authorityを更新し、development evidenceを代替にしない。

## Completion condition

「正しい仕組みを確認した」と報告するには、対象変更に関係するG0-G3について、対応するcontract、semantic invariant、反例、oracle、Known limitationsを示す。test Green、CI Green、利益のどれか一つだけを全体保証の代わりにしない。

問題をその場で修正できる場合は、反例をpermanent regression/property testとして残し、最小修正、再テスト、独立oracle、final diff/CIまで再確認する。
