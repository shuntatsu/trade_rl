# Research Assurance Gate

## 目的

Research Assurance Gate は、実装・CI・再現性の検証より上位で、**研究設計そのものが最終目的に対して意思決定可能な証拠を作る設計になっているか**を確認する。

Trade RL の最終目的は、causal / point-in-time data、realistic execution cost、common accounting、hard risk、複数銘柄・複数期間、unused evidence、robustness / stress の下で再現可能な実運用利益を得ることである。

このGateは「思想が真である」と自動判定しない。機械検証が保証するのは、必要な論点が明示され、claim越権やstale reviewがなく、独立したadversarial reviewがexact authorityへbindしていることまでである。研究仮説の妥当性はreviewerが反証を試みたうえで明示的に判断する。

## 4つの検証層

### 1. Objective / epistemic validity

経済実行前に少なくとも次を固定する。

- この研究が最終的にどの意思決定を支援するか。
- falsifiableなhypothesisとeconomic mechanism。
- counter-hypothesis / plausible alternative explanation。
- current evidenceに対して何のinformation gainがあるか。
- 最も安価なdecisive falsifier。
- stop rule。
- discovery / development / unused validation / prospective paper / production のどのstageか。
- metric / gateが最終目的のproxyとして妥当な理由。
- この研究からは何を主張できないか。

「結果を見る前にpreregisterした」ことだけでは十分ではない。間違った問いを正確にpreregisterしても意思決定品質は上がらない。

### 2. Mechanism validity

研究対象を次のchainで説明できなければならない。

```text
source
-> availability
-> feature/state
-> model/strategy
-> intent
-> order
-> fill
-> accounting
-> evidence
-> decision
```

高影響boundaryでは、unit、event/availability time、requested/realized state、sign、risk、execution、accounting、evidence identityのauthorityを一意にする。

変更した経済boundaryには、production codeと同じ計算を写しただけではないcounterexample、metamorphic test、state-machine invariant、独立算術のいずれかを持つ。

Issue #667 はこの層のruntime/economic/state semanticsとcanonical connectivityを監査する。Research Assurance Gateは#667を置き換えず、「そのmechanismを検証対象に選ぶこと自体が適切か」を追加で問う。

### 3. Evidence sufficiency

予定された証拠がhypothesisを判別できるかを結果前に確認する。

最低限、claimに応じて次を扱う。

- causal / point-in-time sourceとavailability。
- fit / development / unused / finalの境界。
- common accounting、realistic cost / capacity。
- hard risk、terminal state、actual fills。
- claimに十分なsymbol / period / seed coverage。
- cash / exposure / beta / trivial strategy等のalternative explanationを露出できるcontrols。
- result前に固定したrobustness / stress。
- byte reproductionだけでは足りない場合の独立economic/state reconstruction。
- 同じdevelopment evidence上でのresult-dependent rescueを禁止するstop rule。

証拠を再生成できることと、証拠が研究命題を識別できることは別である。

## Claim validity

Research decisionは必ず次へ写像する。

```text
evidence
-> decision rule
-> permitted claims
-> forbidden claims
-> next authorized action
```

主張レベルは次の順で強くなる。

1. `software_validity`
2. `mechanism_validity`
3. `relative_improvement`
4. `development_profitability`
5. `unused_validation`
6. `prospective_paper`
7. `production`

弱いevidence levelから強いclaim levelへ昇格しない。特に、

- CI Greenはeconomic evidenceではない。
- relative improvementはabsolute profitabilityではない。
- development profitabilityはunused-data validityではない。
- unused-data validityはprospective execution evidenceではない。
- prospective paper evidenceも自動的なlive authorizationではない。

## Machine record

Repository toolingはstrictな`research_assurance_v1` recordを扱う。

```bash
uv run python -m tools.agent_repo assurance digest assurance.json
uv run python -m tools.agent_repo assurance check assurance.json
```

`digest` はreview envelopeを除くauthored research contractへbindする。`check` の状態は次だけである。

- `UNREVIEWED`: 構造・claim境界は有効だが独立reviewがない。economic executionをauthorizeしない。
- `BLOCKED`: contract不備、claim越権、stale review、またはreviewerが研究設計をrejectした。authorizeしない。
- `PASS`: strict contractが有効で、独立reviewがexact record digest、protocol HEAD、implementation HEADへbindしてPASSした。

項目を全部埋めただけでは`PASS`にならない。

Generic exampleは `tools/agent_repo/examples/research_assurance.json` に置く。これはtask-specificなanswer keyではなく、意図的に`UNREVIEWED`の例である。

## Independent adversarial review

recordを作成したAgentと、最終review stepを論理的に分離する。

reviewerは少なくとも次を反証する。

- hypothesis以外の説明で同じPASSが出ないか。
- metric/gateがproxyだけを最適化していないか。
- cheap falsifierを先に実行できないか。
- discovery gateとdeployment gateを混同していないか。
- data scope、controls、stressがclaimに十分か。
- deterministic decision ruleから意図したclaimが論理的に導けるか。
- evidence reconstructionが同じbug/assumptionを共有していないか。

reviewはrecord digest、protocol HEAD、implementation HEADへbindする。いずれかが変われば以前のPASSはstaleであり再reviewする。

## Repository workflow

新しいeconomic research protocol、active protocolのmaterial semantic change、evaluation/decision ruleの変更では、economic execution前にResearch Assurance Recordを作成し、independent reviewを通す。

通常のsource implementationでは従来どおりTDD、#667型semantic audit、full CIを行う。Research Assurance Gateはそれらに**追加**される。

推奨順序:

```text
research question
-> assurance draft
-> cheapest falsifiers
-> adversarial assurance review
-> implementation TDD
-> semantic/mechanism audit
-> exact-head full CI
-> result-blind activation
-> economic execution
-> fresh reconstruction / audit
-> bounded claim review
```

material mechanism changeが入った場合、実装CIだけを再利用してeconomic runへ進まない。assurance review authorityもstaleとして扱う。

## Failure Modes

Gate自体も次をfail closedに扱う。

- counter-hypothesisやcheapest falsifierがない。
- stop ruleやlimitationsがない。
- mechanism chainまたはauthority ownerが欠ける。
- economic evidenceなのにPIT/common accounting/cost/risk/terminal boundaryが明示されない。
- profitability claimなのにsymbol/period/robustnessが不足する。
- evidence levelより強いclaimを宣言する。
- development stageでproduction/liveをauthorizeする。
- review digest / protocol HEAD / implementation HEADが現recordと一致しない。
- reviewなしでexecution authorizationを要求する。

## 保証しないこと

このGateが`PASS`でも、strategyがprofitを出すこと、市場仮説が真であること、production deploymentが安全であることは保証しない。

`PASS`が意味するのは、**現時点の研究命題・mechanism・証拠設計・claim境界について独立反証reviewを通り、そのexact authorityの下で次の研究行為を実行してよい**ということだけである。
