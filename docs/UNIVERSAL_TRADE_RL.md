# Universal Trade RL U0: Zero-shot Universe Isolation

Universal Trade RL U0は、銘柄横断RL研究を始める前に、**どの銘柄を学習に使い、どの銘柄を設計確認に使い、どの銘柄を最後のzero-shot Admissionに残すか**を、再現可能なArtifactとして固定するための境界です。

> **Production status: NO-GO**
>
> U0はRLを学習しません。収益性を証明しません。Production運用を認可しません。U0が証明するのは、Universeの役割分離、source-data identity、Train-only fit境界、Admissionのfail-closed access、およびそれらの再現可能なmaterializationです。

## 1. 目的

将来のUniversal Trade RLは、複数のTrain銘柄から**symbol-independentなBase RL**を学び、未学習銘柄へzero-shotで適用できることを最初の仮説とします。ただし、推論・評価・将来のDeployment単位は常に1つのconcrete instrumentです。

```text
one deployment candidate
  = one selected symbol
  = one fixed capital budget
  = one target-exposure policy application
```

BTC・ETH・SOLなどへ1つのActionで同時に資金配分するPortfolio policyをU0は定義しません。複数銘柄へ資本を配る場合は、銘柄ごとの独立した実行単位の外側にCapital allocation責務を置きます。

## 2. 固定する3つの役割

U0では利用可能な全source symbolを、次のいずれか1つへ完全に割り当てます。

| Role | 用途 | Fitへの利用 | Phase evaluation |
| --- | --- | --- | --- |
| Train | Base RL、normalization、calibration、threshold/reward estimationなどのfit source | **可** | Train phaseではevaluation scopeへ公開しない |
| Development | 設計判断、失敗分析、candidate selectionの開発用外部確認 | **不可** | Development phaseでDevelopmentのみ |
| Admission | frozen generationを最後に1回だけzero-shot確認するholdout | **不可** | authorization後、Admission phaseでAdmissionのみ |
| Excluded | データ不足など、事前に理由を明記した非対象symbol | 不可 | 不可 |

Train、Development、Admissionは非空・sorted・unique・pairwise disjointです。Excluded symbolも理由付きで明示し、他のroleと重複できません。

ExclusionをArtifactに残す理由は、結果を見た後で都合の悪い銘柄を黙って外すcherry-pickingを防ぐためです。除外理由そのものが妥当かどうかは別途研究判断ですが、少なくとも「何を外したか」はUniverse identityに固定されます。

## 3. Source identityとcomplete manifest

role configだけではUniverseを固定したことになりません。各symbolは次のsource-data identityにも結び付けられます。

- dataset SHA-256 digest
- first / last timestamp
- row count

`universe.json`は、利用可能なsource catalogの**全symbol**をroleまたはexclusionへ割り当てたcomplete manifestです。未割当の利用可能symbol、設定だけに存在してsourceがないsymbol、role overlap、source digest driftはfail-closedです。

Manifestは、entriesからrole configとsource catalogを再構築し、両方のdigestを再計算して検証します。したがって、Artifact digestだけを作り直してroleやsource identityを差し替えることは契約整合性を満たしません。

## 4. Phase firewall

U0 access contractはphaseごとに利用可能な集合を固定します。evaluation scopeは累積公開しません。

```text
TRAIN
  evaluate = none
  fit      = Train

DEVELOPMENT
  evaluate = Development
  fit      = Train

ADMISSION
  evaluate = Admission
  fit      = none
```

Admissionへ入るには、次の3つを一致させたauthorizationが必要です。

1. universe manifest digest
2. frozen generation digest
3. Selection evidence digest

DevelopmentやTrainの時点でAdmission authorizationを持ち込むことも拒否します。Admissionを開いた後でも`fit_symbols`は空で、新しいfit、normalization、calibration、threshold estimation、reward coefficient estimation、RL trainingはすべて禁止です。

Admission phaseで許されるのは、Admission銘柄を**frozen evaluation targetとして評価すること**と、Admission開封前にTrainだけから作成・固定されたmodel、normalization、calibration、fit provenanceなどを参照することです。Admission銘柄自身から新しい学習済み状態や統計量を作ることはできません。

Admission metadataは、dataset digestやtimestampなどの**integrity verification**には利用できます。しかし、その値をnormalization、calibration、threshold estimation、reward coefficient estimation、RL trainingその他のfitへ使うことはできません。

## 5. Train-only provenance

U0はfit artifactに、少なくとも次を結び付けます。

- Universe manifest digest
- fit purpose
- Train source symbols
- symbolごとのdataset digest
- knowledge cutoff

fit provenance builderはsource lookupより先にTrain-only scopeを検証します。したがってDevelopment / Admission symbolを「同じデータが存在するから」という理由でfitへ混入させることはできません。

現在U0で定義しているfit purposeは、feature normalization、calibration、threshold estimation、reward coefficient estimation、RL trainingを含みます。どのpurposeでもAdmission symbolはfitできません。

## 6. Versioned run identity

U0のrun identityはstageごとに必要なidentityを変えます。

| Stage | Universe | Model config | Fit provenance | Admission authorization |
| --- | --- | --- | --- | --- |
| `UNIVERSE_MATERIALIZATION` | 必須 | 禁止 | 禁止 | 禁止 |
| `BASE_TRAINING` | 必須 | 必須 | 必須 | 禁止 |
| `DEVELOPMENT_SELECTION` | 必須 | 必須 | 必須 | 禁止 |
| `ZERO_SHOT_ADMISSION` | 必須 | 必須 | 必須 | 必須 |

U0のidentityには将来のtransfer-specific fieldを先取りして入れません。transfer契約の追加は、後続U4でversioned identityとして扱う前提です。

## 7. Atomic materialization

CLIはstrict role configとstrict source catalogから、次の2 Artifactを同時にmaterializeします。

```text
output-root/
  universe.json
  identity.json
```

実行例:

```powershell
trade-rl-universe `
  --config examples/binance/universal-trade-rl-universe.example.json `
  --source-catalog examples/binance/universal-trade-rl-source-catalog.example.json `
  --output-root var/runs/universal-trade-rl-u0-example
```

同梱exampleのsymbol、digest、timestamp、row countは**契約形式を示すためのillustrative data**です。Production datasetの証拠ではありません。

Materializationはstaging directory内で両Artifactを完成・flush/fsyncした後、directory単位で1回だけpublishします。途中失敗時に`universe.json`だけが最終outputとして残ることはありません。

既存outputへの再実行は、2 Artifactがcanonical byte単位で完全一致するときだけidempotent successです。既存Artifactの編集、role config drift、source digest drift、余分なfileは自動修復せず拒否します。

成功時のterminal statusにも`production_status: "NO-GO"`を含めます。

## 8. U0が証明しないこと

U0の成功から、次を推論してはいけません。

- RL policyが学習できた
- zero-shot transferに経済的価値がある
- backtest / walk-forward / Admissionで利益が出る
- Execution modelが実市場を完全に再現する
- 実資金投入してよい
- Production認可済みである

U0は**研究デザインの汚染を防ぐ境界**であり、モデル品質や収益性そのものではありません。

## 9. U1へのhandoff

U1を開始する前提は、研究用exampleではなく、実際に使用するproduction-candidate dataについて次が揃うことです。

1. 事前に確定したTrain / Development / Admission / Excluded role config
2. 各symbolの実source identityを持つsource catalog
3. `trade-rl-universe`がそれらをfail-closedにmaterializeできること
4. 得られた`universe.json`と`identity.json`のdigestが、以後のrunで変更されず参照できること

Roleやdataset generationを変更した場合は同じ研究runを継続せず、新しいUniverse generationとしてmaterializeし直します。

U1以降の学習・Selection・Admissionは、このfrozen U0 identityを入力として進めます。U0のAdmission dataを見てからTrain/Development roleへ戻すことは、同一experimentのzero-shot契約を破ります。
