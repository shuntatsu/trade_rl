# Japanese Code Explorer Design

Status: Active

## 結論

Interactive Human Guideを、抽象概念を眺める説明UIから、**Trade RLの実装を日本語で追跡できるコード連動型ガイド**へ再設計する。

画面上の説明・見出し・図ラベル・状態名は日本語を主表示とし、実際のPython識別子・型名・ファイルパスは照合用の補助情報として併記する。Python sourceの関数名・変数名・公開API自体は日本語へ改名しない。

実装理解の主経路は、概念説明ではなく、実コードの呼出し・状態遷移・データ変換を辿る対話型の図とする。最初のコード連動経路は `trade_rl.evaluation.replay.run_single_symbol_replay`、次にPPO学習・推論経路を扱う。

## Objective

初見の開発者がGitHub Pagesだけを入口にして、次を実コードまで追跡できる状態を作る。

1. 市場データがどこから来るか
2. どこで特徴量・availability・stalenessを構築するか
3. strategy/PPOが何を観測し、何を返すか
4. strategy intentがhard riskでどのように制約されるか
5. `MarketExecutor + BookState` がfill/cost/accountingをどこで処理するか
6. replay結果がreturns/diagnostics/evidenceへどう接続されるか
7. その説明を裏付けるsourceとtestがどこにあるか

## Non-goals

- `trade_rl/**` のPython識別子を日本語へrenameしない。
- runtime packageの責務、依存方向、strategy/risk/execution semanticsを変更しない。
- Guideをarchitecture/research authorityへ昇格しない。
- GitHub上で任意sourceを実行するIDEや完全なAPI referenceを作らない。
- Python ASTからcall graphやsemantic dependencyを推測してsource truthとして表示しない。
- 全関数を一枚へ詰め込む巨大graphを作らない。
- force-directed layoutを使わない。

## Current problem

現行Guideには検索、テーマ、可視化、source fingerprint、accessibility/E2E gateがあり、基盤は再利用できる。一方で次の問題がある。

- 日本語文中に英語専門語が密集し、意味理解より用語解読が先に必要になる。
- topic schemaは概念説明中心で、関数・クラス・変数・入力・出力・例外・不変条件・関連テストを表現できない。
- 図を選択しても実ファイル・実symbolへ降りる経路がない。
- `ArchitectureFlow.tsx` は `visualization.edges` を描画せず、`nodes` 配列順に矢印を出すため、data modelと異なる関係を表示し得る。
- `FlowStepper` はactor間message順序や状態変化を表すsequence diagramではない。
- Markdown section fingerprintはあるが、Guideが参照するPython symbol・主要変数の存在やsource変更をfail-closedに検出できない。

## Information architecture

最上位ナビゲーションは「概念カテゴリ」ではなく「理解したいこと」に寄せる。

1. **まず全体をつかむ** — 市場データ → データセット → 戦略 → リスク → 約定・会計 → 評価
2. **1本のバーを追う** — single-symbol replayのruntime sequence
3. **PPOを追う** — 観測、action、risk、execution、reward、学習、frozen strategy replay
4. **コード地図** — package responsibilityと主要owner symbolへのdrill-down
5. **データと特徴量** — causal / availability / staleness / Dataset identity
6. **約定・会計** — cost/fill/funding/borrow/liquidationのauthority
7. **実験と証拠** — Run / Study / EvidenceSet / comparison / freeze
8. **現在の研究状態** — current-statusの非正本要約

既存topicの有用な内容はこの構造へ移す。同じ情報を図・カード・本文で三重表示しない。

## Language policy

### 日本語を主表示するもの

- ナビゲーション
- topic title / summary
- section heading / body
- diagram actor/node/message label
- input/output/state/errorの説明
- search resultの主見出し
- accessibility label

### 照合用に残すもの

- Pythonの関数名・クラス名・変数名
- module/file path
- package名
- `PPO`, `SHA-256`, `JSON`, `NPZ` 等の固有名
- protocol/schema/version literal

表示例:

```text
希望保有数量
desired_quantity

リスク制約を適用
PreTradeRisk.constrain()
```

日本語名は説明上のaliasであり、source identifierを置換しない。

## Code-linked content model

### Build-time symbol index

`guide/tools/code_symbols.py` が `trade_rl/` をPython ASTで静的解析し、production moduleをimport/executeせずに `guide/.generated/code-symbols.json` を生成する。生成物はtracked sourceへcommitしない。

各symbolは最低限次を持つ。

```ts
type CodeSymbol = {
  qualified_name: string;
  kind: "function" | "class" | "method";
  path: string;
  start_line: number;
  end_line: number;
  signature: string;
  source_sha256: string;
  local_names: string[];
};
```

`local_names` は関数・methodのparameterと同一scope内の代入対象を含む。nested function/classのlocalは親symbolへ混ぜない。

### Reviewed code reference

人間がレビューするGuide contentはtopic直下に `code_references` を持つ。

```ts
type CodeReference = {
  id: string;
  symbol: string;
  source_sha256: string;
  label_ja: string;
  description_ja: string;
  variables?: Array<{
    name: string;
    label_ja: string;
    description_ja: string;
  }>;
  tests?: string[];
};
```

source-checkは次をfail-closedに検証する。

- `symbol` が完全修飾名で一意に存在する
- symbol kindが期待contractと一致する
- `source_sha256` がcurrent sourceと一致する
- `variables[].name` が対象symbolの `local_names` に存在する
- test pathがrepository root外へescapeせず、実在する

コード変更でsource digestが変わった場合は、説明を読み直したtopicだけ明示的にrefreshする。digest更新だけで内容レビューを省略しない。

## Source revision / source links

Pages上のsource linkはfloating `main` ではなく、**そのPages buildが説明しているexact repository revision**へ固定する。

生成index metadataは `source_revision` を持つ。CI/deploymentではcheckoutしたexact SHAを渡し、source URLは次の情報から決定的に構築する。

- repository URL
- `source_revision`
- `CodeSymbol.path`
- `start_line` / `end_line`

これにより、後からmainが進んでも表示中の説明とsource line rangeがずれない。

## Primary sequence: single-symbol replay

`trade_rl.evaluation.replay.run_single_symbol_replay` を最初のコード連動sequence diagramとする。

actor:

1. `MarketDataset` — 市場データセット
2. `run_single_symbol_replay` — リプレイ統括
3. `SingleSymbolStrategy` — 戦略
4. `PreTradeRisk` — ハードリスク
5. `MarketExecutor` — 約定エンジン
6. `BookState` — 帳簿状態
7. `SingleSymbolReplayResult` — 評価用結果

source順序と一致させる主要message:

1. `_observation(...)` でcausal observationを構築
2. `strategy.decide(observation)` で `PositionIntent` を取得
3. intent変更時に `target_weight_for_intent(...)` と `_desired_quantity_from_weight(...)`
4. `_weight_for_desired_quantity(...)` でquantity-preserving holdをweightへ再表現
5. `risk_controller.constrain(...)` でhard risk projection
6. 必要時にrisk後targetから `desired_quantity` を更新
7. `executor.execute_interval(...)` でfill/cost/accounting
8. `book`, `returns`, `current_intent`, `index` を更新
9. terminationを評価
10. `ExecutionDiagnostics` と `SingleSymbolReplayResult` を構築

message選択時のinspectorは、日本語での役割、実symbol、exact-revision source path/line、入力、出力、主要変数の日本語alias、不変条件/fail-closed条件、関連testを表示する。

## PPO sequence

PPOはObservation説明だけで終わらず、trainingとfrozen replayの境界を見せる。

主sequence:

1. Dataset / feature index validation
2. `PPOTradingEnv` 構築
3. Observation v2構築
4. PPO policy action
5. logical intent / target exposure
6. hard risk
7. `MarketExecutor.execute_interval`
8. `BookState` / reward更新
9. PPO update
10. fitted/frozen strategy
11. independent per-symbol replay

Observation v2は次の5segmentを日本語主表示にし、source field名を併記する。

- 選択したローカル特徴量
- 利用可能/有限マスク
- 正規化staleness
- 現在の売買意図
- 現在のウェイト

symbol IDとdataset-global featureを含めないことも明示する。

## Architecture / code map semantics

「コード地図」の矢印が何を意味するかを曖昧にしない。

v1のtop-level mapは**静的import graphではなく、review済みの責務間主処理フロー**を表す。edgeにはrelationを明示し、配列順から意味を推測しない。

```ts
type CodeMapEdge = {
  from: string;
  to: string;
  relation: "data-flow" | "calls";
  label_ja: string;
};
```

- `data-flow`: architecture docsでレビューされた主要データ/処理の流れ
- `calls`: Guideで明示的にレビューした主要call relation

自動AST解析でcall graphを生成しない。package/file/symbolのdrill-downはCodeReferenceへ接続し、実在性とsource freshnessを別contractで保証する。

Graph familyはdirectional DAGを基本とし、deterministic layered layoutを使う。既存規模では新しいlayout libraryを追加せず、純粋関数でrank/orderを決定する。実際にbranching/edge routing要件を満たせないことをテストで確認した場合だけ外部layout dependencyを再検討する。

## Sequence model

sequenceはactor/messageを明示modelとして持つ。

```ts
type SequenceActor = {
  id: string;
  label_ja: string;
  code_ref?: string;
};

type SequenceMessage = {
  id: string;
  from: string;
  to: string;
  label_ja: string;
  code_ref?: string;
  state_changes_ja?: string[];
};
```

unknown actor/code reference、duplicate idはparse時にfailする。message配列順だけは実行順序として明示contractにする。

## Interaction design

### Desktop

- 中央: sequence / code map
- 右: 選択中symbolのinspector
- 上部: 日本語/コード名共通検索
- node/messageはクリックとkeyboardで選択可能
- 選択pathだけを強調し、未選択edgeは静かに表示
- resetで全体表示へ戻る

### Mobile portrait

- diagram全体を読めない大きさへ縮小しない
- sequenceは1 stepずつ「前へ / 次へ」で移動できる
- code mapは選択node近傍をfocus表示する
- inspectorは選択後に同一flow内で到達できる
- pan/zoomなしでも主要workflowを完遂できる

## URL state

hash routeはtopicとselectionを保持する。

```text
#implementation-replay?step=risk-constrain
#implementation-ppo?step=observation-v2
#code-map?symbol=trade_rl.evaluation.replay.run_single_symbol_replay
```

既存の `#overview` 等は互換維持する。unknown topicはhome、unknown selectionはtopic defaultへ戻し、blank/exceptionにしない。

## Search

検索はtopicだけでなくCodeReferenceとvariable aliasを返す。

検索対象:

- topic title/summary/body/keyword
- 日本語alias
- qualified symbol
- variable name
- file path

`希望保有数量` と `desired_quantity`、`リスク制約` と `PreTradeRisk.constrain` が同じ実装項目へ到達する。

結果型はtopic navigationだけに限定せず、`topic_id + selection` を持つ。主見出しは日本語、副表示はidentifier/pathとする。

## Relationship to authoritative sources

- architecture/research truth: `docs/architecture/*`, `docs/research/current-status.md`
- implementation truth: current `trade_rl/**` source + contract tests
- Guide: 上記を日本語で説明・可視化する非正本派生層

Markdown section fingerprintとPython symbol source digestは別oracleとして維持する。

## Current-main integration boundary

再レビュー時点でPR #513は `main` へmerge済みで、`docs/research/current-status.md` とGuideの研究状態は「Portable Controlled Experiment 0001 started; result not interpreted」を反映している。

`guide/japanese-code-explorer` はそのmerge commit `de2eefcdfb1999823344fae65f8701224f582fe8` をnon-force mergeで取り込み済みである。今回のredesignはこの研究状態を保持し、profitability / winner / final-test / live authorizationを新たに主張しない。

integration前にmainがさらに進んだ場合も、tested PR HEADがfinal current mainを含むまでnon-force同期して再検証する。

## Error handling / fail-closed behavior

build/test failure:

- missing / duplicate / wrong-kind CodeReference
- stale `source_sha256`
- CodeReference variableが対象symbolのlocal_namesに存在しない
- test/source path escapeまたはmissing
- architecture/code-map edgeが未知nodeを参照する
- sequence messageが未知actorを参照する
- required Japanese labelが空
- Markdown source fingerprint stale
- generated source revisionがCI/deploymentのcheckout SHAと一致しない

runtimeではunknown URL selectionとsearch no-resultを例外にせず、安全なdefaultまたは明示的な空結果へ戻す。

## Accessibility

既存のaxe `serious` / `critical` 0件contractを維持する。

追加要件:

- diagram node/messageはbutton相当のkeyboard操作を持つ
- selectionを色だけで表さない
- sequence順序はDOM上でも意味のある順序にする
- diagramと同じ情報へ到達できるテキストoutlineを提供する
- inspector更新は必要な範囲で`aria-live`へ通知する
- 320px幅でpage-level horizontal overflowを発生させない
- local horizontal scrollを使っても主要操作はstep-throughで代替できる

## Visual design

- 主役はdiagram
- inspectorは一つ
- explanationはdiagram selectionと同期
- 日本語本文の行長を抑える
- identifier/pathはmonospaceで分離する
- accent colorはselection/focusに限定する
- information cardを細分化しすぎない

## Expected file responsibilities

- `guide/tools/code_symbols.py`: AST symbol/local-name/source-digest index生成
- `guide/.generated/code-symbols.json`: untracked build artifact
- `guide/src/content/codeSymbols.ts`: generated index parse/lookup
- `guide/src/content/schema.ts`: CodeReference / sequence / code-map parse contract
- `guide/src/content/search.ts`: topic + code ref + variable search
- `guide/src/app/useHashRoute.ts`: topic + selection URL state
- `guide/src/visualizations/SequenceDiagram.tsx`: ordered runtime sequence
- `guide/src/visualizations/ArchitectureFlow.tsx`: existing edge correctness fix
- `guide/src/visualizations/CodeMap.tsx`: layered responsibility/code map
- `guide/src/visualizations/graphLayout.ts`: deterministic DAG rank/order helper
- `guide/src/components/CodeInspector.tsx`: symbol/variable/test/source detail
- `guide/content/topics/*.json`: 日本語中心copy + reviewed code references
- `guide/tests/*`: schema/search/routing/interaction unit/component tests
- `guide/e2e/*`: desktop/mobile/accessibility/visual/public smoke
- `tests/architecture/test_human_guide_code_symbols.py`: AST/source-binding contract
- `guide/README.md`, `docs/AGENTS.md`: durable maintenance contract

## Acceptance Criteria

1. Homeから「1本のバーを追う」へ2操作以内で到達できる。
2. `run_single_symbol_replay()` の主要処理順をsourceと一致するsequenceで表示する。
3. `PreTradeRisk.constrain()` と `MarketExecutor.execute_interval()` を選ぶと、日本語説明、実symbol、exact-revision source、主要入出力、関連testへ到達できる。
4. `desired_quantity` 等の主要変数は日本語aliasを持ち、存在しないvariable referenceはsource-checkで失敗する。
5. PPO viewでObservation v2の5segmentと除外情報を日本語で理解できる。
6. architecture viewは実edgeだけを描画し、node配列順から偽の矢印を作らない。
7. code mapのedge意味は `data-flow` / `calls` として明示され、静的import graphと誤認させない。
8. package →主要symbolへdrill-downできる。
9. `希望保有数量` と `desired_quantity` が同じ実装項目を検索できる。
10. 表示copyは日本語を主とし、英語識別子は照合用の副表示になる。
11. missing/stale CodeReferenceを入れるとsource-check/buildがfailする。
12. source linkはPages buildのexact revisionへ固定される。
13. Markdown source fingerprint contractを維持する。
14. desktop keyboardだけで主要diagram selectionとsource detailへ到達できる。
15. 320px viewportでpage-level horizontal overflowがない。
16. light/dark両方でaxe serious/critical violationが0件。
17. existing public Pages smoke contractを維持する。
18. Guide変更は `trade_rl/**` runtime semanticsを変更しない。
19. final tested PR HEADはfinal current `main` を含む。

## Invariants

- `docs/` とsource/testsがauthority、Guideは非正本。
- Python identifierはrenameしない。
- execution/accounting authorityは `MarketExecutor + BookState` のまま。
- strategy/risk責務分離を説明UIの都合で崩さない。
- generated code indexはcurrent treeへcommitしない。
- source inspectionのためproduction moduleをimport/executeしない。
- edge/message relationshipは明示modelを正本とし、配列順から未定義の関係を推測しない。
- stale/unknown source bindingはfail-closed。
- current research stateをGuide redesignで巻き戻さない。

## Failure Modes / Risks

### 説明と実装の再乖離

Markdown fingerprint + exact symbol source digest + variable existenceを別々に検証する。

### 図が高密度化する

sequenceは1 runtime questionに限定し、code mapはoverview-plus-detailを使う。全関数を一枚へ載せない。

### 日本語化でsource検索性が落ちる

日本語alias直下に実identifierを併記し、検索indexへ両方を入れる。

### layoutがrevisionごとに揺れる

deterministic layered layout、stable source order/rank、fixed actor orderを使う。

### AST parserがPython semanticsを誤推測する

symbol location/signature/source digest/local-name抽出だけを責務とする。call graphを自動推測しない。

### line linkがmain更新で壊れる

source revisionをexact build SHAへ固定する。

### accessibility regression

keyboard component tests、axe desktop/mobile/light/dark、320px E2EをTest Oracleへ含める。

## Test Oracle

### Python contract

- AST indexが `run_single_symbol_replay`、`PreTradeRisk.constrain`、`MarketExecutor.execute_interval`、PPO主要symbolを正しいfileへ解決する
- `desired_quantity` 等の主要local nameを同一scopeから抽出する
- nested scopeのlocalを混ぜない
- production sourceをimportせずindex生成する
- missing/stale symbol、unknown variable、path escapeをfailする
- generated artifactがtracked sourceへ残らない

### TypeScript unit/component

- sequence actor/message reference validation
- code-map edge validation
- architecture edge-only rendering
- Japanese aliasとidentifierの検索同一結果
- URL selection parse/format/unknown fallback
- selected messageとinspector同期
- code source URLがexact revisionを含む

### E2E / visual

- home → replay sequence → risk message → source detail
- search `希望保有数量` / `desired_quantity`
- desktop keyboard traversal
- mobile step-through
- light/dark axe
- 320px overflow
- deterministic desktop/mobile screenshots
- public hash/search/theme/source navigation

### Repository regression

- `python3 guide/tools/content_contract.py --check`
- `npm --prefix guide run check`
- `npm --prefix guide run e2e`
- affected `tests/architecture/`
- exact final HEADのpermanent repository CI

## Completion criteria

- Acceptance Criteria 1–19を確認する。
- architecture edge bugを再現するRED testを追加しGREEN化する。
- code symbol/variable/source-digest bindingのRED→GREENを確認する。
- replay/PPO sequenceをcurrent sourceと独立に照合する。
- final diffにruntime semantic change、debug code、generated catalog、temporary workaroundがない。
- final current mainをPR HEADが包含する。
- exact final HEADのpermanent CIが成功する。
- merge後Pages deploymentと実公開URL smokeが成功する。
- durable Guide contractを `guide/README.md` / `docs/AGENTS.md` / architecture testsへ昇格する。
- 実装・verification完了後、このActive spec/planをcurrent treeから削除する。
