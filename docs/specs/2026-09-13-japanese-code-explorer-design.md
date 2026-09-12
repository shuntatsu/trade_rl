# Japanese Code Explorer Design

Status: Active

## 結論

Interactive Human Guideを、抽象概念を眺める説明UIから、**Trade RLの実装を日本語で追跡できるコード連動型ガイド**へ再設計する。

画面上の説明・見出し・図ラベル・状態名は日本語を主表示とし、実際のPython識別子・型名・ファイルパスは照合用の補助情報として併記する。Python sourceの関数名・変数名・公開API自体は日本語へ改名しない。

実装理解の主経路は、概念説明ではなく、実コードの呼出し・状態遷移・データ変換を辿る対話型の図とする。特に `run_single_symbol_replay()` とPPO学習/推論経路を、実ファイル・実関数・主要変数・関連テストへ接続する。

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
- GitHub上で任意sourceを実行するIDEや完全なコードブラウザを作らない。
- すべてのprivate helperを網羅するAPI referenceを作らない。主要な実装経路に必要なsymbolへ限定する。
- force-directedな「それっぽい」図で依存関係を推測しない。

## Current problem

現行Guideには検索、テーマ、可視化、source fingerprint、accessibility/E2E gateがあり、説明層としての基盤は再利用できる。一方で次の問題がある。

- 日本語文中に `Point-in-time evidence`、`portable numerics`、`Strategy / PPO`、`Execution`、`EvidenceSet` 等が密集し、概念理解より用語解読が先に必要になる。
- topic schemaは概念説明中心で、関数・クラス・変数・入力・出力・例外・不変条件・関連テストを表現できない。
- 図を選択しても実ファイル・実symbolへ降りる経路がない。
- `ArchitectureFlow.tsx` は `visualization.edges` を描画せず、`nodes` 配列順に矢印を出す。このためdata model上の依存方向と画面表示が食い違い得る。
- `FlowStepper` は説明stepの選択UIであり、actor間message順序や状態変化を表すsequence diagramではない。
- source freshnessはMarkdown section fingerprintには存在するが、Guideが参照するPython symbolの存在/署名に対するfail-closed contractがない。

## Information architecture

### 最上位ナビゲーション

利用者の読み方を「概念カテゴリ」ではなく「理解したいこと」に寄せる。

1. **まず全体をつかむ** — 市場データ → データセット → 戦略 → リスク → 約定・会計 → 評価
2. **1本のバーを追う** — single-symbol replayのruntime sequence
3. **PPOを追う** — 観測、action、risk、execution、reward、学習、frozen strategy replay
4. **コード地図** — package/file/symbol dependency explorer
5. **データと特徴量** — causal / availability / staleness / Dataset identity
6. **約定・会計** — cost/fill/funding/borrow/liquidationのauthority
7. **実験と証拠** — Run / Study / EvidenceSet / comparison / freeze
8. **現在の研究状態** — current-statusの非正本要約

既存topicの有用な内容はこの構造へ移す。重複カードを増やさず、一つの問いに一つの主ビューを割り当てる。

## Language policy

### 主表示

次は原則として日本語にする。

- ナビゲーション
- topic title / summary
- section heading / body
- diagram actor/node/message label
- input/output/state/errorの説明
- search resultの説明
- accessibility label

### 補助表示として残す英語

次は実装照合性を守るため残す。

- Pythonの関数名・クラス名・変数名
- module/file path
- package名
- `PPO`, `SHA-256`, `JSON`, `NPZ` 等の標準的な固有名
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

Guide topicからPython sourceを直接推測せず、build-timeにASTからsymbol indexを生成する。

### Symbol index

生成物はGuide build workspace内だけに置き、current treeへcommitしない。各entryは最低限次を持つ。

```ts
type CodeSymbol = {
  qualified_name: string;
  kind: "function" | "class" | "method";
  path: string;
  start_line: number;
  end_line: number;
  signature: string;
};
```

AST parserはproduction codeをimport/executeしない。対象rootは `trade_rl/` に限定する。

Guideの手書きcontentは次のreferenceを持てる。

```ts
type CodeReference = {
  symbol: string;
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

`symbol` が生成indexに存在しない場合、Guideのsource check/buildはfail-closedに失敗する。曖昧な末尾一致は使わず完全修飾名でbindする。

## Primary implementation sequence: single-symbol replay

`trade_rl.evaluation.replay.run_single_symbol_replay` を最初のコード連動sequence diagramとする。

actorは次に固定する。

1. `MarketDataset` — 市場データセット
2. `run_single_symbol_replay` — リプレイ統括
3. `SingleSymbolStrategy` — 戦略
4. `PreTradeRisk` — ハードリスク
5. `MarketExecutor` — 約定エンジン
6. `BookState` — 帳簿状態
7. `SingleSymbolReplayResult` — 評価用結果

主要messageはsource順序と一致させる。

1. `_observation(...)` でcausal observationを構築
2. `strategy.decide(observation)` で `PositionIntent` を得る
3. intent変更時に `target_weight_for_intent(...)` と `_desired_quantity_from_weight(...)`
4. `_weight_for_desired_quantity(...)` でquantity-preserving holdをweightへ再表現
5. `risk_controller.constrain(...)` でhard risk projection
6. 必要時にrisk後targetから `desired_quantity` を更新
7. `executor.execute_interval(...)` でfill/cost/accounting
8. `book`, `returns`, `current_intent`, `index` を更新
9. terminationを評価
10. `ExecutionDiagnostics` と `SingleSymbolReplayResult` を構築

各messageをクリック/タップするとinspectorへ以下を表示する。

- 日本語での役割
- 実symbol
- source path + line range
- 入力
- 出力
- 主要変数の日本語alias
- 不変条件またはfail-closed条件
- 関連test path
- GitHub source link

## PPO sequence

PPOは「PPO Observation v2の説明」だけで終わらず、trainingとfrozen replayの境界を見せる。

主sequence:

1. Dataset / feature index validation
2. `PPOTradingEnv` 構築
3. Observation v2構築
4. PPO policy action
5. logical intent / target exposure
6. hard risk
7. `MarketExecutor.execute_interval`
8. BookState / reward更新
9. PPO update
10. fitted/frozen strategy
11. independent per-symbol replay

Observationの各segmentは、日本語名を主表示し、source field名を併記する。

- 選択したローカル特徴量
- 利用可能/有限マスク
- 正規化staleness
- 現在の売買意図
- 現在のウェイト

symbol IDとdataset-global featureがObservation v2へ入らないことも「含めない情報」として明示する。

## Architecture / code map

package dependencyは有向graphとして表示し、**edge modelを唯一の接続正本**とする。node配列順からedgeを推測しない。

package overviewは現行authorityに合わせる。

- `artifacts`
- `integrations`
- `data`
- `strategies`
- `risk`
- `simulation`
- `evaluation`

選択したpackageはfile一覧ではなく、まず主要public/owner symbolを表示する。その後file/symbolへdrill-downする。

Graph familyはdirectional dependency graphなのでlayered layoutを使う。desktopは左→右または上→下の安定したrank、edge crossingの最小化、明示的な矢印を優先する。mobile portraitは全graphを縮小せず、選択packageの近傍とoutline/step-throughを主経路にする。

## Interaction design

### Desktop

- 中央: sequence / dependency diagram
- 右: 選択中symbolのinspector
- 上部: 日本語/コード名共通検索
- diagram node/messageはクリックだけで選択可能
- 選択時のみ関係pathを強調し、未選択edgeは静かに表示
- resetで全体表示へ戻る

### Mobile portrait

- diagramを極小縮小しない
- sequenceは1 stepずつ「前へ / 次へ」で移動できる
- dependency graphは選択node近傍をfocus表示する
- inspectorは選択後に同一flow内で読める位置へ出す
- pan/zoomがなくても主要workflowを完遂できる

### URL state

hash routeへviewとselectionを保持し、特定実装stepを共有可能にする。

例:

```text
#implementation/replay?step=risk-constrain
#implementation/ppo?step=observation-v2
#code-map?symbol=trade_rl.evaluation.replay.run_single_symbol_replay
```

無効なselectionは安全にdefaultへ戻し、blank/exceptionにしない。

## Search

検索は日本語aliasと実コード識別子を同じ結果へ正規化する。

例:

- `希望保有数量` と `desired_quantity`
- `リスク制約` と `PreTradeRisk.constrain`
- `約定` と `execute_interval`

検索対象:

- topic title/summary/body
- Japanese alias
- qualified symbol
- variable name
- file path
- keyword

検索結果は「日本語名」を主見出し、「code identifier / path」を副表示にする。

## Source links

既存のauthoritative Markdown source linkに加え、code referenceはGitHub sourceの該当file/line rangeへ遷移できるようにする。

Guideはsource URLをcontentへ手書きせず、repository metadata + `CodeSymbol.path/start_line/end_line` から決定的に構築する。

## Relationship to authoritative docs

Guideは引き続き非正本である。

- architecture/research truth: `docs/architecture/*`, `docs/research/current-status.md`
- implementation truth: current `trade_rl/**` source + contract tests
- Guide: 上記を人間向けに説明・可視化する派生層

既存Markdown section fingerprintは維持する。Python sourceについてはsymbol existence/signature contractを追加し、docs freshnessとcode freshnessを別oracleとして扱う。

## Open PR #513 integration boundary

実装開始時点でPR #513 `docs/issue511-experiment-in-progress` がopenで、`docs/research/current-status.md` と影響するGuide topicsを更新している。

今回のbranchはcurrent `main` から分離して開始するが、研究状態を含むGuide contentを最終編集する前に、#513がmainへ入った場合はnon-forceでcurrent mainを取り込み、その研究状態を保持する。#513が未mergeのまま並行する場合、同じresearch-state factsを今回のPRで独自に再定義しない。

最終PRはtested HEADがその時点のcurrent mainを含むことを必須とする。

## Error handling / fail-closed behavior

次はbuild/test failureとし、黙って表示を続けない。

- CodeReferenceが存在しないsymbolを指す
- symbol kindが期待と異なる
- duplicate symbol key
- architecture edgeが未知nodeを参照する
- sequence messageが未知actorを参照する
- required Japanese labelが空
- source fingerprint stale
- generated source linkがrepository root外pathを指す

一方、URL上のunknown selectionやsearch no-resultはruntime errorにせず、default stateまたは明示的な空結果を表示する。

## Accessibility

既存のaxe `serious` / `critical` 0件contractを維持する。

追加要件:

- diagram node/messageはbutton相当のkeyboard操作を持つ
- selectionを色だけで表さない
- sequence順序はDOM上でも意味のある順序にする
- diagramと同じ情報へ到達できるテキストoutlineを提供する
- inspector更新は必要な範囲で`aria-live`へ通知する
- 320px幅でpage-level horizontal overflowを発生させない
- wide diagramに局所horizontal scrollを使う場合も主要操作はscrollなしのstep-throughで代替できる

## Visual design

現行のcalm/light-dark対応は維持するが、cardを増やして情報量を分散させない。

- 主役はdiagram
- inspectorは一つ
- explanationはdiagram selectionと同期
- 同じ情報を「図・カード・本文」で三重表示しない
- 日本語本文の行長を抑え、識別子はmonospaceで視覚的に分離する
- accent colorはselection/focusに限定し、全nodeを派手に塗らない

## Expected file responsibilities

実装時の責務境界は次を基本とする。詳細なtask分割はimplementation planで確定する。

- `guide/tools/code_symbols.py`: ASTベースのsource symbol index生成/検証
- `guide/src/content/codeSymbols.ts`: generated JSONを型付きで読み込む薄い境界、または同等のbuild-time import seam
- `guide/src/content/schema.ts`: CodeReference / sequence / code-map modelのparse contract
- `guide/src/visualizations/SequenceDiagram.tsx`: runtime interaction sequence
- `guide/src/visualizations/CodeMap.tsx`: layered dependency/code map
- `guide/src/components/CodeInspector.tsx`: 選択symbol/variable/test/source detail
- `guide/src/content/search.ts`: 日本語alias + identifier search
- `guide/src/app/useHashRoute.ts`: view/selection URL state
- `guide/content/topics/*.json`: 日本語中心copyとcode reference
- `guide/tests/*`, `guide/e2e/*`: contract/interaction/accessibility/visual regression
- `tests/architecture/test_human_guide*.py`: build-time source bindingのarchitecture contract

必要以上に新libraryを追加しない。既存React/Vite stackで明確に実装できないlayout要件が確認された場合のみ、deterministic layered layout dependencyを検討する。

## Acceptance Criteria

1. Homeから「1本のバーを追う」へ2操作以内で到達できる。
2. `run_single_symbol_replay()` の主要処理順を、sourceと一致するsequenceとして表示できる。
3. sequenceの `PreTradeRisk.constrain()` と `MarketExecutor.execute_interval()` を選ぶと、日本語説明、実symbol、path、主要入出力、関連testへ到達できる。
4. PPO viewでObservation v2の5segmentと除外情報を日本語で理解できる。
5. architecture viewはcontent modelの実edgeを描画し、node配列順から偽の依存矢印を作らない。
6. package → file/symbolへdrill-downできる。
7. `希望保有数量` と `desired_quantity` が同じ実装項目を検索できる。
8. 表示copyは日本語を主とし、英語識別子は照合用の副表示になる。
9. 存在しないCodeReferenceを入れるとsource-checkまたはbuildがfailする。
10. Markdown source fingerprint contractを維持する。
11. desktop keyboardだけで主要diagram selectionとsource detailへ到達できる。
12. 320px viewportでpage-level horizontal overflowがない。
13. light/dark両方でaxe serious/critical violationが0件。
14. existing public Pages smoke contractを維持する。
15. Guide変更は `trade_rl/**` runtime semanticsを変更しない。
16. final tested PR HEADはfinal current `main` を含む。

## Invariants

- `docs/` とsource/testsがauthority、Guideは非正本。
- Python identifierはrenameしない。
- execution/accounting authorityは `MarketExecutor + BookState` のまま。
- strategy/risk責務分離を説明UIの都合で崩さない。
- generated code indexはcurrent treeへcommitしない。
- source inspectionのためproduction moduleをimport/executeしない。
- edge/message relationshipは明示modelを正本とし、配列順から意味を推測しない。
- stale/unknown source bindingはfail-closed。
- #513または後続current-status更新をGuide redesignで巻き戻さない。

## Failure Modes / Risks

### 説明と実装の再乖離

対策: docs fingerprint + exact Python symbol bindingを別々にCI検証する。

### 図が高密度化し再び読めなくなる

対策: sequenceは1 runtime questionに限定し、code mapはoverview-plus-detailとfocus neighborhoodを使う。全関数を一枚へ載せない。

### 日本語化でsource検索性が落ちる

対策: 日本語aliasの直下に実identifierを併記し、検索indexへ両方を入れる。

### 自動layoutがrevisionごとに大きく揺れる

対策: deterministic layered layout、stable source order/rank、fixed actor orderを使う。random/force layoutを使わない。

### AST parserがPython semanticsを誤推測する

対策: parserの責務をsymbol location/signature extractionへ限定し、call graphを静的に自動推測してsource truthとして表示しない。sequence relationshipはreview済みcontent modelに明示する。

### PR #513とのresearch content conflict

対策: research factを独自再定義せず、final content edit前とintegration前にcurrent mainを同期する。

### 新UIでaccessibilityが退行する

対策: keyboard interaction/unit tests、axe desktop/mobile/light/dark、320px E2EをTest Oracleへ含める。

## Test Oracle

単にbuild成功を完了条件にしない。

### Contract / unit

- nonexistent/duplicate/wrong-kind CodeReferenceがfailする
- sequence actor/message reference validation
- architecture edge validation
- Japanese aliasとidentifierの検索同一結果
- URL state parse/unknown fallback
- selected messageとinspector detailの同期

### Source-derived oracle

- AST indexが `trade_rl.evaluation.replay.run_single_symbol_replay`、`trade_rl.risk.pretrade.PreTradeRisk.constrain`、`trade_rl.simulation.execution.MarketExecutor.execute_interval`、PPO主要symbolを正しいfileへ解決する
- production sourceをimportせずindex生成できる
- generated artifactがtracked sourceへ残らない

### Interaction / E2E

- home → replay sequence → risk message → source detail
- search `希望保有数量` / `desired_quantity`
- desktop keyboard traversal
- mobile step-through
- light/dark axe
- 320px overflow
- public hash/search/theme/source navigation

### Regression

- current `npm --prefix guide run check`
- current `npm --prefix guide run e2e`
- affected architecture tests
- repository required full CI on exact final HEAD

## Completion criteria

実装完了は次をすべて満たした場合のみとする。

- Acceptance Criteria 1–16を確認
- architecture edge bugを再現するRED testを追加しGREEN化
- code bindingのRED→GREENを確認
- replay/PPO主要sequenceがsource reviewと一致
- final diffにruntime semantic change、debug code、generated catalog、temporary workaroundがない
- final current mainをPR HEADが包含
- exact final HEADのpermanent CIが成功
- public deployment後の実URL smokeが成功
- durable Guide contractを `guide/README.md` / architecture testsへ昇格
- implementation完了後、このActive spec/planはcurrent treeから削除する
