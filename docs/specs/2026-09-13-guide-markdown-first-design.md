# Human Guide Markdown-first redesign

Date: 2026-09-13
Status: written design pending user review; implementation not started
Base: `main` at `e91d8d51d5b2dcce038311221967b7701c72ecb3`

## 結論

Human Guideを「インタラクティブな可視化を操作して理解するUI」から、**上から読むだけで主要な理解が完結する技術文書 + 必要時だけ使う実装探索補助**へ変更する。

最重要原則は次の一文とする。

> 読むだけで8〜9割理解でき、操作すると残り1〜2割の実装詳細まで掘れる。

現在の `App.tsx` は `TopicHeader` の直後に `visual-stage` を置き、その後に説明文を置くため、情報設計として「読む」より「触る」を先に要求している。新設計ではこれを反転し、本文・静的フロー・不変条件を先に置く。

## Objective

- 概要と詳細を異なる情報密度・異なるテンプレートで提示する。
- 各ページをMarkdown中心の読み物として成立させ、操作しなくても処理順と責務を理解できるようにする。
- replay / PPOなど順序が重要な処理は、ページ初期表示で全stepが見える静的シーケンスとして示す。
- exact-SHA source link、日本語alias ↔ Python identifier、source digest、関連test、検索など既存のtraceabilityは維持する。
- 可視化・Code Inspectorは理解の前提ではなく、実装確認の補助へ降格する。
- 現在の500,000-byte JavaScript chunk budgetとaccessibility/public smoke契約を維持する。

## Non-goals

- `trade_rl/**` の売買runtime、研究ロジック、PPO、execution、riskの挙動を変更しない。
- 研究結論やprofitability claimを変更しない。
- 汎用CMS、MDX実行環境、自由なHTML埋め込み機構を作らない。
- 初期版ではMermaid、D3、追加の大型図描画依存を導入しない。
- インタラクションを完全廃止しない。検索、テーマ、折りたたみ、exact source navigationは残す。
- 既存docsをGuideへコピーして正本化しない。Guideは引き続き非正本説明層とする。

## 現状の問題

### 1. 概要と詳細が同じUI構造

`overview`、`implementation-replay`、`implementation-ppo`、`code-map`などが共通して「header → large visualization → explanation cards」の順で表示される。

そのため概要でも詳細でも最初に大きな図へ注意が向き、ページの役割差が弱い。

### 2. sequenceの理解に操作を要求する

replay/PPOのsequenceはstep選択とCode Inspectorを組み合わせて詳細を見せるため、全体の順序を読むだけなら通常の静的シーケンスより視線移動が多い。

### 3. 説明文が二次情報になっている

現在の説明文自体は内容が強いが、visual-stageより後ろにあるため「まず読むべき本文」ではなく「図の補足」に見える。

### 4. Code Mapが学習導線へ入り込みすぎる

ownership/referenceとしては有用だが、初見読者の理解順序では実装地図より、system purpose → causal flow → replay → PPO/execution → research statusが先である。

## 採用案

### Markdown-first hybrid

本文を実Markdownへ移し、機械検証metadataだけをsidecar JSONへ分離する。

```text
guide/content/
  manifest.json
  pages/
    overview.md
    implementation-replay.md
    implementation-ppo.md
    data-flow.md
    execution-economics.md
    experiment-loop.md
    research-status.md
    code-map.md
  meta/
    overview.json
    implementation-replay.json
    implementation-ppo.json
    data-flow.json
    execution-economics.json
    experiment-loop.json
    research-status.json
    code-map.json
```

Markdownが人間向け文章・表・静的 `text` 図を所有する。sidecar JSONは次だけを所有する。

- `id`
- `title`
- `nav_label`
- `summary`
- `role`
- `keywords`
- `source_sections` + SHA-256 fingerprint
- `code_references`

現在の `sections` と primary `visualization` はsidecarから外す。文章と処理順の正面表示をJSON component modelへ閉じ込めない。

Markdown内へ独自directive、MDX component、実行可能JavaScriptを埋め込まない。sequence/flowは通常のMarkdown `text` blockとtableだけで表現する。

### Manifest v2

flatなtopic配列をやめ、navigation groupとdefault reading orderを明示する。

概念形:

```json
{
  "schema_version": "document-guide-v2",
  "home": "overview",
  "groups": [
    {"id": "overview", "label": "概要", "topics": ["overview"]},
    {"id": "mechanics", "label": "仕組み", "topics": ["data-flow", "implementation-replay", "implementation-ppo", "execution-economics", "experiment-loop"]},
    {"id": "status", "label": "現在地", "topics": ["research-status"]},
    {"id": "reference", "label": "参照", "topics": ["code-map"]}
  ],
  "reading_order": ["overview", "data-flow", "implementation-replay", "implementation-ppo", "execution-economics", "experiment-loop", "research-status"]
}
```

`code-map` はnavigationには存在するが `reading_order` へ含めない。

## なぜMermaidを初期採用しないか

現行の正本Markdown `docs/architecture/lean-core.md` は、すでに `text` fenced blockのフロー図と文章・箇条書きで高い可読性を持つ。

初期 redesign ではこれに合わせる。

```text
MarketDataset
    ↓
StrategyObservation
    ↓
Strategy.decide
    ↓
PreTradeRisk.constrain
    ↓
MarketExecutor.execute_interval
    ↓
BookState / interval net return
```

sequenceは必要ならlifeline風のmonospace `text` blockまたは番号付きstep表を併記する。JS描画がなくても内容が失われないことを優先する。

Mermaid等を将来導入する場合も、静的本文が完全なfallbackであり、図のrender成功を理解の前提にしないことを条件とする。

## ページrole

meta JSONへ `role` を追加し、値を `overview | detail | status | reference` に限定する。

### `overview`

対象: `overview`

目的: 3〜5分で「何のシステムか」「どう流れるか」「どこまで証明済みか」を理解する。

表示順:

1. 何を作っているか
2. 5段階程度の静的core flow
3. 実証済み / 未証明の表
4. 次に読むページの地図

contract:

- `code_references` は**空配列必須**。
- CodeInspector / ImplementationDetailsをrenderしない。
- Python local variable一覧を出さない。
- selectable node/stepを出さない。
- 詳細なclass/function ownershipを主表示しない。

必要に応じcanonical type名をMarkdown本文の補助表記として書けるが、実装探索を目的にしない。

### `detail`

対象:

- `implementation-replay`
- `implementation-ppo`
- `data-flow`
- `execution-economics`
- `experiment-loop`

目的: 1つの仕組みを、処理順・理由・不変条件・実装対応まで理解する。

表示順:

1. このページで答える問い
2. 結論 / 要点
3. 静的sequence / flow / table
4. step-by-step説明
5. 重要なinvariant / failure mode
6. 記事末尾の折りたたみ式「実装参照」

sequenceの主要stepは**初期DOMに全て存在**し、click/tapしないと次stepが読めない設計を禁止する。

### `status`

対象: `research-status`

目的: 現時点の研究結果と未証明事項を事実として読む。

表示順:

1. 結論
2. confirmed evidence
3. negative / inconclusive evidence
4. 未証明事項
5. source/research artifact参照

contract:

- `code_references` は**空配列必須**。
- 研究statusは文章と表を使い、装飾的な可視化を置かない。

### `reference`

対象: `code-map`

目的: 既に概念を理解した読者がownership、class/function、testsを引く。

通常の「次に読む」学習順から外し、navigationの `参照` groupへ置く。

主表示はstatic ownership tableとdependency direction。exact source linkを持てるが、clickable graphは理解の前提にしない。

## Navigation

視覚的に次のgroupへ分ける。

```text
概要
  全体像

仕組み
  データと特徴量
  1本のバーを追う
  PPOの実装を追う
  約定・会計
  実験・検証証拠

現在地
  研究の現在地

参照
  コード地図
```

`次に見る` はmanifestの `reading_order` だけを辿る。

## Rendering architecture

### Content loading

Markdown本文はViteのraw importで読み込み、meta JSONと `id` でpairする。実装時は `import.meta.glob(..., { eager: true, query: "?raw", import: "default" })` 相当の仕組みを使い、Markdownを実行可能moduleへ変換しない。

各pageは同名Markdown + metaが必須。manifestにないorphan page/meta、manifestが指すmissing page/meta、id不一致はfail-closed。

### App shell

`AppShell`、navigation、theme、searchは維持する。

`App.tsx` のprimary responsibilityを「visualization selection stateの解決」から「document pageのroutingと表示」へ変更する。

新しいtop-level構造は概念的に次とする。

```text
AppShell
  TopicHeader
  MarkdownArticle
  ImplementationReferenceAppendix?   # detail/referenceのみ
  NextReadingLink
```

`visual-stage` を全topic共通の固定位置として置かない。

### Markdown renderer

初期実装は `react-markdown` + `remark-gfm` を使う。`rehype-raw` は導入せず、raw HTML / MDX / arbitrary JSを実行しない。

必須support:

- ATX headings (`#` / `##` / `###`)
- paragraphs
- ordered/unordered lists
- GFM tables
- inline code
- fenced code blocks
- blockquotes
- links

Guide contentのheadingはATX形式へ限定する。heading slugはdeterministicに生成し、同一page内のduplicate normalized slugはcontent checkでfailさせる。

link rendererは `http:`, `https:`, relative path, fragmentのみを許可し、`javascript:` / `data:` 等を拒否する。

`react-markdown`導入後に500,000-byte budgetを超える場合、budgetを上げない。第一選択はrendererのcode splitting、なお超える場合はbuild-time Markdown precompileへ切り替える。

### Static sequence / flow

初期versionではMarkdownの `text` code blockとtableを使用する。これによりraw sourceでも意味が読め、JavaScript failureでも内容が残る。

wideなsequenceはpage全体をoverflowさせず、そのblockだけ `overflow-x: auto` とする。320px document-level horizontal overflowは禁止する。

## Implementation reference appendix

現在のside-by-side `CodeInspector` はprimary layoutから外す。

Markdown本文へ独自component insertion markerを持ち込まず、page末尾に `実装を確認する` appendixを置く。sidecar `code_references` を順に折りたたみ表示する。

```text
実装を確認する

▶ 単銘柄リプレイを実行
    run_single_symbol_replay
    path / line / signature
    主要変数
    関連test
    exact-SHA source link

▶ ハードリスクを適用
    PreTradeRisk.constrain
    ...
```

初期状態は全てcollapsed。

本文だけで概念・順序・不変条件が完結し、appendixは確認用である。

ただしsearchで特定symbolを選んだdeep linkでは、そのsymbolのdetailsだけ自動openしてよい。これにより探索能力を維持しながら、通常閲覧では注意を奪わない。

## Search

検索は残す。

検索indexは次を対象にする。

- Markdown本文のplain text
- heading
- page title / summary / keywords
- 日本語code alias
- Python symbol
- variable alias / identifier
- source path

検索結果は `page` / `heading` / `symbol` の種別を持つ。

- page/heading結果: 該当document位置へ移動
- symbol結果: 該当pageへ移動し、appendixの該当implementation detailsだけopen

sequence step selectionをsearch routingの前提にしない。

Markdown plain text extractionはrendererとは別のcontent utilityへ閉じ、fenced codeを含めるかどうかをテストで固定する。検索ではidentifier発見性のためinline/fenced code textを含める。

## URL contract

page deep linkは維持する。

新しいcanonical targetは `topic + heading` または `topic + symbol` とする。

旧 `step=` deep linkはmigration期間中に既知のheading/symbolへredirectできるものだけcompat mappingを持ち、意味のないvisual step stateを恒久APIにしない。unknown legacy stepはpage rootへ安全にfallbackする。

source linkは引き続きfloating `main` ではなく `GUIDE_SOURCE_REV` のexact SHAへ固定する。

## Content / source contract

### sidecar pairing

各meta JSONは同名Markdownを1つだけ持つ。missing/duplicate/orphanはfail-closed。

### role contract

- `overview`: `code_references=[]` 必須。
- `status`: `code_references=[]` 必須。
- `detail`: code referencesは0件以上。
- `reference`: code referencesは1件以上。

### docs freshness

`source_sections` fingerprint契約は維持する。

Markdownへ移行しても、対応する正本docs sectionが変わった場合は`source-check`を失敗させる。refreshは現在同様、reviewしたpageだけ明示的に行う。

### code freshness

full AST index / compact runtime indexの分離は維持する。

runtime symbol indexはsidecar `code_references`からunique symbol集合を作る。missing symbol、wrong kind、stale source digest、unknown variable、missing/escaping test pathはfail-closed。

### research facts

`research-status` とoverviewの現在地記述は `docs/research/current-status.md` のfingerprintへbindする。Guide移行時に研究結論を再解釈・強化しない。

## Content migration

### overview

現在のarchitecture nodesは、5段階程度のMarkdown `text` flowへ縮約する。

「現在の位置」は、

- verified
- observed but insufficient
- not proven

の区別が一目で分かる表へする。

### implementation-replay

現在の8 sequence messageは順序を保持してMarkdownへ移す。

最初に全sequenceを静的表示し、その後に少なくとも次のstepを本文headingとして説明する。

1. 観測を作る
2. 戦略がintentを返す
3. 希望数量 / proposalへ変換
4. hard riskを適用
5. 約定・会計
6. BookStateを引き継ぐ
7. 結果を確定する

`desired_quantity`、`proposal_weight`、`target_weight`等は記事末尾の実装参照appendixで説明し、ページ冒頭へ露出しない。

### implementation-ppo

静的表示の主軸を次へ固定する。

```text
Observation v2
  → PPO action
  → PositionIntent / proposal
  → hard risk
  → execution/accounting
  → interval net return
  → reward
```

観測5区分はtableで常時見えるようにする。

学習時と実行時で同じencoderを使う点を本文のinvariantとして明示する。

### data-flow / execution-economics / experiment-loop

現在のinteractive card切替をMarkdownのflow / table / invariantへ移す。

クリックしないとassumptionやfailure stateが見えない構造を禁止する。

### research-status

文章と表へ寄せる。visual status groupは削除対象。

### code-map

interactive graphをreference tableへ変更する。

最低限の列:

- 責務
- package/module
- canonical class/function
- owns
- does not own
- test

## Invariants

1. Guideだけを読んでも、source codeを開かずにcore flowを説明できる。
2. replay/PPOの主要処理順は初期表示だけで読める。
3. exact source linkとcode digest contractは弱めない。
4. overview/statusにcode referenceを持ち込まない。
5. implementation appendixを開かなくても主要説明が欠落しない。
6. Guideはdocs/research/sourceを正本として扱い続ける。
7. `trade_rl/**` runtime semanticsを変更しない。
8. page-level horizontal overflowを作らない。
9. keyboard / screen readerでdocument readingとdetails disclosureを完結できる。
10. 500,000-byte JavaScript chunk budgetを維持する。
11. Markdown本文へ独自UI directiveや実行可能componentを埋め込まない。

## Failure modes / Risks

### Markdownへ移した結果source traceabilityが弱まる

対策: machine metadataはsidecar JSONへ残し、既存fingerprint / AST contractを移植する。Markdownへhashやpathを手書きしない。

### Markdown parserが新しいXSS面を作る

対策: `react-markdown`をraw HTML無効で使用し、`rehype-raw` / MDXを導入しない。link protocolもallowlistする。

### static diagramが横に長くmobileを壊す

対策: local code-block scrollerだけを許可し、document root overflowをE2Eで禁止する。同じ内容を番号付きtext/tableでも保持する。

### インタラクティブ機能を減らした結果sourceへ辿れなくなる

対策: article末尾のimplementation appendixとsymbol searchを残す。source linkはexact SHAを維持する。

### 概要が再び詳細化する

対策: `role=overview`は `code_references=[]` を必須にし、overview rendererはImplementationReferenceAppendixをmountしないことをunit/E2Eで固定する。

### 旧step deep linkが壊れる

対策:既知のstepだけcompat mappingし、unknown stepはpage rootへ安全にfallbackする。新規contentはstep routingを増やさない。

### bundleが再び肥大化する

対策:500,000-byte hard budgetを維持。Markdown renderer導入後のproduction chunkを実bytesで検証し、超過時はbudget引上げではなくcode splitting / build-time precompileで対処する。

### content migrationで研究事実が変わる

対策:移行は表現変更のみ。`source_sections` fingerprintを維持し、`docs/research/current-status.md`と差分レビューする。

### Markdownとmetaが別々にdriftする

対策:同名pair、manifest inclusion、role contractをcontent contractで検証する。orphan/missingを許可しない。

## Acceptance Criteria

### Information architecture

- [ ] manifest schemaが `document-guide-v2` になる。
- [ ] overview/detail/status/referenceのroleがmetaで明示される。
- [ ] navigationが `概要 / 仕組み / 現在地 / 参照` へgroup化される。
- [ ] `code-map` はdefault `reading_order`から外れる。

### Reading-first UI

- [ ] 全pageでMarkdown本文がprimary contentとして実装参照より先にrenderされる。
- [ ] overview/statusにselectable visualization / CodeInspector / implementation appendixがない。
- [ ] replay/PPOの主要sequence stepが初期DOMに全て存在する。
- [ ] core explanationの理解にclick/tapを要求しない。
- [ ] detail/referenceのimplementation appendixは通常collapsedである。

### Content

- [ ] 現在の8 topicsがMarkdown + sidecar metadataへ移行される。
- [ ] replay/PPOの既存semantic stepsが欠落しない。
- [ ] research-status/overviewの事実がcurrent research docsと一致する。
- [ ] static flow/sequenceはraw Markdownでも意味が読める。
- [ ] Markdownにcustom UI directive / MDX / raw HTML dependencyがない。

### Traceability

- [ ] source section fingerprintsが引き続きfail-closed。
- [ ] code reference digest/kind/variable/test checksが引き続きfail-closed。
- [ ] source URLがdeployed exact SHAを指す。
- [ ] 日本語alias / Python symbol / variable searchを維持する。

### Accessibility / responsive

- [ ] 320px document-level horizontal overflow 0。
- [ ] wide text diagramはlocal scrollで利用可能。
- [ ] detailsはkeyboardでopen/close可能。
- [ ] axe serious/critical violation 0。
- [ ] reduced motionで情報欠落なし。

### Performance / deployment

- [ ] production JS chunk <= 500,000 bytes。
- [ ] Vite oversized chunk warningなし。
- [ ] Lean Core全gate Green。
- [ ] Human Guide unit/type/lint/build/E2E Green。
- [ ] post-merge Pages build/deploy/public smoke Green。

## Test Oracle

Tests Greenだけでは完了としない。次を観測する。

### Static/content contract

- all Markdown/meta pairing complete
- manifest role/group/reading_order valid
- overview/status code references empty
- reference code references non-empty
- duplicate ATX heading slug rejects
- source fingerprint stale rejects
- code symbol stale/missing rejects
- orphan meta/Markdown rejects
- runtime code symbol index contains only referenced symbols

### Unit

- `react-markdown + remark-gfm`でheading/list/table/codeをrenderできる
- raw HTML / unsafe protocolを実行しない
- search indexes Markdown heading/body and code alias/symbol
- symbol deep link opens only the relevant implementation appendix item
- overview/status renderer does not mount implementation appendix
- next-reading follows manifest `reading_order` and skips reference pages

### Browser E2E

Desktop + 320px mobileで最低限:

1. overviewを開き、クリックなしでpurpose/core flow/proven-vs-unprovenが読める。
2. replayを開き、クリックなしで観測→strategy→risk→execution→bookの順序を読める。
3. PPOを開き、観測5区分とaction→risk→execution→rewardをクリックなしで読める。
4. implementation appendixが初期collapsedで、keyboardでopenできる。
5. `desired_quantity`検索からreplayの該当implementation appendixへ到達できる。
6. exact-SHA source linkを検証する。
7. page-level horizontal overflowがない。
8. serious/critical accessibility violationがない。

### Post-merge public oracle

実公開 `https://shuntatsu.github.io/trade_rl/` に対して、overview reading path、replay/PPO static content、symbol search、exact source link、320px overflowをsmokeする。

## Migration strategy

1. 新しいMarkdown/meta loaderとcontractをlegacy JSONと並存させ、RED/GREENで基盤を作る。
2. overviewを最初に移し、overview専用layout contractを固定する。
3. replay / PPOを移し、reading-first sequence oracleを固定する。
4. data-flow / execution-economics / experiment-loop / research-statusを移す。
5. code-mapをreference templateへ移す。
6. 全8 topic parity確認後にlegacy `sections` / `visualization` schemaと不要なinteractive renderer/stateを削除する。
7. final diff reviewでdead visualization code、stale CSS、旧step testsを確認し、単純削除ではなく新しいreading oracleへ置換されたことを確認する。
8. exact-head CI → merge → main CI → Pages deploy → public smokeで閉じる。

途中状態をpublic mainへmergeしない。最終PRでは全8 topicが新contractへ移行していることを要求する。

## Design decision summary

採用するのは「Markdownをprimary、静的図を本文の一部、実装探索をsecondary disclosure」にする構成である。

採用しないもの:

- interactive visualization first
- overviewとdetailの共通template
- step selectionが必要なsequence comprehension
- Mermaid/MDX等の追加複雑性を初期 redesignへ持ち込むこと
- custom Markdown UI directive
- warning/bundle閾値を緩めること

この設計の成功条件は「図が格好良いこと」ではなく、**初見読者が上から読むだけで処理順、責務、現在の研究状態を説明できること**である。
