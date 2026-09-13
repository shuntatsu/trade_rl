# Trade RL Human Guide

This guide is non-authoritative. Technical truth lives in `docs/architecture/*`, `docs/research/current-status.md`, current source, and contract tests.

公開URL: <https://shuntatsu.github.io/trade_rl/>

## 結論

Human Guideは**Markdown-first technical document + secondary implementation explorer**です。日本語の本文、静的flow、表、不変条件を上から読むだけで主要な理解が完結し、必要なときだけ記事末尾の「実装を確認する」disclosureから実identifier、path、signature、関連test、exact-SHA sourceへ降ります。

設計原則は次です。

> 読むだけで8〜9割理解でき、操作すると残り1〜2割の実装詳細まで掘れる。

インタラクティブな図を理解の前提にはしません。replay/PPOの処理順、研究状態、execution economics、実験ループは初期DOMの本文だけで読めます。

## Content layout

人間向け本文と機械検証metadataを分離します。

```text
content/
  manifest.json
  pages/
    overview.md
    data-flow.md
    implementation-replay.md
    implementation-ppo.md
    execution-economics.md
    experiment-loop.md
    research-status.md
    code-map.md
  meta/
    <same topic ids>.json
```

- `pages/*.md`: 説明本文、静的`text` flow、表、不変条件、failure mode。
- `meta/*.json`: `id`, title/nav/summary, `role`, keywords, authoritative source fingerprints, reviewed `code_references`。
- `manifest.json`: `document-guide-v2`。navigation groupと`reading_order`を固定する。

roleは`overview | detail | status | reference`です。`overview`と`status`は`code_references=[]`必須、`reference`は1件以上必須です。`code-map`はnavigationの「参照」には存在しますが、通常の`reading_order`から外します。

MarkdownへMDX、raw HTML renderer、custom UI directiveを持ち込みません。描画は`react-markdown` + `remark-gfm`、raw HTML無効で行います。wideなtext diagram/tableだけを局所scroll可能にし、page全体のhorizontal overflowは許可しません。

## Traceability / code-linked contract

GuideはPython sourceをimport/executeせず、`tools/code_symbols.py`がASTからcode-symbol indexを生成します。

- `.generated/code-symbols.json`: source-check/review用full index。
- `.generated/code-symbols-runtime.json`: `content/meta/*.json`のreview済み`code_references`が参照するunique symbolだけを含むbrowser配布用index。

両indexはcommitしません。各entryは完全修飾symbol、kind、source path/line、signature、source SHA-256、主要local名、**exact revision**を保持します。runtime indexはfull indexを置き換えず、full indexから参照集合へ射影します。unknown symbolなら生成時にfail closedです。

`tools/content_contract.py --check`は次をfail closedで検証します。

- `pages/*.md` と `meta/*.json` の1対1対応、orphan/missing。
- manifest group / reading order / role contract。
- duplicate normalized Markdown heading slug。
- authoritative docs section fingerprint。
- code symbolのmissing/wrong kind/stale digest。
- unknown variable、missing/escaping test path。
- overview/statusへのcode reference混入。
- legacy `content/topics` の再導入。

正本docs/sourceを読み直し、説明がまだ正しいと人間が確認したtopicだけ明示的にrefreshします。

```bash
python3 guide/tools/content_contract.py --refresh-code <reviewed-topic-id>
python3 guide/tools/content_contract.py --refresh <reviewed-topic-id>
python3 guide/tools/content_contract.py --check
```

`--refresh-code`はCodeReferenceのsource digestだけ、`--refresh`はauthoritative Markdown section fingerprintだけを更新します。CIを通すための一括承認として使いません。

source linkはfloating `main`ではなく、そのGuide buildの`GUIDE_SOURCE_REV` exact SHAへ固定します。

## Reading and navigation contract

navigationは次の4 groupです。

```text
概要
  全体像

仕組み
  データと特徴量
  1本のバーを追う
  PPOの実装を追う
  約定・会計
  実験と証拠

現在地
  研究の現在地

参照
  コード地図
```

canonical deep linkはtopic root、`?heading=<slug>`、`?symbol=<qualified-name>`です。旧`step=`は既知のreplay/PPO stepだけ対応headingへ互換変換し、unknown stepはpage rootへ安全にfallbackします。新規contentではstep routingを増やしません。

検索はMarkdown本文/heading/title/summary/keywords、日本語code alias、Python symbol、variable alias/identifier、source pathを対象にします。symbol/variable検索は対象pageへ遷移し、該当implementation disclosureだけを自動openします。

## Development

Node.js 24 LTSを使用します。

```bash
cd guide
npm ci
npm run dev
```

標準quality gate:

```bash
python3 tools/content_contract.py --check
npm run check
npm run e2e
```

`npm run check`はsource contract、ESLint、TypeScript、Vitest、production buildを検証します。production build後は`tools/bundle_budget.py`が`dist/**/*.js`の実bytesを検査し、1 chunkでも**500,000 bytes**を超えればfailします。Vite warning thresholdを引き上げて隠しません。

`npm run e2e`はdesktopと320px mobileで次を検証します。

- overview/replay/PPOをclickなしで読める。
- implementation appendixは通常collapsedでkeyboard disclosure可能。
- 日本語alias / Python identifier検索から同じsymbolへ到達する。
- exact-SHA source link。
- local scroll regionをkeyboardで操作可能。
- document-level horizontal overflow 0。
- light/darkでaxe `serious`/`critical` violation 0。
- reduced motionでも情報欠落なし。

`node_modules/`, `dist/`, `coverage/`, `playwright-report/`, `test-results/`, `.generated/`はcommitしません。

## GitHub Pages deployment

`.github/workflows/ci.yml`が品質ゲートです。`.github/workflows/deploy-guide.yml`は同一Repositoryの`main` pushに対するCI success後だけ、`workflow_run.head_sha`をexact checkoutしてPagesへdeployします。

production deployでもsource contractと500,000-byte bundle budgetを再実行します。公開済み判定はworkflow fileの存在ではなく、deployment成功と実公開URL smokeで確認します。

```bash
GUIDE_PUBLIC_URL="https://shuntatsu.github.io/trade_rl/" \
GUIDE_SOURCE_REV="<deployed-40-char-sha>" \
npm run e2e:public
```

public smokeは実公開URLで、Markdown-first replay/PPO、symbol search、exact source link、theme、320px overflowを確認します。

## Content / UI boundary

- `content/pages/*.md`: 人間が読む説明の主役。
- `content/meta/*.json`: traceabilityとsearchに必要なmachine metadata。
- `src/components/MarkdownArticle.tsx`: safe Markdown renderer。
- `src/components/ImplementationReferenceAppendix.tsx`: secondary implementation disclosure。
- `src/components/CodeInspector.tsx`: 実identifier/path/signature/test/sourceの詳細表示。
- `src/content/documentSchema.ts`: `document-guide-v2` / role / metadata runtime parser。
- `src/content/codeSymbols.ts`: compact runtime AST indexのparse/lookup。
- `src/content/search.ts`: Markdown + alias + symbol search。
- `tools/code_symbols.py`: full/runtime AST index生成。
- `tools/content_contract.py`: docs/code/content stalenessをfail closedで検出。
- `tools/bundle_budget.py`: production JS chunkを500,000-byte budget内へ固定。

Guideは正本の代替ではありません。説明とsource/docsが食い違った場合はGuide側を修正します。
