# Trade RL Interactive Guide

This guide is non-authoritative.
Technical truth lives in docs/architecture/* and docs/research/current-status.md.
Human-facing content lives in guide/content/topics/*.json.

このGuideは、人間がTrade RLの全体像と実装を日本語で追跡するための非正本説明層です。**日本語を主表示**し、実identifier・型名・ファイルpathはsource照合の副表示として残します。Pythonの関数名・変数名・公開API自体は日本語へrenameしません。

技術仕様や研究状態の正本を置き換えません。実装と説明が食い違う場合は、`docs/architecture/`、`docs/research/current-status.md`、現行source、contract testsを正本として確認します。

公開URL: <https://shuntatsu.github.io/trade_rl/>

## コード連動契約

GuideはPython sourceをimport/executeせず、`guide/tools/code_symbols.py` がASTからcode-symbol indexを生成します。`guide/.generated/code-symbols.json` はsource-check・review用の**full index**、`guide/.generated/code-symbols-runtime.json` は現行topicの `code_references` が参照するunique symbolだけを保持する**browser配布用index**です。どちらもcommitせず、`guide/.gitignore` で除外します。各entryは完全修飾symbol、kind、source path/line、signature、source SHA-256、同一function scopeの主要local名、**exact revision**を保持します。

runtime indexはfull indexを置き換えるものではありません。full AST indexを先に構築した上でreview済みCodeReferenceの参照集合へ射影し、参照symbolがfull indexに存在しなければ生成時にfail-closedで停止します。browserへ未参照の全symbolを配布しないことで、traceabilityを維持したままbundleを小さく保ちます。

各topicの `code_references` は人間がレビューした日本語aliasと実symbolの対応です。`content_contract.py --check` はmissing symbol、wrong kind、stale source digest、存在しないvariable、missing/escaping test pathをfail-closedで拒否します。

sourceを読み直して説明がまだ正しいことを確認したtopicだけ、明示的にrefreshします。

```bash
python3 guide/tools/content_contract.py --refresh-code <reviewed-topic-id>
python3 guide/tools/content_contract.py --refresh <reviewed-topic-id>
python3 guide/tools/content_contract.py --check
```

`--refresh-code` は検証済みCodeReferenceのsource digestだけを更新し、日本語説明や関係を自動生成しません。`--refresh` もMarkdown section fingerprintだけを更新します。どちらもCIを通すための一括承認として使いません。

sequenceとcode-mapのactor、message、edge、`data-flow` / `calls` relationは**review済みcontent**です。ASTから自動call graphを推測して正本扱いしません。実在性とsource freshnessだけを生成indexで検証します。

source linkはfloating `main` ではなく、そのGuide buildが説明している **exact revision** に固定します。CIとPages deploymentは `GUIDE_SOURCE_REV` としてcheckoutしたexact SHAを渡し、GitHub source URLもそのSHAを使用します。

## 開発

Node.js 24 LTSを使用します。React / Vite / Tailwindの依存関係とビルドはこの`guide/` workspace内に閉じ込めます。

```bash
cd guide
npm ci
npm run dev
```

標準のGuide quality gate:

```bash
python3 tools/content_contract.py --check
npm run check
npm run e2e
```

`npm run check` はsource-doc/code freshness、ESLint、TypeScript、Vitest、production buildを順に検証します。production build後は `tools/bundle_budget.py` が `dist/**/*.js` の**実ファイルサイズ**を検査し、1 chunkでも500,000 bytesを超えればfailします。Viteのwarning閾値を引き上げて大きなbundleを隠す運用はしません。`npm run e2e` はChromiumでdesktop/mobileの主要操作・accessibility・visual evidenceを確認します。Light/Darkの両テーマでaxeの`serious`/`critical` violationを0件に保つことを必須とします。

GitHub Actionsでは既存Python gateの`Lean Core`と独立した`Human Guide` jobで同じ検査を実行します。Human Guide job全体へPR headの `GUIDE_SOURCE_REV` を渡すため、unit/buildだけでなくPlaywright dev serverが再生成するcode indexも同じSHAへbindされます。Playwright reportやscreenshotはCI artifactとして短期保持します。

`guide/node_modules/`、`guide/dist/`、`guide/coverage/`、`guide/playwright-report/`、`guide/test-results/`はroot `.gitignore`、`guide/.generated/`は `guide/.gitignore` で除外し、current treeへcommitしません。

## GitHub Pages deployment

`.github/workflows/ci.yml` は品質ゲートです。`.github/workflows/deploy-guide.yml` は、同一Repositoryの `main` pushに対する `CI` がsuccessになった場合だけ、その `workflow_run.head_sha` をexact checkoutして `guide/dist/` だけをGitHub Pagesへdeployします。PR、fork、failed/cancelled CIからPages writeを実行しません。

Repository Pages sourceは **GitHub Actions** を使用します。公開済み判定はworkflow fileの存在ではなく、Pages state、deployment成功、実公開URL smokeを合わせて確認します。

生成済み`dist/`はcommitしません。deployment buildでも`source-check`とbundle budgetを再実行し、正本とのfingerprintがstale、またはJavaScript chunkがbudget超過ならpublishをfail-closedに停止します。

公開後の検証はローカルE2Eとは分離します。

```bash
GUIDE_PUBLIC_URL="https://shuntatsu.github.io/trade_rl/" \
GUIDE_SOURCE_REV="<deployed-40-char-sha>" \
npm run e2e:public
```

`e2e:public` は実公開URLに対してreplay sequence、hash/search navigation、theme、exact-SHA source link、320px overflowを確認します。通常の `npm run e2e` はlocal Vite serverを検証し、public smokeを実行しません。

## 内容を更新する

説明内容は`content/topics/*.json`を更新し、UI componentへ研究固有の説明を直接埋め込みません。

正本docsの関連sectionを変更すると、topicに保存されたSHA-256 fingerprintが古くなり`npm run source-check`が失敗します。影響するtopicと正本sectionを読み直して説明を更新し、確認したtopicだけ明示的にrefreshします。Python実装symbolを変更した場合も同様に、該当CodeReferenceと日本語説明を再確認してから `--refresh-code` を実行します。

## 内容とUIの境界

- `content/topics/*.json`: 日本語中心の説明、検索語、review済み可視化model、正本traceability、CodeReference。
- `src/visualizations/`: 汎用的な図の描画・interaction。研究固有の真実を持たない。
- `src/components/CodeInspector.tsx`: 日本語説明を先に、実identifier/path/signature/test/sourceを副表示する。
- `src/content/codeSymbols.ts`: compact runtime AST indexのparse/lookup。
- `src/components/` / `src/app/`: navigation、search、theme、layout、URL selection。
- `tools/code_symbols.py`: production moduleを実行せずfull AST indexを生成し、browser用runtime indexをreview済み参照symbolへ射影する。
- `tools/content_contract.py`: docs/codeとのstalenessをfail-closedで検出。
- `tools/bundle_budget.py`: production buildのJavaScript chunkを500,000-byte budget内へ固定する。

Desktopではsequence/code-mapと一つのinspectorを並べ、mobileではpan/zoomを必須にせずstep-throughまたは選択node近傍から同じ実装情報へ到達できるようにします。selectionを色だけで表現せず、keyboardとテキストoutlineでも意味を追える状態を維持します。
