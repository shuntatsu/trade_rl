# Trade RL Interactive Guide

This guide is non-authoritative.
Technical truth lives in docs/architecture/* and docs/research/current-status.md.
Human-facing content lives in guide/content/topics/*.json.

このGuideは、人間がTrade RLの全体像・データフロー・責務境界・実験手順・現在の研究状態を視覚的かつ対話的に理解するための説明層です。

技術仕様や研究状態の正本を置き換えません。実装と説明が食い違う場合は、`docs/architecture/`、`docs/research/current-status.md`、現行source、contract testsを正本として確認します。

## 開発

Node.js 24 LTSを使用します。React / Vite / Tailwindの依存関係とビルドはこの`guide/` workspace内に閉じ込めます。

```bash
cd guide
npm ci
npm run dev
```

標準のGuide quality gate:

```bash
npm run check
npm run e2e
```

`npm run check` はsource-doc freshness、ESLint、TypeScript、Vitest、production buildを順に検証します。`npm run e2e` はChromiumでdesktop/mobileの主要操作・accessibility・visual evidenceを確認します。Light/Darkの両テーマでaxeの`serious`/`critical` violationを0件に保つことを必須とします。

GitHub Actionsでは既存Python gateの`Lean Core`と独立した`Human Guide` jobで同じ検査を実行します。Playwright reportやscreenshotはCI artifactとして短期保持します。`guide/node_modules/`、`guide/dist/`、`guide/coverage/`、`guide/playwright-report/`、`guide/test-results/`はroot `.gitignore` で除外し、current treeへcommitしません。

## GitHub Pages deployment

`.github/workflows/ci.yml` は引き続き品質ゲートです。`.github/workflows/deploy-guide.yml` は、同一Repositoryの `main` pushに対する `CI` がsuccessになった場合だけ、その `workflow_run.head_sha` をexact checkoutして `guide/dist/` だけをGitHub Pagesへdeployします。PR、fork、failed/cancelled CIからPages writeを実行しません。

Pagesの一回限りのRepository設定はGit tree外です。公開を有効にする場合はGitHubの `Settings → Pages → Build and deployment → Source` を **GitHub Actions** に設定します。この設定をread-backでき、実deployと公開URL smokeが成功するまで「公開済み」とは扱いません。

生成済み`dist/`はcommitしません。deployment buildでも`source-check`を再実行し、正本とのfingerprintがstaleならpublishをfail-closedに停止します。

公開後の検証はローカルE2Eとは分離します。

```bash
GUIDE_PUBLIC_URL="https://shuntatsu.github.io/trade_rl/" npm run e2e:public
```

`e2e:public` は実公開URLに対してroot表示、hash/search navigation、theme、正本link、320px overflowを確認します。通常の `npm run e2e` はlocal Vite serverを検証し、public smokeを実行しません。

## 内容を更新する

説明内容は`content/topics/*.json`を更新し、UI componentへ研究固有の説明を直接埋め込みません。

正本docsの関連sectionを変更すると、topicに保存されたSHA-256 fingerprintが古くなり`npm run source-check`が失敗します。失敗を一括承認してはいけません。影響するtopicと正本sectionを読み直して説明を更新し、確認したtopicだけ明示的にrefreshします。

```bash
python3 tools/content_contract.py --refresh execution-economics
npm run source-check
```

複数topicを確認した場合もIDを列挙します。

```bash
python3 tools/content_contract.py --refresh overview data-flow architecture
```

`--refresh`はsource sectionのfingerprintだけを更新します。正本の変更が人間向け説明へ影響する場合は、先にtopicのcopy/visualization dataを直してからrefreshしてください。

## 内容とUIの境界

- `content/topics/*.json`: 人間向けの説明、検索語、可視化data、正本traceability。
- `src/visualizations/`: 汎用的な図の描画・interaction。研究固有の真実を持たない。
- `src/components/` / `src/app/`: navigation、search、theme、layout。
- `tools/content_contract.py`: 正本とのstalenessをfail-closedで検出。

Light/Darkは同じcontentとinformation architectureを使います。低コントラスト/低彩度の見た目を優先しますが、可読性・focus・status識別はaccessibility gateを下回らせません。
