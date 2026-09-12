# Interactive Human Guide v1 Design

Status: Active

## Objective

`trade_rl` の現在の正本docsを置き換えず、人間が「何をするシステムか」「データがどう流れるか」「各責務がどう分かれるか」「研究はいまどこまで進んだか」を、視覚的・対話的に理解できるInteractive Guideを追加する。

Guideは**説明層でありauthorityではない**。技術仕様・研究状態の正本は引き続き `docs/architecture/*` と `docs/research/current-status.md` に置く。

## Non-goals

- `docs/` のcurrent-only authority構造を別のdocs siteへ置き換えない。
- GitHub Pages公開をv1の必須条件にしない。
- Guide内に第二の研究結果・設定authorityを作らない。
- Guideから取引・学習・研究Runを実行しない。
- backend/API/server/databaseを追加しない。
- profitability、winner、Production authorizationを示唆するUIを作らない。
- 正本docsの内容を自動要約して無検証で公開しない。
- production Python package `trade_rl/**` にfrontend責務を混ぜない。

## Audience

主対象は、Repositoryを初めて読む人、研究/実装の全体像を短時間で理解したい人、コードを読む前に責務と研究状態を把握したい開発者・レビュアーである。

Agent向けの最短routingは既存 `AGENTS.md` / `docs/AGENTS.md` を維持する。GuideはAgent authorityではない。

## Technology

GuideはRepository rootの独立frontend workspaceとして実装する。

- Node.js 24 LTS
- React 19.3
- TypeScript
- Vite 8 current supported line
- Tailwind CSS 4.3 + `@tailwindcss/vite`
- Vitest + React Testing Library
- Playwright Chromium for browser/UX verification
- Python 3.12 standard library for authoritative-doc source fingerprint validation

Node dependenciesは `guide/package.json` / `guide/package-lock.json` に閉じ込める。root Python package metadataへfrontend dependencyを混ぜない。

## Information architecture

Guide v1は次の7領域を持つ。

1. **Home / 全体像** — Market data → Build → Strategy/PPO → Execution → Experiment の関係。
2. **Data Flow** — source freeze、point-in-time data、feature build、Dataset artifact、Replayまでの流れ。
3. **Architecture / Responsibilities** — data / strategy / risk / execution / evaluation の責務境界。
4. **Execution Economics** — Dataset-authoritative fee/spread/participation/borrow、zero overlay、二重課金禁止。
5. **PPO Observation v2** — local values / availability / staleness / intent / weight の順序と、global/symbol IDを入れない理由。
6. **Controlled Experiment Loop** — Study → baseline EvidenceSet → preregistration → one-factor experiment → verification → decision → freeze。
7. **Research Status** — 完了、未開始、未証明、残存limitationsを分けて表示。

各領域には「正本を見る」を必ず持たせる。

## Repository placement

`docs/` はcurrent authority専用のまま維持するため、GuideはRepository rootの `guide/` に置く。

```text
guide/
  README.md
  package.json
  package-lock.json
  index.html
  vite.config.ts
  tsconfig.json
  src/
    main.tsx
    app/
      App.tsx
      routes.ts
      theme.ts
    components/
      AppShell.tsx
      Sidebar.tsx
      SearchPalette.tsx
      SourceLinks.tsx
      TopicHeader.tsx
    visualizations/
      ArchitectureFlow.tsx
      FlowStepper.tsx
      ObservationVector.tsx
      EconomicsAuthorityPath.tsx
      ExperimentLoop.tsx
      ResearchStatusBoard.tsx
    content/
      loadTopics.ts
      schema.ts
    styles/
      app.css
  content/
    manifest.json
    topics/
      overview.json
      data-flow.json
      architecture.json
      execution-economics.json
      ppo-observation-v2.json
      experiment-loop.json
      research-status.json
  tools/
    content_contract.py
  tests/
    setup.ts
    content.test.ts
    routing.test.tsx
    interactions.test.tsx
  e2e/
    guide.spec.ts
    accessibility.spec.ts
```

`guide/dist/`, Playwright reports/screenshots、coverageはgenerated outputでありcommitしない。

Repositoryのroot `README.md` と `docs/README.md` はGuideへの入口を追加するが、「Guideは非正本」「正本はdocs」という境界を明記する。

## Content architecture

### Authoring source

人間向け説明内容のauthoring sourceは `guide/content/topics/*.json` とする。説明copy・diagram node/edge・step・statusをReact componentへ直書きしない。

各topicは少なくとも次を持つ。

```json
{
  "id": "execution-economics",
  "title": "実行コスト",
  "summary": "取引コストがどこで決まり、どう計上されるか",
  "keywords": ["fee", "spread", "cost"],
  "source_sections": [
    {
      "path": "docs/research/current-status.md",
      "heading": "Execution economics contract",
      "sha256": "<reviewed-section-digest>"
    }
  ],
  "sections": [],
  "visualization": {}
}
```

`src/content/loadTopics.ts` はViteのJSON module importを使い、manifest順でtyped topic dataへ変換する。JSON schema/型検査はruntime boundaryでもfail closedにする。

### Source-section fingerprint

`guide/tools/content_contract.py` はMarkdownをheading単位で抽出し、topicが宣言した `path + heading` のsection SHA-256を再計算する。

- headingが消えた → fail closed。
- pathが消えた → fail closed。
- digestが変わった → Guide staleとしてfail closed。
- authorがGuideを読み直し、説明内容を必要に応じて更新した後だけ明示的なrefresh commandでdigestを更新できる。

これにより「正本docsが変わったらGuideも確認する」を機械的に強制する。

### No generated runtime content

Viteが `guide/content/topics/*.json` を直接bundleする。説明内容から別のgenerated JSを作らない。

これにより更新時のauthority chainを次の1本にする。

```text
Authoritative docs
      ↓ reviewed fingerprint
Guide topic JSON
      ↓ typed import
React visualization
```

## React architecture

`App.tsx` はlayout compositionだけを担当し、巨大なone-file appにしない。

- `app/`: routing/theme/global state
- `components/`: navigation/search/common presentation
- `visualizations/`: visualization family単位のinteractive component
- `content/`: typed content loading/validation

Topicごとの特殊copyはJSONへ置き、componentはgeneric visualization kindをrenderする。

Heavy third-party visualization frameworkはv1では導入しない。主要図はsemantic HTML + responsive SVG/CSSで構成し、keyboard/touch selectionをReact stateで管理する。

## UX principles

優先順位は次の通り。

1. **わかりやすさ**
2. **インタラクティブな可視化**
3. **UX / navigation**
4. 情報量
5. 装飾

### Progressive disclosure

最初から全説明を見せない。

- first view: 全体像 + 現在地 + 3つ程度の入口。
- componentをclick/tap → detail panel。
- 「図で見る / リストで見る」を切替可能。
- flowはstepperで1段ずつ追える。
- deep technical detailsは折りたたみ。
- 各detailからsource docへ戻れる。

### Interactive visualization

最低限次を実装する。

- Architecture flow: component選択でresponsibility / input / output / not-ownedを表示。
- Data flow stepper: source → freeze → build → artifact → strategy → replay → evidence。
- Execution economics explorer: fee/spread/participation/borrowとzero-overlayのauthority pathを可視化する。値をいじって研究結果をsimulationするUIにはしない。
- PPO Observation v2 explorer: vector segmentを選択すると意味、shape contribution、禁止情報を表示。
- Experiment loop: baseline / prereg / execute / verify / decide / freezeを状態遷移として辿れる。
- Research status: `verified / pending / not claimed / limitation` を明確に分離。

## Visual design

ユーザー承認済み方向は、**情報密度は維持しつつ、コントラストと彩度を少し落とし、わかりやすさ・可視化・UXを優先**する。

### Light mode

- background: `#f5f7fa`
- primary surface: `#f9fafb`
- secondary surface: `#f1f4f8`
- primary text: `#334155`
- muted text: `#64748b`
- border: `#dbe3ec`
- primary accent: `#6f8faa`

### Dark mode

- background: `#17202b`
- primary surface: `#1d2835`
- secondary surface: `#243140`
- primary text: `#bec9d8`
- muted text: `#95a5b8`
- border: `#334254`
- primary accent: `#7899b6`

pure black / pure whiteをprimary surface/textに使わない。persistent content categoryの色は低彩度blue / teal / violet / amber / roseを使い、accent色はselection/focus/actionへ優先的に使う。

### Contrast policy

「低コントラスト」は可読性を落とす意味ではない。

- normal text contrast: WCAG AA 4.5:1以上。
- large text / nonessential decorationは穏やかにする。
- focus ringは明確に残す。
- statusは色だけでなくicon + textで表現する。
- selected stateはborder/colorだけでなく背景・label・ARIA stateで判別できる。

### Theme behavior

- defaultはOS `prefers-color-scheme`。
- UI toggleでlight/darkを切替可能。
- explicit selectionだけ`localStorage`へ保存する。
- storage不可でもsystem preferenceへfail-safeする。
- light/darkでcopy・topic state・意味論を分岐させない。

## Navigation / search

- 左navigationは7 topic + Home。
- narrow screenではcompact navigation/drawerへreflow。
- searchはtitle / summary / keywords / section headingsをlocal indexで検索。
- `Ctrl/Cmd+K` でsearch paletteを開く。
- hash routing (`#execution-economics` 等) で直接リンクできる。
- browser back/forwardを壊さない。
- selected topic/detailはURL hashから復元可能にする。

React Routerはv1では追加せず、small hash-router hookで十分なsurfaceに限定する。

## Accessibility

- semantic `header/nav/main/section/aside/button` を使用。
- diagram nodeはnative buttonまたは同等のkeyboard semantics。
- selected stateを`aria-pressed` / `aria-current`等で公開。
- focus-visibleを隠さない。
- animationは `prefers-reduced-motion` で停止/短縮。
- 320px幅でpage-level horizontal overflowを出さない。
- primary interactionはhover依存にしない。
- Playwrightでkeyboard-only primary pathを実行する。

## Update workflow

通常のproduction/docs変更時は次の流れにする。

1. 正本docsを更新する。
2. `uv run python guide/tools/content_contract.py --check` が関連section fingerprint mismatchならREDになる。
3. 対応topic JSONを読み直して説明copy/visualization dataを更新する。
4. `uv run python guide/tools/content_contract.py --refresh-sources` でreview済みsection digestを更新する。
5. `npm --prefix guide run typecheck` / `test` / `build` でGuideを検証する。

Guideだけ変更する場合も、source fingerprintを勝手にrefreshせず、正本docsとの整合性を確認する。

## Development commands

```bash
npm --prefix guide ci
npm --prefix guide run dev
npm --prefix guide run typecheck
npm --prefix guide run test
npm --prefix guide run build
npm --prefix guide run e2e
uv run python guide/tools/content_contract.py --check
```

`package-lock.json`をcommitし、CIは`npm ci`で再現する。

## CI integration

既存Python `Lean Core` gateを弱めない。Guide用Node検証を同workflowの独立job `Human Guide` として追加し、PR/main pushで実行する。

`Human Guide` jobは少なくとも次を実行する。

```text
checkout exact head
setup Node 24 LTS
npm --prefix guide ci
source fingerprint check
TypeScript typecheck
Vitest
Vite production build
Playwright Chromium primary UX / responsive / theme / keyboard tests
```

Node jobのfailureでPython core testsをskipしない。二つのjobは独立して走らせる。

## Invariants

- `guide/` は非正本である。
- source docsへのtraceabilityが全topicに存在する。
- stale source fingerprint / missing source headingはCI Greenにならない。
- explanation copy/visualization modelはstructured topic JSONをsingle authoring sourceにする。
- React componentはtopic固有research truthを埋め込まない。
- Guide変更は `trade_rl/**` runtime behaviorを変えない。
- Node dependenciesは`guide/`に閉じる。
- production Guideはremote runtime/CDNなしでbuildできる。
- light/dark双方を同じinformation architectureで提供する。

## Failure modes / risks

- Guideが第二の仕様書になる。
- 正本更新後もGuideが古いままGreenになる。
- componentに説明copyが散ってJSON更新だけでは済まなくなる。
- research metricを静的copyして古くする。
- visual emphasisが「verified baseline = profitable」のような誤解を生む。
- low-contrast化でaccessibilityを落とす。
- diagramがdesktopでは良いがmobileで読めない。
- mouse hoverだけで情報を出す。
- dark modeだけ別copy/別stateになり意味がずれる。
- React state/effectが複雑化し、不要なre-renderやURL/state二重authorityを作る。
- Node dependency更新がPython core開発を不必要にblockする。

## Test Oracle

完了判定は少なくとも次を観測する。

- 全topic JSON required fields / visualization kind / referenced IDsがvalid。
- source path + headingが存在しsection digestが一致。
- TypeScript typecheck Green。
- Vitestでcontent loading、routing、search、theme、stepper/selection stateがGreen。
- Vite production build Green。
- Playwrightでlight/dark desktopの全primary interactionが成立。
- Playwrightで320px narrow layoutにpage-level horizontal overflowがない。
- keyboard-onlyでtopic selection/search/theme/diagram selectionが可能。
- `prefers-reduced-motion`でも情報が欠落しない。
- WCAG AA text contrastをtoken-level testで確認。
- architecture testでGuide非正本境界、root/docs routing、Active docs lifecycleを確認。
- final diffに`dist/`、Playwright report、debug output、temporary screenshotがない。
- exact-head Python CIとHuman Guide CIがGreen。

## Acceptance Criteria

1. Repository rootからInteractive Guideへ到達できる。
2. `npm --prefix guide run dev` でローカルGuideを起動でき、`npm --prefix guide run build` でself-contained static asset bundleを生成できる。
3. Light/Darkを備え、双方ともapproved low-contrast visual directionを保つ。
4. 全体像、Data Flow、Responsibilities、Execution Economics、PPO Observation v2、Experiment Loop、Research Statusを対話的に理解できる。
5. 各topicからsource authorityへ遷移できる。
6. content更新は`guide/content/topics/*.json`中心で行え、layout component変更を原則不要にする。
7. source authorityの関連section変更をCIが検出する。
8. malformed/duplicate/broken content referenceをCIが検出する。
9. 320pxからdesktopまで主要操作が成立する。
10. keyboard/focus/reduced-motion/contrastのaccessibility contractを満たす。
11. `trade_rl/**` runtime/public API/artifact identityへ変更を加えない。
12. Node toolchainはGuide workspaceに隔離され、既存Python Lean Core gateを弱めない。
13. 実装完了時、Active spec/planはcurrent treeから削除し、恒久的なGuide maintenance ruleだけを `docs/README.md` / `docs/AGENTS.md` に残す。
