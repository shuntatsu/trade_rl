# Interactive Human Guide v1 Design

Status: Active

## Objective

`trade_rl` の現在の正本docsを置き換えず、人間が「何をするシステムか」「データがどう流れるか」「各責務がどう分かれるか」「研究はいまどこまで進んだか」を視覚的・対話的に理解できる、build不要の静的Interactive Guideを追加する。

Guideは**説明層でありauthorityではない**。技術仕様・研究状態の正本は引き続き `docs/architecture/*` と `docs/research/current-status.md` に置く。

## Non-goals

- `docs/` のcurrent-only authority構造を別のdocs siteへ置き換えない。
- GitHub Pages公開をv1の必須条件にしない。
- React/Vite/npm等のfrontend package/toolchainを導入しない。
- Guide内に第二の研究結果・設定authorityを作らない。
- Guideから取引・学習・研究Runを実行しない。
- backend/API/server/databaseを追加しない。
- profitability、winner、Production authorizationを示唆するUIを作らない。
- 正本docsの内容を自動要約して無検証で公開しない。

## Audience

主対象は、Repositoryを初めて読む人、研究/実装の全体像を短時間で理解したい人、コードを読む前に責務と研究状態を把握したい開発者・レビュアーである。

Agent向けの最短routingは既存 `AGENTS.md` / `docs/AGENTS.md` を維持する。GuideはAgent authorityではない。

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
  index.html
  README.md
  assets/
    app.css
    core.js
    app.js
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
  generated/
    guide-data.js
  tools/
    build.py
  tests/
    core.test.cjs
```

Repositoryのroot `README.md` と `docs/README.md` はGuideへの入口を追加するが、「Guideは非正本」「正本はdocs」という境界を明記する。

## Content authority / freshness contract

### Authoring source

人間向け説明内容のauthoring sourceは `guide/content/topics/*.json` とする。UI copyを `index.html` / `app.js` に散在させない。

各topicは少なくとも次を持つ。

```json
{
  "id": "execution-economics",
  "title": "実行コスト",
  "summary": "取引コストがどこで決まり、どう計上されるか",
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

### Section fingerprint

`guide/tools/build.py` はMarkdownをheading単位で抽出し、topicが宣言した `path + heading` のsection SHA-256を再計算する。

- headingが消えた → fail closed。
- pathが消えた → fail closed。
- digestが変わった → Guide staleとしてfail closed。
- authorがGuideを読み直し、説明内容を必要に応じて更新した後だけ明示的なrefresh commandでdigestを更新できる。

これにより「docsが変わったらGuideも確認する」を機械的に強制する。

### Generated runtime data

ブラウザ表示用 `guide/generated/guide-data.js` はcontent JSONから決定論的に生成する。Guide自体はbuild不要で、`guide/index.html` を直接開いても動作する。

生成物はcommitし、CIではgeneratorを再実行した結果とbyte-equalであることを確認する。stale generated dataを許可しない。

## Runtime architecture

外部runtime dependencyは使わない。

```text
content JSON
    ↓ build.py
reviewed source-section fingerprints
    ↓
generated/guide-data.js
    ↓
index.html
 ├─ assets/core.js    pure state/transform helpers
 ├─ assets/app.js     DOM/event/rendering
 └─ assets/app.css    light/dark/responsive tokens
```

`core.js` はUMD/CommonJS-compatibleなpure helperとして、browserとNode built-in test runnerの両方から利用できるようにする。DOM処理は `app.js` に限定する。

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
- Execution economics explorer: fee/spread/participation/borrowとzero-overlayの関係を図示。値変更simulationではなく、authority pathのON/OFF可視化を行う。
- PPO Observation v2 explorer: vector segmentを選択すると意味、shape contribution、禁止情報を表示。
- Experiment loop: baseline / prereg / execute / verify / decide / freezeを状態遷移として辿れる。
- Research status: `verified / pending / not claimed / limitation` を明確に分離。

## Visual design

### Light mode

- background: cool off-white / very light blue-gray。
- surfaces: backgroundとの差は小さく、border + small tonal change中心。
- primary textは濃紺だがpure blackを避ける。
- accentは低彩度blue/teal中心。

### Dark mode

- background: blue-charcoal。pure blackは使わない。
- surfaces: backgroundよりわずかに明るい程度。
- primary text: pure whiteではなくsoft blue-gray。
- muted textはWCAG AAを下回らない範囲で輝度差を抑える。
- accentの彩度を下げ、選択状態だけ少し強める。
- borders/shadowsは極めて弱く、カード境界を色差だけで強調しすぎない。

### Contrast policy

「低コントラスト」は可読性を落とす意味ではない。

- normal text contrast: WCAG AA 4.5:1以上。
- large text / nonessential decorationは穏やかにする。
- interaction focus ringは明確に残す。
- statusは色だけでなくicon + textで表現する。

### Theme behavior

- defaultはOS `prefers-color-scheme`。
- UI toggleでlight/darkを切替可能。
- user selectionはlocal browser storageへ保存してよい。
- storage不可でもsystem/defaultへfail-safeする。

## Navigation / search

- 左navigationは7 topic + Home。
- narrow screenではdrawerへreflow。
- searchはtitle / summary / keywords / section headingsをlocal indexで検索。
- `Ctrl/Cmd+K` はGuide内にfocusがある場合だけsearchへ移動。
- hash routing (`#execution-economics` 等) で直接リンクできる。
- back/forward操作を壊さない。

## Accessibility

- semantic `nav/main/section/button` を使用。
- clickable diagram nodeはbutton相当のkeyboard target。
- selected stateをARIAで公開。
- focus-visibleを隠さない。
- animationは `prefers-reduced-motion` で停止/短縮。
- 320px幅でpage-level horizontal overflowを出さない。
- primary interactionはhover依存にしない。

## Update workflow

通常の実装変更時は次の流れにする。

1. 正本docsを更新する。
2. `uv run python guide/tools/build.py --check` がsource fingerprint mismatchでREDになる。
3. 対応topicを読み直して説明copy/visualization dataを更新する。
4. `uv run python guide/tools/build.py --refresh-sources` でreview済みsection digestを更新する。
5. `uv run python guide/tools/build.py --write` でgenerated JSを更新する。
6. Guide contract tests + JS tests + full CIを通す。

Guideだけ変更する場合も、source fingerprintを勝手に更新せず、source docsと説明の整合性を確認する。

## Invariants

- `guide/` は非正本である。
- source docsへのtraceabilityが全topicに存在する。
- generated dataはauthoring contentから決定論的に生成される。
- stale source fingerprint / missing source heading / stale generated JSはCI Greenにならない。
- Guide変更は `trade_rl/**` runtime behaviorを変えない。
- frontend package manager/dependencyを追加しない。
- Guideはnetworkなしで閲覧できる。
- light/dark双方を同じinformation architectureで提供する。

## Failure modes / risks

- Guideが第二の仕様書になる。
- 正本更新後もGuideが古いままGreenになる。
- generated JSだけ手編集されcontent sourceと乖離する。
- research metricを静的copyして古くする。
- visual emphasisが「verified baseline = profitable」のような誤解を生む。
- low-contrast化でaccessibilityを落とす。
- diagramがdesktopでは良いがmobileで読めない。
- mouse hoverだけで情報を出す。
- dark modeだけ別copy/別stateになり意味がずれる。
- UI logicが巨大化し、content updateのたびにJS変更が必要になる。

## Test Oracle

完了判定は少なくとも次を観測する。

- 全topic JSON schema/required fieldsがvalid。
- source path + headingが存在しsection digestが一致。
- generated dataがfresh generationとbyte-equal。
- `node --check`でJS syntax Green。
- Node built-in testsでrouting/search/step state/pure transforms Green。
- architecture testでGuide非正本境界、root/docs routing、Active docs lifecycleを確認。
- light/dark desktopで全primary interactionをbrowser実行。
- mobile-widthでnavigation/reflow/diagram/detailを確認。
- keyboard-onlyでtopic selection/search/theme/diagram selectionが可能。
- WCAG AA text contrastをtoken-levelで確認。
- final diffにtemporary screenshot/debug/generated stray fileがない。
- exact-head repository CIがGreen。

## Acceptance Criteria

1. Repository rootからInteractive Guideへ到達できる。
2. `guide/index.html` はbuild/serverなしでも主要機能が使える。
3. Light/Darkを備え、双方ともapproved low-contrast visual directionを保つ。
4. 全体像、Data Flow、Responsibilities、Execution Economics、PPO Observation v2、Experiment Loop、Research Statusを対話的に理解できる。
5. 各topicからsource authorityへ遷移できる。
6. content更新はstructured files中心で行え、layout JSの編集を原則不要にする。
7. source authorityの関連section変更をCIが検出する。
8. stale generated dataをCIが検出する。
9. 320pxからdesktopまで主要操作が成立する。
10. keyboard/focus/reduced-motion/contrastのaccessibility contractを満たす。
11. `trade_rl/**` runtime/public API/artifact identityへ変更を加えない。
12. 実装完了時、Active spec/planはcurrent treeから削除し、恒久的なGuide maintenance ruleだけを `docs/README.md` / `docs/AGENTS.md` に残す。
