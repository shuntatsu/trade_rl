# Japanese Code Explorer Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GitHub PagesのHuman Guideを、日本語中心の説明から実ファイル・実関数・主要変数・関連テストまで追跡できるコード連動型実装エクスプローラへ変更する。

**Architecture:** 既存React/Vite Guideを維持し、Python ASTで作るuntracked symbol indexをcode freshnessのoracleとして追加する。sequence/code-mapの関係は人間がレビューしたcontent modelを正本とし、ASTからcall graphを推測しない。UIは日本語主表示、実identifier副表示、exact commit SHA固定source link、desktop diagram + inspector / mobile step-throughで構成する。

**Tech Stack:** Python 3 AST / hashlib / pathlib、React 19、TypeScript 6、Vite 8、Vitest、Testing Library、Playwright、axe-core、既存CSS/Tailwind。新しいlayout libraryは追加しない。

**Spec:** `docs/specs/2026-09-13-japanese-code-explorer-design.md`

## Global Constraints

- `trade_rl/**` runtime semanticsとPython identifierを変更しない。
- Guideは非正本。architecture/research truthは `docs/architecture/*` と `docs/research/current-status.md`、implementation truthはcurrent source/tests。
- PR #513 merge後の研究状態を保持し、profitability / winner / final-test / live authorizationを新たに主張しない。
- `guide/.generated/code-symbols.json` は生成物でありcommitしない。
- production moduleをsymbol inspectionのためimport/executeしない。
- code mapはreview済み `data-flow` / `calls` relationだけを表示し、静的import graphと称さない。
- source linkはPages buildのexact SHAへ固定する。
- Node.jsは既存contractどおり24系。既存React/Vite依存を優先し、新dependencyを追加しない。
- axe serious/critical 0、320px page-level horizontal overflowなし、既存public Pages smokeを維持する。
- final tested PR HEADはfinal current `main` を含む。

---

### Task 1: ArchitectureFlowの偽矢印をRED→GREENで修正する

**Files:**
- Create: `guide/src/visualizations/graphLayout.ts`
- Create: `guide/tests/architecture-flow.test.tsx`
- Modify: `guide/src/visualizations/ArchitectureFlow.tsx`
- Modify: `guide/src/content/schema.ts`

**Interfaces:**
- Produces: `layerDirectedGraph(nodeIds: readonly string[], edges: readonly DirectedEdge[]): GraphLayout`
- Produces: `DirectedEdge = { from: string; to: string }`
- Invariant: rendererは `visualization.edges` に存在しない矢印を生成しない。

- [ ] **Step 1: 現行bugを再現するcomponent testを書く**

```tsx
it("renders only declared architecture edges even when node order disagrees", () => {
  const visualization = {
    kind: "architecture" as const,
    nodes: [node("data"), node("integrations"), node("strategies")],
    edges: [
      { from: "integrations", to: "data" },
      { from: "data", to: "strategies" },
    ],
  };

  render(<ArchitectureFlow visualization={visualization} />);

  expect(screen.getByTestId("edge-integrations-data")).toBeInTheDocument();
  expect(screen.getByTestId("edge-data-strategies")).toBeInTheDocument();
  expect(screen.queryByTestId("edge-data-integrations")).not.toBeInTheDocument();
});
```

- [ ] **Step 2: REDを確認する**

Run:

```bash
npm --prefix guide run test -- --run guide/tests/architecture-flow.test.tsx
```

Expected: 現行`ArchitectureFlow`にedge DOMがなく、またnode配列順に接続するためFAIL。

- [ ] **Step 3: deterministic DAG layoutのunit testを書く**

```ts
it("layers a directed graph without inventing adjacency", () => {
  expect(
    layerDirectedGraph(
      ["data", "integrations", "strategies"],
      [
        { from: "integrations", to: "data" },
        { from: "data", to: "strategies" },
      ],
    ).layers,
  ).toEqual([["integrations"], ["data"], ["strategies"]]);
});
```

cycleは `directed graph contains a cycle` でfailさせる。

- [ ] **Step 4: 最小layout helperとedge-driven rendererを実装する**

`graphLayout.ts` はKahn方式でindegreeを計算し、同rank内は元node orderで安定化する。`ArchitectureFlow.tsx` はlayout順にnodeを置き、edgeは `visualization.edges.map(...)` だけから描画する。各edgeへ `data-testid={\`edge-${from}-${to}\`}` を付ける。

- [ ] **Step 5: targeted testsをGREENにする**

```bash
npm --prefix guide run test -- --run guide/tests/architecture-flow.test.tsx guide/tests/content.test.ts
```

- [ ] **Step 6: commitする**

```bash
git add guide/src/visualizations/graphLayout.ts guide/src/visualizations/ArchitectureFlow.tsx guide/src/content/schema.ts guide/tests/architecture-flow.test.tsx guide/tests/content.test.ts
git commit -m "fix: render declared Guide architecture edges"
```

---

### Task 2: Python ASTからexact symbol・主要local名・source digestを生成する

**Files:**
- Create: `guide/tools/code_symbols.py`
- Create: `tests/architecture/test_human_guide_code_symbols.py`
- Modify: `.gitignore`
- Modify: `guide/package.json`

**Interfaces:**
- Produces: `build_symbol_index(root: Path, *, revision: str) -> dict[str, object]`
- Produces: `write_symbol_index(root: Path, output: Path, *, revision: str) -> None`
- Generated schema: `guide-code-symbols-v1`
- `CodeSymbol`: `qualified_name`, `kind`, `path`, `start_line`, `end_line`, `signature`, `source_sha256`, `local_names`

- [ ] **Step 1: AST indexのRED testsを書く**

```python
def test_index_resolves_replay_risk_execution_and_local_names() -> None:
    index = build_symbol_index(ROOT / "trade_rl", revision="a" * 40)
    symbols = {entry["qualified_name"]: entry for entry in index["symbols"]}

    replay = symbols["trade_rl.evaluation.replay.run_single_symbol_replay"]
    assert replay["kind"] == "function"
    assert replay["path"] == "trade_rl/evaluation/replay.py"
    assert "desired_quantity" in replay["local_names"]
    assert "target_weight" in replay["local_names"]

    risk = symbols["trade_rl.risk.pretrade.PreTradeRisk.constrain"]
    assert risk["kind"] == "method"
    assert risk["path"] == "trade_rl/risk/pretrade.py"

    execution = symbols[
        "trade_rl.simulation.execution.MarketExecutor.execute_interval"
    ]
    assert execution["kind"] == "method"
```

別fixtureでmodule top-levelに `raise RuntimeError("must not execute")` を置き、AST index生成が成功することを確認する。nested functionのlocal nameが親へ漏れないtestも追加する。

- [ ] **Step 2: REDを確認する**

```bash
uv run pytest -q tests/architecture/test_human_guide_code_symbols.py
```

Expected: `guide.tools.code_symbols` が存在せずFAIL。

- [ ] **Step 3: AST collectorを実装する**

実装規則:

```python
SCHEMA_VERSION = "guide-code-symbols-v1"


def _source_digest(lines: list[str], start: int, end: int) -> str:
    text = "".join(lines[start - 1 : end]).replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

`FunctionDef` / `AsyncFunctionDef` / `ClassDef` を走査する。method qualified nameは `module.Class.method`、top-level functionは `module.function`。local name collectorはparametersと同一function scope内の`Store` targetsだけを集め、nested function/class/lambdaへ降りない。

- [ ] **Step 4: revision resolutionとJSON writerを実装する**

priorityは `--revision` → `GUIDE_SOURCE_REV` → `GITHUB_SHA` → `git rev-parse HEAD`。40桁hex以外はfailする。JSONはqualified name順に安定sortし、UTF-8 + `indent=2` + trailing newlineで書く。

- [ ] **Step 5: generated pathをignoreし、npm lifecycleへ接続する**

`.gitignore`へ追加:

```gitignore
guide/.generated/
```

`guide/package.json`へ追加:

```json
"code-index": "python3 tools/code_symbols.py --write .generated/code-symbols.json",
"predev": "npm run code-index",
"pretypecheck": "npm run code-index",
"pretest": "npm run code-index",
"prebuild": "npm run code-index"
```

- [ ] **Step 6: GREENとdeterminismを確認する**

```bash
uv run pytest -q tests/architecture/test_human_guide_code_symbols.py
npm --prefix guide run code-index
cp guide/.generated/code-symbols.json /tmp/code-symbols-a.json
npm --prefix guide run code-index
diff -u /tmp/code-symbols-a.json guide/.generated/code-symbols.json
```

- [ ] **Step 7: generated fileがtracked diffへ出ないことを確認してcommitする**

```bash
git status --short
git add .gitignore guide/package.json guide/tools/code_symbols.py tests/architecture/test_human_guide_code_symbols.py
git commit -m "feat: index Guide code symbols from Python AST"
```

---

### Task 3: CodeReferenceをfail-closedで検証しTypeScriptへ渡す

**Files:**
- Create: `guide/src/content/codeSymbols.ts`
- Modify: `guide/tools/content_contract.py`
- Modify: `guide/src/content/schema.ts`
- Modify: `tests/architecture/test_human_guide_content_contract.py`
- Modify: `guide/tests/content.test.ts`

**Interfaces:**
- Produces: `CodeReference` with `id`, `symbol`, `kind`, `source_sha256`, `label_ja`, `description_ja`, `variables`, `tests`
- Produces: `getCodeSymbol(qualifiedName: string): CodeSymbol | undefined`
- Produces CLI: `python3 guide/tools/content_contract.py --refresh-code <topic-id> [...]`

- [ ] **Step 1: Python contract RED testsを書く**

以下をそれぞれ独立testにする。

```python
def test_code_reference_rejects_missing_symbol(tmp_path: Path) -> None: ...
def test_code_reference_rejects_wrong_kind(tmp_path: Path) -> None: ...
def test_code_reference_rejects_stale_source_digest(tmp_path: Path) -> None: ...
def test_code_reference_rejects_unknown_local_name(tmp_path: Path) -> None: ...
def test_code_reference_rejects_missing_or_escaping_test_path(tmp_path: Path) -> None: ...
def test_refresh_code_updates_only_source_digest(tmp_path: Path) -> None: ...
```

`refresh_code` testでは `label_ja` / `description_ja` / `variables` がbyte-for-byte維持され、`source_sha256`だけがcurrent digestへ変わることをassertする。

- [ ] **Step 2: TypeScript parser RED testsを書く**

`guide/tests/content.test.ts`へ、duplicate `code_references[].id`、unknown sequence `code_ref`、empty `label_ja` をrejectするcaseを追加する。

- [ ] **Step 3: REDを確認する**

```bash
uv run pytest -q tests/architecture/test_human_guide_content_contract.py
npm --prefix guide run test -- --run guide/tests/content.test.ts
```

- [ ] **Step 4: `content_contract.py`へAST index validationを追加する**

`guide.tools.code_symbols.build_symbol_index`をimportし、`check_content()`の一回の走査でindexを構築する。topicごとにCodeReferenceを検証する。`--refresh-code` は明示topic idのみ許可し、unknown idはfailする。

- [ ] **Step 5: `schema.ts`へ型とparse contractを追加する**

追加型:

```ts
export type CodeReference = {
  id: string;
  symbol: string;
  kind: "function" | "class" | "method";
  source_sha256: string;
  label_ja: string;
  description_ja: string;
  variables: CodeVariableReference[];
  tests: string[];
};
```

`GuideTopic`へ `code_references: CodeReference[]` を追加する。既存topicは移行完了まで空配列を許す。

- [ ] **Step 6: generated index readerを実装する**

`codeSymbols.ts` は `../../.generated/code-symbols.json` をparseし、`Map<string, CodeSymbol>`をmodule scopeで一度だけ構築する。lookupはO(1)にする。

- [ ] **Step 7: GREENを確認する**

```bash
python3 guide/tools/content_contract.py --check
uv run pytest -q tests/architecture/test_human_guide_content_contract.py tests/architecture/test_human_guide_code_symbols.py
npm --prefix guide run test -- --run guide/tests/content.test.ts
npm --prefix guide run typecheck
```

- [ ] **Step 8: commitする**

```bash
git add guide/tools/content_contract.py guide/src/content/schema.ts guide/src/content/codeSymbols.ts guide/tests/content.test.ts tests/architecture/test_human_guide_content_contract.py
git commit -m "feat: bind Guide content to reviewed code symbols"
```

---

### Task 4: topic + selectionをhash routeで復元し、検索結果をsymbolまで拡張する

**Files:**
- Modify: `guide/src/app/useHashRoute.ts`
- Modify: `guide/src/content/search.ts`
- Modify: `guide/src/components/SearchPalette.tsx`
- Modify: `guide/tests/routing.test.ts`
- Modify: `guide/tests/search.test.ts`
- Modify: `guide/tests/interactions.test.tsx`

**Interfaces:**
- Produces: `GuideRoute = { topicId: string; step?: string; symbol?: string }`
- Produces: `normalizeHashRoute(hash, validIds, home): GuideRoute`
- Produces: `formatHashRoute(route): string`
- Produces: `GuideSearchResult = { topicId, step?, symbol?, title, subtitle, kind, score }`

- [ ] **Step 1: routing RED testsを書く**

```ts
expect(normalizeHashRoute("#implementation-replay?step=risk-constrain", ids, "overview"))
  .toEqual({ topicId: "implementation-replay", step: "risk-constrain" });
expect(normalizeHashRoute("#missing?step=x", ids, "overview"))
  .toEqual({ topicId: "overview" });
expect(formatHashRoute({ topicId: "code-map", symbol: "trade_rl.evaluation.replay.run_single_symbol_replay" }))
  .toBe("#code-map?symbol=trade_rl.evaluation.replay.run_single_symbol_replay");
```

既存 `#overview` が同じtopicへ復元されるcaseも残す。

- [ ] **Step 2: search RED testsを書く**

fixture topicに `希望保有数量 / desired_quantity` を持つCodeReferenceを置き、両queryが同一 `topicId + step` を返すことをassertする。

- [ ] **Step 3: REDを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/routing.test.ts guide/tests/search.test.ts
```

- [ ] **Step 4: route parser/formatter/hookを実装する**

query paramは `step` と `symbol` だけを受け入れ、両方来た場合はtopic種別に関係なく `step` を優先して `symbol` を捨てる。decode失敗時はhomeへ戻す。

- [ ] **Step 5: searchをtopic/code-ref/variableへ拡張する**

結果scoreはtopic title > code reference Japanese label > exact identifier/variable > body keywordの順にする。検索時に毎回CodeSymbol Mapを再構築しない。

- [ ] **Step 6: SearchPaletteを日本語主見出し + identifier/path副表示へ変更する**

`choose(result)` は `onNavigate({ topicId, step, symbol })` を呼ぶ。dialog `aria-label` も `ガイド検索` へ日本語化する。

- [ ] **Step 7: GREENを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/routing.test.ts guide/tests/search.test.ts guide/tests/interactions.test.tsx
```

- [ ] **Step 8: commitする**

```bash
git add guide/src/app/useHashRoute.ts guide/src/content/search.ts guide/src/components/SearchPalette.tsx guide/tests/routing.test.ts guide/tests/search.test.ts guide/tests/interactions.test.tsx
git commit -m "feat: navigate Guide code selections from search"
```

---

### Task 5: single-symbol replayを対話型sequence + inspectorとして実装する

**Files:**
- Create: `guide/content/topics/implementation-replay.json`
- Create: `guide/src/visualizations/SequenceDiagram.tsx`
- Create: `guide/src/components/CodeInspector.tsx`
- Create: `guide/src/content/sourceLinks.ts`
- Create: `guide/tests/sequence-diagram.test.tsx`
- Modify: `guide/content/manifest.json`
- Modify: `guide/src/content/schema.ts`
- Modify: `guide/src/visualizations/VisualizationRenderer.tsx`
- Modify: `guide/src/app/App.tsx`
- Modify: `guide/src/styles/app.css`
- Modify: `guide/tests/interactions.test.tsx`

**Interfaces:**
- New visualization kind: `sequence`
- `SequenceDiagram({ visualization, selectedStep, onSelectStep })`
- `CodeInspector({ referenceId, topic })`
- `buildCodeSourceUrl(symbol, repositoryUrl): string`

- [ ] **Step 1: schemaとinteractionのRED testsを書く**

unknown actor、duplicate message id、unknown `code_ref` をparseでrejectする。component testでは `risk-constrain` をclickするとinspectorに `リスク制約を適用`、`PreTradeRisk.constrain`、`trade_rl/risk/pretrade.py` が出ることをassertする。

- [ ] **Step 2: source URLのRED testを書く**

```ts
expect(buildCodeSourceUrl(symbol, "https://github.com/shuntatsu/trade_rl"))
  .toContain(`/blob/${index.source_revision}/trade_rl/risk/pretrade.py#L`);
```

- [ ] **Step 3: REDを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/sequence-diagram.test.tsx guide/tests/interactions.test.tsx
```

- [ ] **Step 4: reviewed replay contentを書く**

CodeReference idは最低限次を持つ。

```text
replay-run -> trade_rl.evaluation.replay.run_single_symbol_replay
replay-observation -> trade_rl.evaluation.replay._observation
risk-constrain -> trade_rl.risk.pretrade.PreTradeRisk.constrain
execute-interval -> trade_rl.simulation.execution.MarketExecutor.execute_interval
```

`replay-run` のvariablesには `desired_quantity`, `proposal_weight`, `target_weight`, `book`, `current_intent`, `index` を日本語alias付きで登録する。test pathは `tests/evaluation/test_single_symbol_replay.py`、`tests/risk/test_pretrade.py`、`tests/simulation/test_execution_v2.py` 等、実際に該当contractを検証するfileだけを付ける。

- [ ] **Step 5: code digestを明示refreshする**

```bash
python3 guide/tools/content_contract.py --refresh-code implementation-replay
python3 guide/tools/content_contract.py --check
```

- [ ] **Step 6: SequenceDiagramを実装する**

Desktopはactor header + ordered message rows。messageはnative `<button>` とし、`aria-current="step"` を選択messageだけに付ける。Mobileは同じordered dataからstep-throughを表示し、「前へ」「次へ」で全messageへ到達可能にする。

- [ ] **Step 7: CodeInspectorとexact source linkを実装する**

日本語説明を先に、identifier/path/signatureを副表示にする。source/test linkは外部linkとして明示し、source linkはgenerated `source_revision` を必ず含む。

- [ ] **Step 8: App routeとselectionを接続する**

`#implementation-replay?step=risk-constrain` 直アクセスで同じmessage/inspectorが選択される。unknown stepはfirst messageへfallbackする。

- [ ] **Step 9: targeted GREENを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/sequence-diagram.test.tsx guide/tests/interactions.test.tsx guide/tests/routing.test.ts
npm --prefix guide run typecheck
```

- [ ] **Step 10: commitする**

```bash
git add guide/content/manifest.json guide/content/topics/implementation-replay.json guide/src/visualizations/SequenceDiagram.tsx guide/src/components/CodeInspector.tsx guide/src/content/sourceLinks.ts guide/src/content/schema.ts guide/src/visualizations/VisualizationRenderer.tsx guide/src/app/App.tsx guide/src/styles/app.css guide/tests/sequence-diagram.test.tsx guide/tests/interactions.test.tsx
git commit -m "feat: explain replay implementation with code-linked sequence"
```

---

### Task 6: PPO Observation v2から学習・実行までを同じsequence modelで追えるようにする

**Files:**
- Create: `guide/content/topics/implementation-ppo.json`
- Create: `guide/tests/ppo-sequence.test.tsx`
- Modify: `guide/content/manifest.json`
- Modify: `guide/src/styles/app.css`
- Modify: `guide/tests/content.test.ts`

**Interfaces:**
- Reuses: `SequenceDiagram`, `CodeInspector`, `CodeReference`
- Required code refs:
  - `trade_rl.strategies.rl.ppo._encode_observation`
  - `trade_rl.strategies.rl.ppo.PPOTradingEnv.step`
  - `trade_rl.strategies.rl.ppo.fit_ppo_strategy`
  - `trade_rl.strategies.rl.ppo.PPOIntentStrategy.decide`

- [ ] **Step 1: PPO sequence content testをREDで追加する**

5 observation segmentの日本語label、`PPOTradingEnv.step` のrisk/execution/reward順、`fit_ppo_strategy` の `model.learn` 後にfitted strategyが返る説明をassertする。

- [ ] **Step 2: REDを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/ppo-sequence.test.tsx
```

- [ ] **Step 3: current sourceを根拠にreviewed contentを書く**

Observation segment:

```text
選択したローカル特徴量 -> local_values
利用可能かつ有限のマスク -> local_available / finite
正規化された鮮度遅延 -> local_staleness
現在の売買意図 -> current_intent
現在のウェイト -> current_weight
```

除外欄にはsymbol IDとdataset-global featureを明示する。

`PPOTradingEnv.step` のvariablesには `desired_quantity`, `proposal_weight`, `target_weight`, `reward`, `index` を登録する。

- [ ] **Step 4: code/docs fingerprintをrefreshする**

```bash
python3 guide/tools/content_contract.py --refresh-code implementation-ppo
python3 guide/tools/content_contract.py --refresh implementation-ppo
python3 guide/tools/content_contract.py --check
```

- [ ] **Step 5: existing sequence rendererで表示し、追加分岐を作らない**

PPO専用rendererは作らず、content modelで表現する。Observation 5segmentだけはmessage detail内のordered listとして出す。

- [ ] **Step 6: GREENを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/ppo-sequence.test.tsx guide/tests/content.test.ts
npm --prefix guide run typecheck
```

- [ ] **Step 7: commitする**

```bash
git add guide/content/manifest.json guide/content/topics/implementation-ppo.json guide/tests/ppo-sequence.test.tsx guide/tests/content.test.ts guide/src/styles/app.css
git commit -m "feat: trace PPO training and execution in the Guide"
```

---

### Task 7: 「コード地図」をreview済み責務フロー + symbol drill-downとして実装する

**Files:**
- Create: `guide/content/topics/code-map.json`
- Create: `guide/src/visualizations/CodeMap.tsx`
- Create: `guide/tests/code-map.test.tsx`
- Modify: `guide/content/manifest.json`
- Modify: `guide/src/content/schema.ts`
- Modify: `guide/src/visualizations/VisualizationRenderer.tsx`
- Modify: `guide/src/styles/app.css`

**Interfaces:**
- New visualization kind: `code-map`
- `CodeMapEdge.relation` is exactly `data-flow | calls`
- Reuses: `layerDirectedGraph`, `CodeInspector`

- [ ] **Step 1: relation/edge RED testsを書く**

unknown node、unknown relation、cycleをrejectする。renderer testは `integrations -> data` が宣言されていればそのedgeだけを出し、node配列順由来の `data -> integrations` が出ないことをassertする。

- [ ] **Step 2: REDを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/code-map.test.tsx
```

- [ ] **Step 3: top-level reviewed modelを書く**

nodeは `artifacts`, `integrations`, `data`, `strategies`, `risk`, `simulation`, `evaluation`。edge labelは「市場sourceを内部契約へ変換」「観測から売買意図を生成」「hard riskを適用」「約定・会計を実行」「結果を評価Evidenceへ変換」等、日本語で何が渡るかを書く。

このviewを「import依存図」と記述しない。

- [ ] **Step 4: packageごとの主要owner symbolをCodeReferenceへbindする**

最低限 `MarketDataset`, `SingleSymbolStrategy`, `PreTradeRisk`, `MarketExecutor`, `run_single_symbol_replay` へdrill-downできるようにする。

- [ ] **Step 5: fingerprintをrefreshする**

```bash
python3 guide/tools/content_contract.py --refresh-code code-map
python3 guide/tools/content_contract.py --refresh code-map
python3 guide/tools/content_contract.py --check
```

- [ ] **Step 6: CodeMapを実装する**

Desktopはlayered overview、mobileは選択nodeのincoming/outgoing neighborhood + outlineを表示する。node選択でCodeInspectorを同期する。pan/zoomを必須操作にしない。

- [ ] **Step 7: GREENを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/code-map.test.tsx guide/tests/content.test.ts
npm --prefix guide run typecheck
```

- [ ] **Step 8: commitする**

```bash
git add guide/content/manifest.json guide/content/topics/code-map.json guide/src/visualizations/CodeMap.tsx guide/src/content/schema.ts guide/src/visualizations/VisualizationRenderer.tsx guide/src/styles/app.css guide/tests/code-map.test.tsx
git commit -m "feat: add code-linked responsibility map"
```

---

### Task 8: 情報構造を8トピックへ整理し、表示文を日本語中心へ移行する

**Files:**
- Modify: `guide/content/manifest.json`
- Modify: `guide/content/topics/overview.json`
- Modify: `guide/content/topics/data-flow.json`
- Modify: `guide/content/topics/execution-economics.json`
- Modify: `guide/content/topics/experiment-loop.json`
- Modify: `guide/content/topics/research-status.json`
- Delete: `guide/content/topics/architecture.json`
- Delete: `guide/content/topics/ppo-observation-v2.json`
- Modify: `guide/tests/content.test.ts`
- Modify: `guide/tests/search.test.ts`

**Interfaces:**
- Final topic order:
  1. `overview`
  2. `implementation-replay`
  3. `implementation-ppo`
  4. `code-map`
  5. `data-flow`
  6. `execution-economics`
  7. `experiment-loop`
  8. `research-status`

- [ ] **Step 1: final manifest expectationを先にREDへ変更する**

```ts
expect(topics.map((topic) => topic.id)).toEqual([
  "overview",
  "implementation-replay",
  "implementation-ppo",
  "code-map",
  "data-flow",
  "execution-economics",
  "experiment-loop",
  "research-status",
]);
```

- [ ] **Step 2: REDを確認する**

```bash
npm --prefix guide run test -- --run guide/tests/content.test.ts
```

- [ ] **Step 3: copyを日本語主表示へ改稿する**

原則:

```text
Point-in-time evidence -> その時点で利用可能だった市場証拠（point-in-time evidence）
Strategy / PPO -> 売買判断（Strategy / PPO）
Execution -> 約定・会計（Execution）
EvidenceSet -> 比較用の検証証拠（EvidenceSet）
portable numerics -> CPU差を抑えた再現可能数値計算（portable numerics）
```

最初の出現で日本語意味を示し、以後必要のない英語反復を減らす。

- [ ] **Step 4: #513後のresearch-state factを保持する**

`research-status.json` の「portable Experiment 0001の結果前preregistrationとfresh verifierを完了し、bound runを開始」等、current main由来の状態を削除・巻き戻さない。

- [ ] **Step 5: 廃止topicを削除しmanifestを更新する**

`architecture` の責務説明は `code-map`、`ppo-observation-v2` は `implementation-ppo` へ統合してから削除する。

- [ ] **Step 6: 影響topicのMarkdown fingerprintだけを明示refreshする**

```bash
python3 guide/tools/content_contract.py --refresh overview data-flow execution-economics experiment-loop research-status
python3 guide/tools/content_contract.py --check
```

正本sectionが変わっていないtopicも、content source bindingが既存digestと一致するなら無意味なdigest変更は発生しないことをdiffで確認する。

- [ ] **Step 7: content/search testsをGREENにする**

```bash
npm --prefix guide run test -- --run guide/tests/content.test.ts guide/tests/search.test.ts
```

- [ ] **Step 8: commitする**

```bash
git add guide/content guide/tests/content.test.ts guide/tests/search.test.ts
git commit -m "docs: make the Human Guide Japanese-first"
```

---

### Task 9: exact revisionをCI/Pagesへbindし、desktop/mobile/a11y E2Eを更新する

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/deploy-guide.yml`
- Modify: `guide/e2e/guide.spec.ts`
- Modify: `guide/e2e/accessibility.spec.ts`
- Modify: `guide/e2e/visual.spec.ts`
- Modify: `guide/e2e/public-smoke.spec.ts`
- Modify: `tests/architecture/test_guide_pages_deployment.py`

**Interfaces:**
- CI Guide checkout SHA is passed as `GUIDE_SOURCE_REV`.
- Deployment build uses `${{ github.event.workflow_run.head_sha }}` as `GUIDE_SOURCE_REV`.

- [ ] **Step 1: deployment contract RED testを書く**

`tests/architecture/test_guide_pages_deployment.py` でdeploy workflowのBuild Guide stepに `GUIDE_SOURCE_REV: ${{ github.event.workflow_run.head_sha }}` が存在することをassertする。

- [ ] **Step 2: E2Eを新workflowへ更新しREDを確認する**

`guide.spec.ts` の主経路を次へ変更する。

```text
/#overview
→ 「1本のバーを追う」
→ risk-constrain選択
→ inspectorでPreTradeRisk.constrainを確認
→ source link hrefに40桁revision + pretrade.py line anchor
```

mobile-320では `implementation-replay` を開き、「次へ」でrisk stepまで移動できることを確認する。

- [ ] **Step 3: targeted REDを確認する**

```bash
uv run pytest -q tests/architecture/test_guide_pages_deployment.py
npm --prefix guide run e2e -- --grep "replay|mobile|source"
```

- [ ] **Step 4: CI/deploy workflowへexact revision envを追加する**

CI:

```yaml
env:
  GUIDE_SOURCE_REV: ${{ github.event.pull_request.head.sha || github.sha }}
```

Deploy Build Guide:

```yaml
env:
  GUIDE_SOURCE_REV: ${{ github.event.workflow_run.head_sha }}
```

- [ ] **Step 5: accessibility coverageを追加する**

light/dark両方でoverviewとimplementation-replayをaxe検査する。keyboard testはTab/Enterだけでsequence message選択とsource linkまで到達する。selectionは`aria-current`とtext labelで判別できることをassertする。

- [ ] **Step 6: visual evidenceを更新する**

Desktop light/darkは `#implementation-replay?step=risk-constrain`、mobile-320は同topicのstep-through状態をcaptureする。animationsは既存どおりdisabled。

- [ ] **Step 7: public smokeを更新する**

公開siteでhash selection復元、日本語検索、theme、exact-SHA source link、320px overflowを確認する。

- [ ] **Step 8: targeted GREENを確認する**

```bash
uv run pytest -q tests/architecture/test_guide_pages_deployment.py
npm --prefix guide run e2e
```

- [ ] **Step 9: commitする**

```bash
git add .github/workflows/ci.yml .github/workflows/deploy-guide.yml guide/e2e tests/architecture/test_guide_pages_deployment.py
git commit -m "test: bind Guide UX and source links to exact revisions"
```

---

### Task 10: durable maintenance contractへ昇格し、全検証・反証・cleanupを行う

**Files:**
- Modify: `guide/README.md`
- Modify: `docs/AGENTS.md`
- Modify: `docs/README.md`
- Modify: `tests/architecture/test_human_guide.py`
- Modify: `tests/architecture/test_human_guide_content_contract.py`
- Delete after durable promotion: `docs/specs/2026-09-13-japanese-code-explorer-design.md`
- Delete after durable promotion: `docs/plans/2026-09-13-japanese-code-explorer.md`

**Interfaces:**
- Durable docs define Japanese-first copy rule, `--refresh-code`, generated index, exact-revision link contract, sequence/code-map ownership, and cleanup rule.

- [ ] **Step 1: durable contract testを先に更新してREDを確認する**

`tests/architecture/test_human_guide.py` でREADMEに以下markerが存在することをassertする。

```text
Japanese-first
--refresh-code
.guide generated code symbol indexの非tracked性
exact revision source link
sequence / code-map relationship is reviewed content, not inferred call graph
```

Run:

```bash
uv run pytest -q tests/architecture/test_human_guide.py tests/architecture/test_human_guide_content_contract.py
```

- [ ] **Step 2: `guide/README.md` と `docs/AGENTS.md` を更新する**

maintenance手順を次の順で固定する。

```bash
python3 guide/tools/content_contract.py --check
python3 guide/tools/content_contract.py --refresh-code <reviewed-topic-id>
python3 guide/tools/content_contract.py --refresh <reviewed-topic-id>
npm --prefix guide run check
npm --prefix guide run e2e
```

`--refresh-code` / `--refresh` はCIを通すための一括更新に使わず、人間がsourceを読み直したtopicだけ指定することを明記する。

- [ ] **Step 3: targeted Guide gateを実行する**

```bash
python3 guide/tools/content_contract.py --check
npm --prefix guide run check
npm --prefix guide run e2e
uv run pytest -q tests/architecture/test_human_guide.py tests/architecture/test_human_guide_content_contract.py tests/architecture/test_human_guide_code_symbols.py tests/architecture/test_guide_pages_deployment.py
```

- [ ] **Step 4: runtime非変更とgenerated漏れを反証する**

```bash
git diff --name-only main...HEAD
git status --short
git ls-files guide/.generated
```

Expected:

- `trade_rl/**` のdiffは0件。
- `guide/.generated` tracked fileは0件。
- debug/temporary outputは0件。

- [ ] **Step 5: final mainを取り込む**

```bash
git fetch origin main
git merge --no-ff origin/main
```

conflictがあればcurrent mainのresearch statusをauthorityとしてGuide copyを再照合する。force push/rebase history rewriteは使わない。

- [ ] **Step 6: final current-main包含を確認する**

```bash
git merge-base --is-ancestor origin/main HEAD
git rev-parse HEAD
```

- [ ] **Step 7: repository full gateを実行する**

```bash
uv run ruff check trade_rl tests tools
uv run ruff format --check trade_rl tests tools
uv run mypy trade_rl
uv run mypy tools/agent_repo tests/architecture/distribution.py
uv run pytest -q
npm --prefix guide run check
npm --prefix guide run e2e
```

- [ ] **Step 8: final diffを自己レビューする**

確認対象:

```text
Acceptance Criteria 1–19
architecture edge correctness
source digest / variable binding fail-closed
sequence source順序
PPO Observation v2 contract
Japanese-first copy
exact SHA links
keyboard/mobile/accessibility
runtime semantics unchanged
research status unchanged
no generated/debug/temp files
```

誤実装を通すテストがないか、特に「edgeを配列順へ戻してもtestsが通る」「unknown variableでもsource-checkが通る」「source linkがmainへ戻ってもE2Eが通る」を反証し、不十分ならtestを追加して修正する。

- [ ] **Step 9: durable docsへ昇格後、Active spec/planを削除する**

```bash
git rm docs/specs/2026-09-13-japanese-code-explorer-design.md
git rm docs/plans/2026-09-13-japanese-code-explorer.md
```

`docs/README.md` を「Active spec / planなし」へ戻し、耐久的な契約が `guide/README.md` / `docs/AGENTS.md` / testsへ残っていることを確認する。

- [ ] **Step 10: cleanup後の最終HEADでtargeted/full gateを再実行する**

```bash
python3 guide/tools/content_contract.py --check
npm --prefix guide run check
npm --prefix guide run e2e
uv run pytest -q tests/architecture/test_human_guide.py tests/architecture/test_human_guide_content_contract.py tests/architecture/test_human_guide_code_symbols.py tests/architecture/test_guide_pages_deployment.py
uv run pytest -q
```

- [ ] **Step 11: final commitとPRを作る**

```bash
git add -A
git commit -m "docs: finalize Japanese code explorer"
git status --short
git rev-parse HEAD
```

PR bodyには、RED証拠、主要GREEN gate、runtime diff 0、current main包含、未検証事項を記載する。mergeはユーザーの明示許可なしに行わない。

- [ ] **Step 12: exact final PR HEADのCIだけを有効な統合証拠として確認する**

CI後にHEADまたはmainが進んだ場合は古いGreenを破棄し、同期後の新HEADで再検証する。

- [ ] **Step 13: merge後deployment oracleを確認する**

`.github/workflows/deploy-guide.yml` のbuild/deploy/smokeがmerge SHAに対して成功し、公開URLで次を確認する。

```text
日本語主表示
replay sequence
risk/execute inspector
exact merge-SHA source link
日本語/identifier検索
mobile step-through
320px overflowなし
```

このpost-merge public smokeまで成功して初めてGitHub Pages改修を完了とする。
