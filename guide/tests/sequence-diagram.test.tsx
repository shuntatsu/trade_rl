import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CodeInspector } from "../src/components/CodeInspector";
import { codeSymbolIndex, getCodeSymbol } from "../src/content/codeSymbols";
import type { GuideTopic, SequenceVisualization } from "../src/content/schema";
import { buildCodeSourceUrl } from "../src/content/sourceLinks";
import { SequenceDiagram } from "../src/visualizations/SequenceDiagram";

const visualization: SequenceVisualization = {
  kind: "sequence",
  actors: [
    { id: "dataset", label_ja: "市場データセット" },
    { id: "replay", label_ja: "リプレイ統括", code_ref: "replay-run" },
    { id: "risk", label_ja: "ハードリスク", code_ref: "risk-constrain" },
  ],
  messages: [
    {
      id: "observe",
      from: "dataset",
      to: "replay",
      label_ja: "その時点の観測を組み立てる",
      code_ref: "replay-observation",
    },
    {
      id: "risk-constrain",
      from: "replay",
      to: "risk",
      label_ja: "リスク制約を適用する",
      code_ref: "risk-constrain",
      state_changes_ja: ["目標weightをhard limit内へ射影する"],
    },
  ],
};

const inspectorTopic: GuideTopic = {
  id: "implementation-replay",
  title: "1本のバーを追う",
  nav_label: "1本のバーを追う",
  summary: "実装順を追跡する。",
  keywords: [],
  source_sections: [
    {
      path: "docs/architecture/lean-core.md",
      heading: "Data contract",
      sha256: "0".repeat(64),
    },
  ],
  code_references: [
    {
      id: "risk-constrain",
      symbol: "trade_rl.risk.pretrade.PreTradeRisk.constrain",
      kind: "method",
      source_sha256: "0".repeat(64),
      label_ja: "ハードリスクを適用",
      description_ja: "提案weightをhard riskの内側へ射影します。",
      variables: [],
      tests: ["tests/risk/test_pretrade.py"],
    },
  ],
  sections: [],
  visualization,
};

describe("Guide replay sequence", () => {
  it("renders messages in declared execution order and selects one", async () => {
    const user = userEvent.setup();
    const onSelectStep = vi.fn();
    render(
      <SequenceDiagram
        visualization={visualization}
        selectedStep="observe"
        onSelectStep={onSelectStep}
      />,
    );

    const messages = screen.getAllByRole("button", { name: /観測|リスク制約/ });
    expect(messages.map((message) => message.textContent)).toEqual([
      expect.stringContaining("その時点の観測を組み立てる"),
      expect.stringContaining("リスク制約を適用する"),
    ]);
    expect(messages[0]).toHaveAttribute("aria-current", "step");

    await user.click(messages[1]!);
    expect(onSelectStep).toHaveBeenCalledWith("risk-constrain");
  });

  it("builds source links against the generated exact revision", () => {
    const symbol = getCodeSymbol("trade_rl.risk.pretrade.PreTradeRisk.constrain");
    expect(symbol).toBeDefined();
    if (!symbol) return;

    expect(buildCodeSourceUrl(symbol, "https://github.com/shuntatsu/trade_rl")).toContain(
      `/blob/${codeSymbolIndex.source_revision}/trade_rl/risk/pretrade.py#L`,
    );
  });

  it("shows Japanese purpose before exact implementation details and links", () => {
    render(<CodeInspector referenceId="risk-constrain" topic={inspectorTopic} />);

    expect(
      screen.getByRole("heading", { level: 3, name: "ハードリスクを適用" }),
    ).toBeInTheDocument();
    expect(screen.getByText("提案weightをhard riskの内側へ射影します。")).toBeInTheDocument();
    expect(
      screen.getByText("trade_rl.risk.pretrade.PreTradeRisk.constrain"),
    ).toBeInTheDocument();

    const sourceLink = screen.getByRole("link", { name: "実装をGitHubで開く" });
    expect(sourceLink).toHaveAttribute(
      "href",
      expect.stringContaining(`/blob/${codeSymbolIndex.source_revision}/trade_rl/risk/pretrade.py#L`),
    );
    expect(screen.getByRole("link", { name: "関連テストを開く" })).toHaveAttribute(
      "href",
      expect.stringContaining("/blob/"),
    );
  });
});
