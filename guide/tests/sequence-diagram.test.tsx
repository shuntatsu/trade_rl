import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { buildCodeSourceUrl } from "../src/content/sourceLinks";
import { codeSymbolIndex, getCodeSymbol } from "../src/content/codeSymbols";
import type { SequenceVisualization } from "../src/content/schema";
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
});
