import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeInspector } from "../src/components/CodeInspector";
import { MarkdownArticle } from "../src/components/MarkdownArticle";
import { codeSymbolIndex, getCodeSymbol } from "../src/content/codeSymbols";
import { loadTopics } from "../src/content/loadTopics";
import { buildCodeSourceUrl } from "../src/content/sourceLinks";

function replayTopic() {
  const topic = loadTopics().find((item) => item.id === "implementation-replay");
  if (!topic) throw new Error("missing replay topic");
  return topic;
}

describe("Guide replay sequence", () => {
  it("renders the complete replay order as static document content", () => {
    const topic = replayTopic();
    render(<MarkdownArticle topic={topic} />);

    const markdown = topic.markdown;
    const observe = markdown.indexOf("## 1. 観測を作る");
    const risk = markdown.indexOf("## 4. hard riskを適用する");
    const execution = markdown.indexOf("## 5. 約定と会計を行う");
    const book = markdown.indexOf("## 6. BookStateを引き継ぐ");

    expect(observe).toBeGreaterThanOrEqual(0);
    expect(observe).toBeLessThan(risk);
    expect(risk).toBeLessThan(execution);
    expect(execution).toBeLessThan(book);
    expect(screen.getByText(/desired_quantity → proposal_weight/)).toBeInTheDocument();
    expect(screen.getAllByText(/PreTradeRisk\.constrain/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MarketExecutor\.execute_interval/).length).toBeGreaterThan(0);
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
    render(<CodeInspector referenceId="risk-constrain" topic={replayTopic()} />);

    expect(
      screen.getByRole("heading", { level: 3, name: "ハードリスクを適用" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/提案ウェイトへturnover、単一ウェイト上限/),
    ).toBeInTheDocument();
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
