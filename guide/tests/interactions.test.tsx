import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/app/App";
import { SearchPalette } from "../src/components/SearchPalette";
import type { DocumentGuideTopic } from "../src/content/documentSchema";
import { extractMarkdownHeadings } from "../src/content/markdown";

function searchableImplementationTopic(): DocumentGuideTopic {
  const markdown = "## 処理順\n\n観測 → hard risk → 約定・会計\n";
  return {
    id: "implementation-replay",
    title: "1本のバーを追う",
    nav_label: "1本のバーを追う",
    summary: "単銘柄リプレイの実装順を追跡する。",
    role: "detail",
    keywords: ["リプレイ"],
    source_sections: [
      {
        path: "docs/architecture/lean-core.md",
        heading: "Core flow",
        sha256: "0".repeat(64),
      },
    ],
    code_references: [
      {
        id: "replay-run",
        symbol: "trade_rl.evaluation.replay.run_single_symbol_replay",
        kind: "function",
        source_sha256: "1".repeat(64),
        label_ja: "単銘柄リプレイを実行",
        description_ja: "市場データを順番に処理して評価結果を作る。",
        variables: [
          {
            name: "desired_quantity",
            label_ja: "希望保有数量",
            description_ja: "リスク制約前後で保持したい数量。",
          },
        ],
        tests: ["tests/evaluation/test_single_symbol_replay.py"],
      },
    ],
    markdown,
    headings: extractMarkdownHeadings(markdown),
  };
}

describe("Markdown-first guide workflow", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.location.hash = "#overview";
    document.documentElement.classList.remove("dark");
    Object.defineProperty(window, "scrollTo", { value: vi.fn(), configurable: true });
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      value: vi.fn(),
      configurable: true,
    });
  });

  it("shows the system purpose and core flow without interaction", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "実データで動く、検証可能なトレーディングRLシステム",
      }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "何をするシステムか" })).toBeInTheDocument();
    expect(screen.getByText(/MarketDatasetを構築・固定/)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "実装を確認する" })).not.toBeInTheDocument();
  });

  it("opens search from Ctrl+K and navigates using Japanese terminology", async () => {
    const user = userEvent.setup();
    render(<App />);

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("textbox", { name: "ガイドを検索" });
    await user.type(input, "手数料");
    await user.click(screen.getByRole("button", { name: /約定コストと会計/ }));

    expect(window.location.hash).toBe("#execution-economics");
    expect(
      await screen.findByRole("heading", { level: 1, name: "約定コストと会計" }),
    ).toBeInTheDocument();
  });

  it("navigates a code-search result with its symbol selection", async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    render(
      <SearchPalette topics={[searchableImplementationTopic()]} onNavigate={onNavigate} />,
    );

    await user.click(screen.getByRole("button", { name: "ガイドを検索" }));
    const input = await screen.findByRole("textbox", { name: "ガイドを検索" });
    await user.type(input, "希望保有数量");
    await user.click(screen.getByRole("button", { name: /希望保有数量/ }));

    expect(onNavigate).toHaveBeenCalledWith({
      topicId: "implementation-replay",
      symbol: "trade_rl.evaluation.replay.run_single_symbol_replay",
    });
  });

  it("maps a known legacy replay step to the matching document heading", () => {
    window.location.hash = "#implementation-replay?step=risk-constrain";
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "1本のバーを追う" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "4. hard riskを適用する" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /ハードリスクで目標を制約/ })).not.toBeInTheDocument();
  });

  it("falls back to the document root for an unknown legacy step", () => {
    window.location.hash = "#implementation-replay?step=missing";
    render(<App />);

    expect(screen.getByRole("heading", { level: 2, name: "処理順" })).toBeInTheDocument();
    expect(screen.getByText(/この順序はクリックしなくても全体を読める/)).toBeInTheDocument();
  });

  it("renders code-map ownership as a static reference page", () => {
    window.location.hash = "#code-map?step=risk";
    render(<App />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "コード地図：どこが何を所有するか",
      }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Ownership" })).toBeInTheDocument();
    expect(screen.getAllByText("hard riskの所有者").length).toBeGreaterThanOrEqual(2);
    const details = screen.getByText("実装詳細: hard riskの所有者").closest("details");
    expect(details).not.toHaveAttribute("open");
  });

  it("switches dark mode without changing content", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "ダークモードへ" }));
    expect(document.documentElement).toHaveClass("dark");
    expect(window.localStorage.getItem("trade-rl-guide-theme")).toBe("dark");
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
  });
});
