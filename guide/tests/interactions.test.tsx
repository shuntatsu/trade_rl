import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/app/App";
import { SearchPalette } from "../src/components/SearchPalette";
import type { GuideTopic } from "../src/content/schema";

function searchableImplementationTopic(): GuideTopic {
  return {
    id: "implementation-replay",
    title: "1本のバーを追う",
    nav_label: "1本のバーを追う",
    summary: "単銘柄リプレイの実装順を追跡する。",
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
    sections: [],
    visualization: {
      kind: "data-flow",
      steps: [
        {
          id: "replay",
          label: "リプレイ",
          detail: "単銘柄を順に処理する。",
        },
      ],
    },
  };
}

describe("interactive guide workflow", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.location.hash = "#overview";
    document.documentElement.classList.remove("dark");
  });

  it("keeps diagram detail selectable by click", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "実データで動く、検証可能なトレーディングRLシステム",
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^Execution/ }));
    expect(
      screen.getByRole("heading", { level: 3, name: "Execution" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/MarketExecutor \+ BookState/)).toBeInTheDocument();
  });

  it("opens search from Ctrl+K and navigates using Japanese terminology", async () => {
    const user = userEvent.setup();
    render(<App />);

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("textbox", { name: "ガイドを検索" });
    await user.type(input, "手数料");
    await user.click(screen.getByRole("button", { name: /実行コスト/ }));

    expect(window.location.hash).toBe("#execution-economics");
    expect(
      await screen.findByRole("heading", { level: 1, name: "実行コスト" }),
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

  it("switches dark mode without changing content", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "ダークモードへ" }));
    expect(document.documentElement).toHaveClass("dark");
    expect(window.localStorage.getItem("trade-rl-guide-theme")).toBe("dark");
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
  });
});
