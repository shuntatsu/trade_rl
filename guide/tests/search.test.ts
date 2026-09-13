import { describe, expect, it } from "vitest";

import type { DocumentGuideTopic } from "../src/content/documentSchema";
import { loadTopics } from "../src/content/loadTopics";
import { extractMarkdownHeadings } from "../src/content/markdown";
import { searchGuide, searchTopics } from "../src/content/search";

function implementationTopic(): DocumentGuideTopic {
  const markdown = "## 処理\n\nリスク制約と約定を順に適用する。\n";
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

describe("guide search", () => {
  it("matches titles, summaries, keywords, and Markdown copy", () => {
    const topics = loadTopics();

    expect(searchTopics(topics, "PPO")[0]?.id).toBe("implementation-ppo");
    expect(searchTopics(topics, "手数料").map((topic) => topic.id)).toContain(
      "execution-economics",
    );
    expect(searchTopics(topics, "winner").map((topic) => topic.id)).toContain(
      "research-status",
    );
  });

  it("returns manifest order for an empty query", () => {
    const topics = loadTopics();
    expect(searchTopics(topics, "   ").map((topic) => topic.id)).toEqual(
      topics.map((topic) => topic.id),
    );
  });

  it("maps real Japanese aliases and Python identifiers to the same replay symbol", () => {
    const topics = loadTopics();
    const japanese = searchGuide(topics, "希望保有数量");
    const identifier = searchGuide(topics, "desired_quantity");

    expect(japanese[0]?.topicId).toBe("implementation-replay");
    expect(identifier[0]?.topicId).toBe("implementation-replay");
    expect(japanese[0]?.symbol).toBe(
      "trade_rl.evaluation.replay.run_single_symbol_replay",
    );
    expect(identifier[0]?.symbol).toBe(japanese[0]?.symbol);
    expect(japanese[0]?.title).toBe("希望保有数量");
    expect(japanese[0]?.subtitle).toContain("desired_quantity");
  });

  it("ranks a Japanese code-reference label above body-only matches", () => {
    const implementation = implementationTopic();
    const bodyMarkdown = "## 説明\n\n単銘柄リプレイを実行する処理についての補足。\n";
    const bodyOnly: DocumentGuideTopic = {
      ...implementation,
      id: "body-only",
      title: "補足",
      nav_label: "補足",
      code_references: [],
      markdown: bodyMarkdown,
      headings: extractMarkdownHeadings(bodyMarkdown),
    };

    const results = searchGuide([bodyOnly, implementation], "単銘柄リプレイを実行");
    expect(results[0]?.topicId).toBe("implementation-replay");
    expect(results[0]?.kind).toBe("symbol");
  });
});
