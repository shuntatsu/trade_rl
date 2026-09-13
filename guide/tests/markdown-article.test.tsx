import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownArticle } from "../src/components/MarkdownArticle";
import type { GuideTopic } from "../src/content/schema";

function topic(markdown: string): GuideTopic {
  return {
    id: "demo",
    title: "デモ",
    nav_label: "デモ",
    summary: "デモ",
    role: "detail",
    keywords: [],
    source_sections: [],
    code_references: [],
    markdown,
    headings: [],
  } as GuideTopic;
}

describe("MarkdownArticle", () => {
  it("renders headings, lists, GFM tables, and fenced text without interaction", () => {
    render(
      <MarkdownArticle
        topic={topic(`## 処理順

- 観測
- hard risk

| step | state |
| --- | --- |
| 1 | causal |

\`\`\`text
MarketDataset
    ↓
Strategy
\`\`\`
`)}
      />,
    );

    expect(screen.getByRole("heading", { name: "処理順" })).toBeInTheDocument();
    expect(screen.getByText("hard risk")).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText(/MarketDataset/)).toBeInTheDocument();
  });

  it("does not execute raw HTML or emit unsafe javascript/data links", () => {
    const { container } = render(
      <MarkdownArticle
        topic={topic(`## 安全性

<script>window.__guide_xss = true</script>

[危険](javascript:alert(1))

[データ](data:text/html,boom)

[安全](https://example.com/)
`)}
      />,
    );

    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(container.querySelector('a[href^="data:"]')).toBeNull();
    expect(screen.getByRole("link", { name: "安全" })).toHaveAttribute(
      "href",
      "https://example.com/",
    );
  });
});
