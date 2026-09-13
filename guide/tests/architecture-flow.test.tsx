import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownArticle } from "../src/components/MarkdownArticle";
import { loadTopics } from "../src/content/loadTopics";


describe("overview architecture flow", () => {
  it("renders the full core flow without selectable graph nodes", () => {
    const topic = loadTopics().find((item) => item.id === "overview");
    if (!topic) throw new Error("missing overview topic");

    render(<MarkdownArticle topic={topic} />);

    expect(screen.getByRole("heading", { name: "何をするシステムか" })).toBeInTheDocument();
    expect(screen.getByText(/MarketDatasetを構築・固定/)).toBeInTheDocument();
    expect(screen.getByText(/共通execution\/accounting/)).toBeInTheDocument();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
});
