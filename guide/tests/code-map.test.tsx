import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ImplementationReferenceAppendix } from "../src/components/ImplementationReferenceAppendix";
import { MarkdownArticle } from "../src/components/MarkdownArticle";
import { loadTopics } from "../src/content/loadTopics";

function codeMapTopic() {
  const topic = loadTopics().find((item) => item.id === "code-map");
  if (!topic) throw new Error("missing code-map topic");
  return topic;
}

describe("Guide code map reference", () => {
  it("renders ownership and non-ownership as a static table", () => {
    render(<MarkdownArticle topic={codeMapTopic()} />);

    expect(screen.getByRole("heading", { name: "Ownership" })).toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(table).toHaveTextContent("責務");
    expect(table).toHaveTextContent("owns");
    expect(table).toHaveTextContent("does not own");
    expect(table).toHaveTextContent("hard risk");
  });

  it("keeps code-map outside the normal reading order", async () => {
    const manifest = (await import("../content/manifest.json")).default;
    expect(manifest.reading_order).not.toContain("code-map");
    expect(manifest.groups.at(-1)?.id).toBe("reference");
    expect(manifest.groups.at(-1)?.topics).toContain("code-map");
  });

  it("offers implementation details only as collapsed disclosures", async () => {
    const user = userEvent.setup();
    render(<ImplementationReferenceAppendix topic={codeMapTopic()} />);

    const summary = screen.getByText("実装詳細: hard riskの所有者");
    const details = summary.closest("details");
    expect(details).not.toHaveAttribute("open");

    await user.click(summary);
    expect(details).toHaveAttribute("open");
    expect(
      screen.getByText("trade_rl.risk.pretrade.PreTradeRisk.constrain"),
    ).toBeVisible();
  });
});
