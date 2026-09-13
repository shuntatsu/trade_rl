import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ImplementationReferenceAppendix } from "../src/components/ImplementationReferenceAppendix";
import { loadTopics } from "../src/content/loadTopics";

const TOPICS = loadTopics();

function replayTopic() {
  const topic = TOPICS.find((item) => item.id === "implementation-replay");
  if (!topic) throw new Error("missing replay topic");
  return topic;
}

describe("ImplementationReferenceAppendix", () => {
  it("starts collapsed and opens with the native summary control", async () => {
    const user = userEvent.setup();
    render(<ImplementationReferenceAppendix topic={replayTopic()} />);

    const summary = screen.getByText("実装詳細: ハードリスクを適用");
    const details = summary.closest("details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");

    await user.click(summary);
    expect(details).toHaveAttribute("open");
  });

  it("opens exactly the reference selected by a symbol deep link", () => {
    render(
      <ImplementationReferenceAppendix
        topic={replayTopic()}
        openSymbol="trade_rl.risk.pretrade.PreTradeRisk.constrain"
      />,
    );

    const opened = screen
      .getByText("実装詳細: ハードリスクを適用")
      .closest("details");
    const replay = screen
      .getByText("実装詳細: 単銘柄リプレイを実行")
      .closest("details");
    expect(opened).toHaveAttribute("open");
    expect(replay).not.toHaveAttribute("open");
  });
});
