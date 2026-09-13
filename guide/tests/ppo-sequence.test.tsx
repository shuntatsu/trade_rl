import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownArticle } from "../src/components/MarkdownArticle";
import { loadTopics } from "../src/content/loadTopics";

function ppoTopic() {
  const topic = loadTopics().find((item) => item.id === "implementation-ppo");
  if (!topic) throw new Error("missing PPO implementation topic");
  return topic;
}

describe("PPO implementation sequence", () => {
  it("shows all five observation segments without interaction", () => {
    render(<MarkdownArticle topic={ppoTopic()} />);

    const table = screen.getByRole("table");
    for (const text of [
      "local_values",
      "local_available",
      "local_staleness",
      "current_intent",
      "current_weight",
    ]) {
      expect(within(table).getByText(new RegExp(text))).toBeInTheDocument();
    }
  });

  it("keeps risk, execution, and reward in the static source order", () => {
    const markdown = ppoTopic().markdown;
    const risk = markdown.indexOf("## 4. hard riskを通す");
    const execution = markdown.indexOf("## 5. 約定・会計を通す");
    const reward = markdown.indexOf("## 6. net returnからrewardを作る");

    expect(risk).toBeGreaterThanOrEqual(0);
    expect(risk).toBeLessThan(execution);
    expect(execution).toBeLessThan(reward);
  });

  it("states that fitting and runtime decisions share the same encoder", () => {
    render(<MarkdownArticle topic={ppoTopic()} />);

    expect(
      screen.getByRole("heading", { name: "学習後も同じ観測契約を使う" }),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/_encode_observation/).length).toBeGreaterThan(0);
    expect(screen.getByText(/実行時も学習時と同じencoderを使う/)).toBeInTheDocument();
  });
});
