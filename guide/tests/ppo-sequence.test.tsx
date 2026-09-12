import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { loadTopics } from "../src/content/loadTopics";
import { SequenceDiagram } from "../src/visualizations/SequenceDiagram";

describe("PPO implementation sequence", () => {
  it("explains the five observation segments in implementation order", () => {
    const topic = loadTopics().find((item) => item.id === "implementation-ppo");
    expect(topic).toBeDefined();
    if (!topic || topic.visualization.kind !== "sequence") return;

    const observation = topic.visualization.messages.find(
      (message) => message.id === "encode-observation",
    );
    expect(observation?.state_changes_ja).toEqual([
      "選択したローカル特徴量 → local_values",
      "利用可能かつ有限のマスク → local_available / finite",
      "正規化された鮮度遅延 → local_staleness",
      "現在の売買意図 → current_intent",
      "現在のウェイト → current_weight",
    ]);

    render(
      <SequenceDiagram
        visualization={topic.visualization}
        selectedStep="encode-observation"
        onSelectStep={() => undefined}
      />,
    );
    expect(screen.getByText("現在のウェイト → current_weight")).toBeInTheDocument();
  });

  it("keeps environment risk, execution, and reward in source order", () => {
    const topic = loadTopics().find((item) => item.id === "implementation-ppo");
    expect(topic).toBeDefined();
    if (!topic || topic.visualization.kind !== "sequence") return;

    const ids = topic.visualization.messages.map((message) => message.id);
    expect(ids.indexOf("env-risk")).toBeLessThan(ids.indexOf("env-execute"));
    expect(ids.indexOf("env-execute")).toBeLessThan(ids.indexOf("env-reward"));
  });

  it("shows that fitting returns the fitted intent strategy used by decide", () => {
    const topic = loadTopics().find((item) => item.id === "implementation-ppo");
    expect(topic).toBeDefined();
    if (!topic || topic.visualization.kind !== "sequence") return;

    const fit = topic.visualization.messages.find((message) => message.id === "fit-strategy");
    const decide = topic.visualization.messages.find(
      (message) => message.id === "runtime-decide",
    );
    expect(fit?.label_ja).toContain("学習済み売買判断を返す");
    expect(fit?.code_ref).toBe("fit-ppo");
    expect(decide?.code_ref).toBe("ppo-decide");
  });
});
