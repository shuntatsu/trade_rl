import { describe, expect, it } from "vitest";

import { loadTopics } from "../src/content/loadTopics";
import { parseTopic } from "../src/content/schema";


describe("guide content", () => {
  it("loads the seven reviewed topics in manifest order", () => {
    const topics = loadTopics();
    expect(topics.map((topic) => topic.id)).toEqual([
      "overview",
      "data-flow",
      "architecture",
      "execution-economics",
      "ppo-observation-v2",
      "experiment-loop",
      "research-status",
    ]);
    expect(topics.every((topic) => topic.source_sections.length > 0)).toBe(true);
    expect(
      topics.every((topic) =>
        topic.source_sections.every((source) => /^[0-9a-f]{64}$/.test(source.sha256)),
      ),
    ).toBe(true);
  });

  it("rejects an unknown visualization kind", () => {
    expect(() =>
      parseTopic({
        id: "bad",
        title: "Bad",
        nav_label: "Bad",
        summary: "Bad",
        keywords: [],
        source_sections: [
          {
            path: "docs/architecture/lean-core.md",
            heading: "Core flow",
            sha256: "0".repeat(64),
          },
        ],
        sections: [],
        visualization: { kind: "mystery" },
      }),
    ).toThrow(/visualization kind/i);
  });

  it("rejects broken architecture edge references", () => {
    expect(() =>
      parseTopic({
        id: "broken",
        title: "Broken",
        nav_label: "Broken",
        summary: "Broken",
        keywords: [],
        source_sections: [
          {
            path: "docs/architecture/lean-core.md",
            heading: "Core flow",
            sha256: "0".repeat(64),
          },
        ],
        sections: [],
        visualization: {
          kind: "architecture",
          nodes: [
            {
              id: "a",
              label: "A",
              subtitle: "A",
              description: "A",
              input: "A",
              output: "A",
              not_owned: "A",
            },
          ],
          edges: [{ from: "a", to: "missing" }],
        },
      }),
    ).toThrow(/edge reference/i);
  });
});
