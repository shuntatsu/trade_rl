import { describe, expect, it } from "vitest";

import { loadTopics } from "../src/content/loadTopics";
import { parseTopic } from "../src/content/schema";

function minimalTopic() {
  return {
    id: "demo",
    title: "デモ",
    nav_label: "デモ",
    summary: "デモ",
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
      edges: [],
    },
  };
}

function codeReference() {
  return {
    id: "run",
    symbol: "trade_rl.demo.run",
    kind: "function",
    source_sha256: "a".repeat(64),
    label_ja: "デモ処理",
    description_ja: "デモ処理です。",
    variables: [],
    tests: [],
  };
}

describe("guide content", () => {
  it("loads the eight reviewed topics in manifest order", () => {
    const topics = loadTopics();
    expect(topics.map((topic) => topic.id)).toEqual([
      "overview",
      "implementation-replay",
      "implementation-ppo",
      "code-map",
      "data-flow",
      "execution-economics",
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
        ...minimalTopic(),
        id: "bad",
        visualization: { kind: "mystery" },
      }),
    ).toThrow(/visualization kind/i);
  });

  it("rejects broken architecture edge references", () => {
    expect(() =>
      parseTopic({
        ...minimalTopic(),
        id: "broken",
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

  it("rejects duplicate code reference ids", () => {
    const reference = { ...codeReference(), id: "same" };

    expect(() =>
      parseTopic({
        ...minimalTopic(),
        code_references: [reference, reference],
      }),
    ).toThrow(/code_references.*duplicate/i);
  });

  it("rejects empty Japanese code reference labels", () => {
    expect(() =>
      parseTopic({
        ...minimalTopic(),
        code_references: [{ ...codeReference(), label_ja: "" }],
      }),
    ).toThrow(/label_ja/i);
  });

  it("rejects sequence messages with unknown actors", () => {
    expect(() =>
      parseTopic({
        ...minimalTopic(),
        code_references: [codeReference()],
        visualization: {
          kind: "sequence",
          actors: [{ id: "replay", label_ja: "リプレイ統括", code_ref: "run" }],
          messages: [
            {
              id: "decide",
              from: "replay",
              to: "missing",
              label_ja: "戦略判断を取得",
            },
          ],
        },
      }),
    ).toThrow(/actor reference/i);
  });

  it("rejects sequence references to unknown code references", () => {
    expect(() =>
      parseTopic({
        ...minimalTopic(),
        code_references: [codeReference()],
        visualization: {
          kind: "sequence",
          actors: [{ id: "replay", label_ja: "リプレイ統括", code_ref: "missing" }],
          messages: [],
        },
      }),
    ).toThrow(/code reference/i);
  });

  it("rejects duplicate sequence message ids", () => {
    const message = {
      id: "same",
      from: "replay",
      to: "strategy",
      label_ja: "戦略判断を取得",
    };
    expect(() =>
      parseTopic({
        ...minimalTopic(),
        code_references: [codeReference()],
        visualization: {
          kind: "sequence",
          actors: [
            { id: "replay", label_ja: "リプレイ統括", code_ref: "run" },
            { id: "strategy", label_ja: "戦略" },
          ],
          messages: [message, message],
        },
      }),
    ).toThrow(/messages.*duplicate/i);
  });
});
