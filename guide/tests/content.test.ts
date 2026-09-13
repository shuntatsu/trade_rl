import { describe, expect, it } from "vitest";

import {
  parseDocumentGuideManifest,
  parseDocumentTopic,
} from "../src/content/documentSchema";
import { loadTopics } from "../src/content/loadTopics";

function sourceSection() {
  return {
    path: "docs/architecture/lean-core.md",
    heading: "Core flow",
    sha256: "0".repeat(64),
  };
}

function codeReference(id = "run") {
  return {
    id,
    symbol: "trade_rl.demo.run",
    kind: "function",
    source_sha256: "a".repeat(64),
    label_ja: "デモ処理",
    description_ja: "デモ処理です。",
    variables: [],
    tests: [],
  };
}

function topicMetadata(overrides: Record<string, unknown> = {}) {
  return {
    id: "demo",
    title: "デモ",
    nav_label: "デモ",
    summary: "デモ",
    role: "detail",
    keywords: [],
    source_sections: [sourceSection()],
    code_references: [],
    ...overrides,
  };
}

describe("Markdown-first Guide content", () => {
  it("loads all reviewed pages in grouped navigation order", () => {
    expect(loadTopics().map((topic) => topic.id)).toEqual([
      "overview",
      "data-flow",
      "implementation-replay",
      "implementation-ppo",
      "execution-economics",
      "experiment-loop",
      "research-status",
      "code-map",
    ]);
  });

  it("parses non-empty Markdown with an explicit role", () => {
    const topic = parseDocumentTopic(topicMetadata(), "## 説明\n\n本文\n");
    expect(topic.role).toBe("detail");
    expect(topic.headings.map((heading) => heading.text)).toEqual(["説明"]);
  });

  it("rejects unsupported document roles", () => {
    expect(() =>
      parseDocumentTopic(topicMetadata({ role: "interactive" }), "## 説明\n"),
    ).toThrow(/unsupported Guide page role/i);
  });

  it("forbids implementation references on overview and status pages", () => {
    for (const role of ["overview", "status"] as const) {
      expect(() =>
        parseDocumentTopic(
          topicMetadata({ role, code_references: [codeReference()] }),
          "## 説明\n",
        ),
      ).toThrow(/must not expose code_references/i);
    }
  });

  it("requires implementation references on reference pages", () => {
    expect(() =>
      parseDocumentTopic(topicMetadata({ role: "reference" }), "## 説明\n"),
    ).toThrow(/requires code_references/i);
  });

  it("rejects duplicate code-reference ids", () => {
    expect(() =>
      parseDocumentTopic(
        topicMetadata({
          code_references: [codeReference("same"), codeReference("same")],
        }),
        "## 説明\n",
      ),
    ).toThrow(/code reference ids must be unique/i);
  });

  it("rejects empty Markdown pages", () => {
    expect(() => parseDocumentTopic(topicMetadata(), "   \n")).toThrow(
      /Markdown page must be non-empty/i,
    );
  });

  it("rejects invalid manifest grouping and reading order", () => {
    expect(() =>
      parseDocumentGuideManifest({
        schema_version: "document-guide-v2",
        home: "overview",
        groups: [
          { id: "overview", label: "概要", topics: ["overview"] },
          { id: "reference", label: "参照", topics: ["overview"] },
        ],
        reading_order: ["overview"],
      }),
    ).toThrow(/grouped topics must be unique/i);

    expect(() =>
      parseDocumentGuideManifest({
        schema_version: "document-guide-v2",
        home: "overview",
        groups: [{ id: "overview", label: "概要", topics: ["overview"] }],
        reading_order: ["missing"],
      }),
    ).toThrow(/unknown topic/i);
  });
});
