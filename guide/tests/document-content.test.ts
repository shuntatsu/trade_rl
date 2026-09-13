import { describe, expect, it } from "vitest";

import manifestRaw from "../content/manifest.json";
import { loadTopics } from "../src/content/loadTopics";


describe("Markdown-first Guide content", () => {
  it("uses the document-guide-v2 manifest and reading order", () => {
    const manifest = manifestRaw as unknown as {
      schema_version?: string;
      groups?: Array<{ id?: string; topics?: string[] }>;
      reading_order?: string[];
    };

    expect(manifest.schema_version).toBe("document-guide-v2");
    expect(manifest.groups?.map((group) => group.id)).toEqual([
      "overview",
      "mechanics",
      "status",
      "reference",
    ]);
    expect(manifest.reading_order).not.toContain("code-map");
  });

  it("loads each topic with Markdown and an explicit document role", () => {
    const topics = loadTopics() as unknown as Array<{
      id: string;
      markdown?: string;
      role?: string;
      code_references: unknown[];
    }>;

    expect(topics).toHaveLength(8);
    expect(
      topics.every(
        (topic) => typeof topic.markdown === "string" && topic.markdown.length > 0,
      ),
    ).toBe(true);
    expect(Object.fromEntries(topics.map((topic) => [topic.id, topic.role]))).toEqual({
      overview: "overview",
      "implementation-replay": "detail",
      "implementation-ppo": "detail",
      "code-map": "reference",
      "data-flow": "detail",
      "execution-economics": "detail",
      "experiment-loop": "detail",
      "research-status": "status",
    });

    const overview = topics.find((topic) => topic.id === "overview");
    const status = topics.find((topic) => topic.id === "research-status");
    expect(overview?.code_references).toEqual([]);
    expect(status?.code_references).toEqual([]);
  });
});
