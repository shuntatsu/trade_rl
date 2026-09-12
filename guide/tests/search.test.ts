import { describe, expect, it } from "vitest";

import { loadTopics } from "../src/content/loadTopics";
import { searchTopics } from "../src/content/search";


describe("guide search", () => {
  it("matches titles, summaries, keywords, and section copy", () => {
    const topics = loadTopics();

    expect(searchTopics(topics, "PPO")[0]?.id).toBe("ppo-observation-v2");
    expect(searchTopics(topics, "手数料").map((topic) => topic.id)).toContain(
      "execution-economics",
    );
    expect(searchTopics(topics, "winner").map((topic) => topic.id)).toContain(
      "research-status",
    );
  });

  it("returns manifest order for an empty query", () => {
    const topics = loadTopics();
    expect(searchTopics(topics, "   ").map((topic) => topic.id)).toEqual(
      topics.map((topic) => topic.id),
    );
  });
});
