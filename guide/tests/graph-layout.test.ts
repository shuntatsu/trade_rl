import { describe, expect, it } from "vitest";

import { extractMarkdownHeadings, slugMarkdownHeading } from "../src/content/markdown";


describe("Markdown heading contract", () => {
  it("creates deterministic slugs for Japanese and implementation terms", () => {
    expect(slugMarkdownHeading("4. hard riskを適用する")).toBe(
      "4-hard-riskを適用する",
    );
    expect(slugMarkdownHeading("PPO Observation v2は5区分")).toBe(
      "ppo-observation-v2は5区分",
    );
  });

  it("rejects duplicate normalized heading targets", () => {
    expect(() =>
      extractMarkdownHeadings("## Hard Risk\n\ntext\n\n## hard-risk\n"),
    ).toThrow(/duplicate Markdown heading slug/i);
  });
});
