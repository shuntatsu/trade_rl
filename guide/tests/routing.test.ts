import { describe, expect, it } from "vitest";

import { normalizeHashRoute } from "../src/app/useHashRoute";


describe("guide hash routing", () => {
  const ids = ["overview", "data-flow", "architecture"] as const;

  it("accepts a known topic hash", () => {
    expect(normalizeHashRoute("#data-flow", ids, "overview")).toBe("data-flow");
  });

  it("falls back to home for missing or unknown hashes", () => {
    expect(normalizeHashRoute("", ids, "overview")).toBe("overview");
    expect(normalizeHashRoute("#unknown", ids, "overview")).toBe("overview");
  });
});
