import { describe, expect, it } from "vitest";

import { layerDirectedGraph } from "../src/visualizations/graphLayout";

describe("layerDirectedGraph", () => {
  it("layers a directed graph without inventing adjacency", () => {
    expect(
      layerDirectedGraph(
        ["data", "integrations", "strategies"],
        [
          { from: "integrations", to: "data" },
          { from: "data", to: "strategies" },
        ],
      ).layers,
    ).toEqual([["integrations"], ["data"], ["strategies"]]);
  });

  it("rejects cycles", () => {
    expect(() =>
      layerDirectedGraph(
        ["a", "b"],
        [
          { from: "a", to: "b" },
          { from: "b", to: "a" },
        ],
      ),
    ).toThrow("directed graph contains a cycle");
  });
});
