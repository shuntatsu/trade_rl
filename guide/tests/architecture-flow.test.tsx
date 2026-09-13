import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ArchitectureNode } from "../src/content/schema";
import { ArchitectureFlow } from "../src/visualizations/ArchitectureFlow";

function node(id: string): ArchitectureNode {
  return {
    id,
    label: id,
    subtitle: `${id} subtitle`,
    description: `${id} description`,
    input: `${id} input`,
    output: `${id} output`,
    not_owned: `${id} not owned`,
  };
}

describe("ArchitectureFlow", () => {
  it("renders only declared architecture edges when node order disagrees", () => {
    const visualization = {
      kind: "architecture" as const,
      nodes: [node("data"), node("integrations"), node("strategies")],
      edges: [
        { from: "integrations", to: "data" },
        { from: "data", to: "strategies" },
      ],
    };

    render(<ArchitectureFlow visualization={visualization} />);

    expect(screen.getByTestId("edge-integrations-data")).toBeInTheDocument();
    expect(screen.getByTestId("edge-data-strategies")).toBeInTheDocument();
    expect(screen.queryByTestId("edge-data-integrations")).not.toBeInTheDocument();
  });
});
