import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { parseTopic, type CodeMapVisualization } from "../src/content/schema";
import { CodeMap } from "../src/visualizations/CodeMap";

function topicWithVisualization(visualization: unknown) {
  return {
    id: "code-map",
    title: "コード地図",
    nav_label: "コード地図",
    summary: "責務フロー",
    keywords: [],
    source_sections: [
      {
        path: "docs/architecture/package-boundaries.md",
        heading: "Ownership",
        sha256: "0".repeat(64),
      },
    ],
    code_references: [],
    sections: [],
    visualization,
  };
}

describe("Guide code map contract", () => {
  it("rejects unknown nodes and unknown relations", () => {
    expect(() =>
      parseTopic(
        topicWithVisualization({
          kind: "code-map",
          nodes: [{ id: "data", label_ja: "データ", description_ja: "データ責務" }],
          edges: [
            {
              from: "data",
              to: "missing",
              relation: "data-flow",
              label_ja: "渡す",
            },
          ],
        }),
      ),
    ).toThrow(/edge reference/i);

    expect(() =>
      parseTopic(
        topicWithVisualization({
          kind: "code-map",
          nodes: [
            { id: "data", label_ja: "データ", description_ja: "データ責務" },
            { id: "risk", label_ja: "リスク", description_ja: "リスク責務" },
          ],
          edges: [
            {
              from: "data",
              to: "risk",
              relation: "imports",
              label_ja: "渡す",
            },
          ],
        }),
      ),
    ).toThrow(/relation/i);
  });

  it("rejects cyclic responsibility flows", () => {
    expect(() =>
      parseTopic(
        topicWithVisualization({
          kind: "code-map",
          nodes: [
            { id: "data", label_ja: "データ", description_ja: "データ責務" },
            { id: "risk", label_ja: "リスク", description_ja: "リスク責務" },
          ],
          edges: [
            {
              from: "data",
              to: "risk",
              relation: "data-flow",
              label_ja: "渡す",
            },
            {
              from: "risk",
              to: "data",
              relation: "calls",
              label_ja: "呼ぶ",
            },
          ],
        }),
      ),
    ).toThrow(/cycle/i);
  });
});

describe("Guide code map renderer", () => {
  it("renders only declared directed edges and selects nodes", async () => {
    const user = userEvent.setup();
    const onSelectNode = vi.fn();
    const visualization: CodeMapVisualization = {
      kind: "code-map",
      nodes: [
        { id: "data", label_ja: "データ", description_ja: "内部契約" },
        {
          id: "integrations",
          label_ja: "外部接続",
          description_ja: "市場source adapter",
        },
      ],
      edges: [
        {
          from: "integrations",
          to: "data",
          relation: "data-flow",
          label_ja: "市場sourceを内部契約へ変換",
        },
      ],
    };

    const { container } = render(
      <CodeMap
        visualization={visualization}
        selectedNode="data"
        onSelectNode={onSelectNode}
      />,
    );

    expect(container.querySelector('[data-edge="integrations->data"]')).not.toBeNull();
    expect(container.querySelector('[data-edge="data->integrations"]')).toBeNull();
    expect(screen.getByText("市場sourceを内部契約へ変換")).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "外部接続の実装詳細" }),
    );
    expect(onSelectNode).toHaveBeenCalledWith("integrations");
  });
});
