import { ArrowRight, Box, CircleDot } from "lucide-react";
import { useState } from "react";

import type { ArchitectureVisualization } from "../content/schema";
import { layerDirectedGraph } from "./graphLayout";

export function ArchitectureFlow({
  visualization,
}: {
  visualization: ArchitectureVisualization;
}) {
  const layout = layerDirectedGraph(
    visualization.nodes.map((node) => node.id),
    visualization.edges,
  );
  const nodesById = new Map(visualization.nodes.map((node) => [node.id, node]));
  const orderedNodes = layout.layers
    .flat()
    .map((id) => nodesById.get(id))
    .filter((node): node is ArchitectureVisualization["nodes"][number] => node !== undefined);

  const [selectedId, setSelectedId] = useState(orderedNodes[0]?.id ?? "");
  const selected =
    orderedNodes.find((node) => node.id === selectedId) ?? orderedNodes[0];

  if (!selected) return null;

  return (
    <div className="viz-stack" data-testid="architecture-flow">
      <div className="architecture-flow" aria-label="システム構成図">
        {orderedNodes.map((node) => {
          const outgoing = visualization.edges.filter((edge) => edge.from === node.id);
          return (
            <div className="architecture-flow__item" key={node.id}>
              <button
                type="button"
                className="diagram-node"
                data-active={node.id === selected.id}
                aria-pressed={node.id === selected.id}
                onClick={() => setSelectedId(node.id)}
              >
                <span className="diagram-node__icon" aria-hidden="true">
                  <Box size={20} strokeWidth={1.7} />
                </span>
                <strong>{node.label}</strong>
                <span>{node.subtitle}</span>
              </button>
              {outgoing.map((edge) => {
                const target = nodesById.get(edge.to);
                return (
                  <ArrowRight
                    key={`${edge.from}-${edge.to}`}
                    className="architecture-flow__arrow"
                    size={18}
                    role="img"
                    aria-label={`${node.label} から ${target?.label ?? edge.to} への接続`}
                    data-testid={`edge-${edge.from}-${edge.to}`}
                  />
                );
              })}
            </div>
          );
        })}
      </div>

      <div className="detail-panel" aria-live="polite">
        <div className="detail-panel__lead">
          <CircleDot size={18} aria-hidden="true" />
          <div>
            <p className="eyeline">選択中の責務</p>
            <h3>{selected.label}</h3>
          </div>
        </div>
        <p>{selected.description}</p>
        <dl className="responsibility-grid">
          <div>
            <dt>入力</dt>
            <dd>{selected.input}</dd>
          </div>
          <div>
            <dt>出力</dt>
            <dd>{selected.output}</dd>
          </div>
          <div>
            <dt>ここでは持たない</dt>
            <dd>{selected.not_owned}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
