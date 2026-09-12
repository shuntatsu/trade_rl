import { ArrowRight } from "lucide-react";

import type { CodeMapVisualization } from "../content/schema";
import { layerDirectedGraph } from "./graphLayout";

export function CodeMap({
  visualization,
  selectedNode,
  onSelectNode,
}: {
  visualization: CodeMapVisualization;
  selectedNode?: string;
  onSelectNode: (nodeId: string) => void;
}) {
  const nodes = new Map(visualization.nodes.map((node) => [node.id, node]));
  const layout = layerDirectedGraph(
    visualization.nodes.map((node) => node.id),
    visualization.edges,
  );

  return (
    <div className="code-map" aria-label="コード責務地図">
      <div className="code-map__layers">
        {layout.layers.map((layer, layerIndex) => (
          <div className="code-map__layer" key={layer.join(":")}>
            <span className="code-map__layer-label">層 {layerIndex + 1}</span>
            {layer.map((nodeId) => {
              const node = nodes.get(nodeId);
              if (!node) return null;
              return (
                <button
                  type="button"
                  className="code-map__node"
                  key={node.id}
                  data-selected={node.id === selectedNode ? "true" : "false"}
                  aria-current={node.id === selectedNode ? "true" : undefined}
                  onClick={() => onSelectNode(node.id)}
                >
                  <strong>{node.label_ja}</strong>
                  <span>{node.description_ja}</span>
                  {node.code_ref ? <code>{node.code_ref}</code> : null}
                </button>
              );
            })}
          </div>
        ))}
      </div>

      <div className="code-map__edges" aria-label="宣言された責務の流れ">
        {visualization.edges.map((edge) => (
          <div
            className="code-map__edge"
            key={`${edge.from}->${edge.to}:${edge.relation}`}
            data-edge={`${edge.from}->${edge.to}`}
          >
            <span>{nodes.get(edge.from)?.label_ja ?? edge.from}</span>
            <ArrowRight size={15} aria-hidden="true" />
            <span>{nodes.get(edge.to)?.label_ja ?? edge.to}</span>
            <strong>{edge.label_ja}</strong>
            <small>{edge.relation === "calls" ? "呼び出し" : "データの流れ"}</small>
          </div>
        ))}
      </div>
    </div>
  );
}
