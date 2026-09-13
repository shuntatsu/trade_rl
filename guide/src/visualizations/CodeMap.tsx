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
  const neighborhood = new Set<string>(selectedNode ? [selectedNode] : []);
  for (const edge of visualization.edges) {
    if (edge.from === selectedNode) neighborhood.add(edge.to);
    if (edge.to === selectedNode) neighborhood.add(edge.from);
  }
  const visibleOnMobile = neighborhood.size ? neighborhood : new Set(nodes.keys());

  return (
    <div className="code-map" aria-label="コード責務地図">
      <div className="code-map__outline" aria-label="責務の全体一覧">
        {visualization.nodes.map((node) => (
          <button
            type="button"
            key={node.id}
            aria-label={`${node.label_ja}を一覧で選択`}
            data-selected={node.id === selectedNode ? "true" : "false"}
            onClick={() => onSelectNode(node.id)}
          >
            {node.label_ja}
          </button>
        ))}
      </div>

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
                  aria-label={`${node.label_ja}の実装詳細`}
                  data-selected={node.id === selectedNode ? "true" : "false"}
                  data-mobile-visible={visibleOnMobile.has(node.id) ? "true" : "false"}
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
        {visualization.edges.map((edge) => {
          const mobileVisible = edge.from === selectedNode || edge.to === selectedNode;
          return (
            <div
              className="code-map__edge"
              key={`${edge.from}->${edge.to}:${edge.relation}`}
              data-edge={`${edge.from}->${edge.to}`}
              data-mobile-visible={mobileVisible ? "true" : "false"}
            >
              <span>{nodes.get(edge.from)?.label_ja ?? edge.from}</span>
              <ArrowRight size={15} aria-hidden="true" />
              <span>{nodes.get(edge.to)?.label_ja ?? edge.to}</span>
              <strong>{edge.label_ja}</strong>
              <small>{edge.relation === "calls" ? "呼び出し" : "データの流れ"}</small>
            </div>
          );
        })}
      </div>
    </div>
  );
}
