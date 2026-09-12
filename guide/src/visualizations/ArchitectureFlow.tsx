import { ArrowRight, Box, CircleDot } from "lucide-react";
import { useState } from "react";

import type { ArchitectureVisualization } from "../content/schema";

export function ArchitectureFlow({
  visualization,
}: {
  visualization: ArchitectureVisualization;
}) {
  const [selectedId, setSelectedId] = useState(visualization.nodes[0]?.id ?? "");
  const selected =
    visualization.nodes.find((node) => node.id === selectedId) ?? visualization.nodes[0];

  if (!selected) return null;

  return (
    <div className="viz-stack" data-testid="architecture-flow">
      <div className="architecture-flow" aria-label="システム構成図">
        {visualization.nodes.map((node, index) => (
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
            {index < visualization.nodes.length - 1 ? (
              <ArrowRight className="architecture-flow__arrow" size={18} aria-hidden="true" />
            ) : null}
          </div>
        ))}
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
