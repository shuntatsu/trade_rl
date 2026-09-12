import { ArrowRight, Ban, CheckCircle2 } from "lucide-react";
import { useState } from "react";

import type { EconomicsVisualization } from "../content/schema";

export function EconomicsAuthorityPath({
  visualization,
}: {
  visualization: EconomicsVisualization;
}) {
  const [selectedId, setSelectedId] = useState(
    visualization.components[0]?.id ?? "",
  );
  const selected =
    visualization.components.find((item) => item.id === selectedId) ??
    visualization.components[0];
  if (!selected) return null;

  return (
    <div className="viz-stack" data-testid="economics-path">
      <div className="authority-path" aria-label="実行コストauthority path">
        {visualization.components.map((component, index) => (
          <div className="authority-path__item" key={component.id}>
            <button
              type="button"
              className="authority-node"
              data-active={component.id === selected.id}
              data-disabled={component.status === "disabled"}
              aria-pressed={component.id === selected.id}
              onClick={() => setSelectedId(component.id)}
            >
              <span aria-hidden="true">
                {component.status === "disabled" ? (
                  <Ban size={18} />
                ) : (
                  <CheckCircle2 size={18} />
                )}
              </span>
              <strong>{component.label}</strong>
              <small>{component.role}</small>
            </button>
            {index < visualization.components.length - 1 ? (
              <ArrowRight size={17} className="authority-path__arrow" aria-hidden="true" />
            ) : null}
          </div>
        ))}
      </div>

      <div className="detail-panel" aria-live="polite">
        <p className="eyeline">{selected.status}</p>
        <h3>{selected.label}</h3>
        <p>{selected.description}</p>
      </div>

      <div className="assumption-grid" aria-label="実行コスト前提">
        {visualization.assumptions.map((assumption) => (
          <div className="assumption-row" key={assumption.label}>
            <span>{assumption.label}</span>
            <strong>{assumption.state}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}
