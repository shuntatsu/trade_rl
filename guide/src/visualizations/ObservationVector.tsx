import { EyeOff } from "lucide-react";
import { useMemo, useState } from "react";

import type { ObservationVisualization } from "../content/schema";

export function ObservationVector({
  visualization,
}: {
  visualization: ObservationVisualization;
}) {
  const [selectedId, setSelectedId] = useState(
    visualization.segments[0]?.id ?? "",
  );
  const totalWeight = useMemo(
    () => visualization.segments.reduce((total, segment) => total + segment.weight, 0),
    [visualization.segments],
  );
  const selected =
    visualization.segments.find((segment) => segment.id === selectedId) ??
    visualization.segments[0];
  if (!selected || totalWeight <= 0) return null;

  return (
    <div className="viz-stack" data-testid="observation-vector">
      <div className="observation-vector" aria-label="PPO Observation v2 vector">
        {visualization.segments.map((segment, index) => (
          <button
            key={segment.id}
            type="button"
            className="observation-segment"
            data-segment={index + 1}
            data-active={segment.id === selected.id}
            aria-pressed={segment.id === selected.id}
            style={{ flexGrow: segment.weight }}
            onClick={() => setSelectedId(segment.id)}
          >
            <span>{segment.label}</span>
          </button>
        ))}
      </div>

      <div className="detail-panel" aria-live="polite">
        <p className="eyeline">Observation segment</p>
        <h3>{selected.label}</h3>
        <p>{selected.description}</p>
      </div>

      <div className="excluded-panel">
        <div className="excluded-panel__title">
          <EyeOff size={18} aria-hidden="true" />
          <h3>意図的に入れない情報</h3>
        </div>
        <ul className="compact-list">
          {visualization.excluded.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
