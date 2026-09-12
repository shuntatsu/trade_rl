import { AlertTriangle, ArrowRight } from "lucide-react";
import { useState } from "react";

import type { ExperimentLoopVisualization } from "../content/schema";

export function ExperimentLoop({
  visualization,
}: {
  visualization: ExperimentLoopVisualization;
}) {
  const [selectedId, setSelectedId] = useState(visualization.steps[0]?.id ?? "");
  const selected =
    visualization.steps.find((step) => step.id === selectedId) ?? visualization.steps[0];
  if (!selected) return null;

  return (
    <div className="viz-stack" data-testid="experiment-loop">
      <div className="experiment-loop" aria-label="Controlled Experiment state flow">
        {visualization.steps.map((step, index) => (
          <div className="experiment-loop__item" key={step.id}>
            <button
              type="button"
              className="loop-node"
              data-phase={step.phase ?? "default"}
              data-active={step.id === selected.id}
              aria-pressed={step.id === selected.id}
              onClick={() => setSelectedId(step.id)}
            >
              <span>{index + 1}</span>
              <strong>{step.label}</strong>
            </button>
            {index < visualization.steps.length - 1 ? (
              <ArrowRight className="experiment-loop__arrow" size={16} aria-hidden="true" />
            ) : null}
          </div>
        ))}
      </div>

      <div className="detail-panel" aria-live="polite">
        <p className="eyeline">{selected.phase ?? "step"}</p>
        <h3>{selected.label}</h3>
        <p>{selected.detail}</p>
      </div>

      <div className="failure-note">
        <AlertTriangle size={18} aria-hidden="true" />
        <div>
          <strong>Fail closed</strong>
          <p>
            契約違反や証拠不整合は {visualization.failure_states.join(" / ")} として終端し、
            baselineを都合よく書き換えて続行しません。
          </p>
        </div>
      </div>
    </div>
  );
}
