import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { useState } from "react";

import type { FlowStep } from "../content/schema";

export function FlowStepper({ steps }: { steps: FlowStep[] }) {
  const [index, setIndex] = useState(0);
  const safeIndex = Math.min(index, Math.max(steps.length - 1, 0));
  const selected = steps[safeIndex];
  if (!selected) return null;

  return (
    <div className="viz-stack" data-testid="flow-stepper">
      <ol className="stepper" aria-label="処理ステップ">
        {steps.map((step, stepIndex) => (
          <li key={step.id}>
            <button
              type="button"
              className="stepper__step"
              data-active={stepIndex === safeIndex}
              data-complete={stepIndex < safeIndex}
              aria-current={stepIndex === safeIndex ? "step" : undefined}
              onClick={() => setIndex(stepIndex)}
            >
              <span className="stepper__number" aria-hidden="true">
                {stepIndex < safeIndex ? <Check size={15} /> : stepIndex + 1}
              </span>
              <span>
                <strong>{step.label}</strong>
                {step.summary ? <small>{step.summary}</small> : null}
              </span>
            </button>
          </li>
        ))}
      </ol>

      <div className="detail-panel flow-detail" aria-live="polite">
        <div>
          <p className="eyeline">
            Step {safeIndex + 1} / {steps.length}
          </p>
          <h3>{selected.label}</h3>
          <p>{selected.detail}</p>
        </div>
        <div className="flow-detail__actions">
          <button
            type="button"
            className="icon-button"
            disabled={safeIndex === 0}
            onClick={() => setIndex((value) => Math.max(0, value - 1))}
            aria-label="前のステップ"
          >
            <ArrowLeft size={18} />
          </button>
          <button
            type="button"
            className="icon-button"
            disabled={safeIndex === steps.length - 1}
            onClick={() => setIndex((value) => Math.min(steps.length - 1, value + 1))}
            aria-label="次のステップ"
          >
            <ArrowRight size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
