import { AlertCircle, CheckCircle2, CircleDashed, ShieldAlert } from "lucide-react";
import { useState } from "react";

import type { ResearchStatusVisualization } from "../content/schema";

const GROUP_ICONS = {
  verified: CheckCircle2,
  pending: CircleDashed,
  "not-claimed": ShieldAlert,
  limitations: AlertCircle,
} as const;

export function ResearchStatusBoard({
  visualization,
}: {
  visualization: ResearchStatusVisualization;
}) {
  const [selectedId, setSelectedId] = useState(visualization.groups[0]?.id ?? "");

  return (
    <div className="status-board" data-testid="research-status-board">
      {visualization.groups.map((group) => {
        const Icon = GROUP_ICONS[group.id as keyof typeof GROUP_ICONS] ?? CircleDashed;
        const active = group.id === selectedId;
        return (
          <section className="status-group" data-status={group.id} data-active={active} key={group.id}>
            <button
              type="button"
              className="status-group__header"
              aria-expanded={active}
              onClick={() => setSelectedId(group.id)}
            >
              <Icon size={19} aria-hidden="true" />
              <span>{group.label}</span>
              <small>{group.items.length}</small>
            </button>
            <ul className="status-group__items" hidden={!active}>
              {group.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
