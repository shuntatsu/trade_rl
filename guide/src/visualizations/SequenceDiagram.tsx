import { ArrowLeft, ArrowRight } from "lucide-react";

import type { SequenceVisualization } from "../content/schema";

export function SequenceDiagram({
  visualization,
  selectedStep,
  onSelectStep,
}: {
  visualization: SequenceVisualization;
  selectedStep?: string;
  onSelectStep: (stepId: string) => void;
}) {
  const selectedIndex = Math.max(
    0,
    visualization.messages.findIndex((message) => message.id === selectedStep),
  );
  const selected = visualization.messages[selectedIndex];
  const actorLabels = new Map(
    visualization.actors.map((actor) => [actor.id, actor.label_ja]),
  );

  const move = (offset: number) => {
    if (!visualization.messages.length) return;
    const next = Math.min(
      visualization.messages.length - 1,
      Math.max(0, selectedIndex + offset),
    );
    const message = visualization.messages[next];
    if (message) onSelectStep(message.id);
  };

  return (
    <div className="sequence-diagram" aria-label="実装シーケンス">
      <div className="sequence-actors" aria-label="登場する実装要素">
        {visualization.actors.map((actor) => (
          <div className="sequence-actor" key={actor.id}>
            <strong>{actor.label_ja}</strong>
            {actor.code_ref ? <code>{actor.code_ref}</code> : null}
          </div>
        ))}
      </div>

      <div className="sequence-messages">
        {visualization.messages.map((message, index) => {
          const current = message.id === selected?.id;
          return (
            <button
              type="button"
              className="sequence-message"
              key={message.id}
              aria-current={current ? "step" : undefined}
              data-selected={current ? "true" : "false"}
              onClick={() => onSelectStep(message.id)}
            >
              <span className="sequence-message__index">{index + 1}</span>
              <span className="sequence-message__body">
                <strong>{message.label_ja}</strong>
                <small>
                  {actorLabels.get(message.from) ?? message.from} →{" "}
                  {actorLabels.get(message.to) ?? message.to}
                </small>
                {message.state_changes_ja?.length ? (
                  <ul>
                    {message.state_changes_ja.map((change) => (
                      <li key={change}>{change}</li>
                    ))}
                  </ul>
                ) : null}
              </span>
            </button>
          );
        })}
      </div>

      {visualization.messages.length ? (
        <div className="sequence-mobile-controls" aria-label="シーケンス移動">
          <button
            type="button"
            onClick={() => move(-1)}
            disabled={selectedIndex === 0}
          >
            <ArrowLeft size={16} aria-hidden="true" />
            前へ
          </button>
          <span>
            {selectedIndex + 1} / {visualization.messages.length}
          </span>
          <button
            type="button"
            onClick={() => move(1)}
            disabled={selectedIndex === visualization.messages.length - 1}
          >
            次へ
            <ArrowRight size={16} aria-hidden="true" />
          </button>
        </div>
      ) : null}
    </div>
  );
}
