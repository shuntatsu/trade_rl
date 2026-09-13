import {
  Activity,
  BrainCircuit,
  CircleDollarSign,
  Database,
  FlaskConical,
  Layers3,
  Route,
  ScrollText,
} from "lucide-react";

import type {
  DocumentGuideTopic,
  GuideManifestGroup,
} from "../content/documentSchema";

const ICONS = {
  overview: Layers3,
  "data-flow": Route,
  "implementation-replay": ScrollText,
  "implementation-ppo": BrainCircuit,
  "execution-economics": CircleDollarSign,
  "experiment-loop": FlaskConical,
  "research-status": Activity,
  "code-map": Database,
} as const;

export function Sidebar({
  topics,
  groups,
  activeId,
  onNavigate,
}: {
  topics: DocumentGuideTopic[];
  groups: GuideManifestGroup[];
  activeId: string;
  onNavigate: (id: string) => void;
}) {
  const byId = new Map(topics.map((topic) => [topic.id, topic]));

  return (
    <aside className="sidebar" aria-label="Guide navigation">
      <a className="brand" href="#overview" onClick={() => onNavigate("overview")}>
        <span className="brand__mark" aria-hidden="true">↗</span>
        <span>
          <strong>trade_rl</strong>
          <small>Technical Guide</small>
        </span>
      </a>

      <nav>
        {groups.map((group) => (
          <section className="sidebar__group" key={group.id} aria-labelledby={`nav-${group.id}`}>
            <p className="sidebar__label" id={`nav-${group.id}`}>
              {group.label}
            </p>
            <ul className="sidebar__nav">
              {group.topics.map((topicId) => {
                const topic = byId.get(topicId);
                if (!topic) return null;
                const Icon = ICONS[topic.id as keyof typeof ICONS] ?? Layers3;
                const active = topic.id === activeId;
                return (
                  <li key={topic.id}>
                    <a
                      href={`#${topic.id}`}
                      className="sidebar__link"
                      data-active={active}
                      aria-current={active ? "page" : undefined}
                      onClick={() => onNavigate(topic.id)}
                    >
                      <Icon size={18} strokeWidth={1.7} aria-hidden="true" />
                      <span>{topic.nav_label}</span>
                    </a>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </nav>

      <div className="sidebar__note">
        <strong>人間向け説明層</strong>
        <p>技術仕様・研究状態の正本は docs/ にあります。</p>
      </div>
    </aside>
  );
}
