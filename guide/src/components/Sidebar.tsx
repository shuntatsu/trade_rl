import {
  Activity,
  BrainCircuit,
  CircleDollarSign,
  Database,
  FlaskConical,
  Layers3,
  Route,
} from "lucide-react";

import type { GuideTopic } from "../content/schema";

const ICONS = {
  overview: Layers3,
  "data-flow": Route,
  architecture: Database,
  "execution-economics": CircleDollarSign,
  "ppo-observation-v2": BrainCircuit,
  "experiment-loop": FlaskConical,
  "research-status": Activity,
} as const;

export function Sidebar({
  topics,
  activeId,
  onNavigate,
}: {
  topics: GuideTopic[];
  activeId: string;
  onNavigate: (id: string) => void;
}) {
  return (
    <aside className="sidebar" aria-label="Guide navigation">
      <a className="brand" href="#overview" onClick={() => onNavigate("overview")}>
        <span className="brand__mark" aria-hidden="true">↗</span>
        <span>
          <strong>trade_rl</strong>
          <small>Interactive Guide</small>
        </span>
      </a>

      <nav>
        <p className="sidebar__label">理解する</p>
        <ul className="sidebar__nav">
          {topics.map((topic) => {
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
      </nav>

      <div className="sidebar__note">
        <strong>人間向け説明層</strong>
        <p>技術仕様の正本は docs/ にあります。</p>
      </div>
    </aside>
  );
}
