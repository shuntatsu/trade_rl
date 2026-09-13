import { Menu, X } from "lucide-react";
import { useState } from "react";

import type {
  DocumentGuideTopic,
  GuideManifestGroup,
} from "../content/documentSchema";

export function MobileNav({
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
  const [open, setOpen] = useState(false);
  const active = topics.find((topic) => topic.id === activeId) ?? topics[0];
  const byId = new Map(topics.map((topic) => [topic.id, topic]));

  return (
    <div className="mobile-nav">
      <button
        type="button"
        className="mobile-nav__trigger"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? <X size={18} /> : <Menu size={18} />}
        <span>{active?.nav_label ?? "トピック"}</span>
      </button>
      {open ? (
        <nav className="mobile-nav__panel" aria-label="Mobile Guide navigation">
          {groups.map((group) => (
            <div className="mobile-nav__group" key={group.id}>
              <strong>{group.label}</strong>
              {group.topics.map((topicId) => {
                const topic = byId.get(topicId);
                if (!topic) return null;
                return (
                  <a
                    key={topic.id}
                    href={`#${topic.id}`}
                    data-active={topic.id === activeId}
                    onClick={() => {
                      onNavigate(topic.id);
                      setOpen(false);
                    }}
                  >
                    {topic.nav_label}
                  </a>
                );
              })}
            </div>
          ))}
        </nav>
      ) : null}
    </div>
  );
}
