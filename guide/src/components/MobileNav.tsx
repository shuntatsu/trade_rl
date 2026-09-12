import { Menu, X } from "lucide-react";
import { useState } from "react";

import type { GuideTopic } from "../content/schema";

export function MobileNav({
  topics,
  activeId,
  onNavigate,
}: {
  topics: GuideTopic[];
  activeId: string;
  onNavigate: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const active = topics.find((topic) => topic.id === activeId) ?? topics[0];

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
          {topics.map((topic) => (
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
          ))}
        </nav>
      ) : null}
    </div>
  );
}
