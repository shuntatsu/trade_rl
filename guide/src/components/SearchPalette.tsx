import { Search, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { searchTopics } from "../content/search";
import type { GuideTopic } from "../content/schema";

export function SearchPalette({
  topics,
  onNavigate,
}: {
  topics: GuideTopic[];
  onNavigate: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const results = useMemo(() => searchTopics(topics, query).slice(0, 7), [query, topics]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
      }
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (open) window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [open]);

  const choose = (id: string) => {
    onNavigate(id);
    setOpen(false);
    setQuery("");
  };

  return (
    <>
      <button type="button" className="search-trigger" onClick={() => setOpen(true)}>
        <Search size={17} aria-hidden="true" />
        <span>ガイドを検索</span>
        <kbd>⌘K</kbd>
      </button>
      {open ? (
        <div className="search-backdrop" role="presentation" onMouseDown={() => setOpen(false)}>
          <section
            className="search-palette"
            role="dialog"
            aria-modal="true"
            aria-label="Guide search"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="search-palette__input">
              <Search size={18} aria-hidden="true" />
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="例: Observation、実行コスト、Study"
                aria-label="ガイドを検索"
              />
              <button type="button" className="icon-button" onClick={() => setOpen(false)} aria-label="検索を閉じる">
                <X size={18} />
              </button>
            </div>
            <div className="search-palette__results" aria-live="polite">
              {results.length ? (
                results.map((topic) => (
                  <button
                    type="button"
                    key={topic.id}
                    className="search-result"
                    onClick={() => choose(topic.id)}
                  >
                    <strong>{topic.title}</strong>
                    <span>{topic.summary}</span>
                  </button>
                ))
              ) : (
                <p className="empty-state">一致するトピックがありません。</p>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </>
  );
}
