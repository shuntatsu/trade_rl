import { ExternalLink } from "lucide-react";

import type { SourceSection } from "../content/documentSchema";

const REPOSITORY = "https://github.com/shuntatsu/trade_rl/blob/main/";

export function SourceLinks({ sources }: { sources: SourceSection[] }) {
  return (
    <div className="source-links" aria-label="この説明の正本">
      <span>正本</span>
      {sources.map((source) => (
        <a
          key={`${source.path}#${source.heading}`}
          href={`${REPOSITORY}${source.path}`}
          target="_blank"
          rel="noreferrer"
        >
          {source.path.replace("docs/", "")} · {source.heading}
          <ExternalLink size={13} aria-hidden="true" />
        </a>
      ))}
    </div>
  );
}
