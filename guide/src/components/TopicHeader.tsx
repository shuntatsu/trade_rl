import type { DocumentGuideTopic } from "../content/documentSchema";
import { SourceLinks } from "./SourceLinks";

export function TopicHeader({ topic }: { topic: DocumentGuideTopic }) {
  return (
    <header className="topic-header">
      <div>
        <p className="eyeline">Technical Guide</p>
        <h1>{topic.title}</h1>
        <p className="topic-header__summary">{topic.summary}</p>
      </div>
      <SourceLinks sources={topic.source_sections} />
    </header>
  );
}
