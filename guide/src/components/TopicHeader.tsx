import type { GuideTopic } from "../content/schema";
import { SourceLinks } from "./SourceLinks";

export function TopicHeader({ topic }: { topic: GuideTopic }) {
  return (
    <header className="topic-header">
      <div>
        <p className="eyeline">Interactive explanation</p>
        <h1>{topic.title}</h1>
        <p className="topic-header__summary">{topic.summary}</p>
      </div>
      <SourceLinks sources={topic.source_sections} />
    </header>
  );
}
