import { ArrowRight, Info } from "lucide-react";
import { useEffect } from "react";

import { AppShell } from "../components/AppShell";
import { ImplementationReferenceAppendix } from "../components/ImplementationReferenceAppendix";
import { MarkdownArticle } from "../components/MarkdownArticle";
import { TopicHeader } from "../components/TopicHeader";
import { guideManifest, loadTopics } from "../content/loadTopics";
import { type GuideRoute, useHashRoute } from "./useHashRoute";
import { useTheme } from "./useTheme";

const TOPICS = loadTopics();
const TOPIC_IDS = TOPICS.map((topic) => topic.id);

function nextReadingTopic(route: GuideRoute) {
  const index = guideManifest.reading_order.indexOf(route.topicId);
  if (index < 0) return undefined;
  const nextId = guideManifest.reading_order[index + 1];
  return nextId ? TOPICS.find((topic) => topic.id === nextId) : undefined;
}

export function App() {
  const [route, navigate] = useHashRoute(TOPIC_IDS, guideManifest.home);
  const { theme, resolvedTheme, setTheme } = useTheme();
  const topic = TOPICS.find((item) => item.id === route.topicId) ?? TOPICS[0];
  const nextTopic = nextReadingTopic(route);

  useEffect(() => {
    if (!topic) return;
    if (!route.heading) {
      window.scrollTo({ top: 0 });
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      document.getElementById(route.heading ?? "")?.scrollIntoView({ block: "start" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [route.heading, topic]);

  if (!topic) return null;

  return (
    <AppShell
      topics={TOPICS}
      groups={guideManifest.groups}
      activeId={topic.id}
      onNavigate={navigate}
      theme={theme}
      resolvedTheme={resolvedTheme}
      onThemeChange={setTheme}
    >
      <div className="guide-notice" role="note">
        <Info size={17} aria-hidden="true" />
        <span>
          これは人間向けの説明層です。技術仕様・研究状態の正本は <strong>docs/</strong> です。
        </span>
      </div>

      <TopicHeader topic={topic} />
      <MarkdownArticle topic={topic} />
      <ImplementationReferenceAppendix
        key={`${topic.id}:${route.symbol ?? ""}`}
        topic={topic}
        openSymbol={route.symbol}
      />

      {nextTopic ? (
        <button type="button" className="next-topic" onClick={() => navigate(nextTopic.id)}>
          <span>
            <small>次に読む</small>
            <strong>{nextTopic.title}</strong>
          </span>
          <ArrowRight size={20} aria-hidden="true" />
        </button>
      ) : null}
    </AppShell>
  );
}
