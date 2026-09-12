import { ArrowRight, BookOpen, Info } from "lucide-react";

import { guideManifest, loadTopics } from "../content/loadTopics";
import { AppShell } from "../components/AppShell";
import { TopicHeader } from "../components/TopicHeader";
import { VisualizationRenderer } from "../visualizations/VisualizationRenderer";
import { useHashRoute } from "./useHashRoute";
import { useTheme } from "./useTheme";

const TOPICS = loadTopics();
const TOPIC_IDS = TOPICS.map((topic) => topic.id);

export function App() {
  const [activeId, navigate] = useHashRoute(TOPIC_IDS, guideManifest.home);
  const { theme, resolvedTheme, setTheme } = useTheme();
  const topic = TOPICS.find((item) => item.id === activeId) ?? TOPICS[0];
  if (!topic) return null;

  const currentIndex = TOPICS.findIndex((item) => item.id === topic.id);
  const nextTopic = TOPICS[(currentIndex + 1) % TOPICS.length];

  return (
    <AppShell
      topics={TOPICS}
      activeId={topic.id}
      onNavigate={navigate}
      theme={theme}
      resolvedTheme={resolvedTheme}
      onThemeChange={setTheme}
    >
      <div className="guide-notice" role="note">
        <Info size={17} aria-hidden="true" />
        <span>
          これは人間向けの説明UIです。技術仕様・研究状態の正本は <strong>docs/</strong> です。
        </span>
      </div>

      <TopicHeader topic={topic} />

      <section className="visual-stage" aria-labelledby="visual-stage-title">
        <div className="section-heading">
          <div>
            <p className="eyeline">Explore</p>
            <h2 id="visual-stage-title">図を触って理解する</h2>
          </div>
          <span className="section-heading__hint">クリック / タップで詳細を切替</span>
        </div>
        <VisualizationRenderer visualization={topic.visualization} />
      </section>

      <section className="explanation-grid" aria-label="説明">
        {topic.sections.map((section) => (
          <article className="explanation-card" key={section.title}>
            <BookOpen size={18} aria-hidden="true" />
            <div>
              <h2>{section.title}</h2>
              {section.body.map((paragraph) => (
                <p key={paragraph}>{paragraph}</p>
              ))}
            </div>
          </article>
        ))}
      </section>

      {nextTopic ? (
        <button type="button" className="next-topic" onClick={() => navigate(nextTopic.id)}>
          <span>
            <small>次に見る</small>
            <strong>{nextTopic.title}</strong>
          </span>
          <ArrowRight size={20} aria-hidden="true" />
        </button>
      ) : null}
    </AppShell>
  );
}
