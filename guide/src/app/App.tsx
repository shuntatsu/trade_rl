import { ArrowRight, BookOpen, Info } from "lucide-react";

import { AppShell } from "../components/AppShell";
import { CodeInspector } from "../components/CodeInspector";
import { TopicHeader } from "../components/TopicHeader";
import { guideManifest, loadTopics } from "../content/loadTopics";
import type { GuideTopic } from "../content/schema";
import { VisualizationRenderer } from "../visualizations/VisualizationRenderer";
import { type GuideRoute, useHashRoute } from "./useHashRoute";
import { useTheme } from "./useTheme";

const TOPICS = loadTopics();
const TOPIC_IDS = TOPICS.map((topic) => topic.id);

type InspectorSelection = {
  selectedStep?: string;
  selectedNode?: string;
  referenceId?: string;
};

function resolveSequenceSelection(
  topic: GuideTopic,
  route: GuideRoute,
): InspectorSelection {
  if (topic.visualization.kind !== "sequence") return {};

  const messages = topic.visualization.messages;
  if (!messages.length) return {};
  const stepMessage = route.step
    ? messages.find((message) => message.id === route.step)
    : undefined;
  const symbolReference = route.symbol
    ? topic.code_references.find((reference) => reference.symbol === route.symbol)
    : undefined;
  const symbolMessage = symbolReference
    ? messages.find((message) => message.code_ref === symbolReference.id)
    : undefined;
  const selectedMessage = stepMessage ?? symbolMessage ?? messages[0];
  return {
    selectedStep: selectedMessage?.id,
    referenceId: symbolReference?.id ?? selectedMessage?.code_ref,
  };
}

function resolveCodeMapSelection(
  topic: GuideTopic,
  route: GuideRoute,
): InspectorSelection {
  if (topic.visualization.kind !== "code-map") return {};

  const nodes = topic.visualization.nodes;
  if (!nodes.length) return {};
  const stepNode = route.step ? nodes.find((node) => node.id === route.step) : undefined;
  const symbolReference = route.symbol
    ? topic.code_references.find((reference) => reference.symbol === route.symbol)
    : undefined;
  const symbolNode = symbolReference
    ? nodes.find((node) => node.code_ref === symbolReference.id)
    : undefined;
  const defaultNode = nodes.find((node) => node.code_ref) ?? nodes[0];
  const selectedNode = stepNode ?? symbolNode ?? defaultNode;
  return {
    selectedNode: selectedNode?.id,
    referenceId: symbolReference?.id ?? selectedNode?.code_ref,
  };
}

export function App() {
  const [route, navigate] = useHashRoute(TOPIC_IDS, guideManifest.home);
  const { theme, resolvedTheme, setTheme } = useTheme();
  const topic = TOPICS.find((item) => item.id === route.topicId) ?? TOPICS[0];
  if (!topic) return null;

  const currentIndex = TOPICS.findIndex((item) => item.id === topic.id);
  const nextTopic = TOPICS[(currentIndex + 1) % TOPICS.length];
  const sequenceSelection = resolveSequenceSelection(topic, route);
  const codeMapSelection = resolveCodeMapSelection(topic, route);
  const isSequence = topic.visualization.kind === "sequence";
  const isCodeMap = topic.visualization.kind === "code-map";
  const hasInspector = isSequence || isCodeMap;

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
          <span className="section-heading__hint">
            {hasInspector
              ? "項目を選ぶと対応する実装詳細を確認できます"
              : "クリック / タップで詳細を切替"}
          </span>
        </div>
        {isSequence ? (
          <div className="implementation-stage">
            <VisualizationRenderer
              visualization={topic.visualization}
              selectedStep={sequenceSelection.selectedStep}
              onSelectStep={(stepId) => navigate({ topicId: topic.id, step: stepId })}
            />
            <CodeInspector referenceId={sequenceSelection.referenceId} topic={topic} />
          </div>
        ) : isCodeMap ? (
          <div className="implementation-stage">
            <VisualizationRenderer
              visualization={topic.visualization}
              selectedNode={codeMapSelection.selectedNode}
              onSelectNode={(nodeId) => navigate({ topicId: topic.id, step: nodeId })}
            />
            <CodeInspector referenceId={codeMapSelection.referenceId} topic={topic} />
          </div>
        ) : (
          <VisualizationRenderer visualization={topic.visualization} />
        )}
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
