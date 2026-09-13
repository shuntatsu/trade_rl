import { getCodeSymbol } from "./codeSymbols";
import type { GuideTopic } from "./schema";

export type GuideSearchResult = {
  topicId: string;
  step?: string;
  symbol?: string;
  title: string;
  subtitle: string;
  kind: "topic" | "code" | "variable";
  score: number;
};

function normalize(value: string): string {
  return value.trim().toLocaleLowerCase("ja-JP");
}

function searchableText(topic: GuideTopic): string {
  const sectionText = topic.sections.flatMap((section) => [section.title, ...section.body]);
  return [
    topic.title,
    topic.nav_label,
    topic.summary,
    ...topic.keywords,
    ...sectionText,
  ]
    .join(" ")
    .toLocaleLowerCase("ja-JP");
}

function matchScore(value: string, query: string, exact: number, contains: number): number {
  const normalized = normalize(value);
  if (normalized === query) return exact;
  return normalized.includes(query) ? contains : 0;
}

export function searchTopics(topics: readonly GuideTopic[], query: string): GuideTopic[] {
  const normalized = normalize(query);
  if (!normalized) return [...topics];

  return topics
    .map((topic, index) => {
      const haystack = searchableText(topic);
      const title = topic.title.toLocaleLowerCase("ja-JP");
      const nav = topic.nav_label.toLocaleLowerCase("ja-JP");
      const score =
        (title.includes(normalized) ? 4 : 0) +
        (nav.includes(normalized) ? 3 : 0) +
        (topic.keywords.some((keyword) =>
          keyword.toLocaleLowerCase("ja-JP").includes(normalized),
        )
          ? 2
          : 0) +
        (haystack.includes(normalized) ? 1 : 0);
      return { topic, score, index };
    })
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map((item) => item.topic);
}

type RankedResult = GuideSearchResult & {
  topicIndex: number;
  itemIndex: number;
};

function publicResult(result: RankedResult): GuideSearchResult {
  return {
    topicId: result.topicId,
    ...(result.step ? { step: result.step } : {}),
    ...(result.symbol ? { symbol: result.symbol } : {}),
    title: result.title,
    subtitle: result.subtitle,
    kind: result.kind,
    score: result.score,
  };
}

export function searchGuide(
  topics: readonly GuideTopic[],
  query: string,
): GuideSearchResult[] {
  const normalized = normalize(query);
  if (!normalized) {
    return topics.map((topic) => ({
      topicId: topic.id,
      title: topic.title,
      subtitle: topic.summary,
      kind: "topic",
      score: 0,
    }));
  }

  const results: RankedResult[] = [];
  topics.forEach((topic, topicIndex) => {
    const titleScore = Math.max(
      matchScore(topic.title, normalized, 1000, 950),
      matchScore(topic.nav_label, normalized, 900, 875),
    );
    const keywordScore = topic.keywords.some((keyword) =>
      normalize(keyword).includes(normalized),
    )
      ? 350
      : 0;
    const bodyScore = searchableText(topic).includes(normalized) ? 300 : 0;
    const topicScore = Math.max(titleScore, keywordScore, bodyScore);
    if (topicScore > 0) {
      results.push({
        topicId: topic.id,
        title: topic.title,
        subtitle: topic.summary,
        kind: "topic",
        score: topicScore,
        topicIndex,
        itemIndex: -1,
      });
    }

    topic.code_references.forEach((reference, referenceIndex) => {
      const symbol = getCodeSymbol(reference.symbol);
      const path = symbol?.path;
      const codeScore = Math.max(
        matchScore(reference.label_ja, normalized, 850, 825),
        matchScore(reference.symbol, normalized, 800, 750),
        path ? matchScore(path, normalized, 700, 650) : 0,
        reference.description_ja.toLocaleLowerCase("ja-JP").includes(normalized)
          ? 400
          : 0,
      );
      if (codeScore > 0) {
        results.push({
          topicId: topic.id,
          symbol: reference.symbol,
          title: reference.label_ja,
          subtitle: [reference.symbol, path].filter(Boolean).join(" · "),
          kind: "code",
          score: codeScore,
          topicIndex,
          itemIndex: referenceIndex * 1000,
        });
      }

      reference.variables.forEach((variable, variableIndex) => {
        const variableScore = Math.max(
          matchScore(variable.label_ja, normalized, 810, 790),
          matchScore(variable.name, normalized, 800, 780),
          variable.description_ja.toLocaleLowerCase("ja-JP").includes(normalized)
            ? 390
            : 0,
        );
        if (variableScore <= 0) return;
        results.push({
          topicId: topic.id,
          symbol: reference.symbol,
          title: variable.label_ja,
          subtitle: [variable.name, reference.label_ja, path].filter(Boolean).join(" · "),
          kind: "variable",
          score: variableScore,
          topicIndex,
          itemIndex: referenceIndex * 1000 + variableIndex + 1,
        });
      });
    });
  });

  return results
    .sort(
      (left, right) =>
        right.score - left.score ||
        left.topicIndex - right.topicIndex ||
        left.itemIndex - right.itemIndex,
    )
    .map(publicResult);
}
