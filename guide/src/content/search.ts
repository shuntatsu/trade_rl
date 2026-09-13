import { getCodeSymbol } from "./codeSymbols";
import type { DocumentGuideTopic } from "./documentSchema";
import { extractMarkdownText } from "./markdown";

export type GuideSearchResult = {
  topicId: string;
  heading?: string;
  symbol?: string;
  title: string;
  subtitle: string;
  kind: "page" | "heading" | "symbol" | "variable";
  score: number;
};

function normalize(value: string): string {
  return value.trim().toLocaleLowerCase("ja-JP");
}

function searchableText(topic: DocumentGuideTopic): string {
  return [
    topic.title,
    topic.nav_label,
    topic.summary,
    ...topic.keywords,
    extractMarkdownText(topic.markdown),
  ]
    .join(" ")
    .toLocaleLowerCase("ja-JP");
}

function matchScore(value: string, query: string, exact: number, contains: number): number {
  const normalized = normalize(value);
  if (normalized === query) return exact;
  return normalized.includes(query) ? contains : 0;
}

export function searchTopics(
  topics: readonly DocumentGuideTopic[],
  query: string,
): DocumentGuideTopic[] {
  const normalized = normalize(query);
  if (!normalized) return [...topics];

  return topics
    .map((topic, index) => {
      const haystack = searchableText(topic);
      const title = normalize(topic.title);
      const nav = normalize(topic.nav_label);
      const score =
        (title.includes(normalized) ? 4 : 0) +
        (nav.includes(normalized) ? 3 : 0) +
        (topic.keywords.some((keyword) => normalize(keyword).includes(normalized)) ? 2 : 0) +
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
    ...(result.heading ? { heading: result.heading } : {}),
    ...(result.symbol ? { symbol: result.symbol } : {}),
    title: result.title,
    subtitle: result.subtitle,
    kind: result.kind,
    score: result.score,
  };
}

export function searchGuide(
  topics: readonly DocumentGuideTopic[],
  query: string,
): GuideSearchResult[] {
  const normalized = normalize(query);
  if (!normalized) {
    return topics.map((topic) => ({
      topicId: topic.id,
      title: topic.title,
      subtitle: topic.summary,
      kind: "page",
      score: 0,
    }));
  }

  const results: RankedResult[] = [];
  topics.forEach((topic, topicIndex) => {
    const titleScore = Math.max(
      matchScore(topic.title, normalized, 1000, 950),
      matchScore(topic.nav_label, normalized, 900, 875),
    );
    const keywordScore = topic.keywords.some((keyword) => normalize(keyword).includes(normalized))
      ? 350
      : 0;
    const bodyScore = searchableText(topic).includes(normalized) ? 300 : 0;
    const pageScore = Math.max(titleScore, keywordScore, bodyScore);
    if (pageScore > 0) {
      results.push({
        topicId: topic.id,
        title: topic.title,
        subtitle: topic.summary,
        kind: "page",
        score: pageScore,
        topicIndex,
        itemIndex: -1,
      });
    }

    topic.headings.forEach((heading, headingIndex) => {
      const score = matchScore(heading.text, normalized, 880, 840);
      if (score <= 0) return;
      results.push({
        topicId: topic.id,
        heading: heading.slug,
        title: heading.text,
        subtitle: `${topic.nav_label} · 見出し`,
        kind: "heading",
        score,
        topicIndex,
        itemIndex: headingIndex,
      });
    });

    topic.code_references.forEach((reference, referenceIndex) => {
      const symbol = getCodeSymbol(reference.symbol);
      const path = symbol?.path;
      const codeScore = Math.max(
        matchScore(reference.label_ja, normalized, 850, 825),
        matchScore(reference.symbol, normalized, 800, 750),
        path ? matchScore(path, normalized, 700, 650) : 0,
        normalize(reference.description_ja).includes(normalized) ? 400 : 0,
      );
      if (codeScore > 0) {
        results.push({
          topicId: topic.id,
          symbol: reference.symbol,
          title: reference.label_ja,
          subtitle: [reference.symbol, path].filter(Boolean).join(" · "),
          kind: "symbol",
          score: codeScore,
          topicIndex,
          itemIndex: 1000 + referenceIndex * 1000,
        });
      }

      reference.variables.forEach((variable, variableIndex) => {
        const variableScore = Math.max(
          matchScore(variable.label_ja, normalized, 810, 790),
          matchScore(variable.name, normalized, 800, 780),
          normalize(variable.description_ja).includes(normalized) ? 390 : 0,
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
          itemIndex: 1000 + referenceIndex * 1000 + variableIndex + 1,
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
