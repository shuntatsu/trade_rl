import type { GuideTopic } from "./schema";

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

export function searchTopics(topics: readonly GuideTopic[], query: string): GuideTopic[] {
  const normalized = query.trim().toLocaleLowerCase("ja-JP");
  if (!normalized) return [...topics];

  return topics
    .map((topic, index) => {
      const haystack = searchableText(topic);
      const title = topic.title.toLocaleLowerCase("ja-JP");
      const nav = topic.nav_label.toLocaleLowerCase("ja-JP");
      const score =
        (title.includes(normalized) ? 4 : 0) +
        (nav.includes(normalized) ? 3 : 0) +
        (topic.keywords.some((keyword) => keyword.toLocaleLowerCase("ja-JP").includes(normalized)) ? 2 : 0) +
        (haystack.includes(normalized) ? 1 : 0);
      return { topic, score, index };
    })
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map((item) => item.topic);
}
