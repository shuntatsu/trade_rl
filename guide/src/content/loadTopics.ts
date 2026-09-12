import manifestRaw from "../../content/manifest.json";

import { parseGuideManifest, parseTopic, type GuideTopic } from "./schema";

const topicModules = import.meta.glob("../../content/topics/*.json", {
  eager: true,
  import: "default",
}) as Record<string, unknown>;

export const guideManifest = parseGuideManifest(manifestRaw);

export function loadTopics(): GuideTopic[] {
  const topics = new Map<string, GuideTopic>();
  for (const raw of Object.values(topicModules)) {
    const topic = parseTopic(raw);
    if (topics.has(topic.id)) {
      throw new Error(`duplicate topic id: ${topic.id}`);
    }
    topics.set(topic.id, topic);
  }

  const discovered = [...topics.keys()].sort();
  const expected = [...guideManifest.topics].sort();
  if (JSON.stringify(discovered) !== JSON.stringify(expected)) {
    throw new Error(
      `manifest/topic mismatch: expected ${expected.join(", ")}, got ${discovered.join(", ")}`,
    );
  }

  return guideManifest.topics.map((id) => {
    const topic = topics.get(id);
    if (!topic) throw new Error(`missing topic: ${id}`);
    return topic;
  });
}
