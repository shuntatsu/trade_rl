import manifestRaw from "../../content/manifest.json";

import {
  parseDocumentGuideManifest,
  parseDocumentTopic,
  type DocumentGuideTopic,
} from "./documentSchema";

const metadataModules = import.meta.glob("../../content/meta/*.json", {
  eager: true,
  import: "default",
}) as Record<string, unknown>;

const markdownModules = import.meta.glob("../../content/pages/*.md", {
  eager: true,
  import: "default",
  query: "?raw",
}) as Record<string, string>;

function stem(path: string): string {
  const filename = path.split("/").pop();
  if (!filename) throw new Error(`invalid Guide content path: ${path}`);
  return filename.replace(/\.(json|md)$/, "");
}

export const guideManifest = parseDocumentGuideManifest(manifestRaw);

export function loadTopics(): DocumentGuideTopic[] {
  const metadataById = new Map<string, unknown>();
  for (const [path, raw] of Object.entries(metadataModules)) {
    const id = stem(path);
    if (metadataById.has(id)) throw new Error(`duplicate Guide metadata id: ${id}`);
    metadataById.set(id, raw);
  }

  const markdownById = new Map<string, string>();
  for (const [path, markdown] of Object.entries(markdownModules)) {
    const id = stem(path);
    if (markdownById.has(id)) throw new Error(`duplicate Guide Markdown id: ${id}`);
    markdownById.set(id, markdown);
  }

  const expected = guideManifest.groups.flatMap((group) => group.topics);
  const metadataIds = [...metadataById.keys()].sort();
  const markdownIds = [...markdownById.keys()].sort();
  const expectedIds = [...expected].sort();
  if (JSON.stringify(metadataIds) !== JSON.stringify(expectedIds)) {
    throw new Error(
      `manifest/meta mismatch: expected ${expectedIds.join(", ")}, got ${metadataIds.join(", ")}`,
    );
  }
  if (JSON.stringify(markdownIds) !== JSON.stringify(expectedIds)) {
    throw new Error(
      `manifest/Markdown mismatch: expected ${expectedIds.join(", ")}, got ${markdownIds.join(", ")}`,
    );
  }

  return expected.map((id) => {
    const metadata = metadataById.get(id);
    const markdown = markdownById.get(id);
    if (metadata === undefined || markdown === undefined) {
      throw new Error(`missing Guide page pair: ${id}`);
    }
    const topic = parseDocumentTopic(metadata, markdown);
    if (topic.id !== id) {
      throw new Error(`Guide metadata id/path mismatch: ${id} != ${topic.id}`);
    }
    return topic;
  });
}
