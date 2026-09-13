export type MarkdownHeading = {
  depth: number;
  text: string;
  slug: string;
};

const ATX_HEADING = /^(#{1,6})\s+(.+?)\s*#*\s*$/;
const FENCE = /^\s*(```+|~~~+)/;

function cleanInlineMarkdown(value: string): string {
  return value
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[`*_~]/g, "")
    .trim();
}

export function slugMarkdownHeading(value: string): string {
  const cleaned = cleanInlineMarkdown(value).normalize("NFKC").toLowerCase();
  return cleaned
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
}

export function extractMarkdownHeadings(markdown: string): MarkdownHeading[] {
  const headings: MarkdownHeading[] = [];
  const seen = new Set<string>();
  let fenceMarker: string | undefined;

  for (const line of markdown.split(/\r?\n/)) {
    const fence = line.match(FENCE)?.[1];
    if (fence) {
      const marker = fence[0];
      if (!fenceMarker) fenceMarker = marker;
      else if (marker === fenceMarker) fenceMarker = undefined;
      continue;
    }
    if (fenceMarker) continue;

    const match = line.match(ATX_HEADING);
    if (!match) continue;
    const text = cleanInlineMarkdown(match[2]);
    const slug = slugMarkdownHeading(text);
    if (!text || !slug) throw new Error("Markdown heading must have visible text");
    if (seen.has(slug)) throw new Error(`duplicate Markdown heading slug: ${slug}`);
    seen.add(slug);
    headings.push({ depth: match[1].length, text, slug });
  }
  return headings;
}

export function extractMarkdownText(markdown: string): string {
  return markdown
    .replace(/!\[([^\]]*)\]\(([^)]*)\)/g, "$1 $2")
    .replace(/\[([^\]]+)\]\(([^)]*)\)/g, "$1 $2")
    .replace(/^\s*#{1,6}\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+[.)]\s+/gm, "")
    .replace(/[`*_~|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
