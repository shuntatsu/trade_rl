import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { DocumentGuideTopic } from "../content/documentSchema";
import { slugMarkdownHeading } from "../content/markdown";

function nodeText(value: ReactNode): string {
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (Array.isArray(value)) return value.map(nodeText).join("");
  if (value && typeof value === "object" && "props" in value) {
    const props = (value as { props?: { children?: ReactNode } }).props;
    return nodeText(props?.children);
  }
  return "";
}

function safeUrl(url: string): string {
  const trimmed = url.trim();
  if (!trimmed) return "";
  if (
    trimmed.startsWith("#") ||
    trimmed.startsWith("/") ||
    trimmed.startsWith("./") ||
    trimmed.startsWith("../")
  ) {
    return trimmed;
  }
  const scheme = trimmed.match(/^([a-z][a-z0-9+.-]*):/i)?.[1]?.toLowerCase();
  if (!scheme) return trimmed;
  return scheme === "http" || scheme === "https" ? trimmed : "";
}

function headingId(children: ReactNode): string | undefined {
  const value = slugMarkdownHeading(nodeText(children));
  return value || undefined;
}

export function MarkdownArticle({ topic }: { topic: DocumentGuideTopic }) {
  return (
    <article className="markdown-article" data-page-role={topic.role}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        urlTransform={safeUrl}
        components={{
          h1: ({ children }) => <h1 id={headingId(children)}>{children}</h1>,
          h2: ({ children }) => <h2 id={headingId(children)}>{children}</h2>,
          h3: ({ children }) => <h3 id={headingId(children)}>{children}</h3>,
          h4: ({ children }) => <h4 id={headingId(children)}>{children}</h4>,
          h5: ({ children }) => <h5 id={headingId(children)}>{children}</h5>,
          h6: ({ children }) => <h6 id={headingId(children)}>{children}</h6>,
          a: ({ href, children }) => {
            const safe = safeUrl(href ?? "");
            return safe ? <a href={safe}>{children}</a> : <span>{children}</span>;
          },
          pre: ({ children }) => (
            <pre
              className="markdown-article__code-block"
              tabIndex={0}
              aria-label="横スクロール可能なコードブロック"
            >
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div
              className="markdown-article__table-scroll"
              tabIndex={0}
              aria-label="横スクロール可能な表"
            >
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {topic.markdown}
      </ReactMarkdown>
    </article>
  );
}
