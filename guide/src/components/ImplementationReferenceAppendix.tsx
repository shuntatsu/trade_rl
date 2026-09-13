import { useEffect, useRef } from "react";

import type {
  CodeReference,
  DocumentGuideTopic,
} from "../content/documentSchema";
import { CodeInspector } from "./CodeInspector";

function ImplementationReference({
  reference,
  topic,
  initiallyOpen,
}: {
  reference: CodeReference;
  topic: DocumentGuideTopic;
  initiallyOpen: boolean;
}) {
  const detailsRef = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    if (detailsRef.current) detailsRef.current.open = initiallyOpen;
  }, [initiallyOpen]);

  return (
    <details
      ref={detailsRef}
      className="implementation-reference"
      data-symbol={reference.symbol}
    >
      <summary>実装詳細: {reference.label_ja}</summary>
      <CodeInspector referenceId={reference.id} topic={topic} />
    </details>
  );
}

export function ImplementationReferenceAppendix({
  topic,
  openSymbol,
}: {
  topic: DocumentGuideTopic;
  openSymbol?: string;
}) {
  if (topic.role === "overview" || topic.role === "status" || !topic.code_references.length) {
    return null;
  }

  return (
    <section className="implementation-appendix" aria-labelledby="implementation-appendix-title">
      <div className="section-heading implementation-appendix__heading">
        <div>
          <p className="eyeline">Reference</p>
          <h2 id="implementation-appendix-title">実装を確認する</h2>
        </div>
        <p>本文を読んだ後、必要な実装だけを開けます。</p>
      </div>
      <div className="implementation-appendix__items">
        {topic.code_references.map((reference) => (
          <ImplementationReference
            key={`${reference.id}:${openSymbol ?? ""}`}
            reference={reference}
            topic={topic}
            initiallyOpen={reference.symbol === openSymbol}
          />
        ))}
      </div>
    </section>
  );
}
