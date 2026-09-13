import type { DocumentGuideTopic } from "../content/documentSchema";
import { CodeInspector } from "./CodeInspector";

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
          <details
            className="implementation-reference"
            key={reference.id}
            defaultOpen={reference.symbol === openSymbol}
            data-symbol={reference.symbol}
          >
            <summary>実装詳細: {reference.label_ja}</summary>
            <CodeInspector referenceId={reference.id} topic={topic} />
          </details>
        ))}
      </div>
    </section>
  );
}
