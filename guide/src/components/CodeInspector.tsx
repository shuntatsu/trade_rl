import { ExternalLink } from "lucide-react";

import { codeSymbolIndex, getCodeSymbol } from "../content/codeSymbols";
import type { GuideTopic } from "../content/schema";
import { buildCodeSourceUrl } from "../content/sourceLinks";

const REPOSITORY_URL = "https://github.com/shuntatsu/trade_rl";

function testUrl(path: string): string {
  return `${REPOSITORY_URL}/blob/${codeSymbolIndex.source_revision}/${path}`;
}

export function CodeInspector({
  referenceId,
  topic,
}: {
  referenceId?: string;
  topic: GuideTopic;
}) {
  const reference = topic.code_references.find((item) => item.id === referenceId);
  if (!reference) {
    return (
      <aside className="code-inspector code-inspector--empty" aria-label="実装詳細">
        <p>シーケンスの処理を選ぶと、対応する実装をここで確認できます。</p>
      </aside>
    );
  }

  const symbol = getCodeSymbol(reference.symbol);
  if (!symbol) {
    return (
      <aside className="code-inspector code-inspector--empty" aria-label="実装詳細">
        <h3>{reference.label_ja}</h3>
        <p>{reference.description_ja}</p>
        <p>この実装参照は現在のsource indexで解決できません。</p>
      </aside>
    );
  }

  return (
    <aside className="code-inspector" aria-label="実装詳細">
      <p className="eyeline">Implementation</p>
      <h3>{reference.label_ja}</h3>
      <p className="code-inspector__description">{reference.description_ja}</p>

      <dl className="code-inspector__facts">
        <div>
          <dt>実装識別子</dt>
          <dd>
            <code>{reference.symbol}</code>
          </dd>
        </div>
        <div>
          <dt>定義</dt>
          <dd>
            <code>{symbol.signature}</code>
          </dd>
        </div>
        <div>
          <dt>場所</dt>
          <dd>
            <code>
              {symbol.path}:{symbol.start_line}-{symbol.end_line}
            </code>
          </dd>
        </div>
      </dl>

      {reference.variables.length ? (
        <section className="code-inspector__variables" aria-labelledby="variable-heading">
          <h4 id="variable-heading">主要な値</h4>
          <dl>
            {reference.variables.map((variable) => (
              <div key={variable.name}>
                <dt>
                  {variable.label_ja} <code>{variable.name}</code>
                </dt>
                <dd>{variable.description_ja}</dd>
              </div>
            ))}
          </dl>
        </section>
      ) : null}

      <div className="code-inspector__links">
        <a
          href={buildCodeSourceUrl(symbol, REPOSITORY_URL)}
          target="_blank"
          rel="noreferrer"
        >
          実装をGitHubで開く <ExternalLink size={14} aria-hidden="true" />
        </a>
        {reference.tests.map((testPath) => (
          <a
            key={testPath}
            href={testUrl(testPath)}
            target="_blank"
            rel="noreferrer"
            aria-label="関連テストを開く"
          >
            {testPath} <ExternalLink size={14} aria-hidden="true" />
          </a>
        ))}
      </div>
    </aside>
  );
}
