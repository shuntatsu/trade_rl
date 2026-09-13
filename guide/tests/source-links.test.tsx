import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SourceLinks } from "../src/components/SourceLinks";
import { codeSymbolIndex } from "../src/content/codeSymbols";

describe("SourceLinks", () => {
  it("binds authoritative document links to the exact Guide source revision", () => {
    render(
      <SourceLinks
        sources={[
          {
            path: "docs/architecture/lean-core.md",
            heading: "Core flow",
            sha256: "0".repeat(64),
          },
        ]}
      />,
    );

    expect(screen.getByRole("link", { name: /architecture\/lean-core\.md/ })).toHaveAttribute(
      "href",
      `https://github.com/shuntatsu/trade_rl/blob/${codeSymbolIndex.source_revision}/docs/architecture/lean-core.md`,
    );
  });
});
