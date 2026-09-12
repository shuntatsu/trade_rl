import { describe, expect, it } from "vitest";

import { formatHashRoute, normalizeHashRoute } from "../src/app/useHashRoute";


describe("guide hash routing", () => {
  const ids = [
    "overview",
    "data-flow",
    "architecture",
    "implementation-replay",
    "code-map",
  ] as const;

  it("restores a known topic and step selection", () => {
    expect(
      normalizeHashRoute(
        "#implementation-replay?step=risk-constrain",
        ids,
        "overview",
      ),
    ).toEqual({ topicId: "implementation-replay", step: "risk-constrain" });
  });

  it("prefers step when step and symbol are both present", () => {
    expect(
      normalizeHashRoute(
        "#implementation-replay?step=risk-constrain&symbol=trade_rl.risk.pretrade.PreTradeRisk.constrain",
        ids,
        "overview",
      ),
    ).toEqual({ topicId: "implementation-replay", step: "risk-constrain" });
  });

  it("keeps legacy topic-only hashes compatible", () => {
    expect(normalizeHashRoute("#data-flow", ids, "overview")).toEqual({
      topicId: "data-flow",
    });
  });

  it("falls back to home for missing, unknown, or malformed topic hashes", () => {
    expect(normalizeHashRoute("", ids, "overview")).toEqual({ topicId: "overview" });
    expect(normalizeHashRoute("#missing?step=x", ids, "overview")).toEqual({
      topicId: "overview",
    });
    expect(normalizeHashRoute("#%E0%A4%A", ids, "overview")).toEqual({
      topicId: "overview",
    });
  });

  it("formats a symbol selection deterministically", () => {
    expect(
      formatHashRoute({
        topicId: "code-map",
        symbol: "trade_rl.evaluation.replay.run_single_symbol_replay",
      }),
    ).toBe(
      "#code-map?symbol=trade_rl.evaluation.replay.run_single_symbol_replay",
    );
  });
});
