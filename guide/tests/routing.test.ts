import { describe, expect, it } from "vitest";

import { formatHashRoute, normalizeHashRoute } from "../src/app/useHashRoute";


describe("guide hash routing", () => {
  const ids = [
    "overview",
    "data-flow",
    "architecture",
    "implementation-replay",
    "implementation-ppo",
    "code-map",
  ] as const;

  it("maps a known legacy replay step to its document heading", () => {
    expect(
      normalizeHashRoute(
        "#implementation-replay?step=risk-constrain",
        ids,
        "overview",
      ),
    ).toEqual({
      topicId: "implementation-replay",
      heading: "4-hard-riskを適用する",
    });
  });

  it("prefers canonical heading over symbol when both are present", () => {
    expect(
      normalizeHashRoute(
        "#implementation-replay?heading=4-hard-risk%E3%82%92%E9%81%A9%E7%94%A8%E3%81%99%E3%82%8B&symbol=trade_rl.risk.pretrade.PreTradeRisk.constrain",
        ids,
        "overview",
      ),
    ).toEqual({
      topicId: "implementation-replay",
      heading: "4-hard-riskを適用する",
    });
  });

  it("drops an unknown legacy step instead of preserving visual state", () => {
    expect(
      normalizeHashRoute(
        "#implementation-replay?step=unknown-visual-step",
        ids,
        "overview",
      ),
    ).toEqual({ topicId: "implementation-replay" });
  });

  it("keeps legacy topic-only hashes compatible", () => {
    expect(normalizeHashRoute("#data-flow", ids, "overview")).toEqual({
      topicId: "data-flow",
    });
  });

  it("falls back to home for missing, unknown, or malformed topic hashes", () => {
    expect(normalizeHashRoute("", ids, "overview")).toEqual({ topicId: "overview" });
    expect(normalizeHashRoute("#missing?heading=x", ids, "overview")).toEqual({
      topicId: "overview",
    });
    expect(normalizeHashRoute("#%E0%A4%A", ids, "overview")).toEqual({
      topicId: "overview",
    });
  });

  it("formats heading and symbol selections deterministically", () => {
    expect(
      formatHashRoute({
        topicId: "implementation-replay",
        heading: "4-hard-riskを適用する",
      }),
    ).toBe(
      "#implementation-replay?heading=4-hard-risk%E3%82%92%E9%81%A9%E7%94%A8%E3%81%99%E3%82%8B",
    );
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
