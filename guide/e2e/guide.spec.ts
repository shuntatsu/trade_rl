import { expect, test } from "@playwright/test";


test("overview, replay, and PPO are readable without selecting visualization steps", async ({
  page,
}) => {
  await page.goto("/#overview");
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "実データで動く、検証可能なトレーディングRLシステム",
    }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "何をするシステムか" })).toBeVisible();
  await expect(page.getByText(/MarketDatasetを構築・固定/)).toBeVisible();

  await page.goto("/#implementation-replay");
  for (const heading of [
    "1. 観測を作る",
    "2. 戦略がintentを返す",
    "3. 希望数量とproposalへ変換する",
    "4. hard riskを適用する",
    "5. 約定と会計を行う",
    "6. BookStateを引き継ぐ",
    "7. 結果を確定する",
  ]) {
    await expect(page.getByRole("heading", { level: 2, name: heading })).toBeAttached();
  }

  await page.goto("/#implementation-ppo");
  await expect(page.getByRole("heading", { name: "PPO Observation v2は5区分" })).toBeVisible();
  const table = page.getByRole("table").first();
  for (const segment of [
    "local_values",
    "local_available",
    "local_staleness",
    "current_intent",
    "current_weight",
  ]) {
    await expect(table.getByText(new RegExp(segment))).toBeVisible();
  }
  await expect(page.getByRole("heading", { name: "学習後も同じ観測契約を使う" })).toBeAttached();
});


test("Japanese alias and Python identifier open the same replay implementation", async ({
  page,
}) => {
  for (const query of ["希望保有数量", "desired_quantity"]) {
    await page.goto("/#overview");
    await page.keyboard.press("Control+K");
    const search = page.getByRole("textbox", { name: "ガイドを検索" });
    await expect(search).toBeFocused();
    await search.fill(query);
    await page.getByRole("button", { name: /希望保有数量/ }).click();
    await expect(page).toHaveURL(
      /#implementation-replay\?symbol=trade_rl\.evaluation\.replay\.run_single_symbol_replay$/,
    );

    const details = page.locator(
      'details[data-symbol="trade_rl.evaluation.replay.run_single_symbol_replay"]',
    );
    await expect(details).toHaveAttribute("open", "");
    await expect(
      details.getByText("trade_rl.evaluation.replay.run_single_symbol_replay"),
    ).toBeVisible();
    await expect(details.getByRole("link", { name: "実装をGitHubで開く" })).toHaveAttribute(
      "href",
      /github\.com\/shuntatsu\/trade_rl\/blob\/[0-9a-f]{40}\/trade_rl\/evaluation\/replay\.py#L\d+-L\d+/,
    );
  }
});


test("theme preference persists", async ({ page }) => {
  await page.goto("/#overview");
  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await expect(page.getByRole("button", { name: "ライトモードへ" })).toBeVisible();
});


test("document pages never create horizontal page overflow", async ({ page }) => {
  for (const route of ["overview", "implementation-replay", "implementation-ppo", "code-map"]) {
    await page.goto(`/#${route}`);
    const sizes = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
  }
});


test("mobile replay exposes hard risk without pan, zoom, or step controls", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-320", "mobile project only");
  await page.goto("/#implementation-replay");
  await expect(
    page.getByRole("heading", { level: 2, name: "4. hard riskを適用する" }),
  ).toBeAttached();
  await expect(page.getByRole("button", { name: "次へ" })).toHaveCount(0);

  const sizes = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
});
