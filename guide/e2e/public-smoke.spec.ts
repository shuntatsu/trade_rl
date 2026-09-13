import { expect, test } from "@playwright/test";


test("published Guide serves the Markdown-first replay journey", async ({ page }) => {
  const sourceRevision = process.env.GUIDE_SOURCE_REV;
  expect(sourceRevision).toMatch(/^[0-9a-f]{40}$/);

  const response = await page.goto("./#implementation-replay");
  expect(response?.ok()).toBeTruthy();
  await expect(page.getByRole("heading", { level: 1, name: "1本のバーを追う" })).toBeVisible();
  await expect(
    page.getByRole("heading", { level: 2, name: "4. hard riskを適用する" }),
  ).toBeAttached();
  await expect(page.getByRole("button", { name: "次へ" })).toHaveCount(0);

  await page.keyboard.press("Control+K");
  const search = page.getByRole("textbox", { name: "ガイドを検索" });
  await search.fill("desired_quantity");
  await page
    .getByRole("dialog", { name: "ガイド検索" })
    .getByRole("button", { name: /希望保有数量/ })
    .click();
  await expect(page).toHaveURL(
    /#implementation-replay\?symbol=trade_rl\.evaluation\.replay\.run_single_symbol_replay$/,
  );

  const details = page.locator(
    'details[data-symbol="trade_rl.evaluation.replay.run_single_symbol_replay"]',
  );
  await expect(details).toHaveAttribute("open", "");
  const source = details.getByRole("link", { name: "実装をGitHubで開く" });
  await expect(source).toHaveAttribute(
    "href",
    new RegExp(
      `github\\.com/shuntatsu/trade_rl/blob/${sourceRevision}/trade_rl/evaluation/replay\\.py#L\\d+-L\\d+`,
    ),
  );

  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
});


test("published Guide exposes PPO contract without interaction", async ({ page }) => {
  const response = await page.goto("./#implementation-ppo");
  expect(response?.ok()).toBeTruthy();
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
});


test("published Guide has no 320px horizontal document overflow", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "public-mobile-320", "mobile project only");
  for (const route of ["overview", "implementation-replay", "implementation-ppo"]) {
    await page.goto(`./#${route}`);
    const sizes = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
  }
});
