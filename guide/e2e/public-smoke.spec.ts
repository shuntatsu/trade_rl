import { expect, test } from "@playwright/test";


test("published Guide serves the code-linked replay journey", async ({ page }) => {
  const sourceRevision = process.env.GUIDE_SOURCE_REV;
  expect(sourceRevision).toMatch(/^[0-9a-f]{40}$/);

  const response = await page.goto("./#implementation-replay?step=risk-constrain");
  expect(response?.ok()).toBeTruthy();
  await expect(page.getByRole("heading", { level: 1, name: "1本のバーを追う" })).toBeVisible();
  await expect(
    page.getByRole("heading", { level: 3, name: "ハードリスクを適用" }),
  ).toBeVisible();

  const source = page.getByRole("link", { name: "実装をGitHubで開く" });
  await expect(source).toHaveAttribute(
    "href",
    new RegExp(
      `github\\.com/shuntatsu/trade_rl/blob/${sourceRevision}/trade_rl/risk/pretrade\\.py#L\\d+-L\\d+`,
    ),
  );

  await page.keyboard.press("Control+K");
  const search = page.getByRole("textbox", { name: "ガイドを検索" });
  await search.fill("desired_quantity");
  await page.getByRole("button", { name: /希望保有数量/ }).click();
  await expect(page).toHaveURL(
    /#implementation-replay\?symbol=trade_rl\.evaluation\.replay\.run_single_symbol_replay$/,
  );

  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
});


test("published Guide has no 320px horizontal overflow", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "public-mobile-320", "mobile project only");
  await page.goto("./#implementation-replay?step=risk-constrain");
  const sizes = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
});
