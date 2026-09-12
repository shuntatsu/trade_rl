import { expect, test } from "@playwright/test";


test("core guide interactions work in a real browser", async ({ page }) => {
  await page.goto("/#overview");
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "実データで動く、検証可能なトレーディングRLシステム",
    }),
  ).toBeVisible();

  await page.getByRole("button", { name: /^Execution/ }).click();
  await expect(page.getByRole("heading", { level: 3, name: "Execution" })).toBeVisible();

  await page.keyboard.press("Control+K");
  const search = page.getByRole("textbox", { name: "ガイドを検索" });
  await expect(search).toBeFocused();
  await search.fill("手数料");
  await page.getByRole("button", { name: /実行コスト/ }).click();
  await expect(page).toHaveURL(/#execution-economics$/);
  await expect(page.getByRole("heading", { level: 1, name: "実行コスト" })).toBeVisible();
});


test("theme preference persists", async ({ page }) => {
  await page.goto("/#overview");
  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await expect(page.getByRole("button", { name: "ライトモードへ" })).toBeVisible();
});


test("page never creates horizontal document overflow", async ({ page }) => {
  await page.goto("/#ppo-observation-v2");
  const sizes = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
});


test("mobile navigation remains tap-accessible", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-320", "mobile project only");
  await page.goto("/#overview");
  await page.getByRole("button", { name: /全体像/ }).click();
  await page.getByRole("link", { name: "PPO観測 v2" }).click();
  await expect(page).toHaveURL(/#ppo-observation-v2$/);
  await expect(page.getByRole("heading", { level: 1, name: "PPO Observation v2" })).toBeVisible();
});
