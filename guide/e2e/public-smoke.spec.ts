import { expect, test } from "@playwright/test";


test("published Guide serves the critical human journey", async ({ page }) => {
  const response = await page.goto("./#overview");
  expect(response?.ok()).toBeTruthy();
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "実データで動く、検証可能なトレーディングRLシステム",
    }),
  ).toBeVisible();

  await page.keyboard.press("Control+K");
  const search = page.getByRole("textbox", { name: "ガイドを検索" });
  await search.fill("手数料");
  await page.getByRole("button", { name: /実行コスト/ }).click();
  await expect(page).toHaveURL(/#execution-economics$/);

  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);

  const source = page.getByRole("link", { name: /current-status\.md/ }).first();
  await expect(source).toHaveAttribute(
    "href",
    /github\.com\/shuntatsu\/trade_rl\/blob\/main\/docs\//,
  );
});


test("published Guide has no 320px horizontal overflow", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "public-mobile-320", "mobile project only");
  await page.goto("./#ppo-observation-v2");
  const sizes = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
});
