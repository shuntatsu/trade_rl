import { test } from "@playwright/test";


test("capture approved light and dark desktop states", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "desktop project only");

  await page.goto("/#overview");
  await page.evaluate(() => localStorage.setItem("trade-rl-guide-theme", "light"));
  await page.reload();
  await page.screenshot({ path: testInfo.outputPath("light-desktop.png"), fullPage: true });

  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await page.screenshot({ path: testInfo.outputPath("dark-desktop.png"), fullPage: true });
});


test("capture 320px mobile state", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-320", "mobile project only");
  await page.goto("/#overview");
  await page.screenshot({ path: testInfo.outputPath("light-mobile.png"), fullPage: true });
});
