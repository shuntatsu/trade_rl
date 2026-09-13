import { expect, test } from "@playwright/test";


test("capture approved light and dark replay document states", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "desktop project only");

  await page.goto("/#implementation-replay");
  await page.evaluate(() => localStorage.setItem("trade-rl-guide-theme", "light"));
  await page.reload();
  await expect(
    page.getByRole("heading", { level: 2, name: "4. hard riskを適用する" }),
  ).toBeAttached();
  await page.screenshot({
    path: testInfo.outputPath("replay-document-light-desktop.png"),
    fullPage: true,
    animations: "disabled",
  });

  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await page.screenshot({
    path: testInfo.outputPath("replay-document-dark-desktop.png"),
    fullPage: true,
    animations: "disabled",
  });
});


test("capture 320px replay document state", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-320", "mobile project only");
  await page.goto("/#implementation-replay");
  await expect(page.getByRole("button", { name: "次へ" })).toHaveCount(0);
  await expect(
    page.getByRole("heading", { level: 2, name: "4. hard riskを適用する" }),
  ).toBeAttached();
  await page.screenshot({
    path: testInfo.outputPath("replay-document-light-mobile.png"),
    fullPage: true,
    animations: "disabled",
  });
});
