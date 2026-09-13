import { test } from "@playwright/test";


test("capture approved light and dark replay implementation states", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "desktop project only");

  await page.goto("/#implementation-replay?step=risk-constrain");
  await page.evaluate(() => localStorage.setItem("trade-rl-guide-theme", "light"));
  await page.reload();
  await page.screenshot({
    path: testInfo.outputPath("replay-light-desktop.png"),
    fullPage: true,
    animations: "disabled",
  });

  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await page.screenshot({
    path: testInfo.outputPath("replay-dark-desktop.png"),
    fullPage: true,
    animations: "disabled",
  });
});


test("capture 320px replay step-through state", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-320", "mobile project only");
  await page.goto("/#implementation-replay?step=risk-constrain");
  await page.screenshot({
    path: testInfo.outputPath("replay-light-mobile.png"),
    fullPage: true,
    animations: "disabled",
  });
});
