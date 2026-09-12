import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

async function expectNoBlockingA11yViolations(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page }).analyze();
  const blocking = results.violations.filter(
    (violation) =>
      violation.impact === "serious" || violation.impact === "critical",
  );
  expect(blocking).toEqual([]);
}

test("has no serious or critical accessibility violations in light and dark", async ({
  page,
}) => {
  await page.goto("/#overview");
  await page.evaluate(() => localStorage.setItem("trade-rl-guide-theme", "light"));
  await page.reload();
  await expectNoBlockingA11yViolations(page);

  await page.evaluate(() => localStorage.setItem("trade-rl-guide-theme", "dark"));
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await expectNoBlockingA11yViolations(page);
});

test("keyboard focus can reach the primary interactions", async ({ page }) => {
  await page.goto("/#overview");
  await page.keyboard.press("Tab");
  const focusedTag = await page.evaluate(() => document.activeElement?.tagName);
  expect(["A", "BUTTON", "INPUT"]).toContain(focusedTag);

  await page.keyboard.press("Control+K");
  await expect(page.getByRole("textbox", { name: "ガイドを検索" })).toBeFocused();
});

test("search dialog traps focus and restores the trigger on close", async ({ page }) => {
  await page.goto("/#overview");
  const trigger = page.getByRole("button", { name: "ガイドを検索" });
  const textbox = page.getByRole("textbox", { name: "ガイドを検索" });

  await trigger.focus();
  await page.keyboard.press("Control+K");
  await expect(textbox).toBeFocused();

  await page.keyboard.press("Shift+Tab");
  await expect(page.locator(".search-result").last()).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(textbox).toBeFocused();

  await page.keyboard.press("Escape");
  await expect(trigger).toBeFocused();
});

test("reduced motion keeps the guide usable", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#data-flow");
  await expect(
    page.getByRole("heading", { level: 1, name: "データフロー" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "次のステップ" }).click();
  await expect(page.getByText("raw evidenceを固定")).toBeVisible();
});
