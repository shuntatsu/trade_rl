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

async function tabUntilText(page: Page, text: string): Promise<void> {
  for (let index = 0; index < 100; index += 1) {
    await page.keyboard.press("Tab");
    const activeText = await page.evaluate(() => document.activeElement?.textContent ?? "");
    if (activeText.includes(text)) return;
  }
  throw new Error(`keyboard focus did not reach: ${text}`);
}

test("overview, replay, and PPO have no serious or critical accessibility violations", async ({
  page,
}) => {
  for (const route of ["overview", "implementation-replay", "implementation-ppo"]) {
    for (const theme of ["light", "dark"] as const) {
      await page.goto(`/#${route}`);
      await page.evaluate((value) => localStorage.setItem("trade-rl-guide-theme", value), theme);
      await page.reload();
      if (theme === "dark") await expect(page.locator("html")).toHaveClass(/dark/);
      await expectNoBlockingA11yViolations(page);
    }
  }
});

test("keyboard alone can open implementation details and reach the exact source link", async ({
  page,
}) => {
  await page.goto("/#implementation-replay");

  await tabUntilText(page, "実装詳細: ハードリスクを適用");
  await page.keyboard.press("Enter");
  const details = page.locator(
    'details[data-symbol="trade_rl.risk.pretrade.PreTradeRisk.constrain"]',
  );
  await expect(details).toHaveAttribute("open", "");

  await tabUntilText(page, "実装をGitHubで開く");
  const href = await page.evaluate(() => (document.activeElement as HTMLAnchorElement).href);
  expect(href).toMatch(/\/blob\/[0-9a-f]{40}\/trade_rl\/risk\/pretrade\.py#L/);
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

test("reduced motion keeps all document information available", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#data-flow");
  await expect(
    page.getByRole("heading", { level: 1, name: "データと特徴量の流れ" }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "全体の流れ" })).toBeVisible();
  await expect(page.getByText(/再取得しても照合できる状態/)).toBeVisible();
  await expect(page.getByRole("button", { name: "次のステップ" })).toHaveCount(0);
});
