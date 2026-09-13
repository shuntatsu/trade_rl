import { expect, test } from "@playwright/test";


test("overview reaches replay risk implementation and exact source", async (
  { page },
  testInfo,
) => {
  await page.goto("/#overview");
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "実データで動く、検証可能なトレーディングRLシステム",
    }),
  ).toBeVisible();

  await page.getByRole("button", { name: /1本のバーを追う/ }).click();
  await expect(page).toHaveURL(/#implementation-replay$/);
  await expect(page.getByRole("heading", { level: 1, name: "1本のバーを追う" })).toBeVisible();

  if (testInfo.project.name === "mobile-320") {
    const next = page.getByRole("button", { name: "次へ" });
    await next.click();
    await next.click();
    await next.click();
  } else {
    await page.getByRole("button", { name: /ハードリスクで目標を制約/ }).click();
  }

  await expect(page).toHaveURL(/#implementation-replay\?step=risk-constrain$/);
  await expect(
    page.getByRole("heading", { level: 3, name: "ハードリスクを適用" }),
  ).toBeVisible();
  await expect(
    page.getByText("trade_rl.risk.pretrade.PreTradeRisk.constrain"),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "実装をGitHubで開く" })).toHaveAttribute(
    "href",
    /github\.com\/shuntatsu\/trade_rl\/blob\/[0-9a-f]{40}\/trade_rl\/risk\/pretrade\.py#L\d+-L\d+/,
  );
});


test("Japanese alias and Python identifier find the same replay implementation", async ({
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
    await expect(page.getByText("trade_rl.evaluation.replay.run_single_symbol_replay")).toBeVisible();
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


test("page never creates horizontal document overflow", async ({ page }) => {
  await page.goto("/#implementation-replay?step=risk-constrain");
  const sizes = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
});


test("mobile sequence can reach hard risk without pan or zoom", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-320", "mobile project only");
  await page.goto("/#implementation-replay");
  const next = page.getByRole("button", { name: "次へ" });
  await next.click();
  await next.click();
  await next.click();
  await expect(page).toHaveURL(/#implementation-replay\?step=risk-constrain$/);
  await expect(
    page.getByRole("heading", { level: 3, name: "ハードリスクを適用" }),
  ).toBeVisible();

  const sizes = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
});
