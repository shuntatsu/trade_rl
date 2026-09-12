import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "../src/app/App";


describe("interactive guide workflow", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.location.hash = "#overview";
    document.documentElement.classList.remove("dark");
  });

  it("keeps diagram detail selectable by click", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "実データで動く、検証可能なトレーディングRLシステム",
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^Execution/ }));
    expect(
      screen.getByRole("heading", { level: 3, name: "Execution" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/MarketExecutor \+ BookState/)).toBeInTheDocument();
  });

  it("opens search from Ctrl+K and navigates using Japanese terminology", async () => {
    const user = userEvent.setup();
    render(<App />);

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("textbox", { name: "ガイドを検索" });
    await user.type(input, "手数料");
    await user.click(screen.getByRole("button", { name: /実行コスト/ }));

    expect(window.location.hash).toBe("#execution-economics");
    expect(
      await screen.findByRole("heading", { level: 1, name: "実行コスト" }),
    ).toBeInTheDocument();
  });

  it("switches dark mode without changing content", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "ダークモードへ" }));
    expect(document.documentElement).toHaveClass("dark");
    expect(window.localStorage.getItem("trade-rl-guide-theme")).toBe("dark");
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
  });
});
