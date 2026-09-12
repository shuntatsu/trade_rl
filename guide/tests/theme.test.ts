import { describe, expect, it } from "vitest";

import {
  readStoredTheme,
  resolveTheme,
  THEME_STORAGE_KEY,
  writeStoredTheme,
} from "../src/app/useTheme";


describe("guide theme", () => {
  it("resolves explicit theme before system preference", () => {
    expect(resolveTheme("dark", false)).toBe("dark");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
  });

  it("reads only supported stored values", () => {
    localStorage.setItem(THEME_STORAGE_KEY, "dark");
    expect(readStoredTheme(localStorage)).toBe("dark");
    localStorage.setItem(THEME_STORAGE_KEY, "invalid");
    expect(readStoredTheme(localStorage)).toBe("system");
  });

  it("survives storage write failures", () => {
    const brokenStorage = {
      setItem() {
        throw new Error("blocked");
      },
    } as Pick<Storage, "setItem">;
    expect(() => writeStoredTheme(brokenStorage, "dark")).not.toThrow();
  });
});
