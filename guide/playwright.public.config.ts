import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.GUIDE_PUBLIC_URL;
if (!baseURL) {
  throw new Error("GUIDE_PUBLIC_URL is required for public Guide smoke tests");
}

export default defineConfig({
  testDir: "./e2e",
  testMatch: "public-smoke.spec.ts",
  fullyParallel: true,
  forbidOnly: true,
  retries: 1,
  reporter: "list",
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "public-desktop",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "public-mobile-320",
      use: { ...devices["Desktop Chrome"], viewport: { width: 320, height: 800 } },
    },
  ],
});
