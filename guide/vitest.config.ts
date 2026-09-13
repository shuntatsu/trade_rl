import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    pool: "vmThreads",
    setupFiles: ["./src/test/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
