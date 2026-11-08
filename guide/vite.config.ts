import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              // Keep the eager Guide documents available while retaining the
              // per-chunk 500 KB budget as execution-economics topics grow.
              name: "react-runtime",
              test: /node_modules[\/](react|react-dom|scheduler)[\/]/,
            },
          ],
        },
      },
    },
  },
});
