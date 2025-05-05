import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.pw.ts",
  workers: 1,
  timeout: 180_000,
  use: {
    baseURL: "http://127.0.0.1:8794",
    actionTimeout: 15000,
    trace: "retain-on-failure",
    video: "retain-on-failure",
    launchOptions: process.env.BROWSER_EXECUTABLE
      ? { executablePath: process.env.BROWSER_EXECUTABLE }
      : {},
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1000 } } },
    {
      name: "mobile",
      use: { viewport: { width: 390, height: 844 }, isMobile: true },
    },
  ],
  webServer: {
    command: "uv run python scripts/dev.py",
    cwd: "..",
    url: "http://127.0.0.1:8794/api/health",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
