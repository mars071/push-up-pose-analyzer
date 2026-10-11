import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 120000,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  workers: 1,
  reporter: 'html',
  snapshotPathTemplate: '{testDir}/__snapshots__/{arg}{ext}',
  use: {
    baseURL: 'http://localhost:5174/push-up-pose-analyzer/',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      grepInvert: /.*@gpu/, // Ignore GPU tests in default project
      use: {
        ...devices['Desktop Chrome'],
        headless: true,
        launchOptions: {
          args: [
            '--headless=new',
            '--use-gl=angle',
            '--use-angle=swiftshader',
            '--ignore-gpu-blocklist',
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream'
          ],
        },
      },
    },
    {
      name: 'chromium-gpu',
      // grep: /.*@gpu/, // Run ALL tests on GPU, not just @gpu tagged ones, per user request
      use: {
        ...devices['Desktop Chrome'],
        headless: true,
        launchOptions: {
          args: [
            '--headless=new',
            '--use-gl=angle',
            '--use-angle=swiftshader',
            '--ignore-gpu-blocklist',
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream'
          ],
        },
      },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5174/push-up-pose-analyzer/',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
