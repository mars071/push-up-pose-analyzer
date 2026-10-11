/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { test, expect } from '@playwright/test';

test.describe('Mobile Layout & Navigation', () => {
  // Mobile viewport (iPhone SE 2020)
  test.use({ viewport: { width: 375, height: 667 } });

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should hide sidebar and show mobile header', async ({ page }) => {
    // Check sidebar hidden
    const sidebar = page.locator('.sidebar');
    const display = await sidebar.evaluate((el) => window.getComputedStyle(el).display);
    console.log(`Sidebar display: ${display}`);
    await expect(sidebar).toBeHidden();

    // Check mobile header visible
    const mobileHeader = page.locator('.mobile-header');
    await expect(mobileHeader).toBeVisible();
  });

  test('should show only the push-up analyzer in the dropdown', async ({ page }) => {
    const select = page.locator('#mobile-task-select');
    await expect(select).toBeVisible();
    await expect(select.locator('option')).toHaveCount(1);
    await expect(select).toHaveValue('#/vision/swimming_analyzer');

    // Selecting the only option keeps us on the analyzer.
    await select.selectOption('#/vision/swimming_analyzer');
    await expect(page).toHaveURL(/swimming_analyzer/);
  });

  test('should open the nav drawer and close it via the backdrop', async ({ page }) => {
    const sidebar = page.locator('.sidebar');
    const backdrop = page.locator('.sidebar-backdrop');

    await page.click('.mobile-header .menu-toggle');
    await expect(sidebar).toBeVisible();
    await expect(backdrop).toBeVisible();
    await expect(sidebar).toHaveCSS('position', 'fixed');

    // Tap outside the drawer (backdrop) to dismiss.
    await backdrop.click({ position: { x: 370, y: 400 } });
    await expect(sidebar).toBeHidden();
    await expect(backdrop).toBeHidden();

    // Navigating via a drawer link also closes it.
    await page.click('.mobile-header .menu-toggle');
    await page.click('.sidebar a[href="#/vision/swimming_analyzer"]');
    await expect(page).toHaveURL(/swimming_analyzer/);
    await expect(sidebar).toBeHidden();
  });
});
