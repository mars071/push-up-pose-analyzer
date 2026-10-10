import { test, expect } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Push Up Pose Analyzer', () => {
  let videoPath: string;

  test.beforeAll(() => {
    videoPath = path.resolve(__dirname, 'assets', 'swim_test.mp4');
  });

  test('uploads a video and produces skeleton-marked frames', async ({ page }) => {
    page.on('console', (msg) => {
      if (msg.type() === 'error') console.log(`[BROWSER ERROR] ${msg.text()}`);
    });

    await page.goto('#/vision/swimming_analyzer');
    await page.waitForSelector('h2:has-text("Push Up Pose Analyzer")');

    await page.setInputFiles('#video-upload', videoPath);

    await expect(page.locator('#progress')).toContainText('Done', { timeout: 120000 });

    // Quality report is rendered with one card per metric.
    await expect(page.locator('#report .report-metric')).toHaveCount(4);
    await expect(page.locator('#report .report-total')).toBeVisible();

    const frames = page.locator('#results .swim-frame');
    const count = await frames.count();
    expect(count).toBeGreaterThanOrEqual(15); // 2s video sampled every 0.125s -> 16 frames

    // Every frame canvas should carry green skeleton pixels (#00e676).
    const greenCount = await page.evaluate(() => {
      const canvases = document.querySelectorAll('#results .swim-frame canvas');
      let total = 0;
      canvases.forEach((c) => {
        const ctx = (c as HTMLCanvasElement).getContext('2d')!;
        const data = ctx.getImageData(0, 0, c.width, c.height).data;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i] === 0 && data[i + 1] >= 180 && data[i + 2] >= 80) total++;
        }
      });
      return total;
    });
    expect(greenCount).toBeGreaterThan(0);
  });

  test('exports per-frame landmark data as CSV and JSON', async ({ page }) => {
    await page.goto('#/vision/swimming_analyzer');
    await page.waitForSelector('h2:has-text("Push Up Pose Analyzer")');

    await page.setInputFiles('#video-upload', videoPath);
    await expect(page.locator('#progress')).toContainText('Done', { timeout: 120000 });

    const downloadPromise = page.waitForEvent('download');
    await page.click('#download-data');
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('landmark_data.zip');

    const buf = await fs.promises.readFile(await download.path());
    const JSZip = (await import('jszip')).default;
    const zip = await JSZip.loadAsync(buf);

    const csv = await zip.file('landmarks.csv')!.async('string');
    const lines = csv.trim().split('\n');
    expect(lines[0]).toBe('frame,time,pose_detected,landmark_index,landmark_name,x,y,z,visibility');
    expect(lines.length).toBe(1 + 16 * 33); // header + 16 frames x 33 landmarks
    expect(csv).toContain('left_shoulder');

    const json = JSON.parse(await zip.file('landmarks.json')!.async('string'));
    expect(json.frames.length).toBe(16);
    expect(json.frames[0].landmarks.length).toBe(33);
    expect(json.frames[0].landmarks[11].name).toBe('left_shoulder');

    // Segmentation metadata is present and self-consistent.
    expect(json).toHaveProperty('num_reps');
    expect(json).toHaveProperty('reps');
    expect(json).toHaveProperty('lead_in');
    expect(json).toHaveProperty('lead_out');
    expect(Array.isArray(json.reps)).toBe(true);
    expect(json.num_reps).toBe(json.reps.length);
    for (const rep of json.reps) {
      expect(rep.start_frame).toBeLessThanOrEqual(rep.middle_frame);
      expect(rep.middle_frame).toBeLessThanOrEqual(rep.end_frame);
    }

    // Quality analysis report is embedded with four scored metrics and advice.
    expect(json).toHaveProperty('analysis');
    expect(json.analysis).toHaveProperty('total_score');
    expect(json.analysis).toHaveProperty('grade');
    expect(json.analysis.metrics.length).toBe(4);
    expect(json.analysis.metrics.map((m: { id: string }) => m.id)).toEqual([
      'depth',
      'extension',
      'alignment',
      'tempo',
    ]);
    expect(json.analysis.advice.length).toBeGreaterThan(0);
    expect(json.analysis.frames.length).toBe(16);

    for (const frame of json.frames) {
      expect(frame).toHaveProperty('rep');
      expect(frame).toHaveProperty('phase');
      expect(frame).toHaveProperty('elbow_angle');
      expect(frame).toHaveProperty('shoulder_y');
    }
  });
});
