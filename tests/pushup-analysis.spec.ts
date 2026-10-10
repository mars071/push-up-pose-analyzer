import { test, expect } from '@playwright/test';
import { segmentReps } from '../src/tasks/rep-segmentation';
import { analyzePushup } from '../src/tasks/pushup-analysis';
import type { SegmentFrame } from '../src/tasks/rep-segmentation';

// Shoulder height tracks the elbow angle (high at the top of a rep, low at the
// bottom), matching the signal the segmentation expects.
function shoulderYFor(angle: number): number {
  return 0.55 - (angle - 95) * (0.15 / 80);
}

// Builds a single frame whose left-side landmarks encode an exact elbow angle
// and a straight shoulder-hip-ankle line. Only the left side is populated; the
// midpoint helper falls back to a single landmark when the right is absent.
function frameAt(time: number, elbowAngleDeg: number): SegmentFrame {
  const theta = (elbowAngleDeg * Math.PI) / 180;
  const sy = shoulderYFor(elbowAngleDeg);
  const landmarks = [
    { index: 11, x: 0.5, y: sy, visibility: 1 }, // left shoulder
    { index: 13, x: 0.5, y: sy + 0.2, visibility: 1 }, // left elbow
    { index: 15, x: 0.5 - 0.2 * Math.sin(theta), y: sy + 0.2 - 0.2 * Math.cos(theta), visibility: 1 }, // left wrist
    { index: 23, x: 0.5, y: sy + 0.3, visibility: 1 }, // left hip
    { index: 27, x: 0.5, y: sy + 0.5, visibility: 1 }, // left ankle
  ];
  return { frame: Math.round(time / 0.125), time, landmarks };
}

// Sawtooth elbow angle used by the segmentation tests: phase 0 = top (175°),
// phase 4 = bottom (95°), linear in between.
function sawAngle(phase: number): number {
  if (phase === 0) return 175;
  if (phase === 4) return 95;
  if (phase < 4) return 175 - phase * 20;
  return 95 + (phase - 4) * 20;
}

function sawtoothFrames(n: number): SegmentFrame[] {
  return Array.from({ length: n }, (_, i) => frameAt(i * 0.125, sawAngle(i % 8)));
}

test('scores a near-perfect synthetic push-up', () => {
  const frames = sawtoothFrames(25);
  const segmentation = segmentReps(frames);
  const report = analyzePushup(frames, segmentation);

  expect(segmentation.numReps).toBe(3);
  expect(report.metrics.map((m) => m.id)).toEqual(['depth', 'extension', 'alignment', 'tempo']);
  expect(report.perRep.length).toBe(3);
  expect(report.frames.length).toBe(25);

  const depth = report.metrics.find((m) => m.id === 'depth')!;
  const extension = report.metrics.find((m) => m.id === 'extension')!;
  const alignment = report.metrics.find((m) => m.id === 'alignment')!;

  // Bottom angle 95° -> (160 - 95) / 60 * 100 = 108.3 -> clamped to 100
  expect(depth.score).toBe(100);
  // Top angle 175° -> 100 - (180 - 175) * 3 = 85
  expect(Math.abs(extension.score - 85)).toBeLessThan(0.5);
  // All joints collinear -> 0° deviation -> 100
  expect(alignment.score).toBe(100);

  expect(report.totalScore).toBeGreaterThanOrEqual(0);
  expect(report.totalScore).toBeLessThanOrEqual(100);
  expect(report.totalScore).toBeGreaterThanOrEqual(80);
});

test('penalizes a shallow push-up (bottom never reaches 90°)', () => {
  // Top 180°, bottom 140° — a half rep.
  const angle = [180, 180, 160, 150, 140, 140, 150, 160, 180, 180];
  const frames = angle.map((a, i) => frameAt(i * 0.125, a));
  const segmentation = segmentReps(frames);
  const report = analyzePushup(frames, segmentation);

  const depth = report.metrics.find((m) => m.id === 'depth')!;
  // (160 - 140) / 60 * 100 = 33.3
  expect(Math.abs(depth.score - 33.3)).toBeLessThan(0.5);
  expect(depth.grade).toBe('Poor');
});

test('penalizes a hip that bends off the straight line', () => {
  const frames = sawtoothFrames(25).map((f) => ({
    ...f,
    landmarks: f.landmarks.map((lm) =>
      lm.index === 23 ? { ...lm, x: lm.x + 0.04 } : lm
    ),
  }));
  const segmentation = segmentReps(frames);
  const report = analyzePushup(frames, segmentation);

  const alignment = report.metrics.find((m) => m.id === 'alignment')!;
  expect(alignment.score).toBeLessThan(100);
  expect(alignment.score).toBeGreaterThan(0);
  expect(report.perRep[0].hipDeviation).toBeGreaterThan(0);
});

test('reports zero total score when no reps are detected', () => {
  const frames = Array.from({ length: 10 }, (_, i) => frameAt(i * 0.125, 170));
  const segmentation = segmentReps(frames);
  const report = analyzePushup(frames, segmentation);

  expect(segmentation.numReps).toBe(0);
  expect(report.totalScore).toBe(0);
  expect(report.grade).toBe('Poor');
  expect(report.advice.length).toBeGreaterThan(0);
});

test('debug mode exposes calculation steps for every metric', () => {
  const frames = sawtoothFrames(25);
  const segmentation = segmentReps(frames);
  const report = analyzePushup(frames, segmentation, true);

  expect(report.debug).toBe(true);
  for (const m of report.metrics) {
    expect(m.debug.length).toBeGreaterThan(0);
    expect(m.debug[0].label).toBe('Rule');
  }
});
