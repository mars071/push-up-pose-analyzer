import { test, expect } from '@playwright/test';
import { segmentReps } from '../src/tasks/rep-segmentation';
import { analyzePushup } from '../src/tasks/pushup-analysis';
import type { SegmentFrame, SegmentationResult } from '../src/tasks/rep-segmentation';

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

// A single rep whose top frames put the shoulder and wrist at explicit x
// positions. With `shoulderX = 0.5, wristX = 0.5416` the upper arm is vertical
// (tilt 0°) and the ~168° elbow is a genuine bend; with `shoulderX = 0.5416,
// wristX = 0.5` the shoulder sits forward of the forearm (tilt ~12°) so ~168°
// is its natural full extension. Both read the same top elbow angle.
function topFrame(time: number, shoulderX: number, wristX: number): SegmentFrame {
  const sy = 0.4;
  return {
    frame: Math.round(time / 0.125),
    time,
    landmarks: [
      { index: 11, x: shoulderX, y: sy, visibility: 1 },
      { index: 13, x: 0.5, y: sy + 0.2, visibility: 1 },
      { index: 15, x: wristX, y: sy + 0.4, visibility: 1 },
      { index: 23, x: 0.5, y: sy + 0.3, visibility: 1 },
      { index: 27, x: 0.5, y: sy + 0.5, visibility: 1 },
    ],
  };
}

function repFrames(shoulderTopX: number, wristTopX: number): SegmentFrame[] {
  return [
    topFrame(0.125, shoulderTopX, wristTopX),
    frameAt(0.25, 95), // bottom
    topFrame(0.375, shoulderTopX, wristTopX),
  ];
}

// `analyzePushup` only needs the rep boundaries and phase labels from the
// segmentation; hand-crafting it avoids the rep-segmentation smoothing, which
// needs more frames than this minimal top/bottom/top sequence provides.
function singleRep(frames: SegmentFrame[]): SegmentationResult {
  const phases = ['start', 'middle', 'end'] as const;
  return {
    reps: [
      {
        rep: 1,
        startFrame: 0,
        middleFrame: 1,
        endFrame: 2,
        startTime: frames[0].time,
        middleTime: frames[1].time,
        endTime: frames[2].time,
        frames: [0, 1, 2],
      },
    ],
    frames: frames.map((f, i) => ({ rep: 1, phase: phases[i], elbowAngle: null, shoulderY: null })),
    numReps: 1,
    leadIn: null,
    leadOut: null,
  };
}

// A rep whose wrist stays planted at (0.5, 0.85) while the shoulder drops. The
// elbow is placed off the shoulder-wrist line so it reads a shallow (~121°)
// angle even as the shoulder falls — modelling the real case where the elbow
// landmark is unreliable but the shoulder height is not.
function plantedFrame(time: number, shoulderY: number, elbowX: number, elbowY: number): SegmentFrame {
  return {
    frame: Math.round(time / 0.125),
    time,
    landmarks: [
      { index: 11, x: 0.5, y: shoulderY, visibility: 1 },
      { index: 13, x: elbowX, y: elbowY, visibility: 1 },
      { index: 15, x: 0.5, y: 0.85, visibility: 1 },
      { index: 23, x: 0.5, y: shoulderY + 0.15, visibility: 1 },
      { index: 27, x: 0.5, y: shoulderY + 0.35, visibility: 1 },
    ],
  };
}

test('depth normalizes shoulder drop by upper-arm length and rescues a shallow angle', () => {
  // Top: arm straight (elbow 180°, upper-arm length 0.25, shoulder height
  // 0.50). Bottom: elbow reads ~121° (shallow) but the shoulder drops to 0.25 —
  // a drop equal to the full upper-arm length, which physically means ~90°.
  const cheating = [
    plantedFrame(0.125, 0.35, 0.5, 0.6),
    plantedFrame(0.25, 0.6, 0.43, 0.72),
    plantedFrame(0.375, 0.35, 0.5, 0.6),
  ];
  const cheatingReport = analyzePushup(cheating, singleRep(cheating));
  const cheatingDepth = cheatingReport.metrics.find((m) => m.id === 'depth')!;
  // The shallow angle alone is (160 - 121) / 60 * 100 ≈ 65, but a full
  // upper-arm-length drop (drop / arm = 1.0) rescues it to a full score.
  expect(cheatingDepth.score).toBeGreaterThan(90);

  // Same shallow elbow, but the shoulder never drops: only the angle counts.
  const staticRep = [
    plantedFrame(0.125, 0.6, 0.43, 0.72),
    plantedFrame(0.25, 0.6, 0.43, 0.72),
    plantedFrame(0.375, 0.6, 0.43, 0.72),
  ];
  const staticReport = analyzePushup(staticRep, singleRep(staticRep));
  const staticDepth = staticReport.metrics.find((m) => m.id === 'depth')!;
  expect(staticDepth.score).toBeLessThan(70);
});

test('relaxes extension when the shoulder sits forward of the forearm', () => {
  const straight = analyzePushup(repFrames(0.5, 0.5416), singleRep(repFrames(0.5, 0.5416)));
  const forward = analyzePushup(repFrames(0.5416, 0.5), singleRep(repFrames(0.5416, 0.5)));

  const straightExt = straight.metrics.find((m) => m.id === 'extension')!;
  const forwardExt = forward.metrics.find((m) => m.id === 'extension')!;

  // Same top elbow angle (~168°), but the forward shoulder makes it natural.
  expect(forwardExt.score).toBeGreaterThan(90);
  expect(straightExt.score).toBeLessThan(75);
  expect(forwardExt.score).toBeGreaterThan(straightExt.score + 20);
});
