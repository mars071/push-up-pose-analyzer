import { test, expect } from '@playwright/test';
import { segmentSignals } from '../src/tasks/rep-segmentation';

test('segments a synthetic push-up sawtooth into reps', () => {
  const n = 25;
  const times = Array.from({ length: n }, (_, i) => i * 0.125);
  const angle: number[] = [];
  const shoulder: number[] = [];
  for (let i = 0; i < n; i++) {
    const phase = i % 8;
    let a: number;
    if (phase === 0) a = 175;
    else if (phase === 4) a = 95;
    else if (phase < 4) a = 175 - phase * 20;
    else a = 95 + (phase - 4) * 20;
    angle.push(a);
    shoulder.push(0.55 - (a - 95) * (0.15 / 80));
  }

  const seg = segmentSignals(angle, shoulder, times);

  expect(seg.numReps).toBe(3);
  expect(seg.reps.map((r) => [r.startFrame, r.middleFrame, r.endFrame])).toEqual([
    [0, 4, 7],
    [8, 12, 15],
    [16, 20, 24],
  ]);
  expect(seg.frames[0].rep).toBe(1);
  expect(seg.frames[0].phase).toBe('start');
  expect(seg.frames[4].phase).toBe('middle');
  expect(seg.frames[7].phase).toBe('end');
  expect(seg.frames[8].rep).toBe(2);
  expect(seg.frames[8].phase).toBe('start');
  expect(seg.frames[24].phase).toBe('end');
  expect(seg.leadIn).toBeNull();
  expect(seg.leadOut).toBeNull();
});

test('reports zero reps on a flat signal', () => {
  const n = 10;
  const times = Array.from({ length: n }, (_, i) => i * 0.125);
  const angle = new Array(n).fill(170);
  const shoulder = new Array(n).fill(0.45);

  const seg = segmentSignals(angle, shoulder, times);

  expect(seg.numReps).toBe(0);
  expect(seg.reps).toEqual([]);
  expect(seg.frames.every((f) => f.rep === null)).toBe(true);
});

function sawtoothAngle(phase: number): number {
  if (phase === 0) return 175;
  if (phase === 4) return 95;
  if (phase < 4) return 175 - phase * 20;
  return 95 + (phase - 4) * 20;
}

function sawtoothSignals(
  n: number,
  shoulderLag = 0,
  phaseShift = 0
): { angle: number[]; shoulder: number[] } {
  const angle = Array.from({ length: n }, (_, i) => sawtoothAngle((i + phaseShift) % 8));
  const shoulder: number[] = [];
  for (let i = 0; i < n; i++) {
    let j = i - shoulderLag;
    if (i === 0) j = 0;
    else if (i === n - 1) j = n - 1;
    j = Math.max(0, Math.min(n - 1, j));
    shoulder.push(0.55 - (angle[j] - 95) * (0.15 / 80));
  }
  return { angle, shoulder };
}

test('segments reps when the shoulder lags the elbow by a frame', () => {
  const n = 25;
  const times = Array.from({ length: n }, (_, i) => i * 0.125);
  const { angle, shoulder } = sawtoothSignals(n, 1);

  const seg = segmentSignals(angle, shoulder, times);

  expect(seg.numReps).toBe(3);
  expect(seg.reps.map((r) => [r.startFrame, r.middleFrame, r.endFrame])).toEqual([
    [0, 4, 7],
    [8, 12, 15],
    [16, 20, 24],
  ]);
});

test('brackets reps with lead-in and lead-out segments', () => {
  const n = 25;
  const times = Array.from({ length: n }, (_, i) => i * 0.125);
  const { angle, shoulder } = sawtoothSignals(n, 0, 5);

  const seg = segmentSignals(angle, shoulder, times);

  expect(seg.numReps).toBe(2);
  expect(seg.reps.map((r) => [r.startFrame, r.middleFrame, r.endFrame])).toEqual([
    [3, 7, 10],
    [11, 15, 19],
  ]);
  expect(seg.leadIn).not.toBeNull();
  expect(seg.leadIn!.startFrame).toBe(0);
  expect(seg.leadIn!.endFrame).toBe(2);
  expect(seg.leadOut).not.toBeNull();
  expect(seg.leadOut!.startFrame).toBe(20);
  expect(seg.leadOut!.endFrame).toBe(24);
  expect(seg.frames[2].phase).toBe('lead-in');
  expect(seg.frames[3].phase).toBe('start');
  expect(seg.frames[19].phase).toBe('end');
  expect(seg.frames[20].phase).toBe('lead-out');
});
