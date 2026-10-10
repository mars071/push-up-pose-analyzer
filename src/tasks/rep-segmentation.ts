/**
 * Rule-based push-up rep segmentation from MediaPipe pose landmarks.
 *
 * A rep is delimited using exactly two signals:
 *   1. Shoulder height -> high point = start/end, low point = middle.
 *   2. Elbow angle (shoulder-elbow-wrist) -> local max (~180) = start/end,
 *      local min (the smallest value) = middle.
 *
 * Pure math module: no MediaPipe or DOM dependencies, so it is unit-testable.
 */

const SHOULDER = [11, 12];
const ELBOW = [13, 14];
const WRIST = [15, 16];

// A "top" elbow peak must at least approach extension to count as a start/end
// (filters noise); bottoms use "smallest" with no absolute threshold.
const TOP_ANGLE_MIN = 120;
const SMOOTH_HALF_WINDOW = 2;

// The elbow and shoulder signals can peak a few frames apart (the shoulder keeps
// moving as the elbow locks out), so a shoulder high/low confirms an elbow
// extremum when it lands within this many frames of it.
const SHOULDER_ALIGN_WINDOW = 2;

export interface SegmentLandmark {
  index: number;
  x: number;
  y: number;
  visibility: number;
}

export interface SegmentFrame {
  frame: number;
  time: number;
  landmarks: SegmentLandmark[];
}

export type FramePhase =
  | 'lead-in'
  | 'start'
  | 'down'
  | 'middle'
  | 'up'
  | 'end'
  | 'lead-out'
  | 'unassigned';

export interface RepSegment {
  rep: number;
  startFrame: number;
  middleFrame: number;
  endFrame: number;
  startTime: number;
  middleTime: number;
  endTime: number;
  frames: number[];
}

export interface FrameSegmentation {
  rep: number | null;
  phase: FramePhase;
  elbowAngle: number | null;
  shoulderY: number | null;
}

export interface BoundarySegment {
  startFrame: number;
  endFrame: number;
  startTime: number;
  endTime: number;
  frames: number[];
}

export interface SegmentationResult {
  reps: RepSegment[];
  frames: FrameSegmentation[];
  numReps: number;
  leadIn: BoundarySegment | null;
  leadOut: BoundarySegment | null;
}

interface Point {
  x: number;
  y: number;
}

function averagedJoint(f: SegmentFrame, leftIdx: number, rightIdx: number): Point | null {
  const map = new Map<number, SegmentLandmark>();
  for (const lm of f.landmarks) map.set(lm.index, lm);
  const l = map.get(leftIdx);
  const r = map.get(rightIdx);
  if (!l && !r) return null;
  const lv = l ? l.visibility : 0;
  const rv = r ? r.visibility : 0;
  const total = lv + rv;
  if (total <= 0) return null;
  return {
    x: ((l ? l.x * lv : 0) + (r ? r.x * rv : 0)) / total,
    y: ((l ? l.y * lv : 0) + (r ? r.y * rv : 0)) / total,
  };
}

function angleAt(shoulder: Point, elbow: Point, wrist: Point): number {
  const ux = shoulder.x - elbow.x;
  const uy = shoulder.y - elbow.y;
  const wx = wrist.x - elbow.x;
  const wy = wrist.y - elbow.y;
  const du = Math.hypot(ux, uy);
  const dw = Math.hypot(wx, wy);
  if (du === 0 || dw === 0) return NaN;
  const cos = Math.max(-1, Math.min(1, (ux * wx + uy * wy) / (du * dw)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function computeSignals(frames: SegmentFrame[]): { elbowAngle: number[]; shoulderY: number[] } {
  const elbowAngle = frames.map((f) => {
    const s = averagedJoint(f, SHOULDER[0], SHOULDER[1]);
    const e = averagedJoint(f, ELBOW[0], ELBOW[1]);
    const w = averagedJoint(f, WRIST[0], WRIST[1]);
    if (!s || !e || !w) return NaN;
    return angleAt(s, e, w);
  });
  const shoulderY = frames.map((f) => {
    const s = averagedJoint(f, SHOULDER[0], SHOULDER[1]);
    return s ? s.y : NaN;
  });
  return { elbowAngle, shoulderY };
}

function smooth(values: number[], halfWindow: number): number[] {
  const out = new Array<number>(values.length).fill(NaN);
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    let count = 0;
    const lo = Math.max(0, i - halfWindow);
    const hi = Math.min(values.length - 1, i + halfWindow);
    for (let j = lo; j <= hi; j++) {
      const v = values[j];
      if (!Number.isNaN(v)) {
        sum += v;
        count++;
      }
    }
    if (count > 0) out[i] = sum / count;
  }
  return out;
}

function dedupePlateau(idx: number[]): number[] {
  const out: number[] = [];
  let run: number[] = [];
  const flush = () => {
    if (run.length === 0) return;
    out.push(run[Math.floor(run.length / 2)]);
    run = [];
  };
  for (const i of idx) {
    if (run.length > 0 && i !== run[run.length - 1] + 1) flush();
    run.push(i);
  }
  flush();
  return out;
}

function localExtrema(values: number[], halfWindow: number): { maxima: number[]; minima: number[] } {
  const rawMax: number[] = [];
  const rawMin: number[] = [];
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const b = values[i];
    if (Number.isNaN(b)) continue;
    let geAll = true;
    let leAll = true;
    const lo = Math.max(0, i - halfWindow);
    const hi = Math.min(n - 1, i + halfWindow);
    for (let j = lo; j <= hi; j++) {
      if (j === i) continue;
      const v = values[j];
      if (Number.isNaN(v)) continue;
      if (v > b) geAll = false;
      if (v < b) leAll = false;
    }
    if (geAll) rawMax.push(i);
    if (leAll) rawMin.push(i);
  }
  return { maxima: dedupePlateau(rawMax), minima: dedupePlateau(rawMin) };
}

function frameRange(start: number, end: number): number[] {
  const out: number[] = [];
  for (let f = start; f <= end; f++) out.push(f);
  return out;
}

export function segmentSignals(
  elbowAngle: number[],
  shoulderY: number[],
  times: number[]
): SegmentationResult {
  const n = elbowAngle.length;
  const smoothAngle = smooth(elbowAngle, SMOOTH_HALF_WINDOW);
  const smoothShoulder = smooth(shoulderY, SMOOTH_HALF_WINDOW);

  const { maxima, minima } = localExtrema(smoothAngle, 1);

  // Elbow is the authoritative rep delimiter: extended at top, flexed at bottom.
  const angleTops = maxima.filter((t) => smoothAngle[t] >= TOP_ANGLE_MIN);
  const angleBottoms = minima; // "smallest", no absolute threshold

  // Shoulder height confirms each extremum: a local y-minimum is a high point
  // (top), a local y-maximum is a low point (bottom). Confirm against the
  // nearest shoulder extremum within a small window, since the two signals can
  // peak a few frames apart.
  const { maxima: shoulderLows, minima: shoulderHighs } = localExtrema(smoothShoulder, 1);
  const nearExtremum = (frame: number, candidates: number[]): boolean =>
    candidates.some((c) => Math.abs(c - frame) <= SHOULDER_ALIGN_WINDOW);

  let tops = angleTops.filter((t) => nearExtremum(t, shoulderHighs));
  let bottoms = angleBottoms.filter((b) => nearExtremum(b, shoulderLows));
  // If the shoulder is too noisy to confirm anything, fall back to elbow-only.
  if (tops.length < 2) tops = angleTops;
  if (bottoms.length < 1) bottoms = angleBottoms;

  // Merge into one alternating top/bottom sequence.
  const extrema = [
    ...tops.map((f) => ({ frame: f, kind: 'top' as const })),
    ...bottoms.map((f) => ({ frame: f, kind: 'bottom' as const })),
  ].sort((a, b) => a.frame - b.frame);

  const clean: { frame: number; kind: 'top' | 'bottom' }[] = [];
  for (const e of extrema) {
    const last = clean[clean.length - 1];
    if (last && last.kind === e.kind) {
      if (e.kind === 'top' && smoothAngle[e.frame] > smoothAngle[last.frame]) {
        clean[clean.length - 1] = e;
      } else if (e.kind === 'bottom' && smoothAngle[e.frame] < smoothAngle[last.frame]) {
        clean[clean.length - 1] = e;
      }
    } else {
      clean.push(e);
    }
  }
  if (clean.length > 0 && clean[0].kind === 'bottom') clean.shift();
  if (clean.length > 0 && clean[clean.length - 1].kind === 'bottom') clean.pop();

  // Extrema were detected on the smoothed signal, which shifts each peak/trough
  // by up to SMOOTH_HALF_WINDOW frames from the raw signal. Re-anchor each to the
  // true raw minimum/maximum within that window so the reported frames (and any
  // angle read from them) reflect the actual pose.
  for (const e of clean) {
    const lo = Math.max(0, e.frame - SMOOTH_HALF_WINDOW);
    const hi = Math.min(n - 1, e.frame + SMOOTH_HALF_WINDOW);
    let best = e.frame;
    for (let f = lo; f <= hi; f++) {
      const v = elbowAngle[f];
      if (Number.isNaN(v)) continue;
      const b = elbowAngle[best];
      if (Number.isNaN(b) || (e.kind === 'top' ? v > b : v < b)) best = f;
    }
    e.frame = best;
  }
  // Re-anchoring can collapse a top and its neighbouring bottom onto the same
  // frame in a noisy signal; keep the sequence strictly increasing.
  for (let i = 1; i < clean.length; i++) {
    if (clean[i].frame <= clean[i - 1].frame) clean[i].frame = clean[i - 1].frame + 1;
  }

  // A rep is (top, bottom, top): start at the first top, middle at the bottom,
  // end immediately before the next rep's start (the shared top). The final rep
  // has no successor, so it ends at the last top (its true end position).
  const reps: RepSegment[] = [];
  for (let k = 0; k + 2 < clean.length; k += 2) {
    const startF = clean[k].frame;
    const middleF = clean[k + 1].frame;
    const endF = k + 3 < clean.length ? clean[k + 2].frame - 1 : clean[k + 2].frame;
    if (endF < startF) continue;
    reps.push({
      rep: reps.length + 1,
      startFrame: startF,
      middleFrame: middleF,
      endFrame: endF,
      startTime: times[startF],
      middleTime: times[middleF],
      endTime: times[endF],
      frames: frameRange(startF, endF),
    });
  }

  const firstTop = clean.length > 0 ? clean[0].frame : -1;
  const lastTop = clean.length > 0 ? clean[clean.length - 1].frame : -1;

  const leadIn: BoundarySegment | null =
    firstTop > 0
      ? {
          startFrame: 0,
          endFrame: firstTop - 1,
          startTime: times[0],
          endTime: times[firstTop - 1],
          frames: frameRange(0, firstTop - 1),
        }
      : null;

  const leadOut: BoundarySegment | null =
    lastTop >= 0 && lastTop + 1 < n
      ? {
          startFrame: lastTop + 1,
          endFrame: n - 1,
          startTime: times[lastTop + 1],
          endTime: times[n - 1],
          frames: frameRange(lastTop + 1, n - 1),
        }
      : null;

  const frames: FrameSegmentation[] = [];
  for (let i = 0; i < n; i++) {
    const seg: FrameSegmentation = {
      rep: null,
      phase: 'unassigned',
      elbowAngle: Number.isNaN(elbowAngle[i]) ? null : elbowAngle[i],
      shoulderY: Number.isNaN(shoulderY[i]) ? null : shoulderY[i],
    };
    if (Number.isNaN(elbowAngle[i])) {
      frames.push(seg);
      continue;
    }
    if (firstTop >= 0 && i < firstTop) {
      seg.phase = 'lead-in';
      frames.push(seg);
      continue;
    }
    if (lastTop >= 0 && i > lastTop) {
      seg.phase = 'lead-out';
      frames.push(seg);
      continue;
    }
    let assigned = false;
    for (const rep of reps) {
      if (i < rep.startFrame || i > rep.endFrame) continue;
      seg.rep = rep.rep;
      if (i === rep.startFrame) seg.phase = 'start';
      else if (i === rep.middleFrame) seg.phase = 'middle';
      else if (i === rep.endFrame) seg.phase = 'end';
      else if (i < rep.middleFrame) seg.phase = 'down';
      else seg.phase = 'up';
      assigned = true;
      break;
    }
    if (!assigned) seg.phase = 'unassigned';
    frames.push(seg);
  }

  return { reps, frames, numReps: reps.length, leadIn, leadOut };
}

export function segmentReps(frames: SegmentFrame[]): SegmentationResult {
  const { elbowAngle, shoulderY } = computeSignals(frames);
  return segmentSignals(elbowAngle, shoulderY, frames.map((f) => f.time));
}
