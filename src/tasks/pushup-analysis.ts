/**
 * Rule-based push-up quality analysis.
 *
 * Takes the per-frame pose landmarks and the rep segmentation, then scores the
 * push-up along four dimensions:
 *   1. Depth     – how far the elbows flex at the bottom (~90° is ideal)
 *   2. Extension – how fully the elbows lock out at the top (~180° is ideal)
 *   3. Alignment – how straight the shoulder-hip-ankle line stays (plank)
 *   4. Tempo     – how balanced (down vs up) and consistent the reps are
 *
 * Pure math module: no MediaPipe or DOM dependencies, so it is unit-testable.
 * Set `DEBUG` to include (or, in the UI, hide) the per-metric calculation steps.
 */

import type { SegmentFrame, SegmentationResult } from './rep-segmentation';

export const DEBUG = true;

const SHOULDER = [11, 12];
const ELBOW = [13, 14];
const WRIST = [15, 16];
const HIP = [23, 24];
const ANKLE = [27, 28];

// Metric weights; must sum to 1.
const WEIGHTS = {
  depth: 0.3,
  extension: 0.2,
  alignment: 0.3,
  tempo: 0.2,
} as const;

// Extension tolerance. With a straight upper arm (shoulder directly over the
// elbow, tilt ≈ 0°) the elbow must reach ~180° to count as locked out, so a
// shortfall really is a bent arm and we keep the strict target. In a standard
// push-up the shoulder can sit slightly forward of the forearm, which leans the
// upper arm by φ and makes a fully-extended arm read only 180° − φ; there we
// relax the target by the lean (only past a small noise floor, and capped so an
// extreme forward reach can't zero the metric out).
const LEAN_TOLERANCE_DEG = 3;
const MAX_TILT_DEG = 20;

// Depth shoulder-drop reference. The shoulder's vertical drop toward the
// planted hand, divided by the upper-arm length (shoulder→elbow distance at the
// top of the rep), is a direct proxy for the elbow angle: a drop equal to the
// upper-arm length means the elbow reached ~90°, and a larger drop means it went
// past 90°. This is the height signal used to cross-check the elbow angle.

export type Grade = 'Excellent' | 'Good' | 'Fair' | 'Poor';

export interface DebugStep {
  label: string;
  detail: string;
}

export interface MetricResult {
  id: string;
  name: string;
  weight: number;
  score: number;
  grade: Grade;
  summary: string;
  advice: string;
  debug: DebugStep[];
}

export interface RepMetric {
  rep: number;
  bottomElbowAngle: number | null;
  topElbowAngle: number | null;
  hipDeviation: number | null;
  upperArmTilt: number | null;
  shoulderDropRatio: number | null;
  downTime: number | null;
  upTime: number | null;
}

export interface FrameMetric {
  frame: number;
  time: number;
  rep: number | null;
  phase: string;
  elbowAngle: number | null;
  hipAngle: number | null;
  hipDeviation: number | null;
  upperArmTilt: number | null;
  upperArmLength: number | null;
  shoulderHeight: number | null;
}

export interface PushupReport {
  totalScore: number;
  grade: Grade;
  metrics: MetricResult[];
  perRep: RepMetric[];
  frames: FrameMetric[];
  advice: string[];
  debug: boolean;
}

interface Point {
  x: number;
  y: number;
}

function clamp(v: number, lo = 0, hi = 100): number {
  return Math.min(hi, Math.max(lo, v));
}

function gradeFor(score: number): Grade {
  if (score >= 85) return 'Excellent';
  if (score >= 70) return 'Good';
  if (score >= 55) return 'Fair';
  return 'Poor';
}

function midpoint(f: SegmentFrame, a: number, b: number): Point | null {
  const map = new Map<number, { x: number; y: number; visibility: number }>();
  for (const lm of f.landmarks) map.set(lm.index, lm);
  const l = map.get(a);
  const r = map.get(b);
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

// Angle at `vertex` between the two vectors (a -> vertex) and (c -> vertex).
function jointAngle(a: Point, vertex: Point, c: Point): number {
  const ux = a.x - vertex.x;
  const uy = a.y - vertex.y;
  const vx = c.x - vertex.x;
  const vy = c.y - vertex.y;
  const du = Math.hypot(ux, uy);
  const dv = Math.hypot(vx, vy);
  if (du === 0 || dv === 0) return NaN;
  const cos = Math.max(-1, Math.min(1, (ux * vx + uy * vy) / (du * dv)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function computeFrameMetrics(frames: SegmentFrame[]): FrameMetric[] {
  return frames.map((f) => {
    const shoulder = midpoint(f, SHOULDER[0], SHOULDER[1]);
    const elbow = midpoint(f, ELBOW[0], ELBOW[1]);
    const wrist = midpoint(f, WRIST[0], WRIST[1]);
    const hip = midpoint(f, HIP[0], HIP[1]);
    const ankle = midpoint(f, ANKLE[0], ANKLE[1]);

    let elbowAngle: number | null = null;
    if (shoulder && elbow && wrist) {
      const a = jointAngle(shoulder, elbow, wrist);
      elbowAngle = Number.isNaN(a) ? null : a;
    }

    let hipAngle: number | null = null;
    let hipDeviation: number | null = null;
    if (shoulder && hip && ankle) {
      const a = jointAngle(shoulder, hip, ankle);
      if (!Number.isNaN(a)) {
        hipAngle = a;
        hipDeviation = Math.abs(180 - a);
      }
    }

    // Tilt of the upper arm (shoulder → elbow) away from vertical. 0° = the
    // shoulder sits directly over the elbow; a forward shoulder leans it.
    let upperArmTilt: number | null = null;
    if (shoulder && elbow) {
      const dx = Math.abs(elbow.x - shoulder.x);
      const dy = Math.abs(elbow.y - shoulder.y);
      upperArmTilt = (Math.atan2(dx, dy) * 180) / Math.PI;
    }

    // Length of the upper arm (shoulder→elbow). It is a rigid segment, so it is
    // read where it is most reliable — the top of the rep, when the arm is
    // extended and roughly in-plane with the camera.
    let upperArmLength: number | null = null;
    if (shoulder && elbow) {
      upperArmLength = Math.hypot(elbow.x - shoulder.x, elbow.y - shoulder.y);
    }

    // Height of the shoulder above the planted hand (wrist), taken as the
    // vertical distance wrist→shoulder. This shrinks as the body lowers.
    let shoulderHeight: number | null = null;
    if (shoulder && wrist) {
      shoulderHeight = wrist.y - shoulder.y;
    }

    return {
      frame: f.frame,
      time: f.time,
      rep: null,
      phase: 'unassigned',
      elbowAngle,
      hipAngle,
      hipDeviation,
      upperArmTilt,
      upperArmLength,
      shoulderHeight,
    };
  });
}

function buildDepthMetric(perRep: RepMetric[]): MetricResult {
  const debug: DebugStep[] = [
    {
      label: 'Rule',
      detail:
        "Depth = the deeper of two signals per rep: (1) elbow angle score = (160 − angle) / 60 × 100; (2) shoulder-drop score = drop / upper-arm length × 100, where drop is the shoulder's vertical fall toward the planted hand and upper-arm length is the shoulder→elbow distance at the top of the rep (drop = upper-arm length ⇒ elbow ≈ 90°). A shallow elbow angle is rescued when the shoulder actually dropped, and vice versa.",
    },
  ];
  const scored = perRep.filter((r) => r.bottomElbowAngle != null);
  if (scored.length === 0) {
    debug.push({ label: 'Result', detail: 'No bottom frame with a detected pose → score 0.' });
    return {
      id: 'depth',
      name: 'Depth',
      weight: WEIGHTS.depth,
      score: 0,
      grade: 'Poor',
      summary: 'No reps measured',
      advice: 'No pose was detected at the bottom of any rep.',
      debug,
    };
  }

  const repScores = scored.map((r) => {
    const angleScore = clamp(((160 - r.bottomElbowAngle!) / 60) * 100);
    let s = angleScore;
    if (r.shoulderDropRatio != null) {
      const heightScore = clamp(r.shoulderDropRatio * 100);
      s = Math.max(angleScore, heightScore);
      debug.push({
        label: `Rep ${r.rep}`,
        detail: `bottom angle ${r.bottomElbowAngle!.toFixed(1)}° → angle ${angleScore.toFixed(1)} · shoulder drop ${r.shoulderDropRatio.toFixed(2)}× arm → height ${heightScore.toFixed(1)} → depth ${s.toFixed(1)}`,
      });
    } else {
      debug.push({
        label: `Rep ${r.rep}`,
        detail: `bottom angle ${r.bottomElbowAngle!.toFixed(1)}° → angle ${angleScore.toFixed(1)} (no shoulder drop) → depth ${s.toFixed(1)}`,
      });
    }
    return s;
  });

  const score = mean(repScores)!;
  const avg = mean(scored.map((r) => r.bottomElbowAngle!))!;
  debug.push({
    label: 'Average',
    detail: `mean bottom angle ${avg.toFixed(1)}° → final score ${score.toFixed(1)}`,
  });

  const advice =
    score >= 85
      ? 'Great depth — your chest gets low enough for a full range of motion.'
      : score >= 70
        ? 'Good depth. Aim to lower until your elbows reach ~90° for a full rep.'
        : score >= 55
          ? "You're stopping a bit high. Lower your chest until your elbows reach ~90°. If that's hard, try incline or knee push-ups."
          : 'Very shallow. Build depth first — incline push-ups help you reach full range before full push-ups.';

  return {
    id: 'depth',
    name: 'Depth',
    weight: WEIGHTS.depth,
    score,
    grade: gradeFor(score),
    summary: `Average bottom elbow angle ${avg.toFixed(0)}° (target ≤ 100°)`,
    advice,
    debug,
  };
}

function buildExtensionMetric(perRep: RepMetric[]): MetricResult {
  const debug: DebugStep[] = [
    {
      label: 'Rule',
      detail:
        'Extension = largest elbow angle in each rep (the top lockout). A forward shoulder leans the upper arm by φ, so a fully-extended arm reads 180° − φ there; the target is 180° − φ when φ > 3° (capped at 20°), else 180°. Score = 100 − (target − angle) × 3, clamped to [0, 100].',
    },
  ];
  const scored = perRep.filter((r) => r.topElbowAngle != null);
  if (scored.length === 0) {
    debug.push({ label: 'Result', detail: 'No top frame with a detected pose → score 0.' });
    return {
      id: 'extension',
      name: 'Extension',
      weight: WEIGHTS.extension,
      score: 0,
      grade: 'Poor',
      summary: 'No reps measured',
      advice: 'No pose was detected at the top of any rep.',
      debug,
    };
  }

  const repScores = scored.map((r) => {
    const tilt = r.upperArmTilt ?? 0;
    const effectiveTilt = tilt <= LEAN_TOLERANCE_DEG ? 0 : Math.min(tilt, MAX_TILT_DEG);
    const target = 180 - effectiveTilt;
    const s = clamp(100 - (target - r.topElbowAngle!) * 3);
    debug.push({
      label: `Rep ${r.rep}`,
      detail: `top elbow angle ${r.topElbowAngle!.toFixed(1)}° · upper-arm tilt ${tilt.toFixed(1)}° → target ${target.toFixed(0)}° → score ${s.toFixed(1)}`,
    });
    return s;
  });

  const score = mean(repScores)!;
  const avg = mean(scored.map((r) => r.topElbowAngle!))!;
  debug.push({
    label: 'Average',
    detail: `mean top angle ${avg.toFixed(1)}° → final score ${score.toFixed(1)}`,
  });

  const advice =
    score >= 85
      ? 'Full lockout at the top — excellent.'
      : score >= 70
        ? 'Mostly good. Squeeze to full elbow extension at the top of every rep.'
        : score >= 55
          ? "You're not fully straightening your arms. Press all the way up until your elbows lock out."
          : 'Finish each rep by fully straightening your arms at the top.';

  return {
    id: 'extension',
    name: 'Extension',
    weight: WEIGHTS.extension,
    score,
    grade: gradeFor(score),
    summary: `Average top elbow angle ${avg.toFixed(0)}°`,
    advice,
    debug,
  };
}

function buildAlignmentMetric(perRep: RepMetric[]): MetricResult {
  const debug: DebugStep[] = [
    {
      label: 'Rule',
      detail:
        'Alignment = how far the hip bends off the shoulder-ankle line (0° = perfectly straight plank). Score = (30 − deviation) / 22 × 100, clamped to [0, 100] (≤ 8° → 100, ≥ 30° → 0).',
    },
  ];
  const scored = perRep.filter((r) => r.hipDeviation != null);
  if (scored.length === 0) {
    debug.push({ label: 'Result', detail: 'No frames with a detectable hip → score 0.' });
    return {
      id: 'alignment',
      name: 'Body alignment',
      weight: WEIGHTS.alignment,
      score: 0,
      grade: 'Poor',
      summary: 'No reps measured',
      advice: 'No hip landmarks were detected, so alignment could not be measured.',
      debug,
    };
  }

  scored.forEach((r) => {
    debug.push({
      label: `Rep ${r.rep}`,
      detail: `mean hip deviation ${r.hipDeviation!.toFixed(1)}° from straight`,
    });
  });

  const avg = mean(scored.map((r) => r.hipDeviation!))!;
  const score = clamp(((30 - avg) / 22) * 100);
  debug.push({
    label: 'Average',
    detail: `mean deviation ${avg.toFixed(1)}° → score (30 − ${avg.toFixed(1)}) / 22 × 100 = ${score.toFixed(1)}`,
  });

  const advice =
    score >= 85
      ? 'Strong plank — your body stays in a straight line.'
      : score >= 70
        ? 'Good line. Keep your core and glutes engaged to avoid minor hip sag.'
        : score >= 55
          ? 'Your hips sag or pike. Brace your core and squeeze your glutes to keep a straight line.'
          : 'Your body breaks the straight line significantly. Tighten your core and glutes; add plank holds to build stability.';

  return {
    id: 'alignment',
    name: 'Body alignment',
    weight: WEIGHTS.alignment,
    score,
    grade: gradeFor(score),
    summary: `Average hip deviation ${avg.toFixed(1)}° from a straight plank`,
    advice,
    debug,
  };
}

function buildTempoMetric(perRep: RepMetric[]): MetricResult {
  const debug: DebugStep[] = [
    {
      label: 'Rule',
      detail:
        'Tempo = average of two sub-scores: (1) down/up balance = min(down, up) / max(down, up) × 100; (2) consistency = 100 − CV × 200, where CV = std / mean of rep durations.',
    },
  ];
  const withTimes = perRep.filter((r) => r.downTime != null && r.upTime != null && r.downTime > 0 && r.upTime > 0);
  if (withTimes.length === 0) {
    debug.push({ label: 'Result', detail: 'No complete rep timings → score 0.' });
    return {
      id: 'tempo',
      name: 'Tempo & consistency',
      weight: WEIGHTS.tempo,
      score: 0,
      grade: 'Poor',
      summary: 'No reps measured',
      advice: 'No rep timings could be measured.',
      debug,
    };
  }

  const balances = withTimes.map((r) => {
    const lo = Math.min(r.downTime!, r.upTime!);
    const hi = Math.max(r.downTime!, r.upTime!);
    const b = hi === 0 ? 0 : (lo / hi) * 100;
    debug.push({
      label: `Rep ${r.rep}`,
      detail: `down ${r.downTime!.toFixed(2)}s / up ${r.upTime!.toFixed(2)}s → balance ${b.toFixed(0)}`,
    });
    return b;
  });
  const balanceScore = mean(balances)!;

  const durations = withTimes.map((r) => r.downTime! + r.upTime!);
  const m = mean(durations)!;
  const variance = mean(durations.map((d) => (d - m) * (d - m)))!;
  const std = Math.sqrt(variance);
  const cv = m === 0 ? 1 : std / m;
  const consistencyScore = clamp(100 - cv * 200);
  debug.push({
    label: 'Consistency',
    detail: `rep durations mean ${m.toFixed(2)}s, std ${std.toFixed(2)}s → CV ${cv.toFixed(2)} → score ${consistencyScore.toFixed(1)}`,
  });

  const score = (balanceScore + consistencyScore) / 2;
  debug.push({
    label: 'Average',
    detail: `(balance ${balanceScore.toFixed(1)} + consistency ${consistencyScore.toFixed(1)}) / 2 = ${score.toFixed(1)}`,
  });

  const advice =
    score >= 85
      ? 'Controlled, consistent tempo.'
      : score >= 70
        ? 'Good control. Try to keep each rep the same speed.'
        : score >= 55
          ? 'Inconsistent pace — keep the lowering and pressing phases even and steady.'
          : 'Slow down and make each rep deliberate, with an even lowering and pressing speed.';

  return {
    id: 'tempo',
    name: 'Tempo & consistency',
    weight: WEIGHTS.tempo,
    score,
    grade: gradeFor(score),
    summary: `Balance ${balanceScore.toFixed(0)} · consistency ${consistencyScore.toFixed(0)}`,
    advice,
    debug,
  };
}

function buildRepMetrics(frames: FrameMetric[], segmentation: SegmentationResult): RepMetric[] {
  return segmentation.reps.map((rep) => {
    const devs: number[] = [];
    let bottomAngle = Infinity;
    let topAngle = -Infinity;
    let topArmTilt: number | null = null;
    let heightTop = -Infinity;
    let heightBottom = Infinity;
    let armLengthAtTop: number | null = null;
    for (let i = rep.startFrame; i <= rep.endFrame; i++) {
      const f = frames[i];
      const a = f?.elbowAngle;
      if (a != null) {
        if (a < bottomAngle) bottomAngle = a;
        if (a > topAngle) {
          topAngle = a;
          topArmTilt = f?.upperArmTilt ?? null;
        }
      }
      const d = f?.hipDeviation;
      if (d != null) devs.push(d);
      const h = f?.shoulderHeight;
      if (h != null) {
        if (h > heightTop) {
          heightTop = h;
          armLengthAtTop = f?.upperArmLength ?? null;
        }
        if (h < heightBottom) heightBottom = h;
      }
    }

    // Depth and extension are read as the lowest and highest elbow angle
    // anywhere in the rep. Reading a single start/middle frame is unreliable
    // because the segmentation's extrema are detected on the smoothed signal and
    // can sit ±1 frame off the raw minimum/maximum.
    // The shoulder drop is normalized by the upper-arm length measured at the
    // top of the rep: drop = upper-arm length ⇒ elbow ≈ 90°.
    const shoulderDropRatio =
      heightTop === -Infinity || heightBottom === Infinity || !armLengthAtTop || armLengthAtTop <= 0
        ? null
        : (heightTop - heightBottom) / armLengthAtTop;

    return {
      rep: rep.rep,
      bottomElbowAngle: bottomAngle === Infinity ? null : bottomAngle,
      topElbowAngle: topAngle === -Infinity ? null : topAngle,
      hipDeviation: mean(devs),
      upperArmTilt: topArmTilt,
      shoulderDropRatio,
      downTime: rep.middleTime - rep.startTime,
      upTime: rep.endTime - rep.middleTime,
    };
  });
}

export function analyzePushup(
  frames: SegmentFrame[],
  segmentation: SegmentationResult,
  debug: boolean = DEBUG
): PushupReport {
  const frameMetrics = computeFrameMetrics(frames);
  segmentation.frames.forEach((s, i) => {
    if (i < frameMetrics.length) {
      frameMetrics[i].rep = s.rep;
      frameMetrics[i].phase = s.phase;
    }
  });

  const perRep = buildRepMetrics(frameMetrics, segmentation);

  const metrics = [
    buildDepthMetric(perRep),
    buildExtensionMetric(perRep),
    buildAlignmentMetric(perRep),
    buildTempoMetric(perRep),
  ];

  const totalScore = Math.round(metrics.reduce((acc, m) => acc + m.score * m.weight, 0));
  const grade = gradeFor(totalScore);

  const advice: string[] = [];
  const weak = metrics.filter((m) => m.score < 70);
  if (weak.length === 0) {
    advice.push('Overall form is solid. Keep training with the same depth and control.');
  } else {
    for (const m of weak) {
      advice.push(`${m.name}: ${m.advice}`);
    }
  }

  return { totalScore, grade, metrics, perRep, frames: frameMetrics, advice, debug };
}
