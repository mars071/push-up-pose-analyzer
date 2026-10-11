# Push Up Pose Analyzer

A single-purpose web app that analyzes push-up form from a video using MediaPipe's
Pose Landmarker (BlazePose). Upload a video and it:

- Detects each repetition (down-up cycle) via elbow-angle and shoulder-height signals.
- Scores form across four weighted dimensions: **depth**, **extension**, **alignment**, and **tempo**.
- Produces a quality report with a total score, grade, and per-metric advice.
- Marks every sampled frame with the pose skeleton and exports per-frame landmark
  data (CSV + JSON) as a ZIP.

All inference runs on-device in the browser; no video ever leaves your machine.

## Quick Start

1. **Install dependencies**: `pnpm install`
2. **Run the app**: `pnpm dev`
3. **Open** the local URL printed by Vite (typically `http://localhost:5174/push-up-pose-analyzer/`)

## How it works

- `src/tasks/swimming-analyzer.ts` — UI, video sampling, and landmark extraction.
- `src/tasks/rep-segmentation.ts` — splits the pose stream into reps (lead-in, reps, lead-out).
- `src/tasks/pushup-analysis.ts` — rule-based scoring of depth, extension, alignment, and tempo.

The scoring rules are documented in the companion file `pushup-form-rules.md`.

## Testing

```bash
pnpm test
```

Playwright end-to-end and unit tests cover rep segmentation, form scoring, and the
upload/export flow.
