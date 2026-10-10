import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
// @ts-ignore
import template from '../templates/swimming-analyzer.html?raw';
// @ts-ignore
import { segmentReps, SegmentationResult } from './rep-segmentation';
import { analyzePushup } from './pushup-analysis';
import type { PushupReport } from './pushup-analysis';

const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const INTERVAL = 0.125;
const MAX_EDGE = 1280;

// Keep the landmarker alive across route visits so the model isn't re-downloaded.
let landmarker: PoseLandmarker | null = null;
let initPromise: Promise<PoseLandmarker> | null = null;

function getLandmarker(): Promise<PoseLandmarker> {
  if (landmarker) return Promise.resolve(landmarker);
  if (!initPromise) {
    initPromise = (async () => {
      const wasmPath = new URL(`${import.meta.env.BASE_URL}wasm`, window.location.origin).href.replace(/\/$/, '');
      const vision = await FilesetResolver.forVisionTasks(wasmPath);
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numPoses: 1,
      });
      return landmarker;
    })();
  }
  return initPromise;
}

function seekTo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    if (Math.abs(video.currentTime - time) < 0.001) {
      resolve();
      return;
    }
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      clearTimeout(timer);
      resolve();
    };
    const onSeeked = () => done();
    const onError = () => done();
    const timer = setTimeout(done, 1000);
    video.addEventListener('seeked', onSeeked);
    video.addEventListener('error', onError);
    video.currentTime = time;
  });
}

const LANDMARK_NAMES = [
  'nose',
  'left_eye_inner', 'left_eye', 'left_eye_outer',
  'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear',
  'mouth_left', 'mouth_right',
  'left_shoulder', 'right_shoulder',
  'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist',
  'left_pinky', 'right_pinky',
  'left_index', 'right_index',
  'left_thumb', 'right_thumb',
  'left_hip', 'right_hip',
  'left_knee', 'right_knee',
  'left_ankle', 'right_ankle',
  'left_heel', 'right_heel',
  'left_foot_index', 'right_foot_index',
];

// Face landmarks (indices 0-10) are skipped when drawing; the ears (7, 8) are
// kept as head-position markers and everything from the shoulders down is drawn.
const BODY_CONNECTIONS = PoseLandmarker.POSE_CONNECTIONS.filter(
  (connection) => connection.start >= 11 && connection.end >= 11
);

interface LandmarkRecord {
  index: number;
  name: string;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

interface FrameRecord {
  frame: number;
  time: number;
  landmarks: LandmarkRecord[];
}

function buildCsv(frames: FrameRecord[]): string {
  const lines = ['frame,time,pose_detected,landmark_index,landmark_name,x,y,z,visibility'];
  for (const f of frames) {
    if (f.landmarks.length === 0) {
      lines.push(`${f.frame},${f.time.toFixed(2)},false,,,,,,`);
      continue;
    }
    for (const lm of f.landmarks) {
      lines.push(
        `${f.frame},${f.time.toFixed(2)},true,${lm.index},${lm.name},` +
          `${lm.x.toFixed(6)},${lm.y.toFixed(6)},${lm.z.toFixed(6)},${lm.visibility.toFixed(6)}`
      );
    }
  }
  return lines.join('\n');
}

function buildJson(
  frames: FrameRecord[],
  imageWidth: number,
  imageHeight: number,
  segmentation?: SegmentationResult,
  report?: PushupReport
): string {
  return JSON.stringify(
    {
      task: 'pose_landmarker',
      interval_seconds: INTERVAL,
      image_width: imageWidth,
      image_height: imageHeight,
      num_reps: segmentation ? segmentation.numReps : 0,
      reps: segmentation
        ? segmentation.reps.map((r) => ({
            rep: r.rep,
            start_frame: r.startFrame,
            middle_frame: r.middleFrame,
            end_frame: r.endFrame,
            start_time: Number(r.startTime.toFixed(2)),
            middle_time: Number(r.middleTime.toFixed(2)),
            end_time: Number(r.endTime.toFixed(2)),
          }))
        : [],
      lead_in: segmentation ? segmentation.leadIn : null,
      lead_out: segmentation ? segmentation.leadOut : null,
      analysis: report ? serializeAnalysis(report) : undefined,
      frames: frames.map((f, i) => {
        const s = segmentation?.frames[i];
        return {
          frame: f.frame,
          time: Number(f.time.toFixed(2)),
          pose_detected: f.landmarks.length > 0,
          rep: s ? s.rep : null,
          phase: s ? s.phase : 'unassigned',
          elbow_angle: s && s.elbowAngle !== null ? Number(s.elbowAngle.toFixed(2)) : null,
          shoulder_y: s && s.shoulderY !== null ? Number(s.shoulderY.toFixed(4)) : null,
          landmarks: f.landmarks,
        };
      }),
    },
    null,
    2
  );
}

function serializeAnalysis(report: PushupReport) {
  return {
    total_score: report.totalScore,
    grade: report.grade,
    metrics: report.metrics.map((m) => ({
      id: m.id,
      name: m.name,
      weight: m.weight,
      score: Number(m.score.toFixed(1)),
      grade: m.grade,
      summary: m.summary,
      advice: m.advice,
      debug: report.debug ? m.debug : undefined,
    })),
    per_rep: report.perRep.map((r) => ({
      rep: r.rep,
      bottom_elbow_angle: r.bottomElbowAngle !== null ? Number(r.bottomElbowAngle.toFixed(1)) : null,
      top_elbow_angle: r.topElbowAngle !== null ? Number(r.topElbowAngle.toFixed(1)) : null,
      hip_deviation: r.hipDeviation !== null ? Number(r.hipDeviation.toFixed(1)) : null,
      down_time: r.downTime !== null ? Number(r.downTime.toFixed(2)) : null,
      up_time: r.upTime !== null ? Number(r.upTime.toFixed(2)) : null,
    })),
    frames: report.frames.map((f) => ({
      frame: f.frame,
      time: Number(f.time.toFixed(2)),
      rep: f.rep,
      phase: f.phase,
      elbow_angle: f.elbowAngle !== null ? Number(f.elbowAngle.toFixed(2)) : null,
      hip_angle: f.hipAngle !== null ? Number(f.hipAngle.toFixed(2)) : null,
      hip_deviation: f.hipDeviation !== null ? Number(f.hipDeviation.toFixed(2)) : null,
    })),
    advice: report.advice,
  };
}

const PHASE_LABELS: Record<string, string> = {
  'lead-in': 'Lead-in',
  start: 'Start',
  down: 'Down',
  middle: 'Middle',
  up: 'Up',
  end: 'End',
  'lead-out': 'Lead-out',
  unassigned: '—',
};

function phaseLabel(rep: number | null, phase: string): string {
  const phaseText = PHASE_LABELS[phase] ?? phase;
  return rep === null ? phaseText : `Rep ${rep} · ${phaseText}`;
}

function renderGroupedResults(
  results: HTMLElement,
  cards: HTMLElement[],
  labels: HTMLElement[],
  frames: FrameRecord[],
  segmentation: SegmentationResult
): void {
  segmentation.frames.forEach((s, i) => {
    if (i >= labels.length) return;
    const t = frames[i]?.time ?? i * INTERVAL;
    labels[i].textContent = `${phaseLabel(s.rep, s.phase)} · t = ${t.toFixed(2)}s`;
  });

  // Insert a full-width header before the first card of each segment (lead-in,
  // each rep, lead-out). Iterate in reverse so earlier inserts never shift the
  // card nodes we still reference.
  const headers: { text: string; cardIndex: number }[] = [];
  if (segmentation.leadIn) {
    headers.push({
      text:
        `Lead-in · frames ${segmentation.leadIn.startFrame + 1}–${segmentation.leadIn.endFrame + 1} · ` +
        `${segmentation.leadIn.startTime.toFixed(2)}s–${segmentation.leadIn.endTime.toFixed(2)}s`,
      cardIndex: segmentation.leadIn.startFrame,
    });
  }
  for (const rep of segmentation.reps) {
    headers.push({
      text:
        `Rep ${rep.rep} · frames ${rep.startFrame + 1}–${rep.endFrame + 1} · ` +
        `${rep.startTime.toFixed(2)}s–${rep.endTime.toFixed(2)}s`,
      cardIndex: rep.startFrame,
    });
  }
  if (segmentation.leadOut) {
    headers.push({
      text:
        `Lead-out · frames ${segmentation.leadOut.startFrame + 1}–${segmentation.leadOut.endFrame + 1} · ` +
        `${segmentation.leadOut.startTime.toFixed(2)}s–${segmentation.leadOut.endTime.toFixed(2)}s`,
      cardIndex: segmentation.leadOut.startFrame,
    });
  }

  for (let h = headers.length - 1; h >= 0; h--) {
    const header = document.createElement('div');
    header.className = 'swim-rep-header';
    header.textContent = headers[h].text;
    results.insertBefore(header, cards[headers[h].cardIndex]);
  }
}

function renderReport(container: HTMLElement, report: PushupReport): void {
  const metricCards = report.metrics
    .map((m) => {
      const debugHtml = report.debug
        ? `<details class="metric-debug"><summary>How this was calculated</summary><ul>${m.debug
            .map((d) => `<li><strong>${d.label}:</strong> ${d.detail}</li>`)
            .join('')}</ul></details>`
        : '';
      return `
      <div class="report-metric">
        <div class="metric-head">
          <span class="metric-name">${m.name}</span>
          <span class="metric-score">${m.score.toFixed(0)}</span>
        </div>
        <div class="metric-bar"><div class="metric-bar-fill" style="width: ${m.score.toFixed(0)}%"></div></div>
        <div class="metric-grade grade-${m.grade.toLowerCase()}">${m.grade}</div>
        <p class="metric-summary">${m.summary}</p>
        <p class="metric-advice">${m.advice}</p>
        ${debugHtml}
      </div>`;
    })
    .join('');

  const adviceItems = report.advice.map((a) => `<li>${a}</li>`).join('');

  container.innerHTML = `
    <div class="report-card">
      <div class="report-header">
        <h3>Push-up Quality Report</h3>
        <div class="report-score">
          <div class="report-total">${report.totalScore}</div>
          <div class="report-grade grade-${report.grade.toLowerCase()}">${report.grade}</div>
        </div>
      </div>
      <div class="report-metrics">${metricCards}</div>
      <div class="report-advice">
        <h4>Advice</h4>
        <ul>${adviceItems}</ul>
      </div>
    </div>`;
  container.style.display = '';
}

export async function setupSwimmingAnalyzer(container: HTMLElement) {
  container.innerHTML = template;

  const uploadStage = container.querySelector<HTMLElement>('#upload-stage')!;
  const videoUpload = container.querySelector<HTMLInputElement>('#video-upload')!;
  const progress = container.querySelector<HTMLElement>('#progress')!;
  const reportContainer = container.querySelector<HTMLElement>('#report')!;
  const results = container.querySelector<HTMLElement>('#results')!;
  const downloadAll = container.querySelector<HTMLButtonElement>('#download-all')!;
  const downloadData = container.querySelector<HTMLButtonElement>('#download-data')!;

  const markedBlobs: { name: string; blob: Blob }[] = [];
  const framesData: FrameRecord[] = [];
  const cards: HTMLElement[] = [];
  const labels: HTMLElement[] = [];
  let segmentation: SegmentationResult | null = null;
  let report: PushupReport | null = null;
  let imageWidth = 0;
  let imageHeight = 0;

  uploadStage.addEventListener('click', () => videoUpload.click());

  async function processFile(file: File) {
    uploadStage.style.display = 'none';
    results.innerHTML = '';
    reportContainer.innerHTML = '';
    reportContainer.style.display = 'none';
    markedBlobs.length = 0;
    framesData.length = 0;
    cards.length = 0;
    labels.length = 0;
    segmentation = null;
    report = null;
    imageWidth = 0;
    imageHeight = 0;
    downloadAll.style.display = 'none';
    downloadData.style.display = 'none';
    progress.style.display = '';
    progress.textContent = 'Loading pose model...';

    try {
      const lm = await getLandmarker();
      progress.textContent = 'Reading video...';

      const url = URL.createObjectURL(file);
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'auto';
      video.src = url;

      // `loadeddata` (not `loadedmetadata`) guarantees the first frame's pixel
      // data is decoded, so drawing at t=0 isn't blank.
      await new Promise<void>((resolve, reject) => {
        video.addEventListener('loadeddata', () => resolve(), { once: true });
        video.addEventListener('error', () => reject(new Error('Could not load video')), { once: true });
      });

      const duration = video.duration;
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const numFrames = Math.max(1, Math.round(duration / INTERVAL));

      const scale = Math.min(1, MAX_EDGE / Math.max(vw, vh));
      const sw = Math.max(1, Math.round(vw * scale));
      const sh = Math.max(1, Math.round(vh * scale));
      imageWidth = sw;
      imageHeight = sh;

      const source = document.createElement('canvas');
      source.width = sw;
      source.height = sh;
      const sourceCtx = source.getContext('2d')!;
      const drawingUtils = new DrawingUtils(sourceCtx);

      for (let i = 0; i < numFrames; i++) {
        const t = Math.min(i * INTERVAL, Math.max(0, duration - 0.05));
        await seekTo(video, t);
        sourceCtx.drawImage(video, 0, 0, sw, sh);

        const result = lm.detect(source);

        const frameRecord: FrameRecord = { frame: i, time: t, landmarks: [] };
        if (result.landmarks && result.landmarks.length > 0) {
          const landmarks = result.landmarks[0];
          drawingUtils.drawConnectors(landmarks, BODY_CONNECTIONS, {
            color: '#00e676',
            lineWidth: 2,
          });
          const drawnLandmarks = landmarks.filter((_, idx) => idx >= 11 || idx === 7 || idx === 8);
          drawingUtils.drawLandmarks(drawnLandmarks, { color: '#00e676', fillColor: '#00e676', radius: 3 });
          landmarks.forEach((lm, idx) => {
            frameRecord.landmarks.push({
              index: idx,
              name: LANDMARK_NAMES[idx],
              x: lm.x,
              y: lm.y,
              z: lm.z,
              visibility: lm.visibility,
            });
          });
        }
        framesData.push(frameRecord);

        const frameCanvas = document.createElement('canvas');
        frameCanvas.width = sw;
        frameCanvas.height = sh;
        frameCanvas.getContext('2d')!.drawImage(source, 0, 0);

        const blob: Blob = await new Promise((resolve) =>
          frameCanvas.toBlob((b) => resolve(b || new Blob()), 'image/jpeg', 0.9)
        );
        markedBlobs.push({ name: `frame_${String(i).padStart(4, '0')}.jpg`, blob });

        const card = document.createElement('div');
        card.className = 'swim-frame';
        const label = document.createElement('div');
        label.className = 'swim-frame-label';
        label.textContent = `t = ${t.toFixed(2)}s`;
        card.appendChild(frameCanvas);
        card.appendChild(label);
        results.appendChild(card);
        cards.push(card);
        labels.push(label);

        progress.textContent = `Processing frame ${i + 1}/${numFrames} (t = ${t.toFixed(2)}s)`;
        await new Promise((r) => setTimeout(r, 0));
      }

      segmentation = segmentReps(framesData);
      renderGroupedResults(results, cards, labels, framesData, segmentation);

      report = analyzePushup(framesData, segmentation);
      renderReport(reportContainer, report);

      URL.revokeObjectURL(url);
      progress.textContent =
        `Done: ${markedBlobs.length} frames marked · ${segmentation.numReps} reps detected · total score ${report.totalScore}/100.`;
      if (markedBlobs.length > 0) {
        downloadAll.style.display = '';
        downloadData.style.display = '';
      }
    } catch (err) {
      progress.textContent = `Error: ${err instanceof Error ? err.message : String(err)}`;
      uploadStage.style.display = '';
    }
  }

  videoUpload.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    videoUpload.value = '';
    processFile(file);
  });

  uploadStage.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  });

  uploadStage.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) processFile(file);
  });

  downloadAll.addEventListener('click', async () => {
    if (markedBlobs.length === 0) return;
    downloadAll.disabled = true;
    const original = downloadAll.innerHTML;
    downloadAll.textContent = 'Packaging...';
    try {
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();
      for (const { name, blob } of markedBlobs) zip.file(name, blob);
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(zipBlob);
      a.download = 'marked_frames.zip';
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      downloadAll.disabled = false;
      downloadAll.innerHTML = original;
    }
  });

  downloadData.addEventListener('click', async () => {
    if (framesData.length === 0) return;
    downloadData.disabled = true;
    const original = downloadData.innerHTML;
    downloadData.textContent = 'Packaging...';
    try {
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();
      zip.file('landmarks.csv', buildCsv(framesData));
      zip.file('landmarks.json', buildJson(framesData, imageWidth, imageHeight, segmentation ?? undefined, report ?? undefined));
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(zipBlob);
      a.download = 'landmark_data.zip';
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      downloadData.disabled = false;
      downloadData.innerHTML = original;
    }
  });
}

export function cleanupSwimmingAnalyzer() {
  // The pose model is cached at module scope; nothing to tear down per-visit.
}
