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

import './inference-timer.css';

export interface InferenceSample {
  time: number;
  delegate?: 'CPU' | 'GPU';
  timeB?: number;
}

interface CumulativeStats {
  gpuSum: number;
  gpuCount: number;
  cpuSum: number;
  cpuCount: number;
  seriesASum: number;
  seriesACount: number;
  seriesBSum: number;
  seriesBCount: number;
  history: InferenceSample[];
}

export type InferenceTimerMode = 'delegate' | 'dual' | 'single';

export interface InferenceTimerOptions {
  rollingWindowSize?: number;
  maxHistorySize?: number;
  textUpdateIntervalMs?: number;
  mode?: InferenceTimerMode;
  labelA?: string;
  labelB?: string;
}

const GPU_COLOR = '#007f8b';
const GPU_FILL = 'rgba(0, 127, 139, 0.16)';
const CPU_COLOR = '#d97706';
const CPU_FILL = 'rgba(217, 119, 6, 0.16)';

// Persist cumulative averages and history per route since page load
const sessionStatsByRoute = new Map<string, CumulativeStats>();

function getRouteStats(): CumulativeStats {
  const key = window.location.hash || '#/vision/object_detector';
  let stats = sessionStatsByRoute.get(key);
  if (!stats) {
    stats = {
      gpuSum: 0,
      gpuCount: 0,
      cpuSum: 0,
      cpuCount: 0,
      seriesASum: 0,
      seriesACount: 0,
      seriesBSum: 0,
      seriesBCount: 0,
      history: [],
    };
    sessionStatsByRoute.set(key, stats);
  }
  return stats;
}

export class InferenceTimer {
  private readonly rollingWindowSize: number;
  private readonly maxHistorySize: number;
  private readonly textUpdateIntervalMs: number;
  private readonly mode: InferenceTimerMode;
  private readonly labelA: string;
  private readonly labelB: string;

  private rollingSamples: number[] = [];
  private lastTextUpdateMs = 0;
  private lastDisplayedAvg = 0;
  private pendingUpdateTimer: number | undefined;

  private canvas: HTMLCanvasElement | null = null;
  private valueBadgeEl: HTMLElement | null = null;
  private gpuAvgEl: HTMLElement | null = null;
  private cpuAvgEl: HTMLElement | null = null;
  private seriesAAvgEl: HTMLElement | null = null;
  private seriesBAvgEl: HTMLElement | null = null;

  constructor(
    optionsOrWindowSize: number | InferenceTimerOptions = 10,
    maxHistorySize = 50,
    textUpdateIntervalMs = 180
  ) {
    if (typeof optionsOrWindowSize === 'object' && optionsOrWindowSize !== null) {
      this.rollingWindowSize = optionsOrWindowSize.rollingWindowSize ?? 10;
      this.maxHistorySize = optionsOrWindowSize.maxHistorySize ?? 50;
      this.textUpdateIntervalMs = optionsOrWindowSize.textUpdateIntervalMs ?? 180;
      this.mode = optionsOrWindowSize.mode ?? 'delegate';
      this.labelA =
        optionsOrWindowSize.labelA ?? (this.mode === 'dual' ? 'A' : this.mode === 'single' ? 'Query' : 'GPU');
      this.labelB = optionsOrWindowSize.labelB ?? (this.mode === 'dual' ? 'B' : 'CPU');
    } else {
      this.rollingWindowSize = optionsOrWindowSize;
      this.maxHistorySize = maxHistorySize;
      this.textUpdateIntervalMs = textUpdateIntervalMs;
      this.mode = 'delegate';
      this.labelA = 'GPU';
      this.labelB = 'CPU';
    }
  }

  public mount() {
    const statusGroup = document.querySelector('.status-group');
    if (!statusGroup) return;

    const statusEl = document.getElementById('status-message');
    if (statusEl) {
      statusEl.classList.add('sr-only-status');
    }

    const stats = getRouteStats();
    const lastSample = stats.history.length > 0 ? stats.history[stats.history.length - 1] : null;
    let initialBadgeText = '- ms';
    if (lastSample) {
      if (this.mode === 'dual' && lastSample.timeB !== undefined) {
        initialBadgeText = `${(lastSample.time + lastSample.timeB).toFixed(2)} ms`;
      } else {
        initialBadgeText = `${lastSample.time.toFixed(2)} ms`;
      }
    }

    const inferenceTimeEl = document.getElementById('inference-time');
    if (inferenceTimeEl) {
      inferenceTimeEl.className = 'inference-header-row';
      inferenceTimeEl.innerHTML = `<span class="inference-section-title">Inference Time<span class="inference-sr-colon">: </span></span><span id="inference-time-value" class="inference-current-badge">${initialBadgeText}</span>`;
      this.valueBadgeEl = inferenceTimeEl.querySelector('#inference-time-value');
    }

    if (statusEl && inferenceTimeEl) {
      statusGroup.insertBefore(statusEl, inferenceTimeEl.nextSibling);
    }

    let container = document.getElementById('inference-history-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'inference-history-container';
      container.className = 'inference-history-body';

      let legendHtml = '';
      if (this.mode === 'dual') {
        legendHtml = `
          <span class="inference-delegate-stat">
            <span class="inference-legend-dot gpu"></span>
            ${this.labelA}: <strong id="inference-series-a-avg">--</strong>
          </span>
          <span class="inference-delegate-stat">
            <span class="inference-legend-dot cpu"></span>
            ${this.labelB}: <strong id="inference-series-b-avg">--</strong>
          </span>
        `;
      } else if (this.mode === 'single') {
        legendHtml = `
          <span class="inference-delegate-stat">
            <span class="inference-legend-dot gpu"></span>
            ${this.labelA}: <strong id="inference-series-a-avg">--</strong>
          </span>
        `;
      } else {
        legendHtml = `
          <span class="inference-delegate-stat">
            <span class="inference-legend-dot gpu"></span>
            ${this.labelA}: <strong id="inference-gpu-avg">--</strong>
          </span>
          <span class="inference-delegate-stat">
            <span class="inference-legend-dot cpu"></span>
            ${this.labelB}: <strong id="inference-cpu-avg">--</strong>
          </span>
        `;
      }

      container.innerHTML = `
        <div class="inference-graph-wrapper">
          <canvas id="inference-history-canvas" class="inference-history-canvas"></canvas>
        </div>
        <div class="inference-delegate-summary">
          ${legendHtml}
        </div>
      `;

      statusGroup.appendChild(container);
    }

    this.canvas = container.querySelector('#inference-history-canvas') as HTMLCanvasElement | null;
    this.gpuAvgEl = container.querySelector('#inference-gpu-avg') as HTMLElement | null;
    this.cpuAvgEl = container.querySelector('#inference-cpu-avg') as HTMLElement | null;
    this.seriesAAvgEl = container.querySelector('#inference-series-a-avg') as HTMLElement | null;
    this.seriesBAvgEl = container.querySelector('#inference-series-b-avg') as HTMLElement | null;

    this.drawGraph();
    this.updateDelegateSummaries();
  }

  /**
   * Keeps #status-message accessible in the DOM for tests/screen-readers while showing
   * loading/error states in the Inference Time badge until inference time is available.
   */
  public syncStatusVisibility(msg: string, state?: string) {
    const statusEl = document.getElementById('status-message');

    if (!this.valueBadgeEl || !this.valueBadgeEl.isConnected) {
      this.valueBadgeEl = document.getElementById('inference-time-value');
    }
    if (!this.valueBadgeEl) return;

    const isError = state === 'error' || /^error\b/i.test(msg.trim()) || msg.toLowerCase().includes('failed');
    const isIdleOrReady =
      state === 'idle' ||
      state === 'ready' ||
      msg.startsWith('Done') ||
      msg === 'Ready' ||
      msg === 'Webcam running...' ||
      /ready/i.test(msg);

    if (isError) {
      if (statusEl) {
        statusEl.classList.remove('sr-only-status');
        statusEl.classList.add('status-error');
      }
      this.valueBadgeEl.classList.add('error');
      this.valueBadgeEl.textContent = 'Error';
      return;
    }

    if (statusEl) {
      statusEl.classList.remove('status-error');
      statusEl.classList.add('sr-only-status');
    }
    this.valueBadgeEl.classList.remove('error');

    if (!isIdleOrReady) {
      this.valueBadgeEl.textContent = msg.length > 24 ? `${msg.slice(0, 22)}…` : msg;
    } else if (this.lastDisplayedAvg > 0) {
      this.valueBadgeEl.textContent = `${this.lastDisplayedAvg.toFixed(2)} ms`;
    } else {
      this.valueBadgeEl.textContent = '- ms';
    }
  }

  public setStatus(stateOrMsg: string, message?: string) {
    const msg = message !== undefined ? message : stateOrMsg;
    const state = message !== undefined ? stateOrMsg : undefined;
    const statusEl = document.getElementById('status-message');
    if (statusEl) {
      statusEl.textContent = msg;
    }
    this.syncStatusVisibility(msg, state);
  }

  /**
   * Resets the short-term rolling average window (e.g. when switching between CPU and GPU)
   * while preserving cumulative CPU/GPU averages and graph history since page load.
   */
  public resetRollingWindow() {
    this.rollingSamples = [];
    this.lastTextUpdateMs = 0;
    if (this.pendingUpdateTimer !== undefined) {
      window.clearTimeout(this.pendingUpdateTimer);
      this.pendingUpdateTimer = undefined;
    }
  }

  public recordDual(timeA: number, timeB: number): number {
    if (!Number.isFinite(timeA) || timeA < 0 || !Number.isFinite(timeB) || timeB < 0) {
      return this.lastDisplayedAvg;
    }

    const stats = getRouteStats();
    stats.seriesASum += timeA;
    stats.seriesACount += 1;
    stats.seriesBSum += timeB;
    stats.seriesBCount += 1;

    const total = timeA + timeB;
    this.rollingSamples.push(total);
    if (this.rollingSamples.length > this.rollingWindowSize) {
      this.rollingSamples.shift();
    }

    stats.history.push({ time: timeA, timeB: timeB });
    if (stats.history.length > this.maxHistorySize) {
      stats.history.shift();
    }

    const avg = this.computeRollingAverage();
    const now = performance.now();

    if (this.rollingSamples.length === 1 || now - this.lastTextUpdateMs >= this.textUpdateIntervalMs) {
      this.applyTextUpdate(avg);
      this.lastTextUpdateMs = now;
    } else if (this.pendingUpdateTimer === undefined) {
      const remaining = this.textUpdateIntervalMs - (now - this.lastTextUpdateMs);
      this.pendingUpdateTimer = window.setTimeout(
        () => {
          this.pendingUpdateTimer = undefined;
          this.applyTextUpdate(this.computeRollingAverage());
          this.lastTextUpdateMs = performance.now();
        },
        Math.max(16, remaining)
      );
    }

    if (!this.canvas || !this.canvas.isConnected) {
      this.mount();
    } else {
      this.drawGraph();
      this.updateDelegateSummaries();
    }

    return this.lastDisplayedAvg;
  }

  public recordSingle(time: number): number {
    if (!Number.isFinite(time) || time < 0) {
      return this.lastDisplayedAvg;
    }

    const stats = getRouteStats();
    stats.seriesASum += time;
    stats.seriesACount += 1;

    this.rollingSamples.push(time);
    if (this.rollingSamples.length > this.rollingWindowSize) {
      this.rollingSamples.shift();
    }

    stats.history.push({ time });
    if (stats.history.length > this.maxHistorySize) {
      stats.history.shift();
    }

    const avg = this.computeRollingAverage();
    const now = performance.now();

    if (this.rollingSamples.length === 1 || now - this.lastTextUpdateMs >= this.textUpdateIntervalMs) {
      this.applyTextUpdate(avg);
      this.lastTextUpdateMs = now;
    } else if (this.pendingUpdateTimer === undefined) {
      const remaining = this.textUpdateIntervalMs - (now - this.lastTextUpdateMs);
      this.pendingUpdateTimer = window.setTimeout(
        () => {
          this.pendingUpdateTimer = undefined;
          this.applyTextUpdate(this.computeRollingAverage());
          this.lastTextUpdateMs = performance.now();
        },
        Math.max(16, remaining)
      );
    }

    if (!this.canvas || !this.canvas.isConnected) {
      this.mount();
    } else {
      this.drawGraph();
      this.updateDelegateSummaries();
    }

    return this.lastDisplayedAvg;
  }

  public record(time: number, delegate: 'CPU' | 'GPU'): number;
  public record(timeA: number, timeB: number): number;
  public record(timeOrA: number, delegateOrB: 'CPU' | 'GPU' | number = 'GPU'): number {
    if (this.mode === 'dual' && typeof delegateOrB === 'number') {
      return this.recordDual(timeOrA, delegateOrB);
    }
    if (this.mode === 'single') {
      return this.recordSingle(timeOrA);
    }

    const delegate = delegateOrB === 'CPU' || delegateOrB === 'GPU' ? delegateOrB : 'GPU';
    if (!Number.isFinite(timeOrA) || timeOrA < 0) {
      return this.lastDisplayedAvg;
    }

    const stats = getRouteStats();

    if (stats.history.length > 0 && stats.history[stats.history.length - 1].delegate !== delegate) {
      this.rollingSamples = [];
      this.lastTextUpdateMs = 0;
    }

    this.rollingSamples.push(timeOrA);
    if (this.rollingSamples.length > this.rollingWindowSize) {
      this.rollingSamples.shift();
    }

    if (delegate === 'GPU') {
      stats.gpuSum += timeOrA;
      stats.gpuCount += 1;
    } else {
      stats.cpuSum += timeOrA;
      stats.cpuCount += 1;
    }

    stats.history.push({ time: timeOrA, delegate });
    if (stats.history.length > this.maxHistorySize) {
      stats.history.shift();
    }

    const avg = this.computeRollingAverage();
    const now = performance.now();

    if (this.rollingSamples.length === 1 || now - this.lastTextUpdateMs >= this.textUpdateIntervalMs) {
      this.applyTextUpdate(avg);
      this.lastTextUpdateMs = now;
    } else if (this.pendingUpdateTimer === undefined) {
      const remaining = this.textUpdateIntervalMs - (now - this.lastTextUpdateMs);
      this.pendingUpdateTimer = window.setTimeout(
        () => {
          this.pendingUpdateTimer = undefined;
          this.applyTextUpdate(this.computeRollingAverage());
          this.lastTextUpdateMs = performance.now();
        },
        Math.max(16, remaining)
      );
    }

    if (!this.canvas || !this.canvas.isConnected) {
      this.mount();
    } else {
      this.drawGraph();
      this.updateDelegateSummaries();
    }

    return this.lastDisplayedAvg;
  }

  public cleanup() {
    if (this.pendingUpdateTimer !== undefined) {
      window.clearTimeout(this.pendingUpdateTimer);
      this.pendingUpdateTimer = undefined;
    }
  }

  private computeRollingAverage(): number {
    if (this.rollingSamples.length === 0) return 0;
    const sum = this.rollingSamples.reduce((acc, v) => acc + v, 0);
    return sum / this.rollingSamples.length;
  }

  private applyTextUpdate(avg: number) {
    this.lastDisplayedAvg = avg;
    const statusEl = document.getElementById('status-message');
    if (statusEl && statusEl.classList.contains('status-error')) {
      statusEl.classList.remove('status-error');
      statusEl.classList.add('sr-only-status');
    }
    if (!this.valueBadgeEl || !this.valueBadgeEl.isConnected) {
      this.valueBadgeEl = document.getElementById('inference-time-value');
    }
    if (this.valueBadgeEl) {
      this.valueBadgeEl.classList.remove('error');
      this.valueBadgeEl.textContent = `${avg.toFixed(2)} ms`;
    } else {
      const el = document.getElementById('inference-time');
      if (el) {
        el.innerText = `Inference Time: ${avg.toFixed(2)} ms`;
      }
    }
  }

  private updateDelegateSummaries() {
    const stats = getRouteStats();

    if (this.mode === 'dual') {
      if (this.seriesAAvgEl) {
        if (stats.seriesACount > 0) {
          const avg = stats.seriesASum / stats.seriesACount;
          this.seriesAAvgEl.textContent = `${avg.toFixed(1)} ms`;
        } else {
          this.seriesAAvgEl.textContent = '--';
        }
      }
      if (this.seriesBAvgEl) {
        if (stats.seriesBCount > 0) {
          const avg = stats.seriesBSum / stats.seriesBCount;
          this.seriesBAvgEl.textContent = `${avg.toFixed(1)} ms`;
        } else {
          this.seriesBAvgEl.textContent = '--';
        }
      }
      return;
    }

    if (this.mode === 'single') {
      if (this.seriesAAvgEl) {
        if (stats.seriesACount > 0) {
          const avg = stats.seriesASum / stats.seriesACount;
          this.seriesAAvgEl.textContent = `${avg.toFixed(1)} ms`;
        } else {
          this.seriesAAvgEl.textContent = '--';
        }
      }
      return;
    }

    if (this.gpuAvgEl) {
      if (stats.gpuCount > 0) {
        const avg = stats.gpuSum / stats.gpuCount;
        this.gpuAvgEl.textContent = `${avg.toFixed(1)} ms`;
      } else {
        this.gpuAvgEl.textContent = '--';
      }
    }

    if (this.cpuAvgEl) {
      if (stats.cpuCount > 0) {
        const avg = stats.cpuSum / stats.cpuCount;
        this.cpuAvgEl.textContent = `${avg.toFixed(1)} ms`;
      } else {
        this.cpuAvgEl.textContent = '--';
      }
    }
  }

  private drawGraph() {
    if (!this.canvas) return;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;

    const stats = getRouteStats();
    const history = stats.history;

    const dpr = window.devicePixelRatio || 1;
    const cssWidth = this.canvas.clientWidth || 248;
    const cssHeight = this.canvas.clientHeight || 64;

    if (this.canvas.width !== Math.round(cssWidth * dpr) || this.canvas.height !== Math.round(cssHeight * dpr)) {
      this.canvas.width = Math.round(cssWidth * dpr);
      this.canvas.height = Math.round(cssHeight * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const padLeft = 40;
    const padRight = 8;
    const padTop = 8;
    const padBottom = 8;
    const plotWidth = Math.max(10, cssWidth - padLeft - padRight);
    const plotHeight = Math.max(10, cssHeight - padTop - padBottom);

    // Subtle horizontal grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    for (const frac of [0, 0.5, 1]) {
      const y = padTop + plotHeight * frac;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotWidth, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    if (history.length === 0) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px "Google Sans", "Roboto", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No data yet', cssWidth / 2, cssHeight / 2);
      ctx.restore();
      return;
    }

    const maxVal = Math.max(...history.map((d) => (d.timeB !== undefined ? Math.max(d.time, d.timeB) : d.time)), 5);
    const yMax = maxVal * 1.2;

    const formatAxisLabel = (ms: number) => {
      if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
      return `${Math.round(ms)}ms`;
    };

    // Y-axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '9px "Roboto Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(formatAxisLabel(yMax), padLeft - 5, padTop + 2);
    ctx.fillText('0ms', padLeft - 5, padTop + plotHeight - 1);

    const getY = (val: number) => {
      const clamped = Math.max(0, Math.min(yMax, val));
      return padTop + plotHeight - (clamped / yMax) * plotHeight;
    };

    const n = history.length;
    const baseLineY = padTop + plotHeight;
    const getX = (index: number) => (n === 1 ? padLeft + plotWidth / 2 : padLeft + (index / (n - 1)) * plotWidth);

    // 1. DUAL MODE: Draw two distinct series (A and B)
    if (this.mode === 'dual') {
      if (n === 1) {
        const sample = history[0];
        const yA = getY(sample.time);
        ctx.fillStyle = GPU_FILL;
        ctx.fillRect(padLeft, yA, plotWidth, baseLineY - yA);
        ctx.strokeStyle = GPU_COLOR;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padLeft, yA);
        ctx.lineTo(padLeft + plotWidth, yA);
        ctx.stroke();

        if (sample.timeB !== undefined) {
          const yB = getY(sample.timeB);
          ctx.fillStyle = CPU_FILL;
          ctx.fillRect(padLeft, yB, plotWidth, baseLineY - yB);
          ctx.strokeStyle = CPU_COLOR;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(padLeft, yB);
          ctx.lineTo(padLeft + plotWidth, yB);
          ctx.stroke();
        }
        ctx.restore();
        return;
      }

      // Series A (Teal)
      const ptsA = history.map((s, i) => ({ x: getX(i), y: getY(s.time) }));
      ctx.fillStyle = GPU_FILL;
      ctx.beginPath();
      ctx.moveTo(ptsA[0].x, baseLineY);
      ptsA.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.lineTo(ptsA[ptsA.length - 1].x, baseLineY);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = GPU_COLOR;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      ptsA.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.stroke();

      // Series B (Amber)
      const ptsB = history.map((s, i) => ({ x: getX(i), y: getY(s.timeB ?? 0) }));
      ctx.fillStyle = CPU_FILL;
      ctx.beginPath();
      ctx.moveTo(ptsB[0].x, baseLineY);
      ptsB.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.lineTo(ptsB[ptsB.length - 1].x, baseLineY);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = CPU_COLOR;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      ptsB.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.stroke();

      if (n <= 25) {
        ptsA.forEach((p) => {
          ctx.fillStyle = GPU_COLOR;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
        ptsB.forEach((p) => {
          ctx.fillStyle = CPU_COLOR;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      ctx.restore();
      return;
    }

    // 2. SINGLE MODE: Draw single series
    if (this.mode === 'single') {
      if (n === 1) {
        const sample = history[0];
        const y = getY(sample.time);
        ctx.fillStyle = GPU_FILL;
        ctx.fillRect(padLeft, y, plotWidth, baseLineY - y);

        ctx.strokeStyle = GPU_COLOR;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(padLeft + plotWidth, y);
        ctx.stroke();

        ctx.fillStyle = GPU_COLOR;
        ctx.beginPath();
        ctx.arc(padLeft + plotWidth / 2, y, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
        return;
      }

      const pts = history.map((s, i) => ({ x: getX(i), y: getY(s.time) }));
      ctx.fillStyle = GPU_FILL;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, baseLineY);
      pts.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.lineTo(pts[pts.length - 1].x, baseLineY);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = GPU_COLOR;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.stroke();

      if (n <= 25) {
        pts.forEach((p) => {
          ctx.fillStyle = GPU_COLOR;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      ctx.restore();
      return;
    }

    // 3. DELEGATE MODE: Contiguous delegate segments with transitions
    if (n === 1) {
      const sample = history[0];
      const y = getY(sample.time);
      const color = sample.delegate === 'GPU' ? GPU_COLOR : CPU_COLOR;
      const fill = sample.delegate === 'GPU' ? GPU_FILL : CPU_FILL;

      ctx.fillStyle = fill;
      ctx.fillRect(padLeft, y, plotWidth, baseLineY - y);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotWidth, y);
      ctx.stroke();

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(padLeft + plotWidth / 2, y, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
      return;
    }

    // Build contiguous delegate segments splitting at the midpoint when delegate switches
    interface Point {
      x: number;
      y: number;
    }
    interface Segment {
      delegate: 'CPU' | 'GPU';
      points: Point[];
    }

    const segments: Segment[] = [
      {
        delegate: history[0].delegate ?? 'GPU',
        points: [{ x: getX(0), y: getY(history[0].time) }],
      },
    ];
    const transitionXs: number[] = [];

    for (let i = 1; i < n; i++) {
      const prev = history[i - 1];
      const curr = history[i];
      const xPrev = getX(i - 1);
      const yPrev = getY(prev.time);
      const xCurr = getX(i);
      const yCurr = getY(curr.time);
      const prevDelegate = prev.delegate ?? 'GPU';
      const currDelegate = curr.delegate ?? 'GPU';

      if (currDelegate === prevDelegate) {
        segments[segments.length - 1].points.push({ x: xCurr, y: yCurr });
      } else {
        const midX = (xPrev + xCurr) / 2;
        const midY = (yPrev + yCurr) / 2;
        segments[segments.length - 1].points.push({ x: midX, y: midY });
        transitionXs.push(midX);
        segments.push({
          delegate: currDelegate,
          points: [
            { x: midX, y: midY },
            { x: xCurr, y: yCurr },
          ],
        });
      }
    }

    for (const seg of segments) {
      if (seg.points.length < 2) continue;
      const strokeColor = seg.delegate === 'GPU' ? GPU_COLOR : CPU_COLOR;
      const fillColor = seg.delegate === 'GPU' ? GPU_FILL : CPU_FILL;

      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.moveTo(seg.points[0].x, baseLineY);
      for (const pt of seg.points) {
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.lineTo(seg.points[seg.points.length - 1].x, baseLineY);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      seg.points.forEach((pt, idx) => {
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
    }

    for (const markerX of transitionXs) {
      ctx.save();
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(markerX, padTop);
      ctx.lineTo(markerX, baseLineY);
      ctx.stroke();
      ctx.restore();
    }

    if (n <= 15) {
      for (let i = 0; i < n; i++) {
        const x = getX(i);
        const y = getY(history[i].time);
        ctx.fillStyle = (history[i].delegate ?? 'GPU') === 'GPU' ? GPU_COLOR : CPU_COLOR;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();
  }
}
