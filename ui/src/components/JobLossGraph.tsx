'use client';

import { Job } from '@prisma/client';
import type { JobConfig } from '@/types';
import useJobLossLog, { LossPoint } from '@/hooks/useJobLossLog';
import { useMemo, useState, useEffect, useCallback, useRef } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { NumberInput } from '@/components/formInputs';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface Props {
  job: Job;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (v === 0) return '0';
  const abs = Math.abs(v);
  // Very small / very large magnitudes read better as exponents (e.g. 1.00e-5)
  // than as long decimal strings like 0.0000100.
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential(2);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(3);
  if (abs >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// Fallback canvas height used before the container has been measured.
const FALLBACK_CANVAS_HEIGHT = 360;
const MIN_CANVAS_HEIGHT = 160;

// Compute canvas size so uPlot's canvas + its HTML legend fit inside `host`.
// `host` should be a layout-controlled wrapper (NOT the uPlot mount node, since
// uPlot's stylesheet sets `width: min-content` on its mount node).
function computeCanvasSize(host: HTMLElement): { width: number; height: number } | null {
  const { width, height } = host.getBoundingClientRect();
  if (width <= 0 || height <= 0) return null;
  const legend = host.querySelector('.u-legend') as HTMLElement | null;
  const legendH = legend?.getBoundingClientRect().height ?? 0;
  return { width, height: Math.max(MIN_CANVAS_HEIGHT, height - legendH) };
}

// One-directional bias-corrected EMA. The accumulator starts at 0 and each
// output is divided by w = 1-(1-alpha)^n (n = valid points seen so far) so early
// outputs reflect the running mean rather than the raw accumulator. `w` doubles
// as a confidence weight (→0 with one point seen, →1 once warmed up) used to
// combine the two passes below. Nulls are preserved as gaps and do not advance
// the average. `reverse` walks the series back-to-front (the backward pass).
function emaPass(
  ys: (number | null)[],
  alpha: number,
  reverse: boolean,
): { vals: (number | null)[]; weights: number[] } {
  const vals: (number | null)[] = new Array(ys.length).fill(null);
  const weights: number[] = new Array(ys.length).fill(0);
  let s = 0; // raw EMA accumulator
  let n = 0; // valid points incorporated so far
  const start = reverse ? ys.length - 1 : 0;
  const step = reverse ? -1 : 1;
  for (let i = start; i >= 0 && i < ys.length; i += step) {
    const v = ys[i];
    if (v === null || !Number.isFinite(v)) continue;
    s = alpha * (v as number) + (1 - alpha) * s;
    n += 1;
    const w = 1 - Math.pow(1 - alpha, n);
    vals[i] = s / w;
    weights[i] = w;
  }
  return { vals, weights };
}

// Zero-phase (forward-backward) EMA, combined by each pass's confidence weight.
// A one-sided EMA pins the first point to its raw value and the last point is a
// pure causal (lagging) estimate. Running a forward and backward pass and
// blending them by how much data each has seen at that index gives the best of
// both: at the start the forward pass has ~1 point (distrusted) so the
// backward, future-informed pass dominates; at the latest points the backward
// pass has ~1 point so the forward (causal) estimate dominates; the middle is
// ~50/50, which also cancels EMA's lag. Nulls stay null (both passes align).
function emaWithNulls(ys: (number | null)[], alpha: number): (number | null)[] {
  const fwd = emaPass(ys, alpha, false);
  const bwd = emaPass(ys, alpha, true);
  const out: (number | null)[] = new Array(ys.length);
  for (let i = 0; i < ys.length; i++) {
    const f = fwd.vals[i];
    const b = bwd.vals[i];
    if (f === null || b === null) {
      out[i] = null;
      continue;
    }
    const wf = fwd.weights[i];
    const wb = bwd.weights[i];
    const wsum = wf + wb;
    out[i] = wsum > 0 ? (wf * (f as number) + wb * (b as number)) / wsum : ((f as number) + (b as number)) / 2;
  }
  return out;
}

function hashToIndex(str: string, mod: number) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % mod;
}

const PALETTE = [
  'rgba(96,165,250,1)', // blue-400
  'rgba(52,211,153,1)', // emerald-400
  'rgba(167,139,250,1)', // purple-400
  'rgba(251,191,36,1)', // amber-400
  'rgba(244,114,182,1)', // pink-400
  'rgba(248,113,113,1)', // red-400
  'rgba(34,211,238,1)', // cyan-400
  'rgba(129,140,248,1)', // indigo-400
];

function strokeForKey(key: string) {
  return PALETTE[hashToIndex(key, PALETTE.length)];
}

/** Matches Python: `f"{int(round(q * 1_000_000)):07d}"` for loss_log.db metric keys. */
function gradNormQuantileSuffix(q: number): string {
  return `${Math.round(q * 1e6).toString().padStart(7, '0')}`;
}

/** Extra bucket series from job config (grad_norm_log_percentiles). */
function gradNormPercentileMetricCharts(
  jobConfigStr: string | null,
): { key: string; label: string; color: string }[] {
  if (!jobConfigStr) return [];
  try {
    const jc = JSON.parse(jobConfigStr) as JobConfig;
    const qs = jc?.config?.process?.[0]?.train?.grad_norm_log_percentiles;
    if (!Array.isArray(qs)) return [];
    const seenQ = new Set<string>();
    const out: { key: string; label: string; color: string }[] = [];
    for (const raw of qs) {
      const q = Number(raw);
      if (!Number.isFinite(q) || q < 0 || q > 1) continue;
      const suf = gradNormQuantileSuffix(q);
      if (seenQ.has(suf)) continue;
      seenQ.add(suf);
      const preKey = `grad_norm_pre_q${suf}`;
      const postKey = `grad_norm_post_q${suf}`;
      out.push({ key: preKey, label: `Grad pre q=${q}`, color: strokeForKey(preKey) });
      out.push({ key: postKey, label: `Grad post q=${q}`, color: strokeForKey(postKey) });
    }
    return out;
  } catch {
    return [];
  }
}

/** Plot area alignment: same left Y width + right gutter on every chart so step x-lines line up vertically. */
function chartMargins(showLrOnMain: boolean) {
  const right = showLrOnMain ? 52 : 16;
  return { top: 10, right, bottom: 10, left: 8 } as const;
}

const CHART_Y_AXIS_WIDTH = 72;
const CHART_PLOT_LEFT_PX = 8 + CHART_Y_AXIS_WIDTH;

// Returns a solid but duller/darker version of an rgba color string for the trend overlay.
// Persisted, per-URL graph settings. Sliders + display toggles + which loss
// series are visible. Zoom / highlighted window is intentionally NOT persisted.
interface PersistedSettings {
  useLogScale: boolean;
  showTrend: boolean;
  smoothing: number;
  plotStride: number;
  clipOutliers: boolean;
  enabled: Record<string, boolean>;
}

// Key by the exact URL so each job remembers its own settings independently.
function settingsStorageKey(): string | null {
  if (typeof window === 'undefined') return null;
  return `jobLossGraph:${window.location.pathname}${window.location.search}`;
}

function dulledColor(rgba: string): string {
  const m = rgba.match(/rgba?\((\d+),(\d+),(\d+)/);
  if (!m) return 'rgba(120,120,120,1)';
  const r = Math.round(Number(m[1]) * 0.55);
  const g = Math.round(Number(m[2]) * 0.55);
  const b = Math.round(Number(m[3]) * 0.55);
  return `rgba(${r},${g},${b},1)`;
}

/** Optional training scalars stored in loss_log.db (see BaseSDTrainProcess / SDTrainer). */
const AUX_METRIC_OPTIONS = [
  { key: 'learning_rate', label: 'Learning rate', color: 'rgba(251,191,36,1)' },
  { key: 'weight_decay', label: 'Weight decay', color: 'rgba(52,211,153,1)' },
  { key: 'grad_norm_pre', label: 'Grad norm (before clip)', color: 'rgba(167,139,250,1)' },
  { key: 'grad_norm_post', label: 'Grad norm (after clip)', color: 'rgba(96,165,250,1)' },
] as const;

/** Logged when train.grad_norm_log_every > 1 (GPU bucket flush). */
const GRAD_AGG_METRIC_OPTIONS = [
  { key: 'grad_norm_pre_mean', label: 'Grad pre mean (bucket)', color: 'rgba(167,139,250,1)' },
  { key: 'grad_norm_pre_median', label: 'Grad pre median (bucket)', color: 'rgba(192,132,252,1)' },
  { key: 'grad_norm_clip_pct', label: 'Grad clipped % (bucket)', color: 'rgba(248,113,113,1)' },
  { key: 'grad_norm_post_mean', label: 'Grad post mean (bucket)', color: 'rgba(96,165,250,1)' },
] as const;

function formatAuxTooltipValue(key: string, v: number) {
  if (!Number.isFinite(v)) return '';
  if (key === 'learning_rate') return v.toExponential(2);
  if (key === 'weight_decay') return v.toExponential(2);
  if (key.includes('clip_pct')) return `${v.toFixed(2)}%`;
  return formatNum(v);
}

function downsampleAuxPoints(
  points: LossPoint[],
  stride: number,
): { step: number; value: number }[] {
  const s = Math.max(1, stride | 0);
  return points
    .filter(p => p.value !== null && Number.isFinite(p.value as number))
    .filter((_, i) => i % s === 0)
    .map(p => ({ step: p.step, value: p.value as number }));
}

function alignGradNormPairs(pre: LossPoint[], post: LossPoint[]) {
  const postByStep = new Map<number, number>();
  for (const p of post) {
    if (p.value !== null && Number.isFinite(p.value as number)) {
      postByStep.set(p.step, p.value as number);
    }
  }
  const out: { step: number; pre: number; post: number }[] = [];
  for (const p of pre) {
    if (p.value === null || !Number.isFinite(p.value as number)) continue;
    if (!postByStep.has(p.step)) continue;
    out.push({ step: p.step, pre: p.value as number, post: postByStep.get(p.step)! });
  }
  out.sort((a, b) => a.step - b.step);
  return out;
}

function computeGradNormChunkStats(
  pairs: { step: number; pre: number; post: number }[],
  chunkSize: number,
) {
  if (pairs.length === 0) return [];
  const c =
    !chunkSize || chunkSize <= 0
      ? pairs.length
      : Math.max(1, Math.min(Math.floor(chunkSize), pairs.length));
  const out: {
    chunk: number;
    stepStart: number;
    stepEnd: number;
    meanPre: number;
    clipPct: number;
  }[] = [];
  for (let i = 0, chunk = 0; i < pairs.length; i += c, chunk++) {
    const sl = pairs.slice(i, i + c);
    const meanPre = sl.reduce((s, p) => s + p.pre, 0) / sl.length;
    const clips = sl.filter(p => p.pre > p.post + 1e-9).length;
    out.push({
      chunk,
      stepStart: sl[0].step,
      stepEnd: sl[sl.length - 1].step,
      meanPre,
      clipPct: (100 * clips) / sl.length,
    });
  }
  return out;
}

export default function JobLossGraph({ job }: Props) {
  const [showAuxLr, setShowAuxLr] = useState(false);
  const [showAuxWd, setShowAuxWd] = useState(false);
  const [showAuxGradPre, setShowAuxGradPre] = useState(false);
  const [showAuxGradPost, setShowAuxGradPost] = useState(false);
  const [showGradAgg, setShowGradAgg] = useState(false);
  /** 0 = one chunk over all loaded per-step grad points (CPU stats). */
  const [gradNormChunkSize, setGradNormChunkSize] = useState(0);

  const gradPctCharts = useMemo(() => gradNormPercentileMetricCharts(job.job_config), [job.job_config]);

  const extraMetricKeys = useMemo(() => {
    const out: string[] = [];
    if (showAuxLr) out.push('learning_rate');
    if (showAuxWd) out.push('weight_decay');
    if (showAuxGradPre) out.push('grad_norm_pre');
    if (showAuxGradPost) out.push('grad_norm_post');
    if (showGradAgg) {
      for (const o of GRAD_AGG_METRIC_OPTIONS) {
        out.push(o.key);
      }
      for (const o of gradPctCharts) {
        out.push(o.key);
      }
    }
    return out;
  }, [showAuxLr, showAuxWd, showAuxGradPre, showAuxGradPost, showGradAgg, gradPctCharts]);

  const { series, lossKeys, status, refreshLoss } = useJobLossLog(job.id, 2000, extraMetricKeys);

  const syncMargins = useMemo(() => chartMargins(showAuxLr), [showAuxLr]);

  const auxChartsToShow = useMemo((): { key: string; label: string; color: string }[] => {
    const base = AUX_METRIC_OPTIONS.filter(opt => {
      if (opt.key === 'learning_rate') return false;
      if (opt.key === 'weight_decay') return showAuxWd;
      if (opt.key === 'grad_norm_pre') return showAuxGradPre;
      if (opt.key === 'grad_norm_post') return showAuxGradPost;
      return false;
    });
    const agg = showGradAgg ? [...GRAD_AGG_METRIC_OPTIONS, ...gradPctCharts] : [];
    return [...base, ...agg];
  }, [showAuxWd, showAuxGradPre, showAuxGradPost, showGradAgg, gradPctCharts]);

  const gradNormPairs = useMemo(
    () => alignGradNormPairs(series['grad_norm_pre'] ?? [], series['grad_norm_post'] ?? []),
    [series],
  );

  const gradChunkChartData = useMemo(
    () => computeGradNormChunkStats(gradNormPairs, gradNormChunkSize),
    [gradNormPairs, gradNormChunkSize],
  );

  // Controls
  const [useLogScale, setUseLogScale] = useState(false);
  const [showTrend, setShowTrend] = useState(true);

  // 0..100 slider. 100 = no smoothing, 0 = heavy smoothing.
  const [smoothing, setSmoothing] = useState(80);

  // UI-only downsample for rendering speed
  const [plotStride, setPlotStride] = useState(1);

  // show only last N points in the chart (0 = all)
  const [windowSize] = useState<number>(0);

  // quick y clipping for readability
  const [clipOutliers, setClipOutliers] = useState(false);

  // which loss series are enabled (default: all enabled)
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});

  const [isZoomed, setIsZoomed] = useState(false);

  // Gate persistence writes until we've loaded any stored settings, so the
  // initial defaults don't clobber what was saved before the load effect runs.
  const [hydrated, setHydrated] = useState(false);

  // Restored series selection, kept so the lossKeys-sync effect can honor it for
  // keys that arrive after load rather than falling back to the default.
  const persistedEnabledRef = useRef<Record<string, boolean> | null>(null);

  // Load persisted settings for this job's URL. Re-runs when the job changes
  // (navigating between jobs) so each URL restores its own saved state.
  useEffect(() => {
    setHydrated(false);
    persistedEnabledRef.current = null;
    const key = settingsStorageKey();
    if (!key) {
      setHydrated(true);
      return;
    }
    try {
      const raw = localStorage.getItem(key);
      if (raw) {
        const s = JSON.parse(raw) as Partial<PersistedSettings>;
        if (typeof s.useLogScale === 'boolean') setUseLogScale(s.useLogScale);
        if (typeof s.showTrend === 'boolean') setShowTrend(s.showTrend);
        if (typeof s.smoothing === 'number') setSmoothing(s.smoothing);
        if (typeof s.plotStride === 'number') setPlotStride(s.plotStride);
        if (typeof s.clipOutliers === 'boolean') setClipOutliers(s.clipOutliers);
        if (s.enabled && typeof s.enabled === 'object') {
          persistedEnabledRef.current = s.enabled;
          setEnabled(s.enabled);
        }
      }
    } catch {
      // ignore malformed / unavailable storage
    }
    setHydrated(true);
  }, [job.id]);

  // Persist settings whenever they change (after the initial load).
  useEffect(() => {
    if (!hydrated) return;
    const key = settingsStorageKey();
    if (!key) return;
    try {
      const payload: PersistedSettings = { useLogScale, showTrend, smoothing, plotStride, clipOutliers, enabled };
      localStorage.setItem(key, JSON.stringify(payload));
    } catch {
      // ignore unavailable storage
    }
  }, [hydrated, useLogScale, showTrend, smoothing, plotStride, clipOutliers, enabled]);

  // keep enabled map in sync with discovered keys. Only "loss/loss" is on by
  // default; every other metric starts deactivated (user can toggle it on).
  useEffect(() => {
    // Nothing discovered yet — don't prune, or we'd wipe a restored selection
    // before the keys have loaded.
    if (lossKeys.length === 0) return;
    setEnabled(prev => {
      const next = { ...prev };
      for (const k of lossKeys) {
        if (next[k] === undefined) next[k] = persistedEnabledRef.current?.[k] ?? k === 'loss/loss';
      }
      for (const k of Object.keys(next)) {
        if (!lossKeys.includes(k)) delete next[k];
      }
      return next;
    });
  }, [lossKeys]);

  const activeKeys = useMemo(() => lossKeys.filter(k => enabled[k] !== false), [lossKeys, enabled]);

  // Build uPlot-aligned data + series configs.
  const built = useMemo(() => {
    const stride = Math.max(1, plotStride | 0);
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98; // 1.0 -> 0.02
    const fullAlpha = 0.005;

    // Union of all steps across active series.
    const stepSet = new Set<number>();
    for (const key of activeKeys) {
      const pts: LossPoint[] = series[key] ?? [];
      for (const p of pts) {
        if (p.value === null || !Number.isFinite(p.value as number)) continue;
        if (useLogScale && (p.value as number) <= 0) continue;
        stepSet.add(p.step);
      }
    }
    let xs = Array.from(stepSet).sort((a, b) => a - b);
    if (stride > 1) xs = xs.filter((_, i) => i % stride === 0);
    if (windowSize > 0 && xs.length > windowSize) xs = xs.slice(xs.length - windowSize);

    const xsSet = new Set(xs);

    const data: (number[] | (number | null)[])[] = [xs];
    const seriesConfigs: uPlot.Series[] = [{}]; // x

    // Each metric gets its own y-scale (so unrelated magnitudes auto-range
    // independently) plus a matching colored axis.
    const scales: uPlot.Scales = { x: { time: false } };
    const axes: uPlot.Axis[] = [
      {
        stroke: 'rgba(255,255,255,0.55)',
        grid: { stroke: 'rgba(255,255,255,0.06)' },
        ticks: { stroke: 'rgba(255,255,255,0.15)' },
      },
    ];

    // Data columns belonging to each scale, for per-scale clip percentiles.
    const scaleArrays: Record<string, (number | null)[][]> = {};

    for (let ki = 0; ki < activeKeys.length; ki++) {
      const key = activeKeys[ki];
      const scaleKey = `y::${key}`;
      const pts: LossPoint[] = series[key] ?? [];
      const map = new Map<number, number>();
      for (const p of pts) {
        if (p.value === null || !Number.isFinite(p.value as number)) continue;
        if (useLogScale && (p.value as number) <= 0) continue;
        if (!xsSet.has(p.step)) continue;
        map.set(p.step, p.value as number);
      }
      const raw: (number | null)[] = xs.map(s => (map.has(s) ? (map.get(s) as number) : null));
      const smooth = emaWithNulls(raw, alpha);
      const fullSmooth = emaWithNulls(raw, fullAlpha);

      const color = strokeForKey(key);
      const colorDull = dulledColor(color);

      const colArrays: (number | null)[][] = [];

      // Main series: smoothed by the slider (slider at 100% = raw), always shown.
      data.push(smooth);
      seriesConfigs.push({
        label: key,
        scale: scaleKey,
        stroke: color,
        width: 2,
        spanGaps: false,
        points: { show: false },
        value: (_u, value) => formatNum(value),
      });
      colArrays.push(smooth);

      if (showTrend) {
        data.push(fullSmooth);
        seriesConfigs.push({
          label: `${key} (trend)`,
          scale: scaleKey,
          stroke: colorDull,
          width: 2.5,
          spanGaps: false,
          points: { show: false },
          value: (_u, value) => formatNum(value),
        });
        colArrays.push(fullSmooth);
      }

      scaleArrays[scaleKey] = colArrays;

      scales[scaleKey] = {
        distr: useLogScale ? 3 : 1,
        range: (_u, dataMin, dataMax) => {
          const c = yClipRef.current?.[scaleKey];
          if (c) return [c.min, c.max];
          return [dataMin, dataMax];
        },
      };

      axes.push({
        scale: scaleKey,
        side: ki % 2 === 0 ? 3 : 1, // alternate left / right
        stroke: color,
        label: key,
        labelSize: 14,
        // Only the first scale draws gridlines; overlaying grids from multiple
        // independent scales would be unreadable.
        grid: { show: ki === 0, stroke: 'rgba(255,255,255,0.06)' },
        ticks: { stroke: 'rgba(255,255,255,0.15)' },
        size: 60,
        values: (_u, ticks) => ticks.map(tk => formatNum(tk)),
      });
    }

    // Optional learning-rate overlay (separate y-scale, min–max in view, no tick labels).
    if (showAuxLr) {
      const lrPts: LossPoint[] = series['learning_rate'] ?? [];
      const lrMap = new Map<number, number>();
      for (const p of lrPts) {
        if (p.value === null || !Number.isFinite(p.value as number)) continue;
        if (!xsSet.has(p.step)) continue;
        lrMap.set(p.step, p.value as number);
      }
      const lrVals: (number | null)[] = xs.map(s => (lrMap.has(s) ? (lrMap.get(s) as number) : null));
      const finiteLr = lrVals.filter((v): v is number => v !== null && Number.isFinite(v));
      if (finiteLr.length > 0) {
        const lrScale = 'y::learning_rate';
        let lrMin = Math.min(...finiteLr);
        let lrMax = Math.max(...finiteLr);
        if (lrMin === lrMax) {
          const eps = Math.abs(lrMin) * 1e-6 || 1e-12;
          lrMin -= eps;
          lrMax += eps;
        }
        data.push(lrVals);
        seriesConfigs.push({
          label: 'Learning rate',
          scale: lrScale,
          stroke: 'rgba(251,191,36,0.38)',
          width: 1.75,
          spanGaps: false,
          points: { show: false },
          value: (_u, value) => formatAuxTooltipValue('learning_rate', value),
        });
        scales[lrScale] = {
          range: () => [lrMin, lrMax],
        };
        axes.push({
          scale: lrScale,
          side: 1,
          stroke: 'rgba(251,191,36,0.38)',
          grid: { show: false },
          ticks: { show: false },
          values: () => [],
          size: 8,
        });
      }
    }

    // y-domain clipping (2nd–98th percentile), computed per scale.
    let yClip: Record<string, { min: number; max: number }> | null = null;
    if (clipOutliers && xs.length >= 10) {
      yClip = {};
      for (const scaleKey of Object.keys(scaleArrays)) {
        const vals: number[] = [];
        for (const arr of scaleArrays[scaleKey]) {
          for (const v of arr) {
            if (v !== null && Number.isFinite(v)) vals.push(v as number);
          }
        }
        if (vals.length >= 10) {
          vals.sort((a, b) => a - b);
          const lo = vals[Math.floor(vals.length * 0.02)];
          const hi = vals[Math.ceil(vals.length * 0.98) - 1];
          if (Number.isFinite(lo) && Number.isFinite(hi) && lo !== hi) {
            yClip[scaleKey] = { min: lo, max: hi };
          }
        }
      }
    }

    return { data: data as uPlot.AlignedData, seriesConfigs, scales, axes, yClip };
  }, [series, activeKeys, smoothing, plotStride, windowSize, useLogScale, showTrend, clipOutliers, showAuxLr]);

  // Layout wrapper we measure for sizing — uPlot collapses its own mount node
  // to width:min-content, so we can't read sizes off it.
  const chartHostRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);

  // Latest per-scale yClip read by the y-scale range fns — kept current via effect.
  const yClipRef = useRef<Record<string, { min: number; max: number }> | null>(null);
  useEffect(() => {
    yClipRef.current = built.yClip;
  }, [built.yClip]);

  // Track zoom state via ref so the data-update effect can decide whether to refit scales.
  const isZoomedRef = useRef(false);
  useEffect(() => {
    isZoomedRef.current = isZoomed;
  }, [isZoomed]);

  // Structural recreate key — recreate uPlot only when the series shape or
  // axis distribution changes. Data updates go through setData.
  const hasData = (built.data[0]?.length ?? 0) > 1;
  const structuralKey = useMemo(
    () => `${activeKeys.join('|')}|trend=${showTrend}|log=${useLogScale}|lr=${showAuxLr}|has=${hasData}`,
    [activeKeys, showTrend, useLogScale, showAuxLr, hasData],
  );

  useEffect(() => {
    if (uplotRef.current) {
      uplotRef.current.destroy();
      uplotRef.current = null;
    }
    if (!containerRef.current || !chartHostRef.current) return;
    if (!hasData) return;

    const host = chartHostRef.current;
    const rect = host.getBoundingClientRect();
    const initialHeight = rect.height > 0 ? Math.max(MIN_CANVAS_HEIGHT, rect.height - 40) : FALLBACK_CANVAS_HEIGHT;
    const opts: uPlot.Options = {
      width: rect.width || 800,
      height: initialHeight,
      padding: [12, 16, 0, 4],
      series: built.seriesConfigs,
      scales: built.scales,
      axes: built.axes,
      cursor: {
        drag: { x: true, y: false, setScale: true },
        points: { size: 6 },
      },
      legend: { show: true },
      hooks: {
        setScale: [
          (u, key) => {
            if (key !== 'x') return;
            const xs = u.data[0] as number[];
            if (!xs || !xs.length) return;
            const sx = u.scales.x;
            const zoomed = sx.min !== xs[0] || sx.max !== xs[xs.length - 1];
            setIsZoomed(zoomed);
          },
        ],
      },
    };

    uplotRef.current = new uPlot(opts, built.data, containerRef.current);
    setIsZoomed(false);

    // Right-size the canvas against the legend height so it fills the remaining
    // vertical space. Defer to the next frame: the legend's height depends on
    // how many series wrap, and that layout isn't settled synchronously after
    // construction — measuring now would read a stale height (the bug that
    // previously required a manual resize to correct).
    const raf = requestAnimationFrame(() => {
      const u = uplotRef.current;
      if (!u) return;
      const fitted = computeCanvasSize(host);
      if (fitted) u.setSize(fitted);
    });

    return () => {
      cancelAnimationFrame(raf);
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structuralKey]);

  // Push new data without recreating — preserves zoom & cursor state.
  useEffect(() => {
    const u = uplotRef.current;
    if (!u) return;
    // When zoomed, pass resetScales=false so the user's view stays put. uPlot
    // skips its commit() in that branch though, so force a redraw to actually
    // re-render the new smoothed/strided values within the zoom window.
    if (isZoomedRef.current) {
      u.setData(built.data, false);
      u.redraw(true, true);
    } else {
      u.setData(built.data, true);
    }
  }, [built]);

  // Resize observer — fit canvas to the wrapper's available space minus the
  // HTML legend uPlot renders below it. Observe the layout wrapper, not the
  // uPlot mount node (which uPlot pins to width:min-content).
  // Re-runs on `hasData` because the observed element only exists once data
  // has loaded (see the `!hasData` branch below).
  useEffect(() => {
    const el = chartHostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const u = uplotRef.current;
      if (!u) return;
      const fitted = computeCanvasSize(el);
      if (fitted) u.setSize(fitted);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasData]);

  const handleResetZoom = useCallback(() => {
    const u = uplotRef.current;
    if (!u) return;
    const xs = u.data[0] as number[];
    if (!xs || !xs.length) return;
    u.setScale('x', { min: xs[0], max: xs[xs.length - 1] });
  }, []);

  const totalPoints = built.data[0]?.length ?? 0;

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col h-full">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-blue-400" />
          <h2 className="text-gray-100 text-sm font-medium">Loss graph</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && hasData && `${totalPoints.toLocaleString()} steps`}
            {status === 'success' && !hasData && 'No data yet'}
          </span>
        </div>

        <button
          type="button"
          onClick={refreshLoss}
          className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
        >
          Refresh
        </button>
      </div>

      {/* Loss chart; optional learning rate on a hidden-scale right axis */}
      <div className="px-4 pt-4">
        <div className="mb-3">
          <h3 className="text-xs font-medium text-gray-200 uppercase tracking-wide">Loss</h3>
          <p className="text-[11px] text-gray-500 mt-1">
            Log Y, smoothing, and outlier clipping apply to loss only. When enabled below, learning rate is drawn on the same
            chart with its own y-scale (min–max in view; no tick labels). Hover the curve for the exact value.
          </p>
        </div>
        <div className="bg-gray-950 rounded-lg border border-gray-800 h-96 relative select-none">
          {!hasData ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
              {status === 'error' ? 'Failed to load loss logs.' : 'Waiting for loss points...'}
            </div>
          ) : (
            <>
              {isZoomed && (
                <button
                  type="button"
                  onClick={handleResetZoom}
                  className="absolute top-2 right-2 z-10 px-2 py-1 rounded text-xs bg-blue-600/80 hover:bg-blue-600 text-white border border-blue-500/50"
                >
                  Reset zoom
                </button>
              )}
              <div ref={chartHostRef} className="absolute top-0 left-0 right-0 bottom-2 overflow-hidden">
                <div ref={containerRef} />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Training scalars (except LR): same plot width as loss for aligned step x */}
      <div className="px-4 pb-4 mt-6 pt-6 border-t border-gray-800">
        <h3 className="text-xs font-medium text-gray-200 uppercase tracking-wide">Training metrics</h3>
        <p className="text-[11px] text-gray-500 mt-1 mb-3">
          Each metric has its own y-axis (not shared with loss or with each other). Margins match the loss chart so a vertical
          line keeps the same training step. Use the Learning rate checkbox to overlay LR on the loss chart above.
        </p>
        <div className="flex flex-wrap gap-x-5 gap-y-2 mb-1">
          <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-gray-600 bg-gray-900 text-blue-500 focus:ring-blue-500/40"
              checked={showAuxLr}
              onChange={e => setShowAuxLr(e.target.checked)}
            />
            Learning rate
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-gray-600 bg-gray-900 text-blue-500 focus:ring-blue-500/40"
              checked={showAuxWd}
              onChange={e => setShowAuxWd(e.target.checked)}
            />
            Weight decay
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-gray-600 bg-gray-900 text-blue-500 focus:ring-blue-500/40"
              checked={showAuxGradPre}
              onChange={e => setShowAuxGradPre(e.target.checked)}
            />
            Grad norm (before clip)
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-gray-600 bg-gray-900 text-blue-500 focus:ring-blue-500/40"
              checked={showAuxGradPost}
              onChange={e => setShowAuxGradPost(e.target.checked)}
            />
            Grad norm (after clip)
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-gray-600 bg-gray-900 text-blue-500 focus:ring-blue-500/40"
              checked={showGradAgg}
              onChange={e => setShowGradAgg(e.target.checked)}
            />
            Grad bucket stats (GPU)
          </label>
        </div>
        <p className="text-[11px] text-gray-500 mb-3">
          Per-step grad norms are optional: enable Log grad norm statistics in the job Advanced tab. When grad norm log
          every is greater than 1, the trainer buffers norms on the GPU and logs bucket statistics (means, clip %, and
          optional quantiles from the Advanced bucket-quantiles field). Weight decay uses the first param group.
        </p>

        {showAuxGradPre && showAuxGradPost && gradNormPairs.length > 0 && (
          <div className="mb-6 p-3 rounded-lg border border-gray-800 bg-gray-950/80">
            <h4 className="text-xs font-medium text-gray-200 mb-1">Grad norm chunk statistics (CPU)</h4>
            <p className="text-[11px] text-gray-500 mb-2">
              For per-step grad logging (log every = 1). Chunk size 0 uses all loaded steps as one chunk. Larger values
              split consecutive optimizer steps into chunks; mean and clipped % are computed on the CPU for each chunk.
            </p>
            <NumberInput
              label="Chunk size (steps, 0 = all loaded)"
              value={gradNormChunkSize}
              onChange={value => setGradNormChunkSize(Math.max(0, Math.floor(Number(value) || 0)))}
              min={0}
              className="max-w-xs mb-3"
            />
            {gradChunkChartData.length === 0 ? null : (
              <div className="grid grid-cols-1 gap-3">
                <div className="h-40 flex flex-col border border-gray-800 rounded-md overflow-hidden bg-gray-900/40">
                  <div className="text-[10px] text-gray-500 px-2 py-1 border-b border-gray-800 shrink-0">
                    Mean pre-clip norm per chunk
                  </div>
                  <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={gradChunkChartData} margin={syncMargins}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="chunk" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }} />
                        <YAxis
                          tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                          width={CHART_Y_AXIS_WIDTH}
                        />
                        <Tooltip
                          contentStyle={{ background: 'rgba(17,24,39,0.96)', border: '1px solid #374151', fontSize: 11 }}
                          formatter={(v: unknown) => [
                            typeof v === 'number' && Number.isFinite(v) ? formatNum(v) : '',
                            'mean',
                          ]}
                          labelFormatter={(label, payload) => {
                            const row = payload?.[0]?.payload as
                              | { stepStart?: number; stepEnd?: number }
                              | undefined;
                            return row?.stepStart != null && row?.stepEnd != null
                              ? `steps ${row.stepStart}–${row.stepEnd}`
                              : `chunk ${label}`;
                          }}
                        />
                        <Bar dataKey="meanPre" fill="rgba(167,139,250,0.85)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="h-40 flex flex-col border border-gray-800 rounded-md overflow-hidden bg-gray-900/40">
                  <div className="text-[10px] text-gray-500 px-2 py-1 border-b border-gray-800 shrink-0">
                    Clipped % per chunk
                  </div>
                  <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={gradChunkChartData} margin={syncMargins}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="chunk" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }} />
                        <YAxis
                          tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                          width={CHART_Y_AXIS_WIDTH}
                          domain={[0, 100]}
                        />
                        <Tooltip
                          contentStyle={{ background: 'rgba(17,24,39,0.96)', border: '1px solid #374151', fontSize: 11 }}
                          formatter={(v: unknown) => [
                            typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(2)}%` : '',
                            'clipped',
                          ]}
                          labelFormatter={(label, payload) => {
                            const row = payload?.[0]?.payload as
                              | { stepStart?: number; stepEnd?: number }
                              | undefined;
                            return row?.stepStart != null && row?.stepEnd != null
                              ? `steps ${row.stepStart}–${row.stepEnd}`
                              : `chunk ${label}`;
                          }}
                        />
                        <Bar dataKey="clipPct" fill="rgba(248,113,113,0.85)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {auxChartsToShow.length > 0 && (
          <div className="grid grid-cols-1 gap-4">
            {auxChartsToShow.map(opt => {
              const pts = downsampleAuxPoints(series[opt.key] ?? [], plotStride);
              const hasAux = pts.length > 0;
              return (
                <div
                  key={opt.key}
                  className="bg-gray-950 rounded-lg border border-gray-800 overflow-hidden flex flex-col min-h-[220px]"
                >
                  <div className="px-3 py-2 border-b border-gray-800 bg-gray-900/80">
                    <span className="text-xs font-medium text-gray-200">{opt.label}</span>
                    <span className="block text-[10px] text-gray-500 mt-0.5">Own y-axis · step on x</span>
                  </div>
                  <div className="relative flex-1 h-52">
                    {!hasAux ? (
                      <div className="h-full w-full flex items-center justify-center text-xs text-gray-500">
                        {status === 'success' ? 'No data for this metric yet.' : 'Loading…'}
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={pts} margin={syncMargins}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                          <XAxis
                            dataKey="step"
                            tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                            tickLine={{ stroke: 'rgba(255,255,255,0.12)' }}
                            axisLine={{ stroke: 'rgba(255,255,255,0.12)' }}
                            minTickGap={40}
                          />
                          <YAxis
                            tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                            tickLine={{ stroke: 'rgba(255,255,255,0.12)' }}
                            axisLine={{ stroke: 'rgba(255,255,255,0.12)' }}
                            width={CHART_Y_AXIS_WIDTH}
                            tickFormatter={v => formatAuxTooltipValue(opt.key, Number(v))}
                            domain={['auto', 'auto']}
                          />
                          <Tooltip
                            contentStyle={{
                              background: 'rgba(17,24,39,0.96)',
                              border: '1px solid rgba(31,41,55,1)',
                              borderRadius: 10,
                              color: 'rgba(255,255,255,0.9)',
                              fontSize: 12,
                            }}
                            labelFormatter={(step: unknown) => `step ${step}`}
                            formatter={(value: unknown) => [
                              formatAuxTooltipValue(opt.key, Number(value ?? NaN)),
                              opt.label,
                            ]}
                          />
                          <Line
                            type="monotone"
                            dataKey="value"
                            name={opt.label}
                            stroke={opt.color}
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="px-4 pb-2 shrink-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Display</label>
            <div className="flex flex-wrap gap-2">
              <ToggleButton checked={showTrend} onClick={() => setShowTrend(v => !v)} label="Trend" />
              <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" />
              <ToggleButton checked={clipOutliers} onClick={() => setClipOutliers(v => !v)} label="Clip outliers" />
            </div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Series</label>
            {lossKeys.length === 0 ? (
              <div className="text-sm text-gray-400">No loss keys found yet.</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {lossKeys.map(k => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setEnabled(prev => ({ ...prev, [k]: !(prev[k] ?? true) }))}
                    className={[
                      'px-3 py-1 rounded-md text-xs border transition-colors',
                      enabled[k] === false
                        ? 'bg-gray-900 text-gray-400 border-gray-800 hover:bg-gray-800/60'
                        : 'bg-gray-900 text-gray-200 border-gray-800 hover:bg-gray-800/60',
                    ].join(' ')}
                    aria-pressed={enabled[k] !== false}
                    title={k}
                  >
                    <span className="inline-block h-2 w-2 rounded-full mr-2" style={{ background: strokeForKey(k) }} />
                    {k}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Smoothing</label>
              <span className="text-xs text-gray-300">{smoothing}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={smoothing}
              onChange={e => setSmoothing(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Plot stride</label>
              <span className="text-xs text-gray-300">every {plotStride} pt</span>
            </div>
            <input
              type="range"
              min={1}
              max={20}
              value={plotStride}
              onChange={e => setPlotStride(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="mt-2 text-[11px] text-gray-500">UI downsample for huge runs.</div>
          </div>
        </div>
      </div>

      <style jsx global>{`
        .uplot,
        .uplot * {
          font-family: inherit;
        }
        .uplot .u-legend {
          color: rgba(255, 255, 255, 0.85);
          font-size: 12px;
          margin-top: 4px;
        }
        .uplot .u-legend th,
        .uplot .u-legend td {
          color: rgba(255, 255, 255, 0.85);
        }
        .uplot .u-legend .u-marker {
          border-radius: 2px;
        }
        .uplot .u-select {
          background: rgba(59, 130, 246, 0.15);
          border: 1px solid rgba(59, 130, 246, 0.4);
        }
      `}</style>
    </div>
  );
}

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-3 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}
