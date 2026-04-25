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

interface Props {
  job: Job;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// EMA smoothing that works on a per-series list.
// alpha=1 -> no smoothing, alpha closer to 0 -> more smoothing.
function emaSmoothPoints(points: { step: number; value: number }[], alpha: number) {
  if (points.length === 0) return [];
  const a = clamp01(alpha);
  const out: { step: number; value: number }[] = new Array(points.length);

  let prev = points[0].value;
  out[0] = { step: points[0].step, value: prev };

  for (let i = 1; i < points.length; i++) {
    const x = points[i].value;
    prev = a * x + (1 - a) * prev;
    out[i] = { step: points[i].step, value: prev };
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

  // Controls
  const [useLogScale, setUseLogScale] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showSmoothed, setShowSmoothed] = useState(true);

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

  // keep enabled map in sync with discovered keys (enable new ones automatically)
  useEffect(() => {
    setEnabled(prev => {
      const next = { ...prev };
      for (const k of lossKeys) {
        if (next[k] === undefined) next[k] = true;
      }
      // drop removed keys
      for (const k of Object.keys(next)) {
        if (!lossKeys.includes(k)) delete next[k];
      }
      return next;
    });
  }, [lossKeys]);

  const activeKeys = useMemo(() => lossKeys.filter(k => enabled[k] !== false), [lossKeys, enabled]);

  // Zoom state for drag-to-zoom
  const [zoomLeft, setZoomLeft] = useState<number | null>(null);
  const [zoomRight, setZoomRight] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  // Selection tracked entirely in refs to avoid re-renders during drag
  const selectStartLabel = useRef<number | null>(null);
  const selectStartPx = useRef<number | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const chartWrapperRef = useRef<HTMLDivElement>(null);

  const perSeries = useMemo(() => {
    // Build per-series processed point arrays (raw + smoothed + fullSmooth), then merge by step for charting.
    const stride = Math.max(1, plotStride | 0);

    // smoothing%: 0 => no smoothing (alpha=1.0), 100 => heavy smoothing (alpha=0.02)
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98; // 1.0 -> 0.02

    // Full smoothing overlay: always max smoothing (alpha=0.02)
    const fullAlpha = 0.005;

    const out: Record<string, { raw: { step: number; value: number }[]; smooth: { step: number; value: number }[]; fullSmooth: { step: number; value: number }[] }> =
      {};

    for (const key of activeKeys) {
      const pts: LossPoint[] = series[key] ?? [];

      let raw = pts
        .filter(p => p.value !== null && Number.isFinite(p.value as number))
        .map(p => ({ step: p.step, value: p.value as number }))
        .filter(p => (useLogScale ? p.value > 0 : true))
        .filter((_, idx) => idx % stride === 0);

      // windowing (applies after stride)
      if (windowSize > 0 && raw.length > windowSize) {
        raw = raw.slice(raw.length - windowSize);
      }

      const smooth = emaSmoothPoints(raw, alpha);
      const fullSmooth = emaSmoothPoints(raw, fullAlpha);

      out[key] = { raw, smooth, fullSmooth };
    }

    return out;
  }, [series, activeKeys, smoothing, plotStride, windowSize, useLogScale]);

  const chartDataLossOnly = useMemo(() => {
    // Merge series into one array of objects keyed by step.
    // Fields: `${key}__raw` and `${key}__smooth`
    const map = new Map<number, any>();

    for (const key of activeKeys) {
      const s = perSeries[key];
      if (!s) continue;

      for (const p of s.raw) {
        const row = map.get(p.step) ?? { step: p.step };
        row[`${key}__raw`] = p.value;
        map.set(p.step, row);
      }
      for (const p of s.smooth) {
        const row = map.get(p.step) ?? { step: p.step };
        row[`${key}__smooth`] = p.value;
        map.set(p.step, row);
      }
      for (const p of s.fullSmooth) {
        const row = map.get(p.step) ?? { step: p.step };
        row[`${key}__fullsmooth`] = p.value;
        map.set(p.step, row);
      }
    }

    const arr = Array.from(map.values());
    arr.sort((a, b) => a.step - b.step);
    return arr;
  }, [activeKeys, perSeries]);

  const lrByStepDownsampled = useMemo(() => {
    if (!showAuxLr) return new Map<number, number>();
    const pts = downsampleAuxPoints(series['learning_rate'] ?? [], plotStride);
    return new Map(pts.map(p => [p.step, p.value]));
  }, [showAuxLr, series, plotStride]);

  const chartData = useMemo(() => {
    if (!showAuxLr) return chartDataLossOnly;
    return chartDataLossOnly.map(row => ({
      ...row,
      learning_rate: lrByStepDownsampled.has(row.step) ? lrByStepDownsampled.get(row.step)! : null,
    }));
  }, [chartDataLossOnly, showAuxLr, lrByStepDownsampled]);

  // Zoomed slice of chartData
  const visibleData = useMemo(() => {
    if (zoomLeft == null || zoomRight == null) return chartData;
    const lo = Math.min(zoomLeft, zoomRight);
    const hi = Math.max(zoomLeft, zoomRight);
    return chartData.filter(d => d.step >= lo && d.step <= hi);
  }, [chartData, zoomLeft, zoomRight]);

  const syncMargins = chartMargins(showAuxLr);

  const lrYDomain = useMemo((): [number, number] => {
    if (!showAuxLr) return [0, 1];
    const vals = visibleData
      .map(d => d.learning_rate)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    if (vals.length === 0) return [0, 1];
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    if (lo === hi) {
      const eps = Math.abs(lo) * 1e-6 || 1e-12;
      lo -= eps;
      hi += eps;
    }
    return [lo, hi];
  }, [showAuxLr, visibleData]);

  const hasLrInView =
    showAuxLr && visibleData.some(d => typeof d.learning_rate === 'number' && Number.isFinite(d.learning_rate));

  // Convert a pixel x within the wrapper to a fractional position [0,1] across the plot area
  const pxToFraction = useCallback((clientX: number) => {
    const wrapper = chartWrapperRef.current;
    if (!wrapper) return 0;
    const rect = wrapper.getBoundingClientRect();
    const m = chartMargins(showAuxLr);
    const plotLeft = CHART_PLOT_LEFT_PX;
    const plotRight = rect.width - m.right;
    const plotWidth = plotRight - plotLeft;
    const localX = clientX - rect.left;
    return Math.max(0, Math.min(1, (localX - plotLeft) / plotWidth));
  }, [showAuxLr]);

  const fractionToStep = useCallback((frac: number) => {
    const data = visibleData;
    if (data.length === 0) return 0;
    const idx = Math.round(frac * (data.length - 1));
    return data[Math.max(0, Math.min(data.length - 1, idx))].step;
  }, [visibleData]);

  // Native DOM events for drag selection
  useEffect(() => {
    const wrapper = chartWrapperRef.current;
    if (!wrapper) return;

    const onDown = (e: MouseEvent) => {
      selectStartPx.current = e.clientX;
      selectStartLabel.current = fractionToStep(pxToFraction(e.clientX));
      setIsDragging(true);
      if (overlayRef.current) overlayRef.current.style.display = 'none';
    };

    const onMove = (e: MouseEvent) => {
      if (selectStartPx.current == null) return;
      const rect = wrapper.getBoundingClientRect();
      const m = chartMargins(showAuxLr);
      const plotLeft = CHART_PLOT_LEFT_PX;
      const plotRight = rect.width - m.right;
      const startLocal = selectStartPx.current - rect.left;
      const curLocal = e.clientX - rect.left;
      // Clamp to plot area
      const clampedStart = Math.max(plotLeft, Math.min(plotRight, startLocal));
      const clampedCur = Math.max(plotLeft, Math.min(plotRight, curLocal));
      const left = Math.min(clampedStart, clampedCur);
      const width = Math.abs(clampedCur - clampedStart);
      if (overlayRef.current) {
        overlayRef.current.style.display = width > 3 ? 'block' : 'none';
        overlayRef.current.style.left = `${left}px`;
        overlayRef.current.style.width = `${width}px`;
      }
    };

    const onUp = (e: MouseEvent) => {
      if (selectStartPx.current == null) return;
      const startStep = selectStartLabel.current!;
      const endStep = fractionToStep(pxToFraction(e.clientX));
      selectStartPx.current = null;
      selectStartLabel.current = null;
      setIsDragging(false);
      if (overlayRef.current) overlayRef.current.style.display = 'none';
      if (startStep !== endStep) {
        setZoomLeft(Math.min(startStep, endStep));
        setZoomRight(Math.max(startStep, endStep));
      }
    };

    wrapper.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      wrapper.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [pxToFraction, fractionToStep, showAuxLr]);

  const handleResetZoom = useCallback(() => {
    setZoomLeft(null);
    setZoomRight(null);
  }, []);

  const hasData = chartData.length > 1;
  const isZoomed = zoomLeft != null && zoomRight != null;

  const auxChartsToShow = useMemo(() => {
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

  const yDomain = useMemo((): [number | 'auto', number | 'auto'] => {
    if (!clipOutliers || chartData.length < 10) return ['auto', 'auto'];

    // Collect visible values (prefer smoothed if shown, else raw)
    const vals: number[] = [];
    for (const row of chartData) {
      for (const key of activeKeys) {
        const k = showSmoothed ? `${key}__smooth` : `${key}__raw`;
        const v = row[k];
        if (typeof v === 'number' && Number.isFinite(v)) vals.push(v);
      }
    }
    if (vals.length < 10) return ['auto', 'auto'];

    vals.sort((a, b) => a - b);
    const lo = vals[Math.floor(vals.length * 0.02)];
    const hi = vals[Math.ceil(vals.length * 0.98) - 1];

    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return ['auto', 'auto'];
    return [lo, hi];
  }, [clipOutliers, chartData, activeKeys, showSmoothed]);

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-blue-400" />
          <h2 className="text-gray-100 text-sm font-medium">Loss graph</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && hasData && `${chartData.length.toLocaleString()} steps`}
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
        <div ref={chartWrapperRef} className="bg-gray-950 rounded-lg border border-gray-800 h-96 relative select-none">
          {/* Drag selection overlay — positioned via refs, no re-renders */}
          <div
            ref={overlayRef}
            style={{ display: 'none', position: 'absolute', top: 10, bottom: 10, pointerEvents: 'none', background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.4)', zIndex: 5 }}
          />
          {!hasData ? (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
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
            <ResponsiveContainer width="100%" height="100%" style={isDragging ? { pointerEvents: 'none' } : undefined}>
              <LineChart data={visibleData} margin={syncMargins}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  minTickGap={40}
                />
                <YAxis
                  yAxisId="loss"
                  scale={useLogScale ? 'log' : 'linear'}
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  width={CHART_Y_AXIS_WIDTH}
                  tickFormatter={formatNum}
                  domain={yDomain}
                  allowDataOverflow={clipOutliers}
                />
                {showAuxLr && hasLrInView && (
                  <YAxis
                    yAxisId="lr"
                    orientation="right"
                    domain={lrYDomain}
                    tick={false}
                    axisLine={false}
                    tickLine={false}
                    width={8}
                    allowDataOverflow
                  />
                )}
                {!isDragging && (
                  <Tooltip
                    cursor={{ stroke: 'rgba(59,130,246,0.25)', strokeWidth: 1 }}
                    contentStyle={{
                      background: 'rgba(17,24,39,0.96)',
                      border: '1px solid rgba(31,41,55,1)',
                      borderRadius: 10,
                      color: 'rgba(255,255,255,0.9)',
                      fontSize: 12,
                    }}
                    labelStyle={{ color: 'rgba(255,255,255,0.75)' }}
                    labelFormatter={(label: any) => `step ${label}`}
                    formatter={(value: any, name: any) => {
                      const n = String(name);
                      if (n === 'Learning rate' || n.includes('learning_rate')) {
                        return [formatAuxTooltipValue('learning_rate', Number(value)), 'Learning rate'];
                      }
                      return [formatNum(Number(value)), name];
                    }}
                  />
                )}

                <Legend
                  wrapperStyle={{
                    paddingTop: 8,
                    color: 'rgba(255,255,255,0.7)',
                    fontSize: 12,
                  }}
                />

                {/* Raw lines */}
                {showRaw && activeKeys.map(k => (
                  <Line
                    key={`${k}__raw`}
                    yAxisId="loss"
                    type="monotone"
                    dataKey={`${k}__raw`}
                    name={`${k} (raw)`}
                    stroke={strokeForKey(k).replace('1)', '0.40)')}
                    strokeWidth={1.25}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                {/* Smoothed lines */}
                {showSmoothed && activeKeys.map(k => (
                  <Line
                    key={`${k}__smooth`}
                    yAxisId="loss"
                    type="monotone"
                    dataKey={`${k}__smooth`}
                    name={`${k}`}
                    stroke={strokeForKey(k)}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                {/* Full-smooth trend overlay — hidden from legend/tooltip */}
                {activeKeys.map(k => (
                  <Line
                    key={`${k}__fullsmooth`}
                    yAxisId="loss"
                    type="monotone"
                    dataKey={`${k}__fullsmooth`}
                    name={`${k}__fullsmooth`}
                    stroke={dulledColor(strokeForKey(k))}
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                    legendType="none"
                    tooltipType="none"
                  />
                ))}
                {showAuxLr && hasLrInView && (
                  <Line
                    yAxisId="lr"
                    type="monotone"
                    dataKey="learning_rate"
                    name="Learning rate"
                    stroke="rgba(251,191,36,0.38)"
                    strokeWidth={1.75}
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                )}

              </LineChart>
            </ResponsiveContainer>
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
                          formatter={(v: number | undefined) => [
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
                          formatter={(v: number | undefined) => [
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
                            labelFormatter={(step: number) => `step ${step}`}
                            formatter={(value: number | undefined) => [
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
      <div className="px-4 pb-2">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Display</label>
            <div className="flex flex-wrap gap-2">
              <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smoothed" />
              <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" />
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
              disabled={!showSmoothed}
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
