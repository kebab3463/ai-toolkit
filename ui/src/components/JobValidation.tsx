'use client';

import { Job } from '@prisma/client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { encodeFilePathForUrl } from '@/utils/basic';

// Results of extensions/turbo_lora/loss_validation.py (one event per validation step).
interface Row {
  image: string;
  prompt?: string; // prompt variant (older results: none)
  resolution: number;
  seed: number;
  sigma: number;
  loss: number;
}
interface ValidationEvent {
  step: number;
  time: number;
  resolutions: number[];
  seeds: number[];
  sigmas: number[];
  images: string[];
  prompts?: string[];
  prompt_texts?: Record<string, Record<string, string>>;
  image_paths?: Record<string, string>;
  sizes?: Record<string, [number, number]>;
  rows: Row[];
}

type CompareMode = 'absolute' | 'delta' | 'percent';
const ALL = 'all';

interface Cell {
  mean: number;
  values: number[]; // per seed, each averaged over the selected sigmas
}

function mean(values: number[]) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;
}

function std(values: number[]) {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((a, b) => a + (b - m) * (b - m), 0) / (values.length - 1));
}

const DEFAULT_VARIANT = 'default';

// cell key: image|prompt|resolution, with "all" for a dimension averaged over
const key = (image: string, prompt: string, res: number | string) => `${image}|${prompt}|${res}`;

// every combination of image / prompt variant / resolution, each also averaged over ("all");
// every average is over seeds (and the selected sigmas, and whatever is "all")
function summarize(event: ValidationEvent, sigma: string): Map<string, Cell> {
  const rows = sigma === ALL ? event.rows : event.rows.filter(r => String(r.sigma) === sigma);
  const groups = new Map<string, Map<number, number[]>>();
  const add = (k: string, row: Row) => {
    let bySeed = groups.get(k);
    if (!bySeed) groups.set(k, (bySeed = new Map()));
    let list = bySeed.get(row.seed);
    if (!list) bySeed.set(row.seed, (list = []));
    list.push(row.loss);
  };
  for (const r of rows) {
    const dims = [r.image, r.prompt ?? DEFAULT_VARIANT, String(r.resolution)];
    for (let m = 0; m < 8; m++) {
      add(key(...(dims.map((d, i) => (m & (1 << i) ? ALL : d)) as [string, string, string])), r);
    }
  }
  const out = new Map<string, Cell>();
  groups.forEach((bySeed, key) => {
    const perSeed = Array.from(bySeed.keys())
      .sort((a, b) => a - b)
      .map(seed => mean(bySeed.get(seed)!));
    out.set(key, { mean: mean(perSeed), values: perSeed });
  });
  return out;
}

function fmt(v: number, mode: CompareMode) {
  if (!Number.isFinite(v)) return '–';
  if (mode === 'percent') return `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
  if (mode === 'delta') return `${v > 0 ? '+' : ''}${v.toPrecision(3)}`;
  return v.toPrecision(4);
}

function compare(value: number, base: number | undefined, mode: CompareMode) {
  if (mode === 'absolute' || base === undefined) return value;
  if (mode === 'delta') return value - base;
  return ((value - base) / base) * 100;
}

// lower loss = better: green for decreases, red for increases, scaled to the table's largest change
function deltaColor(v: number, maxAbs: number) {
  if (!Number.isFinite(v) || maxAbs <= 0) return undefined;
  const a = Math.min(1, Math.abs(v) / maxAbs) * 0.45;
  return v < 0 ? `rgba(34, 197, 94, ${a})` : `rgba(239, 68, 68, ${a})`;
}

const TRAIN_KEY = 'train';

// noise-level stepper: index 0 = average over all noise levels, 1..n = one sigma each;
// one shared index drives all three panels
function NoiseStepper({
  index,
  sigmas,
  onStep,
}: {
  index: number;
  sigmas: number[];
  onStep: (dir: -1 | 1) => void;
}) {
  const btn =
    'p-1.5 rounded-md text-gray-300 hover:bg-gray-700 disabled:opacity-30 disabled:hover:bg-transparent disabled:cursor-default';
  return (
    <div
      className="ml-auto flex items-center gap-1 rounded-md px-1"
      title="Noise level, shared by all panels (← / → keys)"
    >
      <button className={btn} disabled={index <= 0} onClick={() => onStep(-1)} aria-label="Previous noise level">
        <FaChevronLeft className="text-xs" />
      </button>
      <span className="text-xs text-gray-200 tabular-nums min-w-[9.5rem] text-center">
        {index === 0 ? 'All noise levels (mean)' : `σ = ${sigmas[index - 1]}`}
        <span className="text-gray-500"> · {index}/{sigmas.length}</span>
      </span>
      <button
        className={btn}
        disabled={index >= sigmas.length}
        onClick={() => onStep(1)}
        aria-label="Next noise level"
      >
        <FaChevronRight className="text-xs" />
      </button>
    </div>
  );
}

const LINE_COLORS = ['#60a5fa', '#f59e0b', '#a78bfa', '#34d399', '#f472b6', '#22d3ee', '#fb923c'];

export default function JobValidation({ job }: { job: Job }) {
  const [events, setEvents] = useState<ValidationEvent[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [pickedStep, setPickedStep] = useState<number | null>(null); // null: follow the latest
  const [mode, setMode] = useState<CompareMode>('absolute');
  const [noiseIndex, setNoiseIndex] = useState(0);
  const [trainPoints, setTrainPoints] = useState<{ step: number; value: number }[]>([]);
  const trainKeyRef = useRef<string | null>(null);
  const trainSinceRef = useRef<number | null>(null);
  // checkpoint table: stays scrolled to the newest step unless the user scrolled up
  const ckptScrollRef = useRef<HTMLDivElement | null>(null);
  const ckptPinnedRef = useRef(true);

  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${job.id}/validation`)
        .then(res => {
          setEvents(res.data?.events ?? []);
          setLoaded(true);
        })
        .catch(() => setLoaded(true)),
    10000,
    [job.id],
  );

  // training loss (loss_log.db), fetched incrementally, for the chart's per-interval averages
  usePollLoop(
    async () => {
      try {
        if (trainKeyRef.current === null) {
          const res = await apiClient.get(`/api/jobs/${job.id}/loss`, { params: { limit: 1 } });
          const keys: string[] = res.data?.keys ?? [];
          trainKeyRef.current = keys.includes('loss/loss') ? 'loss/loss' : keys.includes('loss') ? 'loss' : '';
        }
        if (!trainKeyRef.current) return;
        const params: Record<string, string | number> = { key: trainKeyRef.current, limit: 20000 };
        if (trainSinceRef.current !== null) params.since_step = trainSinceRef.current;
        const res = await apiClient.get(`/api/jobs/${job.id}/loss`, { params });
        const pts = ((res.data?.points ?? []) as { step: number; value: number | null }[]).filter(
          p => p.value !== null && Number.isFinite(p.value),
        ) as { step: number; value: number }[];
        if (pts.length) {
          trainSinceRef.current = pts[pts.length - 1].step;
          setTrainPoints(prev => [...prev, ...pts]);
        }
      } catch {
        // no loss log yet
      }
    },
    10000,
    [job.id],
  );

  const latest = events.length ? events[events.length - 1] : null;
  const event = (pickedStep !== null && events.find(e => e.step === pickedStep)) || latest;
  const baseline = events.length ? events[0] : null;
  const sigmas = latest ? latest.sigmas : [];
  const noise = Math.min(noiseIndex, sigmas.length);
  const stepNoise = (dir: -1 | 1) => setNoiseIndex(prev => Math.max(0, Math.min(sigmas.length, prev + dir)));
  const sigma = noise === 0 ? ALL : String(sigmas[noise - 1]);
  const ckptSigma = sigma;
  const chartSigma = sigma;

  // ← / → move the noise level of all panels; not while typing
  const stepNoiseRef = useRef(stepNoise);
  stepNoiseRef.current = stepNoise;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(t.tagName))) return;
      e.preventDefault();
      stepNoiseRef.current(e.key === 'ArrowLeft' ? -1 : 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const cells = useMemo(() => (event ? summarize(event, sigma) : new Map<string, Cell>()), [event, sigma]);
  const baseCells = useMemo(
    () => (baseline ? summarize(baseline, sigma) : new Map<string, Cell>()),
    [baseline, sigma],
  );

  const shown = (key: string) => {
    const c = cells.get(key);
    if (!c) return NaN;
    return compare(c.mean, baseCells.get(key)?.mean, mode);
  };

  const maxAbs = useMemo(() => {
    if (mode === 'absolute' || !event) return 0;
    let m = 0;
    cells.forEach((_, key) => {
      const v = shown(key);
      if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
    });
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cells, baseCells, mode, event]);

  const chartData = useMemo(() => {
    const base = baseline ? summarize(baseline, chartSigma) : null;
    return events.map((e, i) => {
      const s = summarize(e, chartSigma);
      const point: Record<string, number> = { step: e.step };
      const keys = [
        key(ALL, ALL, ALL),
        ...e.resolutions.map(r => key(ALL, ALL, r)),
        ...((e.prompts?.length ?? 0) > 1 ? e.prompts!.map(p => key(ALL, p, ALL)) : []),
      ];
      for (const key of keys) {
        const c = s.get(key);
        if (c) point[key] = compare(c.mean, base?.get(key)?.mean, mode);
      }
      // mean training loss over the steps since the previous validation (none before the first)
      if (i > 0) {
        const from = events[i - 1].step;
        const vals = trainPoints.filter(p => p.step > from && p.step <= e.step).map(p => p.value);
        if (vals.length) point[TRAIN_KEY] = mean(vals);
      }
      return point;
    });
  }, [events, baseline, chartSigma, mode, trainPoints]);
  const hasTrain = chartData.some(p => p[TRAIN_KEY] !== undefined);

  // checkpoint table: one row per step, one column per average; the lowest loss per column is marked
  const stepSummaries = useMemo(() => events.map(e => summarize(e, ckptSigma)), [events, ckptSigma]);
  const columnGroups = useMemo(() => {
    type Group = { label: string; cols: { k: string; label: string; title?: string }[] };
    if (!latest) return [] as Group[];
    const groups: Group[] = [
      { label: 'Overall', cols: [{ k: key(ALL, ALL, ALL), label: 'All' }] },
      { label: 'Resolution', cols: latest.resolutions.map(r => ({ k: key(ALL, ALL, r), label: String(r) })) },
    ];
    if ((latest.prompts?.length ?? 0) > 1) {
      groups.push({ label: 'Prompt', cols: latest.prompts!.map(p => ({ k: key(ALL, p, ALL), label: p })) });
    }
    groups.push({
      label: 'Image',
      cols: latest.images.map(name => ({ k: key(name, ALL, ALL), label: name, title: name })),
    });
    return groups;
  }, [latest]);
  const columnMin = useMemo(() => {
    const out = new Map<string, number>();
    for (const g of columnGroups) {
      for (const c of g.cols) {
        let best = Infinity;
        for (const s of stepSummaries) {
          const v = s.get(c.k)?.mean;
          if (v !== undefined && v < best) best = v;
        }
        out.set(c.k, best);
      }
    }
    return out;
  }, [columnGroups, stepSummaries]);

  useEffect(() => {
    const el = ckptScrollRef.current;
    if (el && ckptPinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [events.length, loaded]);

  if (!loaded) return <p className="text-sm text-gray-400">Loading validation results…</p>;
  if (!event) {
    return (
      <p className="text-sm text-gray-400">
        No validation results yet. Add a <code>loss_validation:</code> block to the job config (see
        extensions/turbo_lora/README.md).
      </p>
    );
  }

  const variants = event.prompts && event.prompts.length > 1 ? event.prompts : [];

  const tooltip = (k: string, size?: [number, number]) => {
    const c = cells.get(k);
    if (!c) return '';
    const parts = [
      `mean ${c.mean.toPrecision(5)} over ${c.values.length} seed${c.values.length === 1 ? '' : 's'}`,
      `seed std ${std(c.values).toPrecision(3)}`,
      `per seed: ${c.values.map(v => v.toPrecision(4)).join(', ')}`,
    ];
    const b = baseCells.get(k);
    if (b && baseline && baseline.step !== event.step) parts.push(`step ${baseline.step}: ${b.mean.toPrecision(5)}`);
    if (size) parts.push(`${size[0]}×${size[1]} px`);
    return parts.join('\n');
  };

  const cellClass = 'px-3 py-2 text-right font-mono text-xs tabular-nums';
  const cell = (k: string, extra = '', size?: [number, number]) => {
    const v = shown(k);
    return (
      <td
        key={k}
        className={`${cellClass} ${extra}`}
        title={tooltip(k, size)}
        style={{ backgroundColor: mode === 'absolute' ? undefined : deltaColor(v, maxAbs) }}
      >
        {fmt(v, mode)}
      </td>
    );
  };

  const selectClass =
    'bg-gray-800 border border-gray-700 rounded-md text-xs text-gray-200 px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500';

  return (
    <div className="space-y-6">
      <div
        className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800"
      >
        <div className="bg-gray-800 px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-2">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-green-400" />
            <h2 className="text-gray-100 text-sm font-medium">Validation loss</h2>
            <span className="text-xs text-gray-400">
              {event.images.length} images ×{' '}
              {variants.length ? `${variants.length} prompt versions × ` : ''}
              {event.resolutions.length} resolutions × {event.seeds.length} seeds × {event.sigmas.length} noise levels
            </span>
          </div>
          <label className="flex items-center gap-2 text-xs text-gray-300">
            Step
            <select
              className={selectClass}
              value={pickedStep === null ? 'latest' : String(pickedStep)}
              onChange={e => setPickedStep(e.target.value === 'latest' ? null : Number(e.target.value))}
            >
              <option value="latest">Latest ({latest?.step})</option>
              {[...events].reverse().map(e => (
                <option key={e.step} value={e.step}>
                  {e.step}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-300">
            Show
            <select className={selectClass} value={mode} onChange={e => setMode(e.target.value as CompareMode)}>
              <option value="absolute">Loss</option>
              <option value="delta">Change vs step {baseline?.step}</option>
              <option value="percent">% change vs step {baseline?.step}</option>
            </select>
          </label>
          <NoiseStepper
            index={noise}
            sigmas={sigmas}
            onStep={stepNoise}
          />
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-400">
                <th className="px-3 py-2 text-left font-medium">Image</th>
                {event.resolutions.map(r => (
                  <th key={r} className="px-3 py-2 text-right font-medium">
                    {r}
                  </th>
                ))}
                <th className="px-3 py-2 text-right font-medium border-l border-gray-800">All resolutions</th>
              </tr>
            </thead>
            <tbody>
              {event.images.map(name => {
                const imgPath = event.image_paths?.[name];
                return [
                  <tr key={name} className="border-b border-gray-800/60 hover:bg-gray-800/40">
                    <td className="px-3 py-1.5 text-gray-200 text-xs">
                      <div className="flex items-center gap-2">
                        {imgPath && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={`/api/img/${encodeFilePathForUrl(imgPath)}?thumb=1`}
                            alt=""
                            className="h-8 w-8 object-cover rounded"
                            loading="lazy"
                          />
                        )}
                        <span className="truncate max-w-[16rem]" title={name}>
                          {name}
                        </span>
                        {variants.length > 0 && <span className="text-gray-500">(all prompts)</span>}
                      </div>
                    </td>
                    {event.resolutions.map(r => cell(key(name, ALL, r), '', event.sizes?.[`${name}@${r}`]))}
                    {cell(key(name, ALL, ALL), 'border-l border-gray-800 text-gray-100')}
                  </tr>,
                  ...variants.map(v => (
                    <tr key={`${name}|${v}`} className="border-b border-gray-800/30 hover:bg-gray-800/40 text-gray-400">
                      <td className="pl-14 pr-3 py-1 text-xs" title={event.prompt_texts?.[name]?.[v] ?? ''}>
                        prompt {v}
                      </td>
                      {event.resolutions.map(r => cell(key(name, v, r), 'py-1', event.sizes?.[`${name}@${r}`]))}
                      {cell(key(name, v, ALL), 'py-1 border-l border-gray-800')}
                    </tr>
                  )),
                ];
              })}
            </tbody>
            <tfoot>
              <tr className="border-t-2 border-gray-700 bg-gray-800/40">
                <td className="px-3 py-2 text-xs font-medium text-gray-200">
                  All images{variants.length > 0 && <span className="text-gray-500 font-normal"> (all prompts)</span>}
                </td>
                {event.resolutions.map(r => cell(key(ALL, ALL, r), 'text-gray-100'))}
                {cell(key(ALL, ALL, ALL), 'border-l border-gray-800 font-semibold text-white')}
              </tr>
              {variants.map(v => (
                <tr key={`all|${v}`} className="bg-gray-800/20 text-gray-300">
                  <td className="pl-14 pr-3 py-1.5 text-xs">prompt {v}</td>
                  {event.resolutions.map(r => cell(key(ALL, v, r), 'py-1.5'))}
                  {cell(key(ALL, v, ALL), 'py-1.5 border-l border-gray-800 font-medium text-gray-100')}
                </tr>
              ))}
            </tfoot>
          </table>
        </div>
        <p className="px-4 py-2 text-[11px] text-gray-500 border-t border-gray-800">
          Step {event.step}. Each cell averages over seeds{sigma === ALL ? ' and noise levels' : ` at σ = ${sigma}`}
          {variants.length ? ' (and prompt versions in the "all prompts" rows)' : ''}; hover for the per-seed values
          {variants.length ? ' and, on a prompt label, its text' : ''}. Masked training loss (no preservation terms),
          LoRA at weight 1.
        </p>
      </div>

      <div
        className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800"
      >
        <div className="bg-gray-800 px-4 py-3 flex flex-wrap items-center gap-x-3 gap-y-2">
          <h3 className="text-gray-100 text-sm font-medium">Checkpoints</h3>
          <span className="text-xs text-gray-400">
            Lowest loss per column in bold green. Click a row to show that step above.
          </span>
          <NoiseStepper
            index={noise}
            sigmas={sigmas}
            onStep={stepNoise}
          />
        </div>
        <div
          className="overflow-auto max-h-[32rem]"
          ref={ckptScrollRef}
          onScroll={e => {
            const el = e.currentTarget;
            ckptPinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          <table className="min-w-full text-sm">
            <thead className="sticky top-0 bg-gray-900 z-10">
              <tr className="text-[11px] text-gray-500 uppercase tracking-wide">
                <th className="px-3 pt-2" />
                {columnGroups.map(g => (
                  <th
                    key={g.label}
                    colSpan={g.cols.length}
                    className="px-3 pt-2 text-center font-medium border-l border-gray-800"
                  >
                    {g.label}
                  </th>
                ))}
              </tr>
              <tr className="border-b border-gray-800 text-xs text-gray-400">
                <th className="px-3 py-2 text-left font-medium">Step</th>
                {columnGroups.map(g =>
                  g.cols.map((c, i) => (
                    <th
                      key={c.k}
                      title={c.title}
                      className={`px-3 py-2 text-right font-medium max-w-[8rem] truncate ${i === 0 ? 'border-l border-gray-800' : ''}`}
                    >
                      {c.label}
                    </th>
                  )),
                )}
              </tr>
            </thead>
            <tbody>
              {events.map((e, row) => {
                const s = stepSummaries[row];
                const selected = e.step === event.step;
                return (
                  <tr
                    key={e.step}
                    onClick={() => setPickedStep(e.step === latest?.step ? null : e.step)}
                    className={`border-b border-gray-800/60 cursor-pointer hover:bg-gray-800/40 ${selected ? 'bg-blue-900/30' : ''}`}
                  >
                    <td className="px-3 py-1.5 text-xs text-gray-200 font-mono tabular-nums">{e.step}</td>
                    {columnGroups.map(g =>
                      g.cols.map((c, i) => {
                        const raw = s.get(c.k)?.mean;
                        const best = raw !== undefined && raw === columnMin.get(c.k) && events.length > 1;
                        const v = raw === undefined ? NaN : compare(raw, stepSummaries[0]?.get(c.k)?.mean, mode);
                        return (
                          <td
                            key={c.k}
                            className={`${cellClass} ${i === 0 ? 'border-l border-gray-800' : ''} ${
                              best ? 'font-bold text-green-300 bg-green-900/40' : 'text-gray-300'
                            }`}
                          >
                            {fmt(v, mode)}
                          </td>
                        );
                      }),
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div
        className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800"
      >
        <div className="bg-gray-800 px-4 py-3 flex flex-wrap items-center gap-x-3 gap-y-2">
          <h3 className="text-gray-100 text-sm font-medium">Averages across steps</h3>
          {hasTrain && (
            <span className="text-xs text-gray-400">
              Right axis: training loss, averaged over the steps since the previous validation.
            </span>
          )}
          <NoiseStepper
            index={noise}
            sigmas={sigmas}
            onStep={stepNoise}
          />
        </div>
        <div className="h-80 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid stroke="#1f2937" />
              <XAxis dataKey="step" stroke="#9ca3af" tick={{ fontSize: 11 }} />
              <YAxis
                yAxisId="val"
                stroke="#9ca3af"
                tick={{ fontSize: 11 }}
                domain={['auto', 'auto']}
                width={70}
                tickFormatter={(v: any) => fmt(Number(v), mode)}
              />
              {hasTrain && (
                <YAxis
                  yAxisId="train"
                  orientation="right"
                  stroke="#6b7280"
                  tick={{ fontSize: 11 }}
                  domain={['auto', 'auto']}
                  width={70}
                  tickFormatter={(v: any) => Number(v).toPrecision(3)}
                />
              )}
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', fontSize: 12 }}
                formatter={(v: any, name: any) =>
                  name === 'Training loss' ? Number(v).toPrecision(4) : fmt(Number(v), mode)
                }
                labelFormatter={(s: any) => `step ${s}`}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line
                yAxisId="val"
                type="monotone"
                dataKey={key(ALL, ALL, ALL)}
                name="All"
                stroke="#e5e7eb"
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
              />
              {event.resolutions.map((r, i) => (
                <Line
                  key={r}
                  yAxisId="val"
                type="monotone"
                  dataKey={key(ALL, ALL, r)}
                  name={`${r}`}
                  stroke={LINE_COLORS[i % LINE_COLORS.length]}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
              {variants.map((v, i) => (
                <Line
                  key={`prompt-${v}`}
                  yAxisId="val"
                type="monotone"
                  dataKey={key(ALL, v, ALL)}
                  name={`prompt ${v}`}
                  stroke={LINE_COLORS[(event.resolutions.length + i) % LINE_COLORS.length]}
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
              {hasTrain && (
                <Line
                  yAxisId="train"
                  type="monotone"
                  dataKey={TRAIN_KEY}
                  name="Training loss"
                  stroke="#9ca3af"
                  strokeWidth={1.5}
                  strokeDasharray="2 3"
                  dot={{ r: 2 }}
                  connectNulls
                  isAnimationActive={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
