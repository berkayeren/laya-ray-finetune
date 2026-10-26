// "Training runs" (11-UI-SPEC.md "1. Training runs", D-09, D-10), from GET /api/training: the per-epoch rows that Ray
// Train (Laya, one MPS worker) and the HF Trainer (BERTurk) recorded in each final seed's run record, read by the demo
// server from the final checkpoints during its binding stage. While that stage runs the endpoint reads loading: the
// view shows the loading card and polls GET /api/status every second until its training key reads ready, then fetches
// /api/training again. The rows come from that endpoint only: a model whose checkpoints are missing or changed shows
// "Training curves unavailable" with the server's own line in its card, never rows from another source (the user
// decision of 2026-10-25), and the other model still charts.
//
// Per model a stat-tile card (epochs per seed, device, train messages, total train time over the three seeds), then
// one metric selector for both charts ("Peak memory" opens a second selector: Total, Driver, RSS), then two small
// multiples, Laya left and BERTurk right from a 960px content width, each with its own y axis: three seed lines in the
// model's colour told apart by dash and marker (s0 solid circle, s1 dashed 6 4 square, s2 dotted 2 3 triangle), a
// legend, direct end labels, a crosshair tooltip with the three seeds' exact values and a "Show table" toggle with the
// same values. Every value is a field of a row; the UI only converts units (GiB, MiB) and formats it.

import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatGiB, formatMiB, formatSeconds } from "../format";
import { getJSON } from "../http";
import { COLORS, Card, DataTable, SegmentedControl, ShowTable, SourceCaption, Tile, ViewGate } from "./common";
import type { ViewState } from "./common";
import type { EpochRow, EpochValue, TrainingData, TrainingOk, TrainingReady, TrainingStatus, TrainingSystem } from "./types";
import css from "./dashboard.module.css";

const POLL_MS = 1000;
const RETRY_MS = 1000; // the first retry after an unanswered request
const RETRY_MAX_MS = 5000; // the backoff cap
const INTRO =
  "Per-epoch rows recorded by Ray Train (Laya, one MPS worker) and the HF Trainer (BERTurk) in each final seed's run record.";
const LOSS_CAPTION = "Loss values are not comparable across the two models: they train on different objectives.";
const UNAVAILABLE_TITLE = "Training curves unavailable";
const MARKER = 4; // half of the 8px marker
const LABEL_GAP = 14; // the least vertical distance between two end labels, in px
const TICK_ALL = 10; // up to this many epochs, every epoch gets an x tick
// the chart's vertical geometry: the plot height is CHART_H - MARGIN_TOP - MARGIN_BOTTOM - X_AXIS_H, which places the
// end labels from the values alone (CHART_H is --chart-height)
const CHART_H = 280;
const MARGIN_TOP = 16;
const MARGIN_BOTTOM = 8;
const X_AXIS_H = 30;
const PLOT_H = CHART_H - MARGIN_TOP - MARGIN_BOTTOM - X_AXIS_H;

// --- data -----------------------------------------------------------------------------------------------------------

// GET /api/training; while it reads loading, GET /api/status every POLL_MS until training reads ready, then
// GET /api/training once more. Any HTTP error is that view's error; an unanswered request reads unreachable and
// retries GET /api/status after RETRY_MS, doubling up to RETRY_MAX_MS, so a restarted server is picked up without a
// reload.
function useTraining(): ViewState<TrainingReady> {
  const [state, setState] = useState<ViewState<TrainingReady>>({ kind: "loading" });
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let retry = RETRY_MS;
    const fail = (r: { status: number | null; error: string; source?: string | null }, path: string): void => {
      if (r.status === null) {
        setState({ kind: "unreachable" });
        timer = window.setTimeout(() => void pollStatus(), retry);
        retry = Math.min(retry * 2, RETRY_MAX_MS);
      } else {
        setState({ kind: "error", error: r.error, source: r.source ?? path });
      }
    };
    const pollStatus = async (): Promise<void> => {
      const r = await getJSON<TrainingStatus>("/api/status");
      if (cancelled) {
        return;
      }
      if (!r.ok) {
        fail(r, "/api/status");
        return;
      }
      retry = RETRY_MS;
      if (r.data.training === "ready") {
        void fetchTraining();
      } else {
        setState((prev) => (prev.kind === "loading" ? prev : { kind: "loading" }));
        timer = window.setTimeout(() => void pollStatus(), POLL_MS);
      }
    };
    const fetchTraining = async (): Promise<void> => {
      const r = await getJSON<TrainingData>("/api/training");
      if (cancelled) {
        return;
      }
      if (!r.ok) {
        fail(r, "/api/training");
      } else if (r.data.status === "ready") {
        setState({ kind: "ok", data: r.data });
      } else {
        setState({ kind: "loading" });
        timer = window.setTimeout(() => void pollStatus(), POLL_MS);
      }
    };
    void fetchTraining();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, []);
  return state;
}

// --- metrics --------------------------------------------------------------------------------------------------------

type MetricKey = "loss" | "train_s" | "optimizer_steps" | "memory" | "swap";
type MemoryKey = "peak_total_bytes" | "peak_driver_bytes" | "peak_rss_bytes";

interface Metric {
  field: string;
  title: string; // the chart caption and the table caption
  chart: (v: number) => number; // the plotted value (GiB and MiB for bytes)
  text: (v: number) => string; // the exact value in the tooltip and the table
}

const METRIC_OPTIONS: { value: MetricKey; label: string }[] = [
  { value: "loss", label: "Loss" },
  { value: "train_s", label: "Train time (s)" },
  { value: "optimizer_steps", label: "Optimizer steps" },
  { value: "memory", label: "Peak memory" },
  { value: "swap", label: "Swap delta" },
];

const MEMORY_OPTIONS: { value: MemoryKey; label: string }[] = [
  { value: "peak_total_bytes", label: "Total" },
  { value: "peak_driver_bytes", label: "Driver" },
  { value: "peak_rss_bytes", label: "RSS" },
];

const same = (v: number): number => v;

function metricOf(key: MetricKey, memory: MemoryKey): Metric {
  if (key === "loss") {
    return { field: "loss", title: "Loss", chart: same, text: (v) => v.toFixed(4) };
  }
  if (key === "train_s") {
    return {
      field: "train_s",
      title: "Train time (s)",
      chart: same,
      text: (v) => formatSeconds(v) + " s",
    };
  }
  if (key === "optimizer_steps") {
    return { field: "optimizer_steps", title: "Optimizer steps", chart: same, text: String };
  }
  if (key === "swap") {
    return {
      field: "swap_delta_bytes",
      title: "Swap delta (MiB)",
      chart: (v) => v / 2 ** 20,
      text: (v) => formatMiB(v) + " MiB",
    };
  }
  const label = MEMORY_OPTIONS.find((o) => o.value === memory)?.label ?? "";
  return {
    field: memory,
    title: `Peak memory, ${label} (GiB)`,
    chart: (v) => v / 2 ** 30,
    text: (v) => formatGiB(v) + " GiB",
  };
}

// a plotted or tabled value: only a finite number counts; null and anything else is a gap and reads n/a
function numberOf(v: EpochValue | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// --- one model ------------------------------------------------------------------------------------------------------

interface Model {
  key: "laya" | "berturk";
  name: string;
  trainer: string;
  color: string;
  swatch: string;
}

const MODELS: Model[] = [
  { key: "laya", name: "Laya", trainer: "Ray Train", color: COLORS.laya, swatch: "var(--series-laya)" },
  { key: "berturk", name: "BERTurk", trainer: "HF Trainer", color: COLORS.berturk, swatch: "var(--series-berturk)" },
];

type MarkerShape = "circle" | "square" | "triangle";

// seeds in the server's order: the first solid with circles, the second dashed 6 4 with squares, the third dotted
// 2 3 with triangles
const SEED_STYLES: { dash: string | undefined; marker: MarkerShape }[] = [
  { dash: undefined, marker: "circle" },
  { dash: "6 4", marker: "square" },
  { dash: "2 3", marker: "triangle" },
];

function seedStyle(i: number): { dash: string | undefined; marker: MarkerShape } {
  return SEED_STYLES[i % SEED_STYLES.length];
}

function Marker({ shape, cx, cy, color }: { shape: MarkerShape; cx: number; cy: number; color: string }) {
  if (shape === "square") {
    return <rect x={cx - MARKER} y={cy - MARKER} width={2 * MARKER} height={2 * MARKER} fill={color} />;
  }
  if (shape === "triangle") {
    const h = MARKER + 1;
    return <polygon points={`${cx},${cy - h} ${cx + h},${cy + MARKER} ${cx - h},${cy + MARKER}`} fill={color} />;
  }
  return <circle cx={cx} cy={cy} r={MARKER} fill={color} />;
}

// one chart row per epoch: each seed's plotted value and its raw row value, in the seeds' order
interface Row {
  epoch: number;
  values: (number | null)[];
  raw: (number | null)[];
}

function rowsOf(system: TrainingOk, metric: Metric): Row[] {
  const byEpoch = (rows: EpochRow[], epoch: number): EpochRow | undefined => rows.find((r) => r.epoch === epoch);
  const epochs = [
    ...new Set(system.seeds.flatMap((s) => s.rows.map((r) => numberOf(r.epoch)).filter((e) => e !== null))),
  ] as number[];
  epochs.sort((a, b) => a - b);
  return epochs.map((epoch) => {
    const raw = system.seeds.map((s) => numberOf(byEpoch(s.rows, epoch)?.[metric.field]));
    return { epoch, raw, values: raw.map((v) => (v === null ? null : metric.chart(v))) };
  });
}

const seedLabel = (seed: number): string => `s${seed}`;

function Legend({ system, color }: { system: TrainingOk; color: string }) {
  return (
    <ul className={css.legend}>
      {system.seeds.map((s, i) => {
        const st = seedStyle(i);
        return (
          <li key={s.seed} className={css.legendItem}>
            <svg width="32" height="12" aria-hidden="true">
              <line x1="0" y1="6" x2="32" y2="6" stroke={color} strokeWidth={2} strokeDasharray={st.dash} />
              <Marker shape={st.marker} cx={16} cy={6} color={color} />
            </svg>
            {`${seedLabel(s.seed)} (seed ${s.seed})`}
          </li>
        );
      })}
    </ul>
  );
}

interface TipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: Row }>;
}

function seedTip(system: TrainingOk, metric: Metric) {
  return function SeedTip({ active, payload }: TipProps) {
    const row = active ? payload?.[0]?.payload : undefined;
    if (row === undefined) {
      return null;
    }
    return (
      <div className={css.chartTip}>
        <p className={css.chartTipTitle}>{`epoch ${row.epoch}`}</p>
        {system.seeds.map((s, i) => {
          const v = row.raw[i];
          return (
            <p key={s.seed} className={css.chartTipLine}>
              {`${seedLabel(s.seed)}: ${v === null ? "n/a" : metric.text(v)}`}
            </p>
          );
        })}
      </div>
    );
  };
}

// The y axis of one chart, from its plotted values: a nice step (1, 2 or 5 times a power of ten, about four
// intervals), the domain the multiples of the step around the values, the ticks every step, labelled with as many
// decimals as the step needs.
interface YAxisSpec {
  domain: [number, number];
  ticks: number[];
  format: (v: number) => string;
}

function yAxisOf(values: (number | null)[]): YAxisSpec {
  const xs = values.filter((v): v is number => v !== null);
  let lo = xs.length === 0 ? 0 : Math.min(...xs);
  let hi = xs.length === 0 ? 1 : Math.max(...xs);
  if (hi === lo) {
    lo = lo >= 0 && lo < 1 ? 0 : lo - 1;
    hi = hi + 1;
  }
  const raw = (hi - lo) / 4;
  const e = Math.floor(Math.log10(raw));
  const f = raw / 10 ** e;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * 10 ** e;
  const first = Math.floor(lo / step);
  const last = Math.max(Math.ceil(hi / step), first + 1);
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  const ticks: number[] = [];
  for (let k = first; k <= last; k++) {
    ticks.push(Number((k * step).toFixed(decimals)));
  }
  return { domain: [ticks[0], ticks[ticks.length - 1]], ticks, format: (v) => v.toFixed(decimals) };
}

// The vertical offset of seed i's end label from its own point: every seed's last value placed on the plot height,
// sorted top to bottom (ties in seed order), pushed down until no two labels are closer than LABEL_GAP, then pushed
// back up as a group where the lowest would fall below the plot.
function endOffset(last: (number | null)[], i: number, domain: [number, number]): number {
  const px = PLOT_H / (domain[1] - domain[0]);
  const ys = last
    .map((v, j) => ({ y: v === null ? null : (domain[1] - v) * px, j }))
    .filter((e): e is { y: number; j: number } => e.y !== null);
  ys.sort((a, b) => a.y - b.y || a.j - b.j);
  const placed = ys.map((e) => e.y);
  for (let k = 1; k < placed.length; k++) {
    placed[k] = Math.max(placed[k], placed[k - 1] + LABEL_GAP);
  }
  for (let k = placed.length - 1; k >= 0; k--) {
    placed[k] = Math.min(placed[k], k === placed.length - 1 ? PLOT_H : placed[k + 1] - LABEL_GAP);
  }
  const k = ys.findIndex((e) => e.j === i);
  return k < 0 ? 0 : placed[k] - ys[k].y;
}

interface DotArgs {
  cx?: number;
  cy?: number;
  index: number;
  payload?: Row;
}

interface ChartProps {
  system: TrainingOk;
  model: Model;
  metric: Metric;
}

function SeedChart({ system, model, metric }: ChartProps) {
  const rows = rowsOf(system, metric);
  const axis = yAxisOf(rows.flatMap((r) => r.values));
  // each seed's last plotted value: its end label sits right of that point
  const lastIndex = system.seeds.map((_, i) => rows.reduce((last, r, j) => (r.values[i] === null ? last : j), -1));
  const lastValue = lastIndex.map((j, i) => (j < 0 ? null : rows[j].values[i]));
  const dot = (i: number) =>
    function SeedDot({ cx, cy, index, payload }: DotArgs) {
      if (cx === undefined || cy === undefined || payload === undefined || payload.values[i] === null) {
        return null;
      }
      const st = seedStyle(i);
      if (index !== lastIndex[i]) {
        return <Marker shape={st.marker} cx={cx} cy={cy} color={model.color} />;
      }
      return (
        <g>
          <Marker shape={st.marker} cx={cx} cy={cy} color={model.color} />
          <text
            x={cx + 10}
            y={cy + endOffset(lastValue, i, axis.domain) + 5}
            textAnchor="start"
            fill={COLORS.inkSecondary}
            fontSize={14}
            stroke="#ffffff"
            strokeWidth={3}
            paintOrder="stroke"
          >
            {seedLabel(system.seeds[i].seed)}
          </text>
        </g>
      );
    };
  const epochs = rows.map((r) => r.epoch);
  const ticks = epochs.length <= TICK_ALL ? epochs : epochs.filter((e, i) => i === 0 || e % 5 === 0);
  return (
    <div className={css.chart} style={{ height: CHART_H }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: MARGIN_TOP, right: 40, bottom: MARGIN_BOTTOM, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="epoch"
            domain={["dataMin", "dataMax"]}
            ticks={ticks}
            interval={0}
            height={X_AXIS_H}
            padding={{ left: 12, right: 12 }}
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <YAxis
            type="number"
            domain={axis.domain}
            ticks={axis.ticks}
            interval={0}
            tickFormatter={axis.format}
            width={64}
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <Tooltip content={seedTip(system, metric)} cursor={{ stroke: COLORS.axis }} isAnimationActive={false} />
          {system.seeds.map((s, i) => (
            <Line
              key={s.seed}
              type="linear"
              dataKey={(r: Row) => r.values[i]}
              name={seedLabel(s.seed)}
              stroke={model.color}
              strokeWidth={2}
              strokeDasharray={seedStyle(i).dash}
              dot={dot(i)}
              activeDot={{ r: MARKER + 2, fill: model.color, stroke: "#ffffff", strokeWidth: 2 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function SeedTable({ system, metric }: { system: TrainingOk; metric: Metric }) {
  return (
    <DataTable
      caption={metric.title}
      header={["epoch", ...system.seeds.map((s) => seedLabel(s.seed))]}
      rows={rowsOf(system, metric).map((r) => [
        String(r.epoch),
        ...r.raw.map((v) => (v === null ? "n/a" : metric.text(v))),
      ])}
    />
  );
}

// the distinct values of one seed field, in the seeds' order, or n/a when none is recorded
function distinct(values: (string | number | null)[]): string {
  const seen = [...new Set(values.filter((v) => v !== null).map(String))];
  return seen.length === 0 ? "n/a" : seen.join(", ");
}

function ModelTitle({ model }: { model: Model }) {
  return (
    <h2 className={css.cardTitle}>
      <span className={css.swatch} style={{ background: model.swatch }} aria-hidden="true" />
      {`${model.name} (${model.trainer})`}
    </h2>
  );
}

function StatsCard({ model, system }: { model: Model; system: TrainingSystem }) {
  if (system.status !== "ok") {
    return (
      <Card>
        <ModelTitle model={model} />
        <div className={css.notice} role="status">
          <p className={css.noticeTitle}>{UNAVAILABLE_TITLE}</p>
          <p className={css.noticeBody}>
            {`The per-epoch rows live in the final checkpoints, which are not committed: ${system.message}. ` +
              "Every other dashboard view reads results/ and still works."}
          </p>
        </div>
      </Card>
    );
  }
  return (
    <Card>
      <ModelTitle model={model} />
      <div className={css.tiles}>
        <Tile label="Epochs per seed" value={String(system.seeds[0]?.rows.length ?? 0)} />
        <Tile label="Device" value={distinct(system.seeds.map((s) => s.device))} />
        <Tile label="Train messages" value={distinct(system.seeds.map((s) => s.n_train_messages))} />
        <Tile
          label="Total train time"
          value={formatSeconds(system.total_train_s) + " s"}
          note={`over the ${system.seeds.length} seeds`}
        />
      </div>
    </Card>
  );
}

function ChartCard({ model, system, metric }: { model: Model; system: TrainingOk; metric: Metric }) {
  return (
    <Card>
      <ModelTitle model={model} />
      <p className={css.chartCaption}>{`${metric.title} by epoch, per final seed`}</p>
      <Legend system={system} color={model.color} />
      <ShowTable
        chart={<SeedChart system={system} model={model} metric={metric} />}
        table={<SeedTable system={system} metric={metric} />}
      />
    </Card>
  );
}

function Training({ data }: { data: TrainingReady }) {
  const [metricKey, setMetricKey] = useState<MetricKey>("loss");
  const [memory, setMemory] = useState<MemoryKey>("peak_total_bytes");
  const metric = metricOf(metricKey, memory);
  const systems = MODELS.map((m) => ({ model: m, system: data.systems[m.key] }));
  const charted = systems.filter((x) => x.system.status === "ok");
  const sources = charted.flatMap((x) => (x.system.status === "ok" ? x.system.seeds.map((s) => s.checkpoint_name) : []));
  return (
    <>
      {sources.length > 0 && <SourceCaption sources={sources} />}
      <p className={css.intro}>{INTRO}</p>
      <div className={css.pairWrap}>
        <div className={css.pair}>
          {systems.map(({ model, system }) => (
            <div key={model.key} className={css.column}>
              <StatsCard model={model} system={system} />
            </div>
          ))}
        </div>
      </div>
      {charted.length > 0 && (
        <>
          <div className={css.selectorRow}>
            <SegmentedControl label="Metric" options={METRIC_OPTIONS} value={metricKey} onChange={setMetricKey} />
            {metricKey === "memory" && (
              <SegmentedControl label="Peak memory" options={MEMORY_OPTIONS} value={memory} onChange={setMemory} />
            )}
          </div>
          {metricKey === "loss" && <p className={css.lossCaption}>{LOSS_CAPTION}</p>}
          <div className={css.pairWrap}>
            <div className={css.pair}>
              {systems.map(({ model, system }) => (
                <div key={model.key} className={css.column}>
                  {system.status === "ok" && <ChartCard model={model} system={system} metric={metric} />}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </>
  );
}

export default function TrainingView() {
  const state = useTraining();
  return <ViewGate state={state}>{(data) => <Training data={data} />}</ViewGate>;
}
