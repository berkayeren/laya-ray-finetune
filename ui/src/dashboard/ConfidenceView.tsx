// "Confidence, latency and few-shot" (11-UI-SPEC.md "4. Confidence, latency and few-shot", D-10). Three sections, each
// with its own endpoint and its own error, so one refused record leaves the other two working:
//
//   Reliability diagrams  GET /api/confidence: the committed reliability SVGs, "Val (in-sample)" or "Test", each an
//                         <img> from GET /api/svg/{name} (a name from the server's allow-list, URL-encoded) at the full
//                         card width with its 1120:550 aspect ratio reserved, its run id as a mono caption; the val
//                         records' caveat verbatim above the val images. The SVGs are never inlined, redrawn or
//                         recoloured; a failed load shows the alt text and "Diagram unavailable" naming the SVG's path.
//   Latency per message   GET /api/latency, reported, not gated: a horizontal bar per (system, device) at the pooled
//                         p50 with a 2px tick at the pooled p95 and direct value labels, then one table per report
//                         (p50, p95, mean, min and max in ms per seed, then the pooled row, as in latency.md).
//   Few-shot              GET /api/fewshot: new-tag macro-AP by k, one line per model through the means over the draws
//                         and each draw as an 8px dot; the "Mean delta at k = N" tile with the server's text; the
//                         memo's "Few-shot:" caveat verbatim above the forgetting table and the record's notes below it.
//
// Every number is a field of a view and every sentence (caveats, notes, renders) is shown as received; the UI only
// formats values (ms, 3-decimal metrics).

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatMetric, formatMs } from "../format";
import {
  COLORS,
  Card,
  DataTable,
  FrozenRender,
  SegmentedControl,
  ShowTable,
  SourceCaption,
  Tile,
  ViewGate,
  useView,
} from "./common";
import type { ApAtK, ConfidenceData, Diagram, FewshotData, LatencyData, LatencyReport } from "./types";
import css from "./dashboard.module.css";

const NAMES: Record<string, string> = { laya: "Laya", berturk: "BERTurk" };
const SERIES: Record<string, string> = { laya: COLORS.laya, berturk: COLORS.berturk };

const nameOf = (system: string): string => NAMES[system] ?? system;
const colorOf = (system: string): string => SERIES[system] ?? COLORS.reference;
const metricOrNA = (x: number | null): string => (x === null ? "n/a" : formatMetric(x));

function SectionTitle({ title, suffix }: { title: string; suffix?: string }) {
  return (
    <h2 className={css.sectionTitle}>
      {title}
      {suffix !== undefined && <span className={css.sectionSuffix}>{suffix}</span>}
    </h2>
  );
}

// --- reliability diagrams -------------------------------------------------------------------------------------------

type DiagramSet = "val" | "test";

const SET_OPTIONS: { value: DiagramSet; label: string }[] = [
  { value: "val", label: "Val (in-sample)" },
  { value: "test", label: "Test" },
];

function DiagramCard({ d }: { d: Diagram }) {
  const [failed, setFailed] = useState(false);
  const alt = `${d.svg} (${d.split})`;
  return (
    <Card>
      {failed ? (
        <div className={css.diagramMissing} role="img" aria-label={alt}>
          <p className={css.noticeBody}>{alt}</p>
          <p className={css.noticeBody}>{`Diagram unavailable: ${d.path}`}</p>
        </div>
      ) : (
        <img
          className={css.diagram}
          src={"/api/svg/" + encodeURIComponent(d.svg)}
          alt={alt}
          width={1120}
          height={550}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      )}
      <p className={css.diagramCaption}>{d.run_id}</p>
    </Card>
  );
}

function Reliability({ v }: { v: ConfidenceData }) {
  const [set, setSet] = useState<DiagramSet>("val");
  const diagrams = set === "val" ? v.val : v.test;
  return (
    <>
      <SegmentedControl label="Reliability diagrams" options={SET_OPTIONS} value={set} onChange={setSet} />
      {set === "val" &&
        v.caveats.map((c, i) => (
          <p key={i} className={css.caveatLine}>
            {c}
          </p>
        ))}
      {diagrams.map((d) => (
        <DiagramCard key={d.svg} d={d} />
      ))}
    </>
  );
}

function ReliabilitySection() {
  const state = useView("confidence");
  return (
    <section className={css.section}>
      <SectionTitle title="Reliability diagrams" />
      <ViewGate state={state}>
        {(v) => (
          <>
            <SourceCaption sources={v.sources} />
            <Reliability v={v} />
          </>
        )}
      </ViewGate>
    </section>
  );
}

// --- latency --------------------------------------------------------------------------------------------------------

const BAR_ROW = 48; // the height of one bar's band, in px
const BAR_LABEL_WIDTH = 140;
const TICK_HALF = 4; // the p95 tick reaches this far beyond the bar, in px
const BAR_ROW_NARROW = 88; // below 720px: the band holds the two label lines above a thinner bar
const BAR_SIZE_NARROW = 16;
const NARROW = "(max-width: 719px)"; // the CSS breakpoint every module uses

// true below 720px, following the viewport; the right label column and the y axis do not fit a 375px card
function useNarrow(): boolean {
  const mq = typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(NARROW)
    : null;
  const [narrow, setNarrow] = useState(mq?.matches ?? false);
  useEffect(() => {
    if (mq === null) {
      return;
    }
    const on = (e: MediaQueryListEvent) => setNarrow(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [mq?.media]);
  return narrow;
}

interface LatencyRow {
  label: string;
  system: string;
  p50: number; // ms, plotted
  p95: number; // ms
  p50Text: string;
  p95Text: string;
}

function latencyRows(reports: LatencyReport[]): LatencyRow[] {
  return reports.map((r) => ({
    label: `${nameOf(r.system)} on ${r.device}`,
    system: r.system,
    p50: r.pooled.p50 * 1000,
    p95: r.pooled.p95 * 1000,
    p50Text: formatMs(r.pooled.p50),
    p95Text: formatMs(r.pooled.p95),
  }));
}

interface BarArgs {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: LatencyRow;
}

// the p50 bar in the model's colour, a 2px ink tick at p95 (the axis starts at 0, so p95 sits at width * p95 / p50)
// and the direct label with both values
function LatencyBar({ x, y, width, height, payload }: BarArgs) {
  if (x === undefined || y === undefined || width === undefined || height === undefined || payload === undefined) {
    return <g />;
  }
  const tick = payload.p50 > 0 ? x + (width * payload.p95) / payload.p50 : x;
  return (
    <g>
      <rect x={x} y={y} width={Math.max(width, 1)} height={height} fill={colorOf(payload.system)} />
      <line x1={tick} y1={y - TICK_HALF} x2={tick} y2={y + height + TICK_HALF} stroke={COLORS.ink} strokeWidth={2} />
      <text x={tick + 8} y={y + height / 2 + 5} fill={COLORS.inkSecondary} fontSize={14}>
        {`p50 ${payload.p50Text} ms · p95 ${payload.p95Text} ms`}
      </text>
    </g>
  );
}

interface LatencyTipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: LatencyRow }>;
}

function LatencyTip({ active, payload }: LatencyTipProps) {
  const row = active ? payload?.[0]?.payload : undefined;
  if (row === undefined) {
    return null;
  }
  return (
    <div className={css.chartTip}>
      <p className={css.chartTipTitle}>{row.label}</p>
      <p className={css.chartTipLine}>{`pooled p50 ${row.p50Text} ms`}</p>
      <p className={css.chartTipLine}>{`pooled p95 ${row.p95Text} ms`}</p>
    </div>
  );
}

// the x axis in ms: 0 to the first multiple of a nice step (1, 2 or 5 times a power of ten, about four intervals)
// at or above the largest p95, a tick every step
function msTicks(max: number): number[] {
  const raw = Math.max(max, 1) / 4;
  const e = Math.floor(Math.log10(raw));
  const f = raw / 10 ** e;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * 10 ** e;
  const n = Math.ceil(Math.max(max, 1) / step);
  return Array.from({ length: n + 1 }, (_, k) => Number((k * step).toFixed(6)));
}

// below 720px: no y axis and no right label column (a 28px right margin keeps the last tick label whole); the system
// name and both values sit above the bar, from its start
function LatencyBarNarrow({ x, y, width, height, payload }: BarArgs) {
  if (x === undefined || y === undefined || width === undefined || height === undefined || payload === undefined) {
    return <g />;
  }
  const tick = payload.p50 > 0 ? x + (width * payload.p95) / payload.p50 : x;
  return (
    <g>
      <text x={x} y={y - 26} fill={COLORS.inkSecondary} fontSize={14} fontWeight={600}>
        {payload.label}
      </text>
      <text x={x} y={y - 8} fill={COLORS.inkSecondary} fontSize={14}>
        {`p50 ${payload.p50Text} ms · p95 ${payload.p95Text} ms`}
      </text>
      <rect x={x} y={y} width={Math.max(width, 1)} height={height} fill={colorOf(payload.system)} />
      <line x1={tick} y1={y - TICK_HALF} x2={tick} y2={y + height + TICK_HALF} stroke={COLORS.ink} strokeWidth={2} />
    </g>
  );
}

function LatencyChart({ rows }: { rows: LatencyRow[] }) {
  const ticks = msTicks(Math.max(...rows.map((r) => r.p95)));
  const narrow = useNarrow();
  return (
    <div className={css.chart} style={{ height: rows.length * (narrow ? BAR_ROW_NARROW : BAR_ROW) + 40 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 8, right: narrow ? 28 : 240, bottom: 8, left: 8 }}
          barCategoryGap={12}
          barSize={narrow ? BAR_SIZE_NARROW : undefined}
        >
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="p50"
            domain={[0, ticks[ticks.length - 1]]}
            ticks={ticks}
            interval={0}
            tickFormatter={(v: number) => String(v)}
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={BAR_LABEL_WIDTH}
            hide={narrow}
            tick={{ fill: COLORS.inkSecondary, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <Tooltip content={LatencyTip} cursor={{ fill: "#f0efec" }} isAnimationActive={false} />
          <Bar dataKey="p50" shape={narrow ? LatencyBarNarrow : LatencyBar} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function LatencyTable({ r }: { r: LatencyReport }) {
  const cells = (s: { n: number; p50: number; p95: number; mean: number; min: number; max: number }): string[] => [
    String(s.n),
    formatMs(s.p50),
    formatMs(s.p95),
    formatMs(s.mean),
    formatMs(s.min),
    formatMs(s.max),
  ];
  return (
    <DataTable
      header={["seed", "checkpoint", "n", "p50 ms", "p95 ms", "mean ms", "min ms", "max ms"]}
      rows={[
        ...r.seeds.map((s) => [String(s.seed), s.checkpoint_name, ...cells(s.summary)]),
        ["pooled", `${r.seeds.length} seed(s)`, ...cells(r.pooled)],
      ]}
    />
  );
}

function Latency({ v }: { v: LatencyData }) {
  const rows = latencyRows(v.reports);
  return (
    <>
      <Card title="Pooled p50 per system and device, with the p95 tick (ms)">
        <ShowTable
          chart={<LatencyChart rows={rows} />}
          table={
            <DataTable
              header={["system and device", "pooled p50 ms", "pooled p95 ms"]}
              rows={rows.map((r) => [r.label, r.p50Text, r.p95Text])}
            />
          }
        />
      </Card>
      {v.reports.map((r) => (
        <Card key={r.path} title={`${nameOf(r.system)} on ${r.device}`}>
          <p className={css.chartCaption + " " + css.monoCaption}>{r.path}</p>
          <LatencyTable r={r} />
        </Card>
      ))}
      <FrozenRender renders={[v.render]} />
    </>
  );
}

function LatencySection() {
  const state = useView("latency");
  return (
    <section className={css.section}>
      <SectionTitle title="Latency per message" suffix="Reported, not gated" />
      <ViewGate state={state}>
        {(v) => (
          <>
            <SourceCaption sources={v.sources} />
            <Latency v={v} />
          </>
        )}
      </ViewGate>
    </section>
  );
}

// --- few-shot -------------------------------------------------------------------------------------------------------

const DRAW_R = 4; // the 8px draw dot
const kLabel = (k: number): string => (k === 0 ? "k = 0 (masked base)" : `k = ${k}`);

interface ApRow {
  k: number;
  laya: number | null;
  berturk: number | null;
  layaText: string;
  berturkText: string;
}

function apRows(byK: { laya: ApAtK[]; berturk: ApAtK[] }): ApRow[] {
  const ks = [...new Set([...byK.laya, ...byK.berturk].map((x) => x.k))].sort((a, b) => a - b);
  return ks.map((k) => {
    const l = byK.laya.find((x) => x.k === k);
    const b = byK.berturk.find((x) => x.k === k);
    return {
      k,
      laya: l?.mean ?? null,
      berturk: b?.mean ?? null,
      layaText: l?.mean_text ?? "n/a",
      berturkText: b?.mean_text ?? "n/a",
    };
  });
}

interface DrawPoint {
  k: number;
  ap: number;
}

const drawPoints = (series: ApAtK[]): DrawPoint[] => series.flatMap((x) => x.draws.map((ap) => ({ k: x.k, ap })));

interface ApDotArgs {
  cx?: number;
  cy?: number;
  index: number;
}

// a mean marker; the line's last point also carries the model's name as its direct label
function meanDot(name: string, color: string, lastIndex: number) {
  return function MeanDot({ cx, cy, index }: ApDotArgs) {
    if (cx === undefined || cy === undefined) {
      return null;
    }
    return (
      <g>
        <circle cx={cx} cy={cy} r={DRAW_R + 1} fill="#ffffff" stroke={color} strokeWidth={2} />
        {index === lastIndex && (
          <text
            x={cx + 12}
            y={cy + 5}
            fill={COLORS.inkSecondary}
            fontSize={14}
            fontWeight={600}
            stroke="#ffffff"
            strokeWidth={3}
            paintOrder="stroke"
          >
            {name}
          </text>
        )}
      </g>
    );
  };
}

function drawDot(color: string) {
  return function DrawDot({ cx, cy }: { cx?: number; cy?: number }) {
    if (cx === undefined || cy === undefined) {
      return <g />;
    }
    return <circle cx={cx} cy={cy} r={DRAW_R} fill={color} fillOpacity={0.55} />;
  };
}

interface ApTipProps {
  active?: boolean;
  label?: unknown;
  payload?: ReadonlyArray<{ payload?: { k?: number } }>;
}

function apTip(rows: ApRow[]) {
  return function ApTip({ active, label, payload }: ApTipProps) {
    const k = typeof label === "number" ? label : payload?.[0]?.payload?.k;
    const row = active ? rows.find((r) => r.k === k) : undefined;
    if (row === undefined) {
      return null;
    }
    return (
      <div className={css.chartTip}>
        <p className={css.chartTipTitle}>{kLabel(row.k)}</p>
        <p className={css.chartTipLine}>{`Laya mean ${row.layaText}`}</p>
        <p className={css.chartTipLine}>{`BERTurk mean ${row.berturkText}`}</p>
      </div>
    );
  };
}

function ApChart({ v }: { v: FewshotData }) {
  const rows = apRows(v.ap_by_k);
  const ks = rows.map((r) => r.k);
  const models = [
    { key: "laya" as const, series: v.ap_by_k.laya },
    { key: "berturk" as const, series: v.ap_by_k.berturk },
  ];
  return (
    <div className={css.chart} style={{ height: "var(--chart-height)" }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 16, right: 88, bottom: 8, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="k"
            domain={[Math.min(...ks), Math.max(...ks)]}
            ticks={ks}
            interval={0}
            tickFormatter={kLabel}
            padding={{ left: 64, right: 24 }}
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
            tickFormatter={(x: number) => x.toFixed(1)}
            width={48}
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <Tooltip content={apTip(rows)} cursor={{ stroke: COLORS.axis }} isAnimationActive={false} />
          {models.map(({ key, series }) => (
            <Scatter
              key={key + "-draws"}
              data={drawPoints(series)}
              dataKey="ap"
              shape={drawDot(colorOf(key))}
              isAnimationActive={false}
            />
          ))}
          {models.map(({ key }) => (
            <Line
              key={key}
              type="linear"
              dataKey={key}
              name={nameOf(key)}
              stroke={colorOf(key)}
              strokeWidth={2}
              dot={meanDot(nameOf(key), colorOf(key), rows.length - 1)}
              activeDot={{ r: DRAW_R + 2, fill: colorOf(key), stroke: "#ffffff", strokeWidth: 2 }}
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function ApLegend() {
  return (
    <ul className={css.legend}>
      {(["laya", "berturk"] as const).map((key) => (
        <li key={key} className={css.legendItem}>
          <svg width="32" height="12" aria-hidden="true">
            <line x1="0" y1="6" x2="32" y2="6" stroke={colorOf(key)} strokeWidth={2} />
            <circle cx={16} cy={6} r={DRAW_R + 1} fill="#ffffff" stroke={colorOf(key)} strokeWidth={2} />
          </svg>
          {`${nameOf(key)} (mean over the draws)`}
        </li>
      ))}
      <li className={css.legendItem}>
        <svg width="12" height="12" aria-hidden="true">
          <circle cx={6} cy={6} r={DRAW_R} fill={COLORS.inkMuted} fillOpacity={0.55} />
        </svg>
        one draw
      </li>
    </ul>
  );
}

function ApTable({ v }: { v: FewshotData }) {
  const draws = (series: ApAtK[], k: number): string =>
    (series.find((x) => x.k === k)?.draws ?? []).map(formatMetric).join(", ");
  return (
    <DataTable
      header={["k", "Laya mean", "Laya draws", "BERTurk mean", "BERTurk draws"]}
      rows={apRows(v.ap_by_k).map((r) => [
        kLabel(r.k),
        r.layaText,
        draws(v.ap_by_k.laya, r.k),
        r.berturkText,
        draws(v.ap_by_k.berturk, r.k),
      ])}
    />
  );
}

function Fewshot({ v }: { v: FewshotData }) {
  return (
    <>
      <Card title="New-tag macro-AP by k">
        <ApLegend />
        <ShowTable chart={<ApChart v={v} />} table={<ApTable v={v} />} />
        <div className={css.tiles}>
          <Tile
            label={`Mean delta at k = ${v.primary_k}`}
            value={v.mean_delta_text}
            note="Laya - BERTurk new-tag macro-AP, mean over the draws"
          />
        </div>
      </Card>
      <Card title="Forgetting">
        <p className={css.caveatLine}>{v.caveat}</p>
        <DataTable
          header={["model", "run", "base seed", "new-tag micro-F1", "old-tag micro-F1", "forgetting"]}
          rows={v.forgetting.map((f) => [
            nameOf(f.model),
            f.run,
            String(f.base_seed),
            metricOrNA(f.new),
            metricOrNA(f.old),
            metricOrNA(f.forgetting),
          ])}
        />
        <ul className={css.caveats + " " + css.notes}>
          {v.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      </Card>
      <FrozenRender renders={[v.render]} />
    </>
  );
}

function FewshotSection() {
  const state = useView("fewshot");
  return (
    <section className={css.section}>
      <SectionTitle title="Few-shot new-intent results" />
      <ViewGate state={state}>
        {(v) => (
          <>
            <SourceCaption sources={v.sources} />
            <Fewshot v={v} />
          </>
        )}
      </ViewGate>
    </section>
  );
}

export default function ConfidenceView() {
  return (
    <>
      <ReliabilitySection />
      <LatencySection />
      <FewshotSection />
    </>
  );
}
