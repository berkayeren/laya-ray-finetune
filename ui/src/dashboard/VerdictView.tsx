// "Test results and verdict" (11-UI-SPEC.md "3. Test results and verdict", D-10), from GET /api/verdict, which serves
// results/final/verdict.json behind require_frozen and check_record together with report.py's three renders. In the
// UI-SPEC's order: the verdict hero (the verdict, rule and means lines as the frozen render gives them, then the
// recommendation line), the criteria table with status badges, the deltas chart, the results-on-test table and its
// settings table, and the caveats. "View the frozen render" under the hero, the criteria and the results opens the exact render text.
// Every sentence, status word, number and caveat is a field of the view, shown verbatim; the UI writes no verdict
// sentence and computes nothing. A refused record shows its one-line refusal and no number from it.

import {
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { COLORS, Card, DataTable, FrozenRender, ShowTable, SourceCaption, StatusBadge, ViewGate, useView } from "./common";
import type { Delta, VerdictData } from "./types";
import css from "./dashboard.module.css";

// --- the deltas chart -------------------------------------------------------------------------------------------------
// Laya - BERTurk in points on one horizontal axis, one row per deltas entry: an 8px ink dot at delta_points and its
// 95% CI whisker from ci_points (a reference segment with end caps), a dashed zero line. The axis extends to the
// whiskers (ifOverflow extendDomain); the rows run top to bottom in the served order. The tooltip shows delta_text and ci_text exactly as served. The row label and
// the "Show table" table are the entry's own fields; nothing here is computed.

const DOT_R = 4;
const HIT_R = 12;
const CAP_PX = 6;
const LABEL_WIDTH = 360;

type DeltaRow = Delta & { label: string };

function rowLabel(d: Delta): string {
  return `${d.criterion} ${d.metric} (${d.direction})`;
}

interface DotProps {
  cx?: number;
  cy?: number;
}

// the 8px dot, inside a transparent 24px hit target
function DeltaDot({ cx, cy }: DotProps) {
  if (cx === undefined || cy === undefined) {
    return null;
  }
  return (
    <g>
      <circle cx={cx} cy={cy} r={HIT_R} fill="transparent" />
      <circle cx={cx} cy={cy} r={DOT_R} fill={COLORS.ink} />
    </g>
  );
}

interface SegmentProps {
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
}

// the CI whisker: the segment from ci_points[0] to ci_points[1] with a short cap at each end
function Whisker({ x1, y1, x2, y2 }: SegmentProps) {
  if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
    return <g />;
  }
  return (
    <g stroke={COLORS.ink} strokeWidth={1.5}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} />
      <line x1={x1} y1={y1 - CAP_PX} x2={x1} y2={y1 + CAP_PX} />
      <line x1={x2} y1={y2 - CAP_PX} x2={x2} y2={y2 + CAP_PX} />
    </g>
  );
}

interface TipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: DeltaRow }>;
}

function DeltaTip({ active, payload }: TipProps) {
  const row = active ? payload?.[0]?.payload : undefined;
  if (row === undefined) {
    return null;
  }
  return (
    <div className={css.chartTip}>
      <p className={css.chartTipTitle}>{row.label}</p>
      <p className={css.chartTipLine}>{row.delta_text}</p>
      <p className={css.chartTipLine}>{row.ci_text}</p>
    </div>
  );
}

function DeltasChart({ deltas }: { deltas: Delta[] }) {
  const rows: DeltaRow[] = deltas.map((d) => ({ ...d, label: rowLabel(d) }));
  return (
    <div className={css.chart} style={{ height: "var(--chart-height-sm)" }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
          <XAxis
            type="number"
            dataKey="delta_points"
            name="Laya - BERTurk (points)"
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            reversed
            width={LABEL_WIDTH}
            tick={{ fill: COLORS.inkSecondary, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <ReferenceLine x={0} stroke={COLORS.reference} strokeDasharray="4 4" ifOverflow="extendDomain" />
          {rows.map((r) => (
            <ReferenceLine
              key={r.label}
              segment={[
                { x: r.ci_points[0], y: r.label },
                { x: r.ci_points[1], y: r.label },
              ]}
              ifOverflow="extendDomain"
              shape={Whisker}
            />
          ))}
          <Tooltip content={DeltaTip} cursor={false} isAnimationActive={false} />
          <Scatter data={rows} shape={DeltaDot} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function Deltas({ v }: { v: VerdictData }) {
  return (
    <Card title="Laya - BERTurk on test, in points, with 95% CI">
      <ShowTable
        chart={<DeltasChart deltas={v.deltas} />}
        table={
          <DataTable
            header={["criterion", "metric", "delta", "95% CI", "direction"]}
            rows={v.deltas.map((d) => [d.criterion, d.metric, d.delta_text, d.ci_text, d.direction])}
          />
        }
      />
    </Card>
  );
}

function Hero({ v }: { v: VerdictData }) {
  return (
    <Card className={css.hero} style={{ borderLeftColor: `var(--tone-${v.tone}-fill)` }}>
      <p className={css.verdictLine}>{v.lines.verdict}</p>
      <p className={css.heroLine}>{v.lines.rule}</p>
      <p className={css.heroLine}>{v.lines.means}</p>
      <p className={css.heroRecommendation}>{v.lines.recommendation}</p>
      <FrozenRender renders={[v.renders.go, v.renders.recommendation]} />
    </Card>
  );
}

function Criteria({ v }: { v: VerdictData }) {
  return (
    <Card title="Criteria">
      <DataTable
        wrap
        header={["criterion", "status", "numbers"]}
        rows={v.criteria.map((c) => [c.criterion, <StatusBadge status={c.status} tone={c.tone} />, c.numbers])}
      />
      <FrozenRender renders={[v.renders.go]} />
    </Card>
  );
}

function Results({ v }: { v: VerdictData }) {
  return (
    <>
      <Card title="Results on test">
        <DataTable header={v.results.header} rows={v.results.rows} />
        <FrozenRender renders={[v.renders.final_table]} />
      </Card>
      <Card title="Settings">
        <DataTable header={v.settings.header} rows={v.settings.rows} />
      </Card>
    </>
  );
}

function Caveats({ v }: { v: VerdictData }) {
  return (
    <Card title="Caveats">
      <ul className={css.caveats}>
        {v.caveats.map((c, i) => (
          <li key={i}>{c}</li>
        ))}
      </ul>
    </Card>
  );
}

export default function VerdictView() {
  const state = useView("verdict");
  return (
    <ViewGate state={state}>
      {(v) => (
        <>
          <SourceCaption sources={v.sources} />
          <Hero v={v} />
          <Criteria v={v} />
          <Deltas v={v} />
          <Results v={v} />
          <Caveats v={v} />
        </>
      )}
    </ViewGate>
  );
}
