// "Val selection" (11-UI-SPEC.md "2. Val selection", D-10), from GET /api/selection, which serves both models' postfit
// records (results/postfit/laya.json, berturk.json and their final manifests) behind protocol.json's input pins,
// together with postfit.py's renders. Per model one card: the candidate epochs' val score (the exact score
// 2TP / (2TP + FP + FN)) by epoch as a line with 8px markers, the chosen epoch as a filled 12px marker with the direct
// label "chosen: epoch N", a crosshair tooltip and a "Show table" toggle with the same values; under the chart the
// settings tiles of the chosen candidate (T, θ shipped global, θ raw global, encoder / head LR) and, for Laya, the
// four grid-edge and bound lines of its val selection render, verbatim. Laya's card sits left, BERTurk's right, from a
// 960px content width. Under BERTurk's card: the published reference check (the fraction and its decimal on a 0..1 bar
// against the dashed threshold line, the status badge, the threshold rule and the quoted reference, its url as plain
// text). "Show superseded runs" (off by default) adds the two discarded BERTurk selection runs to BERTurk's chart and
// table as grey dashed lines with the server's own label, never in a model's colour and never marked chosen, and one
// grey "Superseded, never current" card per run with its change sentence and the user's choice, verbatim.
// Every number is a field of the view; the UI only formats it.

import { useId, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatMetric, formatTemperature, formatTheta } from "../format";
import { COLORS, Card, DataTable, ShowTable, SourceCaption, StatusBadge, Tile, ViewGate, useView } from "./common";
import type { Candidate, Chosen, ReferenceCheck, SelectionSystem, SupersededRun } from "./types";
import css from "./dashboard.module.css";

const MARKER_R = 4; // the 8px candidate marker
const CHOSEN_R = 6; // the filled 12px marker of the chosen epoch
const TICK_ALL = 10; // up to this many candidates, every epoch gets an x tick
const SUPERSEDED_DASH = "6 4";

// An LR as Python prints it in the records (2e-05, 0.0001): display formatting only. null (a record whose canonical
// config holds no head_lr, e.g. a superseded-protocol record) reads n/a.
function formatLR(x: number | null): string {
  if (x === null) {
    return "n/a";
  }
  return x !== 0 && Math.abs(x) < 1e-4 ? x.toExponential().replace(/e([+-])(\d)$/, "e$10$2") : String(x);
}

// --- the candidate-epoch chart ----------------------------------------------------------------------------------

// One chart row per epoch: the candidate of that epoch and its served score, then each shown superseded run's served
// score at that epoch (null where the run has no candidate of that epoch).
interface Row {
  epoch: number;
  c: Candidate | null;
  main: number | null;
  sup: { label: string; score: number | null }[];
}

function rowsOf(system: SelectionSystem, runs: SupersededRun[]): Row[] {
  const epochs = [...new Set([...system.candidates, ...runs.flatMap((r) => r.candidates)].map((c) => c.epoch))];
  epochs.sort((a, b) => a - b);
  return epochs.map((epoch) => {
    const c = system.candidates.find((x) => x.epoch === epoch) ?? null;
    return {
      epoch,
      c,
      main: c === null ? null : c.score,
      sup: runs.map((r) => ({ label: r.label, score: r.candidates.find((x) => x.epoch === epoch)?.score ?? null })),
    };
  });
}

interface DotArgs {
  cx?: number;
  cy?: number;
  index: number;
  payload?: Row;
}

function candidateDot(color: string, chosen: Chosen, nRows: number) {
  return function CandidateDot({ cx, cy, index, payload }: DotArgs) {
    if (cx === undefined || cy === undefined || payload?.c == null) {
      return null;
    }
    if (payload.c.epoch !== chosen.epoch) {
      return <circle cx={cx} cy={cy} r={MARKER_R} fill="#ffffff" stroke={color} strokeWidth={2} />;
    }
    // the direct label sits above the marker, anchored away from the nearer chart edge
    const end = index >= nRows / 2;
    return (
      <g>
        <circle cx={cx} cy={cy} r={CHOSEN_R} fill={color} stroke={color} strokeWidth={2} />
        <text
          x={end ? cx + CHOSEN_R : cx - CHOSEN_R}
          y={cy - CHOSEN_R - 6}
          textAnchor={end ? "end" : "start"}
          fill={COLORS.inkSecondary}
          fontSize={14}
          fontWeight={600}
          stroke="#ffffff"
          strokeWidth={3}
          paintOrder="stroke"
        >
          {`chosen: epoch ${chosen.epoch}`}
        </text>
      </g>
    );
  };
}

// A superseded run's line has no markers; its last candidate carries the run's served label, right of the point.
function supersededLabel(run: SupersededRun) {
  const last = run.candidates[run.candidates.length - 1]?.epoch;
  return function SupersededEnd({ cx, cy, payload }: DotArgs) {
    if (cx === undefined || cy === undefined || payload?.epoch !== last) {
      return null;
    }
    return (
      <text
        x={cx + 8}
        y={cy + 4}
        textAnchor="start"
        fill={COLORS.inkSecondary}
        fontSize={14}
        stroke="#ffffff"
        strokeWidth={3}
        paintOrder="stroke"
      >
        {run.label}
      </text>
    );
  };
}

interface TipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: Row }>;
}

function SelectionTip({ active, payload }: TipProps) {
  const row = active ? payload?.[0]?.payload : undefined;
  if (row === undefined) {
    return null;
  }
  return (
    <div className={css.chartTip}>
      <p className={css.chartTipTitle}>{`epoch ${row.epoch}`}</p>
      {row.c !== null && (
        <>
          <p className={css.chartTipLine}>{`score ${formatMetric(row.c.score)}`}</p>
          <p className={css.chartTipLine}>
            {`val micro-F1 at the global θ ${formatMetric(row.c.val_micro_f1_at_global_theta)}`}
          </p>
          <p className={css.chartTipLine}>{`T ${formatTemperature(row.c.temperature)}`}</p>
          <p className={css.chartTipLine}>{`θ ${formatTheta(row.c.theta_global)}`}</p>
        </>
      )}
      {row.sup.map((s) =>
        s.score === null ? null : (
          <p key={s.label} className={css.chartTipLine}>{`${s.label}: score ${formatMetric(s.score)}`}</p>
        ),
      )}
    </div>
  );
}

interface ChartProps {
  system: SelectionSystem;
  color: string;
  runs: SupersededRun[];
}

function CandidateChart({ system, color, runs }: ChartProps) {
  const rows = rowsOf(system, runs);
  // every epoch of a short grid; a long grid (BERTurk's 30) labels its first epoch and every fifth
  const epochs = system.candidates.map((c) => c.epoch);
  const ticks = epochs.length <= TICK_ALL ? epochs : epochs.filter((e, i) => i === 0 || e % 5 === 0);
  return (
    <div className={css.chart} style={{ height: "var(--chart-height)" }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 28, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis
            type="number"
            dataKey="epoch"
            domain={["dataMin", "dataMax"]}
            ticks={ticks}
            interval={0}
            padding={{ left: 12, right: 12 }}
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <YAxis
            type="number"
            domain={["auto", "auto"]}
            tickFormatter={formatMetric}
            width={56}
            tick={{ fill: COLORS.inkMuted, fontSize: 14 }}
            stroke={COLORS.axis}
            tickLine={false}
          />
          <Tooltip content={SelectionTip} cursor={{ stroke: COLORS.axis }} isAnimationActive={false} />
          <Line
            type="linear"
            dataKey={(r: Row) => r.main}
            stroke={color}
            strokeWidth={2}
            dot={candidateDot(color, system.chosen, rows.length)}
            activeDot={{ r: CHOSEN_R, fill: color, stroke: "#ffffff", strokeWidth: 2 }}
            isAnimationActive={false}
          />
          {runs.map((run, i) => (
            <Line
              key={run.name}
              type="linear"
              dataKey={(r: Row) => r.sup[i]?.score ?? null}
              stroke={COLORS.superseded}
              strokeWidth={2}
              strokeDasharray={SUPERSEDED_DASH}
              dot={supersededLabel(run)}
              activeDot={{ r: MARKER_R, fill: COLORS.superseded, stroke: "#ffffff", strokeWidth: 2 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// The chart's values: one row per epoch; with superseded runs shown, one more score column per run (its label).
function CandidateTable({ system, runs }: { system: SelectionSystem; runs: SupersededRun[] }) {
  return (
    <DataTable
      header={[
        "epoch",
        "score",
        "2TP / (2TP + FP + FN)",
        "val micro-F1 at the global θ",
        "T",
        "θ",
        ...runs.map((r) => r.label),
      ]}
      rows={rowsOf(system, runs).map(({ epoch, c, sup }) => [
        c !== null && epoch === system.chosen.epoch ? `${epoch} (chosen)` : String(epoch),
        c === null ? "" : formatMetric(c.score),
        c === null ? "" : `${c.fraction[0]} / ${c.fraction[1]}`,
        c === null ? "" : formatMetric(c.val_micro_f1_at_global_theta),
        c === null ? "" : formatTemperature(c.temperature),
        c === null ? "" : formatTheta(c.theta_global),
        ...sup.map((x) => (x.score === null ? "" : formatMetric(x.score))),
      ])}
    />
  );
}

// --- the settings tiles and the edge lines ------------------------------------------------------------------------

function SettingsTiles({ chosen }: { chosen: Chosen }) {
  return (
    <div className={css.tiles}>
      <Tile label="T" value={formatTemperature(chosen.temperature)} />
      <Tile
        label="θ shipped global"
        value={formatTheta(chosen.theta_shipped_global)}
        note={`${chosen.n_own_theta} tags with their own θ`}
      />
      <Tile
        label="θ raw global"
        value={formatTheta(chosen.theta_raw_global)}
        note={`${chosen.n_own_theta_raw} tags with their own θ`}
      />
      <Tile label="encoder / head LR" value={`${formatLR(chosen.encoder_lr)} / ${formatLR(chosen.head_lr)}`} />
    </div>
  );
}

function EdgeLines({ lines }: { lines: string[] }) {
  if (lines.length === 0) {
    return null;
  }
  return (
    <div className={css.edgeBlock}>
      <p className={css.edgeTitle}>Grid edges and bounds</p>
      <ul className={css.edgeLines}>
        {lines.map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ul>
    </div>
  );
}

// --- the BERTurk reference check ------------------------------------------------------------------------------------

// The served fraction and value on a 0..1 bar, the threshold as a dashed reference line labelled with its served value;
// the bar positions are the served numbers as percentages of the bar width.
function ReferenceCard({ r }: { r: ReferenceCheck }) {
  const valueText = `${r.fraction[0]}/${r.fraction[1]} = ${r.value.toFixed(4)}`;
  const threshold = String(r.threshold);
  return (
    <Card title="BERTurk reference check">
      <div className={css.refHead}>
        <p className={css.refValue}>{valueText}</p>
        <StatusBadge status={r.status} tone={r.tone} />
      </div>
      <div className={css.refBar} role="img" aria-label={`${valueText}; reference line at ${threshold}`}>
        <span className={css.refLineLabel} style={{ left: `${r.threshold * 100}%` }}>
          {threshold}
        </span>
        <div className={css.refTrack}>
          <div className={css.refFill} style={{ width: `${r.value * 100}%` }} />
        </div>
        <span className={css.refMarker} style={{ left: `${r.value * 100}%` }} />
        <span className={css.refLine} style={{ left: `${r.threshold * 100}%` }} />
      </div>
      <div className={css.refScale} aria-hidden="true">
        <span>0</span>
        <span>1</span>
      </div>
      <p className={css.refRule}>{r.threshold_rule}</p>
      <dl className={css.refQuote}>
        <dt>paper</dt>
        <dd>{r.reference.paper}</dd>
        <dt>venue</dt>
        <dd>{r.reference.venue}</dd>
        <dt>metric</dt>
        <dd>{r.reference.metric}</dd>
        <dt>url</dt>
        <dd className={css.mono}>{r.reference.url}</dd>
      </dl>
    </Card>
  );
}

// --- the superseded runs ------------------------------------------------------------------------------------------

function SupersededToggle({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  const id = useId();
  return (
    <div className={css.checkRow}>
      <input id={id} type="checkbox" className={css.check} checked={on} onChange={(e) => onChange(e.target.checked)} />
      <label htmlFor={id} className={css.checkLabel}>
        Show superseded runs
      </label>
    </div>
  );
}

function SupersededCard({ run }: { run: SupersededRun }) {
  const u = run.user_choice;
  return (
    <Card title="Superseded, never current" className={css.supersededCard}>
      <p className={css.mono}>{run.name}</p>
      <p className={css.supersededValue}>{`${run.fraction[0]}/${run.fraction[1]} = ${run.value.toFixed(4)}`}</p>
      <p className={css.supersededText}>{run.change}</p>
      <p className={css.supersededText}>{`${u.date}: ${u.decision} (${u.reply})`}</p>
    </Card>
  );
}

// --- one model's card ---------------------------------------------------------------------------------------------

interface ModelCardProps {
  name: string;
  swatch: string;
  color: string;
  system: SelectionSystem;
  runs: SupersededRun[];
  toggle?: { on: boolean; onChange: (on: boolean) => void };
}

function ModelCard({ name, swatch, color, system, runs, toggle }: ModelCardProps) {
  return (
    <Card>
      <h2 className={css.cardTitle}>
        <span className={css.swatch} style={{ background: swatch }} aria-hidden="true" />
        {name}
      </h2>
      <p className={css.chartCaption}>Val score 2TP / (2TP + FP + FN) by candidate epoch</p>
      {toggle !== undefined && <SupersededToggle on={toggle.on} onChange={toggle.onChange} />}
      <ShowTable
        chart={<CandidateChart system={system} color={color} runs={runs} />}
        table={<CandidateTable system={system} runs={runs} />}
      />
      <SettingsTiles chosen={system.chosen} />
      <EdgeLines lines={system.edge_lines} />
    </Card>
  );
}

export default function SelectionView() {
  const state = useView("selection");
  const [showSuperseded, setShowSuperseded] = useState(false);
  return (
    <ViewGate state={state}>
      {(v) => {
        const runs = showSuperseded ? v.superseded : [];
        return (
          <>
            <SourceCaption sources={v.sources} />
            <div className={css.pairWrap}>
              <div className={css.pair}>
                <div className={css.column}>
                  <ModelCard name="Laya" swatch="var(--series-laya)" color={COLORS.laya} system={v.systems.laya} runs={[]} />
                </div>
                <div className={css.column}>
                  <ModelCard
                    name="BERTurk"
                    swatch="var(--series-berturk)"
                    color={COLORS.berturk}
                    system={v.systems.berturk}
                    runs={runs}
                    toggle={{ on: showSuperseded, onChange: setShowSuperseded }}
                  />
                  {runs.map((run) => (
                    <SupersededCard key={run.name} run={run} />
                  ))}
                  <ReferenceCard r={v.reference} />
                </div>
              </div>
            </div>
          </>
        );
      }}
    </ViewGate>
  );
}
