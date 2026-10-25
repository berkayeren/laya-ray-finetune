// "Val selection" (11-UI-SPEC.md "2. Val selection", D-10), from GET /api/selection, which serves both models' postfit
// records (results/postfit/laya.json, berturk.json and their final manifests) behind protocol.json's input pins,
// together with postfit.py's renders. Per model one card: the candidate epochs' val score (the exact score
// 2TP / (2TP + FP + FN)) by epoch as a line with 8px markers, the chosen epoch as a filled 12px marker with the direct
// label "chosen: epoch N", a crosshair tooltip and a "Show table" toggle with the same values; under the chart the
// settings tiles of the chosen candidate (T, θ shipped global, θ raw global, encoder / head LR) and, for Laya, the
// four grid-edge and bound lines of its val selection render, verbatim. Laya's card sits left, BERTurk's right, from a
// 960px content width. Every number is a field of the view; the UI only formats it.

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatMetric, formatTemperature, formatTheta } from "../format";
import { COLORS, Card, DataTable, ShowTable, SourceCaption, ViewGate, useView } from "./common";
import type { Candidate, Chosen, SelectionSystem } from "./types";
import css from "./dashboard.module.css";

type ModelKey = "laya" | "berturk";

const MODELS: { key: ModelKey; name: string; color: string; swatch: string }[] = [
  { key: "laya", name: "Laya", color: COLORS.laya, swatch: "var(--series-laya)" },
  { key: "berturk", name: "BERTurk", color: COLORS.berturk, swatch: "var(--series-berturk)" },
];

const MARKER_R = 4; // the 8px candidate marker
const CHOSEN_R = 6; // the filled 12px marker of the chosen epoch
const TICK_ALL = 10; // up to this many candidates, every epoch gets an x tick

// An LR as Python prints it in the records (2e-05, 0.0001): display formatting only.
function formatLR(x: number): string {
  return x !== 0 && Math.abs(x) < 1e-4 ? x.toExponential().replace(/e([+-])(\d)$/, "e$10$2") : String(x);
}

// --- the candidate-epoch chart ----------------------------------------------------------------------------------

// One chart row per epoch: the candidate of that epoch and its served score.
interface Row {
  epoch: number;
  c: Candidate | null;
  main: number | null;
}

function rowsOf(system: SelectionSystem): Row[] {
  return system.candidates.map((c) => ({ epoch: c.epoch, c, main: c.score }));
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
    </div>
  );
}

function CandidateChart({ system, color }: { system: SelectionSystem; color: string }) {
  const rows = rowsOf(system);
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
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function CandidateTable({ system }: { system: SelectionSystem }) {
  return (
    <DataTable
      header={["epoch", "score", "2TP / (2TP + FP + FN)", "val micro-F1 at the global θ", "T", "θ"]}
      rows={system.candidates.map((c) => [
        c.epoch === system.chosen.epoch ? `${c.epoch} (chosen)` : String(c.epoch),
        formatMetric(c.score),
        `${c.fraction[0]} / ${c.fraction[1]}`,
        formatMetric(c.val_micro_f1_at_global_theta),
        formatTemperature(c.temperature),
        formatTheta(c.theta_global),
      ])}
    />
  );
}

// --- the settings tiles and the edge lines ------------------------------------------------------------------------

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className={css.tile}>
      <p className={css.tileLabel}>{label}</p>
      <p className={css.tileValue}>{value}</p>
      {note !== undefined && <p className={css.tileNote}>{note}</p>}
    </div>
  );
}

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

// --- one model's card ---------------------------------------------------------------------------------------------

function ModelCard({ name, swatch, color, system }: { name: string; swatch: string; color: string; system: SelectionSystem }) {
  return (
    <Card>
      <h2 className={css.cardTitle}>
        <span className={css.swatch} style={{ background: swatch }} aria-hidden="true" />
        {name}
      </h2>
      <p className={css.chartCaption}>Val score 2TP / (2TP + FP + FN) by candidate epoch</p>
      <ShowTable chart={<CandidateChart system={system} color={color} />} table={<CandidateTable system={system} />} />
      <SettingsTiles chosen={system.chosen} />
      <EdgeLines lines={system.edge_lines} />
    </Card>
  );
}

export default function SelectionView() {
  const state = useView("selection");
  return (
    <ViewGate state={state}>
      {(v) => (
        <>
          <SourceCaption sources={v.sources} />
          <div className={css.pairWrap}>
            <div className={css.pair}>
              {MODELS.map((m) => (
                <div key={m.key} className={css.column}>
                  <ModelCard name={m.name} swatch={m.swatch} color={m.color} system={v.systems[m.key]} />
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </ViewGate>
  );
}
