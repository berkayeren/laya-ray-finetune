// The dashboard's shared pieces (11-UI-SPEC.md "Dashboard view", "Color" status badges, "Interaction Contract",
// "UI Considerations"). No component here writes a status word, a verdict sentence, a caveat or a number of its own:
// each one shows the strings and values its props carry, as React text nodes.
//
//   useView(name)      GET /api/{name} until it answers 200: loading | ok {data} | error {error, source} | unreachable
//   ViewGate           renders the loading block, the per-view error or the unreachable copy, else its children(data)
//   SourceCaption      the one-line mono caption naming the records a view reads (the view's own sources field)
//   Card               white, hairline border, radius 8px, 16px padding; an optional h2 title
//   LoadingCard        a grey block of the chart height, shown until an endpoint answers
//   ViewError          "This view is unavailable: {error}. ..." in that view only: the server's line verbatim (it
//                      names the record it is about and ends in its own fix), with no file or command of the UI's
//   StatusBadge        24px badge: the tone's text, fill and border colours and its icon, the status word as received
//   FrozenRender       "View the frozen render" disclosure opening the exact render text in a preformatted block
//   SegmentedControl   role="radiogroup"; arrow keys move the selection; the selected segment in the accent
//   ShowTable          the "Show table" toggle: swaps a chart for a table of the same values
//   DataTable          header + rows; a wide table scrolls inside its card, never the page
//   Tile               a stat tile: label, value and an optional note
//   COLORS             the series and chart-ink hex values of the UI-SPEC (SVG attributes cannot read CSS variables)
//
// Each dashboard view opens with the h1 that Dashboard.tsx renders, then its SourceCaption.

import { useEffect, useId, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";
import { getJSON } from "../http";
import { AlertTriangleIcon, CheckIcon, ChevronDownIcon, ChevronRightIcon, InfoIcon, XIcon } from "../icons";
import type { Tone, ViewMap } from "./types";
import css from "./dashboard.module.css";

export const COLORS = {
  laya: "#2a78d6",
  berturk: "#eb6834",
  superseded: "#898781",
  reference: "#52514e",
  ink: "#0b0b0b",
  inkSecondary: "#52514e",
  inkMuted: "#6b6a66",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
} as const;

const UNREACHABLE_COPY =
  "The demo server is not responding. Check that `make demo` is still running in your terminal; this page reconnects " +
  "when it answers again.";
const RETRY_MS = 1000; // the first retry after an unanswered request
const RETRY_MAX_MS = 5000; // the backoff cap, and the interval at which a refused view asks again

// --- data -----------------------------------------------------------------------------------------------------------

export type ViewState<T> =
  | { kind: "loading" }
  | { kind: "ok"; data: T }
  | { kind: "error"; error: string; source: string | null }
  | { kind: "unreachable" };

// GET /api/{name}, typed by ViewMap, fetched on mount and again until it answers 200: an unanswered request after
// RETRY_MS, doubling up to RETRY_MAX_MS, and an HTTP error every RETRY_MAX_MS. The views are built once at server
// startup, so a refused view (whose line ends in its fix and a restart) shows the restarted server's view without a
// reload. A 503 carries the record view's {error, source}; any other HTTP error has no source (null), never an
// invented one.
export function useView<K extends keyof ViewMap>(name: K): ViewState<ViewMap[K]> {
  const [state, setState] = useState<ViewState<ViewMap[K]>>({ kind: "loading" });
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let retry = RETRY_MS;
    setState({ kind: "loading" });
    const fetchView = async (): Promise<void> => {
      const r = await getJSON<ViewMap[K]>("/api/" + name);
      if (cancelled) {
        return;
      }
      if (r.ok) {
        setState({ kind: "ok", data: r.data });
      } else if (r.status === null) {
        setState({ kind: "unreachable" });
        timer = window.setTimeout(() => void fetchView(), retry);
        retry = Math.min(retry * 2, RETRY_MAX_MS);
      } else {
        retry = RETRY_MS;
        setState({ kind: "error", error: r.error, source: r.source });
        timer = window.setTimeout(() => void fetchView(), RETRY_MAX_MS);
      }
    };
    void fetchView();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [name]);
  return state;
}

interface ViewGateProps<T> {
  state: ViewState<T>;
  height?: string;
  children: (data: T) => ReactNode;
}

export function ViewGate<T>({ state, height, children }: ViewGateProps<T>) {
  if (state.kind === "loading") {
    return <LoadingCard height={height} />;
  }
  if (state.kind === "error") {
    return <ViewError error={state.error} source={state.source} />;
  }
  if (state.kind === "unreachable") {
    return (
      <div className={css.notice} role="alert">
        <p className={css.noticeBody}>{UNREACHABLE_COPY}</p>
      </div>
    );
  }
  return <>{children(state.data)}</>;
}

// --- frame ----------------------------------------------------------------------------------------------------------

export function SourceCaption({ sources }: { sources: string[] }) {
  return <p className={css.sources}>{sources.join(" · ")}</p>;
}

interface CardProps {
  title?: string;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}

export function Card({ title, className, style, children }: CardProps) {
  return (
    <section className={className === undefined ? css.card : css.card + " " + className} style={style}>
      {title !== undefined && <h2 className={css.cardTitle}>{title}</h2>}
      {children}
    </section>
  );
}

export function LoadingCard({ height }: { height?: string }) {
  return (
    <div
      className={css.loading}
      style={{ height: height ?? "var(--chart-height)" }}
      role="status"
      aria-label="Loading"
    />
  );
}

// The server's line names the record it is about and already ends in its fix (a restore, a rebuild or a restart), so
// it is shown alone: the view's source field is its primary record (pilot.demo.VIEW_SOURCES), not necessarily the
// file a refusal is about (a changed berturk-reference.json breaks the selection view, whose source is laya.json), so
// it is never shown as the failing file. An error without a source is no record view's refusal (an HTTP error with
// no line of the server's), so only it gets the restart hint. The views are built once at startup, and useView asks
// again, so a fixed record shows once `make demo` restarts.
export function ViewError({ error, source }: { error: string; source: string | null }) {
  return (
    <div className={css.notice} role="alert">
      <p className={css.noticeBody}>
        {`This view is unavailable: ${error}. The other views are unaffected.`}
        {source === null && (
          <>
            {" If it repeats, restart "}
            <code className={css.code}>make demo</code>
            {"."}
          </>
        )}
      </p>
    </div>
  );
}

// --- status ---------------------------------------------------------------------------------------------------------

const TONE_ICON: Record<Tone, () => ReactNode> = {
  positive: CheckIcon,
  negative: XIcon,
  caution: AlertTriangleIcon,
  neutral: InfoIcon,
};

function knownTone(tone: string): Tone {
  return tone === "positive" || tone === "negative" || tone === "caution" ? tone : "neutral";
}

// The tone comes from the server (pilot.demo.tone); the word is the record's string unchanged.
export function StatusBadge({ status, tone }: { status: string; tone: string }) {
  const t = knownTone(tone);
  const Icon = TONE_ICON[t];
  return (
    <span
      className={css.badge}
      style={{
        color: `var(--tone-${t}-text)`,
        background: `var(--tone-${t}-fill)`,
        borderColor: `var(--tone-${t}-border)`,
      }}
    >
      <Icon />
      {status}
    </span>
  );
}

// --- frozen render --------------------------------------------------------------------------------------------------

// renders: one or more exact render texts, each in its own preformatted block, never parsed or reflowed
export function FrozenRender({ renders }: { renders: string[] }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className={css.frozen}>
      <button
        type="button"
        className={css.disclosure}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDownIcon /> : <ChevronRightIcon />}
        View the frozen render
      </button>
      <div id={id} hidden={!open}>
        {renders.map((text, i) => (
          <pre key={i} className={css.pre}>
            {text}
          </pre>
        ))}
      </div>
    </div>
  );
}

// --- controls -------------------------------------------------------------------------------------------------------

export interface Segment<T extends string> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string> {
  label: string;
  options: Segment<T>[];
  value: T;
  onChange: (value: T) => void;
}

// Arrow keys (and Home/End) move the selection and the focus; only the selected segment is in the tab order.
export function SegmentedControl<T extends string>({ label, options, value, onChange }: SegmentedControlProps<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const current = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );
  const move = (e: KeyboardEvent<HTMLDivElement>): void => {
    const n = options.length;
    const to =
      e.key === "ArrowRight" || e.key === "ArrowDown"
        ? (current + 1) % n
        : e.key === "ArrowLeft" || e.key === "ArrowUp"
          ? (current - 1 + n) % n
          : e.key === "Home"
            ? 0
            : e.key === "End"
              ? n - 1
              : null;
    if (to === null) {
      return;
    }
    e.preventDefault();
    onChange(options[to].value);
    refs.current[to]?.focus();
  };
  return (
    <div className={css.segmented} role="radiogroup" aria-label={label} onKeyDown={move}>
      {options.map((o, i) => (
        <button
          key={o.value}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="radio"
          aria-checked={i === current}
          tabIndex={i === current ? 0 : -1}
          className={i === current ? css.segment + " " + css.segmentOn : css.segment}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// The chart, or (toggled) a table of the same values: the accessible, exact form of every chart.
export function ShowTable({ chart, table }: { chart: ReactNode; table: ReactNode }) {
  const [shown, setShown] = useState(false);
  return (
    <div className={css.showTable}>
      <div className={css.toggleRow}>
        <button type="button" className={css.toggle} aria-pressed={shown} onClick={() => setShown((s) => !s)}>
          Show table
        </button>
      </div>
      {shown ? table : chart}
    </div>
  );
}

interface DataTableProps {
  header: ReactNode[];
  rows: ReactNode[][];
  caption?: string;
  wrap?: boolean;
}

// wrap false (the default) keeps each cell on one line, so a wide table scrolls inside its card
export function DataTable({ header, rows, caption, wrap = false }: DataTableProps) {
  return (
    <div className={css.tableScroll}>
      <table className={wrap ? css.table + " " + css.tableWrap : css.table}>
        {caption !== undefined && <caption className={css.tableCaption}>{caption}</caption>}
        <thead>
          <tr>
            {header.map((h, i) => (
              <th key={i} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td key={c}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// A stat tile: the label, the value (heading size, tabular figures) and an optional muted note.
export function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className={css.tile}>
      <p className={css.tileLabel}>{label}</p>
      <p className={css.tileValue}>{value}</p>
      {note !== undefined && <p className={css.tileNote}>{note}</p>}
    </div>
  );
}
