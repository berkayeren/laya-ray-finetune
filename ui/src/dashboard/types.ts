// The dashboard's record-view contracts, exactly as pilot/demo.py builds them (plan 11-04): GET /api/verdict and
// GET /api/selection answer 200 with the view, or 503 with ViewError {error, source}. Every string here is shown as
// the server sends it; the UI derives no status, verdict sentence, threshold or temperature from these fields.

// The display tone the server computes for a status (pilot.demo.tone): never derived in the UI.
export type Tone = "positive" | "negative" | "caution" | "neutral";

// A refused record view: the one-line `<what>; <fix>` and the repo-relative record it read.
export interface ViewError {
  error: string;
  source: string;
}

export interface Table {
  header: string[];
  rows: string[][];
}

// --- GET /api/verdict ---------------------------------------------------------------------------------------------

export interface VerdictLines {
  verdict: string;
  rule: string;
  means: string;
  recommendation: string;
}

export interface Criterion {
  criterion: string;
  status: string;
  tone: Tone;
  numbers: string;
}

export interface Delta {
  criterion: string;
  metric: string;
  delta_points: number;
  ci_points: [number, number];
  delta_text: string;
  ci_text: string;
  direction: string;
  status: string | null;
}

export interface VerdictRenders {
  go: string;
  final_table: string;
  recommendation: string;
}

export interface VerdictData {
  sources: string[];
  verdict: string;
  means: string;
  tone: Tone;
  lines: VerdictLines;
  criteria: Criterion[];
  deltas: Delta[];
  results: Table;
  settings: Table;
  caveats: string[];
  renders: VerdictRenders;
}

// --- GET /api/selection -------------------------------------------------------------------------------------------

export interface RecordRef {
  path: string;
  sha256: string;
}

export interface Candidate {
  epoch: number;
  fraction: [number, number];
  score: number;
  val_micro_f1_at_global_theta: number;
  temperature: number;
  theta_global: number;
}

export interface Chosen {
  epoch: number;
  epochs_schedule: number;
  temperature: number;
  theta_shipped_global: number;
  n_own_theta: number;
  theta_raw_global: number;
  n_own_theta_raw: number;
  encoder_lr: number;
  head_lr: number;
}

export interface SelectionSystem {
  kind: string;
  record: RecordRef;
  candidates: Candidate[];
  chosen: Chosen;
  edge_lines: string[];
  render: string;
}

export interface PublishedReference {
  paper: string;
  venue: string;
  url: string;
  metric: string;
  [key: string]: unknown;
}

export interface ReferenceCheck {
  record: RecordRef;
  fraction: [number, number];
  value: number;
  threshold: number;
  status: string;
  tone: Tone;
  reference: PublishedReference;
  threshold_rule: string;
}

export interface UserChoice {
  date: string;
  decision: string;
  reply: string;
}

export interface SupersededRun {
  name: string;
  label: string;
  fraction: [number, number];
  value: number;
  change: string;
  user_choice: UserChoice;
  candidates: { epoch: number; score: number }[];
}

export interface SelectionData {
  sources: string[];
  systems: { laya: SelectionSystem; berturk: SelectionSystem };
  reference: ReferenceCheck;
  superseded: SupersededRun[];
}

// --- the view endpoints -------------------------------------------------------------------------------------------

// GET /api/{name} -> its view type; useView(name) reads this map (plans 11-11 and 11-12 add their views here)
export interface ViewMap {
  verdict: VerdictData;
  selection: SelectionData;
}
