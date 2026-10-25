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

// --- GET /api/training and the training key of GET /api/status ---------------------------------------------------
// The rows come only from each final seed's checkpoint run record (Ray Train for Laya, the HF Trainer for BERTurk);
// a system whose checkpoints are missing or changed reads unavailable with the server's one line (D-09).

export type TrainingReadiness = "loading" | "ready";

// the one key of GET /api/status this view reads
export interface TrainingStatus {
  training: TrainingReadiness;
}

// one epoch row: {field: value} in the system's fields order; a value may be null (peak_driver_bytes off MPS)
export type EpochValue = number | string | boolean | null;
export type EpochRow = Record<string, EpochValue>;

export interface TrainingSeed {
  seed: number;
  checkpoint_name: string;
  train_run_id: string;
  device: string | null;
  n_train_messages: number | null;
  rows: EpochRow[];
}

export interface TrainingOk {
  status: "ok";
  fields: string[];
  total_train_s: number;
  seeds: TrainingSeed[];
}

export interface TrainingUnavailable {
  status: "unavailable";
  message: string;
}

export type TrainingSystem = TrainingOk | TrainingUnavailable;

export interface TrainingReady {
  status: "ready";
  systems: { laya: TrainingSystem; berturk: TrainingSystem };
}

export type TrainingData = { status: "loading" } | TrainingReady;

// --- GET /api/confidence --------------------------------------------------------------------------------------------

export interface Diagram {
  svg: string;
  path: string;
  run_id: string;
  split: string;
}

export interface ConfidenceData {
  sources: string[];
  val: Diagram[];
  test: Diagram[];
  caveats: string[];
}

// --- GET /api/latency -----------------------------------------------------------------------------------------------

// a latency/1 summary, in seconds
export interface LatencySummary {
  n: number;
  min: number;
  p50: number;
  p95: number;
  mean: number;
  max: number;
}

export interface LatencySeed {
  seed: number;
  checkpoint_name: string;
  summary: LatencySummary;
}

export interface LatencyReport {
  path: string;
  system: string;
  device: string;
  seeds: LatencySeed[];
  pooled: LatencySummary;
}

export interface LatencyData {
  sources: string[];
  label: string;
  reports: LatencyReport[];
  render: string;
}

// --- GET /api/fewshot -----------------------------------------------------------------------------------------------

export interface ApAtK {
  k: number;
  mean: number;
  mean_text: string;
  draws: number[];
}

export interface ForgettingRow {
  model: string;
  run: string;
  base_seed: number;
  new: number | null;
  old: number | null;
  forgetting: number | null;
}

export interface FewshotData {
  sources: string[];
  record: RecordRef;
  primary_k: number;
  mean_delta_primary: number;
  mean_delta_text: string;
  ap_by_k: { laya: ApAtK[]; berturk: ApAtK[] };
  forgetting: ForgettingRow[];
  caveat: string;
  notes: string[];
  render: string;
}

// --- the view endpoints -------------------------------------------------------------------------------------------

// GET /api/{name} -> its view type; useView(name) reads this map. GET /api/training is not here: it has its own
// loading protocol (TrainingView polls GET /api/status until training reads ready).
export interface ViewMap {
  verdict: VerdictData;
  selection: SelectionData;
  confidence: ConfidenceData;
  latency: LatencyData;
  fewshot: FewshotData;
}
