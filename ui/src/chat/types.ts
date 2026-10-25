// The JSON shapes of the demo backend's chat endpoints (pilot/demo.py, demo.py), exactly as the server sends them.
//   GET /api/status    -> Status
//   GET /api/intents   -> Intents (tags in INTENTS order, the question as stored, the frozen confidence rule)
//   POST /api/classify -> ClassifyResponse ({"text": str} in; one Answer per system out) or {"error": str}
// Every array of length 48 in an Answer (probabilities, theta, own_theta) is in INTENTS order, the order of
// Intents.tags; ranked and predicted are tag names in the server's order.

export type ChatState = "loading" | "ready" | "refused";

export interface Status {
  chat: ChatState;
  message: string | null;
  device: string;
}

export interface IntentTag {
  tag: string;
  question: string;
}

export interface Intents {
  tags: IntentTag[];
  confidence_rule: string;
}

export interface RecordRef {
  path: string;
  sha256: string;
}

export type SystemKind = "laya" | "berturk";

export interface Answer {
  kind: SystemKind;
  seeds: number[];
  temperature: number;
  theta_global: number;
  n_own_theta: number;
  record: RecordRef;
  probabilities: number[];
  theta: number[];
  own_theta: boolean[];
  ranked: string[];
  predicted: string[];
  log_confidence: number;
  confidence: number;
}

export interface ClassifyResponse {
  laya: Answer;
  berturk: Answer;
}
