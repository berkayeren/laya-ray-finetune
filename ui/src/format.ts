// The number formats of 11-UI-SPEC.md (Typography, "Number formats") as pure functions. They only format a value the
// server sent; no function here derives a threshold, an average or a confidence.

// Probabilities and message confidence: 3 decimals, "< 0.001" below 0.001 (a 2-decimal display could show a passing
// and a failing tag as the same value next to a θ of 0.95).
export function formatProbability(p: number): string {
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

// θ as recorded: 2 decimals (the θ grid step is 0.01).
export function formatTheta(x: number): string {
  return x.toFixed(2);
}

// T: 4 decimals, as the renders show it.
export function formatTemperature(x: number): string {
  return x.toFixed(4);
}

// Metrics (F1, AP, exact match): 3 decimals, as in "Results on test".
export function formatMetric(x: number): string {
  return x.toFixed(3);
}

// Latency: a duration in seconds shown in milliseconds with 1 decimal, as in latency.md.
export function formatMs(seconds: number): string {
  return (seconds * 1000).toFixed(1);
}

// Memory: GiB = bytes / 2^30, 2 decimals.
export function formatGiB(bytes: number): string {
  return (bytes / 2 ** 30).toFixed(2);
}

// Swap delta: MiB = bytes / 2^20, an integer.
export function formatMiB(bytes: number): string {
  return Math.round(bytes / 2 ** 20).toString();
}

// Train time: seconds, 1 decimal.
export function formatSeconds(s: number): string {
  return s.toFixed(1);
}
