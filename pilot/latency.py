"""Torch-free bookkeeping of the Phase 7 latency harness (EVAL-06): per-message times -> latency/1 report -> markdown.

latency.py times every final seed of one system (Laya or BERTurk) on one device, in one process, without Ray; this
module holds what does not need a model: the constants of the timing contract, the timing loop itself (time_messages,
which takes the forward as a callable), the integer nearest-rank statistics, the final-manifest reader, the report
layout and its markdown. latency.py --render imports only this module to rebuild the markdown, so nothing here may
import torch, transformers, laya or Ray.

Timing contract (07-RESEARCH Pattern 5, Pitfall 7): batch 1, in-process; a message is one val JSONL row's state
string, passed unchanged to the checkpoint's own tokenizer; its latency is the wall-clock seconds (time.perf_counter)
from that string to 48 probabilities on the host, through the path that scored the dumps (fp32, autocast off, checked
after the load and after the loop). Warm-up (WARMUP_POLICY, G-07-6): after each seed's load, one untimed pass over
every message the seed then times, in split-file order, then one device sync; only then is every message timed; so
every seed follows the same sequence and every input it times has already run once on its model in this process; the
seeds run in manifest order, so a residual seed-order effect is not separated from a seed effect (07-REVIEW WR-01).
The device is synchronised right before t0 and right after the host copy of every timed message (torch.mps.synchronize
on mps, torch.cuda.synchronize on cuda, nothing on cpu). Only val messages are timed and only times are kept: no model
output, message text, uid list, timestamp or local path enters a report.

Statistics: nearest-rank percentiles with integer ranks, s[(q * n + 99) // 100 - 1] for an integer percent q (never
interpolated; never math.ceil(q / 100 * n), whose float product can land one rank off: 07-RESEARCH Pitfall 3). n 208:
p50 is the 104th and p95 the 198th smallest time. Latency is reported, not gated: LABEL is in every report and in the
markdown (REP-01 (c)); a latency budget is DEC-01 (v2). A report keeps the RULES text in force when it was timed;
render_markdown numbers the current RULES and marks every report whose recorded rules differ from them (rules_drift,
07-REVIEW WR-02).

usage: from pilot import latency
       times = latency.time_messages(fn, texts, sync)          # inside torch.no_grad(), right after the seed's load
       report = latency.build_report(meta, [dict(row, times_s=times, summary=latency.summarize(times, where))])
       print(latency.render_markdown(latency.load_reports("results/latency")))
"""
import json
import math
import os
import time

from pilot import spike
from pilot.data import sha256_bytes

SCHEMA = "latency/1"
LABEL = "reported, not gated"
WARMUP_POLICY = "full_pass_per_seed"    # after each seed's load, one untimed pass over every message it then times
QUANTILES = (50, 95)                                        # integer percents, nearest rank
THREADS = 4                                                 # torch threads (Phase 4 measured Laya on CPU at 4)
MIN_AVAILABLE_GIB = 7.0                                     # the compute checkpoint's quiet-machine threshold
MD_NAME = "latency.md"
FINAL_SCHEMA = "postfit-final/1"
SYSTEMS = ("laya", "berturk")
TIMED_PATH = {
    "laya": "laya.load (digests) -> la.force_fp32 -> la.noul_logits of one message (48 noul questions, one forward) "
            "-> sigmoid(margin / T) on the host",
    "berturk": "BertForSequenceClassification fp32 -> tokenizer of one message (truncated at the record max_len) -> "
               "one forward -> logits to host -> sigmoid(logit / T)",
}
RULES = [
    "batch 1, in-process, no Ray; one process per (system, device), every final seed of the final manifest",
    "a message is one val row's text, passed unchanged to the checkpoint's tokenizer; its latency is the wall-clock "
    "seconds from that text to 48 probabilities on the host",
    "warm-up (%s): after each seed's load, one untimed forward of every message the seed then times, in split-file "
    "order, then one device sync; only then is every message timed; so every seed follows the same sequence and every "
    "input it times has already run once on its model in this process; the seeds run in manifest order, so a residual "
    "seed-order effect is not separated from a seed effect; the report records the policy and the count"
    % WARMUP_POLICY,
    "the device is synchronised right before t0 and right after the host copy (torch.mps.synchronize on mps, "
    "torch.cuda.synchronize on cuda, nothing on cpu), under torch.no_grad",
    "fp32 with autocast off, checked after the load and after the timed loop",
    "val messages only; only times are kept (no outputs, text or uids)",
    "p50 and p95 are nearest-rank with integer ranks s[(q * n + 99) // 100 - 1], never interpolated",
]
RULES_DRIFT_LINE = ("- Recorded rules: this report's `rules` field differs from the rules above at rule(s) %s; it "
                    "keeps the text in force when the report was timed, and the rules above are the current wording")
SUMMARY_KEYS = ("n", "min", "p50", "p95", "mean", "max")


def nearest_rank(sorted_values, q):
    """The nearest-rank q-th percentile (q an integer percent in 1..100) of an ascending list: the value at integer
    rank (q * n + 99) // 100, i.e. sorted_values[(q * n + 99) // 100 - 1]."""
    n = len(sorted_values)
    if not n:
        raise SystemExit("nearest rank of an empty list; time at least one message")
    if isinstance(q, bool) or not isinstance(q, int) or not 0 < q <= 100:
        raise SystemExit("percentile %r is not an integer percent in 1..100" % (q,))
    return sorted_values[(q * n + 99) // 100 - 1]


def summarize(times, where):
    """{"n", "min", "p50", "p95", "mean", "max"} of per-message seconds (unrounded); SystemExit naming `where` (the
    system, seed and device summarised) when there is no timing."""
    if not times:
        raise SystemExit("%s: no timings to summarise" % where)
    s = sorted(float(t) for t in times)
    return {"n": len(s), "min": s[0], "p50": nearest_rank(s, 50), "p95": nearest_rank(s, 95),
            "mean": math.fsum(s) / len(s), "max": s[-1]}


def time_messages(fn, texts, sync, clock=time.perf_counter):
    """Per-message seconds of fn(text) over `texts`, in their order (split-file order).

    The warm-up pass (WARMUP_POLICY): fn runs once on every text, in order, untimed and its output discarded, then
    sync() once; the clock is first read after that. Then, for each text: sync(), t0 = clock(), fn(text), sync(),
    clock() - t0. fn must return host data (probabilities on the CPU), so the second sync and the host copy close the
    interval. No torch here: the caller wraps the call in torch.no_grad(), once per seed right after its load.
    """
    for t in texts:
        fn(t)
    sync()
    out = []
    for t in texts:
        sync()
        t0 = clock()
        fn(t)
        sync()
        out.append(clock() - t0)
    return out


def _plain_name(name):
    """True for a plain directory name: non-empty, no path separator, not made of dots only ('.', '..')."""
    return isinstance(name, str) and bool(name) and "/" not in name and "\\" not in name and set(name) != {"."} \
        and os.path.basename(name) == name


def read_final_manifest(path, system):
    """(final manifest, sha256 of its bytes) from one read; SystemExit unless it is a postfit-final/1 manifest of
    `system` (system_kind, "laya" when absent) whose non-empty seeds each carry an int seed, a train_run_id, a plain
    checkpoint_name (a directory under --checkpoint-root, never a path), a dict checkpoint_identity, a dump_run_id and
    a float temperature."""
    if system not in SYSTEMS:
        raise SystemExit("system %r is not one of %s" % (system, ", ".join(SYSTEMS)))
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit("final manifest %s cannot be read (%s); build it with `postfit.py --final-manifest`"
                         % (path, e.strerror or e))
    try:
        m = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SystemExit("final manifest %s is not valid JSON (%s); rebuild it with `postfit.py --final-manifest`"
                         % (path, e))
    if not isinstance(m, dict) or m.get("schema") != FINAL_SCHEMA:
        raise SystemExit("%s is not a %s final manifest; pass the manifest postfit.py --final-manifest wrote"
                         % (path, FINAL_SCHEMA))
    kind = m.get("system_kind", "laya")
    if kind != system:
        raise SystemExit("%s is a final manifest of %s, not %s; pass --final-manifest results/postfit/%s-final.json"
                         % (path, kind, system, system))
    seeds = m.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise SystemExit("%s lists no final seeds; rebuild it with `postfit.py --final-manifest`" % path)
    seen_seed, seen_name = set(), set()
    for i, s in enumerate(seeds):
        where = "%s seeds[%d]" % (path, i)
        if not isinstance(s, dict):
            raise SystemExit("%s is not an object; rebuild the manifest" % where)
        if isinstance(s.get("seed"), bool) or not isinstance(s.get("seed"), int):
            raise SystemExit("%s has no integer seed; rebuild the manifest" % where)
        for key in ("train_run_id", "dump_run_id"):
            if not isinstance(s.get(key), str) or not s[key]:
                raise SystemExit("%s has no %s; rebuild the manifest" % (where, key))
        if not _plain_name(s.get("checkpoint_name")):
            raise SystemExit("%s checkpoint_name %r is not a plain directory name (no path separator, not '.' or "
                             "'..'); a final seed is a directory under --checkpoint-root"
                             % (where, s.get("checkpoint_name")))
        if not isinstance(s.get("checkpoint_identity"), dict):
            raise SystemExit("%s has no checkpoint_identity; rebuild the manifest" % where)
        if not isinstance(s.get("temperature"), float):
            raise SystemExit("%s has no float temperature; rebuild the manifest" % where)
        if s["seed"] in seen_seed or s["checkpoint_name"] in seen_name:
            raise SystemExit("%s repeats seed %d or checkpoint %s; rebuild the manifest"
                             % (where, s["seed"], s["checkpoint_name"]))
        seen_seed.add(s["seed"])
        seen_name.add(s["checkpoint_name"])
    return m, sha256_bytes(raw)


def build_report(meta, seeds):
    """The latency/1 report: schema, LABEL, RULES, the system's TIMED_PATH, the run's `meta` (system, device, threads,
    warm-up policy and count, versions, code state, preflight, swap, split and final-manifest provenance), the
    per-seed rows and a summary pooled over every seed's times. SystemExit unless meta records warmup_policy
    WARMUP_POLICY and a warm-up count equal to every seed's timed count (the full pass)."""
    if meta.get("warmup_policy") != WARMUP_POLICY:
        raise SystemExit("latency meta records warm-up policy %r, not %s (one untimed pass over every timed message "
                         "per seed); rerun latency.py" % (meta.get("warmup_policy"), WARMUP_POLICY))
    w = meta.get("warmup")
    if isinstance(w, bool) or not isinstance(w, int) or any(w != len(s["times_s"]) for s in seeds):
        raise SystemExit("latency meta records %r warm-up message(s) per seed, but the seeds timed %s message(s); "
                         "under %s the warm-up count is every seed's timed count; rerun latency.py"
                         % (w, sorted({len(s["times_s"]) for s in seeds}), WARMUP_POLICY))
    system, device = meta["system"], meta["device"]
    report = {"schema": SCHEMA, "label": LABEL, "rules": list(RULES), "timed_path": TIMED_PATH[system]}
    report.update((k, v) for k, v in meta.items() if k not in report)
    report["seeds"] = seeds
    report["pooled"] = summarize([t for s in seeds for t in s["times_s"]],
                                 "%s pooled over %d seed(s) on %s" % (system, len(seeds), device))
    return report


def swap_grew(report):
    """True when swap in use grew across the timed loops of a report."""
    swap = report.get("swap_used_bytes") or {}
    before, after = swap.get("before"), swap.get("after")
    return before is not None and after is not None and after > before


def rules_drift(report):
    """The ascending 1-based positions at which a report's recorded `rules` differ from RULES (a position on one side
    only counts; a missing or non-list `rules` is empty); [] when it records RULES exactly. A report keeps the rules
    text in force when it was timed, so a later wording change shows here and in render_markdown (07-REVIEW WR-02)."""
    rec = report.get("rules")
    rec = rec if isinstance(rec, list) else []
    return [i for i in range(1, max(len(rec), len(RULES)) + 1)
            if (rec[i - 1] if i <= len(rec) else None) != (RULES[i - 1] if i <= len(RULES) else None)]


def _ms(v):
    return "%.1f" % (1000.0 * v)


def _row(label, checkpoint, s):
    return "| %s | %s | %d | %s |" % (label, checkpoint, s["n"],
                                      " | ".join(_ms(s[k]) for k in ("p50", "p95", "mean", "min", "max")))


def _report_lines(r):
    v = r.get("versions") or {}
    code = r.get("code") or {}
    sha = code.get("git_sha")
    lines = ["", "#### %s on %s" % (r["system"], r["device"]), "",
             "Timed path: %s." % r["timed_path"], ""]
    if spike.busy_machine(r.get("preflight")) or swap_grew(r):
        lines += ["**Busy-machine run:** the preflight was overridden, swap grew while idle, or swap grew during the "
                  "timed loops; these times may be slower than on a quiet machine.", ""]
    lines.append("- Device %s, %s, autocast %s, torch threads %d, batch 1; warm-up %s: one untimed pass over all %d "
                 "message(s) per seed before its timed loop"
                 % (r["device"], r.get("dtype"), "on" if r.get("amp_enabled") else "off", r["threads"],
                    r.get("warmup_policy"), r["warmup"]))
    drift = rules_drift(r)
    if drift:
        lines.append(RULES_DRIFT_LINE % ", ".join(str(i) for i in drift))
    lines.append("- Versions: %s; PYTORCH_ENABLE_MPS_FALLBACK %s"
                 % (", ".join("%s %s" % kv for kv in sorted(v.items())) or "-", r.get("mps_fallback_env") or "unset"))
    lines.append("- Code: %s%s" % (sha[:12] if sha else "no git",
                                   ", uncommitted changes" if code.get("git_dirty") else ""))
    sp, fm = r.get("split") or {}, r.get("final_manifest") or {}
    lines.append("- Messages: %s val messages of %s (sha256 %s); final manifest %s (sha256 %s, committed: %s)"
                 % (sp.get("n_messages"), sp.get("path"), (sp.get("sha256") or "-")[:12], fm.get("path"),
                    (fm.get("sha256") or "-")[:12], fm.get("is_committed")))
    pf, swap = r.get("preflight"), r.get("swap_used_bytes") or {}
    if pf:
        lines.append("- Preflight: %.2f GiB available (threshold %s%s), swap grew while idle: %s"
                     % (pf["available_gib"], pf["min_available_gib"], ", overridden" if pf["override"] else "",
                        pf["swap_grew_while_idle"]))
    if swap.get("before") is not None and swap.get("after") is not None:
        lines.append("- Swap in use: %.0f MiB before, %.0f MiB after the timed loops"
                     % (swap["before"] / 2**20, swap["after"] / 2**20))
    lines += ["", "| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |",
              "|---|---|---|---|---|---|---|---|"]
    for s in r["seeds"]:
        lines.append(_row(str(s["seed"]), s["checkpoint_name"], s["summary"]))
    lines.append(_row("pooled", "%d seed(s)" % len(r["seeds"]), r["pooled"]))
    return lines


def render_markdown(reports):
    """Markdown of latency/1 reports (sorted by system, then device): the label, the current RULES as a numbered list,
    then per report the timed path, a busy-machine banner when the preflight was overridden or swap grew (idle or
    during the timed loops), the provenance (with a Recorded rules line, RULES_DRIFT_LINE, when the report's recorded
    rules differ from RULES: rules_drift) and one row per seed plus a pooled row. Milliseconds with one decimal, for
    display only."""
    lines = ["### Latency per message (Phase 7, EVAL-06): %s" % LABEL, "",
             "Latency is %s: it is not a decision criterion (a latency budget is DEC-01, v2)." % LABEL, "",
             "Rules:", ""] + ["%d. %s" % (i, rule) for i, rule in enumerate(RULES, 1)]
    for r in sorted(reports, key=lambda r: (r["system"], r["device"])):
        lines += _report_lines(r)
    lines += ["", "Laya's own fp16 autocast default on MPS (laya_default_amp) is not what is timed: every run is fp32 "
              "with autocast off, the path that scored the dumps."]
    return "\n".join(lines) + "\n"


def load_reports(path):
    """The latency/1 reports of a directory's *.json, in sorted path order; SystemExit on a missing or empty
    directory, on a *.json of another schema, or on a report without warmup_policy WARMUP_POLICY (timed before the
    full-pass warm-up, so it is never rendered next to one timed under it)."""
    if not os.path.isdir(path):
        raise SystemExit("%s is not a directory of latency reports; run latency.py --out DIR/NAME.json first" % path)
    names = sorted(n for n in os.listdir(path) if n.endswith(".json"))
    if not names:
        raise SystemExit("%s holds no latency report (*.json); run latency.py --out first" % path)
    out = []
    for n in names:
        p = os.path.join(path, n)
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except ValueError as e:
            raise SystemExit("%s is not valid JSON (%s); rerun latency.py" % (p, e))
        if not isinstance(r, dict) or r.get("schema") != SCHEMA:
            raise SystemExit("%s is not a %s report; keep only latency reports in %s" % (p, SCHEMA, path))
        if r.get("warmup_policy") != WARMUP_POLICY:
            raise SystemExit("%s was timed before the full-pass warm-up policy (warmup_policy %r, not %s: one untimed "
                             "pass over every timed message per seed); regenerate it with latency.py"
                             % (p, r.get("warmup_policy"), WARMUP_POLICY))
        out.append(r)
    return out
