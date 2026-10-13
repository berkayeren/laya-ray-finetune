"""Torch-free logic of the Phase 2 MPS feasibility spike (LAYA-09).

mps_spike.py imports this module in the orchestrator, before any child process sets PYTORCH_ENABLE_MPS_FALLBACK
(torch reads it once, at import), so nothing here may import torch. It holds the grid and protocol constants, the
worker env, neg8 sampling, worst-case padding, the timing summary, the swap readers, the child outcome mapping and
the report builders. Every measurement itself happens in mps_spike.py's worker.

usage: imported by mps_spike.py and tests/test_spike_unit.py
"""
import json
import math
import os
import random
import re
import subprocess

import psutil

from pilot.data import write_atomic

# D-07: (micro_batch, max_len) grid, warm-up and measured optimizer steps per regime; the effective batch is held at
# ray_train.py's defaults (8 x 4) so every grid config projects over the same number of optimizer steps
DEFAULT_GRID = ((4, 128), (4, 256), (8, 128), (8, 256))
WARMUP_STEPS = 5
MEASURED_STEPS = 20
EFFECTIVE_BATCH = 32
# D-02: projected selection-train neg8 epoch limit, seconds
TIME_GATE_S = 7200
# D-04/D-08: the only rungs that may count as FITS, in preference order (embedding freeze is a fallback rung only)
RUNGS = ("none", "embeddings_frozen")
# D-03 tie-break: configs within this fraction of the fastest projected epoch count as tied
TIE_TOLERANCE = 0.05


def parse_grid(text):
    """"4x128,8x256" -> [(4, 128), (8, 256)]; ValueError on an empty, malformed, non-positive or duplicate entry."""
    out = []
    for tok in (text or "").split(","):
        parts = tok.strip().lower().split("x")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            raise ValueError("grid entry %r is not MICRO_BATCHxMAX_LEN (e.g. 4x128)" % tok)
        mb, max_len = int(parts[0]), int(parts[1])
        if mb < 1 or max_len < 1:
            raise ValueError("grid entry %r must have positive numbers" % tok)
        if (mb, max_len) in out:
            raise ValueError("grid entry %r is repeated" % tok)
        out.append((mb, max_len))
    return out


def spike_env(fallback, cpus):
    """Env every spike worker must see (ray.init runtime_env and the child process env)."""
    if fallback not in ("0", "1"):
        raise ValueError("fallback must be '0' or '1', got %r" % (fallback,))
    return {"PYTORCH_ENABLE_MPS_FALLBACK": fallback,
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0",
            # LOW must be set too: HIGH=1.0 alone crashes the first MPS call with "invalid low watermark ratio 1.4"
            # (the unified-memory LOW default exceeds HIGH; RESEARCH Pitfall 1)
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "1.0",
            "OMP_NUM_THREADS": str(cpus),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1"}


def neg8_rows(rows, seed, k=8):
    """Copies of `rows` that label every positive tag plus k random negatives (one random.Random(seed) per pass).

    laya.train.items_from_rows skips questions with no `expected` entry without counting them, so the dropped
    negatives simply produce no item (D-06b). The input rows are never mutated.
    """
    rng = random.Random(seed)
    out = []
    for r in rows:
        pos = [t for t, v in r["expected"].items() if v]
        neg = sorted(t for t, v in r["expected"].items() if not v)
        keep = pos + rng.sample(neg, min(k, len(neg)))
        out.append(dict(r, expected={t: r["expected"][t] for t in keep}))
    return out


def pad_item_to(item, max_len):
    """Copy of a laya item whose state is long enough that build_sequence emits exactly max_len tokens.

    Real tokens repeated, not pad tokens, so the attention mask stays all ones (the honest long-message worst case).
    """
    ids = item["state_ids"] or [5]
    return dict(item, state_ids=(ids * (max_len // len(ids) + 1))[:max_len])


def summarize(times):
    """{"n", "median_s", "p90_s", "min_s", "max_s"} of step durations (p90 nearest-rank); {"n": 0} when empty."""
    if not times:
        return {"n": 0}
    s = sorted(times)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "median_s": round(median, 4), "p90_s": round(s[math.ceil(0.9 * n) - 1], 4),
            "min_s": round(s[0], 4), "max_s": round(s[-1], 4)}


def swap_used():
    """System swap in use, bytes (psutil equals sysctl vm.swapusage "used"); the D-01 gate value."""
    return psutil.swap_memory().used


def swapouts():
    """Monotonic macOS swap-out counter, or None where sysctl is missing or fails (a corroborating signal only).

    Never psutil.swap_memory().sout: on macOS it counts file-backed pageouts, not swap.
    """
    try:
        r = subprocess.run(["sysctl", "-n", "vm.compressor.swapper.swapouts_total"],
                           capture_output=True, text=True, check=True, shell=False)
        return int(r.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def scrub(text):
    """Exception text safe for a tracked report: the home directory becomes "~", cut to 300 characters."""
    home = os.path.expanduser("~")
    text = str(text)
    if len(home) > 1:
        text = text.replace(home, "~")
    return text[:300]


def child_outcome(returncode, payload):
    """Map a child run to its attempt: a measurement passes through, a refusal stops the spike, anything else crashed.

    Refusals are configuration errors (wrong env, params off the device, tampered input), never measurements.
    """
    if isinstance(payload, dict) and payload.get("status") == "refused":
        raise SystemExit("spike run refused: %s" % payload.get("reason"))
    if returncode == 0 and isinstance(payload, dict):
        return payload
    return {"status": "crashed", "returncode": returncode}


def optimizer_steps(n_items, micro_batch, grad_accum):
    """Optimizer steps per epoch, the count laya.train.optimizer_updates(n, mb, accum, 1) gives (the last partial
    accumulation window still steps once)."""
    if n_items < 1:
        raise ValueError("n_items must be a positive integer, got %r" % (n_items,))
    return math.ceil(math.ceil(n_items / micro_batch) / grad_accum)


def project_epoch_s(median_step_s, steps):
    """D-02 projection: real-batch median optimizer-step time x optimizer steps per epoch, seconds."""
    return round(median_step_s * steps, 1)


def is_mps_oom(text):
    """True for the MPS allocator's out-of-memory text (a plain RuntimeError; torch.OutOfMemoryError misses it)."""
    return str(text).lstrip().startswith("MPS backend out of memory")


def memory_ok(attempt, measured_steps):
    """D-01 memory gate -> (ok, reason): every measured padded step ran without OOM and system swap did not grow."""
    status = attempt.get("status")
    if status != "ok":
        if status == "oom":
            o = attempt.get("oom") or {}
            return False, "oom in %s %s step %s" % (o.get("regime"), o.get("stage"), o.get("step"))
        if status == "crashed":
            return False, "crashed (exit %s)" % attempt.get("returncode")
        if status == "error":
            return False, "error: %s" % scrub(attempt.get("error", ""))
        return False, str(status)
    pad = (attempt.get("regimes") or {}).get("padded") or {}
    n = (pad.get("times") or {}).get("n", 0)
    if n < measured_steps:
        return False, "completed %d of %d measured padded steps" % (n, measured_steps)
    swap = pad.get("swap_delta_bytes")
    if swap is None:
        return False, "no swap reading for the measured padded steps"
    if swap > 0:
        return False, "system swap grew by %.1f MiB during the measured padded steps (D-01)" % (swap / 2**20)
    return True, "completed %d measured padded steps without OOM or swap growth" % measured_steps


# D-10: torch names the op the same way in both texts (verified on this torch build):
#   strict (fallback 0) NotImplementedError: "The operator 'aten::linalg_eig' is not currently implemented for the MPS
#   device. ..."; fallback 1 UserWarning: "The operator 'aten::linalg_eig' is not currently supported on the MPS
#   backend and will fall back to run on the CPU. ..."
OP_RE = re.compile(r"operator '([^']+)'")
FALLBACK_MARK = "will fall back to run on the CPU"


def op_name(text):
    """The op an MPS NotImplementedError or fallback warning names ("aten::..."), or None."""
    m = OP_RE.search(str(text))
    return m.group(1) if m else None


def is_fallback_warning(text):
    """True for torch's "falls back to the CPU" UserWarning (PYTORCH_ENABLE_MPS_FALLBACK=1)."""
    return FALLBACK_MARK in str(text)


def tag_fallback_ops(messages, marks):
    """[{"op", "stage"}] for the fallback warnings among `messages`, deduplicated by (op, stage), first appearance.

    marks: (index, stage) pairs in ascending index order, recorded as len(caught) when the worker enters a stage, so
    the message at position i belongs to the last mark whose index is <= i.
    """
    out, seen = [], set()
    for i, text in enumerate(messages):
        if not is_fallback_warning(text):
            continue
        stage = "setup"
        for idx, name in marks:
            if idx > i:
                break
            stage = name
        key = (op_name(text) or "unknown op", stage)
        if key not in seen:
            seen.add(key)
            out.append({"op": key[0], "stage": key[1]})
    return out


def blocks(stage, fallback_median_s, mps_median_s, time_ok):
    """D-11 -> (blocks, reason): a step-path op blocks when its fallback run is more than 10% slower than an all-MPS
    run of the same config and rung or, with no all-MPS baseline, when the fallback run fails the D-02 time gate."""
    if stage != "step":
        return False, "seen only in %s; setup and tokenization ops never block (D-11)" % stage
    if mps_median_s is not None:
        if fallback_median_s is None:
            return True, "the fallback attempt measured no real-batch step to compare with all-MPS %.3f s" % mps_median_s
        slower = fallback_median_s > 1.10 * mps_median_s
        return slower, "fallback step %.3f s vs all-MPS %.3f s (%+.1f%%)" % (
            fallback_median_s, mps_median_s, (fallback_median_s / mps_median_s - 1) * 100)
    return not time_ok, "no all-MPS baseline; the config %s the D-02 time gate" % ("passes" if time_ok else "fails")


def followup(chain, policy):
    """The next attempt a config's chain needs, as {"rung", "fallback", "purpose"}, or None when the chain is done.

    D-10: a strict (fallback 0) attempt that raised NotImplementedError is rerun once with fallback 1, in a fresh
    process, to inventory every op that falls back. (Plan 02-05 adds the swap-rerun and embeddings-frozen rules.)
    """
    last = chain[-1]
    if last.get("status") == "not_implemented" and last.get("fallback") == "0":
        return {"rung": last.get("rung"), "fallback": "1", "purpose": "op_inventory"}
    return None


def standing(chain):
    """Index of the attempt the verdict reads: the last one of the chain."""
    return len(chain) - 1


def _real_median(a):
    t = ((a.get("regimes") or {}).get("real") or {}).get("times") or {}
    return t.get("median_s") if t.get("n") else None


def _projected(a, micro_batch, grad_accum):
    """Projected neg8 epoch seconds of an attempt (D-02), or None without a real-batch timing or an item count."""
    median, n_items = _real_median(a), a.get("n_neg8_items")
    if median is None or not n_items:
        return None
    return project_epoch_s(median, optimizer_steps(n_items, micro_batch, grad_accum))


def op_inventory(configs):
    """D-10/D-11 inventory: every op a strict attempt raised on or its fallback rerun fell back on, with blocks/reason.

    Returns {"entries": [{op, stage, micro_batch, max_len, rung, blocks, reason}], "note"}.
    """
    entries, raised = [], 0
    for c in configs:
        mb, max_len, accum = c["micro_batch"], c["max_len"], c["grad_accum"]
        attempts = c["attempts"]
        failed = [a for a in attempts if a.get("fallback") == "0" and a.get("status") == "not_implemented"]
        if failed:
            raised += 1
        done = set()
        for strict in failed:
            rung = strict.get("rung")
            if rung in done:
                continue
            done.add(rung)
            err = strict.get("error_op") or {}
            found = [{"op": err.get("op") or "unknown op", "stage": err.get("stage") or "setup"}]
            fb = next((a for a in attempts if a.get("fallback") == "1" and a.get("rung") == rung), None)
            if fb is None:
                for f in found:
                    entries.append(dict(f, micro_batch=mb, max_len=max_len, rung=rung, blocks=True,
                                        reason="no fallback attempt ran"))
                continue
            for f in fb.get("fallback_ops") or []:
                if (f["op"], f["stage"]) not in {(g["op"], g["stage"]) for g in found}:
                    found.append({"op": f["op"], "stage": f["stage"]})
            base = next((a for a in attempts if a.get("fallback") == "0" and a.get("rung") == rung
                         and a.get("status") == "ok" and _real_median(a) is not None), None)
            projected = _projected(fb, mb, accum)
            time_ok = projected is not None and projected <= TIME_GATE_S
            for f in found:
                b, reason = blocks(f["stage"], _real_median(fb), _real_median(base) if base else None, time_ok)
                entries.append(dict(f, micro_batch=mb, max_len=max_len, rung=rung, blocks=b, reason=reason))
    note = ("strict PYTORCH_ENABLE_MPS_FALLBACK=0 pass raised nowhere: no op falls back from MPS" if not raised else
            "%d config(s) raised under the strict pass; inventory from their fallback reruns" % raised)
    return {"entries": entries, "note": note}


def blocking_ops(inventory):
    """{(micro_batch, max_len, rung): [op, ...]} of the blocking entries, the verdict's `blocking` argument."""
    out = {}
    for e in inventory["entries"]:
        if e["blocks"]:
            ops = out.setdefault((e["micro_batch"], e["max_len"], e["rung"]), [])
            if e["op"] not in ops:
                ops.append(e["op"])
    return out


def protocol_complete(protocol):
    """True only for the full D-07 protocol; anything else is a smoke run that can never decide Phase 2."""
    grid = protocol["grid"]
    if isinstance(grid, str):
        grid = parse_grid(grid)
    return (sorted(tuple(g) for g in grid) == sorted(DEFAULT_GRID) and protocol["warmup_steps"] == WARMUP_STEPS
            and protocol["measured_steps"] == MEASURED_STEPS and protocol["effective_batch"] == EFFECTIVE_BATCH)


RULES = {
    "D-01": "memory: every measured padded step completes on MPS with no OOM and no system swap growth",
    "D-02": "time: real-batch median optimizer step x optimizer steps per neg8 epoch <= %d s" % TIME_GATE_S,
    "D-03": "choice: fastest projected epoch among FITS configs with max_len >= 128; configs within the tie tolerance "
            "resolve to the smaller max_len, then the larger micro-batch; > 1% of messages over 128 tokens prefers a "
            "passing max_len 256",
    "D-04": "rung: only a full-encoder or embeddings-frozen run on MPS counts; a frozen encoder never FITS",
    "D-08": "preference: rung none configs win over embeddings_frozen ones (embedding freeze is a fallback rung only)",
}


def _assess(c, measured_steps, blocking):
    a = c["attempts"][c.get("standing", 0)]
    mb, max_len, accum = c["micro_batch"], c["max_len"], c["grad_accum"]
    rung, fallback = a.get("rung"), a.get("fallback")
    mem, mem_reason = memory_ok(a, measured_steps)
    n_items = a.get("n_neg8_items")
    steps = optimizer_steps(n_items, mb, accum) if n_items else None
    real = ((a.get("regimes") or {}).get("real") or {}).get("times") or {}
    projected = (project_epoch_s(real["median_s"], steps)
                 if steps and real.get("n") and real.get("median_s") is not None else None)
    time_ok = projected is not None and projected <= TIME_GATE_S
    device_ok = a.get("param_device") == "mps"
    rung_ok = rung in RUNGS
    blocked = list(blocking.get((mb, max_len, rung), []))
    fails = [] if mem else [mem_reason]
    if projected is None:
        fails.append("no real-batch step timing to project an epoch (D-02)")
    elif not time_ok:
        fails.append("projected epoch %.1f min exceeds %d min (D-02)" % (projected / 60, TIME_GATE_S // 60))
    if not device_ok:
        fails.append("parameters on %s, not mps (D-04)" % a.get("param_device"))
    if not rung_ok and rung is not None:
        fails.append("rung %s never counts as FITS (D-04)" % rung)
    elif rung is None:
        fails.append("rung not recorded (D-04)")
    if blocked:
        fails.append("blocking fallback ops: %s (D-11)" % ", ".join(blocked))
    fits = not fails
    return {"micro_batch": mb, "max_len": max_len, "grad_accum": accum, "rung": rung, "fallback": fallback,
            "status": a.get("status"), "memory_ok": mem, "memory_reason": mem_reason, "steps_per_epoch": steps,
            "projected_epoch_s": projected, "time_ok": time_ok, "device_ok": device_ok, "rung_ok": rung_ok,
            "blocking_ops": blocked, "peak_memory_bytes": a.get("peak_memory_bytes"), "fits": fits,
            "reason": "FITS: all gates passed" if fits else "; ".join(fails)}


def verdict(configs, protocol, token_stats=None, tie_tolerance=TIE_TOLERANCE, blocking=None):
    """The Phase 2 outcome (LAYA-09 criterion 4) from the standing attempt of every grid config.

    FITS needs the D-01 memory gate, the D-02 time gate, parameters on mps, an allowed rung (D-04) and no blocking
    fallback op. The Phase 4 start config is chosen by D-08 (rung none first), D-03 (truncation, fastest) and the
    tie-break. A run off the D-07 protocol is a smoke run and never asks for a Kaggle plan.
    """
    blocking = blocking or {}
    complete = protocol_complete(protocol)
    per = [_assess(c, protocol["measured_steps"], blocking) for c in configs]
    pool = [p for p in per if p["fits"] and p["max_len"] >= 128]
    if any(p["rung"] == "none" for p in pool):
        pool = [p for p in pool if p["rung"] == "none"]
    if token_stats is None:
        truncation = "not applied: token stats not computed"
    else:
        pct = token_stats["pct_messages_over_128"]
        if pct > 1.0:
            truncation = "applied: %.1f%% of messages over 128 tokens (> 1%%, a passing max_len 256 is preferred)" % pct
            if any(p["max_len"] >= 256 for p in pool):
                pool = [p for p in pool if p["max_len"] >= 256]
        else:
            truncation = "applied: %.1f%% of messages over 128 tokens (<= 1%%, max_len 128 is allowed)" % pct
    chosen = None
    if pool:
        fastest = min(p["projected_epoch_s"] for p in pool)
        tied = [p for p in pool if p["projected_epoch_s"] <= fastest * (1 + tie_tolerance)]
        best = sorted(tied, key=lambda p: (p["max_len"], -p["micro_batch"], p["projected_epoch_s"]))[0]
        chosen = {k: best[k] for k in ("micro_batch", "max_len", "grad_accum", "rung", "fallback",
                                       "projected_epoch_s", "peak_memory_bytes")}
    result = "FITS" if chosen else "DOES NOT FIT"
    untrusted = sum(1 for c in configs for a in c["attempts"] if a.get("status") in ("error", "crashed"))
    return {"result": result, "protocol_complete": complete, "chosen": chosen,
            "kaggle_plan_required": result == "DOES NOT FIT" and complete, "tie_tolerance": tie_tolerance,
            "truncation_rule": truncation, "untrusted_attempts": untrusted, "per_config": per, "rules": dict(RULES)}


def verdict_line(v):
    """The last line of mps_spike.md (and of the spike's stdout); a smoke run can never pass for the Phase 2 verdict."""
    c = v["chosen"]
    if c:
        line = ("**Verdict: FITS** — Phase 4 starts from micro-batch %d, max_len %d, grad-accum %d, rung %s, "
                "PYTORCH_ENABLE_MPS_FALLBACK=%s (projected neg8 epoch %.1f min)"
                % (c["micro_batch"], c["max_len"], c["grad_accum"], c["rung"], c["fallback"],
                   c["projected_epoch_s"] / 60))
    else:
        line = "**Verdict: DOES NOT FIT** — no grid config passed the D-01 memory and D-02 time gates"
    return line if v["protocol_complete"] else "SMOKE RUN (not the Phase 2 verdict): " + line


def _yn(ok):
    return "yes" if ok else "no"


def _verdict_lines(v):
    lines = ["", "## Verdict", "",
             "| micro-batch | max_len | rung | memory ok | projected epoch min | time ok | device ok | FITS |",
             "|---|---|---|---|---|---|---|---|"]
    for p in v["per_config"]:
        lines.append("| %d | %d | %s | %s (%s) | %s | %s | %s | %s |" % (
            p["micro_batch"], p["max_len"], p["rung"] or "-", _yn(p["memory_ok"]), p["memory_reason"].replace("|", "/"),
            _fmt(p["projected_epoch_s"], 60, 1), _yn(p["time_ok"]), _yn(p["device_ok"]),
            "yes" if p["fits"] else "no: " + p["reason"].replace("|", "/")))
    lines += ["",
              "- Tie-break: FITS configs within %.0f%% of the fastest projected epoch resolve to the smaller max_len, "
              "then the larger micro-batch" % (v["tie_tolerance"] * 100),
              "- Truncation rule (D-03): %s" % v["truncation_rule"]]
    lines += ["- %s: %s" % (k, v["rules"][k]) for k in sorted(v["rules"])]
    if v["untrusted_attempts"]:
        lines += ["", "%d attempts ended in an unexpected error or crash; investigate before trusting this verdict"
                  % v["untrusted_attempts"]]
    if v["kaggle_plan_required"]:
        lines += ["", "Kaggle plan required: kaggle_plan.md next to this report, plus the user's explicit "
                      "self-hosting exception before Phase 3 (D-12)"]
    return lines + ["", verdict_line(v)]


def _inventory_lines(inv):
    lines = ["", "## Fallback-op inventory", "", inv["note"]]
    if inv["entries"]:
        lines += ["", "| op | stage | config | rung | blocks | reason |", "|---|---|---|---|---|---|"]
        lines += ["| `%s` | %s | %dx%d | %s | %s | %s |" % (e["op"], e["stage"], e["micro_batch"], e["max_len"],
                                                          e["rung"], _yn(e["blocks"]), e["reason"].replace("|", "/"))
                  for e in inv["entries"]]
    return lines


def build_report(meta, configs):
    return {"schema": "mps_spike/1", "meta": meta, "configs": configs}


def _fmt(v, scale=1.0, digits=3):
    return "-" if v is None else ("%." + str(digits) + "f") % (v / scale)


def render_markdown(report):
    """Human-readable mps_spike.md: provenance lines, one table row per attempt, then the verdict section."""
    m = report["meta"]
    split, model, versions, plat, proto = m["split"], m["model"], m["versions"], m["platform"], m["protocol"]
    lines = ["# MPS feasibility spike (Phase 2, LAYA-09)", "",
             "- Split: `%s` sha256 `%s` (%d messages)" % (split["path"], split["sha256"], split["n_messages"]),
             "- Model: `%s/%s@%s`" % (model["repo"], model["subfolder"], model["revision"]),
             "- Versions: %s" % ", ".join("%s %s" % (k, versions[k]) for k in sorted(versions)),
             "- macOS %s, %s, %.1f GiB memory" % (plat["macos"], plat["machine"], plat["total_memory_bytes"] / 2**30),
             "- Protocol: grid %s, %d warm-up + %d measured optimizer steps per regime, effective batch %d, seed %d, "
             "%d CPUs" % (proto["grid"], proto["warmup_steps"], proto["measured_steps"], proto["effective_batch"],
                          proto["seed"], proto["cpus"]),
             "",
             "| micro-batch | max_len | grad-accum | rung | fallback | purpose | status | padded median s "
             "| padded p90 s | real median s | real p90 s | peak GiB | swap delta MiB | swapouts delta |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["configs"]:
        for a in c["attempts"]:
            reg = a.get("regimes") or {}
            pad, real = reg.get("padded") or {}, reg.get("real") or {}
            pt, rt = pad.get("times") or {}, real.get("times") or {}
            swap = [r.get("swap_delta_bytes") for r in (pad, real) if r.get("swap_delta_bytes") is not None]
            outs = [r.get("swapouts_delta") for r in (pad, real) if r.get("swapouts_delta") is not None]
            lines.append("| %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                c["micro_batch"], c["max_len"], c["grad_accum"], a.get("rung", "-"), a.get("fallback", "-"),
                a.get("purpose", "-"), a.get("status", "-"), _fmt(pt.get("median_s")), _fmt(pt.get("p90_s")), _fmt(rt.get("median_s")),
                _fmt(rt.get("p90_s")), _fmt(a.get("peak_memory_bytes"), 2**30, 2),
                _fmt(max(swap) if swap else None, 2**20, 1), max(outs) if outs else "-"))
    if report.get("op_inventory"):
        lines += _inventory_lines(report["op_inventory"])
    if report.get("verdict"):
        lines += _verdict_lines(report["verdict"])
    return "\n".join(lines) + "\n"


def write_report(out_dir, report):
    """Write mps_spike.json and mps_spike.md atomically under out_dir; returns the markdown path."""
    os.makedirs(out_dir, exist_ok=True)
    write_atomic(os.path.join(out_dir, "mps_spike.json"),
                 (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    md = os.path.join(out_dir, "mps_spike.md")
    write_atomic(md, render_markdown(report).encode("utf-8"))
    return md
