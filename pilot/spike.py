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
import subprocess

import psutil

from pilot.data import write_atomic

# D-07: (micro_batch, max_len) grid, warm-up and measured optimizer steps per regime; the effective batch is held at
# ray_train.py's defaults (8 x 4) so every grid config projects over the same number of optimizer steps
DEFAULT_GRID = ((4, 128), (4, 256), (8, 128), (8, 256))
WARMUP_STEPS = 5
MEASURED_STEPS = 20
EFFECTIVE_BATCH = 32


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


def build_report(meta, configs):
    return {"schema": "mps_spike/1", "meta": meta, "configs": configs}


def _fmt(v, scale=1.0, digits=3):
    return "-" if v is None else ("%." + str(digits) + "f") % (v / scale)


def render_markdown(report):
    """Human-readable mps_spike.md: provenance lines, then one table row per attempt."""
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
             "| micro-batch | max_len | grad-accum | rung | fallback | status | padded median s | padded p90 s "
             "| real median s | real p90 s | peak GiB | swap delta MiB | swapouts delta |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["configs"]:
        for a in c["attempts"]:
            reg = a.get("regimes") or {}
            pad, real = reg.get("padded") or {}, reg.get("real") or {}
            pt, rt = pad.get("times") or {}, real.get("times") or {}
            swap = [r.get("swap_delta_bytes") for r in (pad, real) if r.get("swap_delta_bytes") is not None]
            outs = [r.get("swapouts_delta") for r in (pad, real) if r.get("swapouts_delta") is not None]
            lines.append("| %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                c["micro_batch"], c["max_len"], c["grad_accum"], a.get("rung", "-"), a.get("fallback", "-"),
                a.get("status", "-"), _fmt(pt.get("median_s")), _fmt(pt.get("p90_s")), _fmt(rt.get("median_s")),
                _fmt(rt.get("p90_s")), _fmt(a.get("peak_memory_bytes"), 2**30, 2),
                _fmt(max(swap) if swap else None, 2**20, 1), max(outs) if outs else "-"))
    return "\n".join(lines) + "\n"


def write_report(out_dir, report):
    """Write mps_spike.json and mps_spike.md atomically under out_dir; returns the markdown path."""
    os.makedirs(out_dir, exist_ok=True)
    write_atomic(os.path.join(out_dir, "mps_spike.json"),
                 (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    md = os.path.join(out_dir, "mps_spike.md")
    write_atomic(md, render_markdown(report).encode("utf-8"))
    return md
