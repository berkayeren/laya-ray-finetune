"""Torch-free logic of the Phase 4 compute checkpoint (LAYA-07): the measured run record -> a projected runtime for the
whole experiment matrix -> compute_checkpoint.json + compute_checkpoint.md.

compute_checkpoint.py imports this module in the orchestrator, before its ray_train.py child sets the MPS env (torch
reads PYTORCH_ENABLE_MPS_FALLBACK once, at import), so nothing here may import torch. The measurement itself happens in
the child; this module only reads its checkpoint config (training.ray_train.record, pilot.runs.read_record), counts the
items of the committed splits without a tokenizer (pilot.runs.neg_items) and projects.

Projection rate (RESEARCH Pitfall 6): s_per_step is the LARGEST train_s / optimizer_steps over the measured epochs, so
the MPS warm-up of epoch 1 (or any slower later epoch) makes the projection conservative, never optimistic. A Laya run
of E epochs on n items costs E x (optimizer_steps(n) x s_per_step + save_s_per_epoch) + load_s + calib_s +
final_save_s, every term measured by the probe run.

usage: from pilot import compute; start = compute.spike_start("results/spike/mps_spike.json")
       measured = compute.measured_from_checkpoint(ckpt_dir); counts = compute.split_counts(manifest)
       projection = compute.project(measured, counts, compute.MATRIX)
       compute.write_report(out_dir, compute.build_report(meta, measured, counts, compute.MATRIX, projection, pf))
"""
import json
import os

from pilot import runs, spike
from pilot.data import parse_jsonl, verify_split_file, write_atomic

SCHEMA = "compute_checkpoint/1"
JSON_NAME, MD_NAME = "compute_checkpoint.json", "compute_checkpoint.md"

# The experiment matrix the projection covers. Every value is provisional (RESEARCH A2) until Phases 5, 6 and 8 freeze
# it; the user confirms or trims them at the compute checkpoint (plan 04-08).
MATRIX = {"lr_grid": 3, "max_epochs": 4, "final_seeds": 3}
MATRIX_SOURCES = {
    "lr_grid": "FEATURES TS-12 example shared LR grid {2e-5, 3e-5, 5e-5} (3 values)",
    "max_epochs": "ray_train.py default --epochs 4, the ceiling of the val-selected epoch",
    "final_seeds": "LAYA-08 / BASE-01: three final seeds per model",
}
# the training split each projected line counts its items on (all messages: Phase 5 calibrates on val, LAYA-05)
TRAIN_SPLITS = ("selection_train", "final_train")


def spike_start(path):
    """verdict.chosen of the Phase 2 spike report as {micro_batch, max_len, grad_accum, rung, fallback}; refuses any
    verdict but FITS, and a chosen config that needs the MPS CPU fallback (ray_train.py trains strict only)."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    v = report.get("verdict") or {}
    if v.get("result") != "FITS" or not v.get("chosen"):
        raise SystemExit("%s: the spike verdict is %s, not FITS; the compute checkpoint starts only from a FITS config"
                         % (path, v.get("result")))
    c = v["chosen"]
    if str(c.get("fallback")) != "0":
        raise SystemExit("%s: the chosen config needs PYTORCH_ENABLE_MPS_FALLBACK=%s; ray_train.py trains with the "
                         "strict spike env (fallback 0) only" % (path, c.get("fallback")))
    return {"micro_batch": int(c["micro_batch"]), "max_len": int(c["max_len"]), "grad_accum": int(c["grad_accum"]),
            "rung": c["rung"], "fallback": 0}


def split_counts(manifest_path, names=TRAIN_SPLITS):
    """{split: {n_messages, n_items_neg8}} for the registered splits `names`, each verified on the exact bytes counted
    (pilot.data.verify_split_file) and counted torch-free (pilot.runs.neg_items)."""
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)["splits"]
    out = {}
    for name in names:
        path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), entries[name]["path"])
        with open(path, "rb") as f:
            raw = f.read()
        verify_split_file(path, manifest_path=manifest_path, data=raw)
        rows = parse_jsonl(raw)
        out[name] = {"n_messages": len(rows), "n_items_neg8": runs.neg_items(rows, 8)}
    return out


def _largest(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


def measured_from_checkpoint(ckpt_dir):
    """The probe run's measured inputs from a ray_train.py checkpoint dir: {"laya": rates, overheads, memory and run
    config, "epochs": the record's epoch rows unchanged (LAYA-06)}."""
    record = runs.read_record(ckpt_dir)
    with open(os.path.join(ckpt_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    train = (cfg.get("training") or {}).get("ray_train") or {}
    rows = record.get("epochs") or []
    if not rows:
        raise SystemExit("the probe run record holds no epoch row; nothing was measured")
    rates = [r["train_s"] / r["optimizer_steps"] for r in rows]
    laya = {"s_per_step": max(rates), "s_per_step_epoch1": rates[0], "epochs_measured": len(rows),
            "save_s_per_epoch": _largest(rows, "save_s"), "load_s": record["load_s"], "calib_s": record["calib_s"],
            "final_save_s": record["final_save_s"], "peak_total_bytes": _largest(rows, "peak_total_bytes"),
            "peak_driver_bytes": _largest(rows, "peak_driver_bytes"), "peak_rss_bytes": _largest(rows, "peak_rss_bytes"),
            "swap_delta_bytes": _largest(rows, "swap_delta_bytes"), "micro_batch": train["micro_batch"],
            "grad_accum": train["grad_accum"], "max_len": cfg["max_len"], "neg": train["neg"],
            "rung": record["rung"], "fallback": record["fallback"], "weaker": record["weaker"],
            "device": record["device"], "param_device": record["param_device"],
            "grad_checkpointing": record["grad_checkpointing"], "amp": record["amp"], "env": list(record["env"]),
            "train_state_bytes": record.get("train_state_bytes")}
    return {"laya": laya, "epochs": rows}


def laya_run_s(n_items, epochs, laya):
    """Seconds of one Laya run of `epochs` epochs on `n_items` items at the measured rate and overheads."""
    steps = spike.optimizer_steps(n_items, laya["micro_batch"], laya["grad_accum"])
    return epochs * (steps * laya["s_per_step"] + laya["save_s_per_epoch"]) + laya["load_s"] + laya["calib_s"] + \
        laya["final_save_s"]


def _line(id_, phase, what, runs_, epochs, units, unit, s_per_unit, basis, source, seconds, formula):
    return {"id": id_, "phase": phase, "what": what, "runs": runs_, "epochs": epochs, "units_per_epoch": units,
            "unit": unit, "s_per_unit": s_per_unit, "basis": basis, "source": source, "seconds": seconds,
            "hours": seconds / 3600, "formula": formula}


def _laya_train_line(id_, phase, what, runs_, epochs, split, laya, counts, keys):
    n = counts[split]["n_items_neg8"]
    steps = spike.optimizer_steps(n, laya["micro_batch"], laya["grad_accum"])
    seconds = runs_ * laya_run_s(n, epochs, laya)
    formula = ("%s (%d) x [%s (%d) x (%d optimizer steps (%d %s neg8 items, micro-batch %d, grad-accum %d) x %.4f s "
               "+ %.1f s save) + %.1f s load + %.1f s calibration + %.1f s final save]"
               % (keys[0], runs_, keys[1], epochs, steps, n, split, laya["micro_batch"], laya["grad_accum"],
                  laya["s_per_step"], laya["save_s_per_epoch"], laya["load_s"], laya["calib_s"], laya["final_save_s"]))
    source = ("probe run on %s: s/step = the largest train_s / optimizer_steps over %d measured epoch(s); save, load, "
              "calibration and final-save seconds from the same record" % (laya["device"], laya["epochs_measured"]))
    return _line(id_, phase, what, runs_, epochs, steps, "optimizer step", laya["s_per_step"], "measured", source,
                 seconds, formula)


def project(measured, counts, matrix):
    """{"lines": [...], "total_s", "total_h"}: one line per experiment-matrix item, each with its runs, epochs, units,
    seconds per unit, basis (measured, estimate or provisional), source and formula; the total is their sum."""
    laya = measured["laya"]
    lines = [
        _laya_train_line("laya_selection", 5, "Laya selection runs (LR grid x val-selected epoch ceiling)",
                         matrix["lr_grid"], matrix["max_epochs"], "selection_train", laya, counts,
                         ("lr_grid", "max_epochs")),
        _laya_train_line("laya_final", 5, "Laya final seeds on final_train (projected at the epoch ceiling)",
                         matrix["final_seeds"], matrix["max_epochs"], "final_train", laya, counts,
                         ("final_seeds", "max_epochs")),
    ]
    total = sum(l["seconds"] for l in lines)
    return {"lines": lines, "total_s": total, "total_h": total / 3600}


def build_report(meta, measured, counts, matrix, projection, preflight):
    return {"schema": SCHEMA, "meta": meta, "preflight": preflight, "measured": measured, "counts": counts,
            "matrix": matrix, "projection": projection}


def _fmt(v, scale=1.0, digits=3):
    return "-" if v is None else ("%." + str(digits) + "f") % (v / scale)


def _measured_lines(measured):
    m = measured["laya"]
    lines = ["", "## Measured run", "",
             "- Device %s (parameters on %s), rung %s%s, neg %d, micro-batch %d, max_len %d, grad-accum %d, gradient "
             "checkpointing %s, AMP %s" % (m["device"], m["param_device"], m["rung"],
                                          " (head-only, weaker)" if m["weaker"] else "", m["neg"], m["micro_batch"],
                                          m["max_len"], m["grad_accum"], "on" if m["grad_checkpointing"] else "off",
                                          "on" if m["amp"] else "off"),
             "- Worker env: %s" % (", ".join("`%s`" % e for e in m["env"]) or "none (not an MPS run)"),
             "- Load %s s, calibration %s s, final save %s s" % (_fmt(m["load_s"], digits=1), _fmt(m["calib_s"], digits=1),
                                                                _fmt(m["final_save_s"], digits=1)),
             "- Projection rate: %s s per optimizer step, the largest train_s / optimizer_steps over %d measured "
             "epoch(s) (epoch 1: %s s; RESEARCH Pitfall 6)" % (_fmt(m["s_per_step"], digits=4), m["epochs_measured"],
                                                               _fmt(m["s_per_step_epoch1"], digits=4)),
             "",
             "| epoch | train min | s/step | optimizer steps | items | peak GiB (driver + RSS) | driver GiB | RSS GiB "
             "| swap delta MiB | save s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in measured["epochs"]:
        lines.append("| %d | %s | %s | %d | %d | %s | %s | %s | %s | %s |" % (
            r["epoch"], _fmt(r["train_s"], 60, 1), _fmt(r["train_s"] / r["optimizer_steps"], digits=4),
            r["optimizer_steps"], r["n_items"], _fmt(r["peak_total_bytes"], 2**30, 2),
            _fmt(r["peak_driver_bytes"], 2**30, 2), _fmt(r["peak_rss_bytes"], 2**30, 2),
            _fmt(r["swap_delta_bytes"], 2**20, 1), _fmt(r["save_s"], digits=1)))
    return lines


def _projection_lines(projection):
    lines = ["", "## Projection", "",
             "| line | phase | what | runs | epochs | units / epoch | unit | s / unit | basis | hours |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for l in projection["lines"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            l["id"], l["phase"], l["what"], l["runs"], "-" if l["epochs"] is None else l["epochs"],
            "-" if l["units_per_epoch"] is None else l["units_per_epoch"], l["unit"], _fmt(l["s_per_unit"], digits=4),
            l["basis"], _fmt(l["hours"], digits=2)))
    lines += ["", "**Total: %.2f h** (%.0f s, the sum of the lines above)" % (projection["total_h"],
                                                                           projection["total_s"]),
              "", "Formulas and sources:", ""]
    for l in projection["lines"]:
        lines.append("- `%s` (%s): %s = %.0f s. Source: %s" % (l["id"], l["basis"], l["formula"], l["seconds"],
                                                              l["source"]))
    return lines


def _provenance_lines(meta):
    split, man, model, versions, plat, proto = (meta["split"], meta["manifest"], meta["model"], meta["versions"],
                                                meta["platform"], meta["protocol"])
    model_text = ("`%s/%s@%s`" % (model["repo"], model["subfolder"], model["revision"]) if "repo" in model
                  else "local checkpoint, model sha256 `%s`" % model.get("model_sha256"))
    start = proto["spike_start"]
    return ["- Generated %s" % meta["generated_utc"],
            "- Split: `%s` sha256 `%s` (%d messages)" % (split["path"], split["sha256"], split["n_messages"]),
            "- Manifest: `%s` sha256 `%s` (committed: %s)" % (man["path"], man["sha256"], man["is_committed"]),
            "- Model: %s" % model_text,
            "- Versions: %s" % ", ".join("%s %s" % (k, versions[k]) for k in sorted(versions)),
            "- macOS %s, %s, %.1f GiB memory" % (plat["macos"] or "-", plat["machine"],
                                                 plat["total_memory_bytes"] / 2**30),
            "- Protocol: device %s, %d epoch(s), seed %d, %d CPUs, neg %d; spike start micro-batch %d, max_len %d, "
            "grad-accum %d, rung %s, PYTORCH_ENABLE_MPS_FALLBACK=%d; protocol complete: %s"
            % (proto["device"], proto["epochs"], proto["seed"], proto["cpus"], proto["neg"], start["micro_batch"],
               start["max_len"], start["grad_accum"], start["rung"], start["fallback"], proto["protocol_complete"])]


def render_markdown(report):
    """Human-readable compute_checkpoint.md: banners, provenance, the measured run, the projection with every line's
    basis and formula."""
    meta = report["meta"]
    lines = ["# Compute checkpoint (Phase 4, LAYA-07)", ""]
    if not meta["protocol"]["protocol_complete"]:
        lines += ["**SMOKE RUN: not a compute checkpoint.** The run left the protocol (not mps, not the pinned weights, "
                  "not the committed selection_train, a start config other than the spike's, or a test hook); its "
                  "numbers must not be used for the approval.", ""]
    if spike.busy_machine(report.get("preflight")):
        lines += ["**Busy-machine run:** the preflight was overridden or swap grew while idle; the measured rate may be "
                  "slower than on a quiet machine.", ""]
    lines += _provenance_lines(meta)
    pf = report.get("preflight")
    if pf:
        lines.append("- Preflight: %.2f GiB available (threshold %s%s), swap grew while idle: %s"
                     % (pf["available_gib"], pf["min_available_gib"], ", overridden" if pf["override"] else "",
                        pf["swap_grew_while_idle"]))
    lines += _measured_lines(report["measured"])
    lines += _projection_lines(report["projection"])
    return "\n".join(lines) + "\n"


def write_report(out_dir, report):
    """Write compute_checkpoint.json, then render compute_checkpoint.md from the re-parsed JSON (so the committed
    markdown always equals render_markdown(json.load(the committed JSON))); returns the markdown path."""
    os.makedirs(out_dir, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    write_atomic(os.path.join(out_dir, JSON_NAME), text.encode("utf-8"))
    md = os.path.join(out_dir, MD_NAME)
    write_atomic(md, render_markdown(json.loads(text)).encode("utf-8"))
    return md
