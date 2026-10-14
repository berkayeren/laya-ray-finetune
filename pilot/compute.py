"""Torch-free logic of the Phase 4 compute checkpoint (LAYA-07): the measured run record -> a projected runtime for the
whole experiment matrix -> compute_checkpoint.json + compute_checkpoint.md.

compute_checkpoint.py imports this module in the orchestrator, before its ray_train.py child sets the MPS env (torch
reads PYTORCH_ENABLE_MPS_FALLBACK once, at import), so nothing here may import torch. The measurement itself happens in
the child; this module only reads its checkpoint config (training.ray_train.record, pilot.runs.read_record), counts the
items of the committed splits without a tokenizer (pilot.runs.neg_items) and projects.

Projection rate (RESEARCH Pitfall 6): s_per_step is the LARGEST train_s / optimizer_steps over the measured epochs, so
the MPS warm-up of epoch 1 (or any slower later epoch) makes the projection conservative, never optimistic. A Laya run
of E epochs on n items costs E x (optimizer_steps(n) x s_per_step + save_s_per_epoch) + load_s + calib_s +
final_save_s + run_overhead_s, every term measured by the probe run; run_overhead_s (run_overhead) is the probe child's
wall clock minus the record's own terms (process and Ray start, the driver's checks, item building, the --out copy),
0 for a report from before it was measured, which assumptions() then discloses.

Transparency (T-04-17): every projection line names its basis and its source. basis is the weakest input of the line:
"estimate" when any of its rates is an estimate (ESTIMATES, used only when the BERTurk timing child did not finish
ok), "provisional" for the
few-shot lines (their protocol is frozen only in Phase 8), "measured" otherwise. Every MATRIX value is provisional
whatever the basis; assumptions() lists each one, with its source, for the user to confirm or trim at plan 04-08.

usage: from pilot import compute; start = compute.spike_start("results/spike/mps_spike.json")
       measured = compute.measured_from_checkpoint(ckpt_dir); counts = compute.split_counts(manifest)
       projection = compute.project(measured, counts, compute.MATRIX)
       report = compute.build_report(meta, measured, counts, compute.MATRIX, projection, pf,
                                     what_if=compute.what_if(measured, counts, compute.MATRIX),
                                     disk=compute.disk_budget(model_bytes, compute.MATRIX),
                                     assumptions=compute.assumptions(compute.MATRIX, measured))
       compute.write_report(out_dir, report)
"""
import json
import math
import os

from pilot import runs, spike
from pilot.data import REPO_ROOT, SPLITS, parse_jsonl, repo_relpath, verify_split_file, write_atomic

SCHEMA = "compute_checkpoint/1"
JSON_NAME, MD_NAME = "compute_checkpoint.json", "compute_checkpoint.md"
# the probe's timing basis: --cpus (worker CPUs and torch threads, the spike's 4), a 2-epoch probe, seed 0; a run with
# other values is off the protocol (compute_checkpoint.protocol_complete), since s/step depends on the thread count
PROTOCOL = {"cpus": 4, "epochs": 2, "seed": 0}

# The experiment matrix the projection covers. Every value is provisional (RESEARCH A2) until Phases 5, 6 and 8 freeze
# it; the user confirms or trims them at the compute checkpoint (plan 04-08).
MATRIX = {"lr_grid": 3, "max_epochs": 4, "final_seeds": 3, "neg": 8, "effective_batch": 32, "berturk_epochs": 10,
          "berturk_batch": 32, "fewshot_tags": 6, "fewshot_k": [5, 10], "fewshot_draws": 3, "fewshot_neg_per_pos": 4,
          "fewshot_replay": 200, "fewshot_epochs": 10, "fewshot_base_runs": 3, "eval_seeds": 3}
MATRIX_SOURCES = {
    "lr_grid": "FEATURES TS-12 example shared LR grid {2e-5, 3e-5, 5e-5}, used by Laya (Phase 5) and BERTurk (Phase 6)",
    "max_epochs": "ray_train.py default --epochs 4, the ceiling of the val-selected epoch (final seeds at it)",
    "final_seeds": "LAYA-08 / BASE-01: three final seeds per model on final_train",
    "neg": "the spike's data_neg8 item mix (ray_train.py --neg 8): every positive plus 8 negatives per message",
    "effective_batch": "the spike's micro-batch 8 x grad-accum 4; the projection uses the measured run's own values",
    "berturk_epochs": "STACK: BERTurk 5-10 epochs; the upper bound is projected",
    "berturk_batch": "BERTurk batch 32 messages (one sequence per message, all 48 labels)",
    "fewshot_tags": "FEATURES TS-14: 6 held-out tags (frozen only in the Phase 8 protocol.json)",
    "fewshot_k": "FEATURES TS-14 / ROADMAP Phase 8: k = 5 and 10 support messages per new tag",
    "fewshot_draws": "ROADMAP Phase 8: at most 3 support-set draws",
    "fewshot_neg_per_pos": "FEATURES TS-14 default: 4 negative support messages per positive",
    "fewshot_replay": "FEATURES TS-14 default: 200 replay messages of the old tags",
    "fewshot_epochs": "FEATURES TS-14 default: about 10 continued-training epochs",
    "fewshot_base_runs": "base runs per model with the held-out labels masked (one per final seed); the pre-declared "
                         "compute fallback is 1",
    "eval_seeds": "ROADMAP Phase 8: 3 seeds x 2 models scored once on val, test and HOTELS",
}
# labelled estimates, used only when compute_checkpoint.py's BERTurk timing child (measured["berturk"]) did not
# finish ok
ESTIMATES = {"berturk_s_per_step": 1.3, "berturk_s_per_msg": 0.01}
ESTIMATE_SOURCES = {
    "berturk_s_per_step": "about 30-60 s per BERTurk epoch of 46 batches of 32 on MPS, upper end [ASSUMED, RESEARCH "
                          "Open Question 2]",
    "berturk_s_per_msg": "a 110M-parameter encoder scoring one message at max_len 128 on MPS [ASSUMED]",
}
# the committed CPU zero-shot scoring timing, the Laya scoring rate when no scoring probe (measured["scoring"]) ran ok
SCORING_DUMP = os.path.join(REPO_ROOT, "results", "dumps", "laya-zeroshot-7b928d82", "val.json")
SPLIT_NAMES = tuple(s["name"] for s in SPLITS)
SCORED = ("val", "test", "hotels_probe")


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


def split_counts(manifest_path, names=SPLIT_NAMES):
    """{split: {n_messages, n_items_neg8, n_items_full}} for the registered splits `names` (default all five), each
    verified on the exact bytes counted (pilot.data.verify_split_file) and counted torch-free (pilot.runs.neg_items)."""
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)["splits"]
    out = {}
    for name in names:
        path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), entries[name]["path"])
        with open(path, "rb") as f:
            raw = f.read()
        verify_split_file(path, manifest_path=manifest_path, data=raw)
        rows = parse_jsonl(raw)
        out[name] = {"n_messages": len(rows), "n_items_neg8": runs.neg_items(rows, 8),
                     "n_items_full": runs.neg_items(rows, 0)}
    return out


def _largest(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


# ---- the probe protocol: interrupt after epoch 1, resume, and the LAYA-03 ladder in fresh children ----

# Ray Train V2's internal checkpoint-manager snapshot (no public API; written before the previous checkpoint is
# deleted, in place). Its format was read from ray SNAPSHOT_RAY: {"ray_version", "checkpoint_results":
# [{"checkpoint_dir_name", "metrics", ...}], "latest_checkpoint_result", ...}; tests/test_compute_unit.py fails when the
# lock moves ray, so a bump re-validates it (04-REVIEW IN-07)
SNAPSHOT = "checkpoint_manager_snapshot.json"
SNAPSHOT_RAY = "2.59.0"


def epoch_registered(storage, run_id, epoch):
    """True once Ray has registered the epoch checkpoint of `run_id`: an entry of the snapshot's checkpoint_results has
    checkpoint_dir_name epoch-NNN and that dir holds train_state.pt, so a kill from here on resumes from it
    (ray_train.py's epoch checkpoints). A missing or half-written snapshot reads as not registered."""
    run_dir, name = os.path.join(storage, run_id), "epoch-%03d" % epoch
    try:
        with open(os.path.join(run_dir, SNAPSHOT), encoding="utf-8") as f:
            snap = json.load(f)
    except (OSError, ValueError):
        return False
    results = snap.get("checkpoint_results") if isinstance(snap, dict) else None
    names = {r.get("checkpoint_dir_name") for r in results or [] if isinstance(r, dict)}
    return name in names and os.path.isfile(os.path.join(run_dir, name, "train_state.pt"))


def attempt_chain(start_rung, launch):
    """(attempts, status): launch(rung) runs one fresh child on `rung` and returns its attempt ({status: ok, oom or
    failed, ...}); an oom moves to pilot.runs.next_rung (LAYA-03), ok and failed stop the chain, and an oom on the
    last rung ends it as ladder_exhausted. Each attempt carries the rung it was launched on."""
    attempts, rung = [], start_rung
    while True:
        attempt = dict(launch(rung), rung=rung)
        attempts.append(attempt)
        if attempt["status"] == "ok":
            return attempts, "ok"
        if attempt["status"] != "oom":
            return attempts, "failed"
        rung = runs.next_rung(rung)
        if rung is None:
            return attempts, "ladder_exhausted"


def resume_summary(record):
    """{segments, resume_load_s, resume_peak_total_bytes, resume_demonstrated} of a run record (LAYA-04).

    resume_demonstrated: two or more segments, epoch rows numbered 1..E without a gap, and resumed flags that start
    False and turn True (the rows a resumed segment trained). The load seconds and peak memory are the largest over
    the resumed segments (None when no segment resumed)."""
    segments, rows = record.get("segments") or [], record.get("epochs") or []
    resumed = [s for s in segments if s.get("resumed")]
    flags = [bool(r.get("resumed")) for r in rows]
    numbered = [r.get("epoch") for r in rows] == list(range(1, len(rows) + 1))
    turns = bool(flags) and not flags[0] and flags[-1] and flags == sorted(flags)
    seg_flags = [bool(s.get("resumed")) for s in segments]
    return {"segments": len(segments),
            "resume_load_s": _largest(resumed, "resume_load_s"),
            "resume_peak_total_bytes": _largest(resumed, "resume_peak_total_bytes"),
            "resume_demonstrated": bool(len(segments) >= 2 and numbered and turns and seg_flags[0] is False
                                        and all(seg_flags[1:]))}


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
            "peak_driver_bytes": _largest(rows, "peak_driver_bytes"),
            "peak_rss_bytes": _largest(rows, "peak_rss_bytes"),
            "swap_delta_bytes": _largest(rows, "swap_delta_bytes"), "micro_batch": train["micro_batch"],
            "grad_accum": train["grad_accum"], "max_len": cfg["max_len"], "neg": train["neg"],
            "rung": record["rung"], "fallback": record["fallback"], "weaker": record["weaker"],
            "device": record["device"], "param_device": record["param_device"],
            "grad_checkpointing": record["grad_checkpointing"], "amp": record["amp"], "env": list(record["env"]),
            "train_state_bytes": record.get("train_state_bytes")}
    return {"laya": laya, "epochs": rows}


def run_overhead(wall_s, record):
    """Seconds of one ray_train.py process outside its run record's terms: `wall_s` (the child's wall clock) minus the
    train and save seconds of the epochs its last segment trained, its load, resume-load, calibration and final-save
    seconds; what is left (process and Ray start, the driver's split and weight checks, item building, the --out copy)
    is paid once by every run (04-REVIEW IN-03). Never negative."""
    seg = record["segments"][-1]
    inside = sum(r["train_s"] + r["save_s"] for r in record["epochs"] if r["segment"] == seg["segment"])
    inside += record["load_s"] + (seg.get("resume_load_s") or 0.0) + record["calib_s"] + record["final_save_s"]
    return max(0.0, wall_s - inside)


def laya_run_s(n_items, epochs, laya):
    """Seconds of one Laya run of `epochs` epochs on `n_items` items at the measured rate and overheads."""
    steps = spike.optimizer_steps(n_items, laya["micro_batch"], laya["grad_accum"])
    return epochs * (steps * laya["s_per_step"] + laya["save_s_per_epoch"]) + laya["load_s"] + laya["calib_s"] + \
        laya["final_save_s"] + (laya.get("run_overhead_s") or 0.0)


# ---- rates: measured when the probe measured them, else the committed CPU scoring rate or a labelled estimate ----

def laya_scoring_rate(measured):
    """(s per message, basis, source) of Laya scoring: measured["scoring"] when present (inference seconds / messages),
    else the committed CPU zero-shot val timing, basis "measured (cpu, wall)": wall seconds / messages, Ray start and
    actor loads included (04-REVIEW IN-03)."""
    s = measured.get("scoring")
    if s and s.get("status", "ok") == "ok":
        return s["s_per_msg"], "measured", ("scoring probe on %s: %s, %d messages"
                                            % (s.get("device"), s.get("basis") or "measured", s.get("n") or 0))
    with open(SCORING_DUMP, encoding="utf-8") as f:
        side = json.load(f)
    n, sec, sc = side["split"]["n_messages"], side["timing"]["seconds"], side["scorer"]
    return sec / n, "measured (cpu, wall)", ("committed zero-shot val dump %s: %.2f s wall for %d messages on %s with %d Ray "
                                 "actors x %d torch threads; not yet measured on the fine-tuned scoring device"
                                 % (repo_relpath(SCORING_DUMP), sec, n, sc["device"], sc["workers"],
                                    sc["torch_threads"]))


def berturk_rates(measured):
    """(s per optimizer step, s per scored message, basis, source) of BERTurk: measured["berturk"] when its probe
    finished ok, else ESTIMATES."""
    b = measured.get("berturk")
    if b and b.get("status") == "ok":
        return b["s_per_step"], b["s_per_msg_infer"], "measured", ("BERTurk timing probe on %s: %s"
                                                                   % (b.get("device"), b.get("basis") or "measured"))
    return (ESTIMATES["berturk_s_per_step"], ESTIMATES["berturk_s_per_msg"], "estimate",
            "ESTIMATES: %s; %s" % (ESTIMATE_SOURCES["berturk_s_per_step"], ESTIMATE_SOURCES["berturk_s_per_msg"]))


def _laya_source(laya):
    over = laya.get("run_overhead_s")
    return ("probe run on %s: s/step = the largest train_s / optimizer_steps over %d measured epoch(s); save, load, "
            "calibration and final-save seconds from the same record%s"
            % (laya["device"], laya["epochs_measured"], "" if over is None else
               "; run overhead %.1f s = the probe child's wall clock minus the record's terms" % over))


# ---- lines and per-model parts ----

def _line(id_, phase, what, runs_, epochs, units, unit, s_per_unit, basis, source, seconds, formula):
    return {"id": id_, "phase": phase, "what": what, "runs": runs_, "epochs": epochs, "units_per_epoch": units,
            "unit": unit, "s_per_unit": s_per_unit, "basis": basis, "source": source, "seconds": seconds,
            "hours": seconds / 3600, "formula": formula}


def _laya_part(runs_, epochs, n_items, laya, **extra):
    steps = spike.optimizer_steps(n_items, laya["micro_batch"], laya["grad_accum"])
    return dict({"model": "laya", "runs": runs_, "epochs": epochs, "units_per_epoch": steps,
                 "unit": "optimizer step", "s_per_unit": laya["s_per_step"], "basis": "measured",
                 "seconds": runs_ * laya_run_s(n_items, epochs, laya), "items": n_items}, **extra)


def _berturk_part(runs_, epochs, n_messages, batch, s_step, basis, **extra):
    steps = math.ceil(n_messages / batch)
    return dict({"model": "berturk", "runs": runs_, "epochs": epochs, "units_per_epoch": steps,
                 "unit": "optimizer step", "s_per_unit": s_step, "basis": basis,
                 "seconds": runs_ * epochs * steps * s_step}, **extra)


def _scoring_part(model, runs_, n_messages, s_msg, basis):
    return {"model": model, "runs": runs_, "epochs": None, "units_per_epoch": n_messages, "unit": "message",
            "s_per_unit": s_msg, "basis": basis, "seconds": runs_ * n_messages * s_msg}


def _same(parts, key):
    vals = {json.dumps(p[key]) for p in parts}
    return parts[0][key] if len(vals) == 1 else None


def _parts_line(id_, phase, what, parts, source, formula, provisional=False):
    """A line made of per-model parts: runs and seconds are their sums; epochs, units, unit and s/unit are shown only
    when every part agrees; basis is the weakest part (estimate), else provisional for an unfrozen protocol, else a
    qualified measurement (measured (cpu, wall)) of any part, else measured."""
    basis = "estimate" if any(p["basis"] == "estimate" for p in parts) else "provisional" if provisional else \
        next((p["basis"] for p in parts if p["basis"] != "measured"), "measured")
    line = _line(id_, phase, what, sum(p["runs"] for p in parts), _same(parts, "epochs"),
                 _same(parts, "units_per_epoch"), _same(parts, "unit"), _same(parts, "s_per_unit"), basis, source,
                 sum(p["seconds"] for p in parts), formula)
    line["parts"] = parts
    return line


def _items_key(matrix):
    if matrix["neg"] == 8:
        return "n_items_neg8"
    if matrix["neg"] == 0:
        return "n_items_full"
    raise ValueError("the projection counts neg8 or full items only, got neg %r" % (matrix["neg"],))


def _laya_train_line(id_, phase, what, runs_, epochs, split, laya, counts, keys, items_key):
    n = counts[split][items_key]
    part = _laya_part(runs_, epochs, n, laya)
    over = laya.get("run_overhead_s")
    formula = ("%s (%d) x [%s (%d) x (%d optimizer steps (%d %s %s items, micro-batch %d, grad-accum %d) x %.4f s "
               "+ %.1f s save) + %.1f s load + %.1f s calibration + %.1f s final save%s]"
               % (keys[0], runs_, keys[1], epochs, part["units_per_epoch"], n, split, items_key[8:],
                  laya["micro_batch"], laya["grad_accum"], laya["s_per_step"], laya["save_s_per_epoch"],
                  laya["load_s"], laya["calib_s"], laya["final_save_s"],
                  "" if over is None else " + %.1f s run overhead" % over))
    return _line(id_, phase, what, runs_, epochs, part["units_per_epoch"], "optimizer step", laya["s_per_step"],
                 "measured", _laya_source(laya), part["seconds"], formula)


def _berturk_train_line(id_, phase, what, runs_, epochs, split, batch, rates, counts, keys):
    s_step, _, basis, source = rates
    n = counts[split]["n_messages"]
    part = _berturk_part(runs_, epochs, n, batch, s_step, basis)
    formula = ("%s (%d) x berturk_epochs (%d) x %d optimizer steps (ceil(%d %s messages / batch %d)) x %.4f s"
               % (keys[0], runs_, epochs, part["units_per_epoch"], n, split, batch, s_step))
    return _line(id_, phase, what, runs_, epochs, part["units_per_epoch"], "optimizer step", s_step, basis, source,
                 part["seconds"], formula)


def project(measured, counts, matrix):
    """{"lines": [...], "total_s", "total_h"}: one line per experiment-matrix item, each with its runs, epochs, units,
    seconds per unit, basis (measured, estimate or provisional), source and formula; the total is their sum."""
    laya, ik, m = measured["laya"], _items_key(matrix), matrix
    s_msg, s_msg_basis, s_msg_source = laya_scoring_rate(measured)
    bert = berturk_rates(measured)
    b_step, b_msg, b_basis, b_source = bert
    sel, fin = counts["selection_train"], counts["final_train"]
    n_val, n_test, n_ood = (counts[s]["n_messages"] for s in SCORED)
    lines = [
        _laya_train_line("laya_selection", 5, "Laya selection runs (LR grid x val-selected epoch ceiling)",
                         m["lr_grid"], m["max_epochs"], "selection_train", laya, counts, ("lr_grid", "max_epochs"),
                         ik),
        _line("laya_val_scoring", 5, "Laya val scoring of every selection epoch export", m["lr_grid"] * m["max_epochs"],
              None, n_val, "message", s_msg, s_msg_basis, s_msg_source,
              m["lr_grid"] * m["max_epochs"] * n_val * s_msg,
              "lr_grid (%d) x max_epochs (%d) epoch exports x %d val messages x %.4f s"
              % (m["lr_grid"], m["max_epochs"], n_val, s_msg)),
        _laya_train_line("laya_final", 5, "Laya final seeds on final_train (projected at the epoch ceiling)",
                         m["final_seeds"], m["max_epochs"], "final_train", laya, counts, ("final_seeds", "max_epochs"),
                         ik),
        _berturk_train_line("berturk_selection", 6, "BERTurk selection runs (LR grid)", m["lr_grid"],
                            m["berturk_epochs"], "selection_train", m["berturk_batch"], bert, counts, ("lr_grid",)),
        _berturk_train_line("berturk_final", 6, "BERTurk final seeds on final_train", m["final_seeds"],
                            m["berturk_epochs"], "final_train", m["berturk_batch"], bert, counts, ("final_seeds",)),
    ]

    # Phase 9 few-shot (protocol frozen only in Phase 8): base runs with the held-out labels masked, then incremental
    # runs on k-shot support sets plus replay, then scoring of every incremental model on test
    base = [_laya_part(m["fewshot_base_runs"], m["max_epochs"], sel[ik], laya),
            _berturk_part(m["fewshot_base_runs"], m["berturk_epochs"], sel["n_messages"], m["berturk_batch"], b_step,
                          b_basis)]
    lines.append(_parts_line(
        "fewshot_base", 9, "Few-shot base runs, held-out labels masked (both models)", base,
        "Laya: %s; BERTurk: %s" % (_laya_source(laya), b_source),
        "per model fewshot_base_runs (%d) on selection_train: Laya at max_epochs (%d), %d optimizer steps per epoch; "
        "BERTurk at berturk_epochs (%d), %d optimizer steps per epoch"
        % (m["fewshot_base_runs"], m["max_epochs"], base[0]["units_per_epoch"], m["berturk_epochs"],
           base[1]["units_per_epoch"]), provisional=True))
    per_msg = sel[ik] / sel["n_messages"]
    inc, msgs = [], {}
    for model in ("laya", "berturk"):
        for k in m["fewshot_k"]:
            n_msgs = m["fewshot_tags"] * k * (1 + m["fewshot_neg_per_pos"]) + m["fewshot_replay"]
            msgs[k] = n_msgs
            if model == "laya":
                inc.append(_laya_part(m["fewshot_draws"], m["fewshot_epochs"], math.ceil(n_msgs * per_msg), laya,
                                      k=k, messages=n_msgs))
            else:
                inc.append(_berturk_part(m["fewshot_draws"], m["fewshot_epochs"], n_msgs, m["berturk_batch"], b_step,
                                         b_basis, k=k, messages=n_msgs))
    lines.append(_parts_line(
        "fewshot_incremental", 9, "Few-shot incremental runs on k-shot support sets plus replay (both models)", inc,
        "Laya: %s; BERTurk: %s" % (_laya_source(laya), b_source),
        "per model and k: fewshot_draws (%d) runs x fewshot_epochs (%d) on fewshot_tags (%d) x k x (1 + "
        "fewshot_neg_per_pos %d) + fewshot_replay (%d) messages (%s); Laya items = ceil(messages x %.3f "
        "selection_train %s items per message)" % (m["fewshot_draws"], m["fewshot_epochs"], m["fewshot_tags"],
                                   m["fewshot_neg_per_pos"], m["fewshot_replay"],
                                   ", ".join("k %d: %d" % (k, msgs[k]) for k in m["fewshot_k"]), per_msg, ik[8:]),
        provisional=True))
    n_inc = len(m["fewshot_k"]) * m["fewshot_draws"]
    fs = [_scoring_part("laya", n_inc, n_test, s_msg, s_msg_basis), _scoring_part("berturk", n_inc, n_test, b_msg,
                                                                                   b_basis)]
    lines.append(_parts_line(
        "fewshot_scoring", 9, "Few-shot scoring of every incremental model on test (both models)", fs,
        "Laya: %s; BERTurk: %s" % (s_msg_source, b_source),
        "per model len(fewshot_k) (%d) x fewshot_draws (%d) models x %d test messages x s/msg (Laya %.4f, BERTurk "
        "%.4f)" % (len(m["fewshot_k"]), m["fewshot_draws"], n_test, s_msg, b_msg), provisional=True))
    n_eval = n_val + n_test + n_ood
    final = [_scoring_part("laya", m["eval_seeds"], n_eval, s_msg, s_msg_basis),
             _scoring_part("berturk", m["eval_seeds"], n_eval, b_msg, b_basis)]
    lines.append(_parts_line(
        "final_scoring", 8, "Final scoring: every seed of both models on val, test and HOTELS", final,
        "Laya: %s; BERTurk: %s" % (s_msg_source, b_source),
        "per model eval_seeds (%d) x (%d val + %d test + %d hotels_probe = %d messages) x s/msg (Laya %.4f, BERTurk "
        "%.4f)" % (m["eval_seeds"], n_val, n_test, n_ood, n_eval, s_msg, b_msg)))
    total = sum(l["seconds"] for l in lines)
    return {"lines": lines, "total_s": total, "total_h": total / 3600}


def what_if(measured, counts, matrix):
    """Total runtime for lr_grid 1-3 x max_epochs 2-4 (the rest of `matrix` unchanged), plus one row for the few-shot
    compute fallback (1 base run per model) at the matrix's own lr_grid and max_epochs."""
    rows = []
    for g in (1, 2, 3):
        for e in (2, 3, 4):
            t = project(measured, counts, dict(matrix, lr_grid=g, max_epochs=e))["total_s"]
            rows.append({"label": "lr_grid %d x max_epochs %d" % (g, e), "lr_grid": g, "max_epochs": e,
                         "fewshot_base_runs": matrix["fewshot_base_runs"], "total_s": t, "total_h": t / 3600})
    t = project(measured, counts, dict(matrix, fewshot_base_runs=1))["total_s"]
    rows.append({"label": "few-shot compute fallback (1 base run per model)", "lr_grid": matrix["lr_grid"],
                 "max_epochs": matrix["max_epochs"], "fewshot_base_runs": 1, "total_s": t, "total_h": t / 3600})
    return rows


def disk_budget(model_bytes, matrix, train_state_bytes=None):
    """Disk the matrix needs on top of the code (bytes): the train state (6 x the fp16 weight bytes unless measured),
    one run's peak (three train states during checkpoint rotation, its per-epoch exports and its final checkpoint),
    the selection runs' epoch exports kept until Phase 5 scores them on val, and the kept final checkpoints."""
    ts = 6 * model_bytes if train_state_bytes is None else train_state_bytes
    return {"model_bytes": model_bytes, "train_state_bytes": ts,
            "train_state_basis": "6 x the fp16 weight bytes" if train_state_bytes is None else "measured",
            "per_run_peak_bytes": 3 * ts + matrix["max_epochs"] * model_bytes + model_bytes,
            "run_floor_bytes": runs.disk_need_bytes(model_bytes, matrix["max_epochs"], True),
            "selection_exports_bytes": matrix["lr_grid"] * matrix["max_epochs"] * model_bytes,
            "laya_final_checkpoints_bytes": (matrix["lr_grid"] + matrix["final_seeds"]) * model_bytes}


def swap_grew_epochs(measured):
    """The measured epochs whose system swap grew (swap_delta_bytes > 0): the spike's D-01 memory gate counts any swap
    growth during a measured step as memory pressure, so the report flags them (04-REVIEW WR-08)."""
    return [r["epoch"] for r in (measured or {}).get("epochs") or [] if (r.get("swap_delta_bytes") or 0) > 0]


def assumptions(matrix, measured=None, counts=None, interrupt=None):
    """What the user must confirm or trim at the compute checkpoint (plan 04-08), one sentence each."""
    measured = measured or {}
    interrupted = bool(interrupt) and interrupt.get("mode") != "none" and not interrupt.get("missed")
    out = ["Matrix value %s = %s is provisional (RESEARCH A2): %s"
           % (k, ", ".join(str(x) for x in v) if isinstance(v, list) else v, MATRIX_SOURCES[k])
           for k, v in matrix.items()]
    out += ["Negatives are drawn per seed (RESEARCH A4): ray_train.py --neg-seed defaults to --seed, so the seed "
            "variance includes the neg8 sampling; pin --neg-seed 0 to exclude it",
            "The git sha is recorded per run segment and kept out of the run_id (RESEARCH A3): a resume across a code "
            "change is flagged code_changed_on_resume, not refused"]
    b = measured.get("berturk")
    if not (b and b.get("status") == "ok"):
        out.append("BERTurk rates are estimates (the BERTurk timing child did not finish ok or did not run): %.2f s "
                   "per optimizer step (%s) and %.3f s per scored message (%s)"
                   % (ESTIMATES["berturk_s_per_step"], ESTIMATE_SOURCES["berturk_s_per_step"],
                      ESTIMATES["berturk_s_per_msg"], ESTIMATE_SOURCES["berturk_s_per_msg"]))
    else:
        out.append("BERTurk is timed with a plain PyTorch loop (fp32, AdamW, batch 32, max_len 128), not the HF "
                   "Trainer Phase 6 trains with; the Trainer's own overhead (data collation, logging, per-epoch "
                   "evaluation hooks) is assumed small")
    sc = measured.get("scoring")
    if not (sc and sc.get("status", "ok") == "ok"):
        out.append("The Laya scoring rate is the committed CPU zero-shot rate (%s) until an MPS scoring probe "
                   "measures it" % repo_relpath(SCORING_DUMP))
    else:
        out.append("Scoring time is measured on the pinned zero-shot weights (a --n %s val subset); a fine-tuned "
                   "checkpoint has the same architecture and forward cost" % sc.get("n"))
    if measured.get("laya") and measured["laya"].get("run_overhead_s") is None:
        out.append("The Laya lines leave out each run's fixed overhead (process and Ray start, the driver's split and "
                   "weight checks, item building, the --out copy): this probe predates its measurement (the probe "
                   "child's wall clock minus the record's terms), so they are slightly optimistic, by that overhead "
                   "once per Laya training run")
    grew = swap_grew_epochs(measured)
    if grew:
        out.append("Swap grew during measured epoch(s) %s (largest %.0f MiB): the run was under memory pressure (the "
                   "spike's D-01 gate), so the measured rate may be slower than on a quiet machine; confirm it before "
                   "approval" % (", ".join(map(str, grew)),
                                 max(r["swap_delta_bytes"] for r in measured["epochs"]) / 2**20))
    s_msg = berturk_rates(measured)[1]                     # the BERTurk scoring rate the projection uses
    out += ["Scoring runs in fp32 on MPS (la.force_fp32 turns laya's fp16 autocast off); the scoring rate assumes fp32",
            "Per-epoch laya exports stay on (--export-epochs) so Phase 5 picks the epoch on val without retraining; "
            "Phase 5 prunes them after val scoring (see Disk)",
            "The measurement probe trains %s epoch(s)%s; the projection uses the slowest measured epoch"
            % ((measured.get("laya") or {}).get("epochs_measured", "-"),
               ", interrupted after epoch 1 and resumed by rerunning the same command (see Interruption and resume), "
               "so the resumed process's warm-up is in the rate" if interrupted else ", not interrupted"),
            "Every split is counted over all its messages, without the 40-message calibration holdout, because Phase 5 "
            "calibrates on val (LAYA-05); slightly conservative",
            "BERTurk's per-epoch val evaluation for its epoch choice is not a separate line (lr_grid x berturk_epochs "
            "val passes at %.3f s per message%s)"
            % (s_msg, "" if not counts else ", about %.0f s" % (matrix["lr_grid"] * matrix["berturk_epochs"]
                                                                 * counts["val"]["n_messages"] * s_msg))]
    return out


def build_report(meta, measured, counts, matrix, projection, preflight, what_if=None, disk=None, assumptions=None,
                 attempts=None, status="ok", interrupt=None, resume=None):
    """The report dict. status other than ok (failed, ladder_exhausted) comes with measured and projection None: the
    attempts are the whole result."""
    return {"schema": SCHEMA, "meta": meta, "status": status, "preflight": preflight, "attempts": attempts,
            "interrupt": interrupt, "resume": resume, "measured": measured, "counts": counts, "matrix": matrix,
            "projection": projection, "what_if": what_if, "disk": disk, "assumptions": assumptions}


# ---- markdown ----

def _fmt(v, scale=1.0, digits=3):
    return "-" if v is None else ("%." + str(digits) + "f") % (v / scale)


def _dash(v):
    return "-" if v is None else v


def _measured_lines(measured):
    m = measured["laya"]
    lines = ["", "## Measured run", "",
             "- Device %s (parameters on %s), rung %s%s, neg %d, micro-batch %d, max_len %d, grad-accum %d, gradient "
             "checkpointing %s, AMP %s" % (m["device"], m["param_device"], m["rung"],
                                          " (head-only, weaker)" if m["weaker"] else "", m["neg"], m["micro_batch"],
                                          m["max_len"], m["grad_accum"], "on" if m["grad_checkpointing"] else "off",
                                          "on" if m["amp"] else "off"),
             "- Worker env: %s" % (", ".join("`%s`" % e for e in m["env"]) or "none (not an MPS run)"),
             "- Load %s s, calibration %s s, final save %s s%s" % (
                 _fmt(m["load_s"], digits=1), _fmt(m["calib_s"], digits=1), _fmt(m["final_save_s"], digits=1),
                 "" if m.get("run_overhead_s") is None else ", run overhead %s s (the child's wall clock minus these "
                 "and the epochs' terms)" % _fmt(m["run_overhead_s"], digits=1)),
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


def _probe_lines(measured):
    """The BERTurk timing probe and the scoring probe, when they ran: their numbers, or why a line keeps its
    estimate."""
    lines, g = [], 2**30
    b = measured.get("berturk")
    if b is not None:
        lines += ["", "## BERTurk timing probe", ""]
        if b.get("status") == "ok":
            lines += ["- %s" % b["basis"],
                      "- Device %s (parameters on %s), %s, %d torch threads, %s parameters" % (
                          b["device"], b["param_device"], b.get("dtype", "-"), b.get("torch_threads", 0),
                          b.get("n_params", "-")),
                      "- %s s per optimizer step (median of %d measured steps after %s warm-up; p90 %s s); %s s per "
                      "message for inference (%s messages, one batch)" % (
                          _fmt(b["s_per_step"], digits=4), b["n_steps"], _dash(b.get("warmup_steps")),
                          _fmt(b.get("p90_s"), digits=4), _fmt(b["s_per_msg_infer"], digits=4),
                          _dash(b.get("infer_msgs"))),
                      "- Peak memory %s GiB (driver %s GiB, RSS %s GiB)" % (
                          _fmt(b.get("peak_total_bytes"), g, 2), _fmt(b.get("peak_driver_bytes"), g, 2),
                          _fmt(b.get("peak_rss_bytes"), g, 2))]
        else:
            lines.append("- **Status %s** (%s): the BERTurk lines keep the estimate (%.2f s per step, %.3f s per "
                         "message)" % (b.get("status"), b.get("reason") or "returncode %s" % b.get("returncode"),
                                       ESTIMATES["berturk_s_per_step"], ESTIMATES["berturk_s_per_msg"]))
    sc = measured.get("scoring")
    if sc is not None:
        lines += ["", "## Scoring probe", ""]
        if sc.get("status", "ok") == "ok":
            lines += ["- %s" % sc["basis"],
                      "- Device %s, %s, autocast %s: %s s per message over %d messages (inference time, the sidecar's "
                      "inference_seconds / n)" % (sc["device"], sc["dtype"], "on" if sc["amp_enabled"] else "off",
                                                  _fmt(sc["s_per_msg"], digits=4), sc["n"])]
        else:
            lines.append("- **Status %s** (%s): the Laya scoring lines keep the committed CPU rate"
                         % (sc.get("status"), sc.get("reason")))
    return lines


def _projection_lines(projection):
    lines = ["", "## Projection", "",
             "Basis is the weakest input of a line: estimate (a labelled estimate, not measured), provisional "
             "(few-shot, protocol frozen only in Phase 8) or measured. Every matrix value is provisional; see "
             "Assumptions to confirm.", "",
             "| line | phase | what | runs | epochs | units / epoch | unit | s / unit | basis | hours |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for l in projection["lines"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            l["id"], l["phase"], l["what"], l["runs"], _dash(l["epochs"]), _dash(l["units_per_epoch"]),
            _dash(l["unit"]), _fmt(l["s_per_unit"], digits=4), l["basis"], _fmt(l["hours"], digits=2)))
    lines += ["", "**Total: %.2f h** (%.0f s, the sum of the lines above)" % (projection["total_h"],
                                                                           projection["total_s"])]
    parted = [l for l in projection["lines"] if l.get("parts")]
    if parted:
        lines += ["", "Per-model parts:", "",
                  "| line | model | k | messages | runs | epochs | units / epoch | unit | s / unit | basis | hours |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for l in parted:
            for p in l["parts"]:
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    l["id"], p["model"], _dash(p.get("k")), _dash(p.get("messages")), p["runs"], _dash(p["epochs"]),
                    p["units_per_epoch"], p["unit"], _fmt(p["s_per_unit"], digits=4), p["basis"],
                    _fmt(p["seconds"], 3600, 2)))
    lines += ["", "Formulas and sources:", ""]
    for l in projection["lines"]:
        lines.append("- `%s` (%s): %s = %.0f s. Source: %s" % (l["id"], l["basis"], l["formula"], l["seconds"],
                                                              l["source"]))
    return lines


def _what_if_lines(rows):
    lines = ["", "## What-if", "", "Total projected hours when the matrix is trimmed (everything else unchanged):", "",
             "| option | lr_grid | max_epochs | few-shot base runs | total h |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %d | %d | %d | %s |" % (r["label"], r["lr_grid"], r["max_epochs"], r["fewshot_base_runs"],
                                                     _fmt(r["total_h"], digits=2)))
    return lines


def _disk_lines(d):
    g = 2**30
    return ["", "## Disk", "",
            "- Base model (fp16 safetensors): %s GiB; train state: %s GiB (%s)"
            % (_fmt(d["model_bytes"], g, 2), _fmt(d["train_state_bytes"], g, 2), d["train_state_basis"]),
            "- One run's peak: %s GiB (three train states during checkpoint rotation, its per-epoch exports, its final "
            "checkpoint); ray_train.py refuses below %s GiB free" % (_fmt(d["per_run_peak_bytes"], g, 2),
                                                                   _fmt(d["run_floor_bytes"], g, 2)),
            "- Selection epoch exports kept until Phase 5 scores them on val: %s GiB; Laya final checkpoints kept: "
            "%s GiB" % (_fmt(d["selection_exports_bytes"], g, 2), _fmt(d["laya_final_checkpoints_bytes"], g, 2))]


def _attempt_lines(attempts, status):
    lines = ["", "## Attempts", "",
             "Every probe attempt runs in a fresh process; an MPS out-of-memory attempt is followed by the next rung "
             "of the LAYA-03 ladder. Chain status: **%s**." % status, "",
             "| # | rung | status | run id | detail |", "|---|---|---|---|---|"]
    for i, a in enumerate(attempts):
        # a failed attempt's reason wins over the child's line: after a resume-stage OOM the line says "rerun the
        # same command to resume", which the orchestrator's deleted storage makes untrue (04-REVIEW IN-06)
        detail = a.get("reason") or a.get("line") or ("weaker: head-only" if runs.weaker(a["rung"]) else "")
        lines.append("| %d | %s | %s | %s | %s |" % (i, a["rung"], a["status"], _dash(a.get("run_id")),
                                                     detail.replace("|", "/") or "-"))
    if status == "ladder_exhausted":
        lines += ["", "**The fallback ladder is exhausted: no rung fit in memory, nothing was measured.**"]
    elif status != "ok":
        lines += ["", "**The probe failed (status %s): nothing was measured.**" % status]
    return lines


def _resume_lines(interrupt, resume):
    lines = ["", "## Interruption and resume", ""]
    if not interrupt or interrupt.get("mode") == "none":
        lines.append("- No interruption (--interrupt none).")
    else:
        lines.append("- Mode %s after epoch %s%s; missed: %s; leftover processes after the kill: %s; available memory "
                     "before the rerun: %s GiB" % (interrupt["mode"], interrupt["after_epoch"],
                                                   "" if interrupt.get("delay_s") is None
                                                   else ", %.0f s into the next epoch" % interrupt["delay_s"],
                                                   interrupt["missed"], _dash(interrupt.get("leftover_processes")),
                                                   _fmt(interrupt.get("post_interrupt_available_gib"), digits=2)))
        if interrupt.get("missed"):
            lines.append("- **The kill came too late: the run finished before it, so it was not interrupted.**")
    if resume:
        lines.append("- Segments %d; resume load %s s; peak memory while resuming %s GiB" % (
            resume["segments"], _fmt(resume["resume_load_s"], digits=1),
            _fmt(resume["resume_peak_total_bytes"], 2**30, 2)))
        if not resume["resume_demonstrated"]:
            lines.append("- **The resume was not demonstrated** (it needs two or more segments, epoch rows 1..E and "
                         "resumed flags False then True).")
        else:
            lines.append("- Resume demonstrated: the rerun of the same command continued from the epoch-1 checkpoint.")
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
    basis, source and formula, the what-if table, the disk budget and the assumptions to confirm."""
    meta = report["meta"]
    lines = ["# Compute checkpoint (Phase 4, LAYA-07)", ""]
    if not meta["protocol"]["protocol_complete"]:
        lines += ["**SMOKE RUN: not a compute checkpoint.** The run left the protocol (not mps, not the pinned "
                  "weights, not the committed selection_train, a start config other than the spike's, --cpus, "
                  "--epochs or --seed other than %s, or a test hook); its numbers must not be used for the approval."
                  % ", ".join("%s %s" % kv for kv in sorted(PROTOCOL.items())), ""]
    if report.get("status", "ok") != "ok":
        lines += ["**No measurement (status %s):** see Attempts; there is no projection to approve."
                  % report["status"], ""]
    if spike.busy_machine(report.get("preflight")):
        lines += ["**Busy-machine run:** the preflight was overridden or swap grew while idle; the measured rate may "
                  "be slower than on a quiet machine.", ""]
    grew = swap_grew_epochs(report.get("measured"))
    if grew:
        lines += ["**Swap grew during measured epoch(s) %s** (largest %s MiB): the run was under memory pressure (the "
                  "spike's D-01 gate); confirm the rate before approval."
                  % (", ".join(map(str, grew)),
                     _fmt(max(r["swap_delta_bytes"] for r in report["measured"]["epochs"]), 2**20, 0)), ""]
    lines += _provenance_lines(meta)
    pf = report.get("preflight")
    if pf:
        lines.append("- Preflight: %.2f GiB available (threshold %s%s), swap grew while idle: %s"
                     % (pf["available_gib"], pf["min_available_gib"], ", overridden" if pf["override"] else "",
                        pf["swap_grew_while_idle"]))
    if report.get("attempts"):
        lines += _attempt_lines(report["attempts"], report.get("status", "ok"))
    if report.get("measured"):
        lines += _resume_lines(report.get("interrupt"), report.get("resume"))
        lines += _measured_lines(report["measured"])
        lines += _probe_lines(report["measured"])
    if report.get("projection"):
        lines += _projection_lines(report["projection"])
    if report.get("what_if"):
        lines += _what_if_lines(report["what_if"])
    if report.get("disk"):
        lines += _disk_lines(report["disk"])
    if report.get("assumptions"):
        lines += ["", "## Assumptions to confirm", ""] + ["- %s" % a for a in report["assumptions"]]
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
