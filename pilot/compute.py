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

Transparency (T-04-17): every projection line names its basis and its source. basis is the weakest input of the line:
"estimate" when any of its rates is an estimate (ESTIMATES, until plan 04-06 measures BERTurk), "provisional" for the
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
# labelled estimates until measured (plan 04-06 replaces both with its BERTurk timing probe)
ESTIMATES = {"berturk_s_per_step": 1.3, "berturk_s_per_msg": 0.01}
ESTIMATE_SOURCES = {
    "berturk_s_per_step": "about 30-60 s per BERTurk epoch of 46 batches of 32 on MPS, upper end [ASSUMED, RESEARCH "
                          "Open Question 2]",
    "berturk_s_per_msg": "a 110M-parameter encoder scoring one message at max_len 128 on MPS [ASSUMED]",
}
# the committed CPU zero-shot scoring timing, the Laya scoring rate until a scoring probe measures it
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


def laya_run_s(n_items, epochs, laya):
    """Seconds of one Laya run of `epochs` epochs on `n_items` items at the measured rate and overheads."""
    steps = spike.optimizer_steps(n_items, laya["micro_batch"], laya["grad_accum"])
    return epochs * (steps * laya["s_per_step"] + laya["save_s_per_epoch"]) + laya["load_s"] + laya["calib_s"] + \
        laya["final_save_s"]


# ---- rates: measured when the probe measured them, else the committed CPU scoring rate or a labelled estimate ----

def laya_scoring_rate(measured):
    """(s per message, basis, source) of Laya scoring: measured["scoring"] when present, else the committed CPU
    zero-shot val timing (wall seconds / messages)."""
    s = measured.get("scoring")
    if s:
        return s["s_per_msg"], "measured", "scoring probe on %s" % s.get("device")
    with open(SCORING_DUMP, encoding="utf-8") as f:
        side = json.load(f)
    n, sec, sc = side["split"]["n_messages"], side["timing"]["seconds"], side["scorer"]
    return sec / n, "measured", ("committed zero-shot val dump %s: %.2f s wall for %d messages on %s with %d Ray "
                                 "actors x %d torch threads; not yet measured on the fine-tuned scoring device"
                                 % (repo_relpath(SCORING_DUMP), sec, n, sc["device"], sc["workers"],
                                    sc["torch_threads"]))


def berturk_rates(measured):
    """(s per optimizer step, s per scored message, basis, source) of BERTurk: measured["berturk"] when its probe
    finished ok, else ESTIMATES."""
    b = measured.get("berturk")
    if b and b.get("status") == "ok":
        return b["s_per_step"], b["s_per_msg_infer"], "measured", "BERTurk timing probe on %s" % b.get("device")
    return (ESTIMATES["berturk_s_per_step"], ESTIMATES["berturk_s_per_msg"], "estimate",
            "ESTIMATES: %s; %s" % (ESTIMATE_SOURCES["berturk_s_per_step"], ESTIMATE_SOURCES["berturk_s_per_msg"]))


def _laya_source(laya):
    return ("probe run on %s: s/step = the largest train_s / optimizer_steps over %d measured epoch(s); save, load, "
            "calibration and final-save seconds from the same record" % (laya["device"], laya["epochs_measured"]))


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
    when every part agrees; basis is the weakest part (estimate), else provisional for an unfrozen protocol."""
    basis = "estimate" if any(p["basis"] == "estimate" for p in parts) else \
        ("provisional" if provisional else "measured")
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
    formula = ("%s (%d) x [%s (%d) x (%d optimizer steps (%d %s %s items, micro-batch %d, grad-accum %d) x %.4f s "
               "+ %.1f s save) + %.1f s load + %.1f s calibration + %.1f s final save]"
               % (keys[0], runs_, keys[1], epochs, part["units_per_epoch"], n, split, items_key[8:],
                  laya["micro_batch"], laya["grad_accum"], laya["s_per_step"], laya["save_s_per_epoch"],
                  laya["load_s"], laya["calib_s"], laya["final_save_s"]))
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


def assumptions(matrix, measured=None, counts=None):
    """What the user must confirm or trim at the compute checkpoint (plan 04-08), one sentence each."""
    measured = measured or {}
    out = ["Matrix value %s = %s is provisional (RESEARCH A2): %s"
           % (k, ", ".join(str(x) for x in v) if isinstance(v, list) else v, MATRIX_SOURCES[k])
           for k, v in matrix.items()]
    out += ["Negatives are drawn per seed (RESEARCH A4): ray_train.py --neg-seed defaults to --seed, so the seed "
            "variance includes the neg8 sampling; pin --neg-seed 0 to exclude it",
            "The git sha is recorded per run segment and kept out of the run_id (RESEARCH A3): a resume across a code "
            "change is flagged code_changed_on_resume, not refused"]
    b = measured.get("berturk")
    if not (b and b.get("status") == "ok"):
        out.append("BERTurk rates are estimates until plan 04-06's BERTurk timing probe measures them: %.2f s per "
                   "optimizer step (%s) and %.3f s per scored message (%s)"
                   % (ESTIMATES["berturk_s_per_step"], ESTIMATE_SOURCES["berturk_s_per_step"],
                      ESTIMATES["berturk_s_per_msg"], ESTIMATE_SOURCES["berturk_s_per_msg"]))
    if not measured.get("scoring"):
        out.append("The Laya scoring rate is the committed CPU zero-shot rate (%s) until an MPS scoring probe "
                   "measures it" % repo_relpath(SCORING_DUMP))
    s_msg = berturk_rates(measured)[1]                     # the BERTurk scoring rate the projection uses
    out += ["Scoring runs in fp32 on MPS (la.force_fp32 turns laya's fp16 autocast off); the scoring rate assumes fp32",
            "Per-epoch laya exports stay on (--export-epochs) so Phase 5 picks the epoch on val without retraining; "
            "Phase 5 prunes them after val scoring (see Disk)",
            "The measurement probe trains %s epoch(s) and plan 04-06 adds a kill and resume; the projection uses the "
            "slowest measured epoch" % ((measured.get("laya") or {}).get("epochs_measured", "-")),
            "Every split is counted over all its messages, without the 40-message calibration holdout, because Phase 5 "
            "calibrates on val (LAYA-05); slightly conservative",
            "BERTurk's per-epoch val evaluation for its epoch choice is not a separate line (lr_grid x berturk_epochs "
            "val passes at %.3f s per message%s)"
            % (s_msg, "" if not counts else ", about %.0f s" % (matrix["lr_grid"] * matrix["berturk_epochs"]
                                                                 * counts["val"]["n_messages"] * s_msg))]
    return out


def build_report(meta, measured, counts, matrix, projection, preflight, what_if=None, disk=None, assumptions=None):
    return {"schema": SCHEMA, "meta": meta, "preflight": preflight, "measured": measured, "counts": counts,
            "matrix": matrix, "projection": projection, "what_if": what_if, "disk": disk, "assumptions": assumptions}


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
             "- Load %s s, calibration %s s, final save %s s" % (_fmt(m["load_s"], digits=1),
                                                                _fmt(m["calib_s"], digits=1),
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
                  "weights, not the committed selection_train, a start config other than the spike's, or a test "
                  "hook); its numbers must not be used for the approval.", ""]
    if spike.busy_machine(report.get("preflight")):
        lines += ["**Busy-machine run:** the preflight was overridden or swap grew while idle; the measured rate may "
                  "be slower than on a quiet machine.", ""]
    lines += _provenance_lines(meta)
    pf = report.get("preflight")
    if pf:
        lines.append("- Preflight: %.2f GiB available (threshold %s%s), swap grew while idle: %s"
                     % (pf["available_gib"], pf["min_available_gib"], ", overridden" if pf["override"] else "",
                        pf["swap_grew_while_idle"]))
    lines += _measured_lines(report["measured"])
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
