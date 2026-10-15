"""Postfit (LAYA-05, LAYA-08): selection val dumps -> the val-fitted temperature, the val-selected thresholds and the
chosen (encoder LR, epoch), in one committed, byte-reproducible record. numpy, scipy and pilot modules only: no torch,
Ray or laya, so BERTurk (Phase 6) runs the same functions on its own val dumps.

Inputs are val dumps only. candidate() reads a run dir through pilot.dumps.load_dump(run_dir, "val", manifest) and
never takes a split argument; it refuses a path that is a file (a .jsonl, an .npz), a run dir that holds a test dump,
a --n subset val dump, a candidate trained on another split than selection_train (a final_train model has seen the
val folds, so its val outputs are in-sample), a calibrated candidate and one whose shipped T is not 1.0. Test never
shapes T, θ or the selection (ROADMAP Phase 5 criterion 3).

Protocol (05-RESEARCH Open Questions 1-4, confirmed by the user at plan 05-10: "Protocol decision: confirm"):
- A1: encoder LR grid LR_GRID, head LR fixed at HEAD_LR (laya's default); BERTurk shares the three values
- A2: OBJECTIVE and TIE_ORDER below; the objective is compared exactly, as the fraction 2TP / (2TP + FP + FN) of the
  summed counts, so equal scores tie exactly and the tie order decides, independent of input order
- A3: EPOCH_TRANSFER: the final seeds run the EPOCHS schedule and stop after the chosen epoch
- A4: SELECTION_SEED for every selection run, FINAL_SEEDS for the final runs
Temperature (TEMPERATURE_RULE): one noul T for every (message, tag) pair of val (all 48 tags of every message), the
minimiser of the mean binary cross-entropy of sigmoid(m / T) over log T in [T_MIN, T_MAX] (scipy's bounded scalar
minimiser); the two bounds are scored with the optimiser's point and a bound wins an exact tie, so an optimum outside
the band returns exactly T_MIN or T_MAX, laya.load's clamp band. It equals laya's own fit_temperature_map (float32
LBFGS on the same pairs) within 2e-3 (tests/test_postfit.py, through pilot.laya_adapter as the oracle only).
θ: pilot.metrics.select_theta on each variant's val probabilities: shipped (after the fitted T) and raw (T = 1).

The record (SCHEMA) holds the protocol, the rules, every candidate (sorted by encoder LR, then epoch) and the chosen
settings, from ids, digests and numbers only: no timestamp, no local path, no message text and no rounded value, so it
rebuilds byte for byte from the committed dumps in any argument order. prune_export deletes a ray_train.py epoch export
only after a verified full val dump of exactly that export exists.

usage: from pilot import postfit; record = postfit.fit(run_dirs, manifest_path); postfit.write_record(record, path)
       record, sha = postfit.read_record(path); postfit.prune_export(export_dir, dump_root, manifest_path)
       python postfit.py RUN_DIR [RUN_DIR ...] [--out results/postfit/laya.json] [--manifest splits.json]
       python postfit.py --prune EXPORT_DIR [EXPORT_DIR ...] [--dump-root results/dumps] [--manifest splits.json]
"""
import json
import os
import re
import shutil
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize_scalar

from pilot import dumps, metrics
from pilot.data import DEFAULT_MANIFEST, manifest_is_committed, repo_relpath, sha256_bytes, sha256_file, write_atomic

SCHEMA = "postfit/1"
T_MIN, T_MAX = 0.5, 5.0                     # laya TEMP_MIN / TEMP_MAX: laya.load clamps a served T to this band
LR_GRID = (2e-05, 3e-05, 5e-05)             # A1: encoder LR grid
HEAD_LR = 0.0001                            # A1: head LR, fixed (laya's default)
EPOCHS = 4                                  # the approved max_epochs: every selection run's schedule
SELECTION_SPLIT = "selection_train"
FINAL_SPLIT = "final_train"
SELECTION_SEED = 0                          # A4
FINAL_SEEDS = (0, 1, 2)                     # A4
TEMPERATURE_RULE = ("one noul T over every val (message, tag) pair: the minimiser of the mean binary cross-entropy of "
                    "sigmoid(m / T) over log T in [0.5, 5.0] (scipy bounded scalar minimiser), a bound winning an "
                    "exact tie")
OBJECTIVE = ("val micro-F1 at the val-selected global θ on probabilities after the val-fitted T, compared exactly as "
             "2TP / (2TP + FP + FN) of the summed counts")
TIE_ORDER = "equal objectives: fewer epochs first, then the lower encoder LR"
EPOCH_TRANSFER = ("the final seeds train on final_train with the selection schedule (--epochs %d) and stop after the "
                  "chosen epoch (--stop-after-epoch), never --epochs k*" % EPOCHS)
CANDIDATE_SCORER_KEYS = ("device", "dtype", "amp_enabled")
# canonical run-config keys a final seed may differ in from the chosen selection run (LAYA-08): its split and data,
# its seeds, its stop epoch (checked against the chosen epoch on its own) and the record it is bound to
FINAL_FREE_KEYS = ("split", "data_sha256", "data_role", "seed", "neg_seed", "stop_after_epoch", "postfit_sha256")
EXPORT_RE = re.compile(r"^epoch-[0-9]{3}$")              # ray_train.py's <storage>/<run_id>/exports/epoch-NNN


def _nll(m, y):
    """The mean binary cross-entropy of sigmoid(m / exp(log_t)) against y, as a function of log T."""
    def nll(log_t):
        z = m / np.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, z) - y * z))
    return nll


def fit_temperature(logits, Y):
    """(T, NLL at T, NLL at T = 1) of the val margins `logits` [n, 48] against the gold Y [n, 48], every pair counted.

    SystemExit on empty input or margins that are all exactly 0 (every T scores the same: T is not identifiable).
    """
    m = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(Y, dtype=np.float64).ravel()
    if m.size == 0:
        raise SystemExit("no (message, tag) pairs to fit a temperature on; score the full val split")
    if m.shape != y.shape:
        raise SystemExit("%d margins for %d gold labels; the dump and the gold must cover the same pairs"
                         % (m.size, y.size))
    if not np.any(m != 0.0):
        raise SystemExit("every val margin is 0, so every T fits equally well (T is not identifiable); check the dump")
    nll = _nll(m, y)
    r = minimize_scalar(nll, bounds=(np.log(T_MIN), np.log(T_MAX)), method="bounded", options={"xatol": 1e-8})
    best_t, best = None, None
    for t in (T_MIN, T_MAX, float(np.exp(r.x))):            # bounds first: a bound wins an exact tie
        v = nll(np.log(t))
        if best is None or v < best:
            best_t, best = t, v
    return float(best_t), best, nll(0.0)


def candidate(run_dir, manifest_path=DEFAULT_MANIFEST):
    """The selection candidate of one run dir's full val dump: its training provenance, its val-fitted T, θ for the
    shipped (after T) and raw (T = 1) variants and the objective counts. Reads only the val dump (see the refusals in
    the module docstring)."""
    if os.path.isfile(run_dir):
        raise SystemExit("postfit takes run dirs that hold a val dump; %s is a file (it never reads a .jsonl, a test "
                         "dump or any other split)" % run_dir)
    for ext in (".npz", ".json"):
        if os.path.exists(os.path.join(run_dir, "test" + ext)):
            raise SystemExit("%s holds a test dump; a selection candidate is never scored on test in Phase 5 and "
                             "postfit reads val only" % run_dir)
    logits, uids, side, entry = dumps.load_dump(run_dir, "val", manifest_path)
    if entry["role"] != "val":
        raise SystemExit("%s: postfit reads val dumps only (manifest role %s); test never shapes T, θ or the "
                         "selection" % (run_dir, entry["role"]))
    if side.get("n_subset") is not None:
        raise SystemExit("%s: a --n subset val dump; postfit needs the full val split (rescore val without --n)"
                         % run_dir)
    tr = side.get("training")
    if not isinstance(tr, dict):
        raise SystemExit("%s: the val dump has no training block; postfit selects among ray_train.py exports scored "
                         "with this score.py" % run_dir)
    if tr.get("split") != SELECTION_SPLIT:
        raise SystemExit("%s: trained on %s; postfit selects only on %s runs (a %s model's val is in-sample)"
                         % (run_dir, tr.get("split"), SELECTION_SPLIT, FINAL_SPLIT))
    if tr.get("calibrated") is not False:
        raise SystemExit("%s: calibrated %r; postfit fits T on the raw margins of uncalibrated exports only"
                         % (run_dir, tr.get("calibrated")))
    shipped = (side.get("temperature") or {}).get("shipped")
    if shipped != 1.0:
        raise SystemExit("%s: shipped temperature %r; postfit fits T on the raw margins of uncalibrated exports only "
                         "(T = 1)" % (run_dir, shipped))
    Y = dumps.read_gold(entry, manifest_path, uids)
    T, nll_t, nll_1 = fit_temperature(logits, Y)
    P = metrics.probabilities(logits, T)
    theta = {"shipped": metrics.select_theta(Y, P), "raw": metrics.select_theta(Y, metrics.probabilities(logits, 1.0))}
    at_theta = metrics.metrics(Y, P, theta["shipped"]["global"])
    counts = {k: sum(at_theta["per_tag"][t][k] for t in metrics.TAGS) for k in ("tp", "fp", "fn")}
    scorer = side.get("scorer") or {}
    return {"dump_run_id": side["run_id"], "dump_sha256": side["dump_sha256"], "uids_sha256": dumps.uids_sha256(uids),
            "system_kind": (side.get("system") or {}).get("kind"), "train_run_id": tr.get("run_id"),
            "encoder_lr": tr.get("encoder_lr"), "epoch": tr.get("epoch"), "epochs_schedule": tr.get("epochs_schedule"),
            "seed": side.get("seed"), "rung": tr.get("rung"), "fallback": tr.get("fallback"),
            "weaker": tr.get("weaker"), "loss_by_epoch": tr.get("loss_by_epoch"), "canonical": tr.get("canonical"),
            "scorer": {k: scorer.get(k) for k in CANDIDATE_SCORER_KEYS},
            "temperature": T, "nll_raw": nll_1, "nll_fitted": nll_t, "objective": counts,
            "val_micro_f1_at_0_5": metrics.metrics(Y, P, 0.5)["micro_f1"],
            "val_micro_f1_at_global_theta": at_theta["micro_f1"], "theta": theta}


def _json(v):
    """Exact JSON form of a value (no float tolerance; 1, 1.0 and true differ)."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def score(c):
    """The exact objective of a candidate: Fraction(2TP, 2TP + FP + FN) of its summed counts, 0 on a 0 denominator."""
    o = c["objective"]
    den = 2 * o["tp"] + o["fp"] + o["fn"]
    return Fraction(2 * o["tp"], den) if den else Fraction(0)


def select(cands, lr_grid=LR_GRID, epochs=EPOCHS, selection_seed=SELECTION_SEED, head_lr=HEAD_LR):
    """The chosen candidate: the highest score(); ties go to fewer epochs, then the lower encoder LR. SystemExit unless
    the candidates are exactly one full selection grid (lr_grid x epochs 1..epochs) of one system and recipe."""
    if not cands:
        raise SystemExit("no candidates; pass the selection run dirs that hold the val dumps")
    pairs = [(c["encoder_lr"], c["epoch"]) for c in cands]
    dup = sorted({p for p in pairs if pairs.count(p) > 1}, key=repr)
    if dup:
        raise SystemExit("(encoder_lr, epoch) %s appear more than once; pass each candidate once" % dup)
    shas = [c["dump_sha256"] for c in cands]
    if len(set(shas)) != len(shas):
        raise SystemExit("two candidates hold the same val dump (dump_sha256); pass each candidate once")
    want = {(lr, e) for lr in lr_grid for e in range(1, epochs + 1)}
    got = set(pairs)
    if got != want:
        raise SystemExit("the candidates are not the selection grid %s x epochs 1..%d: missing %s, unexpected %s; "
                         "score every epoch export of every LR run on val"
                         % (list(lr_grid), epochs, sorted(want - got, key=repr), sorted(got - want, key=repr)))
    kinds = sorted({str(c["system_kind"]) for c in cands})
    if len(kinds) != 1:
        raise SystemExit("the candidates mix system kinds %s; postfit selects within one system" % kinds)
    seeds = sorted({repr(c["seed"]) for c in cands})
    if seeds != [repr(selection_seed)]:
        raise SystemExit("candidate seeds %s; every selection run uses seed %r" % (seeds, selection_seed))
    if len({c["uids_sha256"] for c in cands}) != 1:
        raise SystemExit("unpaired: the candidates score different val messages or a different order")
    scorers = sorted({_json(c["scorer"]) for c in cands})
    if len(scorers) != 1:
        raise SystemExit("the candidates were scored on different scorer settings %s; rescore them alike" % scorers)
    runs = {}
    for c in cands:
        canon = c["canonical"]
        if not isinstance(canon, dict):
            raise SystemExit("%s: no canonical run config; retrain with this ray_train.py" % c["dump_run_id"])
        for key, want_v in (("split", SELECTION_SPLIT), ("epochs", epochs), ("head_lr", head_lr),
                            ("encoder_lr", c["encoder_lr"])):
            if _json(canon.get(key)) != _json(want_v):
                raise SystemExit("%s: canonical %s %r, expected %r" % (c["dump_run_id"], key, canon.get(key), want_v))
        if runs.setdefault(_json(c["encoder_lr"]), c["train_run_id"]) != c["train_run_id"]:
            raise SystemExit("encoder LR %r comes from two train runs (%s, %s); one run per LR"
                             % (c["encoder_lr"], runs[_json(c["encoder_lr"])], c["train_run_id"]))
    base = cands[0]["canonical"]
    for c in cands[1:]:
        canon = c["canonical"]
        for key in sorted(set(base) | set(canon)):
            if key != "encoder_lr" and (key not in base or key not in canon or _json(base[key]) != _json(canon[key])):
                raise SystemExit("the candidates' canonical configs differ in %s (%r vs %r); they may differ only in "
                                 "encoder_lr" % (key, base.get(key), canon.get(key)))
    return max(cands, key=lambda c: (score(c), -c["epoch"], -c["encoder_lr"]))


def _val_entry(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    entry = (m.get("splits") or {}).get("val")
    if not entry or entry.get("role") != "val":
        raise SystemExit("%s registers no val split; run `make data`" % manifest_path)
    return entry


def build_record(cands, chosen, manifest_path, lr_grid=LR_GRID, epochs=EPOCHS, selection_seed=SELECTION_SEED,
                 final_seeds=FINAL_SEEDS, head_lr=HEAD_LR):
    """The postfit record (SCHEMA) of the candidates and the chosen one (select's result)."""
    entry = _val_entry(manifest_path)
    for c in cands:
        if c["uids_sha256"] != entry["uids_sha256"]:
            raise SystemExit("%s does not cover the registered val split (uids_sha256); rescore val"
                             % c["dump_run_id"])
    if chosen["dump_sha256"] not in {c["dump_sha256"] for c in cands}:
        raise SystemExit("the chosen candidate is not among the candidates")
    stop = None if chosen["epoch"] == chosen["epochs_schedule"] else chosen["epoch"]
    return {
        "schema": SCHEMA, "system_kind": chosen["system_kind"],
        "split": {"name": "val", "path": entry["path"], "sha256": entry["sha256"], "uids_sha256": entry["uids_sha256"],
                  "n_messages": entry["n_messages"]},
        "manifest": {"path": repo_relpath(manifest_path), "sha256": sha256_file(manifest_path),
                     "is_committed": manifest_is_committed(manifest_path)},
        "protocol": {"selection_split": SELECTION_SPLIT, "final_split": FINAL_SPLIT, "lr_grid": list(lr_grid),
                     "head_lr": head_lr, "epochs": epochs, "selection_seed": selection_seed,
                     "final_seeds": list(final_seeds)},
        "rules": {"temperature": TEMPERATURE_RULE, "temperature_bounds": [T_MIN, T_MAX], "objective": OBJECTIVE,
                  "tie_order": TIE_ORDER, "epoch_transfer": EPOCH_TRANSFER,
                  "theta": {"tie_break": metrics.TIE_RULE, "min_val_positives": metrics.MIN_VAL_POSITIVES,
                            "per_tag": "a tag's own θ only with >= %d val positives, else the global θ"
                                       % metrics.MIN_VAL_POSITIVES}},
        "candidates": sorted(cands, key=lambda c: (c["encoder_lr"], c["epoch"])),
        "chosen": {"train_run_id": chosen["train_run_id"], "dump_run_id": chosen["dump_run_id"],
                   "dump_sha256": chosen["dump_sha256"], "encoder_lr": chosen["encoder_lr"], "epoch": chosen["epoch"],
                   "epochs_schedule": chosen["epochs_schedule"], "stop_after_epoch": stop,
                   "canonical": chosen["canonical"], "temperature": chosen["temperature"], "theta": chosen["theta"],
                   "scorer": chosen["scorer"], "rung": chosen["rung"], "fallback": chosen["fallback"],
                   "weaker": chosen["weaker"], "val_micro_f1_at_global_theta": chosen["val_micro_f1_at_global_theta"]},
        "final": {"split": FINAL_SPLIT, "seeds": list(final_seeds), "epochs": epochs, "stop_after_epoch": stop}}


def fit(run_dirs, manifest_path=DEFAULT_MANIFEST, **grid):
    """candidate() of every run dir, select(), build_record(): the record of one selection grid."""
    cands = [candidate(d, manifest_path) for d in run_dirs]
    chosen = select(cands, **{k: v for k, v in grid.items() if k != "final_seeds"})
    return build_record(cands, chosen, manifest_path, **grid)


def record_bytes(record):
    return (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_record(record, path):
    """Write the record atomically, as indented JSON with a trailing newline (no timestamp: rebuilds byte for byte)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_atomic(path, record_bytes(record))
    return path


def read_record(path):
    """(record, sha256 of the file bytes) of a postfit record; SystemExit on a missing file, invalid JSON, another
    schema or a missing chosen, final or protocol block. The sha256 is of the same bytes that were parsed."""
    if not os.path.isfile(path):
        raise SystemExit("%s not found; build it with postfit.py RUN_DIR... --out %s" % (path, path))
    with open(path, "rb") as f:
        raw = f.read()
    try:
        record = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); rebuild it with postfit.py" % (path, e))
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise SystemExit("%s: schema %r, expected %s; rebuild it with this postfit.py"
                         % (path, record.get("schema") if isinstance(record, dict) else None, SCHEMA))
    for key in ("chosen", "final", "protocol"):
        if not isinstance(record.get(key), dict):
            raise SystemExit("%s has no %s block; rebuild it with this postfit.py" % (path, key))
    return record, sha256_bytes(raw)


def check_final(record, canon):
    """Refuse (SystemExit) a final-seed command whose canonical run config `canon` (pilot.runs.canonical of the
    ray_train.py command) is not bound to the record: the split must be the record's final split, the seed one of its
    final seeds, stop_after_epoch the chosen one (None when the chosen epoch is the last of the schedule), and every
    other key outside FINAL_FREE_KEYS must equal the chosen selection run's canonical value exactly (JSON equality: 1
    and 1.0 differ). The first differing key in sorted order is named with both values (LAYA-08)."""
    chosen, final = record["chosen"], record["final"]
    if canon.get("split") != final.get("split"):
        raise SystemExit("--postfit trains the final seeds on %s; --data is %s" % (final.get("split"), canon.get("split")))
    if canon.get("seed") not in (final.get("seeds") or []):
        raise SystemExit("--seed %r is not one of the record's final seeds %s" % (canon.get("seed"), final.get("seeds")))
    stop = chosen.get("stop_after_epoch")
    if canon.get("stop_after_epoch") != stop:
        want = ("omit --stop-after-epoch" if stop is None else "pass --stop-after-epoch %d" % stop)
        raise SystemExit("--postfit: the chosen candidate is epoch %r of the --epochs %r schedule, so %s (--stop-after-"
                         "epoch is %r here)" % (chosen.get("epoch"), chosen.get("epochs_schedule"), want,
                                                canon.get("stop_after_epoch")))
    want_canon = chosen.get("canonical")
    if not isinstance(want_canon, dict):
        raise SystemExit("the postfit record's chosen run has no canonical config; rebuild it with this postfit.py")
    missing = object()
    for key in sorted((set(canon) | set(want_canon)) - set(FINAL_FREE_KEYS)):
        have, want = canon.get(key, missing), want_canon.get(key, missing)
        if have is missing or want is missing or _json(have) != _json(want):
            raise SystemExit("--postfit: %s is %r here but %r in the chosen selection run %s; pass the chosen value "
                             "(postfit.py --print-final-commands prints each command)"
                             % (key, None if have is missing else have, None if want is missing else want,
                                chosen.get("train_run_id")))


def final_block(record, sha256, path):
    """training.postfit of a final seed's config (and ray_train.py's cfg["postfit"]): the record's sha256 and
    repo-relative path, the chosen settings, the val-fitted T (written as the config's noul temperature) and θ, and
    the selection rules. Ids, digests and numbers only."""
    chosen, rules = record["chosen"], record.get("rules") or {}
    t = chosen.get("temperature")
    if isinstance(t, bool) or not isinstance(t, float) or not T_MIN <= t <= T_MAX:
        raise SystemExit("the postfit record's chosen temperature is %r, not a float in laya's band [%s, %s]; rebuild "
                         "it with this postfit.py" % (t, T_MIN, T_MAX))
    return {"sha256": sha256, "record": repo_relpath(path),
            "chosen": {k: chosen.get(k) for k in ("train_run_id", "dump_run_id", "encoder_lr", "epoch",
                                                  "epochs_schedule", "stop_after_epoch",
                                                  "val_micro_f1_at_global_theta")},
            "temperature": chosen["temperature"], "theta": chosen.get("theta"),
            "rules": {k: rules.get(k) for k in ("objective", "tie_order", "epoch_transfer")}}


def prune_export(export_dir, dump_root=dumps.DUMP_ROOT, manifest_path=DEFAULT_MANIFEST):
    """Delete one ray_train.py epoch export once its verified full val dump exists; returns the dump's run id.

    Refused, and left in place: a symlink (the export or its exports/ dir), a path that is not a real
    <storage>/<run_id>/exports/epoch-NNN dir holding rl_agent_config.json, and an export without a full val dump under
    dump_root whose sidecar checkpoint_identity equals the export's pilot.weights.identity (load_dump verifies the dump
    bytes). Only that one dir is ever deleted.
    """
    from pilot.weights import identity                      # torch-free; imported here, the only caller
    path = os.path.abspath(export_dir)
    if os.path.islink(path) or os.path.islink(os.path.dirname(path)):
        raise SystemExit("%s is a symlink (or sits in a symlinked exports dir); refusing to delete it" % export_dir)
    real = os.path.realpath(path)
    if not (os.path.isdir(real) and EXPORT_RE.match(os.path.basename(real))
            and os.path.basename(os.path.dirname(real)) == "exports"
            and os.path.isfile(os.path.join(real, "rl_agent_config.json"))):
        raise SystemExit("%s is not a ray_train export (<storage>/<run_id>/exports/epoch-NNN); refusing to delete it"
                         % export_dir)
    ident = identity(real)
    rid = dumps.run_id("laya", ident)
    run_dir = os.path.join(dump_root, rid)
    try:
        _, _, side, _ = dumps.load_dump(run_dir, "val", manifest_path)
    except SystemExit as e:
        raise SystemExit("no verified full val dump for this export under %s (%s); score it on val first, then prune"
                         % (dump_root, e))
    if side.get("n_subset") is not None or (side.get("system") or {}).get("checkpoint_identity") != ident:
        raise SystemExit("no verified full val dump for this export under %s (%s is a subset dump or another "
                         "checkpoint's); score it on val first, then prune" % (dump_root, rid))
    shutil.rmtree(real)
    return rid
