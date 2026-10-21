"""Postfit (LAYA-05, LAYA-08): selection val dumps -> the val-fitted temperature, the val-selected thresholds and the
chosen (encoder LR, epoch), in one committed, byte-reproducible record. numpy, scipy and pilot modules only: no torch,
Ray or laya, so BERTurk (Phase 6) runs the same functions on its own val dumps.

Inputs are val dumps only. candidate() reads a run dir through pilot.dumps.load_dump(run_dir, "val", manifest) and
never takes a split argument; it refuses a path that is a file (a .jsonl, an .npz), a run dir that holds a test dump,
a --n subset val dump, a candidate trained on another split than selection_train (a final_train model has seen the
val folds, so its val outputs are in-sample), on the committed manifest a candidate whose canonical data_sha256 is not
the committed selection_train sha256 (06-REVIEW WR-02: the bytes, not the split name), a calibrated candidate and one
whose shipped T is not 1.0. Test never
shapes T, θ or the selection (ROADMAP Phase 5 criterion 3).

Protocol (05-RESEARCH Open Questions 1-4, confirmed by the user at plan 05-10: "Protocol decision: confirm"):
- A1: encoder LR grid LR_GRID, head LR fixed at HEAD_LR (laya's default)
- A2: OBJECTIVE and TIE_ORDER below; the objective is compared exactly, as the fraction 2TP / (2TP + FP + FN) of the
  summed counts, so equal scores tie exactly and the tie order decides, independent of input order
- A3: EPOCH_TRANSFER: the final seeds run the EPOCHS schedule and stop after the chosen epoch
- A4: SELECTION_SEED for every selection run, FINAL_SEEDS for the final runs
Per system kind (protocol(kind), the one source of a selection grid): "laya" is the constants above; "berturk" is
pilot.berturk.PROTOCOL (Phase 6, confirmed at plan 06-02): the same LR grid and the same fixed head LR as Laya (HEAD_LR,
since the BASE-03 fix of 2026-10-21; tests keep both equal), epochs 1..pilot.berturk.EPOCHS (30 since the second BASE-03
fix of 2026-10-21) and the same seeds. The superseded protocols (pilot.berturk.SUPERSEDED, epochs 10, the first with
head_lr None: each candidate's head LR tied to its own encoder LR, which select still accepts) are kept only so their
records still reproduce. fit() with no grid arguments takes the protocol of its
candidates' system kind (one kind, else refused), so BERTurk runs exactly these functions, objective, tie order, T and
θ rules on its own val dumps (BASE-02), and the Laya record is byte for byte what it was before the dispatch.
rules.epoch_transfer names the kind's schedule (EPOCH_TRANSFER_TEMPLATE).
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

Final seeds (LAYA-08): ray_train.py --postfit RECORD calls check_final, which refuses a final-seed command whose
canonical run config is not the chosen selection run's (outside FINAL_FREE_KEYS; neg_seed must still equal the seed, and
a record of the committed manifest binds data_sha256 to the committed final_train), and stamps final_block into the
final config. final_command prints the exact ray_train.py argv of each final seed from the record (nobody retypes a
hyperparameter); final_manifest (FINAL_SCHEMA) lists the finished checkpoints only when they are exactly the record's
final seeds, each bound to its sha256, at the chosen epoch and T, with one shared config_sha256_sans_seed.

render_markdown renders the README's Phase 5 block (postfit.py --render) from a Laya record and its final manifest:
the chosen settings, the rules, the encoder-LR and grid-edge lines against pilot.compute.MATRIX, the θ and T bound
positions, the candidates and the final seeds, rounded for display only (it changes no record byte).

usage: from pilot import postfit; record = postfit.fit(run_dirs, manifest_path); postfit.write_record(record, path)
       record, sha = postfit.read_record(path); postfit.prune_export(export_dir, dump_root, manifest_path)
       postfit.final_command(record, seed, record_path, storage, out_prefix); postfit.final_manifest(path, ckpt_dirs)
       postfit.final_data(record, manifest_path)
       postfit.protocol("laya" | "berturk"); text = postfit.render_markdown(record, final_manifest)
       python postfit.py RUN_DIR [RUN_DIR ...] [--out results/postfit/laya.json] [--manifest splits.json]
       python postfit.py --prune EXPORT_DIR [EXPORT_DIR ...] [--dump-root results/dumps] [--manifest splits.json]
       python postfit.py --print-final-commands RECORD --storage DIR --out-prefix PATH [--base DIR] [--manifest M]
       python postfit.py --final-manifest RECORD CKPT [CKPT ...] [--out results/postfit/laya-final.json]
       python postfit.py --render RECORD [FINAL_MANIFEST]
"""
import json
import os
import re
import shutil
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize_scalar

from pilot import dumps, metrics, pins
from pilot.data import DEFAULT_MANIFEST, manifest_is_committed, repo_relpath, sha256_bytes, sha256_file, write_atomic

SCHEMA = "postfit/1"
FINAL_SCHEMA = "postfit-final/1"            # final_manifest: the three final seeds bound to one record
T_MIN, T_MAX = 0.5, 5.0                     # laya TEMP_MIN / TEMP_MAX: laya.load clamps a served T to this band
LR_GRID = (2e-05,)                          # A1: one encoder LR for now (05-06 user override); a grid search later
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
EPOCH_TRANSFER_TEMPLATE = ("the final seeds train on final_train with the selection schedule (--epochs %d) and stop "
                           "after the chosen epoch (--stop-after-epoch), never --epochs k*")
EPOCH_TRANSFER = EPOCH_TRANSFER_TEMPLATE % EPOCHS           # Laya's; a record names its own kind's schedule
CANDIDATE_SCORER_KEYS = ("device", "dtype", "amp_enabled")
# canonical run-config keys a final seed may differ in from the chosen selection run (LAYA-08): its split and data,
# its seeds, its stop epoch (checked against the chosen epoch on its own) and the record it is bound to
FINAL_FREE_KEYS = ("split", "data_sha256", "data_role", "seed", "neg_seed", "stop_after_epoch", "postfit_sha256")
EXPORT_RE = re.compile(r"^epoch-[0-9]{3}$")              # ray_train.py's <storage>/<run_id>/exports/epoch-NNN
# the chosen block of a final manifest: final_manifest writes exactly these keys and render_markdown checks them
FINAL_CHOSEN_KEYS = ("train_run_id", "encoder_lr", "epoch", "epochs_schedule", "stop_after_epoch", "temperature")


def protocol(kind):
    """The selection grid of a system kind: {lr_grid, epochs, head_lr, selection_seed, final_seeds}, the keyword
    arguments of select and build_record. head_lr None ties each candidate's head LR to its own encoder LR (only the
    superseded BERTurk protocol, pilot.berturk.SUPERSEDED, uses it)."""
    if kind == "laya":
        return {"lr_grid": LR_GRID, "epochs": EPOCHS, "head_lr": HEAD_LR, "selection_seed": SELECTION_SEED,
                "final_seeds": FINAL_SEEDS}
    if kind == "berturk":
        from pilot import berturk                           # lazy: pilot.berturk imports this module
        return dict(berturk.PROTOCOL)
    raise SystemExit("no postfit protocol for system kind %r; expected laya or berturk" % (kind,))


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
    if manifest_is_committed(manifest_path):                 # 06-REVIEW WR-02: the bytes, not the split name
        with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
            want = ((json.load(f).get("splits") or {}).get(SELECTION_SPLIT) or {}).get("sha256")
        have = (tr.get("canonical") or {}).get("data_sha256")
        if have != want:
            raise SystemExit("%s: trained on data sha256 %s, not the committed %s sha256 %s (data/splits.json); a "
                             "record on the committed manifest selects only runs trained on it"
                             % (run_dir, have, SELECTION_SPLIT, want))
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
    the candidates are exactly one full selection grid (lr_grid x epochs 1..epochs) of one system and recipe. head_lr
    None: every candidate's canonical head_lr must equal its own encoder_lr (one LR for encoder and head)."""
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
        if head_lr is None and _json(canon.get("head_lr")) != _json(c["encoder_lr"]):
            raise SystemExit("%s: canonical head_lr %r is not its encoder_lr %r; this protocol ties the head LR to the "
                             "encoder LR (one LR)" % (c["dump_run_id"], canon.get("head_lr"), c["encoder_lr"]))
        for key, want_v in (("split", SELECTION_SPLIT), ("epochs", epochs),
                            ("head_lr", c["encoder_lr"] if head_lr is None else head_lr),
                            ("encoder_lr", c["encoder_lr"])):
            if _json(canon.get(key)) != _json(want_v):
                raise SystemExit("%s: canonical %s %r, expected %r" % (c["dump_run_id"], key, canon.get(key), want_v))
        if runs.setdefault(_json(c["encoder_lr"]), c["train_run_id"]) != c["train_run_id"]:
            raise SystemExit("encoder LR %r comes from two train runs (%s, %s); one run per LR"
                             % (c["encoder_lr"], runs[_json(c["encoder_lr"])], c["train_run_id"]))
    free = ("encoder_lr",) if head_lr is not None else ("encoder_lr", "head_lr")
    base = cands[0]["canonical"]
    for c in cands[1:]:
        canon = c["canonical"]
        for key in sorted(set(base) | set(canon)):
            if key not in free and (key not in base or key not in canon or _json(base[key]) != _json(canon[key])):
                raise SystemExit("the candidates' canonical configs differ in %s (%r vs %r); they may differ only in "
                                 "%s" % (key, base.get(key), canon.get(key), " and ".join(free)))
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
                  "tie_order": TIE_ORDER, "epoch_transfer": EPOCH_TRANSFER_TEMPLATE % epochs,
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
    """candidate() of every run dir, select(), build_record(): the record of one selection grid. Without grid
    arguments the grid is protocol() of the candidates' system kind (SystemExit when they mix kinds)."""
    cands = [candidate(d, manifest_path) for d in run_dirs]
    if not grid and cands:
        kinds = sorted({str(c["system_kind"]) for c in cands})
        if len(kinds) != 1:
            raise SystemExit("the candidates mix system kinds %s; postfit selects within one system" % kinds)
        grid = protocol(cands[0]["system_kind"])
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
    and 1.0 differ). The first differing key in sorted order is named with both values, a key missing on one side as
    <absent> (LAYA-08). Two free keys are
    bound on their own (05-REVIEW WR-01): neg_seed must equal the seed when the chosen run's did (A4: --neg-seed is
    never passed), and when the record was built on the committed manifest (manifest.is_committed) data_sha256 must be
    the committed final_train sha256 of data/splits.json, read at check time (the record stays byte-identical)."""
    chosen, final = record["chosen"], record["final"]
    if canon.get("split") != final.get("split"):
        raise SystemExit("--postfit trains the final seeds on %s; --data is %s"
                         % (final.get("split"), canon.get("split")))
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
    if "neg_seed" in want_canon and want_canon["neg_seed"] == want_canon.get("seed") and \
            canon.get("neg_seed") != canon.get("seed"):
        raise SystemExit("--postfit: neg_seed is %r but the seed is %r; the chosen selection run drew its negatives "
                         "with neg_seed == seed, so omit --neg-seed (it defaults to --seed)"
                         % (canon.get("neg_seed"), canon.get("seed")))
    if (record.get("manifest") or {}).get("is_committed") is True:
        with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
            committed = ((json.load(f).get("splits") or {}).get(final.get("split")) or {}).get("sha256")
        if canon.get("data_sha256") != committed:
            raise SystemExit("--postfit: the final seed's data sha256 is %s, not the committed %s sha256 %s "
                             "(data/splits.json); a record built on the committed manifest trains its final seeds "
                             "on the committed split (drop --manifest, or run `make data`)"
                             % (canon.get("data_sha256"), final.get("split"), committed))
    missing = object()
    for key in sorted((set(canon) | set(want_canon)) - set(FINAL_FREE_KEYS)):
        have, want = canon.get(key, missing), want_canon.get(key, missing)
        if have is missing or want is missing or _json(have) != _json(want):
            shown = ["<absent>" if v is missing else repr(v) for v in (have, want)]   # 05-REVIEW IN-06: not "None"
            raise SystemExit("--postfit: %s is %s here but %s in the chosen selection run %s; pass the chosen value "
                             "(postfit.py --print-final-commands prints each command)"
                             % (key, shown[0], shown[1], chosen.get("train_run_id")))


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


def _device_amp(device):
    """ray_train.resolve's AMP default: on only for cuda."""
    return device == "cuda"


def final_command(record, seed, record_path, storage, out_prefix, base="multilingual",
                  data="data/splits/final_train.jsonl", manifest=None):
    """The ray_train.py argv (starting with "ray_train.py") that trains final seed `seed` with the chosen settings:
    every hyperparameter from the chosen canonical config, the chosen stop epoch, --postfit record_path and
    --no-export-epochs; --out is "<out_prefix>-s<seed>". neg_seed is never passed (it defaults to the seed, A4).
    ray_train.py's check_final is still the binding; this only spares the operator the transcription."""
    final, chosen = record["final"], record["chosen"]
    if seed not in (final.get("seeds") or []):
        raise SystemExit("seed %r is not one of the record's final seeds %s" % (seed, final.get("seeds")))
    c = chosen.get("canonical")
    if not isinstance(c, dict):
        raise SystemExit("the postfit record's chosen run has no canonical config; rebuild it with this postfit.py")
    if base == "multilingual" and c.get("base_model_sha256") != pins.LAYA["sha256"]["model.safetensors"]:
        raise SystemExit("the chosen run trained another base (base_model_sha256 %s, not the pinned Laya weights); "
                         "pass --base DIR" % c.get("base_model_sha256"))
    argv = ["ray_train.py", "--data", data, "--base", base, "--device", c["device"], "--workers", str(c["workers"]),
            "--epochs", str(c["epochs"]), "--micro-batch", str(c["micro_batch"]), "--grad-accum", str(c["grad_accum"]),
            "--encoder-lr", repr(c["encoder_lr"]), "--head-lr", repr(c["head_lr"]), "--loss", c["loss"],
            "--neg", str(c["neg"])]
    for key, flag in (("max_len", "--max-len"), ("head_max_len", "--head-max-len"), ("max_messages", "--max-messages")):
        if c.get(key) is not None:
            argv += [flag, str(c[key])]
    for key, flag in (("freeze_encoder", "--freeze-encoder"), ("freeze_embeddings", "--freeze-embeddings"),
                      ("shuffle_options", "--shuffle-options")):
        if c.get(key):
            argv.append(flag)
    if not c.get("freeze_encoder") and c.get("grad_checkpointing") is False:
        argv.append("--no-grad-checkpointing")
    if c.get("amp") != _device_amp(c["device"]):
        argv.append("--amp" if c.get("amp") else "--no-amp")
    if chosen.get("stop_after_epoch") is not None:
        argv += ["--stop-after-epoch", str(chosen["stop_after_epoch"])]
    argv += ["--seed", str(seed), "--postfit", record_path, "--no-export-epochs", "--storage", storage,
             "--out", "%s-s%d" % (out_prefix, seed)]
    if manifest is not None:
        argv += ["--manifest", manifest]
    return argv


def final_data(record, manifest):
    """The --data of a final-seed command printed with --manifest M (05-REVIEW WR-03): the path M registers for the
    record's final split, joined to M's directory (pilot.data resolves split paths relative to the manifest), so the
    printed command verifies against M. SystemExit when M is unreadable or registers no such split."""
    split = record["final"]["split"]
    try:
        with open(manifest, encoding="utf-8") as f:
            entry = ((json.load(f).get("splits") or {}).get(split) or {})
    except (OSError, ValueError) as e:
        raise SystemExit("--manifest %s cannot be read (%s); pass a splits.json written by `make data`" % (manifest, e))
    if not isinstance(entry.get("path"), str):
        raise SystemExit("--manifest %s registers no %s split; regenerate it with `make data`" % (manifest, split))
    return os.path.join(os.path.dirname(manifest), entry["path"])


def _final_entry(record, sha, ckpt_dir):
    """One final_manifest entry of a finished --postfit checkpoint, or SystemExit naming what does not bind it."""
    from pilot.weights import checkpoint_digests, identity  # torch-free; imported here, like prune_export
    chosen, name = record["chosen"], os.path.basename(os.path.normpath(ckpt_dir))
    path = os.path.join(ckpt_dir, "rl_agent_config.json")
    if not os.path.isfile(path):                            # checked before any hashing (05-REVIEW IN-03)
        raise SystemExit("%s has no rl_agent_config.json; pass the final checkpoint dirs ray_train.py wrote" % name)
    ident = identity(ckpt_dir)
    checkpoint_digests(ckpt_dir, ident)                     # refuses missing weights, tokenizer/ or encoder/ files
    with open(path, "rb") as f:
        raw = f.read()
    if sha256_bytes(raw) != ident["config_sha256"]:
        raise SystemExit("%s: rl_agent_config.json changed while it was read; rerun" % name)
    cfg = json.loads(raw.decode("utf-8"))
    tr = cfg.get("training") if isinstance(cfg, dict) else None
    rt = tr.get("ray_train") if isinstance(tr, dict) else None
    if not isinstance(rt, dict) or not isinstance(rt.get("canonical"), dict):
        raise SystemExit("%s holds no ray_train run config; train it with ray_train.py --postfit" % name)
    canon, rec, prov = rt["canonical"], rt.get("record") or {}, rt.get("provenance") or {}
    if prov.get("manifest_is_committed") is not True or prov.get("data_sha256") != canon.get("data_sha256"):
        raise SystemExit("%s: provenance manifest_is_committed %r (data sha256 %s, canonical %s); a final seed trains "
                         "on the committed data/splits.json (05-REVIEW WR-01)"
                         % (name, prov.get("manifest_is_committed"), prov.get("data_sha256"), canon.get("data_sha256")))
    if (tr.get("postfit") or {}).get("sha256") != sha or canon.get("postfit_sha256") != sha:
        raise SystemExit("%s is bound to postfit record %s, not %s; train it with --postfit of this record"
                         % (name, (tr.get("postfit") or {}).get("sha256"), sha))
    if canon.get("split") != record["final"]["split"]:
        raise SystemExit("%s trained on %s; the final seeds train on %s" % (name, canon.get("split"),
                                                                            record["final"]["split"]))
    check_final(record, canon)
    if tr.get("epoch") != chosen["epoch"]:
        raise SystemExit("%s stopped at epoch %r; the chosen epoch is %r" % (name, tr.get("epoch"), chosen["epoch"]))
    temp = cfg.get("temperature")
    if not (isinstance(temp, list) and len(temp) == 3 and _json(temp[2]) == _json(chosen["temperature"])
            and cfg.get("temperature_by_options") == {} and tr.get("calibrated") is True):
        raise SystemExit("%s serves temperature %r (by options %r, calibrated %r); a final seed serves the record's T "
                         "%r" % (name, temp, cfg.get("temperature_by_options"), tr.get("calibrated"),
                                 chosen["temperature"]))
    rows = rec.get("epochs") or []
    return {"seed": canon["seed"], "train_run_id": rt.get("run_id"), "checkpoint_name": name,
            "checkpoint_identity": ident, "dump_run_id": dumps.run_id("laya", ident), "rung": rec.get("rung"),
            "fallback": rec.get("fallback"), "weaker": rec.get("weaker"),
            "code_changed_on_resume": rec.get("code_changed_on_resume"), "temperature": temp[2],
            "loss_by_epoch": [row.get("loss") for row in rows],
            "epochs": [{k: row.get(k) for k in ("epoch", "loss", "train_s", "peak_total_bytes", "swap_delta_bytes")}
                       for row in rows],
            "config_sha256_sans_seed": dumps.training_digest(canon, tr.get("epoch"))}


def final_manifest(record_path, ckpt_dirs):
    """The FINAL_SCHEMA manifest of the final checkpoints of one postfit record: refused unless every dir is bound to
    the record's sha256 (config and canonical), trained under the committed data/splits.json (its provenance
    manifest_is_committed True, its provenance data sha256 the canonical one), on its final split with the chosen
    recipe (check_final), stops
    at the chosen epoch, serves the chosen T exactly with no temperature_by_options, the seeds are exactly the record's
    final seeds (no duplicate, none missing) and every seed shares one config_sha256_sans_seed. Seeds sorted; ids,
    digests and numbers only (checkpoint_name is a base name; no timestamp, no local path)."""
    record, sha = read_record(record_path)
    entries = [_final_entry(record, sha, d) for d in ckpt_dirs]
    seeds = sorted(e["seed"] for e in entries)
    want = sorted(record["final"].get("seeds") or [])
    if seeds != want:
        raise SystemExit("the checkpoints are seeds %s; the record's final seeds are %s (each exactly once)"
                         % (seeds, want))
    digests = sorted({e["config_sha256_sans_seed"] for e in entries})
    if len(digests) != 1:
        raise SystemExit("the final seeds differ in more than the seed (config_sha256_sans_seed %s); retrain them with "
                         "postfit.py --print-final-commands" % digests)
    chosen = record["chosen"]
    return {"schema": FINAL_SCHEMA, "postfit": {"path": repo_relpath(record_path), "sha256": sha},
            "final_split": record["final"]["split"],
            "chosen": {k: chosen.get(k) for k in FINAL_CHOSEN_KEYS},
            "config_sha256_sans_seed": digests[0],
            "seeds": [{k: v for k, v in e.items() if k != "config_sha256_sans_seed"}
                      for e in sorted(entries, key=lambda e: e["seed"])]}


def _md(header, rows):
    """A markdown table in report.py's _md layout (pilot modules never import the CLI scripts)."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _f1(c):
    """Val micro-F1 at the global θ with its exact fraction 2TP / (2TP + FP + FN) of the summed counts, not reduced
    (05-REVIEW IN-02: the counts stay readable), e.g. "0.8107 (852/1051)"."""
    o = c["objective"]
    return "%.4f (%d/%d)" % (c["val_micro_f1_at_global_theta"], 2 * o["tp"], 2 * o["tp"] + o["fp"] + o["fn"])


def theta_edges(cands, name):
    """(candidates on the upper θ-grid edge, on the lower edge) of variant `name` (shipped or raw): those whose global
    θ equals their own grid's stop or start. A clipped θ means the candidate is scored at the edge, not at its best θ
    (05-REVIEW CR-01, WR-04)."""
    up = [c for c in cands if c["theta"][name]["global"] == c["theta"][name]["grid"]["stop"]]
    lo = [c for c in cands if c["theta"][name]["global"] == c["theta"][name]["grid"]["start"]]
    return up, lo


def per_tag_edge_line(cands, th):
    """The README line of per-tag θ on a grid edge (06-REVIEW IN-02): the chosen candidate's (`th`, its theta block)
    by tag, and the count over every candidate, shipped and raw. A per-tag θ on an edge is clipped there for its tag;
    per-tag θ serve the val-selected rows and the final seeds, while the selection objective uses the global θ only."""
    def on_edge(t):
        return sorted(k for k, v in (t.get("per_tag") or {}).items() if v in (t["grid"]["start"], t["grid"]["stop"]))
    names = ("shipped", "raw")
    mine = ["%s %s" % (n, ", ".join("%s %.2f" % (k, th[n]["per_tag"][k]) for k in on_edge(th[n])) or "none")
            for n in names]
    every = ["%d %s per-tag θ (%d candidates)" % (sum(len(on_edge(c["theta"][n])) for c in cands), n,
                                         sum(bool(on_edge(c["theta"][n])) for c in cands)) for n in names]
    return ("- Per-tag θ grid edge: the chosen candidate's per-tag θ on an edge of the grid %s..%s: %s (of %d / %d tags "
            "with their own θ); over all %d candidates, %s sit on an edge. A per-tag θ on an edge is clipped there for "
            "its tag; per-tag θ serve the val-selected rows and the final seeds, and the selection objective uses the "
            "global θ only." % (th["shipped"]["grid"]["start"], th["shipped"]["grid"]["stop"], "; ".join(mine),
                                len(th["shipped"]["per_tag"]), len(th["raw"]["per_tag"]), len(cands),
                                " and ".join(every)))


def render_markdown(record, final=None):
    """The README's Phase 5 block (markdown) of a Laya postfit record and, optionally, its final manifest: the chosen
    settings, the rule line, the encoder-LR line against the approved pilot.compute.MATRIX lr_grid, the grid-edge line
    against its max_epochs, the θ-grid and T-bound positions, the θ-edge line (how many candidates' global θ is clipped
    at a θ-grid edge, theta_edges; while any shipped θ is, the epoch is never called a plain interior optimum), the
    per-tag θ edge line (per_tag_edge_line), the candidates (chosen and runner-up marked) and the final seeds. Display rounding only; every number comes from the
    record; no timestamp, no local path, no test number. SystemExit on a record of another system kind (BERTurk has its
    own renderer) or a manifest whose chosen block or final split is not the record's."""
    from pilot import compute                               # torch-free; lazy like the other optional imports
    if record.get("system_kind") != "laya":
        raise SystemExit("render_markdown renders postfit records of system kind 'laya' only (BERTurk has its own "
                         "renderer); this record's system kind is %r" % (record.get("system_kind"),))
    chosen, proto, rules = record["chosen"], record["protocol"], record.get("rules") or {}
    if final is not None:
        for key in FINAL_CHOSEN_KEYS:
            if _json((final.get("chosen") or {}).get(key)) != _json(chosen.get(key)):
                raise SystemExit("the final manifest's chosen %s is %r, the record's %r; it belongs to another record"
                                 % (key, (final.get("chosen") or {}).get(key), chosen.get(key)))
        if final.get("final_split") != record["final"]["split"]:
            raise SystemExit("the final manifest's final split is %r, the record's %r"
                             % (final.get("final_split"), record["final"]["split"]))
    cands = record["candidates"]
    by_id = {c["dump_run_id"]: c for c in cands}
    if chosen["dump_run_id"] not in by_id:
        raise SystemExit("the record's chosen dump %s is not among its candidates" % chosen["dump_run_id"])
    cc = by_id[chosen["dump_run_id"]]
    th = chosen["theta"]
    epoch, sched, stop = chosen["epoch"], chosen["epochs_schedule"], chosen["stop_after_epoch"]
    lr_grid = proto["lr_grid"]
    out = ["**Chosen on val** (selection run on %s, seed %s, %d val messages x %d tags; val numbers, in-sample)"
           % (proto["selection_split"], proto["selection_seed"], record["split"]["n_messages"], len(metrics.TAGS)), ""]
    out.append(_md(("setting", "value"), [
        ("train run id", chosen["train_run_id"]),
        ("encoder LR", "%g (head LR %g)" % (chosen["encoder_lr"], proto["head_lr"])),
        ("epoch", "%d of %d, %s" % (epoch, sched, "--stop-after-epoch omitted (last epoch)" if stop is None
                                    else "--stop-after-epoch %d" % stop)),
        ("val-fitted T", "%.4f (bounds [%s, %s])" % (chosen["temperature"], T_MIN, T_MAX)),
        ("val NLL at T / at T = 1", "%.5f / %.5f" % (cc["nll_fitted"], cc["nll_raw"])),
        ("global θ shipped / raw", "%.2f / %.2f" % (th["shipped"]["global"], th["raw"]["global"])),
        ("tags with their own θ (shipped / raw)", "%d / %d" % (len(th["shipped"]["per_tag"]), len(th["raw"]["per_tag"]))),
        ("val micro-F1 at the global θ", _f1(cc)),
        ("val micro-F1 at θ 0.5", "%.4f" % cc["val_micro_f1_at_0_5"])]))
    out += ["", "- Rule: %s; tie order: %s." % (rules.get("objective"), rules.get("tie_order"))]
    approved = compute.MATRIX["lr_grid"]
    if len(lr_grid) == 1:
        out.append("- Encoder LR grid: %g (one value, fixed, not tuned; the approved budget allows lr_grid %d)."
                   % (lr_grid[0], approved))
    else:
        out.append("- Encoder LR grid: %s (%d values, val-selected; the approved budget allows lr_grid %d)."
                   % (", ".join("%g" % x for x in lr_grid), len(lr_grid), approved))
    edge = ("epoch %d of the epoch grid 1..%d (approved max_epochs %d)"
            % (epoch, proto["epochs"], compute.MATRIX["max_epochs"]))
    edges = {name: theta_edges(cands, name) for name in ("shipped", "raw")}
    clipped = edges["shipped"][0] + edges["shipped"][1]
    if epoch == proto["epochs"]:
        out.append("- Grid edge: upper edge: %s; the objective may still rise beyond it, and extending the grid would "
                   "be a protocol change after a val outcome, decided only by the user." % edge)
    elif epoch == 1:
        out.append("- Grid edge: lower edge: %s." % edge)
    elif clipped:
        out.append("- Grid edge: interior in the epoch grid: %s; θ clipped at the grid edge (see the θ-edge line), so "
                   "the epoch is not a clean optimum of the objective." % edge)
    else:
        out.append("- Grid edge: interior optimum: %s." % edge)
    pos = []
    for name in ("shipped", "raw"):
        g, grid = th[name]["global"], th[name]["grid"]
        where = ("at the upper edge" if g == grid["stop"] else "at the lower edge" if g == grid["start"]
                 else "inside")
        pos.append("%s %.2f (%s)" % (name, g, where))
    temps = [c["temperature"] for c in cands]
    on_bound = sum(t in (T_MIN, T_MAX) for t in temps)
    out.append("- Bounds: on the θ grid %s..%s the chosen global θ is %s; the candidates' fitted T span %.4f..%.4f "
               "in laya's band [%s, %s] (%d of %d on a bound), and the chosen T %.4f is %.4f below the upper bound %s."
               % (th["shipped"]["grid"]["start"], th["shipped"]["grid"]["stop"], " and ".join(pos), min(temps),
                  max(temps), T_MIN, T_MAX, on_bound, len(temps), chosen["temperature"],
                  T_MAX - chosen["temperature"], T_MAX))
    (s_up, s_lo), (r_up, r_lo) = edges["shipped"], edges["raw"]
    n, grid = len(cands), th["shipped"]["grid"]
    counts = ("%d of %d candidates choose the shipped global θ on the upper edge %.2f and %d on the lower edge %.2f "
              "(raw: %d upper, %d lower)" % (len(s_up), n, grid["stop"], len(s_lo), grid["start"], len(r_up), len(r_lo)))
    if clipped:
        cut = [c["temperature"] * float(np.log(c["theta"]["shipped"]["global"] / (1 - c["theta"]["shipped"]["global"])))
               for c in clipped]
        out.append("- θ grid edge: %s, so the epoch ranking is a ranking at a T-dependent margin cutoff (T x logit(θ): "
                   "%.2f..%.2f over the clipped candidates), not at each candidate's best θ; the objective is clipped at "
                   "the edge, and a θ outside the pre-declared grid may rank the epochs differently." % (
                       counts, min(cut), max(cut)))
    else:
        out.append("- θ grid edge: %s; no candidate's objective is clipped at a θ-grid edge." % counts)
    out.append(per_tag_edge_line(cands, th))
    ranked = sorted(cands, key=lambda c: (score(c), -c["epoch"], -c["encoder_lr"]), reverse=True)
    runner = ranked[1]["dump_run_id"] if len(ranked) > 1 else None
    rows = []
    for c in cands:
        mark = ("chosen" if c["dump_run_id"] == chosen["dump_run_id"] else
                "runner-up" if c["dump_run_id"] == runner else "")
        rows.append(("%d" % c["epoch"], "%g" % c["encoder_lr"], "%.4f" % c["temperature"], "%.5f" % c["nll_fitted"],
                     "%.5f" % c["nll_raw"], "%.4f" % c["val_micro_f1_at_0_5"], _f1(c),
                     "%.2f" % c["theta"]["shipped"]["global"], "%.2f" % c["theta"]["raw"]["global"], mark))
    out += ["", "**Candidates** (one full val dump per epoch export; the objective is micro-F1 at the global θ)", "",
            _md(("epoch", "encoder LR", "T", "NLL at T", "NLL at T = 1", "micro-F1 @0.5", "micro-F1 @global θ",
                 "θ shipped", "θ raw", "pick"), rows)]
    if final is not None:
        frows = []
        for e in final["seeds"]:
            ep = e.get("epochs") or []
            # an unmeasured value (None) is n/a, never 0 (05-REVIEW IN-01)
            peaks = [r["peak_total_bytes"] for r in ep if r.get("peak_total_bytes") is not None]
            swaps = [r["swap_delta_bytes"] for r in ep if r.get("swap_delta_bytes") is not None]
            loss = ep[-1].get("loss") if ep else None
            frows.append(("%d" % e["seed"], e["train_run_id"], "%d" % len(ep),
                          "n/a" if loss is None else "%.4f" % loss,
                          "n/a" if not peaks else "%.2f" % (max(peaks) / 2 ** 30),
                          "n/a" if not swaps else "%.0f" % (max(swaps) / 2 ** 20), str(e.get("rung")),
                          str(e.get("code_changed_on_resume"))))
        out += ["", "**Final seeds** (%s, --epochs %d stopped after epoch %d, serving T %.4f; not scored before Phase 8)"
                % (final["final_split"], chosen["epochs_schedule"], epoch, chosen["temperature"]), "",
                _md(("seed", "train run id", "epochs trained", "final-epoch loss", "peak GiB", "largest swap delta MiB",
                     "rung", "code changed on resume"), frows)]
    return "\n".join(out) + "\n"
