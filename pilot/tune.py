"""The torch-free half of the exploratory Ray Tune search (tune.py): the tuning partition, the dev objective, the
search constants and the tune/1 record.

A search is exploratory and not a decision input: protocol.json, the frozen verdict, every committed record and every
existing run id stay unchanged, no trial model is scored on val, test or HOTELS, and no trial weights are kept. The
record says so in NOTES.

Partition: a search trains on selection_train folds 0-11 (TRAIN_FOLDS) and scores a tuning-dev subset, folds 12-13
(DEV_FOLDS), both derived in memory from the verified selection_train bytes. selection_train holds no fold field, but
every uid is bank_<fold>_<index> and pilot.data.prepare writes the split in ascending folds, so partition() reads each
row's fold from its uid and refuses any row that does not fit (another uid shape, a fold outside selection_train, a
duplicate uid, folds out of order, an empty side). Both sides are digested with the manifest's uids_sha256 formula (the
sha256 of the newline-joined uids, UTF-8), so data/ is never written and a reader can recompute the digests from the
committed split. The driver binds the partition block; a trial's worker re-derives it from the bytes it reads and
refuses another (split_rows), before any message becomes a training item.

Objective (OBJECTIVE): after every epoch a trial scores tuning-dev in-process at fp32, probabilities at T = 1, and
reports dev_micro_f1, the micro-F1 at the global θ pilot.metrics.select_theta picks over its grid on those same rows,
with that θ, the macro-F1 and the summed TP, FP and FN (dev_objective). The record keeps the integer counts and
recomputes every F1 from them exactly.

Record (tune/1): the canonical part (search seed, space, scheduler, objective, the partition digests and counts, the
fixed recipe, the base digests, versions, code state, NOTES and per trial its config, attempts and per-epoch curve of
CURVE_KEYS) and the derived part build_record recomputes from it (each curve row's dev_micro_f1, each trial's best
epoch, stop epoch and whether ASHA stopped it, the best trial overall and the number ASHA stopped). check_record and
read_record refuse a record that does not rebuild byte for byte from its own canonical part. A record holds no path,
hostname, timestamp, uid or message text.

Render (render_markdown): the deterministic markdown of checked records, results/tune/tune.md: the exploratory
paragraph (RENDER_INTRO), then per record its settings, its highest tuning-dev score and a table of every trial, then
NOTES. It holds no path, timestamp or uid, and never calls a configuration a decision input.

usage: from pilot import tune as pt
       train_rows, dev_rows, block = pt.partition(rows); pt.split_rows(rows, block)
       pt.dev_objective(pt.labels(dev_rows), margins)
       record = pt.build_record(canonical); pt.write_record(record, "results/tune/laya.json")
       record, sha256 = pt.read_record("results/tune/laya.json")
       text = pt.render_markdown([laya_record, berturk_record])   # results/tune/tune.md (tune.py --render)
"""
import copy
import json
import math
import os
import re
import time
from fractions import Fraction
from typing import Any

import numpy as np

from intents import INTENTS
from pilot import metrics
from pilot.data import SPLITS, parse_jsonl, sha256_bytes, write_atomic

SCHEMA = "tune/1"
SYSTEMS = ("laya", "berturk")
SEED = 0                                                    # the search seed (BasicVariantGenerator random_state)
TRIALS = 20                                                 # the equal budget per system
SOURCE = "selection_train"
TRAIN_FOLDS = tuple(range(12))
DEV_FOLDS = (12, 13)
UID_RE = re.compile(r"^bank_(\d+)_(\d+)$")

_SOURCE_FOLDS = next((list(s["folds"]) for s in SPLITS if s["name"] == SOURCE), None)
if list(TRAIN_FOLDS + DEV_FOLDS) != _SOURCE_FOLDS:          # a raise, not an assert: it holds under python -O too
    raise RuntimeError("pilot.tune: TRAIN_FOLDS + DEV_FOLDS %r are not the %s folds %r of pilot.data.SPLITS"
                       % (TRAIN_FOLDS + DEV_FOLDS, SOURCE, _SOURCE_FOLDS))

SPACES = {"laya": {"encoder_lr": ["loguniform", 5e-06, 0.0001], "head_lr": ["loguniform", 3e-05, 0.001],
                   "neg": ["choice", [4, 8, 16]], "epochs": ["choice", [4, 6, 8]]},
          "berturk": {"encoder_lr": ["loguniform", 5e-06, 0.0001], "head_lr": ["loguniform", 3e-05, 0.001],
                      "batch": ["choice", [16, 32]], "warmup_ratio": ["choice", [0.06, 0.1]],
                      "weight_decay": ["choice", [0.0, 0.01]], "epochs": ["choice", [10, 20, 30]]}}
FIXED = {"laya": {"workers": 1, "micro_batch": 8, "grad_accum": 4, "max_len": 128, "freeze_encoder": False,
                  "freeze_embeddings": False, "grad_checkpointing": True, "loss": "rlcd", "seed": 0, "neg_seed": 0,
                  "precision": "fp32"},
         "berturk": {"max_len": 128, "seed": 0, "loss": "masked_bce", "optim": "adamw_torch", "lr_scheduler": "linear",
                     "max_grad_norm": 1.0, "precision": "fp32"}}
METRIC = "dev_micro_f1"
MODE = "max"
ASHA = {"name": "ASHAScheduler", "time_attr": "epoch", "metric": METRIC, "mode": MODE, "grace_period": 1,
        "reduction_factor": 3}
CURVE_KEYS = ("epoch", "dev_tp", "dev_fp", "dev_fn", "dev_theta", "dev_macro_f1", "loss", "train_s", "dev_s",
              "peak_total_bytes", "swap_delta_bytes")
OBJECTIVE = ("dev_micro_f1: the micro-F1 over the tuning-dev rows (selection_train folds 12-13) at the global θ "
             "pilot.metrics.select_theta picks on those same rows, with probabilities at T = 1 from the in-memory fp32 "
             "model scored in-process after every epoch; a trial's score is its best epoch")
NOTES = (
    "This search is exploratory and not a decision input: protocol.json, the frozen verdict, every committed record "
    "and every existing run id are unchanged by it.",
    "No trial model is scored on val, test or HOTELS, and no trial weights are kept.",
    "Tuning-train is selection_train folds 0-11 and tuning-dev is selection_train folds 12-13, split in memory by the "
    "fold in each uid; no trial reads val (folds 14-15), test or HOTELS.",
    "θ is chosen on the same tuning-dev rows it scores (in-sample) at T = 1, so dev micro-F1 is optimistic and not "
    "comparable to the val or test numbers.",
    "Dev scoring uses the in-memory fp32 weights (Laya: the training-side noul route, equal to laya.load's; BERTurk: "
    "the classifier logits), not fp16 exports.",
    "ASHA compares trials at the same epoch across schedule lengths (Laya 4, 6 or 8 epochs; BERTurk 10, 20 or 30), "
    "which favours short schedules at the rungs.",
    "A trial with attempts above 1 was restarted after a kill and may have passed a rung it had been judged at, and "
    "Ray Tune snapshots the experiment at most every 10 s, so a trial that ended seconds before a kill can rerun.",
    "Duplicate message texts across the two partitions are kept (the duplicate audit is report-only); "
    "dev_rows_with_train_text counts the tuning-dev rows whose exact text is also a tuning-train text.",
    "peak_total_bytes and swap_delta_bytes cover the epoch's training steps only.",
    "The pipeline's single encoder LR 2e-05 stays fixed, not tuned; this search reopens it for exploration only.",
)
RENDER_TITLE = "# Exploratory hyperparameter search (Ray Tune)"
# rendered only when a trial's curve holds a null loss (a non-finite training loss)
DIVERGED_LINE = "- Diverged trials (a non-finite training loss at some epoch, recorded as null): %s"
# rendered only for a record of more than one driver start (never in NOTES, which committed records hold verbatim)
RESTORE_LINE = ("- Restored %d time(s): Ray Tune 2.59 does not save ASHA's rung table with the experiment, so after "
                "each restore ASHA judged a trial at a rung only against the results reported since that restore; its "
                "stopping decisions after a restore are not the ones an uninterrupted search would have made")
RENDER_NAMES = {"laya": "Laya", "berturk": "BERTurk"}
RENDER_INTRO = (
    "This search is exploratory and not a decision input: it does not change the frozen verdict, protocol.json or any "
    "committed record. Every trial trains on tuning-train, %s folds %d-%d, and is scored on tuning-dev, %s folds "
    "%d-%d, never val, test or HOTELS. It is a seeded random search (Ray Tune's BasicVariantGenerator, one trial at a "
    "time; each system's seed and trial count are given below) with %s early stopping (time attribute %s, grace period "
    "%d, reduction factor %d, max_t the longest epoch schedule). Objective: %s."
    % (SOURCE, TRAIN_FOLDS[0], TRAIN_FOLDS[-1], SOURCE, DEV_FOLDS[0], DEV_FOLDS[-1], ASHA["name"], ASHA["time_attr"],
       ASHA["grace_period"], ASHA["reduction_factor"], OBJECTIVE))
ATTEMPTS_FILE = "attempts.json"
DISK_FLOOR_BYTES = 2 * 2**30                                # trials write no checkpoints: small files only

# the tune/1 layout: canonical keys in order (build_record refuses any other set or order)
TOP_KEYS = ("schema", "system", "search", "data", "fixed", "base", "device", "cpus", "versions", "segments", "notes",
            "trials")
SEARCH_KEYS = ("seed", "n_trials", "search_alg", "max_concurrent", "points_to_evaluate", "space", "scheduler",
               "objective")
DATA_KEYS = ("split", "sha256", "manifest_is_committed", "train_folds", "dev_folds", "train_uids_sha256",
             "dev_uids_sha256", "n_train", "n_dev", "dev_rows_with_train_text")
BASE_KEYS = ("name", "model_sha256", "config_sha256")
SEGMENT_KEYS = ("git_sha", "git_dirty", "git_diff_sha256", "preflight")
TRIAL_KEYS = ("index", "config", "attempts", "curve")
DEVICES = ("mps", "cpu")
# derived by build_record, never trusted from input
DERIVED_TOP = ("best", "n_asha_stopped")
DERIVED_TRIAL = ("best_epoch", "best_dev_micro_f1", "stop_epoch", "asha_stopped")
DERIVED_ROW = "dev_micro_f1"
BLOCK_KEYS = ("source", "train_folds", "dev_folds", "train_uids_sha256", "dev_uids_sha256")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TRIAL_ID_RE = re.compile(r"^[0-9a-f]+_(\d{5})$")
_THETAS = frozenset(float(g) for g in metrics.GRID)


# ---- the tuning partition ----

def fold_of(uid: str) -> int:
    """The fold of a selection_train uid bank_<fold>_<index>; SystemExit for any other shape."""
    m = UID_RE.fullmatch(uid) if isinstance(uid, str) else None
    if m is None:
        raise SystemExit("uid %r is not bank_<fold>_<index>; the tuning partition reads each row's fold from its uid; "
                         "regenerate the split with `make data`" % (uid,))
    return int(m.group(1))


def _uids_sha256(rows: list[dict[str, Any]]) -> str:
    """The manifest's uids_sha256 formula: sha256 of the newline-joined uids, UTF-8."""
    return sha256_bytes("\n".join(r["uid"] for r in rows).encode("utf-8"))


def partition(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(tuning-train rows, tuning-dev rows, block) of the selection_train rows, both sides in file order; block holds
    exactly source, train_folds, dev_folds, train_uids_sha256 and dev_uids_sha256.

    SystemExit, naming the row index, for a uid that is not bank_<fold>_<index>, a fold outside selection_train's, a
    duplicate uid or folds that decrease in file order (pilot.data.prepare writes ascending folds), and for an empty
    side."""
    train, dev, seen, last = [], [], set(), -1
    for i, r in enumerate(rows):
        uid = r.get("uid") if isinstance(r, dict) else None
        m = UID_RE.fullmatch(uid) if isinstance(uid, str) else None
        if m is None:
            raise SystemExit("%s row %d: uid %r is not bank_<fold>_<index>; the tuning partition reads each row's fold "
                             "from its uid; regenerate the split with `make data`" % (SOURCE, i, uid))
        fold = int(m.group(1))
        if fold not in _SOURCE_FOLDS:
            raise SystemExit("%s row %d: uid %s is in fold %d, outside %s's folds %d-%d; regenerate the split with "
                             "`make data`" % (SOURCE, i, uid, fold, SOURCE, _SOURCE_FOLDS[0], _SOURCE_FOLDS[-1]))
        if uid in seen:
            raise SystemExit("%s row %d: duplicate uid %s; regenerate the split with `make data`" % (SOURCE, i, uid))
        if fold < last:
            raise SystemExit("%s row %d: fold %d follows fold %d, but the split is written in ascending folds; "
                             "regenerate the split with `make data`" % (SOURCE, i, fold, last))
        seen.add(uid)
        last = fold
        (dev if fold in DEV_FOLDS else train).append(r)
    for side, got, folds in (("tuning-train", train, TRAIN_FOLDS), ("tuning-dev", dev, DEV_FOLDS)):
        if not got:
            raise SystemExit("the %s side (folds %s) of these %d rows is empty; a search needs both sides of %s; "
                             "regenerate the split with `make data`" % (side, list(folds), len(rows), SOURCE))
    block = {"source": SOURCE, "train_folds": list(TRAIN_FOLDS), "dev_folds": list(DEV_FOLDS),
             "train_uids_sha256": _uids_sha256(train), "dev_uids_sha256": _uids_sha256(dev)}
    return train, dev, block


def split_rows(rows: list[dict[str, Any]],
               block: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(tuning-train rows, tuning-dev rows) of `rows`; SystemExit, naming the first differing key, unless their
    partition block equals `block` (the one the driver bound)."""
    train, dev, have = partition(rows)
    if not isinstance(block, dict):
        raise SystemExit("the search's tuning partition is %r, not a block; rerun tune.py, which binds the partition "
                         "on the driver" % (block,))
    for k in list(have) + [k for k in block if k not in have]:
        if have.get(k) != block.get(k):
            raise SystemExit("the tuning partition of these rows differs from the search's (key %s); rerun tune.py, "
                             "which binds the partition on the driver" % k)
    return train, dev


def read_partition(path: str, sha256: str,
                   block: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """split_rows of the split at `path`, whose bytes (read once) must hash to `sha256` (the driver-verified digest)
    before they are parsed."""
    with open(path, "rb") as f:
        raw = f.read()
    have = sha256_bytes(raw)
    if have != sha256:
        raise SystemExit("%s sha256 %s != %s verified on the driver; regenerate with `make data`"
                         % (path, have, sha256))
    return split_rows(parse_jsonl(raw), block)


def shared_texts(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> int:
    """The number of tuning-dev rows whose exact message text also occurs in the tuning-train rows (a count, never the
    text)."""
    texts = {r["state"] for r in train_rows}
    return sum(1 for r in dev_rows if r["state"] in texts)


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    """bool [n, 48] gold of `rows` in INTENTS order from each row's expected; SystemExit naming the uid and tag of a
    row that does not label a tag with a bool."""
    tags = list(INTENTS)
    Y = np.zeros((len(rows), len(tags)), dtype=bool)
    for i, r in enumerate(rows):
        exp = r.get("expected") if isinstance(r, dict) else None
        for j, t in enumerate(tags):
            v = exp.get(t) if isinstance(exp, dict) else None
            if not isinstance(v, bool):
                raise SystemExit("row %d (uid %s) does not label tag %s; every split row labels all 48 tags; "
                                 "regenerate the split with `make data`"
                                 % (i, r.get("uid") if isinstance(r, dict) else None, t))
            Y[i, j] = v
    return Y


def dev_objective(Y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """The per-epoch dev score: probabilities at T = 1, the global θ of pilot.metrics.select_theta over its grid on
    these rows, and at that θ {dev_micro_f1, dev_theta, dev_macro_f1, dev_tp, dev_fp, dev_fn} (the counts summed over
    the 48 tags, so the micro-F1 is 2TP / (2TP + FP + FN) exactly)."""
    P = metrics.probabilities(logits, 1.0)
    theta = metrics.select_theta(Y, P)["global"]
    m = metrics.metrics(Y, P, theta)
    per = m["per_tag"].values()
    return {"dev_micro_f1": m["micro_f1"], "dev_theta": float(theta), "dev_macro_f1": m["macro_f1"],
            "dev_tp": int(sum(t["tp"] for t in per)), "dev_fp": int(sum(t["fp"] for t in per)),
            "dev_fn": int(sum(t["fn"] for t in per))}


# ---- search constants ----

def _space(system: str) -> dict[str, list[Any]]:
    if system not in SPACES:
        raise SystemExit("system %r is not one of %s" % (system, ", ".join(SYSTEMS)))
    return SPACES[system]


def max_t(system: str) -> int:
    """ASHA's max_t: the longest epoch schedule of the system's space (8 Laya, 30 BERTurk)."""
    return max(_space(system)["epochs"][1])


def rungs(system: str) -> list[int]:
    """ASHA's rung epochs below max_t: grace_period * reduction_factor ** k ([1, 3] Laya, [1, 3, 9, 27] BERTurk)."""
    out, t = [], ASHA["grace_period"]
    while t < max_t(system):
        out.append(t)
        t *= ASHA["reduction_factor"]
    return out


def check_point(system: str, point: Any) -> dict[str, Any]:
    """`point` in SPACES[system] key order; SystemExit naming the key unless it is a dict with exactly the space's
    keys, every loguniform value a float in [low, high] and every choice value one of its list (of the same type)."""
    space = _space(system)
    if not isinstance(point, dict):
        raise SystemExit("a %s search point must be a dict of %s, got %r" % (system, ", ".join(space), point))
    missing = [k for k in space if k not in point]
    extra = sorted(str(k) for k in point if k not in space)
    if missing or extra:
        raise SystemExit("a %s search point needs exactly the keys %s (missing %s; unexpected %s)"
                         % (system, ", ".join(space), ", ".join(missing) or "none", ", ".join(extra) or "none"))
    for k, (kind, *spec) in space.items():
        v = point[k]
        if kind == "loguniform":
            low, high = spec
            if type(v) is not float or not math.isfinite(v) or not low <= v <= high:
                raise SystemExit("search point %s %r is not a float in [%r, %r]" % (k, v, low, high))
        elif not any(type(v) is type(c) and v == c for c in spec[0]):
            raise SystemExit("search point %s %r is not one of %r" % (k, v, spec[0]))
    return {k: point[k] for k in space}


# ---- trials ----

def trial_index(trial_id: str) -> int:
    """The creation index of a Tune trial id (<hex experiment prefix>_<5 digits>); SystemExit for another shape."""
    m = _TRIAL_ID_RE.fullmatch(trial_id) if isinstance(trial_id, str) else None
    if m is None:
        raise SystemExit("Tune trial id %r is not <hex>_<5 digits>; this tune.py expects the Ray 2.59 trial ids"
                         % (trial_id,))
    return int(m.group(1))


def trial_dirname_of(index: int) -> str:
    """The trial dir name of trial `index`: trial_NNNNN (no config values, no date)."""
    return "trial_%05d" % index


def _attempts(trial_dir: str) -> dict[str, Any] | None:
    """The parsed attempts file of trial_dir, None when it is missing; SystemExit when it is invalid."""
    path = os.path.join(trial_dir, ATTEMPTS_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit("%s is not readable JSON (%s); delete the search's storage dir to start it again"
                         % (path, e)) from None
    n = doc.get("attempts") if isinstance(doc, dict) else None
    started = doc.get("started") if isinstance(doc, dict) else None
    if type(n) is not int or n < 1 or not isinstance(started, list) or len(started) != n:
        raise SystemExit("%s holds no valid attempt count; delete the search's storage dir to start it again" % path)
    return doc


def bump_attempts(trial_dir: str) -> int:
    """Count one more start of the trial in trial_dir/attempts.json (written atomically, with every start time) and
    return the new count. The start times stay in the gitignored storage and never enter a record."""
    doc = _attempts(trial_dir) or {"attempts": 0, "started": []}
    n = doc["attempts"] + 1
    os.makedirs(trial_dir, exist_ok=True)
    write_atomic(os.path.join(trial_dir, ATTEMPTS_FILE),
                 (json.dumps({"attempts": n, "started": doc["started"] + [time.time()]}) + "\n").encode("utf-8"))
    return n


def read_attempts(trial_dir: str) -> int:
    """The number of starts trial_dir/attempts.json counts; SystemExit when the file is missing or invalid."""
    doc = _attempts(trial_dir)
    if doc is None:
        raise SystemExit("%s has no %s: the trial never started under this tune.py; rerun the same command"
                         % (trial_dir, ATTEMPTS_FILE))
    return doc["attempts"]


def _num(key: str, v: Any, kind: type, low: float | None = None) -> None:
    """SystemExit naming `key` unless v is a finite `kind` (never a bool) at least `low`."""
    ok = type(v) is kind and (kind is not float or math.isfinite(v)) and (low is None or v >= low)
    if not ok:
        raise SystemExit("a trial result's %s is %r, expected a finite %s%s" % (
            key, v, kind.__name__, "" if low is None else " at least %s" % low))


def _f1(tp: int, fp: int, fn: int) -> float:
    """2TP / (2TP + FP + FN), 0.0 when TP is 0 (pilot.metrics._prf's F1)."""
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def curve_row(result: dict[str, Any]) -> dict[str, Any]:
    """Exactly CURVE_KEYS of one Tune result dict (a trial's per-epoch report), type-checked; SystemExit naming the
    key. Tune's own fields (hostname, pid, timestamps, trial id) never pass. loss is a finite float, or None for a
    diverged epoch (a non-finite training loss: the trials report None, never NaN). A result that also carries
    dev_micro_f1 must hold the F1 its counts give."""
    if not isinstance(result, dict):
        raise SystemExit("a trial result is %r, not a dict" % (result,))
    missing = [k for k in CURVE_KEYS if k not in result]
    if missing:
        raise SystemExit("a trial result misses %s; was it written by this tune.py?" % ", ".join(missing))
    _num("epoch", result["epoch"], int, 1)
    for k in ("dev_tp", "dev_fp", "dev_fn"):
        _num(k, result[k], int, 0)
    if type(result["dev_theta"]) is not float or result["dev_theta"] not in _THETAS:
        raise SystemExit("a trial result's dev_theta is %r, not a point of the θ grid %.2f..%.2f"
                         % (result["dev_theta"], float(metrics.GRID[0]), float(metrics.GRID[-1])))
    mf = result["dev_macro_f1"]
    if mf is not None and not (type(mf) is float and 0.0 <= mf <= 1.0):
        raise SystemExit("a trial result's dev_macro_f1 is %r, expected a float in [0, 1] or None" % (mf,))
    if result["loss"] is not None:                          # None: a diverged epoch
        _num("loss", result["loss"], float)
    for k in ("train_s", "dev_s"):
        _num(k, result[k], float)
    _num("peak_total_bytes", result["peak_total_bytes"], int, 0)
    _num("swap_delta_bytes", result["swap_delta_bytes"], int)
    if METRIC in result and result[METRIC] != _f1(result["dev_tp"], result["dev_fp"], result["dev_fn"]):
        raise SystemExit("a trial result's %s %r is not the F1 of its counts" % (METRIC, result[METRIC]))
    return {k: result[k] for k in CURVE_KEYS}


# ---- the tune/1 record ----

def _keys(where: str, d: Any, keys: tuple[str, ...]) -> None:
    """SystemExit unless `d` is a dict with exactly `keys`, in that order."""
    if not isinstance(d, dict) or tuple(d) != keys:
        raise SystemExit("%s: keys %s, expected %s in this order; rebuild the record with tune.py"
                         % (where, list(d) if isinstance(d, dict) else type(d).__name__, list(keys)))


def _want(where: str, have: Any, want: Any) -> None:
    if have != want or type(have) is not type(want):
        raise SystemExit("%s is %r, expected %r; rebuild the record with tune.py" % (where, have, want))


def _hex(where: str, v: Any) -> None:
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise SystemExit("%s is %r, not a sha256 hex digest; rebuild the record with tune.py" % (where, v))


def _int(where: str, v: Any, low: int) -> None:
    if type(v) is not int or v < low:
        raise SystemExit("%s is %r, expected an int at least %d; rebuild the record with tune.py" % (where, v, low))


def _check_canonical(c: dict[str, Any]) -> None:
    """SystemExit naming the first part of a canonical tune/1 dict that is not the layout or not consistent."""
    _keys("record", c, TOP_KEYS)
    _want("schema", c["schema"], SCHEMA)
    if c["system"] not in SYSTEMS:
        raise SystemExit("system %r is not one of %s" % (c["system"], ", ".join(SYSTEMS)))
    system = c["system"]
    s = c["search"]
    _keys("search", s, SEARCH_KEYS)
    _int("search.seed", s["seed"], 0)
    _int("search.n_trials", s["n_trials"], 1)
    _want("search.search_alg", s["search_alg"], "BasicVariantGenerator")
    _want("search.max_concurrent", s["max_concurrent"], 1)
    points = s["points_to_evaluate"]
    if points is not None:
        if not isinstance(points, list) or not 1 <= len(points) <= s["n_trials"]:
            raise SystemExit("search.points_to_evaluate must be None or a list of 1..n_trials points")
        for p in points:
            if check_point(system, p) != p or list(p) != list(SPACES[system]):
                raise SystemExit("search.points_to_evaluate holds a point out of key order: %r" % (p,))
    _want("search.space", s["space"], SPACES[system])
    _want("search.scheduler", s["scheduler"], dict(ASHA, max_t=max_t(system)))
    _want("search.objective", s["objective"], OBJECTIVE)
    d = c["data"]
    _keys("data", d, DATA_KEYS)
    _want("data.split", d["split"], SOURCE)
    for k in ("sha256", "train_uids_sha256", "dev_uids_sha256"):
        _hex("data." + k, d[k])
    if type(d["manifest_is_committed"]) is not bool:
        raise SystemExit("data.manifest_is_committed is %r, not a bool" % (d["manifest_is_committed"],))
    _want("data.train_folds", d["train_folds"], list(TRAIN_FOLDS))
    _want("data.dev_folds", d["dev_folds"], list(DEV_FOLDS))
    _int("data.n_train", d["n_train"], 1)
    _int("data.n_dev", d["n_dev"], 1)
    _int("data.dev_rows_with_train_text", d["dev_rows_with_train_text"], 0)
    if d["dev_rows_with_train_text"] > d["n_dev"]:
        raise SystemExit("data.dev_rows_with_train_text %d exceeds n_dev %d" % (d["dev_rows_with_train_text"],
                                                                                d["n_dev"]))
    _want("fixed", c["fixed"], FIXED[system])
    b = c["base"]
    _keys("base", b, BASE_KEYS)
    if not isinstance(b["name"], str) or not b["name"]:
        raise SystemExit("base.name is %r, not a name" % (b["name"],))
    _hex("base.model_sha256", b["model_sha256"])
    _hex("base.config_sha256", b["config_sha256"])
    if c["device"] not in DEVICES:
        raise SystemExit("device %r is not one of %s" % (c["device"], ", ".join(DEVICES)))
    _int("cpus", c["cpus"], 1)
    v = c["versions"]
    if not isinstance(v, dict) or not v or not all(isinstance(k, str) and isinstance(x, str) for k, x in v.items()):
        raise SystemExit("versions must map package names to version strings, got %r" % (v,))
    segs = c["segments"]
    if not isinstance(segs, list) or not segs:
        raise SystemExit("segments must be a non-empty list of driver starts")
    for i, seg in enumerate(segs):
        _keys("segments[%d]" % i, seg, SEGMENT_KEYS)
        if seg["git_sha"] is not None and not (isinstance(seg["git_sha"], str) and re.fullmatch(r"[0-9a-f]{40}",
                                                                                               seg["git_sha"])):
            raise SystemExit("segments[%d].git_sha is %r, not a git sha or None" % (i, seg["git_sha"]))
        if seg["git_dirty"] is not None and type(seg["git_dirty"]) is not bool:
            raise SystemExit("segments[%d].git_dirty is %r, not a bool or None" % (i, seg["git_dirty"]))
        if seg["git_diff_sha256"] is not None:
            _hex("segments[%d].git_diff_sha256" % i, seg["git_diff_sha256"])
        if seg["preflight"] is not None and not isinstance(seg["preflight"], dict):
            raise SystemExit("segments[%d].preflight is %r, not a dict or None" % (i, seg["preflight"]))
    _want("notes", c["notes"], list(NOTES))
    trials = c["trials"]
    if not isinstance(trials, list) or len(trials) != s["n_trials"]:
        raise SystemExit("trials holds %s entries, search.n_trials is %d"
                         % (len(trials) if isinstance(trials, list) else type(trials).__name__, s["n_trials"]))
    for i, t in enumerate(trials):
        where = "trials[%d]" % i
        _keys(where, t, TRIAL_KEYS)
        if t["index"] != i or type(t["index"]) is not int:
            raise SystemExit("%s.index is %r; trials are indexed 0..n_trials-1 in order" % (where, t["index"]))
        cfg = t["config"]
        if check_point(system, cfg) != cfg or list(cfg) != list(SPACES[system]):
            raise SystemExit("%s.config is out of the space's key order: %r" % (where, cfg))
        if points is not None and i < len(points) and cfg != points[i]:
            raise SystemExit("%s.config %r is not points_to_evaluate[%d] %r (evaluated first, in order)"
                             % (where, cfg, i, points[i]))
        _int(where + ".attempts", t["attempts"], 1)
        curve = t["curve"]
        if not isinstance(curve, list) or not curve:
            raise SystemExit("%s.curve must be a non-empty list of per-epoch rows" % where)
        for e, row in enumerate(curve, 1):
            _keys("%s.curve[%d]" % (where, e - 1), row, CURVE_KEYS)
            if curve_row(row) != row:
                raise SystemExit("%s.curve[%d] is not a valid curve row" % (where, e - 1))
            if row["epoch"] != e:
                raise SystemExit("%s.curve[%d].epoch is %r; a curve holds epochs 1..k in order"
                                 % (where, e - 1, row["epoch"]))
        if len(curve) > cfg["epochs"]:
            raise SystemExit("%s.curve holds %d epochs, longer than its schedule of %d" % (where, len(curve),
                                                                                         cfg["epochs"]))


def build_record(canonical: dict[str, Any]) -> dict[str, Any]:
    """The full tune/1 record of a canonical dict (validated: key sets and order, spaces, scheduler, schedules, curve
    lengths, NOTES): a deep copy with the derived fields recomputed, never trusted from input. Each curve row gains
    dev_micro_f1 (from its counts, right after dev_fn); each trial best_epoch, best_dev_micro_f1, stop_epoch (its last
    curve epoch) and asha_stopped (stop_epoch below its schedule); the record best {index, epoch, dev_micro_f1,
    config} (the exact-fraction argmax; ties go to the lower index, then the earlier epoch) and n_asha_stopped, right
    before trials."""
    _check_canonical(canonical)
    c = copy.deepcopy(canonical)
    best_key, best, trials = None, None, []
    for t in c["trials"]:
        curve = []
        for row in t["curve"]:
            out = {}
            for k in CURVE_KEYS:
                out[k] = row[k]
                if k == "dev_fn":
                    out[DERIVED_ROW] = _f1(row["dev_tp"], row["dev_fp"], row["dev_fn"])
            curve.append(out)
        fr = [Fraction(2 * r["dev_tp"], 2 * r["dev_tp"] + r["dev_fp"] + r["dev_fn"]) if r["dev_tp"] else Fraction(0)
              for r in t["curve"]]
        top = max(fr)
        e = fr.index(top)                                   # the earliest epoch of the trial's best
        stop = curve[-1]["epoch"]
        trials.append(dict(t, curve=curve, best_epoch=curve[e]["epoch"], best_dev_micro_f1=curve[e][DERIVED_ROW],
                           stop_epoch=stop, asha_stopped=stop < t["config"]["epochs"]))
        if best_key is None or top > best_key:              # strictly better: ties keep the lower index
            best_key = top
            best = {"index": t["index"], "epoch": curve[e]["epoch"], "dev_micro_f1": curve[e][DERIVED_ROW],
                    "config": copy.deepcopy(t["config"])}
    out = {k: c[k] for k in TOP_KEYS if k != "trials"}
    out["best"] = best
    out["n_asha_stopped"] = sum(1 for t in trials if t["asha_stopped"])
    out["trials"] = trials
    return out


def canonical_part(record: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of `record` without the fields build_record derives (best, n_asha_stopped, each trial's best_epoch,
    best_dev_micro_f1, stop_epoch and asha_stopped, each curve row's dev_micro_f1)."""
    if not isinstance(record, dict):
        raise SystemExit("a tune/1 record is a JSON object, got %s" % type(record).__name__)
    c = {k: copy.deepcopy(v) for k, v in record.items() if k not in DERIVED_TOP}
    if isinstance(c.get("trials"), list):
        trials = []
        for t in c["trials"]:
            if isinstance(t, dict):                         # key order kept: curve stays where it was
                t = {k: v for k, v in t.items() if k not in DERIVED_TRIAL}
                if isinstance(t.get("curve"), list):
                    t["curve"] = [{k: v for k, v in r.items() if k != DERIVED_ROW} if isinstance(r, dict) else r
                                  for r in t["curve"]]
            trials.append(t)
        c["trials"] = trials
    return c


def record_bytes(record: dict[str, Any]) -> bytes:
    """The record's bytes: indented JSON with a trailing newline (no NaN or infinity)."""
    return (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def check_record(record: dict[str, Any]) -> None:
    """SystemExit, naming the first top-level key that differs, unless `record` is build_record of its own canonical
    part byte for byte (an edited best, curve F1, stop epoch or key order is refused)."""
    rebuilt = build_record(canonical_part(record))
    if record_bytes(rebuilt) == record_bytes(record):
        return
    for k in list(rebuilt) + [k for k in record if k not in rebuilt]:
        if k not in record or k not in rebuilt or record_bytes(rebuilt[k]) != record_bytes(record[k]):
            raise SystemExit("the tune/1 record does not rebuild from its own canonical part (key %s differs); "
                             "rebuild it with tune.py, never by hand" % k)
    raise SystemExit("the tune/1 record does not rebuild from its own canonical part (key order differs); rebuild it "
                     "with tune.py, never by hand")


def write_record(record: dict[str, Any], path: str) -> str:
    """Write the record atomically (no timestamp: it rebuilds byte for byte); returns path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_atomic(path, record_bytes(record))
    return path


def read_record(path: str) -> tuple[dict[str, Any], str]:
    """(record, sha256 of the file bytes) of a tune/1 record; SystemExit on a missing file, invalid JSON, another
    schema, a record that does not rebuild from its canonical part, or bytes other than record_bytes(record)."""
    if not os.path.isfile(path):
        raise SystemExit("%s not found; build it with tune.py --system SYSTEM --out %s" % (path, path))
    with open(path, "rb") as f:
        raw = f.read()
    try:
        record = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); rebuild it with tune.py" % (path, e)) from None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise SystemExit("%s: schema %r, expected %s; rebuild it with this tune.py"
                         % (path, record.get("schema") if isinstance(record, dict) else None, SCHEMA))
    check_record(record)
    if record_bytes(record) != raw:
        raise SystemExit("%s is not in the record's own byte layout (indent 2, UTF-8, one trailing newline); rebuild "
                         "it with tune.py, never by hand" % path)
    return record, sha256_bytes(raw)


# ---- the rendered summary ----

def _value(system: str, key: str, v: Any) -> str:
    """A search value as rendered: a log-uniform value as %.1e, a choice as written."""
    return "%.1e" % v if SPACES[system][key][0] == "loguniform" else str(v)


def _segment(seg: dict[str, Any]) -> str:
    """One driver start: its short git sha and dirty flag, and its preflight (none off MPS)."""
    code = "no git" if seg["git_sha"] is None else "git %s, %s" % (seg["git_sha"][:8],
                                                                    "dirty" if seg["git_dirty"] else "clean")
    pf = seg["preflight"]
    if pf is None:
        return "%s; preflight none" % code
    gib = pf.get("available_gib")
    return "%s; preflight %s GiB available%s" % (code, "%.1f" % gib if isinstance(gib, (int, float)) else "?",
                                                   ", overridden (busy machine)" if pf.get("override") else "")


def render_markdown(records: list[dict[str, Any]]) -> str:
    """The markdown of checked tune/1 records, in the given order (results/tune/tune.md): RENDER_TITLE, RENDER_INTRO,
    per record a section with its settings (trials, search seed, device, CPUs, partition counts, the shared-text count,
    base, versions, driver starts with each one's short git sha and preflight, RESTORE_LINE when there was more than
    one driver start (ASHA's rung table is lost on a restore), restarted and ASHA-stopped trials, DIVERGED_LINE when a
    trial's curve holds a null loss), the
    highest tuning-dev score (trial, epoch, dev micro-F1 to 4 decimals, TP/FP/FN, config) and a table of every trial
    (its space values, log-uniform ones as %.1e, attempts, epochs run of its schedule, best epoch, best dev micro-F1,
    θ at the best epoch, ASHA stopped), then NOTES as bullets. Deterministic; no path, timestamp or uid."""
    for r in records:
        check_record(r)
    out = [RENDER_TITLE, "", RENDER_INTRO]
    for r in records:
        system, s, d, trials = r["system"], r["search"], r["data"], r["trials"]
        keys = list(SPACES[system])
        restarted = sum(1 for t in trials if t["attempts"] > 1)
        base = r["base"]["name"] if r["base"]["name"] in ("multilingual", "berturk") else "a local base"
        out += ["", "## %s" % RENDER_NAMES[system], "",
                "- Search: %d trials, search seed %d, %s, rungs at epochs %s, max_t %d; device %s, %d CPUs per trial"
                % (s["n_trials"], s["seed"], s["search_alg"], ", ".join(str(e) for e in rungs(system)),
                   s["scheduler"]["max_t"], r["device"], r["cpus"]),
                "- Data: tuning-train %d messages (folds %d-%d), tuning-dev %d messages (folds %d-%d); %d tuning-dev "
                "rows share their exact text with a tuning-train row; committed manifest: %s"
                % (d["n_train"], d["train_folds"][0], d["train_folds"][-1], d["n_dev"], d["dev_folds"][0],
                   d["dev_folds"][-1], d["dev_rows_with_train_text"], "yes" if d["manifest_is_committed"] else "no"),
                "- Base: %s (model sha256 %s)" % (base, r["base"]["model_sha256"][:12]),
                "- Versions: %s" % ", ".join("%s %s" % kv for kv in r["versions"].items()),
                "- Driver starts: %d (%s)" % (len(r["segments"]), "; ".join("%d: %s" % (i + 1, _segment(g))
                                                                            for i, g in enumerate(r["segments"])))]
        if len(r["segments"]) > 1:
            out.append(RESTORE_LINE % (len(r["segments"]) - 1))
        out += ["- Restarted trials (attempts above 1): %d of %d" % (restarted, len(trials)),
                "- ASHA-stopped trials: %d of %d" % (r["n_asha_stopped"], len(trials))]
        diverged = [str(t["index"]) for t in trials if any(row["loss"] is None for row in t["curve"])]
        if diverged:                                        # rendered only when a trial diverged
            out.append(DIVERGED_LINE % ", ".join(diverged))
        b = r["best"]
        row = trials[b["index"]]["curve"][b["epoch"] - 1]
        out += ["", "Highest tuning-dev score: trial %d, epoch %d, dev micro-F1 %.4f (TP %d, FP %d, FN %d); %s"
                % (b["index"], b["epoch"], b["dev_micro_f1"], row["dev_tp"], row["dev_fp"], row["dev_fn"],
                   ", ".join("%s %s" % (k, _value(system, k, v)) for k, v in b["config"].items())), ""]
        head = ["trial"] + keys + ["attempts", "epochs run", "best epoch", "best dev micro-F1", "θ at best",
                                   "ASHA stopped"]
        out += ["| %s |" % " | ".join(head), "|%s|" % "|".join("---" for _ in head)]
        for t in trials:
            cells = ([str(t["index"])] + [_value(system, k, t["config"][k]) for k in keys]
                     + [str(t["attempts"]), "%d of %d" % (t["stop_epoch"], t["config"]["epochs"]),
                        str(t["best_epoch"]), "%.4f" % t["best_dev_micro_f1"],
                        "%.2f" % t["curve"][t["best_epoch"] - 1]["dev_theta"], "yes" if t["asha_stopped"] else "no"])
            out.append("| %s |" % " | ".join(cells))
    out += ["", "## Notes", ""] + ["- %s" % n for n in NOTES]
    return "\n".join(out) + "\n"
