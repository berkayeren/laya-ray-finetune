"""Exploratory Ray Tune hyperparameter search of a fine-tuned system: a seeded random search with ASHA early stopping,
one trial at a time, scored on a tuning-dev subset of selection_train, written to a tune/1 record.

The search is exploratory and not a decision input: protocol.json, the frozen verdict, every committed record and every
existing run id stay unchanged, no trial model is scored on val, test or HOTELS, and no trial weights are kept
(pilot.tune.NOTES, carried by the record). The torch-free half lives in pilot/tune.py: the partition, the objective,
the search constants and the record.

Search: tune.search.BasicVariantGenerator(random_state=--seed, max_concurrent=1) samples --trials configs of
pilot.tune.SPACES[system] (the hidden --points pins the first ones, tests only), and ASHAScheduler(time_attr epoch,
grace period 1, reduction factor 3, max_t the longest epoch schedule) stops the weak ones, on TuneConfig's metric
dev_micro_f1, mode max. The one-at-a-time limit lives on the generator: TuneConfig(max_concurrent_trials) is ignored
for a SearchGenerator such as BasicVariantGenerator, so it is never passed. The generator limits only the trials it
creates, so every trial also holds the one unit of a custom Ray resource (TRIAL_SLOT, ray.init(resources)), which keeps
the trials a restore re-queues one at a time too.

Data: the driver verifies --data (a registered train split, selection_train only), refuses committed held-out content
in it (pilot.data.refuse_held_out, on the driver), and binds the tuning partition (pilot.tune.partition: folds 0-11
train, folds 12-13 dev, by the fold in each uid). A trial receives the verified selection_train path and sha256 and
the partition block only; its worker re-derives the partition from the bytes it reads and refuses another, before
negative sampling.

Laya trials (laya_trial): each trial is a Tune function that builds ray_train.py's own train config (the one builder
ray_train.train_config, plus the partition block as the key tune) and runs its own one-worker TorchTrainer of
ray_train.train_loop_per_worker with TuneReportCallback(), which forwards rank 0's per-epoch metrics to Tune. The worker
trains on tuning-train, scores tuning-dev on rank 0 at fp32 after every epoch, reports metrics only and writes no
checkpoint, export or final save. Each attempt of a trial runs under a fresh inner run name
(<storage>/laya-train/trial_NNNNN-aK), so Ray Train never resumes a killed attempt, and its --out path is never written.
The fixed recipe is pilot.tune.FIXED["laya"]: one worker, the MPS spike's start config (micro-batch 8, grad-accum 4,
max_len 128, full encoder with gradient checkpointing), rlcd, seed 0, fp32.

BERTurk trials (berturk_trial): each trial is one process with no Ray Train (BERTurk trains with the HF Trainer in one
process). It parses berturk_train.py's own argv for the trial's point (encoder LR, head LR, batch, warm-up ratio,
weight decay, epochs) and calls berturk_train.fit, the recipe berturk_train.py runs, with the partition block as the
identity key tune, no export and no --out. fit's on_epoch hook scores tuning-dev after every epoch with the live model
through score.bert_logits (score.py's own batched forward: eval mode, no_grad, fp32, logits at T = 1), puts the model
back in train mode and calls tune.report. Each attempt trains under a fresh run storage
(<storage>/berturk-train/trial_NNNNN-aK). The trial reserves --cpus CPUs and ray.init gets --cpus + 1, so trials run
one at a time; the fixed recipe is pilot.tune.FIXED["berturk"]: max_len 128, seed 0, the masked BCE, adamw_torch, the
linear schedule, max grad norm 1.0, fp32. --base defaults to the system's pinned weights (multilingual, berturk).

Ray 2.59 behaviour this design handles:
- concurrency: max_concurrent=1 on the generator and the one TRIAL_SLOT (above): a restore puts its unfinished and
  errored trials straight into the controller, which would start them together without the slot;
- leaked placement groups: an ASHA-stopped Laya trial leaves its Train placement group behind (Ray's cleaner needs the
  dashboard's State API, which the lock does not install), so ray.init gets logical-CPU headroom, 1 + cpus x (trials +
  1), and the inner trainer's FailureConfig(max_failures=0, controller_failure_limit=0) turns a starved start into a
  trial error instead of an endless retry;
- a restore can let a restarted trial pass an ASHA rung it had been judged at, so every trial counts its starts in
  attempts.json in its storage trial dir (pilot.tune.bump_attempts, trial_store_dir; never Tune's get_trial_dir(), a
  session working dir synced only at a report) and the record carries the count (never repaired);
- ASHA's rung table is not saved with the experiment (Tune's controller leaves its scheduler out of its state), so after
  a restore ASHA judges trials only against the results reported since; the rendered summary of a record of more than
  one driver start says so (pilot.tune.RESTORE_LINE);
- a function trial runs with its cwd under Ray's session dir, so every path a trial receives is absolute.

Storage (default tune_runs/, gitignored): <storage>/<system>/ is the Tune experiment (trial dirs trial_NNNNN, each with
Tune's params.json and result.json and our attempts.json), <storage>/<system>-train/ the inner Train runs and
<storage>/<system>-search.json the search file: the search spec (search_spec: seed, trials, points, space, scheduler,
device, CPUs, the data and partition digests and counts, the base, the fixed recipe, the versions; never a path), one
segment per driver start (the git code state and, on MPS, the quiet-machine preflight), finished, and the record's
sha256 once written. Once every trial ended without an error the record is built only from each trial dir's
params.json, the CURVE_KEYS of its result.json lines and its attempts count (never Tune's own fields: hostname, pid,
timestamps, trial-id prefixes) and the search file's segments, written atomically to --out, and refused over an
existing --out of other content.

Resume: rerun the identical command. A search file of another spec is refused (naming the first differing key) before
ray.init, and so is an experiment dir Ray Tune cannot restore (a kill before its first snapshot) or one without a
search file. Otherwise this start appends its segment and Tuner.restore continues the search, resuming unfinished and
errored trials (resume_unfinished, resume_errored; never restart_errored, which gives a trial a new id without its
_NNNNN index and a new trial dir): a resumed trial keeps its id and trial_NNNNN dir, holds no checkpoint, so it runs
again from epoch 1 under a fresh inner run, Ray starts its result.json afresh, and its attempts count shows the restart.
A trial that finished before the kill is not rerun once Ray's experiment snapshot holds it (Ray snapshots at most about
every 10 s, so one that ended seconds before the kill can rerun). After an errored trial the search ends non-zero,
naming it; rerunning the same command runs it again. A finished search starts no Ray: its rerun rebuilds the record
from the trial dirs and the search file, and must reproduce the sha256 it wrote.

Guards: under results/ a search writes only results/tune/<system>.json, and only as the committed search (--trials 20,
--seed 0, --device mps, the pinned --base, the committed --manifest, no --points, no test hook), refused before any
fetch or ray.init; any other run needs another --out. On MPS the quiet-machine preflight (compute_checkpoint.preflight,
--min-available-gib, default 7.0; 0 records a busy-machine run) runs before ray.init and is the segment's preflight.

Modes: --check RECORD verifies a tune/1 record (pilot.tune.read_record) and prints its sha256; --status prints a
search's progress from --storage alone (no Ray, no write); --render RECORD... prints pilot.tune.render_markdown of
the checked records, or writes it with --out (under results/ only results/tune/tune.md). make report never runs it.

usage:
  python tune.py --system laya                     # the real search: 20 trials on MPS -> results/tune/laya.json
  python tune.py --system laya --device cpu --cpus 1 --trials 3 --base DIR --data SPLIT --manifest M \\
                 --storage STORAGE --out OUT.json   # a CPU smoke search on a local base
  python tune.py --system berturk                  # the real search: 20 trials on MPS -> results/tune/berturk.json
  python tune.py --system laya --min-available-gib 0   # the real search on a busy machine (recorded as such)
  python tune.py --status --system laya            # a running search's progress from tune_runs/, without Ray
  python tune.py --check results/tune/laya.json    # verify a record: ok PATH SHA256
  python tune.py --render results/tune/laya.json results/tune/berturk.json --out results/tune/tune.md
"""
import argparse
import copy
import json
import os
import re
import signal
import time
from fractions import Fraction
from importlib.metadata import version
from typing import Any

from pilot import pins, runs, spike
from pilot import laya_adapter as la
from pilot import tune as pt
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, parse_jsonl, refuse_held_out,
                        repo_relpath, sha256_bytes, sha256_file, verify_split_file, write_atomic)
from pilot.weights import check_tokenizer_config_rewritable, checkpoint_digests, fetch, identity

DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "splits", "selection_train.jsonl")
TRIAL_DIR_RE = re.compile(r"^trial_\d{5}$")
PINNED_BASE = {"laya": "multilingual", "berturk": "berturk"}   # --base default per system: the pinned weights
SEARCH_FILE_KEYS = ("spec", "segments", "finished", "record_sha256")
MIN_AVAILABLE_GIB = 7.0                                     # the spike's and latency's quiet-machine threshold
TRIAL_SLOT = "tune_trial_slot"   # a custom Ray resource of 1 that every trial holds: one trial at a time, restores too


def parser() -> argparse.ArgumentParser:
    """The tune.py argument parser."""
    ap = argparse.ArgumentParser(description="Exploratory Ray Tune search of one system (not a decision input).")
    ap.add_argument("--system", choices=pt.SYSTEMS, default=None, help="the system to search (or, with --status, "
                                                                        "to report on)")
    ap.add_argument("--trials", type=int, default=pt.TRIALS, help="trials of the search (default %d)" % pt.TRIALS)
    ap.add_argument("--seed", type=int, default=pt.SEED, help="the search seed (default %d)" % pt.SEED)
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps", help="training device (default mps)")
    ap.add_argument("--storage", default=os.path.abspath("tune_runs"),
                    help="Ray Tune and Train storage (default tune_runs/, gitignored)")
    ap.add_argument("--out", default=None, help="the tune/1 record (default results/tune/<system>.json)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="the selection_train split (default the committed one)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify --data against (default: the committed data/splits.json)")
    ap.add_argument("--base", default=None,
                    help="laya: 'multilingual' (pinned; default) or a local Laya checkpoint dir; berturk: 'berturk' "
                         "(pinned; default) or a local BERT base dir")
    ap.add_argument("--cpus", type=int, default=None, help="CPUs and torch threads per trial (default 4 on mps, else 1)")
    ap.add_argument("--min-available-gib", type=float, default=MIN_AVAILABLE_GIB,
                    help="MPS quiet-machine preflight threshold (default %.1f); 0 records a busy-machine run"
                         % MIN_AVAILABLE_GIB)
    ap.add_argument("--check", default=None, metavar="RECORD", help="verify a tune/1 record and print its sha256")
    ap.add_argument("--status", action="store_true",
                    help="print a search's progress from --storage (with --system), without Ray")
    ap.add_argument("--render", nargs="+", default=None, metavar="RECORD",
                    help="print the markdown of checked records (with --out: write it, e.g. results/tune/tune.md)")
    ap.add_argument("--points", default=None, help=argparse.SUPPRESS)   # JSON configs evaluated first; tests only
    # test hook: trial N's first attempt SIGKILLs the driver (the resume test); never in the spec or the record
    ap.add_argument("--kill-at-trial", type=int, default=None, help=argparse.SUPPRESS)
    # test hook: trial N's first attempt fails after its first epoch report (repeatable; the errored-trial resume
    # test); never in the spec or the record
    ap.add_argument("--fail-at-trial", type=int, action="append", default=None, help=argparse.SUPPRESS)
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsed and checked args: the mode (--check, --status or a search; one at most), --out and --cpus resolved,
    --points parsed into checked search points (or None)."""
    ap = parser()
    a = ap.parse_args(argv)
    modes = [m for m, on in (("--check", a.check is not None), ("--status", a.status),
                             ("--render", a.render is not None)) if on]
    if len(modes) > 1:
        ap.error("%s exclude each other" % " and ".join(modes))
    if a.check is not None or a.render is not None:
        if a.system is not None:
            ap.error("%s reads the system from the record; drop --system" % modes[0])
        a.check = os.path.abspath(a.check) if a.check is not None else None
        a.render = [os.path.abspath(r) for r in a.render] if a.render is not None else None
        a.out = os.path.abspath(a.out) if a.out is not None else None
        return a
    if a.system is None:
        ap.error("--system is required (laya or berturk)%s" % (" with --status" if a.status else ""))
    a.storage = os.path.abspath(a.storage)
    if a.status:
        return a
    if a.cpus is None:
        a.cpus = 4 if a.device == "mps" else 1
    if a.base is None:
        a.base = PINNED_BASE[a.system]
    for flag, v in (("--trials", a.trials), ("--cpus", a.cpus)):
        if v < 1:
            ap.error("%s must be at least 1" % flag)
    if a.seed < 0:
        ap.error("--seed must be at least 0")
    if a.min_available_gib < 0:
        ap.error("--min-available-gib must be at least 0 (0 records a busy-machine run)")
    if a.kill_at_trial is not None and not 0 <= a.kill_at_trial < a.trials:
        ap.error("--kill-at-trial must be a trial index 0..%d" % (a.trials - 1))
    for n in a.fail_at_trial or ():
        if not 0 <= n < a.trials:
            ap.error("--fail-at-trial must be a trial index 0..%d" % (a.trials - 1))
    if a.points is not None:
        try:
            points = json.loads(a.points)
        except ValueError as e:
            ap.error("--points is not JSON (%s)" % e)
        if not isinstance(points, list) or not points:
            ap.error("--points must be a non-empty JSON list of search points")
        if len(points) > a.trials:
            ap.error("--points holds %d points, more than --trials %d" % (len(points), a.trials))
        a.points = [pt.check_point(a.system, p) for p in points]
    a.out = os.path.abspath(a.out or os.path.join(REPO_ROOT, "results", "tune", "%s.json" % a.system))
    a.data = os.path.abspath(a.data)
    a.manifest = os.path.abspath(a.manifest)
    return a


def within(path: str, root: str) -> bool:
    """True when `path` is `root` or lies under it: by the realpath strings, or by file identity (os.path.samestat of
    the root against every existing ancestor of the path), so a symlink or a case variant on a case-insensitive
    filesystem is caught too."""
    real, top = os.path.realpath(path), os.path.realpath(root)
    if real == top or real.startswith(top.rstrip(os.sep) + os.sep):
        return True
    try:
        st = os.stat(top)
    except OSError:
        return False
    p = real
    while True:
        try:
            if os.path.samestat(os.stat(p), st):
                return True
        except OSError:
            pass
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def same_path(path: str, target: str) -> bool:
    """True when `path` names `target`: the same base name (exact) in the same directory (by realpath or by file
    identity of both existing directories)."""
    if os.path.basename(path) != os.path.basename(target):
        return False
    d, t = os.path.realpath(os.path.dirname(path)), os.path.realpath(os.path.dirname(target))
    if d == t:
        return True
    try:
        return os.path.samestat(os.stat(d), os.stat(t))
    except OSError:
        return False


def check_results_out(a: argparse.Namespace) -> None:
    """Refuse, before bind (no fetch, no split read) and ray.init, a search --out under results/ unless it is
    results/tune/SYS.json and the run is the committed search: --trials pilot.tune.TRIALS, --seed pilot.tune.SEED,
    --device mps, the pinned --base, the committed --manifest, no --points and no test hook. Names the first
    departure."""
    results = os.path.join(REPO_ROOT, "results")
    if not within(a.out, results):
        return
    target = os.path.join(results, "tune", "%s.json" % a.system)
    if not same_path(a.out, target):
        raise SystemExit("--out %s: under results/ a %s search writes only results/tune/%s.json (the committed "
                         "search); pass another --out" % (repo_relpath(a.out), a.system, a.system))
    committed = os.path.isfile(a.manifest) and manifest_is_committed(a.manifest)
    for departs, flag in ((a.trials != pt.TRIALS, "--trials %d" % a.trials), (a.seed != pt.SEED, "--seed %d" % a.seed),
                          (a.device != "mps", "--device %s" % a.device),
                          (a.base != PINNED_BASE[a.system],
                           "another --base than the pinned %s" % PINNED_BASE[a.system]),
                          (not committed, "another --manifest than the committed data/splits.json"),
                          (a.points is not None, "--points"),
                          (a.kill_at_trial is not None, "--kill-at-trial %s" % a.kill_at_trial),
                          (a.fail_at_trial is not None, "--fail-at-trial")):
        if departs:
            raise SystemExit("--out results/tune/%s.json with %s: a record under results/tune/ is the committed search "
                             "(%d trials, seed %d, MPS, the pinned base, the committed manifest); pass another --out "
                             "for this run" % (a.system, flag, pt.TRIALS, pt.SEED))


def bind(a: argparse.Namespace) -> dict[str, Any]:
    """The picklable binding of a search, built on the driver before ray.init: the verified selection_train (path,
    sha256, manifest entry), the tuning partition block and counts, the base (bind_laya or bind_berturk), the device,
    CPUs and worker env, the storage dirs and the versions. Every path is absolute (a trial's cwd is under Ray's session
    dir). SystemExit on a split that is not selection_train, committed held-out content, a partition refusal, a base
    the system's binder refuses, MPS unavailable, or less free disk under --storage than pilot.tune.DISK_FLOOR_BYTES
    (trials write no checkpoints)."""
    if not os.path.isfile(a.data):
        raise SystemExit("%s not found; run `make data`" % a.data)
    with open(a.data, "rb") as f:
        raw = f.read()
    entry = verify_split_file(a.data, manifest_path=a.manifest, roles=("train",), data=raw)
    name = os.path.splitext(os.path.basename(entry["path"]))[0]
    if name != pt.SOURCE:
        raise SystemExit("--data is %s; a search trains and scores on %s only (folds 0-11 train, 12-13 tuning-dev)"
                         % (name, pt.SOURCE))
    rows = parse_jsonl(raw)
    refuse_held_out(a.data, entry, rows)                    # keyed on the content, under any manifest
    train_rows, dev_rows, block = pt.partition(rows)
    base = bind_laya(a) if a.system == "laya" else bind_berturk(a)
    if a.device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps, but MPS is not available here; use --device cpu, or run on Apple silicon "
                             "with the locked torch build")
    free = runs.free_bytes(a.storage)
    if free < pt.DISK_FLOOR_BYTES:
        raise SystemExit("%.1f GiB free under --storage; a search needs at least %.1f GiB (trial logs and Ray state); "
                         "free disk or pass another --storage" % (free / 2**30, pt.DISK_FLOOR_BYTES / 2**30))
    return dict({"system": a.system, "data": a.data, "data_sha256": entry["sha256"], "manifest": a.manifest,
                 "manifest_is_committed": manifest_is_committed(a.manifest), "entry": entry, "block": block,
                 "n_train": len(train_rows), "n_dev": len(dev_rows),
                 "dev_rows_with_train_text": pt.shared_texts(train_rows, dev_rows),
                 "device": a.device, "cpus": a.cpus, "env": spike.spike_env("0", a.cpus) if a.device == "mps" else {},
                 "storage": a.storage, "train_storage": os.path.join(a.storage, "%s-train" % a.system),
                 "kill_at_trial": a.kill_at_trial, "fail_at_trials": sorted(set(a.fail_at_trial or ())),
                 "driver_pid": os.getpid()}, **base)


def bind_laya(a: argparse.Namespace) -> dict[str, Any]:
    """The Laya half of bind: the base (resolved dir, identity, verified digests, its record entry) and the versions;
    SystemExit on another laya or a base that is neither the pin nor a local checkpoint."""
    la.check_version()
    if a.base == "multilingual":                            # download once on the driver, pinned + verified
        base_dir = fetch(pins.LAYA)
    elif os.path.isfile(os.path.join(a.base, "rl_agent_config.json")):
        base_dir = os.path.abspath(a.base)
    else:
        raise SystemExit("--base must be 'multilingual' (pinned) or a local Laya checkpoint dir: %s" % a.base)
    base_name = "multilingual" if a.base == "multilingual" else base_dir
    base_ident = identity(base_name)
    base_sha256 = checkpoint_digests(base_name, base_ident)   # workers verify these bytes before loading
    check_tokenizer_config_rewritable(base_dir, "base checkpoint")
    return {"base": a.base, "base_dir": base_dir, "base_name": base_name, "base_ident": base_ident,
            "base_sha256": base_sha256,
            "base_record": {"name": "multilingual" if a.base == "multilingual" else repo_relpath(base_dir),
                            "model_sha256": base_ident["model_sha256"],
                            "config_sha256": (pins.LAYA["sha256"]["rl_agent_config.json"] if a.base == "multilingual"
                                              else base_ident["config_sha256"])},
            "versions": {"ray": version("ray"), "torch": version("torch"), "laya": version("laya")}}


def bind_berturk(a: argparse.Namespace) -> dict[str, Any]:
    """The BERTurk half of bind, as berturk_train.py binds its base: 'berturk' names the pin's digests and is fetched
    (pinned, sha256-verified) on the driver; a local dir must hold pilot.berturk.BASE_FILES and is digested file by
    file (berturk_train.fit verifies those bytes right before the load); the tokenizer must be cased. Returns the base
    (the name berturk_train.py takes, the resolved dir, the digests, its record entry) and the versions."""
    from pilot import berturk
    if a.base == "berturk":
        digests = dict(pins.BERTURK["sha256"])
        base_dir = fetch(pins.BERTURK)
        base = "berturk"
    else:
        base_dir = base = os.path.abspath(a.base)
        missing = [f for f in berturk.BASE_FILES if not os.path.isfile(os.path.join(base_dir, f))]
        if missing:
            raise SystemExit("--base must be 'berturk' (pinned) or a local BERT base dir holding %s; %s lacks %s"
                             % (", ".join(berturk.BASE_FILES), a.base, ", ".join(missing)))
        digests = {f: sha256_file(os.path.join(base_dir, f)) for f in berturk.BASE_FILES}
    try:
        with open(os.path.join(base_dir, "tokenizer_config.json"), encoding="utf-8") as f:
            tcfg = json.load(f)
    except ValueError as e:
        raise SystemExit("base tokenizer_config.json is not valid JSON (%s)" % e)
    berturk.check_cased(tcfg, "base")
    return {"base": base, "base_dir": base_dir, "digests": digests,
            "base_record": {"name": "berturk" if a.base == "berturk" else repo_relpath(base_dir),
                            "model_sha256": digests["model.safetensors"], "config_sha256": digests["config.json"]},
            "versions": {"ray": version("ray"), "torch": version("torch"), "transformers": version("transformers"),
                         "accelerate": version("accelerate")}}


def param_space(system: str) -> dict[str, Any]:
    """Tune's param_space of pilot.tune.SPACES[system], in its key order: loguniform -> tune.loguniform(low, high),
    choice -> tune.choice(list)."""
    from ray import tune
    out = {}
    for k, (kind, *spec) in pt.SPACES[system].items():
        out[k] = tune.loguniform(spec[0], spec[1]) if kind == "loguniform" else tune.choice(list(spec[0]))
    return out


def trial_dirname(trial: Any) -> str:
    """Tune's trial_dirname_creator: trial_NNNNN from the trial id's creation index (no config values, no date)."""
    return pt.trial_dirname_of(pt.trial_index(trial.trial_id))


def tune_config(system: str, seed: int, trials: int, points: list[dict[str, Any]] | None) -> Any:
    """The TuneConfig of a search: metric dev_micro_f1, mode max, `trials` samples, ASHA (time_attr epoch, grace period
    1, reduction factor 3, max_t the longest schedule; its metric and mode come from TuneConfig) and a seeded
    BasicVariantGenerator holding the one-at-a-time limit (max_concurrent_trials is never passed: Tune ignores it for a
    SearchGenerator); `points` are evaluated first."""
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.basic_variant import BasicVariantGenerator
    return tune.TuneConfig(
        metric=pt.METRIC, mode=pt.MODE, num_samples=trials,
        scheduler=ASHAScheduler(time_attr=pt.ASHA["time_attr"], grace_period=pt.ASHA["grace_period"],
                                reduction_factor=pt.ASHA["reduction_factor"], max_t=pt.max_t(system)),
        search_alg=BasicVariantGenerator(random_state=seed, max_concurrent=1, points_to_evaluate=points),
        trial_dirname_creator=trial_dirname)


def trial_store_dir(bound: dict[str, Any], index: int) -> str:
    """The storage trial dir STORAGE/SYS/trial_NNNNN of trial `index`, where attempts.json lives. Never Tune's
    get_trial_dir(): that is the trial's working dir under Ray's session dir, synced to storage only at a report, so a
    count written there is lost when the driver dies before the trial's first report, and a restored trial starts in
    a new session's empty working dir."""
    return os.path.join(bound["storage"], bound["system"], pt.trial_dirname_of(index))


def kill_hook(bound: dict[str, Any], index: int, attempt: int) -> bool:
    """The hidden --kill-at-trial test hook: on trial `index`'s first attempt, SIGKILL the driver (Tune's controller
    runs there) and return True; the trial then returns without training. False otherwise."""
    if bound.get("kill_at_trial") != index or attempt != 1:
        return False
    print("trial %d attempt 1: --kill-at-trial kills the driver (pid %d)" % (index, bound["driver_pid"]), flush=True)
    os.kill(bound["driver_pid"], signal.SIGKILL)
    return True


def fail_hook(bound: dict[str, Any], index: int, attempt: int) -> bool:
    """The hidden --fail-at-trial test hook: True on the first attempt of a listed trial, which then fails after its
    first epoch report (Laya: ray_train.py's --fail-after-epoch 1; BERTurk: a raise in the epoch hook), so the trial
    ends errored with one reported epoch. False otherwise."""
    return index in bound.get("fail_at_trials", ()) and attempt == 1


def laya_trial(config: dict[str, Any], bound: dict[str, Any]) -> None:
    """One Laya trial: count the start in attempts.json, build ray_train.py's train config for the trial's point (the
    fixed recipe plus encoder LR, head LR, neg K and epochs) with the partition block as tune, and run one TorchTrainer
    of ray_train.train_loop_per_worker under a fresh inner run name, reporting through TuneReportCallback, failing
    instead of retrying (max_failures 0, controller_failure_limit 0)."""
    import ray
    import ray.train
    from ray import tune
    from ray.train.torch import TorchTrainer
    from ray.tune.integration.ray_train import TuneReportCallback

    import ray_train
    ctx = tune.get_context()
    index = pt.trial_index(ctx.get_trial_id())
    attempt = pt.bump_attempts(trial_store_dir(bound, index))
    if kill_hook(bound, index, attempt):
        return
    point = pt.check_point("laya", {k: config.get(k) for k in pt.SPACES["laya"]})
    fixed = pt.FIXED["laya"]
    name = "%s-a%d" % (pt.trial_dirname_of(index), attempt)
    train_storage = bound["train_storage"]
    argv = ["--data", bound["data"], "--manifest", bound["manifest"], "--base", bound["base"],
            "--out", os.path.join(train_storage, name, "out"),   # never written: a trial keeps no weights
            "--device", bound["device"], "--workers", str(fixed["workers"]), "--cpus-per-worker", str(bound["cpus"]),
            "--micro-batch", str(fixed["micro_batch"]), "--grad-accum", str(fixed["grad_accum"]),
            "--max-len", str(fixed["max_len"]), "--loss", fixed["loss"], "--seed", str(fixed["seed"]),
            "--neg-seed", str(fixed["neg_seed"]), "--epochs", str(point["epochs"]), "--neg", str(point["neg"]),
            "--encoder-lr", repr(point["encoder_lr"]), "--head-lr", repr(point["head_lr"]), "--no-export-epochs",
            "--storage", train_storage]
    if fail_hook(bound, index, attempt):                    # test hook; --fail-after-epoch is not in the run id
        argv += ["--fail-after-epoch", "1"]
    a = ray_train.parser().parse_args(argv)
    r = ray_train.resolve(a)
    have = {"grad_checkpointing": r["grad_checkpointing"], "neg_seed": r["neg_seed"],
            "precision": "fp16" if r["amp"] else "fp32", "freeze_encoder": a.freeze_encoder,
            "freeze_embeddings": a.freeze_embeddings}
    off = [k for k in have if have[k] != fixed[k]]
    if off:
        raise SystemExit("the trial's resolved %s differs from pilot.tune.FIXED; fix tune.py's argv" % off[0])
    canon = runs.canonical(ray_train.identity_values(a, r, bound["entry"], bound["base_name"], bound["base_ident"],
                                                     tune=bound["block"]))
    rid = runs.run_id(canon)
    cfg = ray_train.train_config(a, r, bound["entry"], bound["base_dir"], bound["base_ident"], bound["base_sha256"],
                                 bound["env"], ray.get_runtime_context().get_node_id(), rid, canon,
                                 tune=bound["block"])
    print("trial %d attempt %d: %s (run %s)" % (index, attempt, json.dumps(point), rid), flush=True)
    TorchTrainer(
        ray_train.train_loop_per_worker, train_loop_config=cfg,
        scaling_config=ray.train.ScalingConfig(num_workers=fixed["workers"], use_gpu=False,
                                               resources_per_worker={"CPU": bound["cpus"]}),
        run_config=ray.train.RunConfig(name=name, storage_path=train_storage, callbacks=[TuneReportCallback()],
                                       failure_config=ray.train.FailureConfig(max_failures=0,
                                                                              controller_failure_limit=0))).fit()


def berturk_trial(config: dict[str, Any], bound: dict[str, Any]) -> None:
    """One BERTurk trial, in this process with no Ray Train: count the start in attempts.json, read the tuning
    partition from the driver-verified selection_train bytes only, parse berturk_train.py's own argv for the trial's
    point (encoder LR, head LR, batch, warm-up ratio, weight decay, epochs; the fixed recipe otherwise; no export, no
    --out) with the partition block as tune, and run berturk_train.fit under a fresh run storage per attempt
    (<storage>/berturk-train/trial_NNNNN-aK), so a restarted trial never meets its killed run dir. After every epoch
    the hook scores tuning-dev through score.bert_logits (eval mode, no_grad, fp32 on the trial's device, logits at
    T = 1), puts the model back in train mode and reports pilot.tune.dev_objective with the epoch row to Tune."""
    from ray import tune

    import berturk_train
    import ray_train
    import score
    from pilot import berturk
    ctx = tune.get_context()
    index = pt.trial_index(ctx.get_trial_id())
    attempt = pt.bump_attempts(trial_store_dir(bound, index))
    if kill_hook(bound, index, attempt):
        return
    if bound["env"]:
        ray_train.check_env(bound["env"])                   # the spike's MPS env reached this trial's process
    point = pt.check_point("berturk", {k: config.get(k) for k in pt.SPACES["berturk"]})
    fixed = pt.FIXED["berturk"]
    train_rows, dev_rows = pt.read_partition(bound["data"], bound["data_sha256"], bound["block"])
    labels = [berturk.label_vector(r) for r in train_rows]
    Y = pt.labels(dev_rows)
    texts = [r["state"] for r in dev_rows]
    storage = os.path.join(bound["train_storage"], "%s-a%d" % (pt.trial_dirname_of(index), attempt))
    argv = ["--data", bound["data"], "--manifest", bound["manifest"], "--base", bound["base"],
            "--device", bound["device"], "--cpus", str(bound["cpus"]), "--epochs", str(point["epochs"]),
            "--lr", repr(point["encoder_lr"]), "--head-lr", repr(point["head_lr"]), "--batch", str(point["batch"]),
            "--warmup-ratio", repr(point["warmup_ratio"]), "--weight-decay", repr(point["weight_decay"]),
            "--max-len", str(fixed["max_len"]), "--seed", str(fixed["seed"]), "--no-export-epochs",
            "--storage", storage]
    a = berturk_train.parser().parse_args(argv)
    have = {"max_len": a.max_len, "seed": a.seed, "loss": berturk.LOSS, "optim": berturk.OPTIM,
            "lr_scheduler": berturk.LR_SCHEDULER, "max_grad_norm": berturk.MAX_GRAD_NORM, "precision": "fp32"}
    off = [k for k in have if have[k] != fixed[k]]
    if off:
        raise SystemExit("the trial's %s differs from pilot.tune.FIXED; fix tune.py's argv or pilot.tune.FIXED"
                         % off[0])
    canon = berturk.canonical(berturk_train.identity_values(a, bound["entry"], bound["digests"], (), None, None,
                                                            tune=bound["block"]))
    rid = berturk.run_id(canon)

    def hook(model: Any, tok: Any, epoch: int, row: dict[str, Any]) -> None:
        """Score tuning-dev with the live model (eval, then train mode again) and report the epoch to Tune."""
        model.eval()
        try:
            t = time.perf_counter()
            logits, _ = score.bert_logits(tok, model, texts, fixed["max_len"], bound["device"])
            dev_s = time.perf_counter() - t
        finally:
            model.train()
        tune.report(dict(pt.dev_objective(Y, logits), epoch=epoch, loss=float(row["loss"]),
                         train_s=float(row["train_s"]), dev_s=dev_s, peak_total_bytes=row["peak_total_bytes"],
                         swap_delta_bytes=row["swap_delta_bytes"], n_train_messages=len(train_rows),
                         n_dev_messages=len(dev_rows)))
        if epoch == 1 and fail_hook(bound, index, attempt):
            raise RuntimeError("simulated failure after epoch 1 (--fail-at-trial %d)" % index)

    print("trial %d attempt %d: %s (run %s)" % (index, attempt, json.dumps(point), rid), flush=True)
    berturk_train.fit(a, train_rows, labels, bound["base_dir"], bound["digests"], bound["entry"], canon, rid,
                      os.path.join(storage, rid), cpus=bound["cpus"], on_epoch=hook)


def collect(experiment_dir: str, n_trials: int) -> list[dict[str, Any]]:
    """The record's trials, read from each trial dir trial_00000.. in index order: the config from params.json (exactly
    the space's keys), the curve from the CURVE_KEYS of each result.json line (pilot.tune.curve_row: Tune's own
    fields never pass) and the attempts count. SystemExit unless the experiment holds exactly n_trials trial dirs."""
    found = sorted(n for n in os.listdir(experiment_dir) if TRIAL_DIR_RE.fullmatch(n)) \
        if os.path.isdir(experiment_dir) else []
    want = [pt.trial_dirname_of(i) for i in range(n_trials)]
    if found != want:
        raise SystemExit("%s holds trial dirs %s, expected %s..%s; delete it to start the search again"
                         % (experiment_dir, found[:3] + (["..."] if len(found) > 3 else []), want[0], want[-1]))
    system = os.path.basename(os.path.normpath(experiment_dir))
    out = []
    for i, name in enumerate(want):
        d = os.path.join(experiment_dir, name)
        with open(os.path.join(d, "params.json"), encoding="utf-8") as f:
            config = pt.check_point(system, json.load(f))
        with open(os.path.join(d, "result.json"), encoding="utf-8") as f:
            curve = [pt.curve_row(json.loads(line)) for line in f if line.strip()]
        out.append({"index": i, "config": config, "attempts": pt.read_attempts(d), "curve": curve})
    return out


def canonical_of(a: argparse.Namespace, bound: dict[str, Any], trials: list[dict[str, Any]],
                 segments: list[dict[str, Any]]) -> dict[str, Any]:
    """The canonical tune/1 dict of a finished search (pilot.tune.build_record adds the derived part)."""
    return {"schema": pt.SCHEMA, "system": a.system,
            "search": {"seed": a.seed, "n_trials": a.trials, "search_alg": "BasicVariantGenerator", "max_concurrent": 1,
                       "points_to_evaluate": a.points, "space": copy.deepcopy(pt.SPACES[a.system]),
                       "scheduler": dict(pt.ASHA, max_t=pt.max_t(a.system)), "objective": pt.OBJECTIVE},
            "data": {"split": pt.SOURCE, "sha256": bound["data_sha256"],
                     "manifest_is_committed": bound["manifest_is_committed"],
                     "train_folds": list(bound["block"]["train_folds"]), "dev_folds": list(bound["block"]["dev_folds"]),
                     "train_uids_sha256": bound["block"]["train_uids_sha256"],
                     "dev_uids_sha256": bound["block"]["dev_uids_sha256"], "n_train": bound["n_train"],
                     "n_dev": bound["n_dev"], "dev_rows_with_train_text": bound["dev_rows_with_train_text"]},
            "fixed": copy.deepcopy(pt.FIXED[a.system]), "base": dict(bound["base_record"]), "device": a.device,
            "cpus": a.cpus, "versions": dict(bound["versions"]), "segments": segments, "notes": list(pt.NOTES),
            "trials": trials}


def search_spec(a: argparse.Namespace, bound: dict[str, Any]) -> dict[str, Any]:
    """What makes two driver starts the same search, as JSON values: the system, seed, trial count, pinned points,
    space, scheduler, device, CPUs, the data block (the split sha256, whether the manifest is the committed one, the
    partition block and counts), the base record, the fixed recipe and the versions. Never a path, never the test
    hook: a rerun from another checkout or storage location is the same search."""
    data = {"sha256": bound["data_sha256"], "manifest_is_committed": bound["manifest_is_committed"],
            "block": bound["block"], "n_train": bound["n_train"], "n_dev": bound["n_dev"],
            "dev_rows_with_train_text": bound["dev_rows_with_train_text"]}
    spec = {"system": a.system, "seed": a.seed, "n_trials": a.trials, "points": a.points,
            "space": pt.SPACES[a.system], "scheduler": dict(pt.ASHA, max_t=pt.max_t(a.system)), "device": a.device,
            "cpus": a.cpus, "data": data, "base": bound["base_record"], "fixed": pt.FIXED[a.system],
            "versions": bound["versions"]}
    return json.loads(json.dumps(spec, ensure_ascii=False))   # tuples become lists, as the spec file reads back


def spec_path(storage: str, system: str) -> str:
    """STORAGE/SYS-search.json: the search's spec, one segment per driver start, finished and the record sha256."""
    return os.path.join(storage, "%s-search.json" % system)


def read_search(path: str) -> dict[str, Any] | None:
    """The search file at `path` ({"spec", "segments", "finished", "record_sha256"}), None when it does not exist;
    SystemExit when it is not that layout."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit("%s is not readable JSON (%s); delete the search's storage to start over" % (path, e)) \
            from None
    if (not isinstance(doc, dict) or list(doc) != list(SEARCH_FILE_KEYS) or not isinstance(doc["spec"], dict)
            or not isinstance(doc["segments"], list) or type(doc["finished"]) is not bool):
        raise SystemExit("%s is not a search file of this tune.py (keys %s); delete the search's storage to start "
                         "over" % (path, ", ".join(SEARCH_FILE_KEYS)))
    return doc


def write_search(path: str, doc: dict[str, Any]) -> None:
    """Write the search file atomically (UTF-8, indent 2, one trailing newline)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_atomic(path, (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def first_difference(have: Any, want: Any, where: str = "") -> str | None:
    """The dotted path of the first key at which two JSON values differ (None when equal): keys in `want`'s order,
    then any key only `have` holds; a value of another type or a differing list is named as a whole."""
    if isinstance(have, dict) and isinstance(want, dict):
        for k in list(want) + [k for k in have if k not in want]:
            key = "%s.%s" % (where, k) if where else str(k)
            if k not in have or k not in want:
                return key
            d = first_difference(have[k], want[k], key)
            if d is not None:
                return d
        return None
    return None if have == want and type(have) is type(want) else (where or "spec")


def stale_paths(storage: str, system: str) -> str:
    """The three paths a refused search's storage asks to delete, comma-separated."""
    return ", ".join(os.path.join(storage, n) for n in (system, "%s-train" % system, "%s-search.json" % system))


def write_out(record: dict[str, Any], out: str) -> bool:
    """Check the record and write it atomically to `out`; an existing `out` of the same bytes is left as is. Returns
    True when it wrote `out`, False when `out` already held these bytes; SystemExit over an existing `out` of other
    bytes (a committed search record is never overwritten)."""
    pt.check_record(record)
    data = pt.record_bytes(record)
    if os.path.exists(out):
        with open(out, "rb") as f:
            if f.read() != data:
                raise SystemExit("%s exists with other content; a committed search record is never overwritten; pass "
                                 "another --out" % out)
        return False
    pt.write_record(record, out)
    return True


def finish(a: argparse.Namespace, bound: dict[str, Any], doc: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """(record, written) of a search whose trials all ended: the record built from the trial dirs under STORAGE/SYS
    and every segment of the search file, written to --out by write_out."""
    trials = collect(os.path.join(a.storage, a.system), a.trials)
    record = pt.build_record(canonical_of(a, bound, trials, copy.deepcopy(doc["segments"])))
    return record, write_out(record, a.out)


def preflight(a: argparse.Namespace) -> dict[str, Any]:
    """The quiet-machine preflight of an MPS search (compute_checkpoint.preflight at --min-available-gib: refuse below
    it, sample swap twice 5 s apart while idle; 0 records a busy-machine run), recorded as this start's segment's
    preflight."""
    import compute_checkpoint
    return compute_checkpoint.preflight(a.min_available_gib)


def check_mode(path: str) -> None:
    """--check RECORD: read_record (the layout, the rebuild from its canonical part, the bytes) and print
    "ok RECORD SHA256"; its refusal exits non-zero."""
    _, sha = pt.read_record(path)
    print("ok %s %s" % (path, sha), flush=True)


def render_out_ok(path: str) -> bool:
    """Whether --render may write `path`: anywhere outside results/, and under results/ only results/tune/tune.md."""
    results = os.path.join(REPO_ROOT, "results")
    return not within(path, results) or same_path(path, os.path.join(results, "tune", "tune.md"))


def render_mode(paths: list[str], out: str | None) -> None:
    """--render RECORD... [--out PATH]: pilot.tune.render_markdown of the records read through read_record (checked),
    printed, or written atomically to --out (refused, before any record is read, unless render_out_ok)."""
    if out is not None and not render_out_ok(out):
        raise SystemExit("--out %s: under results/ the rendered summary is results/tune/tune.md only; pass another "
                         "--out" % repo_relpath(out))
    text = pt.render_markdown([pt.read_record(p)[0] for p in paths])
    if out is None:
        print(text, end="", flush=True)
        return
    data = text.encode("utf-8")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_atomic(out, data)
    print("summary -> %s (sha256 %s)" % (out, sha256_bytes(data)), flush=True)


def status_mode(storage: str, system: str) -> None:
    """--status: the progress of the search under `storage` from its search file and trial dirs alone (no Ray, no
    write): one summary line, then per started trial its attempts, last reported epoch of its schedule and best dev
    micro-F1 so far."""
    doc = read_search(spec_path(storage, system))
    if doc is None:
        raise SystemExit("no search under %s (no %s); start one with tune.py --system %s"
                         % (storage, os.path.basename(spec_path(storage, system)), system))
    exp = os.path.join(storage, system)
    names = sorted(n for n in os.listdir(exp) if TRIAL_DIR_RE.fullmatch(n)) if os.path.isdir(exp) else []
    n_trials = doc["spec"].get("n_trials")
    print("search %s: %d of %s trials started, %d driver starts, finished %s"
          % (system, len(names), n_trials, len(doc["segments"]), "yes" if doc["finished"] else "no"), flush=True)
    for name in names:
        d = os.path.join(exp, name)
        attempts = pt.read_attempts(d) if os.path.exists(os.path.join(d, pt.ATTEMPTS_FILE)) else 0
        schedule = None
        if os.path.isfile(os.path.join(d, "params.json")):
            with open(os.path.join(d, "params.json"), encoding="utf-8") as f:
                schedule = json.load(f).get("epochs")
        rows = []
        if os.path.isfile(os.path.join(d, "result.json")):
            with open(os.path.join(d, "result.json"), encoding="utf-8") as f:
                rows = [pt.curve_row(json.loads(line)) for line in f if line.strip()]
        index = int(name.split("_")[1])
        if not rows:
            print("trial %d: attempts %d, no epoch reported yet" % (index, attempts), flush=True)
            continue
        f1 = [(Fraction(2 * r["dev_tp"], 2 * r["dev_tp"] + r["dev_fp"] + r["dev_fn"]) if r["dev_tp"] else Fraction(0),
               -r["epoch"]) for r in rows]
        top, neg_epoch = max(f1)                            # the earliest epoch of the best score
        print("trial %d: attempts %d, epoch %d of %s, best dev micro-F1 %.4f (epoch %d)"
              % (index, attempts, rows[-1]["epoch"], schedule if schedule is not None else "?", float(top),
                 -neg_epoch), flush=True)


def main(argv: list[str] | None = None) -> None:
    """Parse, bind on the driver, compare the search file's spec, rebuild a finished search without Ray, or append
    this start's segment, start Ray with headroom, run (or restore) the search, write the record and mark the search
    finished."""
    a = parse_args(argv)
    if a.check is not None:
        check_mode(a.check)
        return
    if a.status:
        status_mode(a.storage, a.system)
        return
    if a.render is not None:
        render_mode(a.render, a.out)
        return
    check_results_out(a)
    bound = bind(a)
    spec = search_spec(a, bound)
    sp = spec_path(a.storage, a.system)
    path = os.path.join(a.storage, a.system)
    doc = read_search(sp)
    if doc is not None:
        key = first_difference(doc["spec"], spec)
        if key is not None:
            raise SystemExit("%s holds a search with another spec (key %s); rerun the command that started it, or "
                             "delete %s to start over" % (a.storage, key, stale_paths(a.storage, a.system)))
    if doc is not None and doc["finished"]:                 # no Ray: the trial dirs and the segments rebuild it
        record, written = finish(a, bound, doc)
        sha = sha256_bytes(pt.record_bytes(record))
        if sha != doc["record_sha256"]:
            raise SystemExit("the finished search in %s rebuilds a record of sha256 %s, not the %s it wrote; its trial "
                             "dirs changed; delete %s to start over" % (a.storage, sha, doc["record_sha256"],
                                                                         stale_paths(a.storage, a.system)))
        print("search finished; record %s -> %s (sha256 %s)" % ("written" if written else "unchanged", a.out, sha),
              flush=True)
        return
    if os.path.isdir(path) and os.listdir(path):
        from ray import tune
        if doc is None:
            raise SystemExit("%s holds a Tune experiment without its search file %s; delete %s, then rerun"
                             % (path, sp, stale_paths(a.storage, a.system)))
        if not tune.Tuner.can_restore(path):
            raise SystemExit("%s holds a Tune experiment Ray Tune cannot restore (killed before its first snapshot); "
                             "delete %s, then rerun" % (path, stale_paths(a.storage, a.system)))
    import ray_train
    segment = dict(ray_train.code_state(), preflight=preflight(a) if a.device == "mps" else None)
    doc = {"spec": spec, "segments": (doc["segments"] if doc else []) + [segment], "finished": False,
           "record_sha256": None}
    write_search(sp, doc)                                   # this start's segment, before ray.init
    import ray
    from ray import tune
    # one trial slot: the generator's max_concurrent=1 limits only the trials it creates, while a restore re-queues
    # its unfinished and errored trials straight into the controller; a leaked inner placement group never holds it
    if a.system == "laya":                                  # leaked placement-group headroom (Ray Train trials)
        init = {"num_cpus": 1 + a.cpus * (a.trials + 1), "include_dashboard": False, "resources": {TRIAL_SLOT: 1}}
        trainable = tune.with_resources(tune.with_parameters(laya_trial, bound=bound), {"CPU": 1, TRIAL_SLOT: 1})
    else:                                                   # one process per trial, no Ray Train: no placement group
        init = {"num_cpus": a.cpus + 1, "include_dashboard": False, "resources": {TRIAL_SLOT: 1}}
        trainable = tune.with_resources(tune.with_parameters(berturk_trial, bound=bound),
                                        {"CPU": a.cpus, TRIAL_SLOT: 1})
    if bound["env"]:
        init["runtime_env"] = {"env_vars": bound["env"]}   # the spike's MPS worker env reaches every trial worker
    ray.init(**init)
    ray_train.require_single_node(ray.nodes())
    if tune.Tuner.can_restore(path):
        print("continuing the search in %s (driver start %d)" % (path, len(doc["segments"])), flush=True)
        # resume (never restart) errored trials: a restart gives a trial a new id without the _NNNNN index and a new
        # trial dir; a resumed trial keeps both, holds no checkpoint, so it runs from epoch 1 again, and Ray starts
        # its result.json afresh
        tuner = tune.Tuner.restore(path, trainable=trainable, resume_unfinished=True, resume_errored=True)
    else:
        tuner = tune.Tuner(trainable, param_space=param_space(a.system),
                           tune_config=tune_config(a.system, a.seed, a.trials, a.points),
                           run_config=tune.RunConfig(name=a.system, storage_path=a.storage))
    grid = tuner.fit()
    errored = sorted("%s (Tune error file %s)" % (os.path.basename(os.path.normpath(r.path)),
                                                  os.path.join(r.path, "error.txt"))
                     for r in grid if r.error is not None)       # trial dirs are trial_NNNNN: the trial index
    if len(grid) != a.trials or errored:
        raise SystemExit("the search ended with %d of %d trials, errored: %s; rerun the same command: errored trials "
                         "run again from epoch 1 in their own trial dirs" % (len(grid), a.trials,
                                                                             ", ".join(errored) or "none"))
    record, _ = finish(a, bound, doc)
    sha = sha256_bytes(pt.record_bytes(record))
    write_search(sp, dict(doc, finished=True, record_sha256=sha))
    print("record -> %s (sha256 %s)" % (a.out, sha), flush=True)


if __name__ == "__main__":
    main()
