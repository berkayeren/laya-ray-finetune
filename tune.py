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
for a SearchGenerator such as BasicVariantGenerator, so it is never passed.

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

Ray 2.59 behaviour this design handles:
- concurrency: max_concurrent=1 on the generator (above);
- leaked placement groups: an ASHA-stopped Laya trial leaves its Train placement group behind (Ray's cleaner needs the
  dashboard's State API, which the lock does not install), so ray.init gets logical-CPU headroom, 1 + cpus x (trials +
  1), and the inner trainer's FailureConfig(max_failures=0, controller_failure_limit=0) turns a starved start into a
  trial error instead of an endless retry;
- a restore can let a restarted trial pass an ASHA rung it had been judged at, so every trial counts its starts in
  attempts.json in its trial dir (pilot.tune.bump_attempts) and the record carries the count (never repaired);
- a function trial runs with its cwd under Ray's session dir, so every path a trial receives is absolute.

Storage (default tune_runs/, gitignored): <storage>/<system>/ is the Tune experiment (trial dirs trial_NNNNN, each with
Tune's params.json and result.json and our attempts.json), <storage>/<system>-train/ the inner Train runs. Rerunning the
identical command after a kill continues the search (Tuner.can_restore, then Tuner.restore with unfinished and errored
trials restarted from scratch), never rerunning a finished trial. Once every trial ended without an error the record is
built only from each trial dir's params.json, the CURVE_KEYS of its result.json lines and its attempts count (never
Tune's own fields: hostname, pid, timestamps, trial-id prefixes), written atomically to --out, and refused over an
existing --out of other content.

usage:
  python tune.py --system laya                     # the real search: 20 trials on MPS -> results/tune/laya.json
  python tune.py --system laya --device cpu --cpus 1 --trials 3 --base DIR --data SPLIT --manifest M \\
                 --storage STORAGE --out OUT.json   # a CPU smoke search on a local base
"""
import argparse
import copy
import json
import os
import re
from importlib.metadata import version
from typing import Any

from pilot import pins, runs, spike
from pilot import laya_adapter as la
from pilot import tune as pt
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, parse_jsonl, refuse_held_out,
                        repo_relpath, sha256_bytes, verify_split_file)
from pilot.weights import check_tokenizer_config_rewritable, checkpoint_digests, fetch, identity

DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "splits", "selection_train.jsonl")
TRIAL_DIR_RE = re.compile(r"^trial_\d{5}$")


def parser() -> argparse.ArgumentParser:
    """The tune.py argument parser."""
    ap = argparse.ArgumentParser(description="Exploratory Ray Tune search of one system (not a decision input).")
    ap.add_argument("--system", choices=pt.SYSTEMS, required=True)
    ap.add_argument("--trials", type=int, default=pt.TRIALS, help="trials of the search (default %d)" % pt.TRIALS)
    ap.add_argument("--seed", type=int, default=pt.SEED, help="the search seed (default %d)" % pt.SEED)
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps", help="training device (default mps)")
    ap.add_argument("--storage", default=os.path.abspath("tune_runs"),
                    help="Ray Tune and Train storage (default tune_runs/, gitignored)")
    ap.add_argument("--out", default=None, help="the tune/1 record (default results/tune/<system>.json)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="the selection_train split (default the committed one)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify --data against (default: the committed data/splits.json)")
    ap.add_argument("--base", default="multilingual", help="'multilingual' (pinned) or a local Laya checkpoint dir")
    ap.add_argument("--cpus", type=int, default=None, help="CPUs and torch threads per trial (default 4 on mps, else 1)")
    ap.add_argument("--points", default=None, help=argparse.SUPPRESS)   # JSON configs evaluated first; tests only
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsed and checked args: --out and --cpus resolved, --points parsed into checked search points (or None)."""
    ap = parser()
    a = ap.parse_args(argv)
    if a.cpus is None:
        a.cpus = 4 if a.device == "mps" else 1
    for flag, v in (("--trials", a.trials), ("--cpus", a.cpus)):
        if v < 1:
            ap.error("%s must be at least 1" % flag)
    if a.seed < 0:
        ap.error("--seed must be at least 0")
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
    a.storage = os.path.abspath(a.storage)
    a.data = os.path.abspath(a.data)
    a.manifest = os.path.abspath(a.manifest)
    return a


def bind(a: argparse.Namespace) -> dict[str, Any]:
    """The picklable binding of a search, built on the driver before ray.init: the verified selection_train (path,
    sha256, manifest entry), the tuning partition block and counts, the base (resolved dir, identity, verified
    digests), the device, CPUs and worker env, the storage dirs and the versions. Every path is absolute (a trial's
    cwd is under Ray's session dir). SystemExit on a split that is not selection_train, committed held-out content, a
    partition refusal, another laya, a base that is neither the pin nor a local checkpoint, MPS unavailable, or less
    free disk under --storage than pilot.tune.DISK_FLOOR_BYTES (trials write no checkpoints)."""
    if a.system != "laya":
        raise SystemExit("tune.py --system %s: this tune.py runs Laya searches only; run --system laya" % a.system)
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
    if a.device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps, but MPS is not available here; use --device cpu, or run on Apple silicon "
                             "with the locked torch build")
    free = runs.free_bytes(a.storage)
    if free < pt.DISK_FLOOR_BYTES:
        raise SystemExit("%.1f GiB free under --storage; a search needs at least %.1f GiB (trial logs and Ray state); "
                         "free disk or pass another --storage" % (free / 2**30, pt.DISK_FLOOR_BYTES / 2**30))
    return {"system": a.system, "data": a.data, "data_sha256": entry["sha256"], "manifest": a.manifest,
            "manifest_is_committed": manifest_is_committed(a.manifest), "entry": entry, "block": block,
            "n_train": len(train_rows), "n_dev": len(dev_rows),
            "dev_rows_with_train_text": pt.shared_texts(train_rows, dev_rows),
            "base": a.base, "base_dir": base_dir, "base_name": base_name, "base_ident": base_ident,
            "base_sha256": base_sha256,
            "base_record": {"name": "multilingual" if a.base == "multilingual" else repo_relpath(base_dir),
                            "model_sha256": base_ident["model_sha256"],
                            "config_sha256": (pins.LAYA["sha256"]["rl_agent_config.json"] if a.base == "multilingual"
                                              else base_ident["config_sha256"])},
            "device": a.device, "cpus": a.cpus, "env": spike.spike_env("0", a.cpus) if a.device == "mps" else {},
            "storage": a.storage, "train_storage": os.path.join(a.storage, "%s-train" % a.system),
            "versions": {"ray": version("ray"), "torch": version("torch"), "laya": version("laya")}}


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
    attempt = pt.bump_attempts(ctx.get_trial_dir())
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


def write_out(record: dict[str, Any], out: str) -> str:
    """Check the record and write it atomically to `out` (an existing `out` of the same bytes is left as is); SystemExit
    over an existing `out` of other bytes (a committed search record is never overwritten). Returns out."""
    pt.check_record(record)
    data = pt.record_bytes(record)
    if os.path.exists(out):
        with open(out, "rb") as f:
            if f.read() != data:
                raise SystemExit("%s exists with other content; a committed search record is never overwritten; pass "
                                 "another --out" % out)
    else:
        pt.write_record(record, out)
    return out


def main(argv: list[str] | None = None) -> None:
    """Parse, bind on the driver, start Ray with headroom, run (or restore) the search, then write the record."""
    a = parse_args(argv)
    bound = bind(a)
    import ray
    from ray import tune

    import ray_train
    init = {"num_cpus": 1 + a.cpus * (a.trials + 1), "include_dashboard": False}   # leaked placement-group headroom
    if bound["env"]:
        init["runtime_env"] = {"env_vars": bound["env"]}   # the spike's MPS worker env reaches every trial worker
    ray.init(**init)
    ray_train.require_single_node(ray.nodes())
    trainable = tune.with_resources(tune.with_parameters(laya_trial, bound=bound), {"CPU": 1})
    path = os.path.join(a.storage, a.system)
    if tune.Tuner.can_restore(path):
        print("continuing the search in %s" % path, flush=True)
        tuner = tune.Tuner.restore(path, trainable=trainable, resume_unfinished=True, restart_errored=True)
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
                         "restart from scratch" % (len(grid), a.trials, ", ".join(errored) or "none"))
    trials = collect(path, a.trials)
    segments = [dict(ray_train.code_state(), preflight=None)]
    record = pt.build_record(canonical_of(a, bound, trials, segments))
    write_out(record, a.out)
    print("record -> %s (sha256 %s)" % (a.out, sha256_bytes(pt.record_bytes(record))), flush=True)


if __name__ == "__main__":
    main()
