"""Torch-free run-record schema of ray_train.py (LAYA-06), the neg-K item count and the LAYA-03 fallback ladder.

ray_train.py writes one record per run into the checkpoint config, at training.ray_train.record: the device the
parameters landed on, the worker env (KEY=VALUE strings, MPS only), the precision and checkpointing choices, the
load, calibration and final-save times, and one row per completed epoch with exactly EPOCH_FIELDS (wall clock of the
epoch's training steps, save time, optimizer steps, items, peak memory, the system swap delta and the run segment that
trained it), one SEGMENT_FIELDS entry per run segment (a killed run's rerun adds one). Nothing here may import torch: compute_checkpoint.py imports it before any child sets the MPS env, and torch reads
PYTORCH_ENABLE_MPS_FALLBACK once, at import.

The ladder (SPIKE_START, RUNG_NEXT, rung_label, rung_flags, oom_message, failure_line) turns a fallback into flags and
a derived label: a run on any rung other than none says so in its record, and an MPS out-of-memory exit names the
flags of the next rung, except after the last epoch (RESUME_STAGES), where it says to rerun the same command.

Run identity (LAYA-04): IDENTITY lists every value that changes what a run trains; canonical() keeps exactly those and
run_id() names the run by them (split, neg K, seed and the first 8 hex of the sha256 of the sorted canonical JSON). The
run_id is the Ray experiment name, so storage/<run_id>/ is the run's only resume directory: rerunning the same config
resumes, another seed or config gets another dir. Every ray_train.py flag is in exactly one of IDENTITY (directly or
through its resolved value) and NOT_IDENTITY (a test enforces it), so a new flag cannot silently drop out of the key.
check_state refuses a resume from another run's train state; disk_need_bytes is the driver's free-disk floor.

usage: from pilot import runs; record = runs.read_record(ckpt_dir); [row["train_s"] for row in record["epochs"]]
       runs.neg_items(rows, 8); runs.rung_label(cfg); runs.oom_message(epoch, micro_batch, rung)
       rid = runs.run_id(runs.canonical(values))
"""
import hashlib
import json
import os

from pilot.data import repo_relpath

# ---- LAYA-04 run identity ----

# every value that changes what a run trains: the split by name and bytes, the base model by digest, the laya version,
# the resolved device and its precision / checkpointing / optimizer choices, every hyperparameter and the seeds
IDENTITY = ("split", "data_sha256", "data_role", "base_model_sha256", "base_config_sha256", "laya_version", "device",
            "workers", "epochs", "micro_batch", "grad_accum", "encoder_lr", "head_lr", "weight_decay", "clip_grad_norm",
            "loss", "neg", "neg_seed", "freeze_encoder", "freeze_embeddings", "grad_checkpointing", "amp", "max_len",
            "head_max_len", "max_messages", "calib_max_messages", "calib_frac", "shuffle_options", "seed",
            "adamw_foreach", "adamw_fused")
# ray_train.py flags that never change training, so never the run_id:
#   data, base, manifest: paths, represented in IDENTITY by the verified sha256s (data_sha256, base_*_sha256)
#   out, storage: where the result and the Ray run dir go
#   cpus_per_worker, object_store_gb: resources (Ray scheduling, torch threads, object store size)
#   calib_batch: the batch size of the calibration forward pass (eval only, after training)
#   gpu: an alias, represented by its resolved value (device)
#   print_run_id: prints the run_id and exits
#   fail_after_epoch: a test hook that kills the run after an epoch's checkpoint; the rerun must resume the same run
#   export_epochs: whether per-epoch laya exports are written next to the run (outputs, not training)
NOT_IDENTITY = ("data", "base", "out", "storage", "manifest", "cpus_per_worker", "object_store_gb", "calib_batch", "gpu",
                "print_run_id", "fail_after_epoch", "export_epochs")


def canonical(values):
    """{key: values[key]} for exactly the IDENTITY keys; ValueError naming any missing or unexpected key."""
    missing = [k for k in IDENTITY if k not in values]
    extra = sorted(k for k in values if k not in IDENTITY)
    if missing or extra:
        raise ValueError("run identity: missing %s; unexpected %s" % (", ".join(missing) or "none",
                                                                    ", ".join(extra) or "none"))
    return {k: values[k] for k in IDENTITY}


def check_state(state, run_id, canonical):
    """Refuse a train state that is not this run's: its kind, run_id and canonical config must all match (every rank
    checks the same file, so every rank refuses together and no rank waits at a report barrier)."""
    for key, want in (("kind", "train_state"), ("run_id", run_id), ("canonical", canonical)):
        if state.get(key) != want:
            raise SystemExit("Ray storage holds a train state from another run (%s differs); delete storage/%s or "
                             "choose another --storage" % (key, run_id))


def disk_need_bytes(model_bytes, epochs, export_epochs):
    """Bytes a run needs under --storage: three train states alive during checkpoint rotation (the temp dir, Ray's
    copy, the previous epoch), each 6 x the fp16 weight bytes (fp32 weights + two AdamW moments), the per-epoch laya
    exports, the final checkpoint and a 1 GiB margin."""
    return 3 * 6 * model_bytes + (epochs * model_bytes if export_epochs else 0) + model_bytes + 2**30


def run_id(canon):
    """laya-<split>-<negK or full>-s<seed>-<first 8 hex of the sha256 of the sorted, compact canonical JSON>.

    sort_keys and fixed separators make the id independent of key and flag order."""
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "laya-%s-%s-s%d-%s" % (canon["split"], "neg%d" % canon["neg"] if canon["neg"] else "full", canon["seed"],
                                  hashlib.sha256(blob).hexdigest()[:8])

# one row per completed epoch; peak_driver_bytes and current_allocated_bytes are None on CPU (never 0), where
# peak_total_bytes is the peak RSS; on MPS peak_total_bytes is the peak of driver-allocated memory + RSS (the spike's
# peak_memory_bytes basis); segment is the run segment that trained the epoch, resumed whether that segment started from
# a checkpoint (a killed run's rerun)
EPOCH_FIELDS = ("epoch", "loss", "train_s", "save_s", "optimizer_steps", "n_items", "peak_total_bytes",
                "peak_driver_bytes", "peak_rss_bytes", "current_allocated_bytes", "swap_delta_bytes", "segment",
                "resumed")
# one entry per run segment (segment 0 = the first start; every rerun after a kill adds one); git_diff_sha256 is the
# sha256 of the segment's uncommitted tracked diff (ray_train.code_state), so an edit between a kill and its rerun shows
# even when HEAD did not move; resume_load_s and resume_peak_total_bytes are None on a segment that started at epoch 0
SEGMENT_FIELDS = ("segment", "start_epoch", "resumed", "git_sha", "git_dirty", "git_diff_sha256", "device",
                  "resume_load_s", "resume_peak_total_bytes")


def neg_items(rows, k):
    """Training items a neg-K run builds from `rows`, counted without a tokenizer (the compute projection's count).

    Per row: every positive plus min(k, labels - positives) negatives (pilot.spike.neg8_rows keeps exactly these), or
    every label for k 0 (data_full); labels = the row's `expected` entries. laya's items_from_rows skips nothing on a
    well-formed split, so this equals the number of items laya builds (ray_train.py records both).
    """
    if k < 0:
        raise ValueError("k must be 0 (all labels) or a positive count, got %r" % (k,))
    n = 0
    for r in rows:
        labels, pos = len(r["expected"]), sum(1 for v in r["expected"].values() if v)
        n += pos + min(k, labels - pos) if k else labels
    return n


def env_strings(env):
    """Sorted "KEY=VALUE" strings of an env dict, the only form the worker env takes in a run record."""
    return sorted("%s=%s" % (k, v) for k, v in env.items())


def read_record(ckpt_dir):
    """training.ray_train.record of a ray_train.py checkpoint dir; refuses a missing record or an incomplete epoch row."""
    with open(os.path.join(ckpt_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    record = ((cfg.get("training") or {}).get("ray_train") or {}).get("record")
    if not record:
        raise SystemExit("%s holds no ray_train run record; train it with this ray_train.py" % repo_relpath(ckpt_dir))
    for row in record.get("epochs") or []:
        missing = [k for k in EPOCH_FIELDS if k not in row]
        if missing:
            raise SystemExit("run record epoch %s lacks %s" % (row.get("epoch"), ", ".join(missing)))
    return record


# ---- LAYA-03 fallback ladder: flags, never an in-process retry ----

# verdict.chosen of results/spike/mps_spike.json (Phase 2, rung none): the config every Phase 4 run starts from
SPIKE_START = {"micro_batch": 8, "max_len": 128, "grad_accum": 4}
# rung -> the rung to try after an MPS out-of-memory exit; the smaller batch keeps the effective batch 32 (4 x 8)
RUNG_NEXT = {"none": "embeddings_frozen",
             "embeddings_frozen": "embeddings_frozen+smaller_batch",
             "smaller_batch": "embeddings_frozen+smaller_batch",
             "embeddings_frozen+smaller_batch": "encoder_frozen",
             "encoder_frozen": None}
_SMALLER = ["--micro-batch", "4", "--grad-accum", "8"]
OOM_MARK = "MPS out of memory"
# stages after the last epoch is registered: every epoch is saved and the eval pass needs less memory than training
# (the optimizer state is freed first), so the line says to rerun the same command, which resumes there; a rung flag
# would change the run_id and retrain from epoch 0 (04-REVIEW WR-01). stage -> what the rerun does
RESUME_STAGES = {"while calibrating after the last epoch": "it only recalibrates; a smaller --calib-batch keeps the "
                                                           "run id",
                 "while saving the final checkpoint": "it recalibrates and saves again"}
RESUME_TODO = "every epoch is saved: rerun the same command to resume"


def rung_label(cfg):
    """The fallback rung a resolved config runs on, derived from its flags (never a typed claim).

    encoder_frozen wins (head-only); otherwise embeddings_frozen and/or smaller_batch (micro-batch below the spike's
    start) joined by "+", or none.
    """
    if cfg["freeze_encoder"]:
        return "encoder_frozen"
    parts = (["embeddings_frozen"] if cfg["freeze_embeddings"] else []) + \
            (["smaller_batch"] if cfg["micro_batch"] < SPIKE_START["micro_batch"] else [])
    return "+".join(parts) or "none"


def weaker(label):
    """True for the head-only rung, the one that must be labelled weaker wherever it appears."""
    return label == "encoder_frozen"


def next_rung(label):
    return RUNG_NEXT[label]


def rung_flags(label):
    """The ray_train.py flags that put a spike-start command on `label`."""
    return {"none": [], "embeddings_frozen": ["--freeze-embeddings"], "smaller_batch": list(_SMALLER),
            "embeddings_frozen+smaller_batch": ["--freeze-embeddings"] + _SMALLER,
            "encoder_frozen": ["--freeze-encoder"]}[label]


def spike_departures(cfg, max_len):
    """{key: value} where a run leaves the spike's measured start (max_len = the resolved one), plus effective_batch
    when micro_batch x grad_accum is not the spike's 32; {} at the spike start."""
    have = {"micro_batch": cfg["micro_batch"], "max_len": max_len, "grad_accum": cfg["grad_accum"]}
    out = {k: v for k, v in have.items() if v != SPIKE_START[k]}
    eff = cfg["micro_batch"] * cfg["grad_accum"]
    if eff != SPIKE_START["micro_batch"] * SPIKE_START["grad_accum"]:
        out["effective_batch"] = eff
    return out


def oom_message(epoch, step, label, stage=None):
    """One line for an MPS out-of-memory exit: where it happened, on which rung, and the command change to try next.

    stage names an allocation outside the training step (e.g. "while moving the model to the device"); epoch is then
    the epoch it happened in or after, or None when it is not known, and step is ignored. A RESUME_STAGES stage gets
    RESUME_TODO instead of the next rung (resume_oom tells the two apart)."""
    nxt = next_rung(label)
    if stage in RESUME_STAGES:
        todo = "%s (%s)" % (RESUME_TODO, RESUME_STAGES[stage])
    elif nxt is None:
        todo = "no further rung: the fallback ladder is exhausted"
    else:
        todo = "rerun with %s" % " ".join(rung_flags(nxt))
        if rung_flags(label):
            todo += " in place of %s" % " ".join(rung_flags(label))
    if stage is None:
        where = "at epoch %d micro-batch %d" % (epoch, step)
    else:
        where = stage if epoch is None else "%s at epoch %d" % (stage, epoch)
    return "%s %s (rung %s); %s" % (OOM_MARK, where, label, todo)


def failure_line(text):
    """The oom_message line inside a (multi-line) Ray failure text, from OOM_MARK on, or None."""
    for line in str(text).splitlines():
        if OOM_MARK in line:
            return line[line.index(OOM_MARK):].strip()
    return None


def resume_oom(line):
    """True for an oom_message line of a RESUME_STAGES stage: rerun the same command, never the next rung."""
    return bool(line) and OOM_MARK in line and RESUME_TODO in line
