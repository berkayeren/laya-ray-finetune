"""Torch-free run-record schema of ray_train.py (LAYA-06), the neg-K item count and the LAYA-03 fallback ladder.

ray_train.py writes one record per run into the checkpoint config, at training.ray_train.record: the device the
parameters landed on, the worker env (KEY=VALUE strings, MPS only), the precision and checkpointing choices, the
load, calibration and final-save times, and one row per completed epoch with exactly EPOCH_FIELDS (wall clock of the
epoch's training steps, save time, optimizer steps, items, peak memory and the system swap delta). Nothing here may
import torch: compute_checkpoint.py imports it before any child sets the MPS env, and torch reads
PYTORCH_ENABLE_MPS_FALLBACK once, at import.

The ladder (SPIKE_START, RUNG_NEXT, rung_label, rung_flags, oom_message, failure_line) turns a fallback into flags and
a derived label: a run on any rung other than none says so in its record, and an MPS out-of-memory exit names the
flags of the next rung.

usage: from pilot import runs; record = runs.read_record(ckpt_dir); [row["train_s"] for row in record["epochs"]]
       runs.neg_items(rows, 8); runs.rung_label(cfg); runs.oom_message(epoch, micro_batch, rung)
"""
import json
import os

from pilot.data import repo_relpath

# one row per completed epoch; peak_driver_bytes and current_allocated_bytes are None on CPU (never 0), where
# peak_total_bytes is the peak RSS; on MPS peak_total_bytes is the peak of driver-allocated memory + RSS (the spike's
# peak_memory_bytes basis)
EPOCH_FIELDS = ("epoch", "loss", "train_s", "save_s", "optimizer_steps", "n_items", "peak_total_bytes",
                "peak_driver_bytes", "peak_rss_bytes", "current_allocated_bytes", "swap_delta_bytes")


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


def oom_message(epoch, step, label):
    """One line for an MPS out-of-memory exit: where it happened, on which rung, and the command change to try next."""
    nxt = next_rung(label)
    if nxt is None:
        todo = "no further rung: the fallback ladder is exhausted"
    else:
        todo = "rerun with %s" % " ".join(rung_flags(nxt))
        if rung_flags(label):
            todo += " in place of %s" % " ".join(rung_flags(label))
    return "%s at epoch %d micro-batch %d (rung %s); %s" % (OOM_MARK, epoch, step, label, todo)


def failure_line(text):
    """The oom_message line inside a (multi-line) Ray failure text, from OOM_MARK on, or None."""
    for line in str(text).splitlines():
        if OOM_MARK in line:
            return line[line.index(OOM_MARK):].strip()
    return None
