"""Torch-free run-record schema of ray_train.py (LAYA-06): the per-epoch fields and the reader.

ray_train.py writes one record per run into the checkpoint config, at training.ray_train.record: the device the
parameters landed on, the worker env (KEY=VALUE strings, MPS only), the precision and checkpointing choices, the
load, calibration and final-save times, and one row per completed epoch with exactly EPOCH_FIELDS (wall clock of the
epoch's training steps, save time, optimizer steps, items, peak memory and the system swap delta). Nothing here may
import torch: compute_checkpoint.py imports it before any child sets the MPS env, and torch reads
PYTORCH_ENABLE_MPS_FALLBACK once, at import.

usage: from pilot import runs; record = runs.read_record(ckpt_dir); [row["train_s"] for row in record["epochs"]]
"""
import json
import os

from pilot.data import repo_relpath

# one row per completed epoch; peak_driver_bytes and current_allocated_bytes are None on CPU (never 0), where
# peak_total_bytes is the peak RSS; on MPS peak_total_bytes is the peak of driver-allocated memory + RSS (the spike's
# peak_memory_bytes basis)
EPOCH_FIELDS = ("epoch", "loss", "train_s", "save_s", "optimizer_steps", "n_items", "peak_total_bytes",
                "peak_driver_bytes", "peak_rss_bytes", "current_allocated_bytes", "swap_delta_bytes")


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
