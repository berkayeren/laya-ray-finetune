"""Distributed fine-tuning of a Laya checkpoint with Ray Train (PyTorch DDP).

`laya.train.finetune` / `laya-train` run on one device. This script keeps Laya's
own building blocks -- item encoding, RLCD / soft-CE loss, temperature
calibration and the checkpoint layout `laya.load` reads, all reached through
pilot/laya_adapter.py (the only private-API module, pinned to one laya version)
-- and runs the training loop under Ray's TorchTrainer, so it scales from 1 CPU worker to N GPUs on one
node with only --workers / --device changed. It is single-node: workers read the
base weights and the split from driver-local paths, so it refuses a multi-node
Ray cluster.

Per worker:  load base -> build items -> shard by rank -> DDP train
Rank 0:      fit calibration temperatures on whole messages held out before training
             (every question of a held-out message; none of their text is trained on),
             save the final checkpoint and report it to Ray.

Devices (--device cpu|cuda|mps, default cpu; --gpu is an alias for --device cuda): cpu and cuda take the device from
Ray (use_gpu on cuda, DDP across --workers). Ray has no MPS device manager, so --device mps runs one Ray worker with
use_gpu False on an explicit torch.device("mps"), with the Phase 2 spike's worker env (strict fallback 0, HIGH and LOW
MPS watermarks 1.0) passed through ray.init(runtime_env) and checked in the worker. On every device the worker
refuses to train unless every model parameter landed on the requested device (no silent CPU run). Gradient
checkpointing (encoder and head) is on for every full-encoder run (--no-grad-checkpointing turns it off); AMP is a
separate --amp choice, on by default only on cuda.

Run record (LAYA-06): the checkpoint config holds training.ray_train.record: device, parameter device, torch threads,
worker env, precision and checkpointing choices, load, calibration and final-save seconds, and one row per completed
epoch with the pilot.runs.EPOCH_FIELDS (device-synchronised wall clock of the training steps, save time, optimizer
steps, items, peak memory sampled after every micro-batch backward and optimizer step, system swap delta).

Items (data_neg8 by default): the split files label all 48 tags. After the --max-messages subsample and the
message-level calibration holdout, every training message keeps all its positives plus --neg K random negatives
(pilot.spike.neg8_rows, one random.Random(--neg-seed) draw per run, --neg-seed defaulting to --seed); --neg 0 trains on
all 48 labels (data_full). The calibration messages always keep all 48 labels. The record carries the items laya built
and the torch-free count pilot.runs.neg_items gives, so the compute projection can count any split without a tokenizer.

Fallback ladder (LAYA-03): every run starts from the Phase 2 spike's measured config (pilot.runs.SPIKE_START: micro-batch
8, max_len 128, grad-accum 4, full encoder). The fallbacks are flags, never an in-process retry (the MPS allocator keeps a
process-wide high-water mark): --freeze-embeddings (rung embeddings_frozen), then also --micro-batch 4 --grad-accum 8
(rung embeddings_frozen+smaller_batch, effective batch 32 kept), then --freeze-encoder (rung encoder_frozen, head-only,
weaker). pilot.runs.rung_label derives the rung from the resolved flags; the record carries rung, fallback, weaker,
spike_departures and the trainable/total parameter counts. An MPS out-of-memory error ends the run with one line naming
the epoch, micro-batch, rung and the flags of the next rung.

Run identity (LAYA-04): the run_id is pilot.runs.run_id of the command's canonical config (pilot.runs.IDENTITY: split,
data and base digests, laya version, the resolved device and precision choices, every hyperparameter and the seeds; never
a path, a resource or the time), e.g. laya-selection_train-neg8-s0-1a2b3c4d. It is the Ray experiment name, so
storage/<run_id>/ is the run's only directory and the same command always lands in it; --print-run-id prints it after
the split, base and pin checks and exits without starting Ray.

--data must be a split registered in data/splits.json with role train and a matching sha256; anything else
(tampered, unregistered, val/test/probe) is refused before Ray starts. --base multilingual fetches the pinned,
sha256-verified Laya weights (pilot.weights) before Ray starts; any other value must be a local checkpoint dir.
--manifest selects the prepare-written splits.json to verify against (default: the committed data/splits.json; use
OTHER_DIR/splits.json for a `prepare_data.py --out OTHER_DIR` variant). Verification itself cannot be skipped, a
manifest cannot change any split's role, path or folds, and the checkpoint provenance records manifest_is_committed.

usage:
  python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
                      --workers 4 --device cuda    # 4 GPUs on one node (--gpu is the same)
  python ray_train.py ... --device mps --workers 1  # the full encoder on the Apple GPU
  python ray_train.py ... --device mps --workers 1 --freeze-embeddings   # fallback rung 1 after an MPS OOM
  python ray_train.py ... --workers 2 --freeze-encoder --max-messages 100   # CPU smoke test
"""
import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from contextlib import contextmanager, nullcontext

import psutil
import ray
import ray.train
import torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from pilot import laya_adapter as la
from pilot import pins, runs, spike
from pilot.data import (DEFAULT_MANIFEST, manifest_is_committed, parse_jsonl, repo_relpath, sha256_file,
                        verify_split_file)
from pilot.weights import checkpoint_digests, fetch, identity, verify_files

WEIGHT_DECAY = 0.01                                         # AdamW weight decay (part of the run identity)
CLIP_GRAD_NORM = 1.0                                        # gradient-norm clip per optimizer step (same)


def read_verified_rows(path, sha256):
    """Rows of a JSONL split whose bytes must hash to `sha256` (the driver-verified digest); parses those same bytes.

    Workers re-check because they read the file minutes after the driver verified it (fetch, ray.init).
    Rows match laya.train.read_jsonl (pilot.data.parse_jsonl splits on "\n" only).
    """
    with open(path, "rb") as f:
        raw = f.read()
    have = hashlib.sha256(raw).hexdigest()
    if have != sha256:
        raise SystemExit("%s sha256 %s != %s verified on the driver; regenerate with `make data`"
                         % (path, have, sha256))
    return parse_jsonl(raw)


def require_single_node(nodes):
    """Refuse a multi-node cluster: base_dir and the split path are the driver's local paths (no copy elsewhere)."""
    alive = [n for n in nodes if n.get("Alive")]
    if len(alive) > 1:
        raise SystemExit("ray_train.py is single-node: workers read the driver-local base weights and split, "
                         "but the Ray cluster has %d alive nodes" % len(alive))


def require_driver_node(node_id, driver_node_id):
    """Worker-side half of require_single_node: an autoscaler can add a node after the driver's one-time check."""
    if node_id != driver_node_id:
        raise SystemExit("ray_train.py is single-node: this worker runs on node %s, not the driver's node %s, so the "
                         "driver-local base weights and split are not here" % (node_id, driver_node_id))


def check_env(expected):
    """{name: value seen} for the expected env vars; refuses on the first one the worker did not get."""
    seen = {k: os.environ.get(k) for k in expected}
    for k in sorted(expected):
        if seen[k] != expected[k]:
            raise SystemExit("worker env %s=%r, expected %r; env vars must reach the worker through "
                             "ray.init(runtime_env)" % (k, seen[k], expected[k]))
    return seen


def require_param_device(model, device_type):
    """Refuse unless every model parameter sits on `device_type` (LAYA-02: a run asserts where the model landed)."""
    have = {p.device.type for p in model.parameters()}
    if have != {device_type}:
        raise SystemExit("model parameters on %s, expected %s; refusing to train on the wrong device (no silent CPU "
                         "run)" % (", ".join(sorted(have)), device_type))


def resolve(a):
    """The one resolved device and its precision, checkpointing, thread and optimizer choices for parsed args `a`.

    --gpu is an alias for --device cuda (ValueError next to any other --device). AMP defaults on only for cuda;
    checkpointing is on for every full-encoder run unless --no-grad-checkpointing. AdamW's foreach and fused are pinned
    off on mps (what the Phase 2 spike measured; torch's default there anyway) and left to torch on cpu and cuda.
    """
    if a.gpu and a.device not in (None, "cuda"):
        raise ValueError("--gpu is an alias for --device cuda; it conflicts with --device %s" % a.device)
    device = "cuda" if a.gpu else (a.device or "cpu")
    mps = device == "mps"
    return {"device": device,
            "amp": (device == "cuda") if a.amp is None else a.amp,
            "grad_checkpointing": bool(a.grad_checkpointing and not a.freeze_encoder),
            "cpus_per_worker": a.cpus_per_worker or (4 if mps else 1),   # 4 = the spike's timing basis
            "neg_seed": a.seed if a.neg_seed is None else a.neg_seed,
            "adamw_foreach": False if mps else None,
            "adamw_fused": False if mps else None}


@contextmanager
def mps_oom_exit(epoch, step, rung):
    """Turn an MPS out-of-memory RuntimeError into the one-line rung message (pilot.runs.oom_message); no retry, since
    the MPS allocator keeps a process-wide high-water mark. Any other error propagates unchanged."""
    try:
        yield
    except RuntimeError as e:
        if spike.is_mps_oom(str(e)):
            raise RuntimeError(runs.oom_message(epoch, step, rung)) from e
        raise


def fit_or_exit(trainer):
    """trainer.fit(); a worker's MPS out-of-memory exit reaches the driver inside Ray's TrainingFailedError and ends
    the run with that one line (SystemExit); every other failure is re-raised as Ray gave it."""
    try:
        return trainer.fit()
    except ray.train.TrainingFailedError as e:
        line = runs.failure_line(str(e))
        if line is None:
            raise
        raise SystemExit(line) from None


def provenance(entry, manifest_path, base, base_dir, ident=None):
    """Checkpoint provenance: split, manifest and base model by repo-relative path and sha256, never a local path."""
    return {"data": entry["path"], "data_role": entry["role"], "data_sha256": entry["sha256"],
            "manifest": repo_relpath(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "manifest_is_committed": manifest_is_committed(manifest_path),
            "base": identity("multilingual" if base == "multilingual" else base_dir) if ident is None else ident}


def epoch_config(cfg, max_len, head_max_len, ray_cfg, epoch):
    """Config of an intermediate epoch checkpoint: marked fine-tuned and uncalibrated, and traceable to its data
    (provenance, calibration holdout) like the final one, so a best-epoch pick from ray_results stays auditable."""
    return dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                training=dict(cfg.get("training") or {}, ray_train=ray_cfg, epoch=epoch, calibrated=False))


def identity_values(a, r, entry, base_name, base_ident):
    """The pilot.runs.IDENTITY values of a command: parsed args `a`, resolve(a) `r`, the verified split `entry` and the
    base identity. Paths enter only as their verified digests, aliases (--gpu) only as their resolved value."""
    return {"split": os.path.splitext(os.path.basename(entry["path"]))[0],
            "data_sha256": entry["sha256"], "data_role": entry["role"],
            "base_model_sha256": base_ident["model_sha256"],
            "base_config_sha256": (pins.LAYA["sha256"]["rl_agent_config.json"] if base_name == "multilingual"
                                   else base_ident["config_sha256"]),
            "laya_version": la.LAYA_VERSION,
            "device": r["device"], "amp": r["amp"], "grad_checkpointing": r["grad_checkpointing"],
            "neg_seed": r["neg_seed"], "adamw_foreach": r["adamw_foreach"], "adamw_fused": r["adamw_fused"],
            "weight_decay": WEIGHT_DECAY, "clip_grad_norm": CLIP_GRAD_NORM,
            "workers": a.workers, "epochs": a.epochs, "micro_batch": a.micro_batch, "grad_accum": a.grad_accum,
            "encoder_lr": a.encoder_lr, "head_lr": a.head_lr, "loss": a.loss, "neg": a.neg,
            "freeze_encoder": a.freeze_encoder, "freeze_embeddings": a.freeze_embeddings, "max_len": a.max_len,
            "head_max_len": a.head_max_len, "max_messages": a.max_messages,
            "calib_max_messages": a.calib_max_messages, "calib_frac": a.calib_frac,
            "shuffle_options": a.shuffle_options, "seed": a.seed}


def holdout_messages(rows, calib_max_messages, calib_frac, seed):
    """(train_rows, calib_rows): whole messages held out for temperature calibration.

    Holding out (message, question) items instead would leave a calibration message's other 47 questions in
    training, so the encoder would have trained on the exact text it is calibrated on. laya's split_calibration is
    generic over a sequence and depends only on the seed and the length, so every rank holds out the same messages.
    """
    return la.split_calibration(rows, calib_max_messages, calib_frac, seed)


def worker_device(c):
    """(torch device, env seen) of this worker: explicit mps (Ray has no MPS device manager), else Ray's device."""
    # a local torch: cloudpickle cannot ship torch.backends.mps when it is reached through the module-global torch
    import torch

    if c["device"] == "mps":
        env_seen = check_env(c["expect_env"])               # the spike's env must have reached this process
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available in this Ray worker; run --device mps on Apple silicon with the "
                             "locked torch build, or use --device cpu")
        return torch.device("mps"), env_seen
    device = ray.train.torch.get_device()                   # cpu, or cuda:<local rank> with use_gpu=True
    if device.type != c["device"]:
        raise SystemExit("Ray gave this worker device %s, expected %s; check the cluster's GPUs or use --device cpu"
                         % (device.type, c["device"]))
    return device, {}


def memory_sampler(device):
    """(reset, sync, sample, current, peak) for per-epoch memory peaks, sampled like the Phase 2 spike (torch.mps keeps
    no max_memory_allocated).

    peak["driver"]: torch.mps.driver_allocated_memory() on mps, torch.cuda.max_memory_allocated() on cuda, None on cpu
    (never 0); peak["rss"]: the worker's resident set; peak["total"]: driver + rss (the spike's peak_memory_bytes
    basis), so the peak RSS on cpu.
    """
    proc, kind, peak = psutil.Process(), device.type, {}

    def reset():
        peak.update(total=0, driver=None, rss=0)
        if kind == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if kind == "mps":
            torch.mps.synchronize()
        elif kind == "cuda":
            torch.cuda.synchronize(device)

    def sample():
        driver = (torch.mps.driver_allocated_memory() if kind == "mps"
                  else torch.cuda.max_memory_allocated(device) if kind == "cuda" else None)
        rss = proc.memory_info().rss
        peak["total"], peak["rss"] = max(peak["total"], (driver or 0) + rss), max(peak["rss"], rss)
        if driver is not None:
            peak["driver"] = driver if peak["driver"] is None else max(peak["driver"], driver)

    def current():
        return (torch.mps.current_allocated_memory() if kind == "mps"
                else torch.cuda.memory_allocated(device) if kind == "cuda" else None)

    reset()
    return reset, sync, sample, current, peak


def _rewrite_config(path, cfg):
    """Rewrite a just-saved checkpoint's config in laya's own layout (json indent 2), e.g. to add its save time."""
    tmp = os.path.join(path, "rl_agent_config.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(path, "rl_agent_config.json"))


def train_loop_per_worker(c):
    require_driver_node(ray.get_runtime_context().get_node_id(), c["driver_node_id"])
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    if rank == 0:
        print("run_id %s" % c["run_id"], flush=True)
    device, env_seen = worker_device(c)
    torch.set_num_threads(c["cpus_per_worker"])
    torch.manual_seed(c["seed"])

    # the base must still be the bytes the driver recorded in provenance.base (load_checkpoint verifies nothing)
    t_load = time.perf_counter()
    verify_files(c["base_dir"], c["base_sha256"], "base checkpoint")
    model, tok, cfg = la.load_checkpoint(c["base_dir"])
    load_s = time.perf_counter() - t_load
    max_len = c["max_len"] or cfg.get("max_len", 512)
    head_max_len = c["head_max_len"] or cfg.get("head_max_len", 192)
    parallel = la.uses_parallel_layout(cfg)

    # Every rank builds the same items and the same message-level calibration split (fixed seed),
    # then takes its own shard of the training items.
    rows = read_verified_rows(c["data"], c["data_sha256"])
    if c["max_messages"]:
        random.Random(c["seed"]).shuffle(rows)
        rows = rows[:c["max_messages"]]
    train_rows, calib_rows = holdout_messages(rows, c["calib_max_messages"], c["calib_frac"], c["seed"])
    # neg-K sampling on the training messages only, after the holdout: calibration keeps all 48 labels (a neg8
    # calibration set would carry about 22% positives against about 5% on val and test)
    counted = runs.neg_items(train_rows, c["neg"])
    if c["neg"]:
        train_rows = spike.neg8_rows(train_rows, c["neg_seed"], c["neg"])
    train_items, skipped = la.items_from_rows(tok, train_rows, max_len, head_max_len)
    calib_items, calib_skipped = la.items_from_rows(tok, calib_rows, max_len, head_max_len)
    if not calib_items:                                     # same rows on every rank, so every rank stops here
        raise SystemExit("no calibration items from %d messages (--calib-frac %s); use more messages"
                         % (len(rows), c["calib_frac"]))
    calib_uids = "\n".join(r.get("uid", "") for r in calib_rows).encode("utf-8")
    holdout = {"unit": "message", "n_messages": len(calib_rows), "n_items": len(calib_items),
               "uids_sha256": hashlib.sha256(calib_uids).hexdigest()}
    # driver-local absolute paths (data, base_dir), the run's node id and the raw env dict stay out of every
    # checkpoint (the env is recorded only as KEY=VALUE strings in the run record)
    ray_cfg = dict({k: v for k, v in c.items() if k not in ("data", "base_dir", "driver_node_id", "expect_env")},
                   calibration_holdout=holdout)
    per_rank = len(train_items) // world                    # equal shards keep DDP collectives in step
    if per_rank == 0:                                       # same on every rank: never publish an untrained "fine-tune"
        raise SystemExit("no training items per rank (%d items, %d workers); use more messages or fewer workers"
                         % (len(train_items), world))
    shard = train_items[rank * per_rank:(rank + 1) * per_rank]
    if rank == 0:
        print("messages %d -> train %d, calibration %d | neg %s: items train %d (counted %d, skipped %r, %d per rank "
              "x %d), calibration %d (skipped %r)"
              % (len(rows), len(train_rows), len(calib_rows), c["neg"] or "all", len(train_items), counted, skipped,
                 per_rank, world, len(calib_items), calib_skipped), flush=True)

    if c["freeze_encoder"]:
        la.freeze_encoder(model)
    elif c["grad_checkpointing"] and not la.enable_gradient_checkpointing(model):
        raise SystemExit("encoder has no gradient checkpointing; pass --no-grad-checkpointing to train without it")
    if c["freeze_embeddings"]:                              # rung embeddings_frozen: after checkpointing, as the spike
        la.freeze_embeddings(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print("rung %s%s | trainable params %d of %d" % (c["rung"], " (head-only, weaker)" if runs.weaker(c["rung"])
                                                          else "", trainable_params, total_params), flush=True)
    amp = c["amp"]                                          # decoupled from checkpointing (02-REVIEW CR-01)
    model = ray.train.torch.prepare_model(model, move_to_device=device,
                                          parallel_strategy_kwargs={"find_unused_parameters": True})
    gc.collect()                                            # drop the CPU copy before any memory sampling
    require_param_device(model, device.type)
    inner = model.module if hasattr(model, "module") else model
    inner.train()
    if c["freeze_encoder"]:
        inner.encoder.eval()

    groups = [{"params": [p for n, p in inner.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad], "lr": c["head_lr"]}]
    if not c["freeze_encoder"]:
        groups.insert(0, {"params": [p for n, p in inner.named_parameters()
                                     if n.startswith("encoder.") and p.requires_grad], "lr": c["encoder_lr"]})
    # foreach/fused only where resolve() pins them (mps: the single-tensor loop the spike measured)
    impl = {k: c["adamw_" + k] for k in ("foreach", "fused") if c["adamw_" + k] is not None}
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY, **impl)
    mb, accum = c["micro_batch"], c["grad_accum"]
    steps_per_epoch = -(-len(shard) // mb)
    updates = c["epochs"] * -(-steps_per_epoch // accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    params = [p for g in groups for p in g["params"]]
    rng = random.Random(c["seed"] + rank)
    shuffle = ("choice",) if c["shuffle_options"] else ()

    mem_reset, mem_sync, mem_sample, mem_current, peak = memory_sampler(device)
    rows_out = []
    record = {"device": device.type,
              "param_device": ",".join(sorted({p.device.type for p in inner.parameters()})),
              "torch_threads": torch.get_num_threads(),
              "env": runs.env_strings(env_seen) if device.type == "mps" else [],
              "grad_checkpointing": c["grad_checkpointing"], "amp": amp,
              "adamw_foreach": c["adamw_foreach"], "adamw_fused": c["adamw_fused"],
              "recommended_max_memory_bytes": torch.mps.recommended_max_memory() if device.type == "mps" else None,
              "rung": c["rung"], "fallback": c["rung"] != "none", "weaker": runs.weaker(c["rung"]),
              "spike_departures": runs.spike_departures(c, max_len),
              "trainable_params": trainable_params, "total_params": total_params,
              "neg": c["neg"], "neg_seed": c["neg_seed"], "n_train_messages": len(train_rows),
              "n_train_items": len(train_items), "n_train_items_counted": counted, "items_skipped": skipped,
              "load_s": load_s, "calib_s": None, "final_save_s": None, "run_id": c["run_id"],
              "canonical": c["canonical"], "epochs": rows_out}
    for epoch in range(c["epochs"]):
        t_epoch = time.time()
        order = list(shard)
        random.Random(c["seed"] + 1000 * epoch + rank).shuffle(order)
        sigma = la.sigma_at(epoch, c["epochs"], 0.4, 0.1)
        total, n, steps = 0.0, 0, 0
        mem_reset()
        mem_sync()
        swap0, t0 = spike.swap_used(), time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(order), mb):
            batch = la.collate_items([[la.encode_item(tok, it, max_len, head_max_len,
                                                      la.draw_option_order(it, rng, shuffle), parallel)
                                       for it in order[s:s + mb]]], tok.pad_token_id)
            step_now = (n + 1) % accum == 0 or s + mb >= len(order)
            # skip the gradient all-reduce on accumulation steps
            sync = nullcontext() if step_now or not hasattr(model, "no_sync") else model.no_sync()
            with mps_oom_exit(epoch + 1, n + 1, c["rung"]), sync:
                logits = la.forward(model, batch, device, amp, c["freeze_encoder"])
                mask, target = batch["marker_mask"].to(device), batch["target"].to(device)
                if c["loss"] == "rlcd":
                    loss = la.rlcd_loss(logits, target, mask, batch["qtype"].to(device), sigma)
                else:
                    loss = la.soft_ce_loss(logits, target, mask)
                (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            mem_sample()
            n += 1
            if step_now:
                with mps_oom_exit(epoch + 1, n, c["rung"]):   # the step closes micro-batch n
                    if scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, CLIP_GRAD_NORM)
                    (scaler.step(opt), scaler.update()) if scaler else opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                steps += 1
                mem_sample()
            total += loss.item()
        mem_sync()
        train_s, swap_delta = time.perf_counter() - t0, spike.swap_used() - swap0
        row = dict(zip(runs.EPOCH_FIELDS, (epoch + 1, total / max(1, n), train_s, 0.0, steps, len(shard),
                                           peak["total"], peak["driver"], peak["rss"], mem_current(),
                                           swap_delta)))
        rows_out.append(row)
        metrics = {"epoch": epoch + 1, "loss": row["loss"], "epoch_seconds": round(time.time() - t_epoch, 1),
                   "train_s": train_s}
        if rank == 0:
            print("epoch %d/%d loss %.4f train %.1fs peak %.2f GiB swap %+.0f MiB"
                  % (epoch + 1, c["epochs"], row["loss"], train_s, row["peak_total_bytes"] / 2**30,
                     swap_delta / 2**20), flush=True)
        # every rank reports; only rank 0 attaches a checkpoint
        if rank == 0 and epoch < c["epochs"] - 1:
            ecfg = epoch_config(cfg, max_len, head_max_len, dict(ray_cfg, record=record), epoch + 1)
            with tempfile.TemporaryDirectory() as d:
                t_save = time.perf_counter()
                la.save_checkpoint(inner, tok, ecfg, d)
                row["save_s"] = time.perf_counter() - t_save   # ecfg shares this row
                _rewrite_config(d, ecfg)
                ray.train.report(metrics, checkpoint=Checkpoint.from_directory(d))
        elif epoch < c["epochs"] - 1:
            ray.train.report(metrics)
        last = metrics

    # Calibrate on the held-out messages and publish the final checkpoint from rank 0.
    del opt, sched, scaler, params, groups                 # free optimizer state before the eval pass
    gc.collect()
    if rank == 0:
        inner.eval()
        t_calib = time.perf_counter()
        records = la.calibration_records(inner, tok, calib_items, device, max_len, head_max_len,
                                         batch_size=c["calib_batch"], parallel=parallel)
        fitted = la.fit_temperature_map(records)
        record["calib_s"] = time.perf_counter() - t_calib
        calibration = la.calibration_report(records, fitted["temperature"])
        out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                       temperature=fitted["temperature"])
        out_cfg.pop("temperature_by_options", None)       # inherited buckets would mask the new fit
        if fitted["temperature_by_options"]:
            out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
        out_cfg["training"] = dict(out_cfg.get("training") or {}, ray_train=dict(ray_cfg, record=record),
                                   epoch=c["epochs"], calibrated=True, laya_train_calibration=calibration)
        with tempfile.TemporaryDirectory() as d:
            t_save = time.perf_counter()
            la.save_checkpoint(inner, tok, out_cfg, d)
            record["final_save_s"] = time.perf_counter() - t_save
            _rewrite_config(d, out_cfg)                      # the record now holds the final save time
            ray.train.report(dict(last, temperature=fitted["temperature"]), checkpoint=Checkpoint.from_directory(d))
    else:
        ray.train.report(last)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="multilingual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                    help="training device (default cpu); mps trains in one Ray worker on the Apple GPU")
    ap.add_argument("--gpu", action="store_true", help="alias for --device cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                    help="CUDA mixed precision (default: on for cuda, off elsewhere)")
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="gradient checkpointing on the encoder and the Laya head for full-encoder runs (default on)")
    ap.add_argument("--cpus-per-worker", type=int, default=None, help="CPUs and torch threads per worker "
                    "(default 4 on mps, the spike's timing basis; 1 otherwise)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--encoder-lr", type=float, default=2.5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--loss", choices=("rlcd", "soft-ce"), default="rlcd")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="fallback rung 3: train the head only (labelled head-only, weaker)")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="fallback rung 1: freeze only the token embeddings")
    ap.add_argument("--shuffle-options", action="store_true")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--max-messages", type=int, default=None, help="subsample messages (smoke tests)")
    ap.add_argument("--neg", type=int, default=8,
                    help="negatives kept per training message; 0 = all 48 labels (data_full)")
    ap.add_argument("--neg-seed", type=int, default=None, help="negative-sampling seed; default --seed")
    ap.add_argument("--calib-max-messages", type=int, default=40,
                    help="messages held out (all their questions) for temperature calibration")
    ap.add_argument("--calib-frac", type=float, default=0.1, help="cap on the held-out share of messages")
    ap.add_argument("--calib-batch", type=int, default=16)
    ap.add_argument("--object-store-gb", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--storage", default=os.path.abspath("ray_results"),
                    help="Ray run storage (local dir; the script is single-node)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify --data against (default: the committed data/splits.json)")
    ap.add_argument("--print-run-id", action="store_true",
                    help="print the run id this command trains under (its Ray dir and resume key), then exit without "
                         "starting Ray")
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.epochs < 1:
        ap.error("--epochs must be at least 1")
    if a.neg < 0:
        ap.error("--neg must be 0 (all labels) or a positive count")
    if a.freeze_embeddings and a.freeze_encoder:
        ap.error("--freeze-encoder already freezes the embeddings; pass one of them")
    try:
        r = resolve(a)
    except ValueError as e:
        ap.error(str(e))
    if r["device"] == "mps" and a.workers != 1:
        ap.error("--device mps trains on one Ray worker (one GPU, no MPS collective backend); use --workers 1")
    if a.amp and r["device"] in ("mps", "cpu"):
        ap.error("--amp is CUDA-only (the MPS fine-tune is measured with AMP off); drop --amp")
    entry = verify_split_file(a.data, manifest_path=a.manifest, roles=("train",))   # before ray.init (D-03)
    if not a.print_run_id and os.path.exists(a.out) and (not os.path.isdir(a.out) or os.listdir(a.out)):
        # copying over old files could leave a mixed checkpoint that laya.load still accepts
        raise SystemExit("--out %s exists and is not empty; choose a new dir" % a.out)
    # the device must exist here before any pin check, fetch or ray.init (LAYA-02)
    if r["device"] == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps, but MPS is not available here; use --device cpu, or run on Apple silicon with "
                         "the locked torch build")
    if r["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda (or --gpu), but CUDA is not available here; use --device cpu or mps")
    la.check_version()                                      # the adapter's laya pin, before the fetch and ray.init
    if a.base == "multilingual":                            # download once on the driver, pinned + verified (ENV-02)
        base_dir = fetch(pins.LAYA)
    elif os.path.isfile(os.path.join(a.base, "rl_agent_config.json")):
        base_dir = os.path.abspath(a.base)
    else:
        raise SystemExit("--base must be 'multilingual' (pinned) or a local Laya checkpoint dir: %s" % a.base)
    base_name = "multilingual" if a.base == "multilingual" else base_dir
    base_ident = identity(base_name)
    base_sha256 = checkpoint_digests(base_name, base_ident)   # workers verify these bytes before loading
    canon = runs.canonical(identity_values(a, r, entry, base_name, base_ident))
    rid = runs.run_id(canon)                                # the Ray experiment name: storage/<rid> is the resume dir
    if a.print_run_id:
        print(rid)
        return

    device, cpus = r["device"], r["cpus_per_worker"]
    object_store = int(a.object_store_gb * 1024**3) if a.object_store_gb else None
    env = spike.spike_env("0", cpus) if device == "mps" else {}
    if device == "mps":
        # the spike's worker env (strict fallback, HIGH and LOW watermarks 1.0); one spare CPU for the Train controller
        ray.init(runtime_env={"env_vars": env}, num_cpus=cpus + 1, ignore_reinit_error=True, include_dashboard=False,
                 object_store_memory=object_store)
    else:
        ray.init(ignore_reinit_error=True, include_dashboard=False, object_store_memory=object_store)
    require_single_node(ray.nodes())
    cfg = dict({"data": os.path.abspath(a.data), "data_sha256": entry["sha256"], "base_dir": base_dir,
                "driver_node_id": ray.get_runtime_context().get_node_id(),
                "base_sha256": base_sha256, "epochs": a.epochs,
                "micro_batch": a.micro_batch, "grad_accum": a.grad_accum, "encoder_lr": a.encoder_lr,
                "head_lr": a.head_lr, "loss": a.loss, "freeze_encoder": a.freeze_encoder,
                "freeze_embeddings": a.freeze_embeddings,
                "shuffle_options": a.shuffle_options, "max_len": a.max_len, "head_max_len": a.head_max_len,
                "max_messages": a.max_messages, "neg": a.neg, "calib_max_messages": a.calib_max_messages,
                "calib_frac": a.calib_frac, "calib_batch": a.calib_batch,
                "seed": a.seed, "expect_env": env, "run_id": rid, "canonical": canon,
                "provenance": provenance(entry, a.manifest, a.base, base_dir, base_ident)}, **r)
    cfg["rung"] = runs.rung_label(cfg)                      # derived from the flags, into the record and the config
    trainer = TorchTrainer(
        train_loop_per_worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=a.workers, use_gpu=(device == "cuda"),
                                     resources_per_worker={"CPU": cpus}),
        run_config=RunConfig(name=rid, storage_path=a.storage))
    t0 = time.time()
    result = fit_or_exit(trainer)
    print("metrics:", json.dumps(result.metrics, default=str))
    if os.path.isdir(a.out) and not os.listdir(a.out):
        os.rmdir(a.out)                                     # an empty dir checked before training
    with result.checkpoint.as_directory() as d:
        shutil.copytree(d, a.out)                           # never merges: fails if --out was filled meanwhile
    print("checkpoint -> %s  (%.0fs)  load with laya.load(%r)" % (a.out, time.time() - t0, a.out))


if __name__ == "__main__":
    main()
