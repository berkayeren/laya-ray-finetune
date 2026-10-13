"""Distributed fine-tuning of a Laya checkpoint with Ray Train (PyTorch DDP).

`laya.train.finetune` / `laya-train` run on one device. This script keeps Laya's
own building blocks -- item encoding, RLCD / soft-CE loss, temperature
calibration and the checkpoint layout `laya.load` reads -- and runs the training
loop under Ray's TorchTrainer, so it scales from 1 CPU worker to N GPUs on one
node with only --workers / --gpu changed. It is single-node: workers read the
base weights and the split from driver-local paths, so it refuses a multi-node
Ray cluster.

Per worker:  load base -> build items -> shard by rank -> DDP train
Rank 0:      fit calibration temperatures on a slice held out before training,
             save the final checkpoint and report it to Ray.

--data must be a split registered in data/splits.json with role train and a matching sha256; anything else
(tampered, unregistered, val/test/probe) is refused before Ray starts. The split files label all 48 tags;
per-seed negative sampling arrives with the Phase 4 training harness. --base multilingual fetches the pinned,
sha256-verified Laya weights (pilot.weights) before Ray starts; any other value must be a local checkpoint dir.
--manifest selects the prepare-written splits.json to verify against (default: the committed data/splits.json; use
OTHER_DIR/splits.json for a `prepare_data.py --out OTHER_DIR` variant). Verification itself cannot be skipped.

usage:
  python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
                      --workers 4 --gpu            # 4 GPUs on one node
  python ray_train.py ... --workers 2 --freeze-encoder --max-messages 100   # CPU smoke test
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from contextlib import nullcontext

import ray
import ray.train
import torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from laya import train as lt
from laya.calibrate import fit_temperature_map
from laya.common import collate_items, uses_parallel_layout

from pilot import pins
from pilot.data import DEFAULT_MANIFEST, verify_split_file
from pilot.weights import fetch


def read_verified_rows(path, sha256):
    """Rows of a JSONL split whose bytes must hash to `sha256` (the driver-verified digest); parses those same bytes.

    Workers re-check because they read the file minutes after the driver verified it (fetch, ray.init).
    Line handling matches laya.train.read_jsonl (strip; skip blank and '#' lines).
    """
    with open(path, "rb") as f:
        raw = f.read()
    have = hashlib.sha256(raw).hexdigest()
    if have != sha256:
        raise SystemExit("%s sha256 %s != %s verified on the driver; regenerate with `make data`"
                         % (path, have, sha256))
    lines = (l.strip() for l in raw.decode("utf-8").splitlines())
    return [json.loads(l) for l in lines if l and not l.startswith("#")]


def require_single_node(nodes):
    """Refuse a multi-node cluster: base_dir and the split path are the driver's local paths (no copy elsewhere)."""
    alive = [n for n in nodes if n.get("Alive")]
    if len(alive) > 1:
        raise SystemExit("ray_train.py is single-node: workers read the driver-local base weights and split, "
                         "but the Ray cluster has %d alive nodes" % len(alive))


def train_loop_per_worker(c):
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    device = ray.train.torch.get_device()
    torch.manual_seed(c["seed"])

    model, tok, cfg = lt.load_checkpoint(c["base_dir"])
    max_len = c["max_len"] or cfg.get("max_len", 512)
    head_max_len = c["head_max_len"] or cfg.get("head_max_len", 192)
    parallel = uses_parallel_layout(cfg)

    # Every rank builds the same items and the same calibration split (fixed seed),
    # then takes its own shard of the training items.
    rows = read_verified_rows(c["data"], c["data_sha256"])
    if c["max_messages"]:
        random.Random(c["seed"]).shuffle(rows)
        rows = rows[:c["max_messages"]]
    items, skipped = lt.items_from_rows(tok, rows, max_len, head_max_len)
    train_items, calib_items = lt.split_calibration(items, c["calib_max"], c["calib_frac"], c["seed"])
    per_rank = len(train_items) // world                    # equal shards keep DDP collectives in step
    shard = train_items[rank * per_rank:(rank + 1) * per_rank]
    if rank == 0:
        print("items %d (skipped %r) -> train %d (%d per rank x %d), calibration %d"
              % (len(items), skipped, len(train_items), per_rank, world, len(calib_items)), flush=True)

    amp = device.type == "cuda"
    if c["freeze_encoder"]:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif amp and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model = ray.train.torch.prepare_model(model, parallel_strategy_kwargs={"find_unused_parameters": True})
    inner = model.module if hasattr(model, "module") else model
    inner.train()
    if c["freeze_encoder"]:
        inner.encoder.eval()

    groups = [{"params": [p for n, p in inner.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad], "lr": c["head_lr"]}]
    if not c["freeze_encoder"]:
        groups.insert(0, {"params": [p for n, p in inner.named_parameters()
                                     if n.startswith("encoder.") and p.requires_grad], "lr": c["encoder_lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    mb, accum = c["micro_batch"], c["grad_accum"]
    steps_per_epoch = -(-len(shard) // mb)
    updates = c["epochs"] * -(-steps_per_epoch // accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    params = [p for g in groups for p in g["params"]]
    rng = random.Random(c["seed"] + rank)
    shuffle = ("choice",) if c["shuffle_options"] else ()

    for epoch in range(c["epochs"]):
        t0 = time.time()
        order = list(shard)
        random.Random(c["seed"] + 1000 * epoch + rank).shuffle(order)
        sigma = lt.sigma_at(epoch, c["epochs"], 0.4, 0.1)
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(order), mb):
            batch = collate_items([[lt.encode_item(tok, it, max_len, head_max_len,
                                                   lt.draw_option_order(it, rng, shuffle), parallel)
                                    for it in order[s:s + mb]]], tok.pad_token_id)
            step_now = (n + 1) % accum == 0 or s + mb >= len(order)
            # skip the gradient all-reduce on accumulation steps
            sync = nullcontext() if step_now or not hasattr(model, "no_sync") else model.no_sync()
            with sync:
                logits = lt._forward(model, batch, device, amp, c["freeze_encoder"])
                mask, target = batch["marker_mask"].to(device), batch["target"].to(device)
                if c["loss"] == "rlcd":
                    loss = lt.rlcd_loss(logits, target, mask, batch["qtype"].to(device), sigma, 4, 0.75, 1.0)
                else:
                    loss = lt.soft_ce_loss(logits, target, mask)
                (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            n += 1
            if step_now:
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                (scaler.step(opt), scaler.update()) if scaler else opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            total += loss.item()
        metrics = {"epoch": epoch + 1, "loss": total / max(1, n), "epoch_seconds": round(time.time() - t0, 1)}
        # every rank reports; only rank 0 attaches a checkpoint
        if rank == 0 and epoch < c["epochs"] - 1:
            with tempfile.TemporaryDirectory() as d:
                lt.save_checkpoint(inner, tok, dict(cfg, max_len=max_len, head_max_len=head_max_len), d)
                ray.train.report(metrics, checkpoint=Checkpoint.from_directory(d))
        elif epoch < c["epochs"] - 1:
            ray.train.report(metrics)
        last = metrics

    # Calibrate on the held-out slice and publish the final checkpoint from rank 0.
    del opt, sched, scaler, params, groups                 # free optimizer state before the eval pass
    import gc; gc.collect()
    if rank == 0:
        inner.eval()
        records = lt.calibration_records(inner, tok, calib_items, device, max_len, head_max_len,
                                         batch_size=c["calib_batch"], parallel=parallel)
        fitted = fit_temperature_map(records)
        calibration = lt.calibration_report(records, fitted["temperature"])
        out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                       temperature=fitted["temperature"])
        out_cfg.pop("temperature_by_options", None)       # inherited buckets would mask the new fit
        if fitted["temperature_by_options"]:
            out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
        out_cfg["training"] = dict(out_cfg.get("training") or {}, ray_train=dict(c, base_dir=None),
                                   laya_train_calibration=calibration)
        with tempfile.TemporaryDirectory() as d:
            lt.save_checkpoint(inner, tok, out_cfg, d)
            ray.train.report(dict(last, temperature=fitted["temperature"]), checkpoint=Checkpoint.from_directory(d))
    else:
        ray.train.report(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="multilingual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--cpus-per-worker", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--encoder-lr", type=float, default=2.5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--loss", choices=("rlcd", "soft-ce"), default="rlcd")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--shuffle-options", action="store_true")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--max-messages", type=int, default=None, help="subsample messages (smoke tests)")
    ap.add_argument("--calib-max", type=int, default=400)
    ap.add_argument("--calib-frac", type=float, default=0.1)
    ap.add_argument("--calib-batch", type=int, default=16)
    ap.add_argument("--object-store-gb", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--storage", default=os.path.abspath("ray_results"),
                    help="Ray run storage (local dir; the script is single-node)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify --data against (default: the committed data/splits.json)")
    a = ap.parse_args()
    entry = verify_split_file(a.data, manifest_path=a.manifest, roles=("train",))   # before ray.init (D-03)
    if a.base == "multilingual":                            # download once on the driver, pinned + verified (ENV-02)
        base_dir = fetch(pins.LAYA)
    elif os.path.isfile(os.path.join(a.base, "rl_agent_config.json")):
        base_dir = os.path.abspath(a.base)
    else:
        raise SystemExit("--base must be 'multilingual' (pinned) or a local Laya checkpoint dir: %s" % a.base)

    ray.init(ignore_reinit_error=True, include_dashboard=False,
             object_store_memory=int(a.object_store_gb * 1024**3) if a.object_store_gb else None)
    require_single_node(ray.nodes())
    cfg = {"data": os.path.abspath(a.data), "data_sha256": entry["sha256"], "base_dir": base_dir, "epochs": a.epochs,
           "micro_batch": a.micro_batch, "grad_accum": a.grad_accum, "encoder_lr": a.encoder_lr,
           "head_lr": a.head_lr, "loss": a.loss, "freeze_encoder": a.freeze_encoder,
           "shuffle_options": a.shuffle_options, "max_len": a.max_len, "head_max_len": a.head_max_len,
           "max_messages": a.max_messages, "calib_max": a.calib_max, "calib_frac": a.calib_frac, "calib_batch": a.calib_batch,
           "seed": a.seed}
    trainer = TorchTrainer(
        train_loop_per_worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=a.workers, use_gpu=a.gpu,
                                     resources_per_worker={"CPU": a.cpus_per_worker}),
        run_config=RunConfig(name="laya-multi3nlu", storage_path=a.storage))
    t0 = time.time()
    result = trainer.fit()
    print("metrics:", json.dumps(result.metrics, default=str))
    with result.checkpoint.as_directory() as d:
        shutil.copytree(d, a.out, dirs_exist_ok=True)
    print("checkpoint -> %s  (%.0fs)  load with laya.load(%r)" % (a.out, time.time() - t0, a.out))


if __name__ == "__main__":
    main()
