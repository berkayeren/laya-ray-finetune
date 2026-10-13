"""Phase 2 MPS feasibility spike (LAYA-09): does a full-encoder Laya fine-tune fit on this Mac's MPS device?

Design: the orchestrator (this script without --child) imports no torch. It verifies the selection_train split
against data/splits.json, fetches the pinned Laya weights, and then starts one fresh subprocess per run
(`--child ray CONFIG`), because torch reads PYTORCH_ENABLE_MPS_FALLBACK once at import and the MPS allocator keeps
a process-wide high-water mark. The child starts Ray with the spike env in its runtime_env and runs a 1-worker
TorchTrainer with use_gpu False (Ray has no MPS device manager). The worker picks torch.device("mps") itself, moves
the model with prepare_model, refuses to measure unless every parameter landed there, enables gradient checkpointing
on the encoder and the Laya head with AMP off, and times optimizer steps for batches padded to exactly max_len and
for real neg8 batches while sampling MPS driver memory + RSS and system swap. It writes run.json into a Ray
Checkpoint (Ray Train V2 drops metrics reported without one); the child reads it back and the orchestrator writes
mps_spike.json + mps_spike.md.

Laya internals used (the Phase 3 adapter list): lt.load_checkpoint, lt.items_from_rows, lt.encode_item, lt._forward,
lt.rlcd_loss, laya.common.collate_items, laya.common.uses_parallel_layout, laya.common.DecisionModel,
DecisionModel.head_checkpointing, encoder.gradient_checkpointing_enable, encoder.embeddings.tok_embeddings.

usage:
  python mps_spike.py --out results/spike
  python mps_spike.py --grid 4x128 --warmup-steps 1 --measured-steps 2 --effective-batch 8 --out /tmp/spike  # smoke
"""
import argparse
import gc
import importlib.metadata
import itertools
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time

from pilot import pins, spike
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT, repo_relpath, sha256_file, verify_split_file, write_atomic
from pilot.weights import fetch

SPLIT = os.path.join(REPO_ROOT, "data", "splits", "selection_train.jsonl")
# ray_train.py defaults. foreach AdamW kernels do not exist on MPS and fused is opt-in, so AdamW runs its
# single-tensor loop on every device (a fused A/B is deferred).
HPARAMS = {"encoder_lr": 2.5e-5, "head_lr": 1e-4, "weight_decay": 0.01, "clip_grad_norm": 1.0, "loss": "rlcd",
           "sigma": 0.4, "adamw_foreach": False, "adamw_fused": False}
LAYA_INTERNALS = ["lt.load_checkpoint", "lt.items_from_rows", "lt.encode_item", "lt._forward", "lt.rlcd_loss",
                  "laya.common.collate_items", "laya.common.uses_parallel_layout", "laya.common.DecisionModel",
                  "DecisionModel.head_checkpointing", "encoder.gradient_checkpointing_enable",
                  "encoder.embeddings.tok_embeddings"]
VERSIONED = ("torch", "ray", "laya", "transformers", "psutil")


def check_env(expected):
    """{name: value seen} for the expected env vars; refuses on the first one the worker did not get."""
    seen = {k: os.environ.get(k) for k in expected}
    for k in sorted(expected):
        if seen[k] != expected[k]:
            raise SystemExit("worker env %s=%r, expected %r; env vars must reach the worker through "
                             "ray.init(runtime_env)" % (k, seen[k], expected[k]))
    return seen


def configure_model(model, rung):
    """Gradient checkpointing on (encoder and head, whatever the device) and the D-08 rung; returns trainable params.

    A frozen encoder never counts as FITS (D-04), so the spike refuses to measure any rung but these two.
    """
    if rung not in ("none", "embeddings_frozen"):
        raise SystemExit("rung %r is not measured; use 'none' or 'embeddings_frozen' (a frozen encoder never FITS)"
                         % (rung,))
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    if rung == "embeddings_frozen":
        model.encoder.embeddings.tok_embeddings.weight.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def require_param_device(model, device_type):
    """Refuse unless every model parameter sits on `device_type` (LAYA-09 criterion 1; no silent CPU timing)."""
    have = {p.device.type for p in model.parameters()}
    if have != {device_type}:
        raise SystemExit("model parameters on %s, expected %s; the spike refuses to time a silent CPU run"
                         % (", ".join(sorted(have)), device_type))


def _pinned_source(c, max_len):
    """Pinned Laya model + two batch makers (real neg8, padded to max_len) over the verified selection_train split."""
    from laya import train as lt
    from laya.common import collate_items, uses_parallel_layout
    from pilot.weights import verify_files
    from ray_train import read_verified_rows

    # the worker re-checks the bytes the driver verified minutes ago (T-02-01)
    verify_files(c["base_dir"], c["base_sha256"], "base checkpoint")
    model, tok, cfg = lt.load_checkpoint(c["base_dir"])
    head_max_len = cfg.get("head_max_len", 192)
    parallel = uses_parallel_layout(cfg)
    rows = read_verified_rows(c["data"], c["data_sha256"])
    items, skipped = lt.items_from_rows(tok, spike.neg8_rows(rows, c["seed"]), max_len, head_max_len)
    random.Random(c["seed"]).shuffle(items)

    def maker(pad):
        pool = itertools.cycle(items)

        def make():
            chunk = [next(pool) for _ in range(c["micro_batch"])]
            if pad:
                chunk = [spike.pad_item_to(it, max_len) for it in chunk]
            return collate_items([[lt.encode_item(tok, it, max_len, head_max_len, None, parallel) for it in chunk]],
                                 tok.pad_token_id)
        return make
    return model, maker(False), maker(True), len(items), skipped


def _measure(c):
    import psutil
    import torch
    from laya import train as lt

    if c["transport"] == "checkpoint":
        import ray
        from ray_train import require_driver_node
        require_driver_node(ray.get_runtime_context().get_node_id(), c["driver_node_id"])
    env_seen = check_env(c["expect_env"])
    device = torch.device(c["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this process; the spike cannot run")
    torch.manual_seed(c["seed"])
    torch.set_num_threads(c["cpus"])
    mb, max_len, accum, h = c["micro_batch"], c["max_len"], c["grad_accum"], c["hparams"]

    if c["source"] == "pinned":
        model, real_maker, pad_maker, n_items, skipped = _pinned_source(c, max_len)
    else:
        raise SystemExit("unknown model source %r" % (c["source"],))

    def padded_maker():
        batch = pad_maker()
        if tuple(batch["input_ids"].shape) != (mb, max_len) or not bool(batch["attention_mask"].all()):
            raise SystemExit("padded batch shape %s, expected (%d, %d) with a full attention mask"
                             % (tuple(batch["input_ids"].shape), mb, max_len))
        return batch

    trainable = configure_model(model, c["rung"])
    if c["transport"] == "checkpoint":
        import ray.train.torch
        model = ray.train.torch.prepare_model(model, move_to_device=device)
    else:
        model = model.to(device)
    gc.collect()                                            # drop the CPU copy before any sampling (Pitfall 8)
    require_param_device(model, device.type)
    model.train()

    named = list(model.named_parameters())
    groups = [{"params": [p for n, p in named if n.startswith("encoder.") and p.requires_grad], "lr": h["encoder_lr"]},
              {"params": [p for n, p in named if not n.startswith("encoder.") and p.requires_grad], "lr": h["head_lr"]}]
    opt = torch.optim.AdamW(groups, weight_decay=h["weight_decay"], foreach=h["adamw_foreach"],
                            fused=h["adamw_fused"])
    params = [p for g in groups for p in g["params"]]
    is_mps = device.type == "mps"
    proc = psutil.Process()
    peak = {"total": 0, "driver": 0, "rss": 0}

    def sync():
        if is_mps:
            torch.mps.synchronize()

    def sample():
        driver = torch.mps.driver_allocated_memory() if is_mps else 0
        rss = proc.memory_info().rss
        peak["total"], peak["driver"], peak["rss"] = (max(peak["total"], driver + rss), max(peak["driver"], driver),
                                                      max(peak["rss"], rss))

    last = {"loss": None, "shape": None}

    def step(maker, measured):
        # one optimizer step = accum micro-batches (encoded inside the clock, like the real loop) + clip + AdamW
        sync()
        t0 = time.perf_counter()
        for _ in range(accum):
            batch = maker()
            logits = lt._forward(model, batch, device, False, False)
            loss = lt.rlcd_loss(logits, batch["target"].to(device), batch["marker_mask"].to(device),
                                batch["qtype"].to(device), h["sigma"], 4, 0.75, 1.0)
            (loss / accum).backward()
            if measured:
                sample()
            last["loss"], last["shape"] = loss.detach(), list(batch["input_ids"].shape)
        torch.nn.utils.clip_grad_norm_(params, h["clip_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        dt = time.perf_counter() - t0
        if measured:
            sample()
        return dt

    regimes = {}
    for name, maker in (("padded", padded_maker), ("real", real_maker)):
        for _ in range(c["warmup_steps"]):
            step(maker, False)
        # swap brackets the measured steps only, never warm-up or the model load (Pitfall 8)
        s0, o0 = spike.swap_used(), spike.swapouts()
        times = [step(maker, True) for _ in range(c["measured_steps"])]
        s1, o1 = spike.swap_used(), spike.swapouts()
        regimes[name] = {"warmup_steps": c["warmup_steps"], "times": spike.summarize(times),
                         "batch_shape": last["shape"], "swap_used_before": s0, "swap_used_after": s1,
                         "swap_delta_bytes": s1 - s0,
                         "swapouts_delta": None if o0 is None or o1 is None else o1 - o0}
    return {"status": "ok", "purpose": c["purpose"], "micro_batch": mb, "max_len": max_len, "grad_accum": accum,
            "rung": c["rung"], "fallback": c["fallback"], "device": device.type,
            "param_device": ",".join(sorted({p.device.type for p in model.parameters()})),
            "ddp_wrapped": hasattr(model, "module"), "world_size": 1, "trainable_params": trainable,
            "total_params": sum(p.numel() for p in model.parameters()), "n_neg8_items": n_items,
            "items_skipped": skipped, "env_seen": env_seen, "torch_threads": torch.get_num_threads(),
            "regimes": regimes, "peak_memory_bytes": peak["total"], "peak_driver_allocated_bytes": peak["driver"],
            "peak_rss_bytes": peak["rss"],
            "recommended_max_memory_bytes": torch.mps.recommended_max_memory() if is_mps else None,
            "final_loss": None if last["loss"] is None else float(last["loss"])}


def spike_worker(c):
    """Ray Train worker loop (transport "checkpoint") and in-process runner (transport "return")."""
    try:
        payload = _measure(c)
    except SystemExit as e:
        payload = {"status": "refused", "reason": spike.scrub(str(e))}
    payload["transport"] = c["transport"]
    if c["transport"] != "checkpoint":
        return payload
    import ray.train
    from ray.train import Checkpoint
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        # Ray Train V2 leaves Result.metrics None without a checkpoint, so the payload travels inside one
        ray.train.report({"status": payload["status"]}, checkpoint=Checkpoint.from_directory(d))
    return None


def run_ray(cfg):
    """Start Ray with the spike env and run spike_worker in a 1-worker TorchTrainer; returns the worker's run.json."""
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray_train import require_single_node

    # one spare CPU for the Train controller
    ray.init(runtime_env={"env_vars": cfg["expect_env"]}, num_cpus=cfg["cpus"] + 1, include_dashboard=False)
    try:
        require_single_node(ray.nodes())
        trainer = TorchTrainer(
            spike_worker,
            train_loop_config=dict(cfg, transport="checkpoint", driver_node_id=ray.get_runtime_context().get_node_id()),
            scaling_config=ScalingConfig(num_workers=1, use_gpu=False, resources_per_worker={"CPU": cfg["cpus"]}),
            run_config=RunConfig(name=cfg["run_name"], storage_path=cfg["storage"]))
        result = trainer.fit()
        if result.checkpoint is None:
            raise SystemExit("the spike worker reported no checkpoint, so its run.json is lost")
        with result.checkpoint.as_directory() as d:
            with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
                return json.load(f)
    finally:
        ray.shutdown()


def run_child(mode, path):
    """--child entry point: run one config (ray | inprocess) and write its payload to cfg["out_json"]."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if mode == "ray":
        payload = run_ray(cfg)
    elif mode == "inprocess":
        import torch
        torch.set_num_threads(cfg["cpus"])
        payload = spike_worker(dict(cfg, transport="return"))
    else:
        raise SystemExit("--child mode must be 'ray' or 'inprocess', got %r" % (mode,))
    write_atomic(cfg["out_json"], (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def launch(cfg, mode):
    """Run one config in a fresh subprocess with the spike env; returns its attempt (see spike.child_outcome)."""
    d = tempfile.mkdtemp(prefix="mps-spike-")
    try:
        cfg = dict(cfg, out_json=os.path.join(d, "run.json"))
        path = os.path.join(d, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", mode, path],
                           env=dict(os.environ, **cfg["expect_env"]), cwd=REPO_ROOT, shell=False)
        payload = None
        if os.path.isfile(cfg["out_json"]):
            with open(cfg["out_json"], encoding="utf-8") as f:
                payload = json.load(f)
        return spike.child_outcome(r.returncode, payload)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="MPS feasibility spike for a full-encoder Laya fine-tune (LAYA-09)")
    ap.add_argument("--out", default="results/spike", help="report dir (mps_spike.json + mps_spike.md)")
    ap.add_argument("--grid", default="4x128,4x256,8x128,8x256", help="MICRO_BATCHxMAX_LEN entries, comma-separated")
    ap.add_argument("--warmup-steps", type=int, default=spike.WARMUP_STEPS)
    ap.add_argument("--measured-steps", type=int, default=spike.MEASURED_STEPS)
    ap.add_argument("--effective-batch", type=int, default=spike.EFFECTIVE_BATCH,
                    help="micro-batch x grad-accum, held fixed across the grid")
    ap.add_argument("--cpus", type=int, default=4,
                    help="OMP_NUM_THREADS, the worker's CPU resource and torch threads (the M4's performance cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--storage", default=os.path.abspath("ray_results"), help="Ray run storage (local dir)")
    ap.add_argument("--child", nargs=2, metavar=("MODE", "CONFIG"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        run_child(*a.child)
        return
    try:
        grid = spike.parse_grid(a.grid)
    except ValueError as e:
        ap.error("--grid: %s" % e)
    for mb, max_len in grid:
        if a.effective_batch % mb:
            ap.error("--effective-batch %d is not divisible by micro-batch %d" % (a.effective_batch, mb))
        if max_len < 64:
            ap.error("max_len %d is below 64" % max_len)
    if a.measured_steps < 1:
        ap.error("--measured-steps must be at least 1")
    if a.warmup_steps < 0:
        ap.error("--warmup-steps must not be negative")
    if a.cpus < 1:
        ap.error("--cpus must be at least 1")

    # D-05: the main checkout's locked venv, the hash-verified split and the pinned weights, before any child runs
    entry = verify_split_file(SPLIT, roles=("train",))
    base_dir = fetch(pins.LAYA)
    expect_env = spike.spike_env("0", a.cpus)
    configs = []
    for mb, max_len in grid:
        rung, fallback = "none", "0"
        cfg = {"source": "pinned", "device": "mps", "base_dir": base_dir, "base_sha256": dict(pins.LAYA["sha256"]),
               "data": SPLIT, "data_sha256": entry["sha256"], "micro_batch": mb, "max_len": max_len,
               "grad_accum": a.effective_batch // mb, "rung": rung, "fallback": fallback, "purpose": "initial",
               "warmup_steps": a.warmup_steps, "measured_steps": a.measured_steps, "seed": a.seed, "cpus": a.cpus,
               "storage": a.storage,
               "run_name": "mps-spike-%dx%d-%s-f%s-%s" % (mb, max_len, rung, fallback, time.strftime("%Y%m%d-%H%M%S")),
               "expect_env": expect_env, "hparams": HPARAMS}
        print("spike run: micro-batch %d, max_len %d, grad-accum %d" % (mb, max_len, cfg["grad_accum"]), flush=True)
        result = launch(cfg, "ray")
        configs.append({"micro_batch": mb, "max_len": max_len, "grad_accum": cfg["grad_accum"], "attempts": [result],
                        "standing": 0})

    import psutil
    meta = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split": {"path": entry["path"], "sha256": entry["sha256"], "n_messages": entry["n_messages"]},
            "manifest": {"path": repo_relpath(DEFAULT_MANIFEST), "sha256": sha256_file(DEFAULT_MANIFEST)},
            # the pin identifies the weights; base_dir (a local HF cache path) never enters the report
            "model": {"repo": pins.LAYA["repo"], "subfolder": pins.LAYA["subfolder"],
                      "revision": pins.LAYA["revision"], "model_sha256": pins.LAYA["sha256"]["model.safetensors"]},
            # D-05: these match the lock by construction; the report records them anyway
            "versions": dict({k: importlib.metadata.version(k) for k in VERSIONED},
                             python=platform.python_version()),
            "platform": {"macos": platform.mac_ver()[0], "machine": platform.machine(),
                         "total_memory_bytes": psutil.virtual_memory().total},
            "protocol": {"grid": ",".join("%dx%d" % g for g in grid), "warmup_steps": a.warmup_steps,
                         "measured_steps": a.measured_steps, "effective_batch": a.effective_batch, "seed": a.seed,
                         "cpus": a.cpus},
            "hparams": HPARAMS, "laya_internals": LAYA_INTERNALS}
    md = spike.write_report(a.out, spike.build_report(meta, configs))
    print("report -> %s" % md)


if __name__ == "__main__":
    main()
