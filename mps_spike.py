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

Fallback ops (D-10, D-11): every config first runs with PYTORCH_ENABLE_MPS_FALLBACK=0. The worker records every warning
from before the model exists, tagged by stage (setup, tokenize, step), and turns a NotImplementedError into an attempt
with status not_implemented naming the op. Only such a config is rerun, in a fresh process, with fallback 1, so the
report can list every op that falls back to the CPU and whether it blocks training.

Quiet-machine preflight: right after argument validation the orchestrator records available memory, two swap-used
samples 5 s apart and the swapouts counter, and refuses below --min-available-gib (0 records an override); an override
or idle swap growth is flagged in the markdown as a busy-machine run.

Token lengths (D-09b): before the grid, a `--child stats` subprocess (CPU, no Ray) loads Laya's tokenizer from the
re-verified pinned weights and reports, counts and percentiles only, how many selection_train messages exceed 128 and
256 tokens untruncated; D-03's truncation rule reads that number. The same child counts the neg8 items, and the
orchestrator refuses a report whose measured attempts saw a different count (the projection's basis).

Attempt chain (D-08): after the strict attempt and any fallback rerun, an attempt that completed every measured step
but grew system swap gets one immediate rerun at the same rung (--swap-rerun, default on), and a rung-none attempt
that fails the D-01 memory gate is retried with only tok_embeddings frozen (rung embeddings_frozen), keeping its
fallback setting. The last attempt stands; every attempt stays in the report; meta.policy records the run policy.

Ray overhead (D-09a): after the verdict, the chosen config (or 4x128 as a labelled reference when nothing FITS) reruns
in-process without Ray, in a fresh process with the same env and --cpus threads; the report gives both median step
times per regime and their % delta. This run never changes the verdict.

Laya internals used (reached through pilot/laya_adapter.py; LAYA_INTERNALS keeps the names the committed report
records): lt.load_checkpoint, lt.items_from_rows, lt.encode_item, lt._forward,
lt.rlcd_loss, laya.common.collate_items, laya.common.uses_parallel_layout, laya.common.DecisionModel,
DecisionModel.head_checkpointing, encoder.gradient_checkpointing_enable, encoder.embeddings.tok_embeddings.

usage:
  python mps_spike.py --out results/spike
  python mps_spike.py --grid 4x128 --warmup-steps 1 --measured-steps 2 --effective-batch 8 --min-available-gib 0 \
      --out /tmp/spike  # smoke on a busy machine (flagged as a busy-machine run)
  python mps_spike.py --no-swap-rerun --out results/spike  # no swap-only rerun before the embeddings-frozen rung
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
import warnings

from pilot import laya_adapter as la
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
    if not la.enable_gradient_checkpointing(model):
        raise SystemExit("encoder has no gradient checkpointing; the spike measures with checkpointing on")
    if rung == "embeddings_frozen":
        la.freeze_embeddings(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def require_param_device(model, device_type):
    """Refuse unless every model parameter sits on `device_type` (LAYA-09 criterion 1; no silent CPU timing)."""
    have = {p.device.type for p in model.parameters()}
    if have != {device_type}:
        raise SystemExit("model parameters on %s, expected %s; the spike refuses to time a silent CPU run"
                         % (", ".join(sorted(have)), device_type))


def _pinned_source(c, max_len, mark):
    """Pinned Laya model + two batch makers (real neg8, padded to max_len) over the verified selection_train split.

    mark(stage) tags the warnings that follow with their stage (load = setup, item building = tokenize; D-11).
    """
    from pilot.weights import verify_files
    from ray_train import read_verified_rows

    # the worker re-checks the bytes the driver verified minutes ago (T-02-01)
    verify_files(c["base_dir"], c["base_sha256"], "base checkpoint")
    model, tok, cfg = la.load_checkpoint(c["base_dir"])
    head_max_len = cfg.get("head_max_len", 192)
    parallel = la.uses_parallel_layout(cfg)
    rows = read_verified_rows(c["data"], c["data_sha256"])
    mark("tokenize")
    items, skipped = la.items_from_rows(tok, spike.neg8_rows(rows, c["seed"]), max_len, head_max_len)
    random.Random(c["seed"]).shuffle(items)

    def maker(pad):
        pool = itertools.cycle(items)

        def make():
            chunk = [next(pool) for _ in range(c["micro_batch"])]
            if pad:
                chunk = [spike.pad_item_to(it, max_len) for it in chunk]
            return la.collate_items([[la.encode_item(tok, it, max_len, head_max_len, None, parallel) for it in chunk]],
                                    tok.pad_token_id)
        return make
    return model, maker(False), maker(True), len(items), skipped


# a max_len no message reaches, so build_sequence keeps every state token (the untruncated length)
UNTRUNCATED = 100000


def stats_cfg(base_dir, data_sha256, seed, cpus):
    """Config of the `--child stats` run (CPU, no Ray) over the verified selection_train split."""
    return {"base_dir": base_dir, "base_sha256": dict(pins.LAYA["sha256"]), "data": SPLIT, "data_sha256": data_sha256,
            "seed": seed, "cpus": cpus, "expect_env": spike.spike_env("0", cpus)}


def token_lengths(c):
    """--child stats payload: D-09b token stats with Laya's tokenizer and the neg8 item count, no text or uids.

    Only the tokenizer is needed, so the model is dropped right after the load; the child exits before any measured
    run, so its memory never overlaps the grid.
    """
    import torch
    from pilot.weights import verify_files
    from ray_train import read_verified_rows

    torch.set_num_threads(c["cpus"])
    verify_files(c["base_dir"], c["base_sha256"], "base checkpoint")
    model, tok, cfg = la.load_checkpoint(c["base_dir"])
    del model
    gc.collect()
    head_max_len, parallel = cfg.get("head_max_len", 192), la.uses_parallel_layout(cfg)
    rows = read_verified_rows(c["data"], c["data_sha256"])
    seqs, states = [], []
    for row in rows:
        items = la.items_from_rows(tok, [row], UNTRUNCATED, head_max_len)[0]
        seqs.append([len(la.encode_item(tok, it, UNTRUNCATED, head_max_len, None, parallel)["ids"]) for it in items])
        if items:
            states.append(len(items[0]["state_ids"]))
    n_neg8 = len(la.items_from_rows(tok, spike.neg8_rows(rows, c["seed"]), 128, head_max_len)[0])
    return {"status": "ok", "token_stats": spike.token_stats(seqs, states), "n_neg8_items": n_neg8,
            "n_positives": sum(1 for r in rows for v in r["expected"].values() if v), "n_messages": len(rows)}


def check_neg8(configs, n_items):
    """Refuse when a measured attempt built a different neg8 item count than the stats child (the D-02 basis)."""
    for c in configs:
        for i, att in enumerate(c["attempts"]):
            if att.get("status") == "ok" and att.get("n_neg8_items") != n_items:
                raise SystemExit("neg8 item count %s in attempt %s differs from %d in the stats child; the projection "
                                 "would be wrong" % (att.get("n_neg8_items"),
                                                     "%dx%d #%d" % (c["micro_batch"], c["max_len"], i), n_items))


def tiny_model():
    """Tiny ModernBERT encoder + Laya head (random init, CPU, no download) for the offline self-tests."""
    from transformers import ModernBertConfig, ModernBertModel

    cfg = ModernBertConfig(vocab_size=1000, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                           num_attention_heads=4, global_attn_every_n_layers=3, local_attention=16, pad_token_id=0,
                           bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
    return la.decision_model(ModernBertModel(cfg), 2, 2)


def tiny_items(n, min_len, max_len, seed):
    """n synthetic encoded noul items for tiny_model: [CLS] body [SEP], lengths uniform in [min_len, max_len]."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = [rng.randint(5, 54) for _ in range(rng.randint(min_len, max_len) - 2)]
        out.append({"ids": [1] + body + [2], "markers": [3, 4], "qtype": 2,
                    "target": [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]})
    return out


def _tiny_source(c, max_len, mark):
    """Offline self-test source used by tests/test_spike_*.py (the orchestrator always launches "pinned")."""
    def maker(items):
        pool = itertools.cycle(items)
        return lambda: la.collate_items([[next(pool) for _ in range(c["micro_batch"])]], 0)
    model = tiny_model()
    mark("tokenize")
    real = maker(tiny_items(64, 8, min(24, max_len), c["seed"]))
    padded = maker(tiny_items(64, max_len, max_len, c["seed"]))
    return model, real, padded, 64, {}


def _identity(c):
    """The config fields every attempt carries, whatever its status (ok, oom, error, crashed)."""
    return {k: c[k] for k in ("purpose", "micro_batch", "max_len", "grad_accum", "rung", "fallback", "device")}


def _measure(c):
    import torch

    if c["transport"] == "checkpoint":
        import ray
        from ray_train import require_driver_node
        require_driver_node(ray.get_runtime_context().get_node_id(), c["driver_node_id"])
    env_seen = check_env(c["expect_env"])
    device = torch.device(c["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this process; the spike cannot run")
    # D-10: record every warning before the model exists. Each MPS fallback op warns once per process, so a capture
    # opened after the first MPS op would miss it (RESEARCH Pitfall 3); strict and fallback never share a process.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        marks, got = [(0, "setup")], {"stage": "setup"}

        def mark(stage):
            got["stage"] = stage
            marks.append((len(caught), stage))
        try:
            payload = _steps(c, device, mark, got)
        except NotImplementedError as e:
            # strict pass (PYTORCH_ENABLE_MPS_FALLBACK=0): an op with no MPS kernel; the orchestrator reruns the config
            # with fallback 1 in a fresh process to inventory every op that falls back (spike.followup)
            payload = dict(_identity(c), status="not_implemented", device=device.type,
                           error_op={"op": spike.op_name(str(e)), "stage": got["stage"],
                                     "message": spike.scrub(str(e))},
                           **{k: v for k, v in got.items() if k != "stage"})
        texts = [str(w.message) for w in caught]
        payload["fallback_ops"] = spike.tag_fallback_ops(texts, marks)
        # counts only: warning text never enters the tracked report
        payload["other_warnings"] = sum(1 for t in texts if not spike.is_fallback_warning(t))
    payload["env_seen"] = env_seen
    return payload


def _steps(c, device, mark, got):
    """Load, configure, move and time one config; `got` collects what was measured so far (for a strict failure)."""
    import psutil
    import torch

    torch.manual_seed(c["seed"])
    torch.set_num_threads(c["cpus"])
    mb, max_len, accum, h = c["micro_batch"], c["max_len"], c["grad_accum"], c["hparams"]

    if c["source"] == "pinned":
        model, real_maker, pad_maker, n_items, skipped = _pinned_source(c, max_len, mark)
    elif c["source"] == "tiny":
        model, real_maker, pad_maker, n_items, skipped = _tiny_source(c, max_len, mark)
    else:
        raise SystemExit("unknown model source %r" % (c["source"],))
    got.update(n_neg8_items=n_items, items_skipped=skipped)

    def padded_maker():
        batch = pad_maker()
        if tuple(batch["input_ids"].shape) != (mb, max_len) or not bool(batch["attention_mask"].all()):
            raise SystemExit("padded batch shape %s, expected (%d, %d) with a full attention mask"
                             % (tuple(batch["input_ids"].shape), mb, max_len))
        return batch

    mark("setup")
    trainable = configure_model(model, c["rung"])
    got["trainable_params"] = trainable
    if c["transport"] == "checkpoint":
        import ray.train.torch
        model = ray.train.torch.prepare_model(model, move_to_device=device)
    else:
        model = model.to(device)
    gc.collect()                                            # drop the CPU copy before any sampling (Pitfall 8)
    require_param_device(model, device.type)
    got["param_device"] = device.type
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
            logits = la.forward(model, batch, device, False, False)
            loss = la.rlcd_loss(logits, batch["target"].to(device), batch["marker_mask"].to(device),
                                batch["qtype"].to(device), h["sigma"])
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

    regimes, oom = {}, None
    got["regimes"] = regimes                                # completed regimes survive a strict failure
    mark("step")
    for name, maker in (("padded", padded_maker), ("real", real_maker)):
        times, s0, o0, s1, o1 = [], None, None, None, None
        stage, idx = "warmup", 0
        try:
            for idx in range(c["warmup_steps"]):
                step(maker, False)
            stage, idx = "measured", 0
            # swap brackets the measured steps only, never warm-up or the model load (Pitfall 8)
            s0, o0 = spike.swap_used(), spike.swapouts()
            for idx in range(c["measured_steps"]):
                times.append(step(maker, True))
        except RuntimeError as e:
            # the MPS allocator raises a plain RuntimeError on OOM; anything else is a bug and propagates
            if not spike.is_mps_oom(str(e)):
                raise
            oom = {"regime": name, "stage": stage, "step": idx, "message": spike.scrub(str(e))}
        if s0 is not None:
            s1, o1 = spike.swap_used(), spike.swapouts()
        regimes[name] = {"warmup_steps": c["warmup_steps"], "times": spike.summarize(times),
                         "batch_shape": last["shape"], "swap_used_before": s0, "swap_used_after": s1,
                         "swap_delta_bytes": None if s0 is None else s1 - s0,
                         "swapouts_delta": None if o0 is None or o1 is None else o1 - o0}
        if oom:
            break                                           # post-OOM allocator state is never measured again
    payload = dict(_identity(c), status="oom" if oom else "ok", device=device.type,
                   param_device=",".join(sorted({p.device.type for p in model.parameters()})),
                   ddp_wrapped=hasattr(model, "module"), world_size=1, trainable_params=trainable,
                   total_params=sum(p.numel() for p in model.parameters()), n_neg8_items=n_items,
                   items_skipped=skipped, torch_threads=torch.get_num_threads(), regimes=regimes,
                   peak_memory_bytes=peak["total"], peak_driver_allocated_bytes=peak["driver"],
                   peak_rss_bytes=peak["rss"],
                   recommended_max_memory_bytes=torch.mps.recommended_max_memory() if is_mps else None,
                   final_loss=None if last["loss"] is None else float(last["loss"]))
    if oom:
        payload["oom"] = oom
    return payload


def spike_worker(c):
    """Ray Train worker loop (transport "checkpoint") and in-process runner (transport "return")."""
    try:
        payload = _measure(c)
    except SystemExit as e:
        payload = {"status": "refused", "reason": spike.scrub(str(e))}
    except Exception as e:
        # an unexpected failure stays a visible attempt (counted as untrusted by the verdict), never a lost run
        payload = dict(_identity(c), status="error", error=spike.scrub("%s: %s" % (type(e).__name__, e)))
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
    """--child entry point: run one config (ray | inprocess) or the token stats (stats); payload to cfg["out_json"]."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if mode == "stats":
        try:
            payload = token_lengths(cfg)
        except SystemExit as e:
            payload = {"status": "refused", "reason": spike.scrub(str(e))}
    elif mode == "ray":
        payload = run_ray(cfg)
    elif mode == "inprocess":
        import torch
        torch.set_num_threads(cfg["cpus"])
        payload = spike_worker(dict(cfg, transport="return"))
    else:
        raise SystemExit("--child mode must be 'ray', 'inprocess' or 'stats', got %r" % (mode,))
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


def run_config(base, cpus, policy, launcher=launch):
    """One grid config's attempt chain: the strict (fallback 0) attempt, then whatever spike.followup asks for.

    Every attempt runs in its own fresh subprocess with its own env (the fallback flag is read at `import torch`),
    so strict and fallback attempts never share a process. Returns the config entry of the report.
    """
    chain, nxt = [], {"rung": "none", "fallback": "0", "purpose": "initial"}
    while nxt:
        cfg = dict(base, rung=nxt["rung"], fallback=nxt["fallback"], purpose=nxt["purpose"],
                   expect_env=spike.spike_env(nxt["fallback"], cpus),
                   run_name="mps-spike-%dx%d-%s-f%s-a%d-%s" % (base["micro_batch"], base["max_len"], nxt["rung"],
                                                              nxt["fallback"], len(chain),
                                                              time.strftime("%Y%m%d-%H%M%S")))
        print("spike run: micro-batch %d, max_len %d, grad-accum %d, rung %s, fallback %s (%s)"
              % (cfg["micro_batch"], cfg["max_len"], cfg["grad_accum"], cfg["rung"], cfg["fallback"],
                 cfg["purpose"]), flush=True)
        # the attempt's identity (rung, fallback, purpose, ...) always comes from the spec it was launched with: a
        # crashed child returns no payload, and followup's one-swap-rerun rule reads the chain's purposes
        chain.append(dict(launcher(cfg, "ray"), **_identity(cfg)))
        nxt = spike.followup(chain, policy)
    return {"micro_batch": base["micro_batch"], "max_len": base["max_len"], "grad_accum": base["grad_accum"],
            "attempts": chain, "standing": spike.standing(chain)}


def overhead_target(v, configs):
    """D-09a target -> (basis, config entry, rung, fallback): the verdict's chosen config or, when nothing FITS, the
    4x128 reference (the first grid config if 4x128 is not in the grid) at rung none, fallback 0."""
    c = v.get("chosen")
    if c:
        entry = next(e for e in configs if (e["micro_batch"], e["max_len"]) == (c["micro_batch"], c["max_len"]))
        return "chosen", entry, c["rung"], c["fallback"]
    entry = next((e for e in configs if (e["micro_batch"], e["max_len"]) == (4, 128)), configs[0])
    return "reference", entry, "none", "0"


def measure_overhead(v, configs, bases, cpus, launcher=launch):
    """D-09a: rerun the target config in-process (no Ray) in a fresh process with the same env and thread count, and
    compare its step times with the Ray attempt of the same config, rung and fallback.

    Runs after the verdict and is stored only under report["ray_overhead"], never in any config's attempts, so it
    cannot change the verdict. A refused or crashed in-process run is recorded, not raised: the grid's report survives.
    """
    basis, entry, rung, fallback = overhead_target(v, configs)
    # the Ray side is the last attempt of the chain at this rung and fallback (the standing attempt for "chosen")
    ray_attempt = [a for a in entry["attempts"] if a.get("rung") == rung and a.get("fallback") == fallback][-1]
    cfg = dict(bases[(entry["micro_batch"], entry["max_len"])], rung=rung, fallback=fallback, purpose="ray_overhead",
               expect_env=spike.spike_env(fallback, cpus),
               run_name="mps-spike-%dx%d-%s-f%s-inprocess-%s" % (entry["micro_batch"], entry["max_len"], rung,
                                                                  fallback, time.strftime("%Y%m%d-%H%M%S")))
    print("ray overhead (D-09a, %s): micro-batch %d, max_len %d, rung %s, fallback %s, in-process"
          % (basis, cfg["micro_batch"], cfg["max_len"], rung, fallback), flush=True)
    try:
        inprocess = dict(launcher(cfg, "inprocess"), **_identity(cfg))
    except SystemExit as e:
        inprocess = dict(_identity(cfg), status="refused", reason=spike.scrub(str(e)))
    return spike.ray_overhead(ray_attempt, inprocess, basis)


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
    ap.add_argument("--tie-tolerance", type=float, default=spike.TIE_TOLERANCE,
                    help="FITS configs within this fraction of the fastest projected epoch count as tied (D-03)")
    ap.add_argument("--storage", default=os.path.abspath("ray_results"), help="Ray run storage (local dir)")
    ap.add_argument("--min-available-gib", type=float, default=spike.MIN_AVAILABLE_GIB,
                    help="refuse to measure below this much available memory; 0 records a busy-machine run")
    ap.add_argument("--swap-rerun", action=argparse.BooleanOptionalAction, default=True,
                    help="rerun once, at the same rung, an attempt that completed every measured step but grew swap, "
                         "before the embeddings-frozen rung (every attempt stays in the report; the last one stands)")
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
    if not a.tie_tolerance >= 0:
        ap.error("--tie-tolerance must not be negative")
    if not a.min_available_gib >= 0:
        ap.error("--min-available-gib must not be negative")

    # quiet-machine preflight (RESEARCH Pitfall 2), before the split check, the fetch or any child: background apps
    # move the system swap the D-01 gate reads, so a memory-starved machine could fake a DOES NOT FIT
    import psutil
    refuse = ("%.1f GiB available, below --min-available-gib %.1f: quit heavy apps (browsers, IDE indexers, Docker, "
              "other Python processes), connect AC power and run under `caffeinate -i`, then retry; or pass "
              "--min-available-gib 0 to record a busy-machine run")
    available = psutil.virtual_memory().available
    if available / 2**30 < a.min_available_gib:
        raise SystemExit(refuse % (available / 2**30, a.min_available_gib))
    swap0, outs0 = spike.swap_used(), spike.swapouts()
    time.sleep(5)
    pf = spike.preflight(psutil.virtual_memory().available, [swap0, spike.swap_used()], [outs0, spike.swapouts()],
                         a.min_available_gib)
    if not pf["passed"]:
        raise SystemExit(refuse % (pf["available_gib"], a.min_available_gib))
    print("preflight: %.2f GiB available (threshold %.1f%s), swap grew while idle: %s"
          % (pf["available_gib"], a.min_available_gib, ", overridden" if pf["override"] else "",
             pf["swap_grew_while_idle"]), flush=True)

    # D-05: the main checkout's locked venv, the hash-verified split and the pinned weights, before any child runs
    entry = verify_split_file(SPLIT, roles=("train",))
    base_dir = fetch(pins.LAYA)
    # D-09b: token lengths and the neg8 item count, in a CPU child that exits before the first measured run
    stats = launch(stats_cfg(base_dir, entry["sha256"], a.seed, a.cpus), "stats")
    if stats.get("status") != "ok":
        raise SystemExit("the token-stats child failed (%s, exit %s); run `python mps_spike.py --child stats CONFIG` "
                         "to see why" % (stats.get("status"), stats.get("returncode")))
    print("token stats: %.2f%% of %d messages over 128 tokens, %.2f%% over 256; %d neg8 items"
          % (stats["token_stats"]["pct_messages_over_128"], stats["n_messages"],
             stats["token_stats"]["pct_messages_over_256"], stats["n_neg8_items"]), flush=True)
    policy = {"measured_steps": a.measured_steps, "swap_rerun": a.swap_rerun}
    configs, bases = [], {}
    for mb, max_len in grid:
        base = {"source": "pinned", "device": "mps", "base_dir": base_dir, "base_sha256": dict(pins.LAYA["sha256"]),
                "data": SPLIT, "data_sha256": entry["sha256"], "micro_batch": mb, "max_len": max_len,
                "grad_accum": a.effective_batch // mb, "warmup_steps": a.warmup_steps,
                "measured_steps": a.measured_steps, "seed": a.seed, "cpus": a.cpus, "storage": a.storage,
                "hparams": HPARAMS}
        bases[(mb, max_len)] = base
        configs.append(run_config(base, a.cpus, policy))

    # every measured attempt must have projected over the same neg8 items the stats child counted
    check_neg8(configs, stats["n_neg8_items"])

    protocol = {"grid": ",".join("%dx%d" % g for g in grid), "warmup_steps": a.warmup_steps,
                "measured_steps": a.measured_steps, "effective_batch": a.effective_batch, "seed": a.seed,
                "cpus": a.cpus}
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
            "protocol": protocol,
            "policy": {"min_available_gib": a.min_available_gib, "swap_rerun": a.swap_rerun,
                       "tie_tolerance": a.tie_tolerance},
            "projection_basis": "real neg8 median optimizer step x optimizer steps per epoch over all 1,450 "
                                "selection_train messages at effective batch 32; conservative: ray_train holds out "
                                "up to 40 messages for calibration and Phase 5 calibrates on val (LAYA-05)",
            "hparams": HPARAMS, "laya_internals": LAYA_INTERNALS}
    report = spike.build_report(meta, configs)
    report["preflight"] = pf
    report["token_stats"] = stats["token_stats"]
    report["neg8"] = {"n_items": stats["n_neg8_items"], "n_positives": stats["n_positives"],
                      "n_messages": stats["n_messages"], "seed": a.seed}
    # D-10/D-11: every op a strict attempt raised on, from its fallback rerun; a blocking op keeps its config from FITS
    inventory = spike.op_inventory(configs)
    report["op_inventory"] = inventory
    # D-03: the truncation rule reads pct_messages_over_128 from the stats child
    report["verdict"] = spike.verdict(configs, protocol, token_stats=stats["token_stats"],
                                      tie_tolerance=a.tie_tolerance, blocking=spike.blocking_ops(inventory))
    # D-09a, after the verdict: Ray vs in-process step times of the chosen (or reference) config; informational
    report["ray_overhead"] = measure_overhead(report["verdict"], configs, bases, a.cpus)
    md = spike.write_report(a.out, report)
    print("report -> %s" % md)
    print(spike.verdict_line(report["verdict"]), flush=True)


if __name__ == "__main__":
    main()
