"""Phase 4 compute checkpoint (LAYA-07): measure a real Laya training run from the Phase 2 spike's start config and
project the runtime of the whole experiment matrix, for the user to approve before any Phase 5/6 run.

Design (like mps_spike.py): this orchestrator imports no torch. It runs the spike's quiet-machine preflight (available
memory, swap growth while idle), verifies the training split, reads the start config from the committed spike report
(verdict.chosen, FITS only), checks the laya pin and the base weights, and then starts ONE fresh child process,
`ray_train.py --device mps --workers 1` with the spike's micro-batch, max_len and grad-accum, --neg 8 and the spike's
MPS env (strict fallback 0, HIGH and LOW watermarks 1.0). A fresh process, because torch reads
PYTORCH_ENABLE_MPS_FALLBACK once at import and the MPS allocator keeps a process-wide high-water mark. The child's
checkpoint carries the per-epoch run record (pilot.runs.read_record); pilot.compute turns it into the projection of
the whole experiment matrix (s_per_step = the largest train_s / optimizer_steps over the measured epochs; item counts
of all five verified splits), a what-if table, a disk budget and the assumptions to confirm, and writes
compute_checkpoint.json and compute_checkpoint.md. The probe's checkpoint and Ray storage are deleted afterwards unless --keep-run.

Probe protocol (RESEARCH Open Question 1): a 2-epoch probe that is interrupted and resumed (LAYA-04 on the real
hardware). The orchestrator learns the run_id from `ray_train.py ... --print-run-id`, starts the child in its own
session, waits until storage/<run_id> registers the epoch-001 checkpoint (compute.epoch_registered), waits
--kill-delay-s into epoch 2, SIGKILLs the child's process group, reaps the captured process tree (survivors get a
SIGKILL after 120 s), waits for available memory to recover and reruns the identical command, which resumes from
epoch-001. --interrupt fail uses ray_train.py's hidden --fail-after-epoch 1 instead (deterministic; the CPU test);
--interrupt none runs once. A kill that comes too late is recorded as missed, and the report then says the resume was
not demonstrated. An attempt that ends with the MPS out-of-memory line is followed by a fresh child on the next rung of
the LAYA-03 ladder (pilot.runs.next_rung), each with its own run_id (compute.attempt_chain); every attempt stays in the
report, and an exhausted ladder is reported as status ladder_exhausted. Before any child, the disk under --storage
must hold pilot.runs.disk_need_bytes for the run.

Measured BERTurk and scoring rates (RESEARCH Open Question 2): after the Laya probe, a BERTurk timing child
(`--child berturk CONFIG`, a fresh process with the same MPS env) loads the pinned BERTurk weights, fetched and
verified on the driver and re-verified (sha256) in the child before any load, and times 3 warm-up + 20 AdamW optimizer
steps on real selection_train batches of 32 at max_len 128 in fp32, plus a 64-message inference pass. A scoring probe
runs `score.py <val> --model laya --checkpoint <--base> --device <device> --workers 0 --n 32` into a temp out-root that
is deleted afterwards and reads only the sidecar's timing (never test, HOTELS or a fine-tuned checkpoint, never a
metric). Both replace the estimates in the projection (basis measured); a probe that fails leaves its lines on the
estimate and the report says why. --no-berturk / --no-score-probe skip them.

A run off the protocol (not mps, not the pinned weights, not the committed selection_train under the committed
manifest, a start config other than the spike's at rung none, an interrupt other than kill, BERTurk weights other than
the pinned ones, or a test hook) is bannered SMOKE RUN with
meta.protocol.protocol_complete false; a busy-machine preflight is bannered too. Neither is a compute checkpoint.

Run it on AC power, after quitting heavy apps, under `caffeinate -i` (the machine must not sleep mid-epoch).

usage:
  caffeinate -i python compute_checkpoint.py --out results/compute
  python compute_checkpoint.py --device cpu --base CKPT --data SPLIT --manifest M --interrupt fail \
      --min-available-gib 0 --out /tmp/compute   # smoke (bannered SMOKE RUN)
"""
import argparse
import functools
import glob
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil

from pilot import compute, pins, runs, spike
from pilot import laya_adapter as la
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, repo_relpath, sha256_file,
                        verify_split_file, write_atomic)
from pilot.weights import fetch, identity, verify_files

DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "splits", "selection_train.jsonl")
DEFAULT_SPIKE = os.path.join(REPO_ROOT, "results", "spike", "mps_spike.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "compute")
VERSIONED = ("torch", "ray", "laya", "transformers", "psutil")
NEG = 8
TAIL_LINES = 40
REAP_S = 120                # a killed tree's processes get this long to exit before they are SIGKILLed
RECOVER_S, RECOVER_POLL_S = 120, 5   # wait for available memory to recover after an interrupt, before the rerun
# the BERTurk timing child (RESEARCH Open Question 2): real selection_train batches of 32 at max_len 128, AdamW, fp32
BERTURK = {"batch": 32, "max_len": 128, "lr": 3e-5, "weight_decay": 0.01, "clip_grad_norm": 1.0, "warmup_steps": 3,
           "measured_steps": 20, "infer_msgs": 64}
BERTURK_BASIS = "plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer"
SCORE_PROBE_N = 32
SCORING_BASIS = ("--n 32 val subset of the --base weights (the pinned zero-shot model in the real run; same "
                 "architecture as a fine-tuned checkpoint), in-process, fp32")


def preflight(min_available_gib):
    """The spike's quiet-machine preflight (mps_spike.main): refuse below --min-available-gib, sample swap twice 5 s
    apart while idle; 0 records an override (a busy-machine run)."""
    refuse = ("%.1f GiB available, below --min-available-gib %.1f: quit heavy apps (browsers, IDE indexers, Docker, "
              "other Python processes), connect AC power and run under `caffeinate -i`, then retry; or pass "
              "--min-available-gib 0 to record a busy-machine run")
    available = psutil.virtual_memory().available
    if available / 2**30 < min_available_gib:
        raise SystemExit(refuse % (available / 2**30, min_available_gib))
    swap0, outs0 = spike.swap_used(), spike.swapouts()
    time.sleep(5)
    pf = spike.preflight(psutil.virtual_memory().available, [swap0, spike.swap_used()], [outs0, spike.swapouts()],
                         min_available_gib)
    if not pf["passed"]:
        raise SystemExit(refuse % (pf["available_gib"], min_available_gib))
    print("preflight: %.2f GiB available (threshold %.1f%s), swap grew while idle: %s"
          % (pf["available_gib"], min_available_gib, ", overridden" if pf["override"] else "",
             pf["swap_grew_while_idle"]), flush=True)
    return pf


def child_command(a, start, base, ckpt, storage):
    """The ray_train.py probe command: the spike's start config on its rung, neg 8, one worker."""
    cmd = [sys.executable, os.path.join(REPO_ROOT, "ray_train.py"), "--data", os.path.abspath(a.data),
           "--manifest", os.path.abspath(a.manifest), "--base", base, "--out", ckpt, "--storage", storage,
           "--device", a.device, "--workers", "1", "--cpus-per-worker", str(a.cpus), "--epochs", str(a.epochs),
           "--micro-batch", str(start["micro_batch"]), "--grad-accum", str(start["grad_accum"]),
           "--max-len", str(start["max_len"]), "--neg", str(NEG), "--seed", str(a.seed)]
    cmd += runs.rung_flags(start["rung"])
    if a.max_messages is not None:
        cmd += ["--max-messages", str(a.max_messages)]
    if a.calib_frac is not None:
        cmd += ["--calib-frac", str(a.calib_frac)]
    return cmd


def run_id_of(cmd, env):
    """The run id the probe command trains under (ray_train.py --print-run-id; no Ray started)."""
    r = subprocess.run(cmd + ["--print-run-id"], cwd=REPO_ROOT, env=env, capture_output=True, text=True, shell=False)
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines or not lines[-1].startswith("laya-"):
        raise SystemExit("ray_train.py --print-run-id failed (exit %d): %s"
                         % (r.returncode, (r.stderr or r.stdout).strip()[-2000:]))
    return lines[-1]


def tail(path, n=TAIL_LINES):
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def last_line(path):
    """The last non-empty line of a child log, scrubbed for the report (the child's own SystemExit line)."""
    lines = [ln.strip() for ln in read_text(path).splitlines() if ln.strip()]
    return spike.scrub(lines[-1]) if lines else ""


def protocol_complete(a, start):
    """True only for the real compute checkpoint: mps, the pinned weights, the committed selection_train under the
    committed manifest, no test hook (--max-messages, --calib-frac, --interrupt fail), a kill-and-resume probe, and the
    spike's start config at rung none."""
    return (a.device == "mps" and a.base == "multilingual"
            and os.path.realpath(a.data) == os.path.realpath(DEFAULT_DATA) and manifest_is_committed(a.manifest)
            and a.max_messages is None and a.calib_frac is None and a.interrupt == "kill"
            and (not a.berturk or a.berturk_source == "pinned")
            and {k: start[k] for k in runs.SPIKE_START} == runs.SPIKE_START and start["rung"] == "none")


def nearest_existing(path):
    """`path` or its nearest existing parent (the filesystem a not-yet-created --storage lands on)."""
    p = os.path.abspath(path)
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return p


def disk_preflight(storage, model_bytes, epochs):
    """Refuse before any child when the disk under --storage cannot hold the probe run (pilot.runs.disk_need_bytes
    with per-epoch exports on, the ray_train.py floor)."""
    need = runs.disk_need_bytes(model_bytes, epochs, True)
    free = shutil.disk_usage(nearest_existing(storage)).free
    if free < need:
        raise SystemExit("%.1f GiB free under --storage; the probe needs about %.1f GiB (three train states during "
                         "checkpoint rotation, per-epoch exports, the final checkpoint); free disk or pass another "
                         "--storage" % (free / 2**30, need / 2**30))
    return {"need_bytes": need, "free_bytes": free}


# ---- the interrupt: kill the probe's whole process tree once epoch 1 is registered ----

def kill_tree(proc):
    """SIGKILL the process group of `proc` (started with start_new_session=True), reap `proc`, then wait up to REAP_S
    for every process of its captured tree and SIGKILL the survivors (T-04-19: no Ray worker may keep holding MPS
    memory into the resumed run). Returns the number of survivors that needed the extra SIGKILL."""
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    try:
        pgid = os.getpgid(proc.pid)
        if pgid == os.getpgrp():                            # never our own group: the child must have its own session
            raise SystemExit("the probe child shares the orchestrator's process group; refusing to kill it")
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass                                                # it exited (and its group with it) in the meantime
    proc.wait()
    _, alive = psutil.wait_procs(children, timeout=REAP_S)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)
    return len(alive)


def kill_after_registration(proc, registered, delay_s, poll_s=1.0):
    """Wait until registered() (epoch 1's checkpoint is in Ray's snapshot), then delay_s more into epoch 2, then kill
    the child's process tree (kill_tree). missed is True when the child exited first: it finished (or failed) before
    the kill, so nothing was interrupted."""
    out = {"mode": "kill", "after_epoch": 1, "delay_s": delay_s, "missed": False, "leftover_processes": None}
    while not registered():
        if proc.poll() is not None:
            return dict(out, missed=True)
        time.sleep(poll_s)
    end = time.monotonic() + delay_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            return dict(out, missed=True)
        time.sleep(max(0.0, min(poll_s, end - time.monotonic())))
    if proc.poll() is not None:
        return dict(out, missed=True)
    out["leftover_processes"] = kill_tree(proc)
    # a child that exited on its own between the last poll and the SIGKILL was not interrupted
    out["missed"] = proc.returncode is not None and proc.returncode >= 0
    return out


def wait_for_memory(min_available_gib, timeout_s=RECOVER_S, poll_s=RECOVER_POLL_S):
    """Available GiB once it is back at --min-available-gib, or after timeout_s (the rerun then starts anyway and the
    report shows the number)."""
    end = time.monotonic() + timeout_s
    while True:
        available = psutil.virtual_memory().available / 2**30
        if available >= min_available_gib or time.monotonic() >= end:
            return available
        time.sleep(poll_s)


def drop_run(storage, rid):
    shutil.rmtree(os.path.join(storage, rid), ignore_errors=True)


def drop_storage(storage, own_storage, rids):
    """Delete the probe's Ray storage: the whole dir the orchestrator created, else only the probe's run dirs."""
    if own_storage:
        shutil.rmtree(storage, ignore_errors=True)
    else:
        for rid in rids:
            drop_run(storage, rid)


def launch_rung(rung, ctx):
    """One probe attempt on `rung` in fresh children: the first launch (its own session) interrupted per --interrupt,
    then, after an interrupt that landed, the identical command again, which resumes from epoch 1. Returns the attempt
    ({run_id, status ok|oom|failed, interrupt, resume | line | reason}); an ok attempt leaves its measurement in
    ctx["result"]."""
    a, storage, env = ctx["a"], ctx["storage"], ctx["env"]
    d = os.path.join(ctx["tmp"], "attempt-%d" % len(ctx["rids"]))
    os.makedirs(d)
    ckpt, log = os.path.join(d, "ckpt"), os.path.join(d, "probe.log")
    cmd = child_command(a, dict(ctx["start"], rung=rung), ctx["base"], ckpt, storage)
    rid = run_id_of(cmd, env)
    if os.path.exists(os.path.join(storage, rid)):
        # the probe must train from scratch: an existing dir would resume (or return) another measurement
        raise SystemExit("%s already holds run %s; the probe must start fresh: delete that dir or choose another "
                         "--storage" % (storage, rid))
    ctx["rids"].append(rid)
    ctx["logs"].append(log)
    attempt = {"run_id": rid}
    print("probe run %s on rung %s, interrupt %s -> log %s (tail -f it)" % (rid, rung, a.interrupt, log), flush=True)
    first = cmd + (["--fail-after-epoch", "1"] if a.interrupt == "fail" else [])
    with open(log, "wb") as f:
        proc = subprocess.Popen(first, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, shell=False,
                                start_new_session=True)
        try:
            if a.interrupt == "kill":
                interrupt = kill_after_registration(proc, lambda: compute.epoch_registered(storage, rid, 1),
                                                    a.kill_delay_s)
            else:
                proc.wait()
                interrupt = {"mode": a.interrupt, "after_epoch": 1 if a.interrupt == "fail" else None,
                             "delay_s": None, "missed": False, "leftover_processes": None}
        except BaseException:
            if proc.poll() is None:                         # Ctrl-C never reaches a child in its own session
                kill_tree(proc)
            raise
    oom = runs.failure_line(read_text(log))
    if oom:
        return dict(attempt, status="oom", line=spike.scrub(oom), interrupt=interrupt)
    rc = proc.returncode
    if a.interrupt == "fail" and (rc == 0 or not compute.epoch_registered(storage, rid, 1)):
        return dict(attempt, status="failed", returncode=rc, interrupt=interrupt,
                    reason="--fail-after-epoch 1 did not leave a registered epoch-1 checkpoint (exit %d): %s"
                           % (rc, last_line(log)))
    landed = a.interrupt != "none" and not interrupt["missed"]
    if not landed and rc != 0:
        return dict(attempt, status="failed", returncode=rc, interrupt=interrupt, reason=last_line(log))
    if landed:
        interrupt["post_interrupt_available_gib"] = wait_for_memory(a.min_available_gib)
        print("interrupted after epoch 1 (%s); rerunning the same command to resume" % a.interrupt, flush=True)
        with open(log, "ab") as f:
            r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, shell=False)
        oom = runs.failure_line(read_text(log))
        if oom:
            return dict(attempt, status="oom", line=spike.scrub(oom), interrupt=interrupt)
        if r.returncode != 0:
            return dict(attempt, status="failed", returncode=r.returncode, interrupt=interrupt,
                        reason="the resumed run failed: %s" % last_line(log))
    measured = compute.measured_from_checkpoint(ckpt)
    ctx["result"] = {"measured": measured, "ckpt": ckpt, "rid": rid}
    return dict(attempt, status="ok", interrupt=interrupt, resume=compute.resume_summary(runs.read_record(ckpt)))


def oom_dropping(launch, ctx):
    """launch, but an oom attempt's Ray run dir is deleted at once (unless --keep-run), so the next rung has the disk
    the preflight checked for one run."""
    def run(rung):
        attempt = launch(rung)
        if attempt["status"] == "oom" and not ctx["a"].keep_run:
            drop_run(ctx["storage"], attempt["run_id"])
        return attempt
    return run


# ---- the BERTurk timing child and the scoring probe (measured inputs of the projection, LAYA-07) ----

def berturk_child(cfg):
    """Time BERTurk fine-tuning and inference in this (fresh) process: warm-up + measured AdamW optimizer steps on
    batches of 32 at max_len 128 (forward with labels -> loss -> backward -> clip -> step), then one no_grad pass over
    infer_msgs messages, each timed with a device sync, memory sampled after every backward and step as in the trainer.

    source "pinned": the weights in cfg["dir"] are checked against pins.BERTURK before any load, the 48-label head is
    trained on real selection_train messages (verified bytes, labels in INTENTS order); source "tiny": a tiny random
    BERT on random ids (the offline test). torch and transformers are imported here only (the orchestrator stays
    torch-free)."""
    if cfg["source"] == "pinned":
        verify_files(cfg["dir"], pins.BERTURK["sha256"], "BERTurk at the pinned revision")   # before any load
    elif cfg["source"] != "tiny":
        raise SystemExit("unknown BERTurk source %r" % (cfg["source"],))
    import random

    import torch
    import transformers

    from intents import INTENTS
    from ray_train import memory_sampler, read_verified_rows, require_param_device

    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["cpus"])
    device = torch.device(cfg["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this process; the BERTurk timing child cannot run")
    batch_n, max_len = BERTURK["batch"], BERTURK["max_len"]
    if cfg["source"] == "pinned":
        tok = transformers.AutoTokenizer.from_pretrained(cfg["dir"])
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            cfg["dir"], num_labels=len(INTENTS), problem_type="multi_label_classification", dtype=torch.float32)
        rows = read_verified_rows(cfg["data"], cfg["data_sha256"])
        random.Random(cfg["seed"]).shuffle(rows)
        labels = torch.tensor([[1.0 if r["expected"][t] else 0.0 for t in INTENTS] for r in rows])

        def make(i, n, with_labels=True):
            idx = [(i * n + j) % len(rows) for j in range(n)]
            enc = dict(tok([rows[k]["state"] for k in idx], padding="longest", truncation=True, max_length=max_len,
                           return_tensors="pt"))
            return dict(enc, labels=labels[idx]) if with_labels else enc
    else:
        model = transformers.BertForSequenceClassification(transformers.BertConfig(
            vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
            num_labels=len(INTENTS), problem_type="multi_label_classification"))
        gen = torch.Generator().manual_seed(cfg["seed"])

        def make(i, n, with_labels=True):
            ids = torch.randint(0, 128, (n, 24), generator=gen)
            out = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            return dict(out, labels=(torch.rand(n, len(INTENTS), generator=gen) > 0.9).float()) if with_labels \
                else out
    model = model.to(device=device, dtype=torch.float32)
    require_param_device(model, device.type)
    opt = torch.optim.AdamW(model.parameters(), lr=BERTURK["lr"], weight_decay=BERTURK["weight_decay"],
                            foreach=False, fused=False)
    _, sync, sample, _, peak = memory_sampler(device)
    model.train()

    def step(i):
        b = make(i, batch_n)
        sync()
        t0 = time.perf_counter()
        b = {k: v.to(device) for k, v in b.items()}
        loss = model(**b).loss
        loss.backward()
        sample()
        torch.nn.utils.clip_grad_norm_(model.parameters(), BERTURK["clip_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        dt = time.perf_counter() - t0
        sample()
        return dt

    for i in range(cfg["warmup_steps"]):
        step(i)
    times = [step(cfg["warmup_steps"] + i) for i in range(cfg["measured_steps"])]
    model.eval()
    b = make(0, cfg["infer_msgs"], with_labels=False)
    with torch.no_grad():
        sync()
        t0 = time.perf_counter()
        model(**{k: v.to(device) for k, v in b.items()})
        sync()
        infer_s = time.perf_counter() - t0
    sample()
    s = spike.summarize(times)
    return {"status": "ok", "source": cfg["source"], "device": device.type,
            "param_device": ",".join(sorted({p.device.type for p in model.parameters()})),
            "dtype": str(next(model.parameters()).dtype), "n_params": sum(p.numel() for p in model.parameters()),
            "torch_threads": torch.get_num_threads(), "batch": batch_n, "max_len": max_len,
            "warmup_steps": cfg["warmup_steps"], "s_per_step": s["median_s"], "p90_s": s["p90_s"], "n_steps": s["n"],
            "infer_msgs": cfg["infer_msgs"], "infer_s": infer_s, "s_per_msg_infer": infer_s / cfg["infer_msgs"],
            "peak_total_bytes": peak["total"], "peak_driver_bytes": peak["driver"], "peak_rss_bytes": peak["rss"],
            "basis": BERTURK_BASIS}


def run_child(mode, path):
    """--child entry point: run the BERTurk timing child and write its payload to cfg["out_json"]; a refusal, an MPS
    out-of-memory error or any other failure becomes a payload status (refused, oom, error), never a lost run."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if mode != "berturk":
        raise SystemExit("--child mode must be 'berturk', got %r" % (mode,))
    try:
        payload = berturk_child(cfg)
    except SystemExit as e:
        payload = {"status": "refused", "reason": spike.scrub(str(e))}
    except Exception as e:
        oom = isinstance(e, RuntimeError) and spike.is_mps_oom(str(e))
        payload = {"status": "oom" if oom else "error", "reason": spike.scrub("%s: %s" % (type(e).__name__, e))}
    write_atomic(cfg["out_json"], (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def run_berturk(a, entry, berturk_dir, env):
    """The BERTurk timing child in a fresh process with the probe's MPS env; its payload (status ok, or why not)."""
    d = tempfile.mkdtemp(prefix="compute-berturk-")
    try:
        cfg = {"source": a.berturk_source, "dir": berturk_dir, "data": os.path.abspath(a.data),
               "data_sha256": entry["sha256"], "device": a.device, "cpus": a.cpus, "seed": a.seed,
               "warmup_steps": BERTURK["warmup_steps"], "measured_steps": BERTURK["measured_steps"],
               "infer_msgs": BERTURK["infer_msgs"], "out_json": os.path.join(d, "berturk.json")}
        path = os.path.join(d, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)
        print("BERTurk timing child (%s weights, %s)" % (a.berturk_source, a.device), flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "berturk", path], cwd=REPO_ROOT,
                           env=env, shell=False)
        if not os.path.isfile(cfg["out_json"]):
            return {"status": "crashed", "returncode": r.returncode}
        with open(cfg["out_json"], encoding="utf-8") as f:
            return json.load(f)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_score_probe(a, base, env):
    """Score a --n 32 subset of the manifest's val split with the --base weights on --device, in-process, into a temp
    out-root that is deleted afterwards; only the sidecar's timing and scorer fields are read (T-04-21: never test,
    HOTELS or a fine-tuned checkpoint, never a metric)."""
    with open(a.manifest, encoding="utf-8") as f:
        val = json.load(f)["splits"]["val"]
    split = os.path.join(os.path.dirname(os.path.abspath(a.manifest)), val["path"])
    root = tempfile.mkdtemp(prefix="compute-score-")
    try:
        cmd = [sys.executable, os.path.join(REPO_ROOT, "score.py"), split, "--model", "laya", "--checkpoint", base,
               "--device", a.device, "--workers", "0", "--cpus-per-worker", str(a.cpus), "--n", str(SCORE_PROBE_N),
               "--manifest", os.path.abspath(a.manifest), "--out-root", root]
        print("scoring probe: --n %d of val on %s" % (SCORE_PROBE_N, a.device), flush=True)
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, shell=False)
        sidecars = glob.glob(os.path.join(root, "*", "val.json"))
        if r.returncode != 0 or len(sidecars) != 1:
            lines = [ln.strip() for ln in (r.stderr or r.stdout).splitlines() if ln.strip()]
            return {"status": "failed", "returncode": r.returncode,
                    "reason": spike.scrub(lines[-1] if lines else "no sidecar written")}
        with open(sidecars[0], encoding="utf-8") as f:
            side = json.load(f)
        n, sc = side["n_subset"], side["scorer"]
        return {"status": "ok", "device": sc["device"], "dtype": sc["dtype"], "amp_enabled": sc["amp_enabled"],
                "n": n, "s_per_msg": side["timing"]["inference_seconds"] / n, "basis": SCORING_BASIS}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def build_meta(a, entry, start, model, rid):
    return {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split": {"path": entry["path"], "sha256": entry["sha256"], "n_messages": entry["n_messages"]},
            "manifest": {"path": repo_relpath(a.manifest), "sha256": sha256_file(a.manifest),
                         "is_committed": manifest_is_committed(a.manifest)},
            "spike_report": {"path": repo_relpath(a.spike_report), "sha256": sha256_file(a.spike_report)},
            # the pin (or the local checkpoint's hashes) identifies the weights; no local path enters the report
            "model": model,
            "versions": dict({k: importlib.metadata.version(k) for k in VERSIONED}, python=platform.python_version()),
            "platform": {"macos": platform.mac_ver()[0], "machine": platform.machine(),
                         "total_memory_bytes": psutil.virtual_memory().total},
            "protocol": {"device": a.device, "epochs": a.epochs, "seed": a.seed, "cpus": a.cpus, "neg": NEG,
                         "spike_start": start, "run_id": rid,
                         "interrupt": {"mode": a.interrupt, "kill_delay_s": a.kill_delay_s},
                         "berturk": a.berturk, "score_probe": a.score_probe,
                         "test_hooks": {"max_messages": a.max_messages, "calib_frac": a.calib_frac,
                                        "berturk_source": a.berturk_source},
                         "protocol_complete": protocol_complete(a, start)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="report dir (default results/compute)")
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--data", default=DEFAULT_DATA, help="training split (default data/splits/selection_train.jsonl)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--base", default="multilingual", help="'multilingual' (pinned) or a local Laya checkpoint dir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpus", type=int, default=4, help="CPUs and torch threads of the probe worker (the spike's 4)")
    ap.add_argument("--storage", default=None,
                    help="Ray run storage (default: a fresh compute-* dir under the repo's ray_results/)")
    ap.add_argument("--min-available-gib", type=float, default=7.0,
                    help="refuse to measure below this much available memory (the Phase 2 run policy); 0 records a "
                         "busy-machine run")
    ap.add_argument("--interrupt", choices=("kill", "fail", "none"), default="kill",
                    help="kill (default): SIGKILL the probe's process tree --kill-delay-s into epoch 2 and rerun the "
                         "same command to resume; fail: ray_train.py's --fail-after-epoch 1 (deterministic, for the "
                         "CPU test); none: run once")
    ap.add_argument("--kill-delay-s", type=float, default=60.0,
                    help="seconds into epoch 2 (after epoch 1's checkpoint is registered) before the kill")
    ap.add_argument("--keep-run", action="store_true",
                    help="keep the probe's checkpoint and Ray storage instead of deleting them after the report")
    ap.add_argument("--spike-report", default=DEFAULT_SPIKE, help="the Phase 2 spike report holding verdict.chosen")
    ap.add_argument("--berturk", action=argparse.BooleanOptionalAction, default=True,
                    help="time BERTurk in a fresh child on the pinned weights (default on; off keeps the estimates)")
    ap.add_argument("--score-probe", action=argparse.BooleanOptionalAction, default=True,
                    help="measure the Laya scoring rate on a --n 32 val subset on --device (default on)")
    ap.add_argument("--max-messages", type=int, default=None, help=argparse.SUPPRESS)   # test hook
    ap.add_argument("--calib-frac", type=float, default=None, help=argparse.SUPPRESS)   # test hook
    ap.add_argument("--berturk-source", choices=("pinned", "tiny"), default="pinned", help=argparse.SUPPRESS)  # test
    ap.add_argument("--child", nargs=2, metavar=("MODE", "CONFIG"), default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        run_child(*a.child)
        return
    if a.epochs < 1:
        ap.error("--epochs must be at least 1")
    if a.interrupt != "none" and a.epochs < 2:
        ap.error("--interrupt needs --epochs 2 or more (it interrupts after epoch 1); or pass --interrupt none")
    if not a.kill_delay_s >= 0:
        ap.error("--kill-delay-s must not be negative")
    if a.cpus < 1:
        ap.error("--cpus must be at least 1")
    if not a.min_available_gib >= 0:
        ap.error("--min-available-gib must not be negative")
    if a.max_messages is not None and a.max_messages < 1:
        ap.error("--max-messages must be at least 1")
    if a.calib_frac is not None and not 0 < a.calib_frac <= 1:
        ap.error("--calib-frac must be in (0, 1]")

    pf = preflight(a.min_available_gib)
    entry = verify_split_file(a.data, manifest_path=a.manifest, roles=("train",))
    start = compute.spike_start(a.spike_report)
    la.check_version()                                      # the adapter's laya pin, before the fetch
    if a.base == "multilingual":
        base, model = "multilingual", {"repo": pins.LAYA["repo"], "subfolder": pins.LAYA["subfolder"],
                                       "revision": pins.LAYA["revision"],
                                       "model_sha256": pins.LAYA["sha256"]["model.safetensors"]}
        base_dir = fetch(pins.LAYA)                         # pinned and sha256-verified before any child
    elif os.path.isfile(os.path.join(a.base, "rl_agent_config.json")):
        base = base_dir = os.path.abspath(a.base)
        model = identity(base)                              # hashes only: no local path enters the report
    else:
        raise SystemExit("--base must be 'multilingual' (pinned) or a local Laya checkpoint dir: %s" % a.base)
    model_bytes = os.path.getsize(os.path.join(base_dir, "model.safetensors"))
    # BERTurk at the full pinned revision, sha256-verified on the driver before the long Laya probe (the child verifies
    # the same digests again before it loads anything)
    berturk_dir = fetch(pins.BERTURK) if a.berturk and a.berturk_source == "pinned" else None

    own_storage = a.storage is None
    ray_results = os.path.join(REPO_ROOT, "ray_results")
    disk_preflight(ray_results if own_storage else a.storage, model_bytes, a.epochs)   # before any child
    if own_storage:
        os.makedirs(ray_results, exist_ok=True)
        storage = tempfile.mkdtemp(prefix="compute-", dir=ray_results)
    else:
        storage = os.path.abspath(a.storage)
    tmp = tempfile.mkdtemp(prefix="compute-probe-")
    env = dict(os.environ, **spike.spike_env("0", a.cpus))
    ctx = {"a": a, "start": start, "base": base, "storage": storage, "tmp": tmp, "env": env, "rids": [], "logs": [],
           "result": {}}
    try:
        attempts, status = compute.attempt_chain(start["rung"],
                                                 oom_dropping(functools.partial(launch_rung, ctx=ctx), ctx))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        drop_storage(storage, own_storage, ctx["rids"])
        raise

    rid = attempts[-1]["run_id"]
    meta = build_meta(a, entry, start, model, rid)
    if status != "ok":
        report = compute.build_report(meta, None, None, compute.MATRIX, None, pf, attempts=attempts, status=status)
        md = compute.write_report(a.out, report)
        if not a.keep_run:
            drop_storage(storage, own_storage, ctx["rids"])
        raise SystemExit("probe %s after %d attempt(s) (last rung %s); report -> %s; logs kept in %s. Last %d lines "
                         "of %s:\n%s" % (status, len(attempts), attempts[-1]["rung"], md, tmp, TAIL_LINES,
                                          ctx["logs"][-1], tail(ctx["logs"][-1])))

    ok = attempts[-1]
    measured, ckpt = ctx["result"]["measured"], ctx["result"]["ckpt"]
    if a.berturk or a.score_probe:
        wait_for_memory(a.min_available_gib)               # the Laya probe's memory is back before the next child
    if a.berturk:
        measured["berturk"] = run_berturk(a, entry, berturk_dir, env)
        if measured["berturk"].get("status") != "ok":
            print("BERTurk timing child: %s; the BERTurk lines keep the estimate" % measured["berturk"], flush=True)
    if a.score_probe:
        measured["scoring"] = run_score_probe(a, base, env)
        if measured["scoring"].get("status") != "ok":
            print("scoring probe: %s; the Laya scoring lines keep the committed CPU rate" % measured["scoring"],
                  flush=True)
    counts = compute.split_counts(a.manifest)              # all five registered splits, verified bytes
    projection = compute.project(measured, counts, compute.MATRIX)
    report = compute.build_report(meta, measured, counts, compute.MATRIX, projection, pf,
                                  what_if=compute.what_if(measured, counts, compute.MATRIX),
                                  disk=compute.disk_budget(model_bytes, compute.MATRIX,
                                                           measured["laya"]["train_state_bytes"]),
                                  assumptions=compute.assumptions(compute.MATRIX, measured, counts, ok["interrupt"]),
                                  attempts=attempts, status=status, interrupt=ok["interrupt"], resume=ok["resume"])
    md = compute.write_report(a.out, report)
    print("report -> %s" % md)
    print("projected total: %.2f h over %d lines; resume demonstrated: %s"
          % (projection["total_h"], len(projection["lines"]), ok["resume"]["resume_demonstrated"]), flush=True)
    if a.keep_run:
        print("kept: probe checkpoint %s, Ray run dir(s) under %s" % (ckpt, storage))
    else:
        shutil.rmtree(tmp, ignore_errors=True)
        drop_storage(storage, own_storage, ctx["rids"])


if __name__ == "__main__":
    main()
