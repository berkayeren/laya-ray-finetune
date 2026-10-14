"""Phase 4 compute checkpoint (LAYA-07): measure a real Laya training run from the Phase 2 spike's start config and
project the runtime of the whole experiment matrix, for the user to approve before any Phase 5/6 run.

Design (like mps_spike.py): this orchestrator imports no torch. It runs the spike's quiet-machine preflight (available
memory, swap growth while idle), verifies the training split, reads the start config from the committed spike report
(verdict.chosen, FITS only), checks the laya pin and the base weights, and then starts ONE fresh child process,
`ray_train.py --device mps --workers 1` with the spike's micro-batch, max_len and grad-accum, --neg 8 and the spike's
MPS env (strict fallback 0, HIGH and LOW watermarks 1.0). A fresh process, because torch reads
PYTORCH_ENABLE_MPS_FALLBACK once at import and the MPS allocator keeps a process-wide high-water mark. The child's
checkpoint carries the per-epoch run record (pilot.runs.read_record); pilot.compute turns it into the projection
(s_per_step = the largest train_s / optimizer_steps over the measured epochs) and writes compute_checkpoint.json and
compute_checkpoint.md. The probe's checkpoint and Ray storage are deleted afterwards unless --keep-run.

A run off the protocol (not mps, not the pinned weights, not the committed selection_train under the committed
manifest, a start config other than the spike's at rung none, or a test hook) is bannered SMOKE RUN with
meta.protocol.protocol_complete false; a busy-machine preflight is bannered too. Neither is a compute checkpoint.

Run it on AC power, after quitting heavy apps, under `caffeinate -i` (the machine must not sleep mid-epoch).

usage:
  caffeinate -i python compute_checkpoint.py --out results/compute
  python compute_checkpoint.py --device cpu --base CKPT --data SPLIT --manifest M --epochs 1 --min-available-gib 0 \
      --out /tmp/compute   # smoke (bannered SMOKE RUN)
"""
import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from pilot import compute, pins, runs, spike
from pilot import laya_adapter as la
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, repo_relpath, sha256_file, verify_split_file
from pilot.weights import fetch, identity

DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "splits", "selection_train.jsonl")
DEFAULT_SPIKE = os.path.join(REPO_ROOT, "results", "spike", "mps_spike.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "compute")
VERSIONED = ("torch", "ray", "laya", "transformers", "psutil")
NEG = 8
TAIL_LINES = 40


def preflight(min_available_gib):
    """The spike's quiet-machine preflight (mps_spike.main): refuse below --min-available-gib, sample swap twice 5 s
    apart while idle; 0 records an override (a busy-machine run)."""
    import psutil
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
           "--max-len", str(start["max_len"]), "--neg", str(NEG), "--seed", str(a.seed)] + runs.rung_flags(start["rung"])
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


def protocol_complete(a, start):
    """True only for the real compute checkpoint: mps, the pinned weights, the committed selection_train under the
    committed manifest, no test hook, and the spike's start config at rung none."""
    return (a.device == "mps" and a.base == "multilingual"
            and os.path.realpath(a.data) == os.path.realpath(DEFAULT_DATA) and manifest_is_committed(a.manifest)
            and a.max_messages is None and a.calib_frac is None
            and {k: start[k] for k in runs.SPIKE_START} == runs.SPIKE_START and start["rung"] == "none")


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
    ap.add_argument("--keep-run", action="store_true",
                    help="keep the probe's checkpoint and Ray storage instead of deleting them after the report")
    ap.add_argument("--spike-report", default=DEFAULT_SPIKE, help="the Phase 2 spike report holding verdict.chosen")
    ap.add_argument("--max-messages", type=int, default=None, help=argparse.SUPPRESS)   # test hook
    ap.add_argument("--calib-frac", type=float, default=None, help=argparse.SUPPRESS)   # test hook
    a = ap.parse_args(argv)
    if a.epochs < 1:
        ap.error("--epochs must be at least 1")
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
        fetch(pins.LAYA)                                    # pinned and sha256-verified before any child
    elif os.path.isfile(os.path.join(a.base, "rl_agent_config.json")):
        base = os.path.abspath(a.base)
        model = identity(base)                              # hashes only: no local path enters the report
    else:
        raise SystemExit("--base must be 'multilingual' (pinned) or a local Laya checkpoint dir: %s" % a.base)

    own_storage = a.storage is None
    if own_storage:
        os.makedirs(os.path.join(REPO_ROOT, "ray_results"), exist_ok=True)
        storage = tempfile.mkdtemp(prefix="compute-", dir=os.path.join(REPO_ROOT, "ray_results"))
    else:
        storage = os.path.abspath(a.storage)
    tmp = tempfile.mkdtemp(prefix="compute-probe-")
    ckpt, log = os.path.join(tmp, "ckpt"), os.path.join(tmp, "probe.log")
    cmd = child_command(a, start, base, ckpt, storage)
    env = dict(os.environ, **spike.spike_env("0", a.cpus))
    try:
        rid = run_id_of(cmd, env)
        if os.path.exists(os.path.join(storage, rid)):
            # the probe must train from scratch: an existing dir would resume (or return) another measurement
            raise SystemExit("%s already holds run %s; the probe must start fresh: delete that dir or choose another "
                             "--storage" % (storage, rid))
    except SystemExit:
        shutil.rmtree(tmp, ignore_errors=True)
        if own_storage:
            shutil.rmtree(storage, ignore_errors=True)
        raise
    print("probe run %s -> log %s (tail -f it)" % (rid, log), flush=True)
    with open(log, "wb") as f:
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, shell=False)
    if r.returncode != 0:
        raise SystemExit("ray_train.py exited %d; the probe dir %s is kept. Last %d lines of %s:\n%s"
                         % (r.returncode, tmp, TAIL_LINES, log, tail(log)))

    measured = compute.measured_from_checkpoint(ckpt)
    counts = compute.split_counts(a.manifest)
    projection = compute.project(measured, counts, compute.MATRIX)
    import psutil
    meta = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
                         "test_hooks": {"max_messages": a.max_messages, "calib_frac": a.calib_frac},
                         "protocol_complete": protocol_complete(a, start)}}
    report = compute.build_report(meta, measured, counts, compute.MATRIX, projection, pf)
    md = compute.write_report(a.out, report)
    print("report -> %s" % md)
    print("projected total: %.2f h over %d lines" % (projection["total_h"], len(projection["lines"])), flush=True)
    if a.keep_run:
        print("kept: probe checkpoint %s, Ray run dir %s" % (ckpt, os.path.join(storage, rid)))
    else:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(storage if own_storage else os.path.join(storage, rid), ignore_errors=True)


if __name__ == "__main__":
    main()
