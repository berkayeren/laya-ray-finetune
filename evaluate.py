"""Evaluate a Laya checkpoint on a registered val/test split: a thin wrapper over score.py + report.py.

Kept so the existing evaluate.py commands keep working. Scoring is score.py (the only place a model runs: split
verification, pinned checkpoint resolution, Ray Data or in-process inference, the logit dump); metrics are report.py
(numpy only, from the verified dump). This file forwards its flags to score.main (--model laya), then prints
report.run_report on the dump it wrote, so its output equals `report.py RUN_DIR --split SPLIT` for the same run dir.
It holds no metric, threshold or model-loading code of its own (tests/test_consumers.py checks this).

Every refusal comes from score.py, before any model load or ray.init: a tampered, unregistered, missing or wrong-role
split (only test and val are scored), --n 0 or negative, and a --checkpoint that is neither 'multilingual' (the
pinned, sha256-verified hub weights) nor a local Laya checkpoint dir. The dump lands under --out-root (default
results/dumps/RUN_ID); with --n it is a debug subset and goes to a fresh temp dir (never results/dumps). The report's
val-selected θ needs a full val dump in the same run dir (score val first); otherwise that row is null with a reason.

usage: python evaluate.py data/splits/test.jsonl [--checkpoint multilingual|DIR] [--n N] [--workers W]
                          [--cpus-per-worker C] [--gpus-per-worker G] [--out-root DIR] [--out report.json]
                          [--manifest splits.json]
  python evaluate.py data/splits/test.jsonl --checkpoint ./ckpt --workers 0 --n 20
  python evaluate.py data/splits/test.jsonl --checkpoint multilingual --workers 4 --gpus-per-worker 1 --out results/x.json
"""
import argparse
import json
import os

import report
import score
from pilot.data import DEFAULT_MANIFEST, write_atomic


def main(argv=None):
    ap = argparse.ArgumentParser(description="score a Laya checkpoint on a val/test split and print its report")
    ap.add_argument("split_jsonl")
    ap.add_argument("--checkpoint", default="multilingual",
                    help="multilingual (pinned hub weights) or DIR (local Laya checkpoint)")
    ap.add_argument("--n", default=None, help="score a fixed random subset of N messages (debug only)")
    ap.add_argument("--workers", default="1", help="Ray Data actors; 0 = in-process, no Ray")
    ap.add_argument("--cpus-per-worker", default="2")
    ap.add_argument("--gpus-per-worker", default="0")
    ap.add_argument("--out-root", default=None, help="dump root (default results/dumps; with --n a temp dir)")
    ap.add_argument("--out", default=None, help="also write the report JSON here")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    a = ap.parse_args(argv)
    # every value goes to score.py unparsed, so its checks, exit codes and messages stay the only ones
    score_argv = [a.split_jsonl, "--model", "laya", "--checkpoint", a.checkpoint, "--workers", a.workers,
                  "--cpus-per-worker", a.cpus_per_worker, "--gpus-per-worker", a.gpus_per_worker,
                  "--manifest", a.manifest]
    score_argv += ["--n", a.n] if a.n is not None else []
    score_argv += ["--out-root", a.out_root] if a.out_root is not None else []
    run_dir, split_name = score.main(score_argv)
    rep = report.run_report(run_dir, split_name, a.manifest)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        write_atomic(a.out, (text + "\n").encode("utf-8"))
    return rep


if __name__ == "__main__":
    main()
