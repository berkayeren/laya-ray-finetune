"""Time the final seeds of fine-tuned Laya or BERTurk per message, in-process (EVAL-06): reported, not gated.

One run times every final seed of one system (--system laya|berturk), listed in its final manifest (default
results/postfit/<system>-final.json, schema postfit-final/1), on one device (--device cpu|mps|cuda), in this process,
without Ray, at batch 1. Each message goes from its val text to 48 probabilities on the host through the path that
scored the dumps (07-RESEARCH Pattern 5):
- laya: score.resolve_checkpoint (every file bound to the seed's identity), la.check_version, score.Predictor
  (laya.load with digests, la.force_fp32, check_agent), then per message la.noul_logits of the one text (48 noul
  questions, one forward) and pilot.metrics.probabilities at the served T;
- berturk: score.check_berturk (label order, casing, record bytes), score.load_bert (the loader score.py --model
  berturk uses: every digest verified right before the fp32 load, check_bert), then per message the tokenizer of the
  one text (truncated at the record's max_len), one forward, the logits copied to the host and probabilities at T.
Per seed, pilot.latency.WARMUP (10) warm-up messages run first and are excluded; every val message is then timed with
time.perf_counter between a device sync right before t0 and one right after the host copy (torch.mps.synchronize on
mps, torch.cuda.synchronize on cuda, nothing on cpu), under torch.no_grad; fp32 with autocast off is checked after the
load and after the loop. --threads sets the torch threads (default 4). Never score.py's batch-of-16 inference seconds:
those carry no sync and no warm-up.

Refusals, each a SystemExit before torch is imported or any model loads: a split not registered with role val
(verify_split_file), a split holding committed test or HOTELS content, keyed on the content whatever its manifest says
(pilot.data.holds_committed for test and hotels_probe), an empty split, a final manifest that is not postfit-final/1 of
the requested system with at least one seed, a checkpoint_name that is not a plain directory name, a missing seed dir,
and a seed dir whose identity (pilot.weights.identity for Laya, pilot.berturk.identity for BERTurk) or dump run id is
not the manifest's. After the load, a served T other than the seed's temperature (beyond 1e-12) is refused too.

The latency/1 report (pilot.latency.build_report) holds, per seed, the seed, train run id, checkpoint name and identity,
dump run id, served T, load seconds, the checked {device, dtype, amp_enabled} state, the per-message seconds in
split-file order and their summary; a summary pooled over every seed; the split's name, registered path, sha256 and the
uids_sha256 of the timed messages; the final manifest's repo-relative path, sha256 and is_committed; device, threads,
warm-up, versions, PYTORCH_ENABLE_MPS_FALLBACK as set, the git code state, the quiet-machine preflight
(compute_checkpoint.preflight, --min-available-gib; 0 records a busy-machine run) and swap in use before and after the
timed loops. No message text, uid list, timestamp or local path. --render DIR prints pilot.latency.render_markdown of
DIR's latency/1 reports without importing torch (with --out, writes the same bytes). Run one MPS job at a time.

usage: python latency.py --system laya --device mps --out results/latency/laya-mps.json
       python latency.py --system berturk --device cpu [--final-manifest M] [--checkpoint-root DIR] [--split S]
                         [--manifest SPLITS_JSON] [--threads 4] [--warmup 10] [--min-available-gib 7.0] [--out PATH]
       python latency.py --render results/latency [--out results/latency/latency.md]
"""
import argparse
import gc
import json
import os
import time

import numpy as np

import score
from intents import QUESTIONS
from pilot import berturk, dumps, latency, metrics, spike
from pilot import laya_adapter as la
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, holds_committed, manifest_is_committed, parse_jsonl,
                        repo_relpath, sha256_file, verify_split_file, write_atomic)
from pilot.weights import identity

T_TOLERANCE = 1e-12
HELD_OUT = ("test", "hotels_probe")                         # committed content never timed, whatever the manifest says


def default_final_manifest(system):
    return os.path.join(REPO_ROOT, "results", "postfit", "%s-final.json" % system)


def parser():
    ap = argparse.ArgumentParser(description="time the final seeds of one system per message, in-process "
                                             "(reported, not gated)")
    ap.add_argument("--system", choices=latency.SYSTEMS, default=None)
    ap.add_argument("--render", metavar="DIR", default=None,
                    help="print the markdown of DIR's latency/1 reports (no torch); with --out, write it")
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument("--final-manifest", default=None,
                    help="postfit-final/1 manifest of the seeds (default results/postfit/SYSTEM-final.json)")
    ap.add_argument("--checkpoint-root", default=REPO_ROOT,
                    help="directory holding each seed's checkpoint_name (default: the repo root)")
    ap.add_argument("--split", default=os.path.join(REPO_ROOT, "data", "splits", "val.jsonl"),
                    help="a split registered with role val (default data/splits/val.jsonl)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    ap.add_argument("--threads", type=int, default=latency.THREADS, help="torch threads (default 4)")
    ap.add_argument("--warmup", type=int, default=latency.WARMUP,
                    help="warm-up messages per seed, excluded (default 10)")
    ap.add_argument("--min-available-gib", type=float, default=latency.MIN_AVAILABLE_GIB,
                    help="quiet-machine preflight threshold; 0 records a busy-machine run")
    ap.add_argument("--out", default=None, help="write the report (or, with --render, the markdown) here")
    return ap


def preflight(min_available_gib):
    """The compute checkpoint's quiet-machine preflight (compute_checkpoint.preflight: refuse below the threshold,
    sample swap twice 5 s apart while idle); imported here so --render never reads it."""
    import compute_checkpoint
    return compute_checkpoint.preflight(min_available_gib)


def read_split(a):
    """(manifest entry, parsed rows) of --split: registered with role val and holding no committed test or HOTELS
    content, verified on the bytes that are parsed."""
    if not os.path.isfile(a.split):
        raise SystemExit("%s not found; run `make data` to write the registered splits" % a.split)
    with open(a.split, "rb") as f:
        raw = f.read()
    entry = verify_split_file(a.split, manifest_path=a.manifest, roles=("val",), data=raw)
    cases = parse_jsonl(raw)
    if not cases:
        raise SystemExit("%s: no messages to time" % a.split)
    for name in HELD_OUT:
        if holds_committed(entry, cases, name):
            raise SystemExit("%s holds committed %s content; latency is measured on val messages only: pass the "
                             "committed data/splits/val.jsonl" % (a.split, name))
    return entry, cases


def bind_seed(system, root, s):
    """(checkpoint dir, identity) of a manifest seed, refused unless the dir under --checkpoint-root holds exactly the
    checkpoint the manifest names (its identity and dump run id). Hashes only: no model loads here."""
    ckpt = os.path.join(root, s["checkpoint_name"])
    where = "--checkpoint-root/" + s["checkpoint_name"]
    if not os.path.isdir(ckpt):
        raise SystemExit("%s (seed %d) not found; pass --checkpoint-root DIR holding the final seeds the manifest "
                         "names" % (where, s["seed"]))
    if system == "laya":
        ident = identity(ckpt)
        rid = dumps.run_id("laya", ident)
    else:
        berturk.require_files(ckpt, where)
        ident = berturk.identity(ckpt)
        rid = berturk.dump_run_id(ident)
    if ident != s["checkpoint_identity"] or rid != s["dump_run_id"]:
        raise SystemExit("%s is not the checkpoint the final manifest names for seed %d (identity or dump run id "
                         "differs: %s, manifest %s); time only the final seeds the manifest binds"
                         % (where, s["seed"], rid, s["dump_run_id"]))
    return ckpt, ident


def check_temperature(served, s):
    if abs(float(served) - s["temperature"]) > T_TOLERANCE:
        raise SystemExit("%s serves T %r, the final manifest records %r for seed %d; time only the final seeds the "
                         "manifest binds" % (s["checkpoint_name"], served, s["temperature"], s["seed"]))


def time_laya(s, ckpt, ident, device, threads, texts, warmup, sync):
    """(load seconds, served T, per-message seconds, checked state) of one Laya seed, through score.py's load path."""
    import torch
    t = time.perf_counter()
    model_dir, expected = score.resolve_checkpoint(ckpt, ident)
    la.check_version()
    p = score.Predictor(model_dir, device, expected, threads)
    load_s = time.perf_counter() - t
    temp = p.temperature
    check_temperature(temp, s)
    agent = p.agent

    def fn(text):
        return metrics.probabilities(la.noul_logits(agent, [text], QUESTIONS), temp)
    with torch.no_grad():
        times = latency.time_messages(fn, texts, warmup, sync)
    return load_s, temp, times, score.check_agent(agent, device)


def time_berturk(s, ckpt, ident, device, threads, texts, warmup, sync):
    """(load seconds, served T, per-message seconds, checked state) of one BERTurk seed, through score.load_bert."""
    import torch
    t = time.perf_counter()
    rec = score.check_berturk(ckpt, ident)
    temp = float(rec["temperature"])
    check_temperature(temp, s)
    tok, model, _ = score.load_bert(ckpt, ident, device, threads)
    load_s = time.perf_counter() - t
    max_len = rec["canonical"]["max_len"]

    def fn(text):
        enc = tok([text], truncation=True, max_length=max_len, return_tensors="pt")
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float().cpu().numpy()
        return metrics.probabilities(logits, temp)
    with torch.no_grad():
        times = latency.time_messages(fn, texts, warmup, sync)
    return load_s, temp, times, score.check_bert(model, device)


def _no_sync():
    return None


def measure(a):
    system, device = a.system, a.device
    entry, cases = read_split(a)
    mpath = a.final_manifest or default_final_manifest(system)
    manifest, msha = latency.read_final_manifest(mpath, system)
    bound = [(s,) + bind_seed(system, a.checkpoint_root, s) for s in manifest["seeds"]]   # every seed, before any load
    pf = preflight(a.min_available_gib)
    swap0 = spike.swap_used()

    import torch
    import transformers
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps, but MPS is not available here; use --device cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but CUDA is not available here; use --device cpu or mps")
    sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(device, _no_sync)
    texts = [c["state"] for c in cases]                     # unchanged: no lowercasing, folding or normalization
    run = time_laya if system == "laya" else time_berturk
    rows = []
    for s, ckpt, ident in bound:
        load_s, temp, times, state = run(s, ckpt, ident, device, a.threads, texts, a.warmup, sync)
        rows.append({"seed": s["seed"], "train_run_id": s["train_run_id"], "checkpoint_name": s["checkpoint_name"],
                     "checkpoint_identity": ident, "dump_run_id": s["dump_run_id"], "temperature": temp,
                     "load_s": load_s, "state": state, "times_s": times,
                     "summary": latency.summarize(times, "%s seed %d on %s" % (system, s["seed"], device))})
        gc.collect()                                        # free the seed's model before the next load
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    swap1 = spike.swap_used()

    from ray_train import code_state                       # imports torch and ray: only after the timed loops
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}
    if system == "laya":
        import laya
        versions["laya"] = laya.__version__
    default = default_final_manifest(system)
    meta = {"system": system, "device": device, "dtype": "torch.float32", "amp_enabled": False, "threads": a.threads,
            "warmup": a.warmup, "batch": 1, "versions": versions,
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "code": code_state(),
            "preflight": pf, "swap_used_bytes": {"before": swap0, "after": swap1},
            "split": {"name": os.path.splitext(os.path.basename(entry["path"]))[0], "path": entry["path"],
                      "sha256": entry["sha256"], "uids_sha256": dumps.uids_sha256([c["uid"] for c in cases]),
                      "n_messages": len(cases), "manifest_is_committed": manifest_is_committed(a.manifest)},
            "final_manifest": {"path": repo_relpath(mpath), "sha256": msha,
                               "is_committed": os.path.isfile(default) and msha == sha256_file(default)}}
    report = latency.build_report(meta, rows)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if a.out:
        if os.path.dirname(a.out):
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
        write_atomic(a.out, text.encode("utf-8"))
    else:
        print(text, end="")
    p = report["pooled"]
    print("latency %s on %s (%s): pooled p50 %.1f ms, p95 %.1f ms over %d seed(s) x %d message(s)%s"
          % (system, device, latency.LABEL, 1000 * p["p50"], 1000 * p["p95"], len(rows), len(cases),
             " -> %s" % repo_relpath(a.out) if a.out else ""), flush=True)
    return report


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if (a.system is None) == (a.render is None):
        ap.error("pass exactly one of --system (time the final seeds) or --render DIR (markdown of saved reports)")
    if a.threads < 1:
        ap.error("--threads must be >= 1")
    if a.warmup < 0:
        ap.error("--warmup must be >= 0")
    if a.min_available_gib < 0:
        ap.error("--min-available-gib must be >= 0 (0 records a busy-machine run)")
    if a.render is not None:
        text = latency.render_markdown(latency.load_reports(a.render))
        if a.out:
            write_atomic(a.out, text.encode("utf-8"))
        print(text, end="")
        return text
    return measure(a)


if __name__ == "__main__":
    main()
