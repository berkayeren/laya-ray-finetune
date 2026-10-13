"""Evaluate a Laya checkpoint on Multi3NLU++ Turkish banking (multi-label, 48 noul questions).

Inference is distributed with Ray Data: each worker actor loads the checkpoint
once and scores batches of messages; with --workers 0 it runs in-process.

The input must be a split registered in data/splits.json with a matching sha256 and role test or val (refused
otherwise; the hotels_probe is scored only on ood_scoring_tags, which this script does not implement yet).
--checkpoint multilingual loads the pinned, sha256-verified hub weights (pilot.weights); any other value must be
a local Laya checkpoint dir, so nothing downloads an unpinned revision.

--manifest selects the prepare-written splits.json to verify against (default: the committed data/splits.json; use
OTHER_DIR/splits.json for a `prepare_data.py --out OTHER_DIR` variant). Verification itself cannot be skipped, a
manifest cannot change any split's role, path or folds, and the report records manifest_is_committed.

usage: python evaluate.py data/splits/test.jsonl [--checkpoint multilingual|DIR] [--n N]
                          [--workers W] [--gpus-per-worker G] [--out report.json] [--manifest splits.json]
"""
import argparse
import json
import os
import random
import time

from intents import INTENTS, QUESTIONS
from pilot.data import (DEFAULT_MANIFEST, manifest_is_committed, parse_jsonl, repo_relpath, sha256_file,
                        verify_split_file)

SUBSET_SEED = 0                                             # fixed shuffle before --n, so subsets are comparable
# the majority baseline's tag comes from the training split the evaluated role's models are trained on, never from
# the labels being scored (val -> model selection runs, test -> final models; see pilot.data.SPLITS)
MAJORITY_SOURCE = {"val": "selection_train", "test": "final_train"}


def load_agent(checkpoint, device):
    """'multilingual' = pinned, digest-verified hub weights (ENV-02); anything else must be a local checkpoint dir."""
    if checkpoint == "multilingual":
        import laya
        from pilot import pins
        from pilot.weights import fetch
        return laya.load(fetch(pins.LAYA), device=device, expected_sha256=pins.LAYA["sha256"])
    if not os.path.isfile(os.path.join(checkpoint, "rl_agent_config.json")):
        raise SystemExit("--checkpoint must be 'multilingual' (pinned) or a local Laya checkpoint dir: %s" % checkpoint)
    import laya
    return laya.load(os.path.abspath(checkpoint), device=device)   # abspath: laya treats args with "/" as local


class Predictor:
    """Ray Data actor: one checkpoint load per worker, then batched forward passes."""

    def __init__(self, checkpoint, device):
        self.agent = load_agent(checkpoint, device)

    def __call__(self, batch):
        t = time.perf_counter()                             # scoring only: load/digest time is not per-message cost
        results = self.agent.predict_batch(list(batch["text"]), QUESTIONS, batch_size=8, sort_by_length=True)
        batch["probs"] = [json.dumps({k: a["noul"] for k, a in r["answers"].items()}) for r in results]
        dt = time.perf_counter() - t
        batch["infer_secs"] = [dt / len(results)] * len(results)   # each row's share of its batch
        return batch


def majority_tag(manifest_path, role):
    """(source split name, most frequent gold tag in it); ties go to the earlier tag in INTENTS."""
    name = MAJORITY_SOURCE[role]
    with open(manifest_path, encoding="utf-8") as f:
        e = json.load(f).get("splits", {}).get(name)
    if e is None:
        raise SystemExit("%s registers no %s split for the majority baseline" % (manifest_path, name))
    path = os.path.join(os.path.dirname(os.path.realpath(manifest_path)), e["path"])
    if not os.path.isfile(path):
        raise SystemExit("%s (majority baseline source) is missing; run `make data`" % path)
    with open(path, "rb") as f:
        raw = f.read()
    verify_split_file(path, manifest_path=manifest_path, roles=("train",), data=raw)
    counts = dict.fromkeys(INTENTS, 0)
    for r in parse_jsonl(raw):
        for k, v in r["expected"].items():
            counts[k] += bool(v)
    return name, max(INTENTS, key=counts.get)


def provenance(checkpoint, manifest_path, entry, n_subset, majority_source):
    """Report header: model identity, split, manifest and subset. Paths are repo-relative or hashed, never local."""
    from pilot.weights import identity
    return {"checkpoint": checkpoint if checkpoint == "multilingual" else repo_relpath(checkpoint),
            "checkpoint_identity": identity(checkpoint),
            "split": {"path": entry["path"], "sha256": entry["sha256"], "role": entry["role"]},
            "manifest": {"path": repo_relpath(manifest_path), "sha256": sha256_file(manifest_path)},
            "manifest_is_committed": manifest_is_committed(manifest_path),
            "n_subset": n_subset, "subset_seed": SUBSET_SEED, "majority_source": majority_source}


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def metrics(golds, probs, majority):
    """majority: the baseline tag, fixed beforehand from training labels (majority_tag)."""
    if not golds:
        raise SystemExit("no messages to score")
    out = {"n": len(golds)}
    # top-1 is undefined on no-intent rows (no gold tag to hit), so both top-1 rates use the rows with gold tags;
    # no-intent rows still count in the threshold metrics below (exact match = predict nothing)
    tagged = [(g, p) for g, p in zip(golds, probs) if g]
    out["n_top1"], out["n_no_intent"] = len(tagged), len(golds) - len(tagged)
    out["top1_hit"] = round(sum(max(p, key=p.get) in g for g, p in tagged) / len(tagged), 3) if tagged else None
    out["majority_tag"] = majority
    out["majority_top1_hit"] = round(sum(majority in g for g, _ in tagged) / len(tagged), 3) if tagged else None
    counts = {k: sum(k in g for g in golds) for k in INTENTS}
    out["by_threshold"] = {}
    for t in (0.3, 0.5, 0.7, 0.9):
        tp = fp = fn = exact = 0
        for g, p in zip(golds, probs):
            pred = {k for k, v in p.items() if v >= t}
            tp, fp, fn = tp + len(g & pred), fp + len(pred - g), fn + len(g - pred)
            exact += g == pred
        prec, rec, f = f1(tp, fp, fn)
        out["by_threshold"][str(t)] = {"precision": round(prec, 3), "recall": round(rec, 3),
                                       "micro_f1": round(f, 3), "exact_match": round(exact / len(golds), 3)}
    # per-tag F1 at 0.5, the most and least learnable tags
    per = {}
    for k in INTENTS:
        tp = sum(k in g and p[k] >= 0.5 for g, p in zip(golds, probs))
        fp = sum(k not in g and p[k] >= 0.5 for g, p in zip(golds, probs))
        fn = sum(k in g and p[k] < 0.5 for g, p in zip(golds, probs))
        if counts[k]:
            per[k] = round(f1(tp, fp, fn)[2], 3)
    out["macro_f1@0.5"] = round(sum(per.values()) / len(per), 3) if per else None   # None: no gold tag in these rows
    ranked = sorted(per, key=per.get)
    out["best_tags@0.5"] = {k: per[k] for k in ranked[::-1][:5]}
    out["worst_tags@0.5"] = {k: per[k] for k in ranked[:5]}
    out["mean_tags_predicted@0.5"] = round(sum(sum(v >= 0.5 for v in p.values()) for p in probs) / len(probs), 2)
    out["mean_gold_tags"] = round(sum(len(g) for g in golds) / len(golds), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_jsonl")
    ap.add_argument("--checkpoint", default="multilingual",
                    help="multilingual (pinned hub weights) or DIR (local Laya checkpoint)")
    ap.add_argument("--n", type=int, default=None, help="evaluate a fixed random subset of N messages")
    ap.add_argument("--workers", type=int, default=1, help="Ray Data actors; 0 = no Ray")
    ap.add_argument("--gpus-per-worker", type=float, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    a = ap.parse_args()
    if a.n is not None and a.n <= 0:
        ap.error("--n must be a positive number of messages (omit it to evaluate the whole split)")
    # hard-fail on a tampered/unregistered split (D-03); train splits and the OOD probe (scored only on
    # ood_scoring_tags, a Phase 3 scorer) are refused
    if not os.path.isfile(a.test_jsonl):
        raise SystemExit("%s not found; check the path, or run `make data` to write the registered splits"
                         % a.test_jsonl)
    with open(a.test_jsonl, "rb") as f:
        raw = f.read()                                       # hash and parse the same bytes (no reread window)
    entry = verify_split_file(a.test_jsonl, manifest_path=a.manifest, roles=("test", "val"), data=raw)
    majority_source, majority = majority_tag(a.manifest, entry["role"])

    cases = parse_jsonl(raw)
    random.Random(SUBSET_SEED).shuffle(cases)
    cases = cases[:a.n] if a.n is not None else cases
    if not cases:
        raise SystemExit("%s: no messages to evaluate (--n %r)" % (a.test_jsonl, a.n))
    items = [{"uid": c["uid"], "text": c["state"]} for c in cases]
    gold = {c["uid"]: {k for k, v in c["expected"].items() if v} for c in cases}
    device = "cuda" if a.gpus_per_worker else "cpu"
    # n_subset: messages actually scored when --n was given (--n above the split size scores the whole split)
    header = provenance(a.checkpoint, a.manifest, entry, None if a.n is None else len(cases), majority_source)

    load_secs = None                                        # Ray path: actors load inside the timed region
    if a.workers == 0:
        t0 = time.time()
        p = Predictor(a.checkpoint, device)                 # fetch + digest + model build, timed separately
        load_secs = time.time() - t0
    t0 = time.time()
    if a.workers == 0:
        rows = []
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            b = p({"uid": [x["uid"] for x in chunk], "text": [x["text"] for x in chunk]})
            rows += [{"uid": u, "probs": s, "infer_secs": t} for u, s, t in zip(b["uid"], b["probs"], b["infer_secs"])]
    else:
        import ray
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        ds = ray.data.from_items(items).map_batches(
            Predictor, fn_constructor_args=(a.checkpoint, device), batch_size=16,
            concurrency=a.workers, num_gpus=a.gpus_per_worker, memory=3 * 1024**3,
            batch_format="numpy")
        rows = ds.take_all()
    secs = time.time() - t0
    infer_secs = sum(float(r["infer_secs"]) for r in rows)   # summed over actors: compute time, not wall time

    by_uid = {r["uid"]: json.loads(r["probs"]) for r in rows}
    uids = [c["uid"] for c in cases]
    # seconds = wall time of the scoring pass (includes actor start-up and model loads on the Ray path);
    # sec_per_msg = model scoring time per message, excluding checkpoint fetch, digest checks and model build
    report = {**header, "workers": a.workers, "seconds": round(secs, 1),
              "load_seconds": None if load_secs is None else round(load_secs, 1),
              "inference_seconds": round(infer_secs, 1), "sec_per_msg": round(infer_secs / len(cases), 3),
              **metrics([gold[u] for u in uids], [by_uid[u] for u in uids], majority)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
