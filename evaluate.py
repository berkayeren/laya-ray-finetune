"""Evaluate a Laya checkpoint on Multi3NLU++ Turkish banking (multi-label, 48 noul questions).

Inference is distributed with Ray Data: each worker actor loads the checkpoint
once and scores batches of messages; with --workers 0 it runs in-process.

The input must be a split registered in data/splits.json with a matching sha256 and role test or val (refused
otherwise; the hotels_probe is scored only on ood_scoring_tags, which this script does not implement yet).
--checkpoint multilingual loads the pinned, sha256-verified hub weights (pilot.weights); any other value must be
a local Laya checkpoint dir, so nothing downloads an unpinned revision.

usage: python evaluate.py data/splits/test.jsonl [--checkpoint multilingual|DIR] [--n N]
                          [--workers W] [--gpus-per-worker G] [--out report.json]
"""
import argparse
import json
import os
import random
import time

from intents import INTENTS, QUESTIONS
from pilot.data import verify_split_file


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
        results = self.agent.predict_batch(list(batch["text"]), QUESTIONS, batch_size=8, sort_by_length=True)
        batch["probs"] = [json.dumps({k: a["noul"] for k, a in r["answers"].items()}) for r in results]
        return batch


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def metrics(golds, probs):
    if not golds:
        raise SystemExit("no messages to score")
    out = {"n": len(golds)}
    out["top1_hit"] = round(sum(max(p, key=p.get) in g for g, p in zip(golds, probs)) / len(golds), 3)
    counts = {k: sum(k in g for g in golds) for k in INTENTS}
    maj = max(counts, key=counts.get)
    out["majority_top1_hit"] = round(counts[maj] / len(golds), 3)
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
    a = ap.parse_args()
    # hard-fail on a tampered/unregistered split (D-03); train splits and the OOD probe (scored only on
    # ood_scoring_tags, a Phase 3 scorer) are refused
    verify_split_file(a.test_jsonl, roles=("test", "val"))

    with open(a.test_jsonl, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f]
    random.Random(0).shuffle(cases)
    cases = cases[:a.n] if a.n else cases
    if not cases:
        raise SystemExit("%s: no messages to evaluate (--n %r)" % (a.test_jsonl, a.n))
    items = [{"uid": c["uid"], "text": c["state"]} for c in cases]
    gold = {c["uid"]: {k for k, v in c["expected"].items() if v} for c in cases}
    device = "cuda" if a.gpus_per_worker else "cpu"

    t0 = time.time()
    if a.workers == 0:
        p = Predictor(a.checkpoint, device)
        rows = []
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            b = p({"uid": [x["uid"] for x in chunk], "text": [x["text"] for x in chunk]})
            rows += [{"uid": u, "probs": s} for u, s in zip(b["uid"], b["probs"])]
    else:
        import ray
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        ds = ray.data.from_items(items).map_batches(
            Predictor, fn_constructor_args=(a.checkpoint, device), batch_size=16,
            concurrency=a.workers, num_gpus=a.gpus_per_worker, memory=3 * 1024**3,
            batch_format="numpy")
        rows = ds.take_all()
    secs = time.time() - t0

    by_uid = {r["uid"]: json.loads(r["probs"]) for r in rows}
    uids = [c["uid"] for c in cases]
    report = {"checkpoint": a.checkpoint, "workers": a.workers, "seconds": round(secs, 1),
              "sec_per_msg": round(secs / len(cases), 2),
              **metrics([gold[u] for u in uids], [by_uid[u] for u in uids])}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
