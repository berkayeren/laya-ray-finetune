"""Evaluate a Laya checkpoint on Multi3NLU++ Turkish banking (multi-label, 48 noul questions).

Inference is distributed with Ray Data: each worker actor loads the checkpoint
once and scores batches of messages; with --workers 0 it runs in-process.

The input must be a split registered in data/splits.json with a matching sha256 (refused otherwise).

usage: python evaluate.py data/splits/test.jsonl [--checkpoint multilingual|DIR] [--n N]
                          [--workers W] [--gpus-per-worker G] [--out report.json]
"""
import argparse
import json
import random
import time

from intents import INTENTS, QUESTIONS
from pilot.data import verify_split_file


def load_agent(checkpoint, device):
    import laya
    if checkpoint in ("multilingual", "english", "typed-decisions"):
        sub = None if checkpoint == "english" else checkpoint
        return laya.load("convaiinnovations/laya", subfolder=sub, device=device)
    return laya.load(checkpoint, device=device)


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
    out["macro_f1@0.5"] = round(sum(per.values()) / len(per), 3)
    ranked = sorted(per, key=per.get)
    out["best_tags@0.5"] = {k: per[k] for k in ranked[::-1][:5]}
    out["worst_tags@0.5"] = {k: per[k] for k in ranked[:5]}
    out["mean_tags_predicted@0.5"] = round(sum(sum(v >= 0.5 for v in p.values()) for p in probs) / len(probs), 2)
    out["mean_gold_tags"] = round(sum(len(g) for g in golds) / len(golds), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_jsonl")
    ap.add_argument("--checkpoint", default="multilingual")
    ap.add_argument("--n", type=int, default=None, help="evaluate a fixed random subset of N messages")
    ap.add_argument("--workers", type=int, default=1, help="Ray Data actors; 0 = no Ray")
    ap.add_argument("--gpus-per-worker", type=float, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    verify_split_file(a.test_jsonl)                         # hard-fail on a tampered/unregistered split (D-03)

    with open(a.test_jsonl, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f]
    random.Random(0).shuffle(cases)
    cases = cases[:a.n] if a.n else cases
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
