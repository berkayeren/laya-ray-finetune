"""Convert Multi3NLU++ Turkish banking into Laya JSONL.

Each of the 48 intent tags is one `noul` question (the data is multi-label).
Train = folds 0-15, test = folds 16-19 (the dataset ships 20 folds).

Laya turns every labelled question into one training item, so 48 tags x ~1,600
utterances is ~77k items. `--neg K` keeps all positive tags of a message plus K
random negative tags, which cuts training cost several-fold; unlabelled
questions are simply not trained on. The test file always labels all 48.

usage: python -I prepare_data.py DATA_DIR OUT_DIR [--neg K] [--seed S]
"""
import argparse
import glob
import json
import os
import random

from intents import INTENTS

QUESTIONS = {k: {"type": "noul", "instructions": v} for k, v in INTENTS.items()}


def load(data_dir, folds):
    rows = []
    for i in folds:
        rows += json.load(open(os.path.join(data_dir, "fold%d.json" % i), encoding="utf-8"))
    return [r for r in rows if r.get("intents")]


def to_case(r, rng=None, neg=None):
    gold = set(r["intents"])
    if neg is None:
        labelled = list(INTENTS)
    else:
        negatives = [k for k in INTENTS if k not in gold]
        labelled = sorted(gold) + rng.sample(negatives, min(neg, len(negatives)))
    return {"state": r["text"], "uid": r["uid"],
            "questions": {k: QUESTIONS[k] for k in labelled},
            "expected": {k: k in gold for k in labelled}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--neg", type=int, default=None, help="negatives per message (default: all 48 tags)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    train, test = load(a.data_dir, range(16)), load(a.data_dir, range(16, 20))
    for name, rows, neg in (("train", train, a.neg), ("test", test, None)):
        path = os.path.join(a.out_dir, name + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            n_items = 0
            for r in rows:
                case = to_case(r, rng, neg)
                n_items += len(case["expected"])
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print("%s: %d messages, %d labelled questions -> %s" % (name, len(rows), n_items, path))


if __name__ == "__main__":
    main()
