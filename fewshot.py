"""The few-shot runner of the frozen protocol (REP-01; ROADMAP Phase 8 criterion 3, Phase 9 criterion 1).

protocol.json's "fewshot" block (pilot.fewshot.protocol_block, frozen by plan 08-10) is the whole few-shot protocol:
the held-out tags, k values, draw seeds, negatives per positive, replay size, the source split and the three frozen
support-set draws with their uid lists and digests. Phase 9 trains and scores on those frozen lists only, unchanged.

Phase 8 ships one mode, --support-sets, the runner's entry point (08-RESEARCH Open Question 3). Phase 9 adds the base,
incremental and scoring modes to this entry point; every mode calls pilot.protocol.require_frozen() before anything
else and records the protocol_sha256 it ran under, so the runner refuses a missing, unpinned or changed protocol.json
before it reads a split or loads a model (ROADMAP Phase 8 criterion 1).

--support-sets, in this order:
1. pilot.protocol.require_frozen(): protocol.json exists, PROTOCOL_SHA256 is set, its bytes hash to the pin and parse
   to a protocol/1 object; SystemExit otherwise, before any split is read.
2. the manifest's selection_train bytes, read once and verified (pilot.data.verify_split_file, role train, the exact
   bytes parsed); the entry's sha256 and uids_sha256 must be the block's source, so the rows are the frozen source.
3. pilot.fewshot.check_draws(block, rows): every draw regenerates to exactly its frozen uid lists, realised
   positives and digest; SystemExit naming the draw otherwise.
4. prints, as JSON, the protocol sha256, the source, and per draw its number, seed, base seed, digest, support sizes
   per k, replay size and realised positives; never a uid list or message text.

--counts (09-01), in this order: require_frozen() first; the verified source rows (read_source); pilot.fewshot.
check_source (the frozen source and every frozen draw re-checked over those bytes, D-02); then counts(block, rows),
printed as JSON: the masked-base and per-(draw, k) row, item and optimizer-step counts of both trainers, computed from
the frozen uid lists only (the D-08 projection consumes them). Counts only, never a uid list or message text.

usage:
    python fewshot.py --support-sets                     # refused until protocol.json is frozen (plan 08-10)
    python fewshot.py --support-sets --manifest OTHER_DIR/splits.json
    python fewshot.py --counts                           # rows, items and steps of every Phase 9 training run
"""
import argparse
import json
import math
import os

import pilot.data
from pilot import fewshot, protocol, runs, spike
from pilot.data import parse_jsonl, verify_split_file

SOURCE_SPLIT = fewshot.SOURCE_SPLIT


def read_source(block, manifest_path):
    """The verified rows of the block's source split: the manifest's selection_train bytes, read once, verified and
    bound to the block's source sha256 and uids_sha256; SystemExit naming the source otherwise."""
    return read_source_entry(block, manifest_path)[1]


def read_source_entry(block, manifest_path):
    """(manifest entry, verified rows) of the block's source split, as read_source; a missing manifest is refused in
    one line (08-REVIEW IN-07)."""
    src = block.get("source") or {}
    if src.get("split") != SOURCE_SPLIT:
        raise SystemExit("the frozen few-shot source is %r, not %s; the protocol draws from %s only"
                         % (src.get("split"), SOURCE_SPLIT, SOURCE_SPLIT))
    if not os.path.isfile(manifest_path):
        raise SystemExit("%s not found; run `make data` to write the committed manifest and splits, or pass "
                         "--manifest PATH" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    entry = (m.get("splits") or {}).get(SOURCE_SPLIT)
    if not entry:
        raise SystemExit("%s registers no %s; run `make data`" % (manifest_path, SOURCE_SPLIT))
    path = os.path.join(os.path.dirname(os.path.realpath(manifest_path)), entry["path"])
    if not os.path.isfile(path):
        raise SystemExit("%s not found; run `make data` to write the registered splits" % path)
    with open(path, "rb") as f:
        raw = f.read()
    verify_split_file(path, manifest_path=manifest_path, roles=("train",), data=raw)
    for key in ("sha256", "uids_sha256"):
        if entry.get(key) != src.get(key):
            raise SystemExit("%s has %s %s, the frozen few-shot source is %s; the draws are bound to the source split "
                             "the protocol froze (run `make data`, or pass the manifest it was frozen from)"
                             % (path, key, entry.get(key), src.get(key)))
    return entry, parse_jsonl(raw)


def frozen_block():
    """(fewshot block, pin) of require_frozen(), the first call of every mode: nothing is read without it."""
    proto, pin = protocol.require_frozen()
    block = proto["fewshot"]
    if not isinstance(block, dict):
        raise SystemExit("protocol.json's fewshot block is not an object; restore it with `git checkout -- "
                         "protocol.json`")
    return block, pin


def support_sets(manifest_path):
    """--support-sets: the frozen draws, re-checked against the verified source, as a printable summary."""
    block, pin = frozen_block()                                 # first: nothing is read without the frozen protocol
    rows = read_source(block, manifest_path)
    fewshot.check_draws(block, rows)
    out = {"protocol_sha256": pin, "source": block["source"],
           "draws": [{"draw": d["draw"], "seed": d["seed"], "base_seed": d["base_seed"], "digest": d["digest"],
                      "n_support": {k: len(v) for k, v in d["support"].items()}, "n_replay": len(d["replay"]),
                      "positives": d["positives"]} for d in block["draws"]]}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def counts(block, rows):
    """The rows, items and optimizer steps of every Phase 9 training run, from the frozen uid lists over `rows` (the
    verified source): the masked base (every message, the held-out keys deleted before neg sampling; Laya items by
    pilot.runs.neg_items, steps by pilot.spike.optimizer_steps at the frozen micro-batch and grad-accum; BERTurk
    steps ceil(n / batch)) and per (draw, k), in draw then k order, the support and replay rows, the Laya items of
    the frozen item rule (pilot.fewshot.laya_item_count) and both trainers' steps per epoch. protocol_sha256 is left
    for the caller to fill."""
    tags, base = block["tags"], block["rules"]["base_runs"]
    laya, bert = base["laya"], base["berturk"]
    mb, ga, neg, batch = laya["micro_batch"], laya["grad_accum"], laya["neg"], bert["batch"]
    items = runs.neg_items(fewshot.masked_rows(rows, tags), neg)
    out = {"protocol_sha256": None, "source": block["source"],
           "base": {"messages": len(rows), "laya_items": items,
                    "laya_steps_per_epoch": spike.optimizer_steps(items, mb, ga),
                    "berturk_steps_per_epoch": math.ceil(len(rows) / batch)},
           "continued": [], "laya_rows_rule": fewshot.LAYA_ROWS_RULE}
    for i in range(len(block["draws"])):
        for k in block["k_values"]:
            sel = fewshot.selection(block, i, k)
            n_items = fewshot.laya_item_count(rows, sel, tags, neg)
            n_bert = len(fewshot.berturk_rows(rows, sel, tags))
            out["continued"].append({"draw": i, "k": k, "support_rows": len(sel["support"]),
                                     "replay_rows": len(sel["replay"]), "laya_items": n_items,
                                     "laya_steps_per_epoch": spike.optimizer_steps(n_items, mb, ga),
                                     "berturk_messages": n_bert,
                                     "berturk_steps_per_epoch": math.ceil(n_bert / batch)})
    return out


def counts_mode(manifest_path):
    """--counts: require_frozen first, the verified source, check_source, then counts(), printed as JSON."""
    block, pin = frozen_block()                                 # first: nothing is read without the frozen protocol
    entry, rows = read_source_entry(block, manifest_path)
    fewshot.check_source(block, entry, rows)
    out = counts(block, rows)
    out["protocol_sha256"] = pin
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--support-sets", action="store_true",
                      help="re-check the frozen support-set draws against the verified source and print their digests")
    mode.add_argument("--counts", action="store_true",
                      help="print the rows, items and optimizer steps of every Phase 9 training run")
    ap.add_argument("--manifest", default=None, help="the splits manifest (default data/splits.json)")
    a = ap.parse_args(argv)
    manifest = a.manifest or pilot.data.DEFAULT_MANIFEST
    if a.counts:
        return counts_mode(manifest)
    if not a.support_sets:
        ap.error("pass a mode: --support-sets or --counts (Phase 9 adds the training and scoring modes)")
    return support_sets(manifest)


if __name__ == "__main__":
    main()
