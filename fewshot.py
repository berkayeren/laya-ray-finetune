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

usage:
    python fewshot.py --support-sets                     # refused until protocol.json is frozen (plan 08-10)
    python fewshot.py --support-sets --manifest OTHER_DIR/splits.json
"""
import argparse
import json
import os

import pilot.data
from pilot import fewshot, protocol
from pilot.data import parse_jsonl, verify_split_file

SOURCE_SPLIT = "selection_train"


def read_source(block, manifest_path):
    """The verified rows of the block's source split: the manifest's selection_train bytes, read once, verified and
    bound to the block's source sha256 and uids_sha256; SystemExit naming the source otherwise."""
    src = block.get("source") or {}
    if src.get("split") != SOURCE_SPLIT:
        raise SystemExit("the frozen few-shot source is %r, not %s; the protocol draws from %s only"
                         % (src.get("split"), SOURCE_SPLIT, SOURCE_SPLIT))
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
    return parse_jsonl(raw)


def support_sets(manifest_path):
    """--support-sets: the frozen draws, re-checked against the verified source, as a printable summary."""
    proto, pin = protocol.require_frozen()                      # first: nothing is read without the frozen protocol
    block = proto["fewshot"]
    if not isinstance(block, dict):
        raise SystemExit("protocol.json's fewshot block is not an object; restore it with `git checkout -- "
                         "protocol.json`")
    rows = read_source(block, manifest_path)
    fewshot.check_draws(block, rows)
    out = {"protocol_sha256": pin, "source": block["source"],
           "draws": [{"draw": d["draw"], "seed": d["seed"], "base_seed": d["base_seed"], "digest": d["digest"],
                      "n_support": {k: len(v) for k, v in d["support"].items()}, "n_replay": len(d["replay"]),
                      "positives": d["positives"]} for d in block["draws"]]}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--support-sets", action="store_true",
                    help="re-check the frozen support-set draws against the verified source and print their digests")
    ap.add_argument("--manifest", default=None, help="the splits manifest (default data/splits.json)")
    a = ap.parse_args(argv)
    if not a.support_sets:
        ap.error("pass a mode: --support-sets (Phase 9 adds the training and scoring modes)")
    return support_sets(a.manifest or pilot.data.DEFAULT_MANIFEST)


if __name__ == "__main__":
    main()
