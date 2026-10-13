"""Build the frozen Multi3NLU++ Turkish banking splits (Laya JSONL) and the data/splits.json manifest.

Downloads uoe-nlp/multi3-nlu at the pinned commit (or reads --data-dir), verifies all 40 raw fold sha256s
(pilot/pins.py), and writes four banking splits under OUT/splits/:

  selection_train  folds 0-13   1,450 messages (66 no-intent)   role train
  val              folds 14-15    208 messages  (7 no-intent)   role val
  final_train      folds 0-15   1,658 messages (73 no-intent)   role train
  test             folds 16-19    413 messages (15 no-intent)   role test

Every row labels all 48 tags (one `noul` question per tag). No-intent messages are kept as all-negative rows.
There is no prepare-time negative sampling: neg-K sampling happens at train time, per run seed (Phase 4).
If OUT/splits.json already exists and the freshly built manifest differs, nothing is written; --rewrite-manifest
intentionally refreezes it (and invalidates downstream work).

usage: python prepare_data.py [--data-dir DIR] [--out data] [--rewrite-manifest]
"""
import argparse

from pilot import data, pins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="local snapshot root holding turkish/banking/foldN.json and turkish/hotels/foldN.json; "
                         "must match the pinned raw sha256s (default: download the pinned revision)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--rewrite-manifest", action="store_true",
                    help="intentionally refreeze OUT/splits.json when the rebuilt manifest differs")
    a = ap.parse_args()
    root = a.data_dir if a.data_dir else data.fetch_raw()
    data.prepare(root, a.out, pins.RAW_SHA256, rewrite_manifest=a.rewrite_manifest)


if __name__ == "__main__":
    main()
