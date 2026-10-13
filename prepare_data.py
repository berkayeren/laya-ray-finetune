"""Build the frozen Multi3NLU++ Turkish banking splits, the HOTELS OOD probe (Laya JSONL) and data/splits.json.

Downloads uoe-nlp/multi3-nlu at the pinned commit (or reads --data-dir), verifies all 40 raw fold sha256s
(pilot/pins.py), and writes four banking splits plus the out-of-domain probe under OUT/splits/:

  selection_train  folds 0-13   1,450 messages (66 no-intent)   role train
  val              folds 14-15    208 messages  (7 no-intent)   role val
  final_train      folds 0-15   1,658 messages (73 no-intent)   role train
  test             folds 16-19    413 messages (15 no-intent)   role test
  hotels_probe     hotels 0-19  1,009 messages (274 no-intent)  role ood_probe

hotels_probe holds every Turkish HOTELS message with all 48 banking tags expected=false. Hotel messages genuinely
carry the 26 generic tags banking shares (affirm, thank, ...), so only the 22 banking-only tags recorded as
ood_scoring_tags in the manifest are valid gold on it. Prep re-derives that set (INTENTS minus every HOTELS tag)
and hard-fails if it differs from pins.OOD_SCORING_TAGS. A duplicate uid anywhere in the 3,080 raw banking +
HOTELS rows also hard-fails before any file is written (uid-keyed consumers would silently merge rows).

Every row labels all 48 tags (one `noul` question per tag). No-intent messages are kept as all-negative rows
by default; --no-include-no-intent leaves banking ones out (1,384 / 201 / 1,585 / 398 messages; the probe always
keeps all 1,009) and records
options.include_no_intent=false in the manifest. The committed data/splits.json records the default, so write the
flag-off variant to a different --out dir (prepare refuses to overwrite a differing manifest).
There is no prepare-time negative sampling: neg-K sampling happens at train time, per run seed (Phase 4).
If OUT/splits.json already exists and the freshly built manifest differs, nothing is written; --rewrite-manifest
intentionally refreezes it (and invalidates downstream work).

usage: python prepare_data.py [--data-dir DIR] [--out data] [--no-include-no-intent] [--rewrite-manifest]
"""
import argparse

from pilot import data, pins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="local snapshot root holding turkish/banking/foldN.json and turkish/hotels/foldN.json; "
                         "must match the pinned raw sha256s (default: download the pinned revision)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--include-no-intent", action=argparse.BooleanOptionalAction, default=True,
                    help="keep banking messages with no intents as all-negative rows (default: on); "
                         "the choice is recorded in the manifest, so write the off variant to another --out")
    ap.add_argument("--rewrite-manifest", action="store_true",
                    help="intentionally refreeze OUT/splits.json when the rebuilt manifest differs")
    a = ap.parse_args()
    root = a.data_dir if a.data_dir else data.fetch_raw()
    data.prepare(root, a.out, pins.RAW_SHA256, rewrite_manifest=a.rewrite_manifest,
                 include_no_intent=a.include_no_intent, ood_tags=pins.OOD_SCORING_TAGS)


if __name__ == "__main__":
    main()
