"""Postfit CLI (LAYA-05, LAYA-08): the selection val dumps -> one record with the val-fitted temperature, the
val-selected thresholds and the chosen (encoder LR, epoch). numpy and scipy only: no torch, Ray, laya or model.

Every number comes from verified val dumps (pilot.dumps.load_dump) and the verified val split's gold labels, so the
record rebuilds byte for byte from the committed dumps, in any order of the RUN_DIR arguments. Test input is refused:
a file path (a .jsonl, an .npz), a run dir holding a test dump, a --n subset val dump, a candidate trained on
final_train (its val is in-sample), a calibrated candidate and one whose shipped T is not 1.0 (pilot.postfit).
The rules, the grid and the seeds are the constants of pilot.postfit, as the user confirmed them at plan 05-10.

usage: python postfit.py results/dumps/RUN_ID [RUN_ID2 ...] [--out results/postfit/laya.json] [--manifest splits.json]
"""
import argparse
import json

from pilot import postfit
from pilot.data import DEFAULT_MANIFEST


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit T, θ and the (encoder LR, epoch) choice on selection val dumps")
    ap.add_argument("run_dir", nargs="*", help="run dirs that each hold one candidate's full val dump")
    ap.add_argument("--out", default=None, help="write the record here (e.g. results/postfit/laya.json)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    a = ap.parse_args(argv)
    if not a.run_dir:
        ap.error("give the selection run dirs (RUN_DIR [RUN_DIR ...])")
    record = postfit.fit(a.run_dir, a.manifest)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if a.out:
        postfit.write_record(record, a.out)
    return record


if __name__ == "__main__":
    main()
