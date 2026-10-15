"""Postfit CLI (LAYA-05, LAYA-08): the selection val dumps -> one record with the val-fitted temperature, the
val-selected thresholds and the chosen (encoder LR, epoch). numpy and scipy only: no torch, Ray, laya or model.

Every number comes from verified val dumps (pilot.dumps.load_dump) and the verified val split's gold labels, so the
record rebuilds byte for byte from the committed dumps, in any order of the RUN_DIR arguments. Test input is refused:
a file path (a .jsonl, an .npz), a run dir holding a test dump, a --n subset val dump, a candidate trained on
final_train (its val is in-sample), a calibrated candidate and one whose shipped T is not 1.0 (pilot.postfit).
The rules, the grid and the seeds are the constants of pilot.postfit, as the user confirmed them at plan 05-10.

--prune EXPORT_DIR... deletes ray_train.py epoch exports (<storage>/<run_id>/exports/epoch-NNN) to free disk, each
only after a verified full val dump of exactly that export exists under --dump-root (pilot.postfit.prune_export);
anything else, a symlink included, is refused and left in place.

usage: python postfit.py results/dumps/RUN_ID [RUN_ID2 ...] [--out results/postfit/laya.json] [--manifest splits.json]
       python postfit.py --prune ray_results/RUN_ID/exports/epoch-001 [...] [--dump-root results/dumps]
"""
import argparse
import json
import os

from pilot import dumps, postfit
from pilot.data import DEFAULT_MANIFEST


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit T, θ and the (encoder LR, epoch) choice on selection val dumps, "
                                             "or prune epoch exports whose val dump exists")
    ap.add_argument("run_dir", nargs="*", help="run dirs that each hold one candidate's full val dump")
    ap.add_argument("--out", default=None, help="write the record here (e.g. results/postfit/laya.json)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    ap.add_argument("--prune", nargs="+", metavar="EXPORT_DIR", default=None,
                    help="delete these epoch exports, each only after its verified full val dump exists")
    ap.add_argument("--dump-root", default=dumps.DUMP_ROOT, help="--prune: where the val dumps are (results/dumps)")
    a = ap.parse_args(argv)
    if sum((bool(a.run_dir), a.prune is not None)) != 1:
        ap.error("give exactly one of RUN_DIR [RUN_DIR ...] or --prune EXPORT_DIR [EXPORT_DIR ...]")
    if a.prune is not None:
        if a.out is not None:
            ap.error("--out applies only to RUN_DIR mode")
        pruned = []
        for d in a.prune:
            rid = postfit.prune_export(d, a.dump_root, a.manifest)
            real = os.path.normpath(os.path.abspath(d))
            print("pruned %s/exports/%s (val dump %s)"
                  % (os.path.basename(os.path.dirname(os.path.dirname(real))), os.path.basename(real), rid))
            pruned.append(rid)
        return pruned
    record = postfit.fit(a.run_dir, a.manifest)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if a.out:
        postfit.write_record(record, a.out)
    return record


if __name__ == "__main__":
    main()
