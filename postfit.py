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

--print-final-commands RECORD prints one ray_train.py command per final seed (pilot.postfit.final_command), each line
to be prefixed with the interpreter (and caffeinate -i on the Mac); ray_train.py --postfit still refuses any command
that is not the chosen recipe. --final-manifest RECORD CKPT... builds the postfit-final/1 manifest of the finished
final checkpoints (pilot.postfit.final_manifest), refused unless they are exactly the record's final seeds.

usage: python postfit.py results/dumps/RUN_ID [RUN_ID2 ...] [--out results/postfit/laya.json] [--manifest splits.json]
       python postfit.py --prune ray_results/RUN_ID/exports/epoch-001 [...] [--dump-root results/dumps]
       python postfit.py --print-final-commands results/postfit/laya.json --storage ray_results \
                         --out-prefix ckpt/laya-final
       python postfit.py --final-manifest results/postfit/laya.json ckpt/laya-final-s0 ckpt/laya-final-s1 \
                         ckpt/laya-final-s2 --out results/postfit/laya-final.json
"""
import argparse
import json
import os
import shlex

from pilot import dumps, postfit
from pilot.data import DEFAULT_MANIFEST, write_atomic


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit T, θ and the (encoder LR, epoch) choice on selection val dumps, "
                                             "prune epoch exports whose val dump exists, print the final-seed "
                                             "commands or build the final-seed manifest")
    ap.add_argument("run_dir", nargs="*", help="run dirs that each hold one candidate's full val dump")
    ap.add_argument("--out", default=None, help="write the record (or, with --final-manifest, the manifest) here")
    ap.add_argument("--manifest", default=None,
                    help="splits.json to verify against (default: the committed data/splits.json); with "
                         "--print-final-commands it is passed on to ray_train.py only when given")
    ap.add_argument("--prune", nargs="+", metavar="EXPORT_DIR", default=None,
                    help="delete these epoch exports, each only after its verified full val dump exists")
    ap.add_argument("--dump-root", default=dumps.DUMP_ROOT, help="--prune: where the val dumps are (results/dumps)")
    ap.add_argument("--print-final-commands", metavar="RECORD", default=None,
                    help="print the ray_train.py command of every final seed of this postfit record")
    ap.add_argument("--storage", default=None, help="--print-final-commands: the Ray storage dir of the final runs")
    ap.add_argument("--out-prefix", default=None,
                    help="--print-final-commands: final checkpoint dirs are <out-prefix>-s<seed>")
    ap.add_argument("--base", default="multilingual",
                    help="--print-final-commands: the base the chosen run trained (default the pinned weights)")
    ap.add_argument("--final-manifest", nargs="+", metavar="RECORD CKPT", default=None,
                    help="the postfit record, then the final checkpoint dirs of every final seed")
    a = ap.parse_args(argv)
    modes = (bool(a.run_dir), a.prune is not None, a.print_final_commands is not None, a.final_manifest is not None)
    if sum(modes) != 1:
        ap.error("give exactly one of RUN_DIR [RUN_DIR ...], --prune EXPORT_DIR [...], --print-final-commands RECORD "
                 "or --final-manifest RECORD CKPT [...]")
    if a.print_final_commands is None and (a.storage is not None or a.out_prefix is not None
                                           or a.base != "multilingual"):
        ap.error("--storage, --out-prefix and --base apply only to --print-final-commands")
    manifest = a.manifest or DEFAULT_MANIFEST
    if a.print_final_commands is not None:
        if a.out is not None:
            ap.error("--out does not apply to --print-final-commands (use --out-prefix)")
        if a.storage is None or a.out_prefix is None:
            ap.error("--print-final-commands needs --storage DIR and --out-prefix PATH")
        record, _ = postfit.read_record(a.print_final_commands)
        lines = [shlex.join(postfit.final_command(record, seed, a.print_final_commands, a.storage, a.out_prefix,
                                                  base=a.base, manifest=a.manifest))
                 for seed in record["final"]["seeds"]]
        print("\n".join(lines))
        return lines
    if a.final_manifest is not None:
        if len(a.final_manifest) < 2:
            ap.error("--final-manifest takes the record, then the final checkpoint dirs")
        out = postfit.final_manifest(a.final_manifest[0], a.final_manifest[1:])
        text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
        print(text, end="")
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            write_atomic(a.out, text.encode("utf-8"))
        return out
    if a.prune is not None:
        if a.out is not None:
            ap.error("--out applies only to RUN_DIR mode")
        pruned = []
        for d in a.prune:
            rid = postfit.prune_export(d, a.dump_root, manifest)
            real = os.path.normpath(os.path.abspath(d))
            print("pruned %s/exports/%s (val dump %s)"
                  % (os.path.basename(os.path.dirname(os.path.dirname(real))), os.path.basename(real), rid))
            pruned.append(rid)
        return pruned
    record = postfit.fit(a.run_dir, manifest)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if a.out:
        postfit.write_record(record, a.out)
    return record


if __name__ == "__main__":
    main()
