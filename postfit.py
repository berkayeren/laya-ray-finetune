"""Postfit CLI (LAYA-05, LAYA-08): the selection val dumps -> one record with the val-fitted temperature, the
val-selected thresholds and the chosen (encoder LR, epoch). numpy and scipy only: no torch, Ray, laya or model.

Every number comes from verified val dumps (pilot.dumps.load_dump) and the verified val split's gold labels, so the
record rebuilds byte for byte from the committed dumps, in any order of the RUN_DIR arguments. Test input is refused:
a file path (a .jsonl, an .npz), a run dir holding a test dump, a --n subset val dump, a candidate trained on
final_train (its val is in-sample), a calibrated candidate and one whose shipped T is not 1.0 (pilot.postfit).
The rules are the constants of pilot.postfit, as the user confirmed them at plan 05-10; the grid and the seeds are
pilot.postfit.protocol of the candidates' system kind (laya: pilot.postfit's constants; berturk: pilot.berturk.PROTOCOL,
1 LR x epochs 1..10, plan 06-04), so BERTurk val dumps go through exactly Laya's functions. Mixed kinds are refused.

--prune EXPORT_DIR... deletes epoch exports (<storage>/<run_id>/exports/epoch-NNN) to free disk, each only after a
verified full val dump of exactly that export exists under --dump-root: a dir holding berturk_train.json goes to
pilot.berturk.prune_export, any other to pilot.postfit.prune_export (ray_train.py exports); anything else, a symlink
included, is refused and left in place.

--print-final-commands RECORD prints one training command per final seed, each line to be prefixed with the
interpreter (and caffeinate -i on the Mac), dispatched on the record's system kind: a Laya record prints ray_train.py
commands (pilot.postfit.final_command; --base defaults to multilingual), a BERTurk record berturk_train.py commands
(pilot.berturk.final_command; --base defaults to berturk), which need --reference REF, the record's BASE-03 reference
check, refused here and in berturk_train.py unless it is of that record and reads ok or carries a resolution. The
trainer's --postfit still refuses any command that is not the chosen recipe. --final-manifest RECORD CKPT... builds
the postfit-final/1 manifest of the finished final checkpoints (pilot.postfit.final_manifest for Laya,
pilot.berturk.final_manifest for BERTurk), refused unless they are exactly the record's final seeds.

--reference-check RECORD [--resolution TEXT] builds the BASE-03 check of a BERTurk postfit record
(pilot.berturk.reference_check, schema berturk-reference/1): the chosen candidate's val micro-F1 at its val-selected
global θ against 0.707, exactly; status ok or investigate, the quoted source, the setup differences and the in-sample
caveat. A Laya record is refused; --resolution (the user's documented reason) only on an investigate status.

usage: python postfit.py results/dumps/RUN_ID [RUN_ID2 ...] [--out results/postfit/laya.json] [--manifest splits.json]
       python postfit.py --prune ray_results/RUN_ID/exports/epoch-001 [...] [--dump-root results/dumps]
       python postfit.py --print-final-commands results/postfit/laya.json --storage ray_results \
                         --out-prefix ckpt/laya-final
       python postfit.py --final-manifest results/postfit/laya.json ckpt/laya-final-s0 ckpt/laya-final-s1 \
                         ckpt/laya-final-s2 --out results/postfit/laya-final.json
       python postfit.py --print-final-commands results/postfit/berturk.json \
                         --reference results/postfit/berturk-reference.json --storage berturk_runs \
                         --out-prefix ckpt/berturk-final
       python postfit.py --reference-check results/postfit/berturk.json [--resolution TEXT] \
                         [--out results/postfit/berturk-reference.json]
"""
import argparse
import json
import os
import shlex

from pilot import berturk, dumps, postfit
from pilot.data import DEFAULT_MANIFEST, write_atomic


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit T, θ and the (encoder LR, epoch) choice on selection val dumps, "
                                             "prune epoch exports whose val dump exists, print the final-seed "
                                             "commands or build the final-seed manifest")
    ap.add_argument("run_dir", nargs="*", help="run dirs that each hold one candidate's full val dump")
    ap.add_argument("--out", default=None,
                    help="write the record (with --final-manifest the manifest, with --reference-check the check) here")
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
    ap.add_argument("--base", default=None,
                    help="--print-final-commands: the base the chosen run trained (default the pinned weights: "
                         "multilingual for a Laya record, berturk for a BERTurk record)")
    ap.add_argument("--reference", metavar="REF", default=None,
                    help="--print-final-commands of a BERTurk record: its BASE-03 reference check (postfit.py "
                         "--reference-check --out REF)")
    ap.add_argument("--final-manifest", nargs="+", metavar="RECORD CKPT", default=None,
                    help="the postfit record, then the final checkpoint dirs of every final seed")
    ap.add_argument("--reference-check", metavar="RECORD", default=None,
                    help="the BASE-03 published-reference check of a BERTurk postfit record")
    ap.add_argument("--resolution", default=None,
                    help="--reference-check: the documented resolution of an investigate status")
    a = ap.parse_args(argv)
    modes = (bool(a.run_dir), a.prune is not None, a.print_final_commands is not None, a.final_manifest is not None,
             a.reference_check is not None)
    if sum(modes) != 1:
        ap.error("give exactly one of RUN_DIR [RUN_DIR ...], --prune EXPORT_DIR [...], --print-final-commands RECORD, "
                 "--final-manifest RECORD CKPT [...] or --reference-check RECORD")
    if a.resolution is not None and a.reference_check is None:
        ap.error("--resolution applies only to --reference-check")
    if a.print_final_commands is None and (a.storage is not None or a.out_prefix is not None or a.base is not None):
        ap.error("--storage, --out-prefix and --base apply only to --print-final-commands")
    if a.print_final_commands is None and a.reference is not None:
        ap.error("--reference applies only to --print-final-commands of a BERTurk record")
    manifest = a.manifest or DEFAULT_MANIFEST
    if a.print_final_commands is not None:
        if a.out is not None:
            ap.error("--out does not apply to --print-final-commands (use --out-prefix)")
        if a.storage is None or a.out_prefix is None:
            ap.error("--print-final-commands needs --storage DIR and --out-prefix PATH")
        record, sha = postfit.read_record(a.print_final_commands)
        if record.get("system_kind") == "berturk":
            if a.reference is None:
                ap.error("--print-final-commands of a BERTurk record needs --reference REF (its BASE-03 check)")
            berturk.check_reference_for_final(a.reference, sha)      # no command printed before the gate passes
            argvs = [berturk.final_command(record, seed, a.print_final_commands, a.reference, a.storage, a.out_prefix,
                                           base=a.base or "berturk", manifest=a.manifest)
                     for seed in record["final"]["seeds"]]
        else:
            if a.reference is not None:
                ap.error("--reference applies only to --print-final-commands of a BERTurk record")
            argvs = [postfit.final_command(record, seed, a.print_final_commands, a.storage, a.out_prefix,
                                           base=a.base or "multilingual", manifest=a.manifest)
                     for seed in record["final"]["seeds"]]
        lines = [shlex.join(argv) for argv in argvs]
        print("\n".join(lines))
        return lines
    if a.reference_check is not None:
        record, sha = postfit.read_record(a.reference_check)
        out = berturk.reference_check(record, a.reference_check, sha, a.resolution)
        text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
        print(text, end="")
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            write_atomic(a.out, text.encode("utf-8"))
        return out
    if a.final_manifest is not None:
        if len(a.final_manifest) < 2:
            ap.error("--final-manifest takes the record, then the final checkpoint dirs")
        record, _ = postfit.read_record(a.final_manifest[0])
        build = berturk.final_manifest if record.get("system_kind") == "berturk" else postfit.final_manifest
        out = build(a.final_manifest[0], a.final_manifest[1:])
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
            prune = berturk.prune_export if os.path.isfile(os.path.join(d, berturk.RECORD)) else postfit.prune_export
            rid = prune(d, a.dump_root, manifest)
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
