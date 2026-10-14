"""Dumps -> metrics JSON (EVAL-01, EVAL-02): numpy only, no torch, Ray or laya, and no model.

Every number is recomputed from a verified dump (pilot.dumps.load_dump: npz bytes, keys, dtypes, split, question and
tag-order hashes, coverage; hard-fail, no override) and the verified split's gold labels, so an auditor with the repo,
the committed dumps and `make data` (no weights) can reproduce the report byte for byte. Every system is treated the
same way, with no system-specific branch (D-10, D-12):
- two temperature variants (D-11): `shipped` (the dump's shipped T; the headline) and `raw` (T = 1); probabilities
  are sigmoid(logit / T)
- each variant has a row at θ = 0.5 and a row at the val-selected θ (pilot.metrics.select_theta). θ is chosen only on
  the full val dump of the same run directory (run_id and uids_sha256 verified by load_dump), on that variant's
  val probabilities (shipped: the val sidecar's shipped T; raw: T = 1), and is then applied to the reported split.
  The test dump never shapes θ. A subset (--n) val dump is refused; without a val dump the val-selected row is null
  with a reason; on the val split itself the val-selected row is labelled in_sample
- the full θ (global, per tag, eligible tags, grid, objective, tie rule) and its val dump's hashes are disclosed
The report holds no timestamps or timings and no rounded value.

usage: python report.py results/dumps/RUN_ID [--split val|test] [--out report.json] [--manifest splits.json]
"""
import argparse
import json
import os

from pilot import dumps, metrics
from pilot.data import DEFAULT_MANIFEST, manifest_is_committed, repo_relpath, sha256_file, write_atomic

SCHEMA_VERSION = 1


def theta_source(run_dir, manifest_path):
    """(val logits, gold, sidecar, disclosure) of the run dir's full val dump, or None when it has no val dump.

    load_dump proves the dump is untampered, belongs to this run (run_id == run dir name) and, for a full dump, covers
    the registered val split in order (uids_sha256). A subset val dump is refused: θ is never chosen on part of val.
    """
    npz = os.path.join(run_dir, "val.npz")
    if not (os.path.exists(npz) or os.path.exists(os.path.join(run_dir, "val.json"))):
        return None
    logits, uids, side, entry = dumps.load_dump(run_dir, "val", manifest_path)
    if side["n_subset"] is not None:
        raise SystemExit("%s: θ is selected only on a full val dump; rescore val without --n" % npz)
    return logits, dumps.read_gold(entry, manifest_path, uids), side, {
        "run_id": side["run_id"], "split": "val", "dump_sha256": side["dump_sha256"],
        "uids_sha256": dumps.uids_sha256(uids), "n_messages": len(uids)}


def run_report(run_dir, split="test", manifest_path=DEFAULT_MANIFEST):
    """The report dict of one verified dump: shipped and raw variants, rows at 0.5 and at the val-selected θ."""
    logits, uids, side, entry = dumps.load_dump(run_dir, split, manifest_path)
    Y = dumps.read_gold(entry, manifest_path, uids)
    src = theta_source(run_dir, manifest_path)              # reads only val: the reported split never shapes θ
    val_logits, Y_val, val_side, disclosed = src if src is not None else (None, None, None, None)
    variants = {}
    for name, T, T_val in (("shipped", side["temperature"]["shipped"],
                            None if src is None else val_side["temperature"]["shipped"]),
                           ("raw", 1.0, 1.0)):
        P = metrics.probabilities(logits, T)
        rows = {"0.5": metrics.metrics(Y, P, 0.5)}
        v = {"temperature": T, "rows": rows}
        if src is None:
            rows["val_selected"] = None
            v["val_selected_reason"] = "no val dump in run dir"
        else:                                               # θ on this variant's val probabilities (after T)
            sel = metrics.select_theta(Y_val, metrics.probabilities(val_logits, T_val))
            rows["val_selected"] = dict(metrics.metrics(Y, P, metrics.theta_vector(sel)), theta=sel,
                                        in_sample=(split == "val"))
        variants[name] = v
    return {"schema_version": SCHEMA_VERSION, "run_id": side["run_id"], "split": side["split"],
            "subset": side["n_subset"] is not None, "system": side["system"],
            "manifest": {"path": repo_relpath(manifest_path), "sha256": sha256_file(manifest_path),
                         "is_committed": manifest_is_committed(manifest_path)},
            "dump_sha256": side["dump_sha256"], "headline": "shipped",
            "theta_source": disclosed, "variants": variants}


def main(argv=None):
    ap = argparse.ArgumentParser(description="recompute metrics from a logit dump")
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    a = ap.parse_args(argv)
    report = run_report(a.run_dir, a.split, a.manifest)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        parent = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(parent, exist_ok=True)
        write_atomic(a.out, (text + "\n").encode("utf-8"))
    return report


if __name__ == "__main__":
    main()
