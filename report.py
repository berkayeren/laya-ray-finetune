"""Dumps -> metrics JSON (EVAL-01, EVAL-02): numpy only, no torch, Ray or laya, and no model.

Every number is recomputed from a verified dump (pilot.dumps.load_dump: npz bytes, keys, dtypes, split, question and
tag-order hashes, coverage; hard-fail, no override) and the verified split's gold labels, so an auditor with the repo,
the committed dumps and `make data` (no weights) can reproduce the report byte for byte. Every system is treated the
same way: probabilities are sigmoid(logit / T) with the dump's shipped temperature, then pilot.metrics at theta 0.5.
The report holds no timestamps or timings and no rounded value.

usage: python report.py results/dumps/RUN_ID [--split val|test] [--out report.json] [--manifest splits.json]
"""
import argparse
import json
import os

from pilot import dumps, metrics
from pilot.data import DEFAULT_MANIFEST, manifest_is_committed, repo_relpath, sha256_file, write_atomic

SCHEMA_VERSION = 1


def run_report(run_dir, split="test", manifest_path=DEFAULT_MANIFEST):
    """The report dict of one verified dump."""
    logits, uids, side, entry = dumps.load_dump(run_dir, split, manifest_path)
    Y = dumps.read_gold(entry, manifest_path, uids)
    T = side["temperature"]["shipped"]
    return {"schema_version": SCHEMA_VERSION, "run_id": side["run_id"], "split": side["split"],
            "subset": side["n_subset"] is not None, "system": side["system"],
            "manifest": {"path": repo_relpath(manifest_path), "sha256": sha256_file(manifest_path),
                         "is_committed": manifest_is_committed(manifest_path)},
            "dump_sha256": side["dump_sha256"], "headline": "shipped",
            "variants": {"shipped": {"temperature": T,
                                     "rows": {"0.5": metrics.metrics(Y, metrics.probabilities(logits, T), 0.5)}}}}


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
