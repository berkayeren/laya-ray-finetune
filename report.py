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

--compare A_DIRS B_DIRS (EVAL-07): a paired, message-level bootstrap between two systems, each side a comma-separated
list of per-seed run dirs. Every dir needs a full split dump and a full val dump (subset dumps are refused), and every
dump of both sides must score the same uids in the same order (else "unpaired"). Per side, sigmoid(logit / T_shipped)
is averaged over seeds per message and tag (pilot.metrics.seed_average), θ is chosen by D-12 on the seed-averaged val
probabilities (provisional: Phase 8 freezes the θ rule for seed-averaged systems in protocol.json), and
pilot.metrics.paired_bootstrap resamples the split's messages with one index draw shared by both systems, at θ 0.5 and
at the val-selected θ. The output records B, the RNG seed, the interval method, every run_id and dump_sha256, the
temperatures and θ, and is descriptive only: it holds no decision label.

usage: python report.py results/dumps/RUN_ID [--split val|test] [--out report.json] [--manifest splits.json]
       python report.py --compare A_RUN[,A_RUN2...] B_RUN[,B_RUN2...] [--split test] [--B 10000] [--seed 0] [--out x.json]
"""
import argparse
import json
import os

from pilot import dumps, metrics
from pilot.data import DEFAULT_MANIFEST, manifest_is_committed, repo_relpath, sha256_file, write_atomic

SCHEMA_VERSION = 1
COMPARE_PROCEDURE = ("EVAL-07: per-seed probabilities averaged per message and tag, then thresholded; test messages "
                     "resampled with replacement with one index draw shared by both systems; 95% percentile interval "
                     "of A - B")
COMPARE_THETA_RULE = ("D-12 applied to the seed-averaged val probabilities; Phase 8 freezes the rule for seed-averaged "
                      "systems in protocol.json")
COMPARE_NOTE = "descriptive intervals only; the decision rule and its thresholds are frozen in protocol.json in Phase 8"


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


def _load_side(run_dirs, split, manifest_path):
    """[(run_dir, split dump, val dump)] of one system's per-seed run dirs, each dump verified by load_dump."""
    if not run_dirs:
        raise SystemExit("no seed dumps to average: each side of --compare needs at least one run dir")
    runs, seen = [], set()
    for d in run_dirs:
        if not (os.path.exists(os.path.join(d, "val.npz")) or os.path.exists(os.path.join(d, "val.json"))):
            raise SystemExit("compare needs a full val dump in every run dir; %s has none (score.py on val)" % d)
        pair = (dumps.load_dump(d, split, manifest_path), dumps.load_dump(d, "val", manifest_path))
        rid = pair[0][2]["run_id"]                          # == the run dir name (load_dump checks it)
        if os.path.realpath(d) in seen or rid in seen:
            raise SystemExit("%s (run_id %s) is listed more than once on one side of --compare; each seed counts once"
                             % (d, rid))
        seen.update((os.path.realpath(d), rid))
        runs.append((d,) + pair)
    return runs


def _check_full(runs):
    """SystemExit on any --n subset dump: intervals and θ come only from full dumps."""
    for r in runs:
        for _, _, side, _ in r[1:]:
            if side["n_subset"] is not None:
                raise SystemExit("%s: compare needs full dumps, not a --n subset; rescore without --n" % r[0])


def _check_paired(runs, k):
    """SystemExit unless every dump at position k (1 split, 2 val) of every run scores the first one's uids in order."""
    first, uids = runs[0][0], runs[0][k][1]
    for r in runs[1:]:
        if r[k][1] != uids:
            raise SystemExit("unpaired: %s and %s score different messages or a different order"
                             % (first, r[0]))


def _seed_probs(runs, k):
    """Seed-averaged probabilities of the dumps at position k, each seed at its own sidecar's shipped T."""
    return metrics.seed_average([metrics.probabilities(r[k][0], r[k][2]["temperature"]["shipped"]) for r in runs])


def compare(a_dirs, b_dirs, split="test", manifest_path=DEFAULT_MANIFEST, B=10000, seed=0):
    """EVAL-07 paired bootstrap of system A (its per-seed run dirs) minus system B on `split`, at θ 0.5 and at each
    system's val-selected θ. Descriptive only: no decision label is produced (the rule is frozen in Phase 8)."""
    sides = {"a": _load_side(list(a_dirs), split, manifest_path), "b": _load_side(list(b_dirs), split, manifest_path)}
    every = sides["a"] + sides["b"]
    _check_paired(every, 1)                                 # pairing first: a mismatch is named "unpaired"
    _check_paired(every, 2)
    _check_full(every)
    _, uids, _, entry = every[0][1]
    _, val_uids, _, val_entry = every[0][2]
    Y = dumps.read_gold(entry, manifest_path, uids)
    Y_val = dumps.read_gold(val_entry, manifest_path, val_uids)
    P, sel, info = {}, {}, {}
    for name, runs in sides.items():
        P[name] = _seed_probs(runs, 1)
        sel[name] = metrics.select_theta(Y_val, _seed_probs(runs, 2))
        info[name] = {"run_ids": [r[1][2]["run_id"] for r in runs],
                      "dump_sha256": {"split": [r[1][2]["dump_sha256"] for r in runs],
                                      "val": [r[2][2]["dump_sha256"] for r in runs]},
                      "temperature": {"split": [r[1][2]["temperature"]["shipped"] for r in runs],
                                      "val": [r[2][2]["temperature"]["shipped"] for r in runs]},
                      "theta": sel[name]}
    thresholds = {
        "0.5": metrics.paired_bootstrap(Y, P["a"], 0.5, P["b"], 0.5, B, seed),
        "val_selected": metrics.paired_bootstrap(Y, P["a"], metrics.theta_vector(sel["a"]),
                                                 P["b"], metrics.theta_vector(sel["b"]), B, seed)}
    return {"schema_version": SCHEMA_VERSION, "kind": "paired_bootstrap", "split": split, "n": len(uids),
            "B": int(B), "seed": seed,
            "manifest": {"path": repo_relpath(manifest_path), "sha256": sha256_file(manifest_path),
                         "is_committed": manifest_is_committed(manifest_path)},
            "a": info["a"], "b": info["b"], "theta_in_sample": split == "val", "thresholds": thresholds,
            "procedure": COMPARE_PROCEDURE, "theta_rule": COMPARE_THETA_RULE, "note": COMPARE_NOTE}


def _run_dirs(ap, text):
    dirs = text.split(",")
    if not all(dirs):
        ap.error("--compare: empty run dir in %r (use a comma-separated list of run dirs)" % text)
    return dirs


def main(argv=None):
    ap = argparse.ArgumentParser(description="recompute metrics from a logit dump, or compare two systems' dumps")
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    ap.add_argument("--compare", nargs=2, metavar=("A_DIRS", "B_DIRS"), default=None,
                    help="paired bootstrap of A - B; each side a comma-separated list of per-seed run dirs")
    ap.add_argument("--B", type=int, default=10000, help="bootstrap replicates for --compare (default 10000)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed for --compare (default 0)")
    a = ap.parse_args(argv)
    if (a.run_dir is None) == (a.compare is None):
        ap.error("give exactly one of RUN_DIR or --compare A_DIRS B_DIRS")
    if a.compare is not None:
        report = compare(_run_dirs(ap, a.compare[0]), _run_dirs(ap, a.compare[1]), a.split, a.manifest, a.B, a.seed)
    else:
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
