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
  The test dump never shapes θ. The val dump must come from the same source as the reported dump: their sidecars
  must agree on SAME_SOURCE_KEYS (system kind, checkpoint identity, shipped T, scorer device, dtype and amp; the
  checkpoint path only when no identity binds the scored bytes; 03-REVIEW WR-01), else the report is refused naming
  the key. A subset (--n) val dump next to a reported test dump
  is refused; without a val dump the val-selected row is null with a reason; on the val split itself the val-selected
  row is labelled in_sample, and a --n subset val dump reported on val gets its θ 0.5 rows with the val-selected row
  null and SUBSET_VAL_REASON (03-REVIEW WR-04)
- the full θ (global, per tag, eligible tags, grid, objective, tie rule) and its val dump's hashes are disclosed
The report holds no timestamps or timings and no rounded value.

--compare A_DIRS B_DIRS (EVAL-07): a paired, message-level bootstrap between two systems, each side a comma-separated
list of per-seed run dirs. Every dir needs a full split dump and a full val dump of the same source (SAME_SOURCE_KEYS;
subset dumps are refused), and every dump of both sides must score the same uids in the same order (else "unpaired").
A side of several run dirs must be distinct seeds of one system (03-REVIEW WR-05): one system kind, a training block
in every sidecar with one shared config_sha256_sans_seed (pilot.dumps.training_digest: the training recipe and epoch
without the seed), distinct non-null seeds and pairwise distinct split and val dump_sha256; a run dir listed twice is
refused too. A single-dir side (a baseline, zero-shot Laya, one seed) needs none of these, and the checks run within a
side only, so A = X, B = a copy of X stays a zero-width self-test.
Per side, sigmoid(logit / T_shipped)
is averaged over seeds per message and tag (pilot.metrics.seed_average), θ is chosen by D-12 on the seed-averaged val
probabilities (provisional: Phase 8 freezes the θ rule for seed-averaged systems in protocol.json), and
pilot.metrics.paired_bootstrap resamples the split's messages with one index draw shared by both systems, at θ 0.5 and
at the val-selected θ. The output records B, the RNG seed, the interval method, every run_id and dump_sha256, the
temperatures and θ, and is descriptive only: it holds no decision label.

--table PATH [PATH ...] (D-08): the markdown results table of saved reports (a directory expands to its sorted *.json),
rendered in the given order: one row per single-system report and temperature variant (shipped first, then raw), a
second table for paired-bootstrap comparisons, and a note for every report whose shipped T is 1.0. Labels come from
each report's system block, never from a system-specific constant; values are rounded for display only. A --n subset
report is refused: the table shows full-split numbers only. The README results block is this output, test-enforced.

usage: python report.py results/dumps/RUN_ID [--split val|test] [--out report.json] [--manifest splits.json]
       python report.py --compare A_RUN[,A_RUN2...] B_RUN[,B_RUN2...] [--split test] [--B 10000] [--seed 0] [--out x.json]
       python report.py --table results/reports [--out table.md]
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
NO_VAL_REASON = "no val dump in run dir"
SUBSET_VAL_REASON = "val dump is a --n subset; θ is selected only on a full val dump"
# Sidecar keys the θ-source val dump and the reported dump must share (03-REVIEW WR-01): θ chosen on one checkpoint's,
# temperature's or device's / precision's val probabilities is never applied to another's dump. system.baseline_source
# is excluded on purpose: pilot.data.TRAINED_ON gives a baseline's val and test dumps different training splits.
SAME_SOURCE_KEYS = (("system", "kind"), ("system", "checkpoint"), ("system", "checkpoint_identity"),
                    ("temperature", "shipped"), ("scorer", "device"), ("scorer", "dtype"), ("scorer", "amp_enabled"))
# system.checkpoint, the repo-relative path, counts only when no checkpoint_identity binds every file laya.load reads:
# a local checkpoint's run id is hash-based, so the same checkpoint scored from a renamed dir lands in the same run dir
# with the same identity and is the same source (04-REVIEW IN-12). Bound means the pin (revision), or a local identity
# with files_sha256 (weights, config, tokenizer/ and encoder/; 04-REVIEW IN-05) that holds dumps.LOCAL_REQUIRED_FILES
# (04-REVIEW WR-02); an older local identity (weights and config only) or an incomplete one still compares the path.
# A BERTurk identity (pilot.berturk.identity, kind berturk) binds every file score.py's BERTurk branch reads when its
# model, config and record digests and both BERTURK_FILES digests are set (BASE-02); an incomplete one compares the path
PATH_KEY = ("system", "checkpoint")
BERTURK_FILES = ("tokenizer_config.json", "vocab.txt")    # = pilot.berturk.TOKENIZER_FILES (test-pinned; numpy only)
BERTURK_DIGESTS = ("model_sha256", "config_sha256", "record_sha256")


def binds_every_file(ident):
    """True for a checkpoint_identity that pins every file laya.load reads: the pinned checkpoint's (its revision) or a
    local one whose files_sha256 (pilot.weights.identity) holds every dumps.LOCAL_REQUIRED_FILES file; without
    tokenizer/ or encoder/ laya.load reads them unpinned from the Hub (04-REVIEW WR-02). A BERTurk identity (kind
    berturk) binds when its BERTURK_DIGESTS and the digests of both BERTURK_FILES are all set: those five files are
    everything score.py's BERTurk branch reads (BASE-02)."""
    if not isinstance(ident, dict):
        return False
    files = ident.get("files_sha256")
    if ident.get("kind") == "berturk":
        return (all(ident.get(k) for k in BERTURK_DIGESTS) and isinstance(files, dict)
                and all(files.get(f) for f in BERTURK_FILES))
    return bool(ident.get("revision")) or (isinstance(files, dict)
                                           and all(k in files for k in dumps.LOCAL_REQUIRED_FILES))


def same_source(side, val_side, where):
    """SystemExit naming the first SAME_SOURCE_KEYS path on which the reported dump's sidecar and the val dump's
    sidecar differ (a missing key reads as None on both sides); PATH_KEY is skipped when the reported dump's
    checkpoint_identity binds every file (binds_every_file; the identity key itself is always compared)."""
    bound = binds_every_file((side.get("system") or {}).get("checkpoint_identity"))
    for path in SAME_SOURCE_KEYS:
        if path == PATH_KEY and bound:
            continue
        a, b = side, val_side
        for k in path:
            a = a.get(k) if isinstance(a, dict) else None
            b = b.get(k) if isinstance(b, dict) else None
        if a != b:
            raise SystemExit("%s: the val dump (θ source) and the %s dump differ in %s (%r vs %r); rescore both with "
                             "the same checkpoint, device and precision"
                             % (where, (side.get("split") or {}).get("name"), ".".join(path), b, a))


def theta_source(run_dir, manifest_path):
    """(val logits, gold, sidecar, disclosure) of the run dir's full val dump, or None when it has no val dump.

    load_dump proves the dump is untampered, belongs to this run (run_id == run dir name) and, for a full dump, covers
    the registered val split in order (uids_sha256). A subset val dump is refused: θ is never chosen on part of val.
    run_report does not call it when the reported dump is itself a subset val dump (SUBSET_VAL_REASON).
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
    """The report dict of one verified dump: shipped and raw variants, rows at 0.5 and at the val-selected θ.

    The val-selected row is null with a reason in two cases: the run dir has no val dump (NO_VAL_REASON), or the
    reported dump is a --n subset val dump (SUBSET_VAL_REASON; it is reported at θ 0.5 only). A θ-source val dump
    from another source than the reported dump (same_source) is refused.
    """
    logits, uids, side, entry = dumps.load_dump(run_dir, split, manifest_path)
    Y = dumps.read_gold(entry, manifest_path, uids)
    if split == "val" and side["n_subset"] is not None:     # the reported dump is that subset val dump: no θ source
        src, reason = None, SUBSET_VAL_REASON
    else:
        src, reason = theta_source(run_dir, manifest_path), NO_VAL_REASON   # reads only val: the split never shapes θ
    if src is not None and split != "val":
        same_source(side, src[2], run_dir)
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
            v["val_selected_reason"] = reason
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
        same_source(pair[0][2], pair[1][2], d)
        rid = pair[0][2]["run_id"]                          # == the run dir name (load_dump checks it)
        if os.path.realpath(d) in seen or rid in seen:
            raise SystemExit("%s (run_id %s) is listed more than once on one side of --compare; each seed counts once"
                             % (d, rid))
        seen.update((os.path.realpath(d), rid))
        runs.append((d,) + pair)
    if len(runs) > 1:
        _check_one_system(runs)
    return runs


def _check_one_system(runs):
    """SystemExit unless a multi-dir side holds distinct seeds of one system (03-REVIEW WR-05): one system kind, a
    training block with one shared config_sha256_sans_seed (pilot.dumps.training_digest), distinct recorded seeds and
    distinct split and val dumps. A sidecar without the keys (Phase 3 dumps) reads seed and training as None."""
    sides = [r[1][2] for r in runs]
    kinds = sorted({str((s.get("system") or {}).get("kind")) for s in sides})
    if len(kinds) != 1:
        raise SystemExit("one side of --compare must be seeds of one system; %s mixes system kinds %s"
                         % (",".join(r[0] for r in runs), kinds))
    digests = [(s.get("training") or {}).get("config_sha256_sans_seed") for s in sides]
    if None in digests or len(set(digests)) != 1:
        raise SystemExit("one side of --compare must be seeds of one system (same training config apart from the "
                         "seed); got %s" % digests)
    seeds = [s.get("seed") for s in sides]
    if None in seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("seed dumps on one side need distinct, recorded seeds; got %s" % seeds)
    for k in (1, 2):                                        # the split dumps, then the val dumps
        shas = [r[k][2]["dump_sha256"] for r in runs]
        if len(set(shas)) != len(shas):
            raise SystemExit("two run dirs on one side of --compare hold identical dumps; each seed counts once")


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


SINGLE_HEADER = ("system", "run_id", "temperature", "θ global (val)", "per-tag θ", "micro-F1 @0.5", "micro-F1 @val-θ",
                 "macro-F1 @0.5", "macro-F1 @val-θ", "exact match @0.5", "exact match @val-θ", "top-1",
                 "split, n (no-intent)")
COMPARE_HEADER = ("A", "B", "θ", "metric", "delta (A - B)", "95% CI", "replicates")
THRESHOLD_LABELS = (("0.5", "0.5"), ("val_selected", "val-selected"))


def _num(v, fmt="%.3f"):
    return "n/a" if v is None else fmt % v


def _md(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _system_label(system):
    """kind plus the checkpoint (a model) or the training split its statistics come from (a baseline)."""
    if system.get("checkpoint") is not None:
        return "%s (%s)" % (system["kind"], system["checkpoint"])
    src = system.get("baseline_source") or {}
    return "%s (train: %s)" % (system["kind"], src.get("split", "n/a"))


def _single_rows(r):
    if r.get("subset"):
        raise SystemExit("%s is a --n subset report; the results table shows full-split reports only" % r["run_id"])
    rows = []
    for name in ("shipped", "raw"):
        v = r["variants"][name]
        at5, sel = v["rows"]["0.5"], v["rows"]["val_selected"]
        theta = None if sel is None else sel["theta"]
        rows.append((_system_label(r["system"]), r["run_id"], "%s, T = %.2f" % (name, v["temperature"]),
                     _num(None if theta is None else theta["global"], "%.2f"),
                     "n/a" if theta is None else str(len(theta["per_tag"])),
                     _num(at5["micro_f1"]), _num(None if sel is None else sel["micro_f1"]),
                     _num(at5["macro_f1"]), _num(None if sel is None else sel["macro_f1"]),
                     _num(at5["exact_match"]), _num(None if sel is None else sel["exact_match"]),
                     _num(at5["top1_hit"]), "%s, %d (%d)" % (r["split"]["name"], at5["n"], at5["n_no_intent"])))
    return rows


def _compare_rows(c):
    rows = []
    for key, label in THRESHOLD_LABELS:
        for metric, m in c["thresholds"][key]["metrics"].items():
            ci = "n/a" if m["ci"] is None else "[%.3f, %.3f]" % tuple(m["ci"])
            rows.append((",".join(c["a"]["run_ids"]), ",".join(c["b"]["run_ids"]), label, metric,
                         _num(m["delta"], "%+.3f"), ci, str(c["B"])))
    return rows


def render_table(reports):
    """Markdown of saved report dicts, in the given order (callers pass them in sorted path order)."""
    singles, comps, notes = [], [], []
    for r in reports:
        if r.get("kind") == "paired_bootstrap":
            comps.extend(_compare_rows(r))
        elif "variants" in r:
            singles.extend(_single_rows(r))
            if r["variants"]["shipped"]["temperature"] == 1.0:
                notes.append("- %s: shipped temperature is 1.0, so the T = 1 rows equal the headline rows."
                             % r["run_id"])
        else:
            raise SystemExit("not a report.py report (no variants, not a paired_bootstrap): %s"
                             % sorted(r)[:6])
    if not (singles or comps):
        raise SystemExit("no reports to tabulate")
    parts = [_md(SINGLE_HEADER, singles)] if singles else []
    if comps:
        parts.append(_md(COMPARE_HEADER, comps))
    if notes:
        parts.append("\n".join(notes))
    return "\n\n".join(parts) + "\n"


def table_inputs(paths):
    """Report paths of --table: files as given, a directory expanded to its sorted *.json."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(os.path.join(p, n) for n in sorted(os.listdir(p)) if n.endswith(".json"))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit("%s not found; run `make baseline-reports` to rebuild the reports" % p)
    return out


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
    ap.add_argument("--table", nargs="+", metavar="PATH", default=None,
                    help="markdown table of saved reports (a directory expands to its sorted *.json)")
    a = ap.parse_args(argv)
    if sum(x is not None for x in (a.run_dir, a.compare, a.table)) != 1:
        ap.error("give exactly one of RUN_DIR, --compare A_DIRS B_DIRS or --table PATH [PATH ...]")
    if a.table is not None:
        reports = []
        for p in table_inputs(a.table):
            with open(p, encoding="utf-8") as f:
                reports.append(json.load(f))
        text = render_table(reports)
        print(text, end="")
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            write_atomic(a.out, text.encode("utf-8"))
        return text
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
