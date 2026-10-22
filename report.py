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
  the key. A val dump of a model whose training saw val (a training block whose split is not
  pilot.data.TRAINED_ON["val"], selection_train, or that carries a postfit_sha256: a final seed on final_train, which
  holds the val folds) is refused as a θ source here and in --compare (refuse_in_sample_theta, 05-REVIEW WR-02): its
  θ is the bound postfit record's, never re-selected on its own in-sample val. A subset (--n) val dump next to a
  reported test dump is refused; without a val dump the val-selected row is null with a reason; on the val split
  itself the val-selected row is labelled in_sample, and a --n subset val dump reported on val gets its θ 0.5 rows
  with the val-selected row null and SUBSET_VAL_REASON (03-REVIEW WR-04)
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

--confidence RUN_DIR --split val|test [--postfit RECORD] (EVAL-03): a separate confidence/1 report of one verified dump
(run_report and the committed results/reports/ stay as they are). Two variants, each with pilot.metrics.calibration
(Brier, NLL, classwise ECE over 15 equal-width bins with per-tag ECE and support, reliability bins pooled and per
eligible tag) on sigmoid(logit / T) and the θ that goes with it: `headline` (the val-fitted or shipped T) and `raw`
(T = 1). Each variant also carries a selective block (EVAL-04, pilot.metrics.selective) at θ 0.5 and at its θ:
precision and exact-match accuracy at 50/70/80/90 % coverage and the AURC under the one message-confidence rule
(pilot.metrics.CONFIDENCE_RULE, COVERAGE_RULE and AURC_RULE are in the record's definitions), and a no_intent_fp
block (EVAL-05, pilot.metrics.fp_rate over all 48 tags) at θ 0.5 and at its θ: k of the n no-intent messages predicting
at least one tag, the rate and its Wilson 95% interval (pilot.metrics.FP_RULE and WILSON_TEXT are in the definitions).
Where T and θ come from (confidence_binding; every other case is refused before any metric is computed):
- postfit_candidate: a selection candidate's dump (a training block without postfit_sha256) ships T = 1, so its
  "after T" numbers need the postfit record: only on val and only with --postfit RECORD (schema postfit/1, read as
  JSON plus the sha256 of the same bytes, never through the scipy-based pilot.postfit) whose system_kind is the
  sidecar's and whose candidates hold exactly one entry with the dump's dump_sha256, the same dump_run_id and
  uids_sha256; headline T and θ are that candidate's temperature and theta.shipped, raw θ its theta.raw
- final_seed: a dump whose training.postfit_sha256 is set, only with the record of that sha256, at the sidecar's
  shipped T (which must equal the record's chosen T), with the record's chosen θ (shipped and raw)
- val_selected: a dump without a training block (baseline, zero-shot), only without --postfit, with θ from the run
  dir's full val dump by run_report's rule (theta_source, refuse_in_sample_theta; same_source when the reported split
  is not val); without a full val dump it is refused (θ is chosen on val only)
A --n subset dump is refused. A val record carries in_sample true and IN_SAMPLE_CAVEAT; the record holds the dump,
manifest and record hashes, the binding, the rule texts (pilot.metrics.CALIBRATION_RULES), no timestamps or timings and
no rounded value, so it rebuilds byte for byte (make confidence-reports). With --out X.json the record's reliability
diagram is written next to it as X.svg: render_svg of the re-parsed JSON (A8: plain SVG text, the pooled bins of both
variants side by side with the diagonal and every bin's pair count; no plotting dependency, no timestamp or metadata).

--confidence-table PATH [PATH ...] (EVAL-03): the markdown of saved confidence/1 records (a directory expands to its
sorted *.json; any other JSON is refused), rendered in the given order by render_confidence: one level-3 section per
block (a calibration table with the headline and raw rows of every record, a link to each record's SVG, the
abstention table (EVAL-04) with the confidence, coverage and AURC rule texts and CONFIDENCE_DISCLOSURE, the
no-intent false-positive table (EVAL-05: k/n, rate and Wilson 95% interval per row, the rule texts and the sample size
line), then the in-sample caveat and the definition texts). Values are rounded for display only. The README
confidence block is this output (plan 07-08).

usage: python report.py results/dumps/RUN_ID [--split val|test] [--out report.json] [--manifest splits.json]
       python report.py --compare A_RUN[,A_RUN2...] B_RUN[,B_RUN2...] [--split test] [--B 10000] [--seed 0] [--out x.json]
       python report.py --table results/reports [--out table.md]
       python report.py --confidence results/dumps/RUN_ID --split val [--postfit results/postfit/laya.json]
                        [--out results/confidence/RUN_ID-val.json]
       python report.py --confidence-table results/confidence [--out confidence.md]
"""
import argparse
import json
import os

from pilot import dumps, metrics
from pilot.data import (DEFAULT_MANIFEST, TRAINED_ON, manifest_is_committed, repo_relpath, sha256_bytes, sha256_file,
                        write_atomic)

SCHEMA_VERSION = 1
COMPARE_PROCEDURE = ("EVAL-07: per-seed probabilities averaged per message and tag, then thresholded; test messages "
                     "resampled with replacement with one index draw shared by both systems; 95% percentile interval "
                     "of A - B")
COMPARE_THETA_RULE = ("D-12 applied to the seed-averaged val probabilities; Phase 8 freezes the rule for seed-averaged "
                      "systems in protocol.json")
COMPARE_NOTE = "descriptive intervals only; the decision rule and its thresholds are frozen in protocol.json in Phase 8"
NO_VAL_REASON = "no val dump in run dir"
SUBSET_VAL_REASON = "val dump is a --n subset; θ is selected only on a full val dump"
CONFIDENCE_SCHEMA = "confidence/1"
POSTFIT_SCHEMA = "postfit/1"
IN_SAMPLE_CAVEAT = ("in-sample: θ (and, for a fine-tuned model, T and the epoch) were chosen on these val messages, so "
                    "these numbers are not held-out estimates")
# EVAL-04: the message-confidence rule (pilot.metrics.CONFIDENCE_RULE) was fixed at planning time and its val
# sensitivity disclosed with it (07-04-PLAN definition lock; 07-RESEARCH Pitfall 1); render_confidence prints this text.
CONFIDENCE_DISCLOSURE = ("The message-confidence rule (the joint decision probability) was fixed at planning time on "
                         "principle, because it is the model's own probability of the event the abstention accuracy "
                         "scores (exact match), after the Phase 7 research had computed val AURC under it and under a "
                         "weakest-tag rule (the minimum over tags of q). At the headline (val-fitted) temperature the "
                         "order of the two fine-tuned models' val AURC differs between the two rules (at T = 1 it does "
                         "not); the rule was not chosen by that outcome, and Phase 8 freezes it in protocol.json "
                         "knowing both directions.")
# EVAL-05: the interval text in every confidence record, and the test split's no-intent count (data/splits.json
# test n_no_intent, test-pinned) that the no-intent section names next to the val count it reads from the records.
WILSON_TEXT = "95%% Wilson score interval, z = %r, no continuity correction" % metrics.WILSON_Z
TEST_NO_INTENT = 15
T_MATCH = 1e-12                                             # a final seed's shipped T vs its record's chosen T
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


def refuse_in_sample_theta(val_side, where):
    """SystemExit when the θ-source val dump comes from a model whose training saw val (05-REVIEW WR-02): a training
    block that names a split other than TRAINED_ON["val"] or carries a postfit_sha256 (a final seed). Such a model's θ
    is its postfit record's (training.postfit.theta of its config), never selected again on its own in-sample val
    dump. score.py's training_block always names the split; a block without one names no training data to refuse."""
    tr = val_side.get("training")
    if isinstance(tr, dict) and (tr.get("split") not in (None, TRAINED_ON["val"]) or
                                 tr.get("postfit_sha256") is not None):
        raise SystemExit("%s: the val dump comes from a model trained on %s (postfit record %s); its val is in-sample, "
                         "so report.py does not select θ on it. A final seed's θ is its postfit record's "
                         "(training.postfit.theta); Phase 8 reports final seeds with it"
                         % (where, tr.get("split"), tr.get("postfit_sha256")))


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
    refuse_in_sample_theta(side, run_dir)
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



def read_postfit(path):
    """(record, sha256) of a postfit/1 record: the bytes are read once, hashed and parsed (no reread window). Read as
    JSON, never through pilot.postfit, which imports scipy (report.py stays numpy only)."""
    if not os.path.isfile(path):
        raise SystemExit("%s not found; pass the record postfit.py wrote (e.g. results/postfit/laya.json)" % path)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        record = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise SystemExit("%s: not a postfit/1 record; pass the record postfit.py wrote" % path)
    if not isinstance(record, dict) or record.get("schema") != POSTFIT_SCHEMA:
        raise SystemExit("%s: not a postfit/1 record; pass the record postfit.py wrote" % path)
    return record, sha256_bytes(raw)


def _postfit_block(path, sha, record):
    return {"path": repo_relpath(path), "sha256": sha, "system_kind": record.get("system_kind")}


def confidence_binding(run_dir, split, side, manifest_path, postfit=None):
    """Where the confidence report of one dump takes T and θ from (kind postfit_candidate, final_seed or
    val_selected; see the module docstring). SystemExit naming the reason for every other combination, before any
    metric is computed. Each variant's theta is the full select_theta-shaped dict."""
    tr = side.get("training")
    kind = (side.get("system") or {}).get("kind")
    rid = side["run_id"]
    if isinstance(tr, dict) and tr.get("postfit_sha256") is not None:
        if postfit is None:
            raise SystemExit("%s: a final seed (training.postfit_sha256 %s) is reported only with --postfit, the record "
                             "it was trained from" % (run_dir, tr["postfit_sha256"]))
        record, sha = read_postfit(postfit)
        if sha != tr["postfit_sha256"]:
            raise SystemExit("%s: --postfit %s has sha256 %s, but the final seed was bound to record %s; pass that "
                             "record" % (run_dir, postfit, sha, tr["postfit_sha256"]))
        if record.get("system_kind") != kind:
            raise SystemExit("%s: --postfit %s is a %s record, the dump is %s" % (run_dir, postfit,
                                                                                   record.get("system_kind"), kind))
        chosen = record.get("chosen") or {}
        T = side["temperature"]["shipped"]
        if chosen.get("temperature") is None or abs(T - chosen["temperature"]) > T_MATCH:
            raise SystemExit("%s: the final seed ships T %r, the record's chosen T is %r; a final seed serves its "
                             "record's T" % (run_dir, T, chosen.get("temperature")))
        return {"kind": "final_seed", "postfit": _postfit_block(postfit, sha, record),
                "theta_source": {"postfit_sha256": sha, "chosen": chosen.get("dump_run_id"), "split": "val"},
                "variants": {"headline": {"temperature": T,
                                          "temperature_source": "sidecar shipped T (= the postfit record's chosen T)",
                                          "theta": chosen["theta"]["shipped"]},
                             "raw": {"temperature": 1.0, "temperature_source": "T = 1 (raw)",
                                     "theta": chosen["theta"]["raw"]}}}
    if isinstance(tr, dict):
        if split != "val":
            raise SystemExit("%s: a selection candidate is reported on val only (its T and θ were chosen on val); "
                             "got split %s" % (run_dir, split))
        if postfit is None:
            raise SystemExit("%s: a selection candidate ships T = 1; pass --postfit RECORD (the postfit record that "
                             "lists this dump) so the headline uses its val-fitted T" % run_dir)
        record, sha = read_postfit(postfit)
        if record.get("system_kind") != kind:
            raise SystemExit("%s: --postfit %s is a %s record, the dump is %s" % (run_dir, postfit,
                                                                                   record.get("system_kind"), kind))
        cands = [c for c in record.get("candidates") or [] if c.get("dump_sha256") == side["dump_sha256"]]
        if len(cands) != 1:
            raise SystemExit("%s: --postfit %s lists %d candidate(s) with dump_sha256 %s, expected exactly 1; pass "
                             "the record fitted on this dump" % (run_dir, postfit, len(cands), side["dump_sha256"]))
        c = cands[0]
        if c.get("dump_run_id") != rid or c.get("uids_sha256") != side["split"]["uids_sha256"]:
            raise SystemExit("%s: the record candidate (dump_run_id %s, uids_sha256 %s) is not this dump (%s, %s)"
                             % (run_dir, c.get("dump_run_id"), c.get("uids_sha256"), rid,
                                side["split"]["uids_sha256"]))
        return {"kind": "postfit_candidate", "postfit": _postfit_block(postfit, sha, record),
                "theta_source": {"postfit_sha256": sha, "candidate": rid, "split": "val"},
                "variants": {"headline": {"temperature": c["temperature"],
                                          "temperature_source": "postfit record candidate (val-fitted T)",
                                          "theta": c["theta"]["shipped"]},
                             "raw": {"temperature": 1.0, "temperature_source": "T = 1 (raw)",
                                     "theta": c["theta"]["raw"]}}}
    if postfit is not None:
        raise SystemExit("%s: the dump has no training block (a baseline or zero-shot model); its θ comes from its "
                         "own val dump, so --postfit does not apply" % run_dir)
    src = theta_source(run_dir, manifest_path)
    if src is None:
        raise SystemExit("no full val dump in %s to select θ on; θ is chosen on val only" % run_dir)
    if split != "val":
        same_source(side, src[2], run_dir)
    val_logits, Y_val, val_side, disclosed = src
    T = side["temperature"]["shipped"]
    return {"kind": "val_selected", "postfit": None, "theta_source": disclosed,
            "variants": {"headline": {"temperature": T, "temperature_source": "sidecar shipped T",
                                      "theta": metrics.select_theta(Y_val, metrics.probabilities(
                                          val_logits, val_side["temperature"]["shipped"]))},
                         "raw": {"temperature": 1.0, "temperature_source": "T = 1 (raw)",
                                 "theta": metrics.select_theta(Y_val, metrics.probabilities(val_logits, 1.0))}}}


def confidence_report(run_dir, split, manifest_path=DEFAULT_MANIFEST, postfit=None):
    """The confidence/1 report of one verified full dump (EVAL-03, EVAL-04, EVAL-05): the binding, and per variant
    its T, θ, pilot.metrics.calibration on sigmoid(logit / T), and pilot.metrics.selective and pilot.metrics.fp_rate
    (no-intent) at θ 0.5 and at the variant's θ."""
    logits, uids, side, entry = dumps.load_dump(run_dir, split, manifest_path)
    if side["n_subset"] is not None:
        raise SystemExit("%s: a --n subset dump is never reported; rescore %s without --n" % (run_dir, split))
    Y = dumps.read_gold(entry, manifest_path, uids)
    binding = confidence_binding(run_dir, split, side, manifest_path, postfit)
    variants = {}
    for name in ("headline", "raw"):
        v = binding["variants"][name]
        P = metrics.probabilities(logits, v["temperature"])
        variants[name] = dict(v, calibration=metrics.calibration(Y, P, v["theta"]["eligible_tags"]),
                              selective={"0.5": metrics.selective(Y, P, 0.5),
                                         "theta": metrics.selective(Y, P, metrics.theta_vector(v["theta"]))},
                              no_intent_fp={"0.5": metrics.fp_rate(Y, P, 0.5),
                                            "theta": metrics.fp_rate(Y, P, metrics.theta_vector(v["theta"]))})
    tr = side.get("training")
    in_sample = split == "val"
    return {"schema": CONFIDENCE_SCHEMA, "schema_version": SCHEMA_VERSION, "run_id": side["run_id"],
            "split": side["split"], "system": side["system"],
            "training": ({"run_id": tr.get("run_id"), "split": tr.get("split"), "epoch": tr.get("epoch")}
                         if isinstance(tr, dict) else None),
            "dump_sha256": side["dump_sha256"], "uids_sha256": dumps.uids_sha256(uids), "n_messages": len(uids),
            "manifest": {"path": repo_relpath(manifest_path), "sha256": sha256_file(manifest_path),
                         "is_committed": manifest_is_committed(manifest_path)},
            "binding": {k: v for k, v in binding.items() if k != "variants"},
            "in_sample": in_sample, "caveat": IN_SAMPLE_CAVEAT if in_sample else None,
            "definitions": {"calibration": metrics.CALIBRATION_RULES, "confidence": metrics.CONFIDENCE_RULE,
                            "coverage": metrics.COVERAGE_RULE, "aurc": metrics.AURC_RULE,
                            "false_positive": metrics.FP_RULE, "wilson": WILSON_TEXT},
            "variants": variants}

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
        refuse_in_sample_theta(pair[1][2], d)
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


CONFIDENCE_VARIANTS = ("headline", "raw")
SVG_PANEL = 400                                             # side of one unit-square plot, px
SVG_LEFT, SVG_TOP, SVG_GAP = 70, 60, 120                    # left margin, top margin, gap between the two panels
SVG_WIDTH = SVG_LEFT + 2 * SVG_PANEL + SVG_GAP + 130        # the right panel's title runs past its plot
SVG_HEIGHT = SVG_TOP + SVG_PANEL + 90
SVG_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SVG_FONT = 'font-family="sans-serif"'


def _confidence_record(r):
    """r if it is a confidence/1 record, else a refusal (the renderers draw only what a record holds)."""
    if not isinstance(r, dict) or r.get("schema") != CONFIDENCE_SCHEMA:
        raise SystemExit("not a confidence/1 record: %s" % (sorted(r)[:6] if isinstance(r, dict) else type(r).__name__))
    return r


def _xml(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg_panel(r, name, x0):
    """One reliability panel: frame, ticks, per non-empty pooled bin an outlined bar of height freq over [lo, hi), a dot
    at (mean_p, freq) and its pair count, the dashed diagonal drawn over the bars; the title names the run, variant, T
    and classwise ECE."""
    v = r["variants"][name]
    cal = v["calibration"]
    S, y0 = SVG_PANEL, SVG_TOP

    def X(p):
        return x0 + p * S

    def Y(f):
        return y0 + S - f * S

    out = ['<g>',
           '<text x="%.2f" y="%.2f" font-size="11" %s>%s</text>'
           % (x0, y0 - 14, SVG_FONT, _xml("%s %s, T = %.4f, classwise ECE = %.4f"
                                         % (r["run_id"], name, v["temperature"], cal["classwise_ece"]))),
           '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" stroke="#000000" stroke-width="1"/>'
           % (x0, y0, S, S)]
    for t in SVG_TICKS:
        out.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#000000" stroke-width="1"/>'
                   % (X(t), Y(0.0), X(t), Y(0.0) + 5))
        out.append('<text x="%.2f" y="%.2f" font-size="10" text-anchor="middle" %s>%.1f</text>'
                   % (X(t), Y(0.0) + 18, SVG_FONT, t))
        out.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#000000" stroke-width="1"/>'
                   % (x0 - 5, Y(t), x0, Y(t)))
        out.append('<text x="%.2f" y="%.2f" font-size="10" text-anchor="end" %s>%.1f</text>'
                   % (x0 - 8, Y(t) + 3, SVG_FONT, t))
    for b in cal["reliability"]["pooled"]:
        if not b["count"]:
            continue
        out.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="#c6dbef" fill-opacity="0.6" '
                   'stroke="#08519c" stroke-width="1"/>' % (X(b["lo"]), Y(b["freq"]), (b["hi"] - b["lo"]) * S,
                                                            b["freq"] * S))
        out.append('<circle cx="%.2f" cy="%.2f" r="3.00" fill="#08519c"/>' % (X(b["mean_p"]), Y(b["freq"])))
        out.append('<text x="%.2f" y="%.2f" font-size="8" text-anchor="middle" %s>%d</text>'
                   % (X((b["lo"] + b["hi"]) / 2), Y(b["freq"]) - 8, SVG_FONT, b["count"]))
    out.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#555555" stroke-width="1" '
               'stroke-dasharray="5 4"/>' % (X(0.0), Y(0.0), X(1.0), Y(1.0)))
    out.append('<text x="%.2f" y="%.2f" font-size="11" text-anchor="middle" %s>mean predicted p (%d equal-width '
               'bins)</text>' % (X(0.5), Y(0.0) + 36, SVG_FONT, cal["n_bins"]))
    out.append('<text x="%.2f" y="%.2f" font-size="11" text-anchor="middle" transform="rotate(-90 %.2f %.2f)" '
               '%s>observed frequency</text>' % (x0 - 40, Y(0.5), x0 - 40, Y(0.5), SVG_FONT))
    out.append('</g>')
    return out


def render_svg(report):
    """The reliability diagram of one confidence/1 record as plain SVG text: the pooled bins of the headline (left)
    and raw (right) variants, with the diagonal and every non-empty bin's pair count (A8: no plotting dependency).
    Only g, rect, line, circle and text elements, every number %-formatted with fixed decimals, no timestamp, no
    metadata, no script and no external reference, so it rebuilds byte for byte from the JSON."""
    r = _confidence_record(report)
    n = r["n_messages"]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
           % (SVG_WIDTH, SVG_HEIGHT, SVG_WIDTH, SVG_HEIGHT),
           '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (SVG_WIDTH, SVG_HEIGHT),
           '<text x="%.2f" y="%.2f" font-size="13" %s>%s</text>'
           % (SVG_LEFT, 22, SVG_FONT, _xml("reliability diagram, %s, %d messages x 48 tags = %d (message, tag) "
                                            "pairs, pooled" % (r["split"]["name"], n, n * 48)))]
    for i, name in enumerate(CONFIDENCE_VARIANTS):
        out.extend(_svg_panel(r, name, SVG_LEFT + i * (SVG_PANEL + SVG_GAP)))
    out.append('<text x="%.2f" y="%.2f" font-size="10" %s>%s</text>'
               % (SVG_LEFT, SVG_HEIGHT - 12, SVG_FONT,
                  _xml("bars: observed frequency of y = 1 per bin; dots: (mean p, observed frequency); numbers: pairs "
                       "per bin; dashed: perfect calibration")))
    out.append('</svg>')
    return "\n".join(out) + "\n"


def _confidence_label(r):
    tr = r["training"]
    if tr is not None:
        return "%s (train run %s, epoch %s)" % (r["system"]["kind"], tr["run_id"], tr["epoch"])
    return _system_label(r["system"])


def _calibration_section(records):
    header = ("system", "run_id", "variant", "T", "Brier", "NLL", "classwise ECE (%d bins)" % metrics.ECE_BINS)
    rows, notes = [], []
    for r in records:
        for name in CONFIDENCE_VARIANTS:
            v = r["variants"][name]
            cal = v["calibration"]
            if cal["n_bins"] != metrics.ECE_BINS:
                raise SystemExit("%s: %s has %d bins, not %d" % (r["run_id"], name, cal["n_bins"], metrics.ECE_BINS))
            rows.append((_confidence_label(r), r["run_id"], name, "%.4f" % v["temperature"], "%.4f" % cal["brier"],
                         "%.4f" % cal["nll"], "%.4f" % cal["classwise_ece"]))
        if r["variants"]["headline"]["temperature"] == 1.0:
            notes.append("- %s: headline temperature is 1.0, so its raw rows equal its headline rows." % r["run_id"])
    parts = ["### Calibration on %s (θ-free)" % "/".join(sorted({r["split"]["name"] for r in records})),
             _md(header, rows)]
    if notes:
        parts.append("\n".join(notes))
    return "\n\n".join(parts)


def _diagram_section(records):
    return "\n\n".join(["### Reliability diagrams"] + [
        "![reliability diagram %s (%s)](results/confidence/%s-%s.svg)"
        % (r["run_id"], r["split"]["name"], r["run_id"], r["split"]["name"]) for r in records])


ABSTENTION_RULES = ("confidence", "coverage", "aurc")
SELECTIVE_ROWS = (("0.5", "0.5"), ("theta", "val-selected, global %.2f"))


def _abstention_section(records):
    """EVAL-04: precision and accuracy at every coverage and the AURC per record, variant and θ row, then the rule
    texts (one set, shared by every record) and CONFIDENCE_DISCLOSURE."""
    rules = {json.dumps([r["definitions"].get(k) for k in ABSTENTION_RULES]) for r in records}
    texts = [records[0]["definitions"].get(k) for k in ABSTENTION_RULES]
    if len(rules) != 1 or None in texts:
        raise SystemExit("the confidence records disagree on (or lack) the abstention definitions; rebuild them with "
                         "`make confidence-reports`")
    covs = [str(c) for c in metrics.COVERAGES]
    header = (("system", "run_id", "variant", "θ") + tuple("precision@%s%%" % c for c in covs)
              + tuple("accuracy@%s%%" % c for c in covs) + ("AURC",))
    rows = []
    for r in records:
        for name in CONFIDENCE_VARIANTS:
            v = r["variants"][name]
            for key, label in SELECTIVE_ROWS:
                s = v["selective"][key]
                if sorted(s["coverages"]) != sorted(covs):
                    raise SystemExit("%s: %s has coverages %s, not %s; rebuild with `make confidence-reports`"
                                     % (r["run_id"], name, sorted(s["coverages"]), covs))
                theta = label % v["theta"]["global"] if "%" in label else label
                rows.append((_confidence_label(r), r["run_id"], name, theta)
                            + tuple("%.3f" % s["coverages"][c]["precision"] for c in covs)
                            + tuple("%.3f" % s["coverages"][c]["accuracy"] for c in covs) + ("%.3f" % s["aurc"],))
    parts = ["### Abstention on %s (message confidence)" % "/".join(sorted({r["split"]["name"] for r in records})),
             _md(header, rows),
             "\n".join("- %s: %s" % (k, t) for k, t in zip(ABSTENTION_RULES, texts)),
             "**Definition disclosure:** %s" % CONFIDENCE_DISCLOSURE]
    return "\n\n".join(parts)


NO_INTENT_RULES = ("false_positive", "wilson")


def _no_intent_section(records):
    """EVAL-05: the no-intent false-positive rate per record, variant and θ row, always as k/n with its rate and Wilson
    95% interval (a rate is never shown alone), then the rule texts (one set, shared by every record) and the sample
    size line: the no-intent set is tiny, so every interval is wide."""
    rules = {json.dumps([r["definitions"].get(k) for k in NO_INTENT_RULES]) for r in records}
    texts = [records[0]["definitions"].get(k) for k in NO_INTENT_RULES]
    if len(rules) != 1 or None in texts:
        raise SystemExit("the confidence records disagree on (or lack) the false-positive definitions; rebuild them "
                         "with `make confidence-reports`")
    rows, sizes = [], {}
    for r in records:
        for name in CONFIDENCE_VARIANTS:
            v = r["variants"][name]
            for key, label in SELECTIVE_ROWS:
                f = v["no_intent_fp"][key]
                sizes.setdefault(r["split"]["name"], set()).add(f["n"])
                theta = label % v["theta"]["global"] if "%" in label else label
                ci = "n/a" if f["wilson95"] is None else "[%.3f, %.3f]" % tuple(f["wilson95"])
                rows.append((_confidence_label(r), r["run_id"], name, theta, "%d/%d" % (f["k"], f["n"]),
                             _num(f["rate"]), ci))
    if any(len(n) != 1 for n in sizes.values()):
        raise SystemExit("the confidence records of one split disagree on the no-intent message count %s; rebuild "
                         "them with `make confidence-reports`" % {s: sorted(n) for s, n in sizes.items()})
    splits = sorted(sizes)
    header = ("system", "run_id", "variant", "θ", "k/n", "rate", "Wilson 95%")
    lines = ["The %s no-intent set holds %d messages%s, so every interval is wide."
             % (s, next(iter(sizes[s])), " (%d on test)" % TEST_NO_INTENT if s == "val" else "") for s in splits]
    parts = ["### No-intent false positives on %s" % "/".join(splits), _md(header, rows),
             "\n".join("- %s: %s" % (k, t) for k, t in zip(NO_INTENT_RULES, texts)), "\n".join(lines)]
    return "\n\n".join(parts)


def _definitions_section(records):
    defs = {json.dumps(r["definitions"]["calibration"], sort_keys=True) for r in records}
    if len(defs) != 1:
        raise SystemExit("the confidence records disagree on the calibration definitions; rebuild them with "
                         "`make confidence-reports`")
    parts = ["### Caveat and definitions"]
    in_sample = [r["run_id"] for r in records if r["in_sample"]]
    if in_sample:
        parts.append("**Caveat:** %s (%s)." % (IN_SAMPLE_CAVEAT, ", ".join(in_sample)))
    parts.append("\n".join("- %s: %s" % (k, t) for k, t in sorted(records[0]["definitions"]["calibration"].items())))
    return "\n\n".join(parts)


# render_confidence's sections in order, each a builder of the record list returning markdown that opens with a
# level-3 heading (the README carries it inside its own section, plan 07-08); later Phase 7 plans add theirs before
# the definitions.
CONFIDENCE_SECTIONS = [_calibration_section, _diagram_section, _abstention_section, _no_intent_section,
                       _definitions_section]


def render_confidence(reports):
    """Markdown of saved confidence/1 records, in the given order (callers pass them in sorted path order): the
    calibration table (headline then raw per record), a link to each record's reliability diagram, the abstention
    table with its rule texts and CONFIDENCE_DISCLOSURE, the no-intent false-positive table, the in-sample caveat
    and the definition texts. Values are rounded for display only; identical input gives identical text."""
    records = [_confidence_record(r) for r in reports]
    if not records:
        raise SystemExit("no confidence records to render")
    return "\n\n".join(build(records) for build in CONFIDENCE_SECTIONS) + "\n"


def _run_dirs(ap, text):
    dirs = text.split(",")
    if not all(dirs):
        ap.error("--compare: empty run dir in %r (use a comma-separated list of run dirs)" % text)
    return dirs


def main(argv=None):
    ap = argparse.ArgumentParser(description="recompute metrics from a logit dump, or compare two systems' dumps")
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--split", default=None, choices=("val", "test"),
                    help="the dump's split (default test for RUN_DIR and --compare; required for --confidence)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify against (default: the committed data/splits.json)")
    ap.add_argument("--compare", nargs=2, metavar=("A_DIRS", "B_DIRS"), default=None,
                    help="paired bootstrap of A - B; each side a comma-separated list of per-seed run dirs")
    ap.add_argument("--B", type=int, default=10000, help="bootstrap replicates for --compare (default 10000)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed for --compare (default 0)")
    ap.add_argument("--table", nargs="+", metavar="PATH", default=None,
                    help="markdown table of saved reports (a directory expands to its sorted *.json)")
    ap.add_argument("--confidence", metavar="RUN_DIR", default=None,
                    help="confidence/1 report (Brier, NLL, classwise ECE, reliability) of RUN_DIR's --split dump")
    ap.add_argument("--postfit", metavar="RECORD", default=None,
                    help="with --confidence: the postfit record binding a selection candidate's or final seed's T, θ")
    ap.add_argument("--confidence-table", nargs="+", metavar="PATH", default=None,
                    help="markdown of saved confidence/1 records (a directory expands to its sorted *.json)")
    a = ap.parse_args(argv)
    if sum(x is not None for x in (a.run_dir, a.compare, a.table, a.confidence, a.confidence_table)) != 1:
        ap.error("give exactly one of RUN_DIR, --compare A_DIRS B_DIRS, --table PATH [PATH ...], "
                 "--confidence RUN_DIR or --confidence-table PATH [PATH ...]")
    if a.postfit is not None and a.confidence is None:
        ap.error("--postfit applies only to --confidence")
    if a.confidence is not None and a.split is None:
        ap.error("--confidence needs an explicit --split val|test")
    if a.confidence is not None and a.out is not None and not a.out.endswith(".json"):
        ap.error("--confidence --out must name a .json file (the reliability diagram is written next to it as .svg)")
    if a.table is not None or a.confidence_table is not None:
        reports = []
        for p in table_inputs(a.table if a.table is not None else a.confidence_table):
            with open(p, encoding="utf-8") as f:
                reports.append(json.load(f))
        text = render_table(reports) if a.table is not None else render_confidence(reports)
        print(text, end="")
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            write_atomic(a.out, text.encode("utf-8"))
        return text
    split = "test" if a.split is None else a.split
    if a.confidence is not None:
        report = confidence_report(a.confidence, split, a.manifest, a.postfit)
    elif a.compare is not None:
        report = compare(_run_dirs(ap, a.compare[0]), _run_dirs(ap, a.compare[1]), split, a.manifest, a.B, a.seed)
    else:
        report = run_report(a.run_dir, split, a.manifest)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        parent = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(parent, exist_ok=True)
        write_atomic(a.out, (text + "\n").encode("utf-8"))
        if a.confidence is not None:
            write_atomic(a.out[:-len(".json")] + ".svg", render_svg(json.loads(text)).encode("utf-8"))
    return report


if __name__ == "__main__":
    main()
