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
  θ is the bound postfit record's, never re-selected on its own in-sample val. A final seed's dump is never reported
  on content it trained on: on val (a --n subset too) or on a split holding committed val or final_train content
  under any manifest it is refused, here (so in evaluate.py), in --compare and in the --confidence final_seed binding
  (refuse_final_seed_training_content, 07-REVIEW WR-01, 07-09 review CR-01). A θ-source val split holding committed
  test, HOTELS or selection_train content is refused, keyed on the content under any manifest (refuse_held_out_theta
  over pilot.data.holds_committed, 07-REVIEW CR-01, 07-09 review WR-01), here, in --compare and in the --confidence
  val_selected and postfit_candidate bindings. Both content checks fail closed while a committed split file they read
  is missing, under a non-committed manifest of the pinned dataset (require_committed_files, 07-09 review WR-02). A
  subset (--n) val dump next to a reported test dump is refused; without a val dump the val-selected row is null with
  a reason; on the val split itself the val-selected row is labelled in_sample, and a --n subset val dump reported on
  val gets its θ 0.5 rows with the val-selected row null and SUBSET_VAL_REASON (03-REVIEW WR-04)
- a final seed (training.postfit_sha256 set; Phase 8, REP-02's per-seed rows) never has θ selected: with --postfit
  RECORD, the record its training.postfit_sha256 names, its val-selected rows are at the record's chosen θ
  (theta.shipped at the sidecar's shipped T, theta.raw at T = 1; in_sample false), bound by final_seed_binding, the
  same checks as the --confidence final_seed binding below (record sha256, system kind, refuse_held_out_record, T
  within T_MATCH; 08-RESEARCH A5, Pattern 3), and theta_source names the record, its chosen candidate and split val;
  without --postfit its val-selected rows are null with FINAL_SEED_REASON. --postfit on any other dump is refused
- the full θ (global, per tag, eligible tags, grid, objective, tie rule) and its source are disclosed (theta_source:
  the val dump's hashes, or a final seed's postfit record; its split is always val)
The report holds no timestamps or timings and no rounded value. A dump whose split is not role val or test (a
hotels_probe dump, role ood_probe) is refused naming its role, here and in --compare (refuse_ood_role): its gold is
valid on the 22 banking-only columns only, so micro-F1 over 48 tags is undefined there (07-RESEARCH A7).

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
- final_seed: a dump whose training.postfit_sha256 is set, on test or hotels_probe only, only with the record of that
  sha256, at the sidecar's shipped T (which must equal the record's chosen T), with the record's chosen θ (shipped and
  raw). A final seed was trained on final_train, which holds the val folds, so its numbers on val, or on any split
  holding committed val or final_train content under any manifest, are training-set numbers; they are refused before
  the record is read (refuse_final_seed_training_content, 07-REVIEW WR-01; run_report and --compare refuse them too).
  The record's T and θ must have been fitted on banking val content: a record whose split holds committed test, HOTELS
  or selection_train content (by its hashes, or by its rows read under the manifest it was fitted on) is refused, and
  so is a record of another manifest whose val is not the committed val (refuse_held_out_record, 07-09 review WR-03)
- val_selected: a dump without a training block (baseline, zero-shot), only without --postfit, with θ from the run
  dir's full val dump by run_report's rule (theta_source, refuse_in_sample_theta; same_source when the reported split
  is not val); without a full val dump it is refused (θ is chosen on val only)
In the val_selected and postfit_candidate bindings a val split holding committed test, HOTELS or selection_train
content is refused, keyed on the content under any manifest (refuse_held_out_theta over pilot.data.holds_committed,
07-REVIEW CR-01, 07-09 review WR-01).
--split hotels_probe (EVAL-05 out of domain, 07-RESEARCH A7): the out-of-domain false-positive rate of a hotels_probe
dump (role ood_probe, read by pilot.dumps.load_dump) on its valid columns only, pilot.dumps.gold_valid_tags (the
manifest's ood_scoring_tags; the 22 banking-only tags of pilot.pins.OOD_SCORING_TAGS under the committed manifest).
The binding is the same function, so T and θ come from banking val (the run dir's full val dump by the val rule and
same_source, or a bound final seed's record; a selection candidate is refused off val) and never from HOTELS. Each
variant holds only its T, θ and ood_fp (pilot.metrics.fp_rate at θ 0.5 and at its θ: k of the n hotels messages with
no gold among those columns predicting at least one of them, the rate, its Wilson 95% interval and per-tag counts);
the record has in_sample false and no calibration, coverage or no-intent block, and --out writes its JSON only (no
reliability diagram). Phase 8: score.py scores hotels_probe for laya and berturk, a fine-tuned checkpoint only
through its protocol gate (one of the frozen protocol's final seeds, once, protocol_sha256 in the sidecar; 08-07).
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
line), the out-of-domain false-positive table (only when some record holds one; every section skips a record without
its block), then the in-sample caveat and the definition texts). Values are rounded for display only. The SVG links
follow the inputs' one common directory under the repo (confidence_svg_dir; records in several directories or outside
the repo link to CONFIDENCE_SVG_DIR, results/confidence), so results/final/confidence renders with its own links. The
README confidence block is this output (plan 07-08).

--verdict [--out results/final/verdict.json] (REP-02, REP-03; 08-RESEARCH A3, A5, A6, A16, A18): the verdict/1 record
of the frozen protocol, verdict_report = verdict_from(*pilot.protocol.require_frozen()), so it is refused while
protocol.json is missing or PROTOCOL_SHA256 is unset or differs. verdict_from checks every protocol input's sha256,
loads exactly the protocol's final-seed test dumps (protocol_sha256 == the pin, the seed and training.postfit_sha256
bound, every dump on the protocol's test uids in one order), seed-averages each model's probabilities at its record T
and applies the record's theta.shipped (raw: T 1, theta.raw), runs pilot.verdict.system_deltas with the protocol's
bootstrap block, recomputes the label prior's frozen fractions (pilot.protocol.prior_floors, refused unless equal),
takes the per-seed rows from run_report(..., postfit=RECORD), and pilot.verdict.build_verdict computes the criteria
and the verdict from the protocol's criteria block. --out writes the JSON plus a newline atomically.
--final-table VERDICT_JSON and --go-block VERDICT_JSON print pilot.verdict.render_final_table (REP-02's one results
table: references, each fine-tuned model as mean ± sample std with its per-seed values and its seed-averaged system,
the settings) and pilot.verdict.render_go (REP-03's mechanical GO block) of a saved record (08-12 writes them into the
README), only after pilot.protocol.require_frozen passes and pilot.verdict.check_record binds the record to it: the
protocol sha256 is the pin, the verdict rules are the code's, and every criterion, collapse flag and the verdict
recompute from the record's own numbers and the frozen criteria, so a hand-edited status or verdict is refused, never
rendered (08-REVIEW WR-03).
--verdict --fewshot RECORD (09-08, D-13): the few-shot results record (results/fewshot/fewshot.json, `make
fewshot-reports`) decides criterion (d)'s few-shot path. fewshot_input reads it (schema fewshot-results/1), refuses it
unless its protocol_sha256 is the pin, its primary k is criteria (d)'s fewshot k, the runs manifest it names (under the
repo, sha256-checked, pilot.fewshot.read_runs) is its own and its primary block equals pilot.fewshot_report.primary
recomputed from that manifest and the committed dumps, then hands {mean_delta, draws} to criterion (d) at the frozen
mean_delta_at_least: the path is MET or NOT MET, (d) PASS or FAIL, the provisional mark resolves, and the verdict's
fewshot block holds the path's status, k, mean delta, draws, the record's path and sha256, the mode and the fallback
disclosure. Without --fewshot the verdict is the Phase 8 one byte for byte (the few-shot path PENDING). --fewshot is
refused with any other mode.
The Phase 3 results table (--table) and COMPARE_THETA_RULE stay as they are: the committed comparison reports embed
them and rebuild byte for byte; the frozen seed-averaged θ rule lives in protocol.json.

usage: python report.py results/dumps/RUN_ID [--split val|test] [--out report.json] [--manifest splits.json]
       python report.py results/dumps/FINAL_SEED_RUN_ID --split test --postfit results/postfit/laya.json [--out X.json]
       python report.py --compare A_RUN[,A_RUN2...] B_RUN[,B_RUN2...] [--split test] [--B 10000] [--seed 0] [--out x.json]
       python report.py --table results/reports [--out table.md]
       python report.py --confidence results/dumps/RUN_ID --split val [--postfit results/postfit/laya.json]
                        [--out results/confidence/RUN_ID-val.json]
       python report.py --confidence results/dumps/RUN_ID --split hotels_probe [--postfit RECORD] [--out X.json]
       python report.py --confidence-table results/confidence [--out confidence.md]
       python report.py --verdict [--fewshot results/fewshot/fewshot.json] [--out results/final/verdict.json]
       python report.py --final-table results/final/verdict.json [--out table.md]
       python report.py --go-block results/final/verdict.json [--out go.md]
"""
import argparse
import json
import os

import pilot.fewshot                                        # numpy-free: read_runs (the few-shot runs manifest)
import pilot.fewshot_report                                 # numpy only: read_record, primary (09-08)
import pilot.protocol                                       # stdlib only at import: require_frozen, prior_floors
import pilot.verdict
from pilot import data, dumps, metrics                     # data.DEFAULT_MANIFEST: the committed one, at call time
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, TRAINED_ON, holds_committed, manifest_is_committed, parse_jsonl,
                        repo_relpath, require_committed_files, sha256_bytes, sha256_file, verify_split_file,
                        write_atomic)

SCHEMA_VERSION = 1
COMPARE_PROCEDURE = ("EVAL-07: per-seed probabilities averaged per message and tag, then thresholded; test messages "
                     "resampled with replacement with one index draw shared by both systems; 95% percentile interval "
                     "of A - B")
COMPARE_THETA_RULE = ("D-12 applied to the seed-averaged val probabilities; Phase 8 freezes the rule for seed-averaged "
                      "systems in protocol.json")
COMPARE_NOTE = "descriptive intervals only; the decision rule and its thresholds are frozen in protocol.json in Phase 8"
NO_VAL_REASON = "no val dump in run dir"
FINAL_SEED_REASON = ("a final seed's θ is its postfit record's; pass --postfit RECORD (the record its "
                     "training.postfit_sha256 names)")
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
# EVAL-05 out of domain: the column rule in every hotels_probe confidence record (07-RESEARCH A7)
OOD_COLUMNS_TEXT = ("the manifest's ood_scoring_tags (22 banking-only tags); θ and T come from banking val, never from "
                    "HOTELS")
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


def refuse_ood_role(run_dir, split, entry, what):
    """SystemExit for a dump whose split is not role val or test (07-RESEARCH A7): a hotels_probe dump (role ood_probe)
    has valid gold on the banking-only ood_scoring_tags columns only, all negative, so a 48-tag metric (micro-F1, the
    θ rows, the bootstrap) is undefined on it; only --confidence reads it, as the out-of-domain false-positive rate."""
    if entry["role"] not in dumps.EVAL_ROLES:
        raise SystemExit("%s: split %s has role %s; %s val or test only (micro-F1 needs every tag's gold); use "
                         "report.py --confidence RUN_DIR --split hotels_probe for the out-of-domain false-positive "
                         "rate" % (run_dir, split, entry["role"], what))


# The committed splits whose content never shapes θ or T, whatever the manifest says: test and HOTELS (07-REVIEW
# CR-01), and selection_train, the training messages of the baselines and selection candidates (07-09 review WR-01).
# They are every committed message outside val (final_train = selection_train + val), so a θ source holding any
# committed message holds only committed val ones; final_train is not listed, because it holds the val folds.
THETA_HELD_OUT = ("test", "hotels_probe", "selection_train")


def _split_content(entry, manifest_path, roles):
    """(verified manifest entry, parsed rows) of the split file a manifest entry (or a sidecar's split block) names, at
    dirname(realpath(manifest)) + entry["path"], the path rule of pilot.dumps.read_gold. The bytes are read once,
    verified against the manifest (sha256, registration, role) and parsed, so the rows checked are the rows verified."""
    path = os.path.join(os.path.dirname(os.path.realpath(manifest_path)), entry["path"])
    if not os.path.isfile(path):
        raise SystemExit("%s not found; run `make data` to write the registered splits" % path)
    with open(path, "rb") as f:
        raw = f.read()
    return verify_split_file(path, manifest_path=manifest_path, roles=roles, data=raw), parse_jsonl(raw)


def refuse_held_out_theta(where, entry, manifest_path):
    """SystemExit when the θ-source val split holds committed test, HOTELS or selection_train content (THETA_HELD_OUT),
    keyed on the content under any manifest by pilot.data.holds_committed (the entry's sha256 or uids_sha256 is the
    committed entry's, or any (uid, message) pair is in the committed file), never on the manifest's role label
    (07-REVIEW CR-01, 07-09 review WR-01; the precedents are score.py's test interlock, 05-REVIEW CR-02, the trainers'
    pilot.data.refuse_held_out, 06-REVIEW CR-01/WR-02, and latency.read_split). `entry` is a manifest entry or a
    sidecar's split block; both name the registered path. Fails closed without the committed files
    (require_committed_files, 07-09 review WR-02). Runs before read_gold and any metric."""
    require_committed_files(where, manifest_path, THETA_HELD_OUT)
    verified, rows = _split_content(entry, manifest_path, ("val",))
    for held in THETA_HELD_OUT:
        if holds_committed(verified, rows, held):
            raise SystemExit("%s: the θ-source val split %s holds committed %s content; θ is selected on banking val "
                             "only, never on test, HOTELS or training (selection_train) content, whatever the manifest "
                             "says (drop --manifest, or run `make data`)" % (where, verified["path"], held))


# Committed content a final seed trained on: final_train holds the val folds, and its pairs cover selection_train's
FINAL_SEED_TRAINED = ("val", "final_train")


def is_final_seed(side):
    """True for a dump sidecar whose training block carries a postfit_sha256: a final seed, trained on final_train."""
    tr = side.get("training")
    return isinstance(tr, dict) and tr.get("postfit_sha256") is not None


def refuse_final_seed_training_content(where, split, entry, manifest_path):
    """SystemExit when a final seed would be reported on messages it trained on (07-REVIEW WR-01): split val by name
    (final_train holds the val folds), and any split holding committed val or final_train content (FINAL_SEED_TRAINED)
    under any manifest, keyed on the content by pilot.data.holds_committed, so a manifest copy cannot present
    training-set numbers as held-out under the test name. Every report mode applies it to the reported dump of a final
    seed (is_final_seed): run_report (so evaluate.py), --compare (_load_side) and the --confidence final_seed binding
    (07-09 review CR-01). Fails closed without the committed files (require_committed_files, 07-09 review WR-02). Runs
    before the postfit record is read and before any metric; `entry` is the dump sidecar's split block."""
    if split == "val":
        raise SystemExit("%s: a final seed was trained on final_train, which holds the val folds, so its val numbers "
                         "are training-set numbers (neither held-out nor in-sample); report a final seed on test or "
                         "hotels_probe (Phase 8) with its postfit record" % where)
    require_committed_files(where, manifest_path, FINAL_SEED_TRAINED)
    verified, rows = _split_content(entry, manifest_path, dumps.READ_ROLES)
    for held in FINAL_SEED_TRAINED:
        if holds_committed(verified, rows, held):
            raise SystemExit("%s: split %s holds committed %s content, on which a final seed was trained (final_train "
                             "holds the val folds); its numbers there are training-set numbers; report a final seed "
                             "on the committed test or hotels_probe split (drop --manifest, or run `make data`)"
                             % (where, split, held))


def theta_source(run_dir, manifest_path):
    """(val logits, gold, sidecar, disclosure) of the run dir's full val dump, or None when it has no val dump.

    load_dump proves the dump is untampered, belongs to this run (run_id == run dir name) and, for a full dump, covers
    the registered val split in order (uids_sha256). A subset val dump is refused: θ is never chosen on part of val.
    run_report does not call it when the reported dump is itself a subset val dump (SUBSET_VAL_REASON). A val split
    holding committed test, HOTELS or selection_train content is refused, keyed on the content (refuse_held_out_theta,
    07-REVIEW CR-01, 07-09 review WR-01).
    """
    npz = os.path.join(run_dir, "val.npz")
    if not (os.path.exists(npz) or os.path.exists(os.path.join(run_dir, "val.json"))):
        return None
    logits, uids, side, entry = dumps.load_dump(run_dir, "val", manifest_path)
    if side["n_subset"] is not None:
        raise SystemExit("%s: θ is selected only on a full val dump; rescore val without --n" % npz)
    refuse_held_out_theta(run_dir, entry, manifest_path)
    refuse_in_sample_theta(side, run_dir)
    return logits, dumps.read_gold(entry, manifest_path, uids), side, {
        "run_id": side["run_id"], "split": "val", "dump_sha256": side["dump_sha256"],
        "uids_sha256": dumps.uids_sha256(uids), "n_messages": len(uids)}


def run_report(run_dir, split="test", manifest_path=DEFAULT_MANIFEST, postfit=None):
    """The report dict of one verified dump: shipped and raw variants, rows at 0.5 and at the val-selected θ.

    The val-selected row is null with a reason in three cases: the run dir has no val dump (NO_VAL_REASON), the
    reported dump is a --n subset val dump (SUBSET_VAL_REASON; it is reported at θ 0.5 only), or it is a final seed's
    dump without `postfit` (FINAL_SEED_REASON). A final seed's θ is never selected (its val is training data): with
    `postfit`, the record its training.postfit_sha256 names, its val-selected rows are at the record's chosen θ
    (theta.shipped at the sidecar's shipped T, theta.raw at T = 1) through final_seed_binding, the --confidence
    final_seed checks, and theta_source names the record, its chosen candidate and split val (08-RESEARCH A5,
    Pattern 3). `postfit` for any other dump is refused. A θ-source val dump from another source than the reported
    dump (same_source) is refused, and so is a final seed's dump on val (a subset one too) or on committed val or
    final_train content (refuse_final_seed_training_content, 07-09 review CR-01), before any metric.
    """
    logits, uids, side, entry = dumps.load_dump(run_dir, split, manifest_path)
    refuse_ood_role(run_dir, split, entry, "report.py RUN_DIR reports")
    final = is_final_seed(side)
    if postfit is not None and not final:
        raise SystemExit("%s: --postfit applies to a final seed's dump (training.postfit_sha256); a selection "
                         "candidate, zero-shot model or baseline takes θ from its run dir's val dump" % run_dir)
    Y = dumps.read_gold(entry, manifest_path, uids)
    if split == "val" and side["n_subset"] is not None:     # the reported dump is that subset val dump: no θ source
        src, reason = None, SUBSET_VAL_REASON
    else:
        src, reason = theta_source(run_dir, manifest_path), NO_VAL_REASON   # reads only val: the split never shapes θ
    bound = None
    if final:                                               # never on content it trained on (07-09 review CR-01)
        if postfit is None:
            refuse_final_seed_training_content(run_dir, split, side["split"], manifest_path)
            src, reason = None, FINAL_SEED_REASON
        else:                                               # the record's θ, never re-selected (A5)
            bound = final_seed_binding(run_dir, split, side, manifest_path, postfit)
    if src is not None and split != "val" and bound is None:
        same_source(side, src[2], run_dir)
    val_logits, Y_val, val_side, disclosed = src if src is not None else (None, None, None, None)
    if bound is not None:
        record, sha, chosen, _ = bound
        disclosed = {"postfit": _postfit_block(postfit, sha, record), "chosen": chosen.get("dump_run_id"),
                     "split": "val"}
    variants = {}
    for name, T, T_val in (("shipped", side["temperature"]["shipped"],
                            None if src is None else val_side["temperature"]["shipped"]),
                           ("raw", 1.0, 1.0)):
        P = metrics.probabilities(logits, T)
        rows = {"0.5": metrics.metrics(Y, P, 0.5)}
        v = {"temperature": T, "rows": rows}
        if bound is not None:                               # theta.shipped for shipped, theta.raw for raw
            sel = bound[2]["theta"][name]
            rows["val_selected"] = dict(metrics.metrics(Y, P, metrics.theta_vector(sel)), theta=sel, in_sample=False)
        elif src is None:
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


def refuse_held_out_record(where, path, record, manifest_path):
    """SystemExit unless a final seed's postfit record fitted its T and θ on banking val content (07-09 review WR-03,
    AR-07-02): the record's split block (path, sha256, uids_sha256) holding committed THETA_HELD_OUT content by its
    hashes is refused; a record fitted on the committed val bytes passes; any other record's rows are read under the
    manifest it was fitted on (the record's manifest.sha256 must be this --manifest's) and checked by
    refuse_held_out_theta. A record of another manifest is refused: it holds hashes only, so its rows cannot be checked
    (fail closed). pilot.data.DEFAULT_MANIFEST is read at call time."""
    split = record.get("split") or {}
    for held in THETA_HELD_OUT:
        if holds_committed(split, [], held):
            raise SystemExit("%s: --postfit %s was fitted on a val split that holds committed %s content; a final "
                             "seed's T and θ come from banking val only (refit the record on the committed val)"
                             % (where, path, held))
    with open(data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        if split.get("sha256") == json.load(f)["splits"]["val"]["sha256"]:
            return                                          # the committed val bytes
    if (record.get("manifest") or {}).get("sha256") != sha256_file(manifest_path):
        raise SystemExit("%s: --postfit %s was fitted on val sha256 %s under manifest sha256 %s, neither the committed "
                         "val nor this manifest's, so its val rows cannot be content-checked; report the final seed "
                         "under the manifest the record was fitted on" % (where, path, split.get("sha256"),
                                                                          (record.get("manifest") or {}).get("sha256")))
    refuse_held_out_theta(where, split, manifest_path)


def _postfit_block(path, sha, record):
    return {"path": repo_relpath(path), "sha256": sha, "system_kind": record.get("system_kind")}


def final_seed_binding(run_dir, split, side, manifest_path, postfit):
    """(record, sha256, chosen, T) of a final seed's dump (is_final_seed) bound to its postfit record, the one binding
    of report.py RUN_DIR --postfit (run_report) and the --confidence final_seed branch (confidence_binding). In order:
    the reported split holds no content the seed trained on (refuse_final_seed_training_content, before the record is
    read), --postfit is given, its bytes hash to training.postfit_sha256, its system kind is the sidecar's, it was
    fitted on banking val content (refuse_held_out_record, 07-09 review WR-03) and the sidecar's shipped T is the
    record's chosen T within T_MATCH. SystemExit naming the first failure; θ is then the record's chosen θ, never
    selected again (08-RESEARCH A5)."""
    tr = side["training"]
    kind = (side.get("system") or {}).get("kind")
    refuse_final_seed_training_content(run_dir, split, side["split"], manifest_path)
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
    refuse_held_out_record(run_dir, postfit, record, manifest_path)     # T and θ fitted on banking val (WR-03)
    chosen = record.get("chosen") or {}
    T = side["temperature"]["shipped"]
    if chosen.get("temperature") is None or abs(T - chosen["temperature"]) > T_MATCH:
        raise SystemExit("%s: the final seed ships T %r, the record's chosen T is %r; a final seed serves its "
                         "record's T" % (run_dir, T, chosen.get("temperature")))
    return record, sha, chosen, T


def confidence_binding(run_dir, split, side, manifest_path, postfit=None):
    """Where the confidence report of one dump takes T and θ from (kind postfit_candidate, final_seed or
    val_selected; see the module docstring). SystemExit naming the reason for every other combination, before any
    metric is computed. Each variant's theta is the full select_theta-shaped dict."""
    tr = side.get("training")
    kind = (side.get("system") or {}).get("kind")
    rid = side["run_id"]
    if is_final_seed(side):
        record, sha, chosen, T = final_seed_binding(run_dir, split, side, manifest_path, postfit)
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
        refuse_held_out_theta(run_dir, side["split"], manifest_path)    # the val split its record's T and θ fit on
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
    (no-intent) at θ 0.5 and at the variant's θ. A hotels_probe dump (role ood_probe) gets per variant only its T, θ
    (the same binding: banking val, never HOTELS) and ood_fp, pilot.metrics.fp_rate over pilot.dumps.gold_valid_tags
    at θ 0.5 and at the variant's θ; no calibration, coverage or no-intent block (its gold is valid on those columns
    only, and all of it is negative)."""
    logits, uids, side, entry = dumps.load_dump(run_dir, split, manifest_path)
    if side["n_subset"] is not None:
        raise SystemExit("%s: a --n subset dump is never reported; rescore %s without --n" % (run_dir, split))
    ood = entry["role"] == "ood_probe"
    cols = dumps.gold_valid_tags(entry, manifest_path) if ood else None     # before any metric: the valid columns
    Y = dumps.read_gold(entry, manifest_path, uids)
    binding = confidence_binding(run_dir, split, side, manifest_path, postfit)
    variants = {}
    for name in ("headline", "raw"):
        v = binding["variants"][name]
        P = metrics.probabilities(logits, v["temperature"])
        if ood:                                             # θ and T from banking val; only the OOD rate on HOTELS
            variants[name] = dict(v, ood_fp={"0.5": metrics.fp_rate(Y, P, 0.5, cols),
                                             "theta": metrics.fp_rate(Y, P, metrics.theta_vector(v["theta"]), cols)})
            continue
        variants[name] = dict(v, calibration=metrics.calibration(Y, P, v["theta"]["eligible_tags"]),
                              selective={"0.5": metrics.selective(Y, P, 0.5),
                                         "theta": metrics.selective(Y, P, metrics.theta_vector(v["theta"]))},
                              no_intent_fp={"0.5": metrics.fp_rate(Y, P, 0.5),
                                            "theta": metrics.fp_rate(Y, P, metrics.theta_vector(v["theta"]))})
    tr = side.get("training")
    in_sample = split == "val"
    if ood:
        definitions = {"false_positive": metrics.FP_RULE, "wilson": WILSON_TEXT, "ood_columns": OOD_COLUMNS_TEXT}
    else:
        definitions = {"calibration": metrics.CALIBRATION_RULES, "confidence": metrics.CONFIDENCE_RULE,
                       "coverage": metrics.COVERAGE_RULE, "aurc": metrics.AURC_RULE,
                       "false_positive": metrics.FP_RULE, "wilson": WILSON_TEXT}
    return {"schema": CONFIDENCE_SCHEMA, "schema_version": SCHEMA_VERSION, "run_id": side["run_id"],
            "split": side["split"], "system": side["system"],
            "training": ({"run_id": tr.get("run_id"), "split": tr.get("split"), "epoch": tr.get("epoch")}
                         if isinstance(tr, dict) else None),
            "dump_sha256": side["dump_sha256"], "uids_sha256": dumps.uids_sha256(uids), "n_messages": len(uids),
            "manifest": {"path": repo_relpath(manifest_path), "sha256": sha256_file(manifest_path),
                         "is_committed": manifest_is_committed(manifest_path)},
            "binding": {k: v for k, v in binding.items() if k != "variants"},
            "in_sample": in_sample, "caveat": IN_SAMPLE_CAVEAT if in_sample else None,
            "definitions": definitions, "variants": variants}

def _load_side(run_dirs, split, manifest_path):
    """[(run_dir, split dump, val dump)] of one system's per-seed run dirs, each dump verified by load_dump."""
    if not run_dirs:
        raise SystemExit("no seed dumps to average: each side of --compare needs at least one run dir")
    runs, seen = [], set()
    for d in run_dirs:
        if not (os.path.exists(os.path.join(d, "val.npz")) or os.path.exists(os.path.join(d, "val.json"))):
            raise SystemExit("compare needs a full val dump in every run dir; %s has none (score.py on val)" % d)
        pair = (dumps.load_dump(d, split, manifest_path), dumps.load_dump(d, "val", manifest_path))
        refuse_ood_role(d, split, pair[0][3], "report.py --compare compares")
        refuse_held_out_theta(d, pair[1][3], manifest_path)
        same_source(pair[0][2], pair[1][2], d)
        refuse_in_sample_theta(pair[1][2], d)
        if is_final_seed(pair[0][2]):                       # the compared dump itself (07-09 review CR-01)
            refuse_final_seed_training_content(d, split, pair[0][2]["split"], manifest_path)
        rid = pair[0][2]["run_id"]                        # == the run dir name (load_dump checks it)
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


def _protocol_inputs(protocol, root):
    """SystemExit unless every protocol input exists under root and its bytes still hash to the frozen sha256 (named,
    before anything is read or computed); returns the inputs block."""
    inputs = protocol.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise SystemExit("the protocol lists no inputs; the verdict reads only inputs the frozen protocol binds")
    for rel, want in inputs.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise SystemExit("protocol input %s not found under %s; restore it with git (the protocol is never "
                             "rebuilt)" % (rel, root))
        have = sha256_file(p)
        if have != want:
            raise SystemExit("protocol input %s changed: sha256 %s, the frozen protocol has %s; restore it with git "
                             "(the protocol is never rebuilt)" % (rel, have, want))
    return inputs


def _bound_input(inputs, root, ref, what):
    """root/<ref path> after checking that the protocol lists it as an input with ref's sha256."""
    if not isinstance(ref, dict) or inputs.get(ref.get("path")) is None or inputs[ref["path"]] != ref.get("sha256"):
        raise SystemExit("%s %r is not a protocol input with that sha256; the verdict reads only bound inputs"
                         % (what, ref))
    return os.path.join(root, ref["path"])


def _read_json_input(path, what):
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except ValueError as e:
            raise SystemExit("%s %s is not valid JSON (%s); restore it with git" % (what, path, e))


def _final_manifest(inputs, root, s, kind, rec_sha):
    """(final manifest, {seed: entry}) of a system's postfit-final/1 manifest, refused unless it is bound to the
    system's record and holds every protocol final seed with the same dump and train run ids. A protocol whose final
    seeds are fewer (or other) than its listed seeds is refused: every listed seed is reported, none dropped."""
    final = _read_json_input(_bound_input(inputs, root, s.get("final_manifest"), "%s final manifest" % kind),
                             "%s final manifest" % kind)
    if final.get("schema") != "postfit-final/1" or (final.get("postfit") or {}).get("sha256") != rec_sha:
        raise SystemExit("%s final manifest is not a postfit-final/1 bound to the record %s" % (kind, rec_sha))
    fs = s.get("final_seeds") or []
    if [f.get("seed") for f in fs] != list(s.get("seeds") or []) or not fs:
        raise SystemExit("%s: the protocol has final seeds %s for the listed seeds %s; every listed final seed is "
                         "scored and reported" % (kind, [f.get("seed") for f in fs], s.get("seeds")))
    entries = {e.get("seed"): e for e in final.get("seeds") or []}
    for f in fs:
        e = entries.get(f["seed"]) or {}
        if (e.get("dump_run_id"), e.get("train_run_id")) != (f["dump_run_id"], f["train_run_id"]):
            raise SystemExit("%s final seed %s: the final manifest holds %r, the protocol %r"
                             % (kind, f["seed"], (e.get("dump_run_id"), e.get("train_run_id")),
                                (f["dump_run_id"], f["train_run_id"])))
    return final, entries


def _final_seed_dump(dump_root, kind, f, sha, rec_sha, manifest_path):
    """(run dir, load_dump result) of one protocol final seed's test dump: it exists, carries the protocol's sha256,
    the protocol seed, a final seed's training block bound to the system's postfit record and the system's kind."""
    d = os.path.join(dump_root, f["dump_run_id"])
    if not all(os.path.isfile(os.path.join(d, "test." + ext)) for ext in ("npz", "json")):
        raise SystemExit("%s final seed %s (dump run id %s) has no test dump under %s; score it once through the "
                         "protocol gate (score.py, 08-11)" % (kind, f["seed"], f["dump_run_id"], dump_root))
    dump = dumps.load_dump(d, "test", manifest_path)
    side = dump[2]
    where = "%s final seed %s (%s)" % (kind, f["seed"], f["dump_run_id"])
    if side.get("protocol_sha256") != sha:
        raise SystemExit("%s: the test dump carries protocol_sha256 %s, the frozen protocol is %s; only a dump "
                         "scored through this protocol's gate is read" % (where, side.get("protocol_sha256"), sha))
    if side.get("seed") != f["seed"]:
        raise SystemExit("%s: the test dump records seed %s" % (where, side.get("seed")))
    tr = side.get("training") if is_final_seed(side) else {}
    if tr.get("postfit_sha256") != rec_sha:
        raise SystemExit("%s: the test dump is bound to postfit_sha256 %s, the system's record is %s"
                         % (where, tr.get("postfit_sha256"), rec_sha))
    if (side.get("system") or {}).get("kind") != kind or tr.get("run_id") != f["train_run_id"]:
        raise SystemExit("%s: the test dump is a %s dump of train run %s, not %s"
                         % (where, (side.get("system") or {}).get("kind"), tr.get("run_id"), f["train_run_id"]))
    return d, dump


def _global_counts(Y, P, theta):
    """Micro (tp, fp, fn) summed over the 48 tags at one global θ (the collapse rule's test floor)."""
    tags = metrics.metrics(Y, P, theta)["per_tag"].values()
    return {k: int(sum(t[k] for t in tags)) for k in ("tp", "fp", "fn")}


def _reference_rows(protocol, inputs, root, test):
    """REP-02's reference rows from the protocol's reference reports (bound inputs): each a full report on the
    protocol's test split, its shipped val-selected row (D-12 on its own val dump)."""
    rows = []
    for name, ref in ((protocol.get("references") or {}).get("rows") or {}).items():
        rel = ref.get("report")
        if rel not in inputs:
            raise SystemExit("reference %s: report %s is not a protocol input" % (name, rel))
        r = _read_json_input(os.path.join(root, rel), "reference report")
        split = r.get("split") or {}
        if (split.get("sha256"), split.get("uids_sha256")) != (test["sha256"], test["uids_sha256"]) or r.get("subset"):
            raise SystemExit("reference %s: %s is not a full report on the protocol's test split" % (name, rel))
        v = (r.get("variants") or {}).get("shipped") or {}
        sel = (v.get("rows") or {}).get("val_selected")
        if not sel:
            raise SystemExit("reference %s: %s has no val-selected row" % (name, rel))
        system = r.get("system") or {}
        rows.append(dict({"name": name, "run_id": r.get("run_id"), "report": {"path": rel, "sha256": inputs[rel]},
                          "kind": system.get("kind"), "checkpoint": system.get("checkpoint"),
                          "baseline_split": (system.get("baseline_source") or {}).get("split"),
                          "temperature": v.get("temperature"),
                          "theta": {"global": sel["theta"]["global"], "n_per_tag": len(sel["theta"]["per_tag"])}},
                         **{k: sel[k] for k in pilot.verdict.TABLE_METRICS + ("n", "n_top1")}))
    if not rows:
        raise SystemExit("the protocol lists no reference report; REP-02's table needs its reference rows")
    return rows


def _latency_rows(inputs, root):
    """(c)'s rows {system, device, n, p50, p95} of the pooled summary of every latency report among the inputs."""
    rows = []
    for rel in inputs:
        if rel.startswith("results/latency/") and rel.endswith(".json"):
            r = _read_json_input(os.path.join(root, rel), "latency report")
            p = r.get("pooled") or {}
            rows.append({"system": r.get("system"), "device": r.get("device"), "n": p.get("n"), "p50": p.get("p50"),
                         "p95": p.get("p95")})
    return rows


def frozen_code(protocol):
    """SystemExit unless the frozen protocol's definitions block, and its bootstrap block but for the parameters
    (pilot.verdict.BOOT_KEYS: B, seed, level, coverage, read from the protocol), equal the live code's
    (pilot.protocol.definitions_block, bootstrap_block): the verdict is computed by the code the protocol froze only
    (08-REVIEW WR-04). Names the differing keys."""
    boot = pilot.verdict.BOOT_KEYS
    for key, live in (("definitions", pilot.protocol.definitions_block()),
                      ("bootstrap", pilot.protocol.bootstrap_block())):
        have = protocol.get(key) if isinstance(protocol, dict) else None
        if not isinstance(have, dict):
            raise SystemExit("the frozen protocol has no %s block; the verdict is computed by the frozen code only"
                             % key)
        skip = boot if key == "bootstrap" else ()
        diff = sorted(k for k in set(have) | set(live)
                      if k not in skip and (k not in have or k not in live or have[k] != live[k]))
        if diff:
            raise SystemExit("the code's %s (%s) differ from the frozen protocol's; the verdict is computed by the "
                             "frozen code only (restore the code the protocol froze)" % (key, ", ".join(diff)))


def _root_relpath(path, root):
    """`path` relative to root with forward slashes ("<outside-repo>/NAME" outside it: never a local path)."""
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    if rel == ".." or rel.startswith(".." + os.sep) or os.path.isabs(rel):
        return "<outside-repo>/" + os.path.basename(os.path.normpath(path))
    return rel.replace(os.sep, "/")


FEWSHOT_FIX = "rebuild it with `make fewshot-reports`"


def fewshot_input(protocol, sha, path, root, dump_root, manifest_path):
    """Criterion (d)'s few-shot input {"mean_delta", "draws", "k", "record": {path, sha256}, "mode", "disclosure"} of
    the few-shot results record at `path` (09-08, D-13), refused (one line) unless: its schema is
    pilot.fewshot_report.RESULTS_SCHEMA (read_record); its protocol_sha256 is the pin `sha`; its primary k is the frozen
    criteria (d)'s fewshot k; the runs manifest it names exists under root, passes pilot.fewshot.read_runs (check_runs
    under the D-08 decision record, whose bytes it names) and hashes to the record's runs sha256; the record's decision
    is that manifest's (sha256, mode) with the frozen disclosure; and its primary block equals
    pilot.fewshot_report.primary recomputed from that manifest and the dumps under dump_root (a JSON round trip, no
    tolerance). The numbers criterion (d) reads are thus the dumps', never a hand edit of the record."""
    record, rec_sha = pilot.fewshot_report.read_record(path)
    what = "few-shot record %s" % _root_relpath(path, root)
    if record.get("protocol_sha256") != sha:
        raise SystemExit("%s was built under protocol sha256 %s, not the frozen %s; %s"
                         % (what, record.get("protocol_sha256"), sha, FEWSHOT_FIX))
    k = pilot.verdict._fewshot_k(protocol)
    p = record.get("primary") if isinstance(record.get("primary"), dict) else {}
    if p.get("k") != k:
        raise SystemExit("%s: primary k %r is not criteria (d)'s fewshot k %r; criterion (d) reads the frozen k only"
                         % (what, p.get("k"), k))
    ref = record.get("runs") if isinstance(record.get("runs"), dict) else {}
    if not isinstance(ref.get("path"), str) or not isinstance(ref.get("sha256"), str):
        raise SystemExit("%s names no runs manifest {path, sha256}; %s" % (what, FEWSHOT_FIX))
    runs, runs_sha = pilot.fewshot.read_runs(protocol, sha, os.path.join(root, ref["path"]))
    if runs is None:
        raise SystemExit("%s: its runs manifest %s is not found under %s; restore it with git" % (what, ref["path"],
                                                                                                root))
    if runs_sha != ref["sha256"]:
        raise SystemExit("%s names runs manifest sha256 %s, but %s hashes to %s; %s"
                         % (what, ref["sha256"], ref["path"], runs_sha, FEWSHOT_FIX))
    disclosure = pilot.verdict._fallback_disclosure(protocol) if runs["mode"] == "fallback" else None
    dec = record.get("decision") if isinstance(record.get("decision"), dict) else {}
    if (dec.get("sha256"), dec.get("mode"), dec.get("disclosure")) != (runs["decision"]["sha256"], runs["mode"],
                                                                        disclosure):
        raise SystemExit("%s: decision %r is not the runs manifest's (sha256 %s, mode %s, disclosure %r); %s"
                         % (what, dec, runs["decision"]["sha256"], runs["mode"], disclosure, FEWSHOT_FIX))
    want = pilot.fewshot_report.primary(protocol, sha, runs, runs_sha, dump_root, manifest_path)
    if json.loads(json.dumps(want)) != record.get("primary"):
        raise SystemExit("%s: its primary block does not recompute from the dumps its runs manifest lists; the record "
                         "was edited, %s" % (what, FEWSHOT_FIX))
    return {"mean_delta": want["mean_delta"], "draws": want["draws"], "k": k,
            "record": {"path": _root_relpath(path, root), "sha256": rec_sha}, "mode": runs["mode"],
            "disclosure": disclosure}


def verdict_from(protocol, sha, manifest_path=DEFAULT_MANIFEST, root=None, dump_root=None, fewshot=None):
    """The verdict/1 record (REP-02, REP-03) of a frozen protocol (the dict and its sha256), computed mechanically by
    pilot.verdict.build_verdict from protocol-bound inputs only (08-RESEARCH A3, A5, A6, A16, A18):
    - every protocol input exists under root (default the repo) and hashes to its frozen sha256, and the manifest is
      the protocol's data/splits.json;
    - per system (laya, berturk) its postfit record and final manifest are bound inputs; the final seeds are exactly
      the listed seeds; each final seed's test dump (dump_root, default root/<scoring.dump_root>) exists, carries
      protocol_sha256 == sha, its seed and a final seed's training block bound to the record; all six score the
      protocol's test uids in one order (unpaired otherwise), none a --n subset;
    - the per-seed rows are run_report(..., postfit=RECORD)'s shipped val-selected rows at the record θ (08-04), plus
      the micro counts at the record's global shipped θ (the test floor) and the final manifest's loss_by_epoch;
    - each model's probabilities at each seed's shipped T (the record T, by the binding) are seed-averaged and
      thresholded at the record's theta.shipped (raw: T 1, theta.raw); pilot.verdict.system_deltas with the protocol's
      bootstrap block gives the headline and raw deltas, Laya - BERTurk;
    - the label prior's val and test fractions and θ are recomputed (pilot.protocol.prior_floors) and refused unless
      equal to the frozen ones; the selection runs' objective counts and loss come from each record's chosen candidate;
    - the latency rows come from the protocol's latency reports and the reference rows from its reference reports.
    - the protocol's definitions block and the code texts of its bootstrap block (every key but the parameters
      pilot.verdict.BOOT_KEYS, which the code reads from the protocol) equal the live code's
      (pilot.protocol.definitions_block, bootstrap_block), so a metric or bootstrap text changed after the freeze is
      refused rather than computing a verdict that names the frozen sha256 (08-REVIEW WR-04; verdict/1 holds no code
      state, so a drift in the code under unchanged texts is caught by the data-marked byte rebuild only).
    - fewshot (09-08, D-13): None leaves criterion (d)'s few-shot path PENDING (the Phase 8 verdict, byte for byte); a
      few-shot results record path gives it fewshot_input's numbers, recomputed from the dumps under dump_root, so the
      path is MET or NOT MET at the frozen mean_delta_at_least and the provisional mark resolves.
    numpy only; SystemExit naming the first failure."""
    frozen_code(protocol)
    root = REPO_ROOT if root is None else root
    inputs = _protocol_inputs(protocol, root)
    if "data/splits.json" in inputs and sha256_file(manifest_path) != inputs["data/splits.json"]:
        raise SystemExit("%s is not the protocol's data/splits.json (sha256 %s); drop --manifest"
                         % (manifest_path, inputs["data/splits.json"]))
    dump_root = os.path.join(root, protocol["scoring"]["dump_root"]) if dump_root is None else dump_root
    few = None if fewshot is None else fewshot_input(protocol, sha, fewshot, root, dump_root, manifest_path)
    test = protocol["splits"]["test"]
    systems, runs, recs = {}, [], {}
    for kind in pilot.verdict.KINDS:
        s = (protocol.get("systems") or {}).get(kind)
        if not isinstance(s, dict):
            raise SystemExit("the protocol has no %s system" % kind)
        rec_path = _bound_input(inputs, root, s.get("postfit"), "%s postfit record" % kind)
        record, rec_sha = read_postfit(rec_path)
        if record.get("system_kind") != kind:
            raise SystemExit("%s is a %s record, not %s" % (s["postfit"]["path"], record.get("system_kind"), kind))
        recs[kind] = (record, rec_sha, rec_path) + _final_manifest(inputs, root, s, kind, rec_sha)
        for f in s["final_seeds"]:
            runs.append((kind, f) + _final_seed_dump(dump_root, kind, f, sha, rec_sha, manifest_path))
    _check_paired([(r[2], r[3]) for r in runs], 1)          # one message order over all six seeds (EVAL-07)
    _, uids, _, entry = runs[0][3]
    if (entry["sha256"], dumps.uids_sha256(uids)) != (test["sha256"], test["uids_sha256"]):
        raise SystemExit("the final-seed test dumps score split %s (uids_sha256 %s), not the protocol's test split"
                         % (entry["sha256"], dumps.uids_sha256(uids)))
    for kind, f, d, dump in runs:
        if dump[2]["n_subset"] is not None:
            raise SystemExit("%s: a --n subset dump is never part of the verdict" % d)
    Y = dumps.read_gold(entry, manifest_path, uids)
    P = {k: {"headline": [], "raw": []} for k in pilot.verdict.KINDS}
    for kind, f, d, (logits, _, side, _) in runs:
        record, rec_sha, rec_path, _, entries = recs[kind]
        sel = run_report(d, "test", manifest_path, postfit=rec_path)["variants"]["shipped"]
        row = sel["rows"]["val_selected"]
        T = side["temperature"]["shipped"]
        Ps = metrics.probabilities(logits, T)
        P[kind]["headline"].append(Ps)
        P[kind]["raw"].append(metrics.probabilities(logits, 1.0))
        systems.setdefault(kind, []).append(dict(
            {"seed": f["seed"], "dump_run_id": f["dump_run_id"], "dump_sha256": side["dump_sha256"],
             "train_run_id": side["training"]["run_id"], "temperature": T},
            **{m: row[m] for m in pilot.verdict.TABLE_METRICS + ("n_top1",)},
            test_counts=_global_counts(Y, Ps, record["chosen"]["theta"]["shipped"]["global"]),
            loss_by_epoch=entries[f["seed"]].get("loss_by_epoch")))
    seeded = {}
    A, B = pilot.verdict.KINDS                              # every delta is A - B (D-10: no system name in report.py)
    for variant, key in (("headline", "shipped"), ("raw", "raw")):
        Pv = {k: metrics.seed_average(P[k][variant]) for k in pilot.verdict.KINDS}
        th = {k: metrics.theta_vector(recs[k][0]["chosen"]["theta"][key]) for k in pilot.verdict.KINDS}
        d = pilot.verdict.system_deltas(Y, Pv[A], th[A], Pv[B], th[B], protocol["bootstrap"])
        seeded[variant] = dict(d, a=A, b=B,
                               temperature={k: (recs[k][0]["chosen"]["temperature"] if variant == "headline" else 1.0)
                                            for k in pilot.verdict.KINDS},
                               theta={k: "theta.%s" % key for k in pilot.verdict.KINDS})
    floors = pilot.protocol.prior_floors(manifest_path, root)
    for k in pilot.verdict.PRIOR_KEYS:
        if floors[k] != protocol["collapse"].get(k):
            raise SystemExit("the label prior's %s recomputes to %r, the frozen protocol has %r; restore the prior's "
                             "dumps with git" % (k, floors[k], protocol["collapse"].get(k)))
    selection = []
    for kind in pilot.verdict.KINDS:
        record = recs[kind][0]
        chosen = record["chosen"]
        cands = [c for c in record.get("candidates") or [] if c.get("dump_run_id") == chosen["dump_run_id"]]
        if len(cands) != 1:
            raise SystemExit("%s: the record lists %d candidates for its chosen dump %s" % (kind, len(cands),
                                                                                         chosen["dump_run_id"]))
        selection.append({"kind": kind, "seed": cands[0].get("seed"), "train_run_id": chosen["train_run_id"],
                          "dump_run_id": chosen["dump_run_id"], "objective": cands[0]["objective"],
                          "loss_by_epoch": cands[0]["loss_by_epoch"]})
    out_systems = {}
    for kind in pilot.verdict.KINDS:
        s, (record, _, _, final, _) = protocol["systems"][kind], recs[kind]
        c = record["chosen"]
        out_systems[kind] = {
            "kind": kind, "postfit": dict(s["postfit"]), "final_manifest": dict(s["final_manifest"]),
            "settings": {"chosen": c["dump_run_id"], "train_run_id": c["train_run_id"], "temperature": c["temperature"],
                         "theta": {v: {"global": c["theta"][v]["global"], "n_per_tag": len(c["theta"][v]["per_tag"])}
                                   for v in ("shipped", "raw")},
                         "epoch": c["epoch"], "epochs_schedule": c["epochs_schedule"], "encoder_lr": c["encoder_lr"],
                         "head_lr": (c.get("canonical") or {}).get("head_lr"),
                         "config_sha256_sans_seed": final.get("config_sha256_sans_seed")},
            "seeds": list(s["seeds"]), "per_seed": systems[kind]}
    inputs_v = {"split": {"name": "test", "path": test.get("path"), "sha256": test["sha256"],
                          "uids_sha256": test["uids_sha256"], "n": int(Y.shape[0]),
                          "n_no_intent": int((~Y.any(axis=1)).sum())},
                "systems": out_systems, "references": _reference_rows(protocol, inputs, root, test),
                "seed_averaged": seeded, "collapse": {"selection": selection}, "latency": _latency_rows(inputs, root)}
    if few is not None:
        inputs_v["fewshot"] = few
    return pilot.verdict.build_verdict(protocol, sha, inputs_v)


def verdict_report(manifest_path=DEFAULT_MANIFEST, root=None, dump_root=None, fewshot=None):
    """verdict_from of the frozen protocol: pilot.protocol.require_frozen() (refused while protocol.json is missing
    or PROTOCOL_SHA256 is unset or differs), so the verdict is never computed under an unfrozen rule; `fewshot` is the
    few-shot results record (None: the few-shot path stays PENDING)."""
    protocol, sha = pilot.protocol.require_frozen()
    return verdict_from(protocol, sha, manifest_path, root, dump_root, fewshot)


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
    if not _has(r, "calibration"):
        raise SystemExit("%s (%s): an out-of-domain record has no calibration block, so no reliability diagram"
                         % (r["run_id"], r["split"]["name"]))
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


def _has(r, block):
    """True when every variant of record r holds `block` (an out-of-domain record holds only ood_fp)."""
    return all(block in r["variants"][name] for name in CONFIDENCE_VARIANTS)


def _confidence_label(r):
    tr = r["training"]
    if tr is not None:
        return "%s (train run %s, epoch %s)" % (r["system"]["kind"], tr["run_id"], tr["epoch"])
    return _system_label(r["system"])


def _calibration_section(records):
    records = [r for r in records if _has(r, "calibration")]
    if not records:
        return None
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


def _diagram_section(records, svg_dir=None):
    """A link per record with a calibration block to its reliability diagram, <svg_dir>/<run_id>-<split>.svg
    (default CONFIDENCE_SVG_DIR, the Phase 7 records' directory)."""
    records = [r for r in records if _has(r, "calibration")]
    if not records:
        return None
    return "\n\n".join(["### Reliability diagrams"] + [
        "![reliability diagram %s (%s)](%s/%s-%s.svg)"
        % (r["run_id"], r["split"]["name"], svg_dir or CONFIDENCE_SVG_DIR, r["run_id"], r["split"]["name"])
        for r in records])


ABSTENTION_RULES = ("confidence", "coverage", "aurc")
SELECTIVE_ROWS = (("0.5", "0.5"), ("theta", "val-selected, global %.2f"))


def _abstention_section(records):
    """EVAL-04: precision and accuracy at every coverage and the AURC per record, variant and θ row, then the rule
    texts (one set, shared by every record) and CONFIDENCE_DISCLOSURE."""
    records = [r for r in records if _has(r, "selective")]
    if not records:
        return None
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
    records = [r for r in records if _has(r, "no_intent_fp")]
    if not records:
        return None
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


OOD_RULES = ("false_positive", "wilson", "ood_columns")
OOD_ROWS = (("0.5", "0.5"), ("theta", "banking-val, global %.2f"))


def _ood_section(records):
    """EVAL-05 out of domain: the HOTELS false-positive rate on the banking-only columns per record, variant and θ row
    (θ 0.5 and the banking-val θ), as k/n with its rate and Wilson 95% interval, then the rule texts. Only records that
    hold ood_fp; none at all (every committed Phase 7 record) leaves the section out."""
    records = [r for r in records if _has(r, "ood_fp")]
    if not records:
        return None
    rules = {json.dumps([r["definitions"].get(k) for k in OOD_RULES]) for r in records}
    texts = [records[0]["definitions"].get(k) for k in OOD_RULES]
    if len(rules) != 1 or None in texts:
        raise SystemExit("the out-of-domain records disagree on (or lack) the false-positive definitions; rebuild them "
                         "with report.py --confidence RUN_DIR --split hotels_probe")
    rows, cols = [], set()
    for r in records:
        for name in CONFIDENCE_VARIANTS:
            v = r["variants"][name]
            for key, label in OOD_ROWS:
                f = v["ood_fp"][key]
                cols.add(json.dumps(f["cols"]))
                theta = label % v["theta"]["global"] if "%" in label else label
                ci = "n/a" if f["wilson95"] is None else "[%.3f, %.3f]" % tuple(f["wilson95"])
                rows.append((_confidence_label(r), r["run_id"], name, theta, "%d/%d" % (f["k"], f["n"]),
                             _num(f["rate"]), ci))
    if len(cols) != 1:
        raise SystemExit("the out-of-domain records disagree on the scored columns; rebuild them with report.py "
                         "--confidence RUN_DIR --split hotels_probe")
    n_cols = len(json.loads(next(iter(cols))))
    header = ("system", "run_id", "variant", "θ", "k/n", "rate", "Wilson 95%")
    return "\n\n".join(["### Out-of-domain false positives (HOTELS, %d banking-only tags)" % n_cols, _md(header, rows),
                         "\n".join("- %s: %s" % (k, t) for k, t in zip(OOD_RULES, texts))])


def _definitions_section(records):
    records = [r for r in records if "calibration" in r["definitions"]]
    if not records:
        return None
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
# level-3 heading (the README carries it inside its own section, plan 07-08), or None when no record holds its block
# (an out-of-domain record holds only ood_fp); later Phase 7 plans add theirs before the definitions.
CONFIDENCE_SECTIONS = [_calibration_section, _diagram_section, _abstention_section, _no_intent_section,
                       _ood_section, _definitions_section]


CONFIDENCE_SVG_DIR = "results/confidence"                 # the Phase 7 records' directory: the README's SVG links


def confidence_svg_dir(paths, root=REPO_ROOT):
    """The directory --confidence-table links the reliability diagrams to: the forward-slash path, relative to root,
    of the one directory every input record sits in, when that directory is under root (not root itself); else
    CONFIDENCE_SVG_DIR. Never an absolute or outside-repo path. 08-12 renders the test/HOTELS block from
    results/final/confidence with it."""
    dirs = {os.path.dirname(os.path.realpath(p)) for p in paths}
    if len(dirs) != 1:
        return CONFIDENCE_SVG_DIR
    try:
        rel = os.path.relpath(next(iter(dirs)), os.path.realpath(root))
    except ValueError:                                      # another drive (Windows): not under root
        return CONFIDENCE_SVG_DIR
    if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return CONFIDENCE_SVG_DIR
    return rel.replace(os.sep, "/")


def render_confidence(reports, svg_dir=CONFIDENCE_SVG_DIR):
    """Markdown of saved confidence/1 records, in the given order (callers pass them in sorted path order): the
    calibration table (headline then raw per record), a link to each record's reliability diagram (under svg_dir,
    default CONFIDENCE_SVG_DIR), the abstention
    table with its rule texts and CONFIDENCE_DISCLOSURE, the no-intent false-positive table, the out-of-domain
    false-positive table (only when some record is a hotels_probe one), the in-sample caveat and the definition texts.
    Each section skips a record without its block. Values are rounded for display only; identical input gives
    identical text."""
    records = [_confidence_record(r) for r in reports]
    if not records:
        raise SystemExit("no confidence records to render")
    parts = [build(records, svg_dir) if build is _diagram_section else build(records)     # only it links files
             for build in CONFIDENCE_SECTIONS]
    return "\n\n".join(p for p in parts if p is not None) + "\n"


def _run_dirs(ap, text):
    dirs = text.split(",")
    if not all(dirs):
        ap.error("--compare: empty run dir in %r (use a comma-separated list of run dirs)" % text)
    return dirs


def main(argv=None):
    ap = argparse.ArgumentParser(description="recompute metrics from a logit dump, or compare two systems' dumps")
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--split", default=None, choices=("val", "test", "hotels_probe"),
                    help="the dump's split (default test for RUN_DIR and --compare; required for --confidence; "
                         "hotels_probe only with --confidence)")
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
                    help="with --confidence: the postfit record binding a selection candidate's or final seed's T, "
                         "θ; with RUN_DIR: a final seed's record (its θ)")
    ap.add_argument("--confidence-table", nargs="+", metavar="PATH", default=None,
                    help="markdown of saved confidence/1 records (a directory expands to its sorted *.json)")
    ap.add_argument("--verdict", action="store_true",
                    help="the verdict/1 record of the frozen protocol (pilot.protocol.require_frozen); --out writes it")
    ap.add_argument("--fewshot", metavar="RECORD", default=None,
                    help="with --verdict: the few-shot results record (results/fewshot/fewshot.json) that decides "
                         "criterion (d)'s few-shot path; without it the path stays PENDING")
    ap.add_argument("--final-table", metavar="VERDICT_JSON", default=None,
                    help="markdown of REP-02's final results table of a saved verdict/1 record")
    ap.add_argument("--go-block", metavar="VERDICT_JSON", default=None,
                    help="markdown of REP-03's mechanical GO block of a saved verdict/1 record")
    a = ap.parse_args(argv)
    modes = (a.run_dir, a.compare, a.table, a.confidence, a.confidence_table, a.verdict or None, a.final_table,
             a.go_block)
    if sum(x is not None for x in modes) != 1:
        ap.error("give exactly one of RUN_DIR, --compare A_DIRS B_DIRS, --table PATH [PATH ...], "
                 "--confidence RUN_DIR, --confidence-table PATH [PATH ...], --verdict, --final-table VERDICT_JSON "
                 "or --go-block VERDICT_JSON")
    if a.postfit is not None and a.confidence is None and a.run_dir is None:
        ap.error("--postfit applies only to RUN_DIR or --confidence")
    if a.fewshot is not None and not a.verdict:
        ap.error("--fewshot applies only to --verdict")
    if a.confidence is not None and a.split is None:
        ap.error("--confidence needs an explicit --split val|test|hotels_probe")
    if a.confidence is not None and a.out is not None and not a.out.endswith(".json"):
        ap.error("--confidence --out must name a .json file (the reliability diagram is written next to it as .svg)")
    text = None                                             # the markdown modes
    if a.table is not None or a.confidence_table is not None:
        reports, paths = [], table_inputs(a.table if a.table is not None else a.confidence_table)
        for p in paths:
            with open(p, encoding="utf-8") as f:
                reports.append(json.load(f))
        text = (render_table(reports) if a.table is not None
                else render_confidence(reports, svg_dir=confidence_svg_dir(paths)))
    elif a.final_table is not None or a.go_block is not None:
        path = a.final_table if a.final_table is not None else a.go_block
        if not os.path.isfile(path):
            raise SystemExit("%s not found; write the verdict with report.py --verdict --out PATH" % path)
        v = _read_json_input(path, "verdict record")
        proto, pin = pilot.protocol.require_frozen()        # 08-REVIEW WR-03: render only the frozen protocol's
        pilot.verdict.check_record(v, proto, pin)            # verdict, every status recomputed from its numbers
        text = pilot.verdict.render_final_table(v) if a.final_table is not None else pilot.verdict.render_go(v)
    if text is not None:
        print(text, end="")
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            write_atomic(a.out, text.encode("utf-8"))
        return text
    split = "test" if a.split is None else a.split
    if a.verdict:
        report = verdict_report(a.manifest, fewshot=a.fewshot)
    elif a.confidence is not None:
        report = confidence_report(a.confidence, split, a.manifest, a.postfit)
    elif a.compare is not None:
        report = compare(_run_dirs(ap, a.compare[0]), _run_dirs(ap, a.compare[1]), split, a.manifest, a.B, a.seed)
    else:
        report = run_report(a.run_dir, split, a.manifest, a.postfit)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        parent = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(parent, exist_ok=True)
        write_atomic(a.out, (text + "\n").encode("utf-8"))
        if a.confidence is not None and _has(report, "calibration"):   # an out-of-domain record has no diagram
            write_atomic(a.out[:-len(".json")] + ".svg", render_svg(json.loads(text)).encode("utf-8"))
    return report


if __name__ == "__main__":
    main()
