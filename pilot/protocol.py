"""Frozen decision protocol (REP-01, ROADMAP Phase 8 criterion 1): build, bytes, pinned reader and gate helpers.

REP-01 requires the decision rule, with a numeric threshold behind every criterion, and the few-shot protocol to be
written down and frozen before any fine-tuned model is scored on test. protocol.json at the repo root is that record
(A17). It is built once by build_protocol from the committed inputs and the code constants of pilot.verdict,
pilot.metrics and pilot.fewshot, written by the freeze commit (plan 08-10) and never rebuilt after it: from then on
the file is the source of truth. Its identity is the sha256 of its exact bytes (protocol_bytes: indent 2,
ensure_ascii False, one trailing newline, UTF-8), pinned in PROTOCOL_SHA256 by the same commit; the pin lives here,
never in the file it hashes. Every consumer (score.py's test and HOTELS gate, report.py --verdict, the few-shot
runner) calls require_frozen, which reads the file once, hashes those bytes, compares them to the pin and parses the
same bytes, so a re-serialised but equal file is refused too. require_frozen reads PROTOCOL_PATH and PROTOCOL_SHA256
at call time, so no consumer captures a stale pin.

gated_split(protocol, entry) names the gated split (test or hotels_probe) of a manifest entry whose sha256 and
uids_sha256 are the protocol's; final_seed_allowed(protocol, kind, dump_run_id) is true for exactly the three final
seeds of a system's committed final manifest.

Decisions taken as 08-RESEARCH recommended, under the user's standing delegation (2026-10-21): A5 (a seed-averaged
system uses its postfit record's theta.shipped at the record T), A14 (final seeds are scored on MPS, fp32,
--workers 0: the device the postfit T was fitted on), A15 (zero-shot Laya on HOTELS is an ungated reference row,
scored on cpu with Ray workers 2 x 4 CPUs like its committed val dump), A17 (protocol.json at the repo root, the pin
here) and A18 (every gating number uses the headline variant; raw is reported).

The decision blocks are copies of code, never retyped: criteria is pilot.verdict.DEFAULT_CRITERIA, verdict its
VERDICT_RULES and statuses, definitions the pilot.metrics and pilot.fewshot rule texts, fewshot the sampler's own
pilot.fewshot.protocol_block(); collapse adds to DEFAULT_COLLAPSE the label prior's floors computed from its committed
dumps (prior_floors: 10/41 on val and 520/2163 on test at its global val θ 0.12); known_at_freeze holds the in-sample
val deltas of the two chosen selection candidates, computed by pilot.verdict.system_deltas, the function the verdict
uses, so what was known at freeze time is disclosed; power_context, assumptions (A1-A19, Open Questions 1-3) and
delegation are this module's literals.

Stdlib only at import time: build_protocol and main import the numpy-using modules (pilot.metrics, pilot.dumps,
pilot.verdict) inside the function, so score.py's gate never pays for them.

usage:
    python -m pilot.protocol --build --out protocol.json    # write the protocol's bytes (refuses an existing --out)
    python -m pilot.protocol --check                         # require_frozen(): print the pinned sha256
    python -m pilot.protocol --render [PATH]                 # the README protocol block (markdown) of PATH with
                                                             # PATH's own sha256 (a preview, no pin check); without
                                                             # PATH, of require_frozen()'s protocol

render_markdown(protocol, sha256) is the README protocol block (plan 08-10 writes it between the README markers at the
freeze and test-enforces it): every displayed value is read from the dict passed in, never from a code constant, every
rule and definition text is shown verbatim, and rounding is for display only (2 decimals for points, 4 for metrics).
"""
import argparse
import copy
import json
import os
import subprocess

from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, sha256_bytes, sha256_file,
                        write_atomic)

SCHEMA = "protocol/1"
PROTOCOL_PATH = os.path.join(REPO_ROOT, "protocol.json")
# the sha256 of protocol.json's exact bytes; None until the freeze commit (plan 08-10) sets it
PROTOCOL_SHA256 = None
TITLE = "Laya vs BERTurk on Turkish banking intents: the frozen decision protocol (REP-01)"
UNITS = "fractions in [0, 1]; points = fraction x 100"
FROZEN = {
    "when": "before any fine-tuned model is scored on test or HOTELS (REP-01, ROADMAP Phase 8 criterion 1)",
    "why": ("every threshold, the seed-averaged bootstrap procedure and the few-shot protocol are fixed before a "
            "fine-tuned test number exists, so nothing can be tuned to the result"),
    "identity": ("the sha256 of this file's exact bytes, pinned in pilot.protocol.PROTOCOL_SHA256 by the freeze "
                 "commit; every consumer calls pilot.protocol.require_frozen and refuses a missing or changed file"),
}
GATED = ("test", "hotels_probe")
SPLIT_ORDER = ("test", "hotels_probe", "val", "selection_train", "final_train")
SPLIT_KEYS = ("role", "sha256", "uids_sha256", "n_messages", "n_no_intent")
KINDS = ("laya", "berturk")
FINAL_SEEDS = [0, 1, 2]
POSTFIT_DIR = "results/postfit"
SCORING = {
    "splits": list(GATED),
    "device": "mps",
    "workers": 0,
    "dtype": "torch.float32",
    "amp_enabled": False,
    "dump_root": "results/dumps",
    "once": ("each final seed is scored once per split into results/dumps/<dump_run_id>/<split>; an existing dump is "
             "never replaced"),
    "refused": ["--force", "--n", "an --out-root other than results/dumps",
                "any checkpoint outside systems.*.final_seeds",
                "a device or worker count other than this block's",
                "a manifest other than the committed data/splits.json"],
    "why": "mps, fp32, in-process: the scorer of the val dumps the postfit T and θ were fitted on (A14)",
}
INPUT_PATHS = (
    "data/splits.json",
    "results/postfit/laya.json",
    "results/postfit/berturk.json",
    "results/postfit/laya-final.json",
    "results/postfit/berturk-final.json",
    "results/postfit/berturk-reference.json",
    "results/dumps/prior/val.json",
    "results/dumps/prior/test.json",
    "results/reports/majority.json",
    "results/reports/prior.json",
    "results/reports/laya-zeroshot-7b928d82.json",
    "results/latency/laya-mps.json",
    "results/latency/laya-cpu.json",
    "results/latency/berturk-mps.json",
    "results/latency/berturk-cpu.json",
)
ZEROSHOT = "laya-zeroshot-7b928d82"
REFERENCES = {
    "majority": {"run_id": "majority", "report": "results/reports/majority.json", "gated": False,
                 "hotels_probe": None},
    "prior": {"run_id": "prior", "report": "results/reports/prior.json", "gated": False, "hotels_probe": None},
    ZEROSHOT: {"run_id": ZEROSHOT, "report": "results/reports/%s.json" % ZEROSHOT, "gated": False,
               "hotels_probe": {"device": "cpu", "workers": 2, "cpus_per_worker": 4, "gated": False,
                                "once": True}},
}
REFERENCES_RULE = ("reference rows are reported, never gated; their θ is D-12 on their own committed val dump; "
                   "majority and prior are not scored on hotels_probe; zero-shot Laya is scored once on hotels_probe "
                   "as an ungated reference OOD row, on the scorer of its committed val dump (cpu, Ray workers 2 x 4 "
                   "CPUs; A15)")
VARIANT = {
    "headline": ("each final seed's shipped T (= its postfit record's T) with the record's theta.shipped (global plus "
                 "per-tag, pilot.metrics.theta_vector); every gating number uses it (A18)"),
    "raw": "T = 1 with the record's theta.raw; reported only, never gating",
    "references": "majority, prior and zero-shot Laya keep D-12 on their own val dump (pilot.metrics.select_theta)",
}
SEED_AVERAGING = {
    "rule": ("per seed s, P_s = sigmoid(logit_s / T_s) with T_s the record T; P = the mean over the 3 seeds per "
             "message and tag (pilot.metrics.seed_average); θ = the record's theta.shipped (global plus per-tag, "
             "theta_vector) applied to P; probabilities are averaged before any threshold, never metrics (EVAL-07, "
             "A5)"),
    "raw": "the same with T_s = 1 and the record's theta.raw; reported only",
    "theta_source": ("the postfit record's θ, chosen on val by one selection seed; a final seed has no clean val "
                     "(final_train holds the val folds), so θ is never re-selected on the final seeds or on test"),
}
BOOTSTRAP_PROCEDURE = ("EVAL-07: a paired message-level bootstrap of Laya - BERTurk over the seed-averaged "
                       "probabilities of each system; one index draw (pilot.metrics.bootstrap_weights(n, B, seed)) is "
                       "shared by both systems and by every metric; each replicate's delta is computed at each "
                       "system's fixed θ; the interval is the percentile interval of the replicate deltas; replicates "
                       "where a delta is undefined are counted and left out")
KNOWN_NOTE = ("in-sample val: T, θ and the epoch were chosen on these 208 messages, by one selection seed per model, "
              "while the test system is a 3-seed average; the thresholds are the REP-01 defaults, written before any "
              "result, and are not tuned to these numbers; both (d) paths (Brier, precision at 80% coverage) favour "
              "BERTurk on val and (a) sits near the margin; recorded so that what was known at freeze time is "
              "disclosed")
POWER_CONTEXT = {
    "label": ("research estimate, normal approximation (A19): 08-RESEARCH CI-width research, from the in-sample val "
              "deltas (B = 10,000, seed 0) scaled to the test size; the 3-seed average makes the SE somewhat smaller"),
    "n": 413,
    "micro_f1_delta_se": 0.0098,
    "p_a_lower_at_least_margin": {"0": 0.53, "-0.005": 0.34, "-0.01": 0.17, "-0.0155": 0.07},
    "p_a_lower_note": ("P(the (a) micro-F1 delta CI lower bound >= -0.02) at a true micro-F1 delta of each key; "
                       "-0.0155 is the in-sample val estimate"),
    "margin_80_power": -0.0274,
    "margin_80_power_note": "the non-inferiority margin with 80% power at a true delta of 0",
    "alternative_go_success": {"0.03": 0.5, "0.04": 0.85, "0.05": 0.98},
    "alternative_go_note": "P(the alternative GO is MET) at a true micro-F1 delta of each key",
    "macro_f1_ci_half_width": 0.035,
    "precision_at_80_ci_half_width": 0.03,
    "precision_at_80_note": "extrapolated by the square-root-of-n rule (A19)",
    "margin": ("the -0.02 margin is locked (REP-01) and kept; (a) is three-way, so a non-resolvable result reads "
               "INCONCLUSIVE, which counts as NO-GO plus collect bank data"),
}
TAKEN = "taken as 08-RESEARCH recommended under the standing delegation"
ASSUMPTIONS = [{"id": i, "decision": d, "taken": TAKEN} for i, d in (
    ("A1", "(d)'s calibration path is Brier at the headline T; classwise ECE is reported only, its percentile "
           "bootstrap being biased"),
    ("A2", "(a) is three-way: PASS when the micro-F1 delta CI lower bound >= -0.02 and the macro-F1 delta point >= "
           "-0.03; FAIL when the micro-F1 CI upper bound < -0.02 or the macro-F1 point < -0.03; INCONCLUSIVE otherwise"),
    ("A3", "the overall verdict follows the verdict rules in order; INCONCLUSIVE counts as NO-GO; with (d) PENDING the "
           "verdict is PROVISIONAL only when the few-shot MET and NOT MET outcomes differ"),
    ("A4", "the alternative GO is a micro-F1 delta > +0.03 with a CI lower bound > 0, and still needs (b) PASS"),
    ("A5", "a seed-averaged final system uses its postfit record's theta.shipped at the record T"),
    ("A6", "the collapse rule applies to each selection run (val floor 10/41 and loss) and to each final seed (loss "
           "and a test floor against the prior's 520/2163, a marked addition)"),
    ("A7", "the held-out few-shot tags are the FEATURES TS-14 list; eligibility checks the list, never selects it"),
    ("A8", "one masked base run per model and seed on selection_train at the chosen recipe; seeds 0-2 pair with "
           "draws 0-2"),
    ("A9", "k is nested (5 within 10), 4k negatives per tag, full-label support rows, the union over tags, and 200 "
           "replay messages outside the k = 10 support set"),
    ("A10", "the Laya few-shot item rule: every positive, every new-tag label and 8 old-tag negatives per message"),
    ("A11", "10 epochs of continued training at the base LRs with each trainer's schedule (epoch parity, not step "
            "parity)"),
    ("A12", "new-tag AP has sklearn semantics, macro over the 6 tags, at T = 1, as the mean over draws; (d) reads "
            "k = 10"),
    ("A13", "the compute fallback (1 base run per model, seed 0) triggers on a projection over 10 h or a base-run "
            "failure after the ladder"),
    ("A14", "final seeds are scored on MPS, fp32, --workers 0: the device the postfit T was fitted on"),
    ("A15", "zero-shot Laya on HOTELS is an ungated reference row (cpu, Ray workers 2 x 4 CPUs); majority and prior "
            "are not scored on HOTELS"),
    ("A16", "the table's standard deviation over seeds is the sample std (ddof 1)"),
    ("A17", "protocol.json sits at the repo root; its sha256 pin is in pilot/protocol.py"),
    ("A18", "every gating number uses the headline variant (record T, theta.shipped); raw (T = 1, theta.raw) is "
            "reported only"),
    ("A19", "power numbers are normal-approximation estimates; the precision at 80% coverage widths are extrapolated "
            "by the square-root-of-n rule"),
    ("OQ1", "the test-floor collapse component is frozen, marked as a Claude-recommended addition to the REP-01 "
            "default"),
    ("OQ2", "zero-shot Laya is scored once on HOTELS as an ungated reference OOD row; majority and prior are not"),
    ("OQ3", "Phase 8 ships fewshot.py --support-sets behind require_frozen; Phase 9 extends the same entry point"),
)]
DELEGATION = {
    "standing": {"given": "06-15 (2026-10-21)",
                 "quote": "do the programatical control. i will not do manual control until all phases done.",
                 "meaning": ("every blocking-human gate is resolved by programmatic checks; the recommended option is "
                             "taken and the delegation recorded")},
    "restated": {"given": "2026-10-23",
                 "quote": "i already told you that i will not made a manual control untill all phases are done"},
    "single_lr": {"given": "2026-10-15",
                  "reply": ("assume it okey for now. defer for further learning rate grid search. we know that at "
                            "least one hyperparameter works."),
                  "choice": "Single LR now (Recommended)",
                  "decision": "one encoder LR 2e-05 for both models, fixed, not tuned (head LR 1e-4)"},
    "bank_name": "the bank-name check is deferred to the end-of-milestone review",
}
REQUIRED_KEYS = ("schema", "title", "units", "frozen", "inputs", "splits", "systems", "references", "scoring",
                 "variant", "seed_averaging", "bootstrap", "criteria", "collapse", "verdict", "definitions", "fewshot",
                 "known_at_freeze", "power_context", "assumptions", "delegation", "code")


# ---- bytes, writer and the pinned reader ----

def protocol_bytes(protocol):
    """The exact bytes of a protocol: indent 2, ensure_ascii False, one trailing newline, UTF-8 (REP-01 encoding)."""
    return (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_protocol(protocol, path):
    """Write protocol_bytes(protocol) to `path` atomically; an existing path is refused (the protocol is frozen
    once). Returns the sha256 of the written bytes."""
    if os.path.lexists(path):
        raise SystemExit("%s exists; the protocol is frozen once and never rewritten (write it to a new path)" % path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw = protocol_bytes(protocol)
    write_atomic(path, raw)
    return sha256_bytes(raw)


def require_frozen(path=None, pin=None):
    """(protocol dict, sha256) of the frozen protocol; SystemExit naming the fix unless the file exists, the pin is
    set, the file's bytes hash to the pin, and those same bytes parse to a protocol/1 object. With no argument it
    reads pilot.protocol.PROTOCOL_PATH and PROTOCOL_SHA256 at call time."""
    path = PROTOCOL_PATH if path is None else path
    pin = PROTOCOL_SHA256 if pin is None else pin
    if not os.path.isfile(path):
        raise SystemExit("%s not found; the decision protocol must be frozen (plan 08-10) before any fine-tuned model "
                         "is scored on test or HOTELS, the verdict is computed or the few-shot runner starts" % path)
    if pin is None:
        raise SystemExit("the decision protocol is not frozen: pilot.protocol.PROTOCOL_SHA256 is unset; the freeze "
                         "commit (plan 08-10) pins the sha256 of %s" % path)
    with open(path, "rb") as f:
        raw = f.read()                                      # hash and parse the same bytes (no re-read window)
    have = sha256_bytes(raw)
    if have != pin:
        raise SystemExit("%s has sha256 %s, the frozen protocol is %s; the protocol may not change after the freeze "
                         "(restore it with `git checkout -- protocol.json`)" % (path, have, pin))
    return parse_protocol(raw, path, "restore it with `git checkout -- protocol.json`"), have


def parse_protocol(raw, path, fix):
    """The protocol/1 dict of `raw` (the bytes of `path`); SystemExit ending in `fix` unless they are UTF-8 JSON of an
    object with schema protocol/1 and every REQUIRED_KEYS key."""
    try:
        protocol = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); %s" % (path, e, fix))
    if not isinstance(protocol, dict):
        raise SystemExit("%s is not a JSON object; %s" % (path, fix))
    if protocol.get("schema") != SCHEMA:
        raise SystemExit("%s: schema %r, expected %s; %s" % (path, protocol.get("schema"), SCHEMA, fix))
    missing = [k for k in REQUIRED_KEYS if k not in protocol]
    if missing:
        raise SystemExit("%s lacks the protocol/1 key(s) %s; %s" % (path, ", ".join(missing), fix))
    return protocol


# ---- the gate helpers (score.py, fewshot.py, report.py) ----

def gated_split(protocol, entry):
    """"test" or "hotels_probe" when the manifest `entry` has both the sha256 and the uids_sha256 of that split in
    the protocol, else None."""
    splits = protocol.get("splits") or {}
    for name in GATED:
        s = splits.get(name) or {}
        if s.get("sha256") and entry.get("sha256") == s["sha256"] and entry.get("uids_sha256") == s.get("uids_sha256"):
            return name
    return None


def final_seeds(protocol, kind):
    """The protocol's final seeds of system `kind`: [{seed, train_run_id, checkpoint_name, dump_run_id}]."""
    system = (protocol.get("systems") or {}).get(kind)
    if not isinstance(system, dict) or not isinstance(system.get("final_seeds"), list):
        raise SystemExit("the protocol has no final seeds for system %r; systems: %s"
                         % (kind, sorted(protocol.get("systems") or {})))
    return system["final_seeds"]


def final_seed_allowed(protocol, kind, dump_run_id):
    """True when `dump_run_id` is one of the final seeds of system `kind` in the protocol (an unknown kind: False)."""
    if kind not in (protocol.get("systems") or {}):
        return False
    return any(s.get("dump_run_id") == dump_run_id for s in final_seeds(protocol, kind))


# ---- the blocks ----

def _read_json(root, rel):
    """(parsed JSON, sha256 of the same bytes) of a committed file under root; SystemExit naming the git fix."""
    p = os.path.join(root, rel)
    try:
        with open(p, "rb") as f:
            raw = f.read()
        return json.loads(raw.decode("utf-8")), sha256_bytes(raw)
    except (OSError, ValueError) as e:
        raise SystemExit("%s cannot be read (%s); it is a committed input of the protocol, restore it with git"
                         % (rel, e))


def _committed(manifest_path):
    if not manifest_is_committed(manifest_path):
        raise SystemExit("%s is not the committed data/splits.json; the protocol is built from the committed splits "
                         "only (drop --manifest, or run `make data`)" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def splits_block(manifest_path=DEFAULT_MANIFEST):
    """The protocol's splits (test, hotels_probe with its ood_scoring_tags, val, selection_train, final_train): path,
    role, sha256, uids_sha256 and counts of the committed manifest; another manifest is refused."""
    m = _committed(manifest_path)
    out = {}
    for name in SPLIT_ORDER:
        e = m["splits"][name]
        out[name] = dict({"path": "data/" + e["path"]}, **{k: e[k] for k in SPLIT_KEYS})
        if name == "hotels_probe":
            out[name]["ood_scoring_tags"] = list(m["ood_scoring_tags"])
    return out


def _theta_summary(theta):
    return {"global": theta["global"], "n_per_tag": len(theta["per_tag"])}


def _system(root, kind):
    rec_rel, final_rel = "%s/%s.json" % (POSTFIT_DIR, kind), "%s/%s-final.json" % (POSTFIT_DIR, kind)
    record, rec_sha = _read_json(root, rec_rel)
    final, final_sha = _read_json(root, final_rel)
    if record.get("system_kind") != kind:
        raise SystemExit("%s is a postfit record of system %r, not %r; restore it with git"
                         % (rec_rel, record.get("system_kind"), kind))
    bound = final.get("postfit") or {}
    if final.get("schema") != "postfit-final/1" or bound.get("sha256") != rec_sha or bound.get("path") != rec_rel:
        raise SystemExit("%s is not bound to %s (postfit %r, the record's sha256 %s); restore both with git"
                         % (final_rel, rec_rel, bound, rec_sha))
    seeds = final.get("seeds") or []
    if [s.get("seed") for s in seeds] != FINAL_SEEDS or record["final"].get("seeds") != FINAL_SEEDS:
        raise SystemExit("%s lists seeds %s, the protocol needs exactly %s in order"
                         % (final_rel, [s.get("seed") for s in seeds], FINAL_SEEDS))
    c = record["chosen"]
    if (c.get("scorer") or {}).get("device") != SCORING["device"]:
        raise SystemExit("%s: the chosen candidate was scored on %r, the protocol scores final seeds on %s (the "
                         "device the postfit T was fitted on, A14)" % (rec_rel, (c.get("scorer") or {}).get("device"),
                                                                       SCORING["device"]))
    fs = [{"seed": s["seed"], "train_run_id": s["train_run_id"], "checkpoint_name": s["checkpoint_name"],
           "dump_run_id": s["dump_run_id"]} for s in seeds]
    ids = [s["dump_run_id"] for s in fs]
    if len(set(ids)) != len(ids):
        raise SystemExit("%s repeats a dump run id: %s" % (final_rel, ids))
    return {"kind": kind,
            "final_manifest": {"path": final_rel, "sha256": final_sha},
            "postfit": {"path": rec_rel, "sha256": rec_sha},
            "chosen": {"dump_run_id": c["dump_run_id"], "epoch": c["epoch"], "epochs_schedule": c["epochs_schedule"],
                       "temperature": c["temperature"],
                       "theta": {"shipped": _theta_summary(c["theta"]["shipped"]),
                                 "raw": _theta_summary(c["theta"]["raw"])},
                       "encoder_lr": c["encoder_lr"], "head_lr": c["canonical"]["head_lr"]},
            "seeds": list(FINAL_SEEDS),
            "final_seeds": fs,
            "dump_run_ids": ids}


def systems_block(root=REPO_ROOT):
    """The protocol's systems (laya, berturk) from the committed final manifests and postfit records: each final
    manifest refused unless bound to its record's sha256 and holding exactly seeds 0, 1, 2."""
    return {kind: _system(root, kind) for kind in KINDS}


def inputs_block(root, systems):
    """{repo-relative path: sha256} of every committed file the verdict reads, plus each chosen candidate's val
    sidecar; a missing input is refused."""
    rels = list(INPUT_PATHS) + ["results/dumps/%s/val.json" % systems[k]["chosen"]["dump_run_id"] for k in KINDS]
    out = {}
    for rel in rels:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise SystemExit("%s not found; it is a committed input of the protocol, restore it with git" % rel)
        out[rel] = sha256_file(p)
    return out


def git_code(root=REPO_ROOT):
    """{"git_head", "dirty"} of the checkout the protocol is built in (dirty: a tracked file differs from HEAD)."""
    def git(*args):
        return subprocess.run(("git",) + args, cwd=root, capture_output=True, text=True, check=True).stdout
    try:
        return {"git_head": git("rev-parse", "HEAD").strip(),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip())}
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit("git is needed to record the code state of the protocol (%s)" % e)


def bootstrap_block():
    """The frozen EVAL-07 bootstrap: B 10000, seed 0, level 0.95, coverage 80 ((d)'s abstention path), messages."""
    from pilot import metrics
    return {"B": 10000, "seed": 0, "level": 0.95, "coverage": 80, "unit": "message", "method": metrics.BOOT_METHOD,
            "procedure": BOOTSTRAP_PROCEDURE, "extra": metrics.BOOT_EXTRA_RULE}


def verdict_block():
    """The verdict rules, the INCONCLUSIVE meaning, the provisional text and every status, copied from pilot.verdict."""
    from pilot import verdict as v
    return {"rules": list(v.VERDICT_RULES), "inconclusive_means": v.INCONCLUSIVE_MEANS,
            "provisional": v.PROVISIONAL_TEXT, "units": v.UNITS,
            "statuses": {"a": list(v.A_STATUSES), "b": list(v.B_STATUSES), "c": [v.REPORTED],
                         "d": list(v.D_STATUSES), "alternative_go": list(v.ALT_STATUSES),
                         "overall": [v.GO, v.NO_GO, v.INCONCLUSIVE]}}


def definitions_block():
    """The metric and rule texts the verdict applies, copied verbatim from the code that applies them."""
    from pilot import fewshot, metrics
    cal = metrics.CALIBRATION_RULES
    return {"brier": cal["brier"], "nll": cal["nll"], "classwise_ece": cal["classwise_ece"], "bins": cal["bins"],
            "confidence": metrics.CONFIDENCE_RULE, "coverage": metrics.COVERAGE_RULE, "aurc": metrics.AURC_RULE,
            "theta_tie": metrics.TIE_RULE, "bootstrap_method": metrics.BOOT_METHOD,
            "bootstrap_extra": metrics.BOOT_EXTRA_RULE, "false_positive": metrics.FP_RULE,
            "average_precision": metrics.AP_RULE, "fewshot_sampler": fewshot.SAMPLER_RULE,
            "fewshot_digest": fewshot.DIGEST_RULE}


def _micro_counts(Y, P, theta):
    from pilot import metrics
    tags = metrics.metrics(Y, P, theta)["per_tag"].values()
    return {k: int(sum(t[k] for t in tags)) for k in ("tp", "fp", "fn")}


def prior_floors(manifest_path=DEFAULT_MANIFEST, root=REPO_ROOT):
    """The label prior's collapse floors (A6) from its committed dumps: its global val θ (select_theta on val at
    T 1), and its micro-F1 at that θ on val and on test as exact fractions 2TP / (2TP + FP + FN) ("a/b") with the
    float values for display. Every key starts with prior_. The verdict recomputes them with this function."""
    from pilot import dumps, metrics, verdict
    run = os.path.join(root, "results", "dumps", "prior")
    data = {}
    for split in ("val", "test"):
        logits, uids, side, entry = dumps.load_dump(run, split, manifest_path)
        if side["temperature"]["shipped"] != 1.0 or side["system"]["kind"] != "prior":
            raise SystemExit("results/dumps/prior/%s is not the label prior at T = 1; restore it with git" % split)
        data[split] = (dumps.read_gold(entry, manifest_path, uids), metrics.probabilities(logits, 1.0))
    theta = metrics.select_theta(*data["val"])["global"]
    out = {"prior_theta_global": theta, "prior_counts": {}}
    for split in ("val", "test"):
        c = _micro_counts(data[split][0], data[split][1], theta)
        f = verdict.micro_fraction(c["tp"], c["fp"], c["fn"])
        out["prior_%s_fraction" % split] = "%d/%d" % (f.numerator, f.denominator)
        out["prior_%s_micro_f1" % split] = float(f)
        out["prior_counts"][split] = c
    out["prior_source"] = ("results/dumps/prior/{val,test}: the label prior at T = 1; θ = its global val θ "
                           "(pilot.metrics.select_theta); micro-F1 = 2TP / (2TP + FP + FN) over all 48 tags")
    return out


def collapse_block(manifest_path=DEFAULT_MANIFEST, root=REPO_ROOT):
    """DEFAULT_COLLAPSE (the rule texts) plus the prior's val and test floors (prior_floors)."""
    from pilot import verdict
    return dict(copy.deepcopy(verdict.DEFAULT_COLLAPSE), **prior_floors(manifest_path, root))


def known_at_freeze_block(manifest_path, root, systems, boot):
    """The in-sample val deltas Laya - BERTurk of the two chosen selection candidates, computed by
    pilot.verdict.system_deltas (the verdict's own function): headline at each record's T and theta.shipped, raw at
    T 1 and theta.raw. Each val dump must be the one its record chose (dump_sha256)."""
    from pilot import dumps, metrics, verdict
    got = {}
    for kind in KINDS:
        record, sha = _read_json(root, systems[kind]["postfit"]["path"])
        if sha != systems[kind]["postfit"]["sha256"]:
            raise SystemExit("%s changed while the protocol was built; rebuild it" % systems[kind]["postfit"]["path"])
        c = record["chosen"]
        logits, uids, side, entry = dumps.load_dump(os.path.join(root, "results", "dumps", c["dump_run_id"]), "val",
                                                    manifest_path)
        if side.get("dump_sha256") != c["dump_sha256"]:
            raise SystemExit("results/dumps/%s/val is not the dump %s chose (dump_sha256 %s != %s); restore it with "
                             "git" % (c["dump_run_id"], systems[kind]["postfit"]["path"], side.get("dump_sha256"),
                                      c["dump_sha256"]))
        got[kind] = (logits, uids, entry, c)
    if got["laya"][1] != got["berturk"][1]:
        raise SystemExit("the chosen candidates' val dumps hold different uids; both must score the full val split")
    Y = dumps.read_gold(got["laya"][2], manifest_path, got["laya"][1])
    out = {"split": "val", "in_sample": True, "n": int(Y.shape[0]),
           "laya": got["laya"][3]["dump_run_id"], "berturk": got["berturk"][3]["dump_run_id"]}
    for variant, theta_key in (("headline", "shipped"), ("raw", "raw")):
        T = {k: (got[k][3]["temperature"] if variant == "headline" else 1.0) for k in KINDS}
        P = {k: metrics.probabilities(got[k][0], T[k]) for k in KINDS}
        th = {k: metrics.theta_vector(got[k][3]["theta"][theta_key]) for k in KINDS}
        d = verdict.system_deltas(Y, P["laya"], th["laya"], P["berturk"], th["berturk"], boot)
        out[variant] = dict(d, temperature=T, theta={k: "theta.%s" % theta_key for k in KINDS})
    out["note"] = KNOWN_NOTE
    return out


def build_protocol(manifest_path=DEFAULT_MANIFEST, root=REPO_ROOT, code=None):
    """The complete protocol/1 dict (REQUIRED_KEYS, in order) from the committed inputs and the code constants of
    pilot.verdict, pilot.metrics and pilot.fewshot; deterministic for a given `code` (else git_code(root))."""
    from pilot import fewshot, verdict
    splits = splits_block(manifest_path)
    systems = systems_block(root)
    boot = bootstrap_block()
    p = {
        "schema": SCHEMA,
        "title": TITLE,
        "units": UNITS,
        "frozen": dict(FROZEN),
        "inputs": inputs_block(root, systems),
        "splits": splits,
        "systems": systems,
        "references": {"rule": REFERENCES_RULE, "rows": copy.deepcopy(REFERENCES)},
        "scoring": copy.deepcopy(SCORING),
        "variant": dict(VARIANT),
        "seed_averaging": dict(SEED_AVERAGING),
        "bootstrap": boot,
        "criteria": copy.deepcopy(verdict.DEFAULT_CRITERIA),
        "collapse": collapse_block(manifest_path, root),
        "verdict": verdict_block(),
        "definitions": definitions_block(),
        "fewshot": fewshot.protocol_block(manifest_path),
        "known_at_freeze": known_at_freeze_block(manifest_path, root, systems, boot),
        "power_context": copy.deepcopy(POWER_CONTEXT),
        "assumptions": copy.deepcopy(ASSUMPTIONS),
        "delegation": copy.deepcopy(DELEGATION),
        "code": dict(code) if code is not None else git_code(root),
    }
    if tuple(p) != REQUIRED_KEYS:
        raise SystemExit("build_protocol wrote keys %s, REQUIRED_KEYS is %s" % (list(p), list(REQUIRED_KEYS)))
    return json.loads(json.dumps(p))                        # plain JSON values only (no numpy scalar survives)


# ---- the README protocol block ----

def _num(x):
    """A frozen number as stored (its JSON form: -0.02, 0.0, 2e-05, 80)."""
    return json.dumps(x)


def _value(x, points=False):
    """Display form of a frozen value: a float fraction with its points when `points` (e.g. "-0.02 (-2.00 points)"),
    other numbers and null/booleans in JSON form, lists joined by ", ", dicts as "key: value" joined by "; ", strings
    verbatim."""
    if x is None or isinstance(x, bool):
        return json.dumps(x)
    if isinstance(x, float) and points:
        return "%s (%+.2f points)" % (_num(x), x * 100)
    if isinstance(x, (int, float)):
        return _num(x)
    if isinstance(x, list):
        return ", ".join(_value(v, points) for v in x)
    if isinstance(x, dict):
        return "; ".join("%s: %s" % (k, _value(v, points)) for k, v in x.items())
    return str(x)


def _table(header, rows):
    """A markdown table (the pilot.postfit layout); cells hold names, numbers and hashes only, never rule texts."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def _label(key, name=None):
    """A criterion or status key as shown: "a" -> "(a)" (with its name: "(a) non-inferiority"); a longer key is its
    name when given ("alternative GO"), else the key."""
    if len(key) == 1:
        return "(%s)" % key if name is None else "(%s) %s" % (key, name)
    return key if name is None else name


def _criteria_lines(criteria, units):
    rows, rules, extra = [], [], []
    for key, c in criteria.items():
        label, head = _label(key), _label(key, c["name"])
        rules.append("- %s: %s" % (head, c["rule"]))
        rest = [(k, v) for k, v in c.items() if k not in ("name", "rule", "paths", "reported_only")]
        if rest or not c.get("paths"):
            rows.append((label, c["name"], _value(dict(rest), points=True) or "none"))
        for p in c.get("paths") or []:
            rows.append(("%s %s path" % (label, p["path"]), c["name"],
                         _value({k: v for k, v in p.items() if k != "path"}, points=True)))
        for k, v in (c.get("reported_only") or {}).items():
            extra.append("- %s reported only, %s: %s" % (label, k, v))
    return (["#### Criteria", "", "Thresholds as frozen (units: %s)." % units, ""]
            + _table(("criterion", "name", "frozen values"), rows)
            + ["", "Rules, as frozen:", ""] + rules + extra)


def _collapse_lines(c):
    out = ["#### Collapse rule (b)", "", "- Rule: %s" % c["rule"], "- Applies to:"]
    for a in c["applies_to"]:
        out.append("  - %s (%s): %s" % (a["runs"], a["source"], ", ".join(a["components"])))
    out.append("- Components:")
    for name, comp in c["components"].items():
        out.append("  - %s: %s" % (name, comp["rule"]))
        for k, v in comp.items():
            if k != "rule":
                out.append("    - %s: %s" % ("**marked addition**" if k == "addition" else k, v))
    out += ["- Not applied: %s" % c["not_applied"],
            "- Label prior floors (%s), at the prior's global val θ %s:" % (c["prior_source"],
                                                                          _num(c["prior_theta_global"]))]
    for split, n in c["prior_counts"].items():
        out.append("  - %s: micro-F1 %s = %.4f (TP %d, FP %d, FN %d)"
                   % (split, c["prior_%s_fraction" % split], c["prior_%s_micro_f1" % split], n["tp"], n["fp"], n["fn"]))
    return out


def _verdict_lines(v):
    out = ["#### Verdict", "", "Units: %s." % v["units"], ""]
    out += ["%d. %s" % (i, r) for i, r in enumerate(v["rules"], 1)]
    out += ["", "- INCONCLUSIVE means: %s." % v["inconclusive_means"],
            "- Provisional form: `%s` (each %%s is the overall verdict computed with that few-shot outcome)."
            % v["provisional"],
            "- Statuses: %s." % "; ".join("%s %s" % (_label(k), ", ".join(s)) for k, s in v["statuses"].items())]
    return out


def render_markdown(protocol, sha256):
    """The README protocol block (markdown, level-3 and level-4 headings; 08-10 writes the section heading): the
    protocol's sha256 and frozen-before sentence, the criteria with their thresholds in points, the collapse rule
    with the label prior's floors, the verdict rules. Every value is read from `protocol`, never from a code constant;
    rule texts are verbatim; rounding is display only. Deterministic: the same dict and sha256 give the same text."""
    p = protocol
    fz, code = p["frozen"], p["code"]
    out = ["### Frozen decision protocol (protocol.json)", "", p["title"], "",
           "- sha256: `%s`" % sha256,
           "- Frozen %s: %s." % (fz["when"], fz["why"]),
           "- Identity: %s." % fz["identity"],
           "- Units: %s." % p["units"],
           "- Built at git HEAD `%s` (dirty: %s)." % (code["git_head"], _value(code["dirty"]))]
    for block in (_criteria_lines(p["criteria"], p["units"]), _collapse_lines(p["collapse"]),
                  _verdict_lines(p["verdict"])):
        out += [""] + block
    return "\n".join(out) + "\n"


# ---- CLI ----

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="build the protocol from the committed inputs to --out")
    mode.add_argument("--check", action="store_true", help="require_frozen(): print the pinned sha256")
    mode.add_argument("--render", nargs="?", const="", metavar="PATH",
                      help="print the README protocol block of PATH with PATH's own sha256 (a preview, no pin check); "
                           "without PATH, of require_frozen()'s protocol")
    ap.add_argument("--out", help="--build: the path to write (refused when it exists)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="--build: must be the committed data/splits.json")
    a = ap.parse_args(argv)
    if a.render is not None:
        if a.out:
            ap.error("--out is a --build option")
        if a.render:
            try:
                with open(a.render, "rb") as f:
                    raw = f.read()                          # hash and parse the same bytes
            except OSError as e:
                raise SystemExit("%s cannot be read (%s); --render PATH needs a protocol file" % (a.render, e))
            sha = sha256_bytes(raw)
            protocol = parse_protocol(raw, a.render, "--render PATH needs a protocol/1 file (pilot.protocol --build)")
        else:
            protocol, sha = require_frozen()
        md = render_markdown(protocol, sha)
        print(md, end="")
        return md
    if a.build:
        if not a.out:
            ap.error("--build needs --out PATH")
        if os.path.lexists(a.out):
            raise SystemExit("%s exists; the protocol is frozen once and never rewritten (write it to a new path)"
                             % a.out)
        sha = write_protocol(build_protocol(a.manifest), a.out)
        print("protocol -> %s sha256 %s" % (a.out, sha))
        return sha
    if a.out:
        ap.error("--out is a --build option")
    _, sha = require_frozen()
    print("protocol.json is frozen: sha256 %s" % sha)
    return sha


if __name__ == "__main__":
    main()
