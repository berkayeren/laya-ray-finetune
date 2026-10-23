"""The mechanical GO block of REP-03 under the decision rule frozen in protocol.json (REP-01): per-criterion statuses
and the overall verdict, computed from numbers and thresholds passed in, never from a judgement.

Thresholds always come from the caller's criteria dict: in production the frozen protocol's `criteria` block;
DEFAULT_CRITERIA are the REP-01 defaults that 08-05 writes into protocol.json (and a test there keeps the two equal).
No criterion function holds a threshold literal. Every comparison uses the operator its key names (`_min` and
`_at_least` >=, `_above` >, `_below` <) on the unrounded float64 values; nothing is rounded. Units are fractions in
[0, 1]; the rule texts render points = fraction x 100. Deltas are Laya - BERTurk.

Decisions taken as 08-RESEARCH recommended, under the user's standing delegation (2026-10-21):
- A1: (d)'s calibration path is Brier (strictly proper, additive over messages, so its paired percentile bootstrap is
  well posed); classwise ECE is reported only, its percentile bootstrap being biased (DEFAULT_CRITERIA d reported_only).
- A2: (a) is three-way: PASS when the micro-F1 delta CI lower bound >= micro_ci_lower_min and the macro-F1 delta point
  >= macro_point_min; FAIL when the micro-F1 CI upper bound < micro_fail_ci_upper_below or the macro-F1 point <
  macro_point_min; INCONCLUSIVE otherwise. The macro-F1 CI is carried, labelled macro_ci_label, never gated.
- A3: overall follows VERDICT_RULES in order. INCONCLUSIVE counts as NO-GO (INCONCLUSIVE_MEANS) and is never mapped to
  GO. With (a) PASS and (d) PENDING (the few-shot path waits for Phase 9) decide computes the verdict with the few-shot
  path MET and NOT MET; it is PROVISIONAL (PROVISIONAL_TEXT, both outcomes named) only when the two differ.
- A4: the alternative GO is a micro-F1 delta strictly above delta_above with a CI lower bound strictly above
  ci_lower_above, and still needs (b) PASS ((b) FAIL is rule 1, before it).
- A6: the collapse rule of (b) (DEFAULT_COLLAPSE) applies to each model's selection run (its postfit record's chosen
  candidate: val floor and loss) and to each final seed (loss and a test floor; the val floor is not applied to final
  seeds, whose final_train holds the val folds). The floors compare exact Fractions 2TP / (2TP + FP + FN)
  (micro_fraction, parse_fraction), never floats, and only strictly below the floor is collapsed; a loss list with a
  NaN or infinite value, or whose last epoch is strictly above its first, is collapsed. The test floor is a
  Claude-recommended addition to the REP-01 default (08-RESEARCH Open Question 1), frozen before any fine-tuned test
  number. (b) is PASS only with no collapsed row and exactly seeds_per_model distinct seeds per model; (c) is always
  REPORTED (not gated).
Inputs are validated before any comparison: a missing or non-finite delta, a missing, non-finite or misordered CI and
an unknown status raise SystemExit naming the input, so an empty result is refused, never scored.

The verdict record (08-09): build_verdict(protocol, sha, inputs) turns report.verdict_from's protocol-bound inputs into
verdict/1 (VERDICT_KEYS): the per-seed rows, settings and summary of each fine-tuned system (mean_std: the mean and the
sample std, ddof 1, A16, exact for equal values), the reference rows, the seed-averaged deltas (headline gates, raw is
reported, A18), the collapse rows (A6: selection laya, selection berturk, then each model's final seeds), the latency
rows, the few-shot path, every criterion from the protocol's criteria block and the overall verdict by decide. It
refuses a protocol whose verdict rules are not this module's, holds no timestamp and rounds nothing.
The few-shot path (09-08, D-13): without a few-shot result the fewshot block is {status PENDING, phase 9} (the Phase 8
verdict, byte for byte); with inputs["fewshot"] (report.fewshot_input: the few-shot results record's primary numbers,
recomputed from the dumps) criterion_d reads its {mean_delta, draws} at the frozen mean_delta_at_least (_fewshot,
unchanged), so the path is MET or NOT MET, (d) PASS or FAIL and decide names no few-shot branch; the fewshot block is
then FEWSHOT_KEYS: the path's status, phase 9, k (criteria (d)'s fewshot k), mean_delta, draws, the record {path,
sha256}, the mode (primary or fallback) and the frozen compute-fallback disclosure under the fallback (None under
primary). check_record accepts either form and recomputes (d) from the block's own numbers, so an edited few-shot status,
mean, draw, k, phase or key is refused; render_go adds the few-shot line (and the disclosure line) for the record's form
only, so the PENDING render is unchanged.
render_final_table(v) is REP-02's one results table (references in REFERENCE_ORDER, each fine-tuned model as
mean ± sample std with its per-seed values in seed order, then its seed-averaged EVAL-07 system, and the settings)
and render_go(v) REP-03's GO block (every criterion's status and numbers, the verdict and the frozen rule lines that
fired, INCONCLUSIVE_MEANS whenever INCONCLUSIVE appears); both read only the record and the frozen texts, refuse
another schema or a dropped seed, and round for display only.

numpy and stdlib only: no torch, scipy, Ray, laya or sklearn (report.py imports this module).

usage: from pilot import verdict
       a = verdict.criterion_a({"delta": d, "ci": [lo, hi]}, {"delta": d, "ci": [lo, hi]}, criteria["a"])
       alt = verdict.alternative_go({"delta": d, "ci": [lo, hi]}, criteria["alternative_go"])
       d = verdict.criterion_d(brier, precision_at_80, None, criteria["d"])     # None: the few-shot path is pending
       verdict.decide(a["status"], b["status"], d["status"], alt["status"])   # {"verdict", "provisional", ...}
       rows = [verdict.collapse_val(verdict.micro_fraction(tp, fp, fn), verdict.parse_fraction("10/41")),
               verdict.collapse_loss(loss_by_epoch), verdict.collapse_test_floor(value, floor)]
       b = verdict.criterion_b(rows, {"laya": [0, 1, 2], "berturk": [0, 1, 2]}, criteria["b"])
       c = verdict.criterion_c(latency_rows, criteria["c"])
       deltas = verdict.system_deltas(Y, P_laya, th_laya, P_berturk, th_berturk, protocol["bootstrap"])
       mean, std = verdict.mean_std([x0, x1, x2])                            # sample std, ddof 1
       v = verdict.build_verdict(protocol, sha, inputs)                      # verdict/1 (report.verdict_from)
       verdict.render_final_table(v), verdict.render_go(v)                   # markdown (report.py --final-table, --go-block)
"""
import json
import math
import numbers
import re
import statistics
from fractions import Fraction

import numpy as np

from pilot import metrics

PASS, FAIL, INCONCLUSIVE, PENDING = "PASS", "FAIL", "INCONCLUSIVE", "PENDING"
MET, NOT_MET = "MET", "NOT MET"
REPORTED = "REPORTED (not gated)"
GO, NO_GO = "GO", "NO-GO"
INCONCLUSIVE_MEANS = "counts as NO-GO; collect bank data"
PROVISIONAL_TEXT = "PROVISIONAL: %s if the few-shot path of (d) is met, else %s"
A_STATUSES = (PASS, FAIL, INCONCLUSIVE)
B_STATUSES = (PASS, FAIL)
D_STATUSES = (PASS, FAIL, PENDING)
ALT_STATUSES = (MET, NOT_MET)
# A3: the overall verdict, first matching line wins; decide records the 1-based index of the line that fired
VERDICT_RULES = (
    "(b) FAIL -> NO-GO",
    "alternative GO MET -> GO",
    "(a) FAIL -> NO-GO",
    "(a) INCONCLUSIVE -> INCONCLUSIVE (" + INCONCLUSIVE_MEANS + ")",
    "(a) PASS and (d) PASS -> GO",
    "(a) PASS and (d) FAIL -> NO-GO",
    "(a) PASS and (d) PENDING -> PROVISIONAL: the verdict is computed with the few-shot path of (d) MET and NOT MET; "
    "it is final when both agree",
)
UNITS = "deltas are Laya - BERTurk fractions in [0, 1] (points = fraction x 100), compared unrounded"

DEFAULT_CRITERIA = {
    "a": {
        "name": "non-inferiority",
        "rule": ("PASS when the micro-F1 delta 95% CI lower bound >= micro_ci_lower_min and the macro-F1 delta point "
                 "estimate >= macro_point_min; FAIL when the micro-F1 delta CI upper bound < "
                 "micro_fail_ci_upper_below or the macro-F1 delta point estimate < macro_point_min; INCONCLUSIVE "
                 "otherwise. The macro-F1 CI is reported with macro_ci_label, not gated; " + UNITS),
        "micro_ci_lower_min": -0.02,
        "micro_fail_ci_upper_below": -0.02,
        "macro_point_min": -0.03,
        "macro_ci_label": "not resolvable at this test size (n = 413)",
    },
    "b": {
        "name": "seeds and no collapsed run",
        "rule": ("PASS when no run the collapse rule applies to is collapsed and every model has exactly "
                 "seeds_per_model distinct scored final seeds; FAIL otherwise"),
        "seeds_per_model": 3,
    },
    "c": {
        "name": "latency",
        "rule": "per-message latency of every final seed is reported, not gated, in v1 (a budget is DEC-01, v2)",
        "label": "reported, not gated",
        "status": REPORTED,
    },
    "d": {
        "name": "a clear edge on calibration, abstention or few-shot learning",
        "rule": ("PASS when any one path is MET; PENDING when none is MET and the few-shot path is pending; FAIL "
                 "otherwise. calibration: the Brier delta < delta_below and its CI upper bound < ci_upper_below; "
                 "abstention: the precision at coverage % delta >= delta_at_least and its CI lower bound > "
                 "ci_lower_above; fewshot: the mean over draws of the new-tag macro-AP delta at k >= "
                 "mean_delta_at_least (PENDING until Phase 9); " + UNITS),
        "paths": [
            {"path": "calibration", "metric": "brier", "delta_below": 0.0, "ci_upper_below": 0.0},
            {"path": "abstention", "metric": "precision_at_80", "coverage": 80, "delta_at_least": 0.03,
             "ci_lower_above": 0.0},
            {"path": "fewshot", "metric": "new_tag_macro_ap", "k": 10, "mean_delta_at_least": 0.05,
             "until_phase_9": PENDING},
        ],
        "reported_only": {
            "classwise_ece": ("reported, not gated (A1): its percentile bootstrap is biased (binned ECE replicates "
                              "are inflated, so the observed delta sat at the 93.8th percentile of its own replicates "
                              "on val; Roelofs et al., AISTATS 2022) and its direction flips between T = 1 and the "
                              "fitted T; Brier names the calibration path"),
        },
    },
    "alternative_go": {
        "name": "alternative GO",
        "rule": ("MET when the micro-F1 delta > delta_above and its 95% CI lower bound > ci_lower_above; it gives GO "
                 "only with (b) PASS; " + UNITS),
        "metric": "micro_f1",
        "delta_above": 0.03,
        "ci_lower_above": 0.0,
        "requires": "b PASS",
    },
}


# A6: rule texts only; 08-05 adds the data-derived prior fractions (val and test) when it builds protocol.json
DEFAULT_COLLAPSE = {
    "rule": "a run is collapsed when any component applied to it is collapsed; (b) needs no collapsed run",
    "applies_to": [
        {"runs": "the selection run of each model", "source": "its postfit record's chosen candidate",
         "components": ["val_floor", "loss"]},
        {"runs": "each final seed", "source": "its final-manifest entry and its test dump",
         "components": ["loss", "test_floor"]},
    ],
    "components": {
        "val_floor": {
            "rule": ("collapsed when the chosen candidate's val micro-F1 at its global val θ after T, the exact "
                     "fraction 2TP / (2TP + FP + FN) of the record's objective counts, is strictly below the label "
                     "prior's val micro-F1 at the prior's own global val θ (prior_val_fraction); equal is not "
                     "collapsed"),
        },
        "loss": {
            "rule": ("collapsed when any epoch's mean training loss is NaN or infinite, or the last epoch's is "
                     "strictly above the first epoch's (divergent); equal is not collapsed"),
            "note": ("the per-epoch loss is the unweighted mean over micro-batches (Laya) or steps (BERTurk); Laya's "
                     "RLCD sigma anneal makes its epochs not strictly comparable, but the rule still catches a "
                     "blow-up"),
        },
        "test_floor": {
            "rule": ("collapsed when a final seed's test micro-F1 at its record's global shipped θ, the exact "
                     "fraction 2TP / (2TP + FP + FN), is strictly below the label prior's test micro-F1 at the "
                     "prior's global val θ (prior_test_fraction); equal is not collapsed"),
            "addition": ("Claude-recommended addition to the REP-01 default, frozen before any fine-tuned test number "
                         "(08-RESEARCH Open Question 1)"),
        },
    },
    "not_applied": "the val floor is not applied to final seeds: final_train holds the val folds",
}
FRACTION_RE = re.compile(r"^(0|[1-9][0-9]*)/([1-9][0-9]*)$")


def _status(value, allowed, what):
    if not isinstance(value, str) or value not in allowed:
        raise SystemExit("unknown %s status %r; expected one of: %s" % (what, value, ", ".join(allowed)))
    return value


def _finite(x, what):
    """x as a float64 value, refused unless it is a finite real number (never None, a bool or a string)."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x):
        raise SystemExit("%s must be a finite number, got %r; it is refused, never scored" % (what, x))
    return float(x)


def _delta_ci(s, what):
    """(delta, ci lower, ci upper) of a {"delta", "ci": [lower, upper]} dict, every value finite and lower <= upper."""
    if not isinstance(s, dict) or "delta" not in s or "ci" not in s:
        raise SystemExit("%s needs a {\"delta\", \"ci\"} result, got %r; it is refused, never scored" % (what, s))
    ci = s["ci"]
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        raise SystemExit("%s needs a two-sided CI [lower, upper], got %r" % (what, ci))
    d = _finite(s["delta"], what + " delta")
    lo, hi = _finite(ci[0], what + " CI lower bound"), _finite(ci[1], what + " CI upper bound")
    if lo > hi:
        raise SystemExit("%s CI is misordered: lower %r > upper %r" % (what, lo, hi))
    return d, lo, hi


def _threshold(rule, key, what):
    if not isinstance(rule, dict) or key not in rule:
        raise SystemExit("%s rule has no %r threshold; pass the frozen protocol's criteria block" % (what, key))
    return _finite(rule[key], "%s threshold %s" % (what, key))


def _path_rule(rule, name):
    paths = [p for p in rule.get("paths", ()) if isinstance(p, dict) and p.get("path") == name] \
        if isinstance(rule, dict) else []
    if len(paths) != 1:
        raise SystemExit("criterion (d) rule needs exactly one %r path, found %d" % (name, len(paths)))
    return paths[0]


def criterion_a(micro, macro, rule):
    """(a) non-inferiority, three-way (A2): {"status", "micro", "macro", "macro_ci_label", "rule"}."""
    lo_min = _threshold(rule, "micro_ci_lower_min", "(a)")
    fail_below = _threshold(rule, "micro_fail_ci_upper_below", "(a)")
    macro_min = _threshold(rule, "macro_point_min", "(a)")
    if fail_below > lo_min:
        raise SystemExit("(a) rule would let PASS and FAIL overlap: micro_fail_ci_upper_below %r > "
                         "micro_ci_lower_min %r" % (fail_below, lo_min))
    md, mlo, mhi = _delta_ci(micro, "(a) micro-F1")
    Md, Mlo, Mhi = _delta_ci(macro, "(a) macro-F1")
    if mhi < fail_below or Md < macro_min:
        status = FAIL
    elif mlo >= lo_min and Md >= macro_min:
        status = PASS
    else:
        status = INCONCLUSIVE
    return {"status": status, "micro": {"delta": md, "ci": [mlo, mhi]}, "macro": {"delta": Md, "ci": [Mlo, Mhi]},
            "macro_ci_label": rule.get("macro_ci_label"), "rule": rule.get("rule")}


def alternative_go(micro, rule):
    """The alternative GO path (A4): {"status": MET or NOT_MET, "micro", "requires", "rule"}."""
    above = _threshold(rule, "delta_above", "alternative GO")
    lo_above = _threshold(rule, "ci_lower_above", "alternative GO")
    md, mlo, mhi = _delta_ci(micro, "alternative GO micro-F1")
    status = MET if md > above and mlo > lo_above else NOT_MET
    return {"status": status, "micro": {"delta": md, "ci": [mlo, mhi]}, "requires": rule.get("requires"),
            "rule": rule.get("rule")}


def _fewshot(fewshot, p):
    at_least = _threshold(p, "mean_delta_at_least", "(d) fewshot")
    if fewshot is None:
        return {"path": "fewshot", "metric": p.get("metric"), "status": PENDING, "mean_delta": None, "draws": None}
    if not isinstance(fewshot, dict) or "mean_delta" not in fewshot or "draws" not in fewshot:
        raise SystemExit("(d) fewshot needs None (pending) or a {\"mean_delta\", \"draws\"} result, got %r"
                         % (fewshot,))
    draws = fewshot["draws"]
    if not isinstance(draws, (list, tuple)) or not draws:
        raise SystemExit("(d) fewshot needs a non-empty list of per-draw deltas, got %r" % (draws,))
    draws = [_finite(x, "(d) fewshot draw delta") for x in draws]
    mean = _finite(fewshot["mean_delta"], "(d) fewshot mean_delta")
    if mean != float(np.mean(np.asarray(draws, dtype=np.float64))):
        raise SystemExit("(d) fewshot mean_delta %r is not the mean of its draws %r" % (mean, draws))
    status = MET if mean >= at_least else NOT_MET
    return {"path": "fewshot", "metric": p.get("metric"), "status": status, "mean_delta": mean, "draws": draws}


def criterion_d(calibration, abstention, fewshot, rule):
    """(d) any one path (A1): calibration and abstention are {"delta", "ci"} results, fewshot is None (pending) or
    {"mean_delta", "draws"}. Returns {"status", "paths": [calibration, abstention, fewshot], "reported_only",
    "rule"}."""
    pc, pa, pf = _path_rule(rule, "calibration"), _path_rule(rule, "abstention"), _path_rule(rule, "fewshot")
    c_below = _threshold(pc, "delta_below", "(d) calibration")
    c_hi_below = _threshold(pc, "ci_upper_below", "(d) calibration")
    a_least = _threshold(pa, "delta_at_least", "(d) abstention")
    a_lo_above = _threshold(pa, "ci_lower_above", "(d) abstention")
    cd, clo, chi = _delta_ci(calibration, "(d) calibration")
    ad, alo, ahi = _delta_ci(abstention, "(d) abstention")
    paths = [
        {"path": "calibration", "metric": pc.get("metric"), "status": MET if cd < c_below and chi < c_hi_below
         else NOT_MET, "delta": cd, "ci": [clo, chi]},
        {"path": "abstention", "metric": pa.get("metric"), "status": MET if ad >= a_least and alo > a_lo_above
         else NOT_MET, "delta": ad, "ci": [alo, ahi]},
        _fewshot(fewshot, pf),
    ]
    if any(p["status"] == MET for p in paths):
        status = PASS
    elif paths[-1]["status"] == PENDING:
        status = PENDING
    else:
        status = FAIL
    return {"status": status, "paths": paths, "reported_only": rule.get("reported_only"), "rule": rule.get("rule")}


def _fire(a, b, d, alt):
    """(verdict, 1-based VERDICT_RULES index), or (None, None) for (a) PASS with (d) PENDING."""
    _status(a, A_STATUSES, "(a)")
    _status(b, B_STATUSES, "(b)")
    _status(d, D_STATUSES, "(d)")
    _status(alt, ALT_STATUSES, "alternative GO")
    if b == FAIL:
        return NO_GO, 1
    if alt == MET:
        return GO, 2
    if a == FAIL:
        return NO_GO, 3
    if a == INCONCLUSIVE:
        return INCONCLUSIVE, 4
    if d == PASS:
        return GO, 5
    if d == FAIL:
        return NO_GO, 6
    return None, None


def overall(a, b, d, alt):
    """The overall verdict by VERDICT_RULES (A3): GO, NO_GO, INCONCLUSIVE, or None when it depends on the pending
    few-shot path ((a) PASS with (d) PENDING)."""
    return _fire(a, b, d, alt)[0]


def decide(a, b, d, alt):
    """The GO block's verdict: {"verdict", "provisional", "if_fewshot_met", "if_fewshot_not_met", "rule", "means",
    "statuses"}. With (d) PENDING both branches are computed ((d) PASS: the few-shot path MET; (d) FAIL: NOT MET, the
    other two paths being NOT MET whenever (d) is PENDING); the verdict is PROVISIONAL_TEXT naming both when they
    differ, and rule then lists both branches' rule lines. INCONCLUSIVE carries INCONCLUSIVE_MEANS."""
    verdict, rule = _fire(a, b, d, alt)
    met = not_met = None
    provisional = False
    if d == PENDING:
        met, r_met = _fire(a, b, PASS, alt)
        not_met, r_not = _fire(a, b, FAIL, alt)
        if met != not_met:
            provisional = True
            verdict, rule = PROVISIONAL_TEXT % (met, not_met), [r_met, r_not]
        else:
            verdict, rule = met, r_met
    return {"verdict": verdict, "provisional": provisional, "if_fewshot_met": met, "if_fewshot_not_met": not_met,
            "rule": rule, "means": INCONCLUSIVE_MEANS if verdict == INCONCLUSIVE else None,
            "statuses": {"a": a, "b": b, "d": d, "alternative_go": alt}}


def _count(x, what):
    if isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 0:
        raise SystemExit("%s must be a non-negative integer count, got %r" % (what, x))
    return int(x)


def micro_fraction(tp, fp, fn):
    """Micro-F1 as the exact Fraction 2TP / (2TP + FP + FN) (0 for 0 / 0), the postfit objective's form."""
    tp, fp, fn = _count(tp, "tp"), _count(fp, "fp"), _count(fn, "fn")
    den = 2 * tp + fp + fn
    return Fraction(2 * tp, den) if den else Fraction(0)


def parse_fraction(text):
    """The Fraction of an "a/b" text (non-negative integers, b > 0, no sign, space or leading zero); anything else is
    refused, so a float never stands in for a floor."""
    m = FRACTION_RE.fullmatch(text) if isinstance(text, str) else None
    if m is None:
        raise SystemExit("expected a fraction \"a/b\" of non-negative integers with b > 0, got %r" % (text,))
    return Fraction(int(m.group(1)), int(m.group(2)))


def _fraction(x, what):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        return parse_fraction(x)
    raise SystemExit("%s must be an exact Fraction or \"a/b\" text, never %r (floors compare exactly)" % (what, x))


def _text(f):
    return "%d/%d" % (f.numerator, f.denominator)


def _floor(component, value, floor):
    value, floor = _fraction(value, component + " value"), _fraction(floor, component + " floor")
    return {"component": component, "value": _text(value), "floor": _text(floor), "collapsed": value < floor}


def collapse_val(value, floor):
    """The val floor of a selection run: collapsed when value < floor, both exact Fractions (equal is not)."""
    return _floor("val_floor", value, floor)


def collapse_test_floor(value, floor):
    """The test floor of a final seed (the marked addition): collapsed when value < floor, exactly as the val floor."""
    return _floor("test_floor", value, floor)


def collapse_loss(losses):
    """The loss component: collapsed when any epoch loss is NaN or infinite, or the last is strictly above the first.
    first and last are None when not finite, so the row stays strict JSON."""
    if not isinstance(losses, list) or not losses:
        raise SystemExit("the loss component needs a non-empty list of per-epoch losses, got %r; it is refused, "
                         "never scored" % (losses,))
    for x in losses:
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise SystemExit("a per-epoch loss must be a number, got %r" % (x,))
    vals = [float(x) for x in losses]
    non_finite = not all(math.isfinite(x) for x in vals)
    first, last = vals[0], vals[-1]
    return {"component": "loss", "epochs": len(vals), "first": first if math.isfinite(first) else None,
            "last": last if math.isfinite(last) else None, "non_finite": non_finite,
            "collapsed": non_finite or last > first}


def criterion_b(rows, seeds, rule):
    """(b): PASS when no collapse row is collapsed and every model has exactly seeds_per_model distinct seeds.
    rows are collapse_* results (a caller may add labels such as "run"); seeds maps a model to its scored seeds."""
    n = rule.get("seeds_per_model") if isinstance(rule, dict) else None
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise SystemExit("(b) rule needs a positive integer seeds_per_model, got %r" % (n,))
    if not isinstance(rows, list) or not rows:
        raise SystemExit("(b) needs a non-empty list of collapse rows, got %r; it is refused, never scored" % (rows,))
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("collapsed"), bool):
            raise SystemExit("(b) collapse row %r has no boolean \"collapsed\"" % (r,))
    if not isinstance(seeds, dict) or not seeds:
        raise SystemExit("(b) needs the scored seeds of every model, got %r" % (seeds,))
    check = {}
    for model, s in seeds.items():
        if not isinstance(s, list) or not s:
            raise SystemExit("(b) model %r has no seed list, got %r" % (model, s))
        for x in s:
            if isinstance(x, bool) or not isinstance(x, numbers.Integral):
                raise SystemExit("(b) model %r seed %r is not an integer" % (model, x))
        check[model] = len(s) == n and len(set(s)) == n
    ok = not any(r["collapsed"] for r in rows) and all(check.values())
    return {"status": PASS if ok else FAIL, "rows": rows, "seeds": {m: list(s) for m, s in seeds.items()},
            "seed_check": check, "rule": rule.get("rule")}


def criterion_c(rows, rule):
    """(c) latency: always REPORTED (not gated), whatever the numbers; rows are carried as given."""
    if not isinstance(rows, list) or not rows:
        raise SystemExit("(c) needs a non-empty list of latency rows, got %r; it is refused, never shown" % (rows,))
    if not isinstance(rule, dict) or rule.get("status") != REPORTED or not rule.get("label"):
        raise SystemExit("(c) rule must be reported with a label (status %r), got %r" % (REPORTED, rule))
    return {"status": REPORTED, "label": rule["label"], "rows": rows, "rule": rule.get("rule")}


BOOT_KEYS = ("B", "seed", "level", "coverage")


def system_deltas(Y, PA, thA, PB, thB, boot):
    """The paired Laya - BERTurk deltas of one bootstrap draw (EVAL-07): paired_bootstrap's four metrics (micro-F1,
    macro-F1, exact match, top-1) then paired_bootstrap_extra's brier, classwise_ece and precision_at_<coverage>,
    both drawn with boot's (B, seed, level) on the same n, so every metric shares one index draw. Each metric is
    {a, b, delta, ci, width, n_undefined}. `boot` is the protocol's bootstrap block (B, seed, level, coverage); a
    missing key is refused. The verdict and protocol.json's known_at_freeze use this one function."""
    if not isinstance(boot, dict) or any(k not in boot for k in BOOT_KEYS):
        raise SystemExit("the bootstrap block %r lacks %s; it needs %s" % (
            boot, ", ".join(k for k in BOOT_KEYS if not isinstance(boot, dict) or k not in boot), ", ".join(BOOT_KEYS)))
    base = metrics.paired_bootstrap(Y, PA, thA, PB, thB, B=boot["B"], seed=boot["seed"], level=boot["level"])
    extra = metrics.paired_bootstrap_extra(Y, PA, thA, PB, thB, B=boot["B"], seed=boot["seed"], level=boot["level"],
                                           coverage=boot["coverage"])
    if (base["n"], base["B"]) != (extra["n"], extra["B"]):
        raise SystemExit("the two bootstraps drew on different (n, B): %s vs %s" % ((base["n"], base["B"]),
                                                                                   (extra["n"], extra["B"])))
    return {"n": base["n"], "B": base["B"], "seed": base["seed"], "level": base["level"], "method": base["method"],
            "coverage": extra["coverage"], "metrics": dict(base["metrics"], **extra["metrics"])}


# ---- the verdict record (08-09, REP-02 and REP-03) ----

VERDICT_SCHEMA = "verdict/1"
KINDS = ("laya", "berturk")                                 # A - B: every delta is Laya - BERTurk
TABLE_METRICS = ("micro_f1", "macro_f1", "exact_match", "top1_hit")
VERDICT_KEYS = ("schema", "protocol", "split", "systems", "references", "seed_averaged", "collapse", "latency",
                "fewshot", "criteria", "verdict")
PRIOR_KEYS = ("prior_val_fraction", "prior_test_fraction", "prior_theta_global")
# the seed-averaged deltas: system_deltas plus which system is A and B and each one's T and θ
SEEDED_KEYS = ("metrics", "B", "seed", "level", "method", "a", "b", "temperature", "theta")
FEWSHOT_PHASE = 9                                           # the phase that runs the few-shot path of (d)
# the fewshot block of a verdict built with the few-shot record (09-08, D-13); without it the block is PENDING_FEWSHOT
FEWSHOT_KEYS = ("status", "phase", "k", "mean_delta", "draws", "record", "mode", "disclosure")
FEWSHOT_INPUT_KEYS = ("mean_delta", "draws", "k", "record", "mode", "disclosure")
FEWSHOT_MODES = ("primary", "fallback")                     # pilot.fewshot.MODES (a test keeps the two equal)
FEWSHOT_FIX = "rebuild it with report.py --verdict --fewshot results/fewshot/fewshot.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def mean_std(values):
    """(mean, sample standard deviation with ddof 1) of per-seed values as floats (A16), computed exactly (the
    statistics module sums exact rationals), so equal values give their own value and a std of exactly 0.0. Fewer
    than two values, or a value that is not a finite number, is refused."""
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise SystemExit("mean_std needs at least two per-seed values, got %r; a seed is never dropped" % (values,))
    xs = [_finite(x, "a per-seed value") for x in values]
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def _check_rules(protocol):
    """SystemExit unless the protocol's verdict block holds exactly the rules, INCONCLUSIVE meaning and provisional
    text this module implements: decide runs this code, so it must be the code of the frozen rules."""
    v = protocol.get("verdict") if isinstance(protocol, dict) else None
    want = {"rules": list(VERDICT_RULES), "inconclusive_means": INCONCLUSIVE_MEANS, "provisional": PROVISIONAL_TEXT}
    got = {k: (v or {}).get(k) for k in want}
    if got != want:
        raise SystemExit("the protocol's verdict rules differ from pilot.verdict's (%s); the verdict is computed by "
                         "the code of the frozen rules only" % ", ".join(k for k in want if got[k] != want[k]))


def _run_row(run, kind, seed, train_run_id, dump_run_id, components):
    return {"run": run, "kind": kind, "seed": seed, "train_run_id": train_run_id, "dump_run_id": dump_run_id,
            "components": components, "collapsed": any(c["collapsed"] for c in components)}


def _collapse_rows(protocol, inputs, systems):
    """A6, one row per run in the order selection laya, selection berturk, final laya s0.., final berturk s0..:
    each selection run (its record's chosen candidate) by the val floor and the loss, each final seed by the loss and
    the test floor, against the protocol's frozen prior fractions."""
    col = protocol["collapse"]
    val_floor, test_floor = parse_fraction(col["prior_val_fraction"]), parse_fraction(col["prior_test_fraction"])
    sel = {s.get("kind"): s for s in (inputs.get("collapse") or {}).get("selection") or []}
    rows = []
    for kind in KINDS:
        s = sel.get(kind)
        if s is None:
            raise SystemExit("no selection run of %s in the collapse inputs; it is refused, never scored" % kind)
        o = s["objective"]
        rows.append(_run_row("selection", kind, s["seed"], s["train_run_id"], s["dump_run_id"],
                             [collapse_val(micro_fraction(o["tp"], o["fp"], o["fn"]), val_floor),
                              collapse_loss(s["loss_by_epoch"])]))
    for kind in KINDS:
        for r in systems[kind]["per_seed"]:
            n = r["test_counts"]
            rows.append(_run_row("final", kind, r["seed"], r["train_run_id"], r["dump_run_id"],
                                 [collapse_loss(r["loss_by_epoch"]),
                                  collapse_test_floor(micro_fraction(n["tp"], n["fp"], n["fn"]), test_floor)]))
    return {"rule": col.get("rule"), "prior_val_fraction": col["prior_val_fraction"],
            "prior_test_fraction": col["prior_test_fraction"], "prior_theta_global": col.get("prior_theta_global"),
            "rows": rows}


def _systems(inputs):
    """Each fine-tuned system as given plus its summary: per TABLE_METRICS metric the mean and sample std over its
    seeds and the per-seed values in seed order; the per-seed rows must be exactly the protocol's listed seeds."""
    out = {}
    for kind in KINDS:
        s = (inputs.get("systems") or {}).get(kind)
        if not isinstance(s, dict) or not isinstance(s.get("per_seed"), list):
            raise SystemExit("no per-seed rows of %s in the verdict inputs; it is refused, never scored" % kind)
        got = [r.get("seed") for r in s["per_seed"]]
        if got != list(s.get("seeds") or []):
            raise SystemExit("%s has per-seed rows for seeds %s, the protocol lists seeds %s; every listed final seed "
                             "is reported, none dropped" % (kind, got, s.get("seeds")))
        summary = {}
        for m in TABLE_METRICS:
            values = [r.get(m) for r in s["per_seed"]]
            mean, std = mean_std(values)
            summary[m] = {"mean": mean, "std": std, "values": values}
        out[kind] = dict(s, summary=summary)
    return out


def _pending_fewshot():
    return {"status": PENDING, "phase": FEWSHOT_PHASE}


def _fewshot_k(protocol):
    return _path_rule(protocol["criteria"]["d"], "fewshot").get("k")


def _fewshot_draws(protocol):
    """The number of frozen few-shot draws (the protocol's few-shot block), or None when it lists none."""
    block = protocol.get("fewshot") if isinstance(protocol, dict) else None
    draws = block.get("draws") if isinstance(block, dict) else None
    return len(draws) if isinstance(draws, list) and draws else None


def _fallback_disclosure(protocol):
    """The frozen compute-fallback disclosure of the protocol's few-shot block (None when it names none)."""
    block = protocol.get("fewshot") if isinstance(protocol, dict) else None
    rules = block.get("rules") if isinstance(block, dict) else None
    fallback = rules.get("compute_fallback") if isinstance(rules, dict) else None
    text = fallback.get("disclosure") if isinstance(fallback, dict) else None
    return text if isinstance(text, str) and text else None


def _check_fewshot(block, protocol, what):
    """SystemExit (one line naming `what` and the fix) unless `block` is the few-shot record's form of the fewshot block
    but for its status: exactly FEWSHOT_KEYS, phase FEWSHOT_PHASE, k criteria (d)'s fewshot k, one draw delta per
    frozen draw of the protocol's few-shot block (a draw is never dropped), mode one of FEWSHOT_MODES, disclosure None
    under primary and the protocol's compute-fallback disclosure under fallback, record {path, sha256} with a 64-hex
    sha256. The values of mean_delta and draws are checked by criterion_d."""
    def bad(text):
        raise SystemExit("%s: %s; %s" % (what, text, FEWSHOT_FIX))
    if not isinstance(block, dict) or set(block) != set(FEWSHOT_KEYS):
        bad("keys %s, expected exactly %s (or the PENDING form {status, phase})"
            % (sorted(block) if isinstance(block, dict) else type(block).__name__, ", ".join(FEWSHOT_KEYS)))
    if isinstance(block["phase"], bool) or block["phase"] != FEWSHOT_PHASE:
        bad("phase %r, expected %d" % (block["phase"], FEWSHOT_PHASE))
    k = _fewshot_k(protocol)
    if isinstance(block["k"], bool) or not isinstance(block["k"], int) or block["k"] != k:
        bad("k %r is not criteria (d)'s fewshot k %r" % (block["k"], k))
    n = _fewshot_draws(protocol)
    if n is None:
        bad("the protocol's few-shot block lists no draws, so the few-shot path cannot be bound to them")
    if not isinstance(block["draws"], list) or len(block["draws"]) != n:
        bad("draws %r, expected one delta per frozen draw (%d); a draw is never dropped" % (block["draws"], n))
    if block["mode"] not in FEWSHOT_MODES:
        bad("mode %r is not one of %s" % (block["mode"], ", ".join(FEWSHOT_MODES)))
    want = None if block["mode"] == "primary" else _fallback_disclosure(protocol)
    if block["mode"] == "fallback" and want is None:
        bad("the fallback mode needs the protocol's compute-fallback disclosure, and the protocol names none")
    if block["disclosure"] != want:
        bad("disclosure %r under the %s mode, expected %r" % (block["disclosure"], block["mode"], want))
    rec = block["record"]
    if not isinstance(rec, dict) or set(rec) != {"path", "sha256"} or not isinstance(rec["path"], str) \
            or not rec["path"] or not isinstance(rec["sha256"], str) or not SHA256_RE.fullmatch(rec["sha256"]):
        bad("record %r is not {path, sha256} of the few-shot results record" % (rec,))
    return block


def _fewshot_result(block):
    """criterion_d's few-shot input of a fewshot block: None (PENDING) or {"mean_delta", "draws"}."""
    return None if block is None else {"mean_delta": block["mean_delta"], "draws": block["draws"]}


def _metric(deltas, name):
    m = (deltas.get("metrics") or {}).get(name) if isinstance(deltas, dict) else None
    if not isinstance(m, dict):
        raise SystemExit("the headline deltas have no %r metric; the verdict needs every gated statistic" % name)
    return m


def build_verdict(protocol, sha, inputs):
    """The verdict/1 record (REP-03) from the frozen protocol and verdict_from's inputs: per system the per-seed rows,
    settings and summary (mean and sample std, A16), the seed-averaged deltas (headline gates, raw is reported, A18),
    the collapse rows (A6), the latency and reference rows, and the criteria from the protocol's criteria block: (a)
    on the headline micro-F1 and macro-F1 deltas, (b) on the collapse rows and the scored seeds, (c) on the latency
    rows, (d) on the headline Brier and precision-at-coverage deltas with the few-shot path PENDING until Phase 9, the
    alternative GO on the headline micro-F1 delta, and the overall verdict by decide (A3). No timestamp and no
    rounding: the same inputs give the same bytes. The protocol's verdict rules must be this module's.
    inputs["fewshot"] (09-08), when present, is {FEWSHOT_INPUT_KEYS} from the few-shot record (report.fewshot_input):
    (d)'s few-shot path reads its mean_delta and draws, and the fewshot block is FEWSHOT_KEYS with the path's status;
    absent (or None) the path is PENDING and the block {status PENDING, phase 9}, exactly as in Phase 8."""
    _check_rules(protocol)
    crit = protocol["criteria"]
    systems = _systems(inputs)
    seeded = inputs.get("seed_averaged") or {}
    for variant in ("headline", "raw"):
        if not isinstance(seeded.get(variant), dict) or any(k not in seeded[variant] for k in SEEDED_KEYS):
            raise SystemExit("the verdict inputs lack the %s seed-averaged deltas (keys %s)"
                             % (variant, ", ".join(SEEDED_KEYS)))
    h = seeded["headline"]
    collapse = _collapse_rows(protocol, inputs, systems)
    few = inputs.get("fewshot")
    if few is not None:
        if not isinstance(few, dict) or set(few) != set(FEWSHOT_INPUT_KEYS):
            raise SystemExit("the verdict inputs' fewshot is %r, expected None (PENDING) or {%s} from the few-shot "
                             "record (report.verdict_from)" % (few, ", ".join(FEWSHOT_INPUT_KEYS)))
        few = _check_fewshot(dict({"status": None, "phase": FEWSHOT_PHASE}, **few), protocol,
                             "the verdict inputs' few-shot result")
    cal, abst = _path_rule(crit["d"], "calibration"), _path_rule(crit["d"], "abstention")
    criteria = {
        "a": criterion_a(_metric(h, "micro_f1"), _metric(h, "macro_f1"), crit["a"]),
        "b": criterion_b(collapse["rows"], {k: [r["seed"] for r in systems[k]["per_seed"]] for k in KINDS},
                         crit["b"]),
        "c": criterion_c(inputs.get("latency"), crit["c"]),
        "d": criterion_d(_metric(h, cal.get("metric")), _metric(h, abst.get("metric")), _fewshot_result(few),
                         crit["d"]),
        "alternative_go": alternative_go(_metric(h, "micro_f1"), crit["alternative_go"]),
    }
    for k in criteria:
        criteria[k]["name"] = crit[k].get("name")
    s = {k: criteria[k]["status"] for k in ("a", "b", "d", "alternative_go")}
    if few is None:
        fewshot = _pending_fewshot()
    else:
        path = criteria["d"]["paths"][-1]
        fewshot = dict(few, status=path["status"], mean_delta=path["mean_delta"], draws=path["draws"])
        fewshot = {k: fewshot[k] for k in FEWSHOT_KEYS}
    out = {"schema": VERDICT_SCHEMA, "protocol": {"path": "protocol.json", "sha256": sha},
           "split": inputs.get("split"), "systems": systems, "references": inputs.get("references") or [],
           "seed_averaged": seeded, "collapse": collapse, "latency": inputs.get("latency"),
           "fewshot": fewshot, "criteria": criteria,
           "verdict": decide(s["a"], s["b"], s["d"], s["alternative_go"])}
    try:
        return json.loads(json.dumps(out, allow_nan=False))     # plain JSON values only, strict (no NaN)
    except ValueError as e:
        raise SystemExit("the verdict holds a value that is not strict JSON (%s); it is refused" % e)


# ---- the renders: REP-02's final table and REP-03's GO block (markdown; 08-12 writes them into the README) ----

REFERENCE_ORDER = ("majority", "prior", "laya")             # REP-02's fixed order: majority, label prior, zero-shot Laya
FINAL_HEADING = "### Results on test (Phase 8)"
GO_HEADING = "### GO block (mechanical)"
TABLE_HEADER = ("system", "run", "micro-F1", "macro-F1", "exact match", "top-1")
SETTINGS_HEADER = ("system", "T", "θ shipped global (per-tag)", "θ raw global (per-tag)", "epoch / schedule",
                   "encoder / head LR", "seeds", "train run ids", "selection run (chosen dump)")
DESCRIPTIVE = ("classwise_ece", "exact_match", "top1_hit")  # GO block's closing line, with the raw micro-F1 delta
SIDE = {"laya": "a", "berturk": "b"}                        # the seed-averaged deltas are Laya (a) - BERTurk (b)


def _verdict_record(v):
    if not isinstance(v, dict) or v.get("schema") != VERDICT_SCHEMA:
        raise SystemExit("not a %s record (schema %r); write it with report.py --verdict --out PATH"
                         % (VERDICT_SCHEMA, v.get("schema") if isinstance(v, dict) else type(v).__name__))
    return v


def check_record(v, protocol, sha):
    """SystemExit unless the verdict/1 record `v` is the frozen protocol's (08-REVIEW WR-03): its protocol block is
    {path protocol.json, sha256 `sha`} (the pin), the protocol's verdict rules are this module's (_check_rules), and
    every gated number it renders recomputes from the record's own numbers and the frozen criteria block, exactly as
    build_verdict computed them: each collapse component from its value and the protocol's floor (the loss component
    from its first, last and non_finite), each row's collapsed flag, each final seed's test floor from its test
    counts, the scored seeds the protocol's; then every criterion ((a), (b), (c), (d), the alternative GO) from the
    headline seed-averaged deltas, the collapse rows and the latency rows, and the verdict by decide. A hand-edited
    status, verdict, criterion number or protocol sha256 is refused, never rendered. The headline numbers themselves
    are bound to the dumps only by the byte rebuild (report.py --verdict). The fewshot block (09-08) is either the
    PENDING form {status PENDING, phase 9} ((d)'s few-shot path PENDING) or the few-shot record's form (_check_fewshot:
    exactly FEWSHOT_KEYS, phase 9, criteria (d)'s fewshot k, a mode, the disclosure of that mode, a record {path,
    sha256}), whose mean_delta and draws (d) is recomputed from and whose status must be the recomputed path's; the
    few-shot numbers are bound to the dumps by report.fewshot_input when the verdict is built."""
    v = _verdict_record(v)
    if v.get("protocol") != {"path": "protocol.json", "sha256": sha}:
        raise SystemExit("the verdict names protocol %r, not the frozen protocol.json (sha256 %s); rebuild it with "
                         "report.py --verdict" % (v.get("protocol"), sha))
    _check_rules(protocol)
    crit, col = protocol["criteria"], protocol["collapse"]
    seeded = v.get("seed_averaged") or {}
    h = seeded.get("headline")
    if not isinstance(h, dict) or (h.get("a"), h.get("b")) != (KINDS[0], KINDS[1]):
        raise SystemExit("the verdict's headline deltas are not %s - %s; rebuild it with report.py --verdict" % KINDS)
    floors = {"val_floor": col["prior_val_fraction"], "test_floor": col["prior_test_fraction"]}
    c_rows = (v.get("collapse") or {}).get("rows")
    if not isinstance(c_rows, list) or not c_rows:
        raise SystemExit("the verdict holds no collapse row; rebuild it with report.py --verdict")
    counts = {(k, r.get("seed")): r.get("test_counts") for k in KINDS
              for r in ((v.get("systems") or {}).get(k) or {}).get("per_seed") or []}
    for r in c_rows:
        where = "collapse row %s %s s%s" % (r.get("run"), r.get("kind"), r.get("seed"))
        for c in r.get("components") or []:
            name = c.get("component")
            if name in floors:
                want = _floor(name, c.get("value"), floors[name])
                if name == "test_floor":
                    n = counts.get((r.get("kind"), r.get("seed"))) or {}
                    if parse_fraction(c.get("value")) != micro_fraction(n.get("tp"), n.get("fp"), n.get("fn")):
                        raise SystemExit("%s: the test floor value %s is not the seed's test counts; rebuild the "
                                         "verdict with report.py --verdict" % (where, c.get("value")))
            elif name == "loss":
                first, last, non_finite = c.get("first"), c.get("last"), c.get("non_finite")
                want = dict(c, collapsed=bool(non_finite or first is None or last is None or last > first))
            else:
                raise SystemExit("%s: unknown component %r" % (where, name))
            if c != want:
                raise SystemExit("%s: the %s component %r does not recompute (%r); the record was edited, rebuild it "
                                 "with report.py --verdict" % (where, name, c, want))
        if r.get("collapsed") != any(c["collapsed"] for c in r.get("components") or []):
            raise SystemExit("%s: collapsed %r is not any of its components'; rebuild the verdict with report.py "
                             "--verdict" % (where, r.get("collapsed")))
    seeds = {k: [r.get("seed") for r in ((v.get("systems") or {}).get(k) or {}).get("per_seed") or []] for k in KINDS}
    want_seeds = {k: list(((protocol.get("systems") or {}).get(k) or {}).get("seeds") or []) for k in KINDS}
    if seeds != want_seeds:
        raise SystemExit("the verdict's scored seeds %s are not the protocol's %s; rebuild it with report.py --verdict"
                         % (seeds, want_seeds))
    cal, abst = _path_rule(crit["d"], "calibration"), _path_rule(crit["d"], "abstention")
    block = v.get("fewshot")
    few = None if block == _pending_fewshot() else _check_fewshot(block, protocol, "the verdict's few-shot block")
    try:
        d = criterion_d(_metric(h, cal.get("metric")), _metric(h, abst.get("metric")), _fewshot_result(few),
                        crit["d"])
    except SystemExit as e:
        raise SystemExit("the verdict's few-shot block does not recompute (%s); %s" % (e.code, FEWSHOT_FIX))
    want = {
        "a": criterion_a(_metric(h, "micro_f1"), _metric(h, "macro_f1"), crit["a"]),
        "b": criterion_b(c_rows, seeds, crit["b"]),
        "c": criterion_c(v.get("latency"), crit["c"]),
        "d": d,
        "alternative_go": alternative_go(_metric(h, "micro_f1"), crit["alternative_go"]),
    }
    for k in want:
        want[k]["name"] = crit[k].get("name")
    want = json.loads(json.dumps(want, allow_nan=False))
    got = v.get("criteria") or {}
    bad = [k for k in want if got.get(k) != want[k]]
    if bad or set(got) != set(want):
        raise SystemExit("the verdict's criteria %s do not recompute from its own numbers and the frozen criteria; the "
                         "record was edited, rebuild it with report.py --verdict"
                         % ", ".join(bad or sorted(set(got) ^ set(want))))
    s = {k: want[k]["status"] for k in ("a", "b", "d", "alternative_go")}
    dec = json.loads(json.dumps(decide(s["a"], s["b"], s["d"], s["alternative_go"])))
    if v.get("verdict") != dec:
        raise SystemExit("the verdict %r is not decide() of its recomputed statuses (%r); the record was edited, "
                         "rebuild it with report.py --verdict" % ((v.get("verdict") or {}).get("verdict"),
                                                                  dec["verdict"]))
    if few is not None and few["status"] != want["d"]["paths"][-1]["status"]:
        raise SystemExit("the verdict's few-shot block has status %r, its own numbers give %r; the record was edited, "
                         "%s" % (few["status"], want["d"]["paths"][-1]["status"], FEWSHOT_FIX))
    return v


def _md(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + \
        ["| " + " | ".join(r) + " |" for r in rows]


def _f(x, fmt="%.3f"):
    return "n/a" if x is None else fmt % x


def _points(x):
    return "n/a" if x is None else "%+.2f points" % (x * 100)


def _ci(ci):
    return "n/a" if ci is None else "[%+.2f, %+.2f]" % (ci[0] * 100, ci[1] * 100)


def _seed_text(seeds):
    return ", ".join("%d" % s for s in seeds)


def _references(v):
    """The verdict's reference rows in REFERENCE_ORDER (by system kind, then name); none, an unknown kind or a row
    without a table metric is refused (REP-02 empty edge)."""
    refs = v.get("references") or []
    if not refs:
        raise SystemExit("the verdict holds no reference row; REP-02's table needs the protocol's reference reports "
                         "(rebuild it with report.py --verdict)")
    for r in refs:
        if r.get("kind") not in REFERENCE_ORDER or any(m not in r for m in TABLE_METRICS):
            raise SystemExit("reference row %r is not a %s row with %s" % (r.get("name"), "/".join(REFERENCE_ORDER),
                                                                       ", ".join(TABLE_METRICS)))
    return sorted(refs, key=lambda r: (REFERENCE_ORDER.index(r["kind"]), str(r.get("name"))))


def _ref_label(r):
    if r.get("checkpoint"):
        return "%s (%s)" % (r["kind"], r["checkpoint"])
    return "%s (train: %s)" % (r["kind"], r.get("baseline_split"))


def _fine_tuned(v, kind):
    """(system, seeds) of a fine-tuned system, refused unless its per-seed rows are exactly its listed seeds (at least
    two; a seed is never dropped) and its summary is mean_std of those rows (else the record was edited)."""
    s = (v.get("systems") or {}).get(kind) or {}
    seeds = [r.get("seed") for r in s.get("per_seed") or []]
    if seeds != list(s.get("seeds") or []) or len(seeds) < 2:
        raise SystemExit("%s: per-seed rows for seeds %s, the protocol lists seeds %s; every final seed is shown, none "
                         "dropped" % (kind, seeds, s.get("seeds")))
    for m in TABLE_METRICS:
        values = [r[m] for r in s["per_seed"]]
        mean, std = mean_std(values)
        if (s.get("summary") or {}).get(m) != {"mean": mean, "std": std, "values": values}:
            raise SystemExit("%s: the %s summary is not mean_std of its per-seed values; rebuild the record with "
                             "report.py --verdict" % (kind, m))
    return s, seeds


def render_final_table(v):
    """REP-02's one results table (markdown, level-3 heading) of a verdict/1 record: the reference rows in
    REFERENCE_ORDER (majority, prior, zero-shot Laya), then per fine-tuned model (Laya, BERTurk) a row of mean ± sample
    std (ddof 1, A16) with the per-seed values in seed order and a row of its seed-averaged EVAL-07 system (the one
    the GO block compares), at the val-selected θ and headline T; then the settings table (T, θ shipped and raw with
    the per-tag count, epoch and schedule, encoder and head LR, seeds, train run ids, the selection run) and the bound
    records. Values are rounded for display only (3 decimals); identical input gives identical text."""
    v = _verdict_record(v)
    sp, refs = v["split"], _references(v)
    h = v["seed_averaged"]["headline"]
    rows, settings, records = [], [], []
    for r in refs:
        rows.append((_ref_label(r), r["run_id"]) + tuple(_f(r[m]) for m in TABLE_METRICS))
        settings.append((_ref_label(r), "%.4f" % r["temperature"], "%.2f (%d)" % (r["theta"]["global"],
                                                                                r["theta"]["n_per_tag"]))
                        + ("n/a",) * 6)
    for kind in KINDS:
        s, seeds = _fine_tuned(v, kind)
        st = s["settings"]
        rows.append(("%s fine-tuned (final seeds %s)" % (kind, _seed_text(seeds)),
                     "mean ± std (seed %s)" % " / ".join("%d" % x for x in seeds))
                    + tuple("%.3f ± %.3f (%s)" % (s["summary"][m]["mean"], s["summary"][m]["std"],
                                                  " / ".join(_f(x) for x in s["summary"][m]["values"]))
                            for m in TABLE_METRICS))
        rows.append(("%s fine-tuned, seed-averaged (EVAL-07 system)" % kind, "seeds %s averaged" % _seed_text(seeds))
                    + tuple(_f(h["metrics"][m][SIDE[kind]]) for m in TABLE_METRICS))
        th = st["theta"]
        settings.append(("%s fine-tuned" % kind, "%.4f" % st["temperature"],
                         "%.2f (%d)" % (th["shipped"]["global"], th["shipped"]["n_per_tag"]),
                         "%.2f (%d)" % (th["raw"]["global"], th["raw"]["n_per_tag"]),
                         "%d / %d" % (st["epoch"], st["epochs_schedule"]),
                         "%g / %g" % (st["encoder_lr"], st["head_lr"]), _seed_text(seeds),
                         ", ".join(r["train_run_id"] for r in s["per_seed"]),
                         "%s (%s)" % (st["train_run_id"], st["chosen"])))
        records.append("- %s: postfit record %s `%s`, final manifest %s `%s`, config sha256 without the seed `%s`."
                       % (kind, s["postfit"]["path"], s["postfit"]["sha256"][:12], s["final_manifest"]["path"],
                          s["final_manifest"]["sha256"][:12], str(st.get("config_sha256_sans_seed"))[:12]))
    out = [FINAL_HEADING, "",
           "Split %s: n = %d (%d no-intent), sha256 `%s`, uids sha256 `%s`; protocol sha256 `%s`."
           % (sp["name"], sp["n"], sp["n_no_intent"], sp["sha256"][:12], sp["uids_sha256"][:12],
              v["protocol"]["sha256"][:12]),
           "Every value is at the val-selected θ and headline T: a reference row at its own val θ, a fine-tuned row "
           "as the mean ± sample std (ddof 1) over its final seeds with the per-seed values in seed order, the "
           "seed-averaged row the system the GO block compares (each seed at its record T, averaged, then the "
           "record's theta.shipped).", ""]
    out += _md(TABLE_HEADER, rows)
    out += ["", "Settings (T and θ are each postfit record's, chosen on val, never re-selected on test):", ""]
    out += _md(SETTINGS_HEADER, settings) + [""] + records
    return "\n".join(out) + "\n"


def _criterion_row(key, c, v):
    """(label, status, numbers) of one GO block row; every number and label read from the verdict record."""
    label = "(%s) %s" % (key, c["name"]) if len(key) == 1 else c["name"]
    if key == "a":
        nums = "micro-F1 delta %s, CI %s; macro-F1 delta %s, CI %s (%s)" % (
            _points(c["micro"]["delta"]), _ci(c["micro"]["ci"]), _points(c["macro"]["delta"]), _ci(c["macro"]["ci"]),
            c["macro_ci_label"])
    elif key == "b":
        bad = ["%s %s s%s" % (r["run"], r["kind"], r["seed"]) for r in c["rows"] if r["collapsed"]]
        nums = "%d runs, %d collapsed%s; seeds %s" % (
            len(c["rows"]), len(bad), (": " + ", ".join(bad)) if bad else "",
            "; ".join("%s %s%s" % (m, _seed_text(s), "" if c["seed_check"].get(m) else " (seed check failed)")
                      for m, s in c["seeds"].items()))
    elif key == "c":
        nums = "%s; %s" % (c["label"], "; ".join("%s %s p50 %s s, p95 %s s (n = %s)" % (
            r["system"], r["device"], _f(r["p50"]), _f(r["p95"]), r["n"]) for r in c["rows"]))
    elif key == "d":
        parts = []
        for p in c["paths"]:
            if p["status"] == PENDING:
                parts.append("%s (%s): %s (Phase %s)" % (p["path"], p["metric"], p["status"], v["fewshot"]["phase"]))
            elif "delta" in p:
                parts.append("%s (%s): %s, delta %s, CI %s" % (p["path"], p["metric"], p["status"], _points(p["delta"]),
                                                               _ci(p["ci"])))
            else:
                parts.append("%s (%s): %s, mean delta %s" % (p["path"], p["metric"], p["status"],
                                                             _points(p["mean_delta"])))
        nums = "; ".join(parts)
    else:
        nums = "micro-F1 delta %s, CI %s" % (_points(c["micro"]["delta"]), _ci(c["micro"]["ci"]))
    return label, c["status"], nums


def render_go(v):
    """REP-03's GO block (markdown, level-3 heading) of a verdict/1 record: the protocol sha256 (first 12 hex
    characters), the split and the bootstrap of the seed-averaged deltas, one row per criterion ((a), (b), (c), (d))
    and the alternative GO with its status and numbers (points, 95% CI), the statuses line ((c) REPORTED (not
    gated)), the verdict and the frozen rule line(s) that fired, INCONCLUSIVE_MEANS whenever INCONCLUSIVE appears, and
    a closing descriptive line (classwise ECE, exact match and top-1 deltas, the raw variant's micro-F1 delta). Every
    label and number is a field of the record or a frozen text (VERDICT_RULES, INCONCLUSIVE_MEANS, UNITS); rounding is
    for display only (2 decimals for points, 3 for metrics)."""
    v = _verdict_record(v)
    crit, dec, sp = v["criteria"], v["verdict"], v["split"]
    h, raw = v["seed_averaged"]["headline"], v["seed_averaged"]["raw"]
    keys = ("a", "b", "c", "d", "alternative_go")
    out = [GO_HEADING, "",
           "- Protocol: protocol.json sha256 `%s`." % v["protocol"]["sha256"][:12],
           "- Split %s: n = %d (%d no-intent)." % (sp["name"], sp["n"], sp["n_no_intent"]),
           "- Deltas: %s - %s, seed-averaged, headline (T: %s; θ: %s); B = %d, seed %d, %g%% interval, %s." % (
               h["a"], h["b"], "; ".join("%s %.4f" % kv for kv in h["temperature"].items()),
               "; ".join("%s %s" % kv for kv in h["theta"].items()), h["B"], h["seed"], h["level"] * 100, h["method"]),
           "- Units: %s." % UNITS]
    few = v["fewshot"]
    if few != _pending_fewshot():                           # the few-shot record's form only: PENDING renders as before
        out.append("- Few-shot path (Phase %d): k = %d, mean over %d draws, mode %s; record %s sha256 `%s`." % (
            few["phase"], few["k"], len(few["draws"]), few["mode"], few["record"]["path"],
            few["record"]["sha256"][:12]))
        if few["disclosure"]:
            out.append("- Few-shot compute fallback: %s." % few["disclosure"].rstrip("."))
    out.append("")
    out += _md(("criterion", "status", "numbers"), [_criterion_row(k, crit[k], v) for k in keys])
    out += ["", "Statuses: %s." % "; ".join("%s %s" % ("(%s)" % k if len(k) == 1 else crit[k]["name"],
                                                       crit[k]["status"]) for k in keys), "",
            "**Verdict: %s**" % dec["verdict"], ""]
    fired = dec["rule"] if isinstance(dec["rule"], list) else [dec["rule"]]
    if dec["provisional"]:
        out.append("- Rule %d: %s" % (len(VERDICT_RULES), VERDICT_RULES[-1]))
    out += ["- Rule %d: %s" % (i, VERDICT_RULES[i - 1]) for i in fired if i is not None]
    statuses = [crit[k]["status"] for k in keys] + [dec["verdict"], dec.get("if_fewshot_met"),
                                                    dec.get("if_fewshot_not_met")]
    if INCONCLUSIVE in statuses:
        out.append("- %s %s." % (INCONCLUSIVE, INCONCLUSIVE_MEANS))
    m = h["metrics"]
    out += ["", "Descriptive, not gating: %s; raw (T: %s; θ: %s) micro_f1 delta %s, CI %s." % (
        "; ".join("%s delta %s, CI %s" % (k, _points(m[k]["delta"]), _ci(m[k]["ci"])) for k in DESCRIPTIVE),
        "; ".join("%s %.4f" % kv for kv in raw["temperature"].items()),
        "; ".join("%s %s" % kv for kv in raw["theta"].items()),
        _points(raw["metrics"]["micro_f1"]["delta"]), _ci(raw["metrics"]["micro_f1"]["ci"]))]
    return "\n".join(out) + "\n"
