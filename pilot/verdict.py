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
Inputs are validated before any comparison: a missing or non-finite delta, a missing, non-finite or misordered CI and
an unknown status raise SystemExit naming the input, so an empty result is refused, never scored.

numpy and stdlib only: no torch, scipy, Ray, laya or sklearn (report.py imports this module).

usage: from pilot import verdict
       a = verdict.criterion_a({"delta": d, "ci": [lo, hi]}, {"delta": d, "ci": [lo, hi]}, criteria["a"])
       alt = verdict.alternative_go({"delta": d, "ci": [lo, hi]}, criteria["alternative_go"])
       d = verdict.criterion_d(brier, precision_at_80, None, criteria["d"])     # None: the few-shot path is pending
       verdict.decide(a["status"], b["status"], d["status"], alt["status"])   # {"verdict", "provisional", ...}
"""
import math
import numbers

import numpy as np

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
        raise SystemExit("(d) fewshot needs None (pending) or a {\"mean_delta\", \"draws\"} result, got %r" % (fewshot,))
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
    {"mean_delta", "draws"}. Returns {"status", "paths": [calibration, abstention, fewshot], "reported_only", "rule"}."""
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
