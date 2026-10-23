"""Unit contracts of pilot/verdict.py (REP-01, REP-03): the mechanical GO block, written and tested before any
fine-tuned test number exists and before protocol.json is frozen.

- decide/overall: the frozen order of VERDICT_RULES (A3), checked on every one of the 36 (a, b, d, alternative)
  status combinations against an expectation table written here, independently of the code, from the seven rule lines;
  the provisional form (d PENDING) names both outcomes and is provisional only when they differ.
- criterion_a (A2, three-way), alternative_go (A4) and criterion_d (A1: Brier, precision at 80% coverage, few-shot):
  every threshold comes from the rule dict passed in, every comparison is on unrounded float64 values, and each
  boundary is pinned exactly and one ulp (numpy.nextafter) to the other side.
- the collapse rule (A6) and criteria (b) and (c): exact Fractions for the floors, a non-finite or divergent loss,
  the seed count, and the refusals of empty or missing inputs.
- pilot.verdict imports numpy and the stdlib only (report.py will import it).
Offline: runs in `make test`.
"""
import ast
import copy
import inspect
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pilot.data
from pilot import verdict as v

pytestmark = pytest.mark.unit

HEAVY = ("torch", "scipy", "sklearn", "laya", "ray", "transformers")


def down(x):
    """One ulp below x."""
    return float(np.nextafter(x, -np.inf))


def up(x):
    """One ulp above x."""
    return float(np.nextafter(x, np.inf))


def stat(delta, lo, hi):
    return {"delta": delta, "ci": [lo, hi]}


# ---- the verdict (A3) ----

def test_decide_examples():
    assert v.decide("PASS", "PASS", "PASS", "NOT MET")["verdict"] == "GO"
    assert v.decide("INCONCLUSIVE", "PASS", "FAIL", "NOT MET")["verdict"] == "INCONCLUSIVE"
    assert v.decide("PASS", "FAIL", "PASS", "MET")["verdict"] == "NO-GO"           # (b) FAIL wins over everything
    assert v.decide("FAIL", "PASS", "PASS", "MET")["verdict"] == "GO"              # the alternative GO comes before (a)


def test_decide_provisional_names_both_outcomes():
    d = v.decide("PASS", "PASS", "PENDING", "NOT MET")
    assert d["provisional"] is True
    assert d["if_fewshot_met"] == "GO" and d["if_fewshot_not_met"] == "NO-GO"
    assert d["verdict"] == "PROVISIONAL: GO if the few-shot path of (d) is met, else NO-GO"
    assert d["verdict"] == v.PROVISIONAL_TEXT % ("GO", "NO-GO")
    assert d["verdict"] not in ("GO", "NO-GO")                                      # never shown as final
    assert d["rule"] == [5, 6]                                                      # both branches' rule lines


def test_decide_pending_is_final_when_both_branches_agree():
    d = v.decide("INCONCLUSIVE", "PASS", "PENDING", "NOT MET")
    assert d["provisional"] is False and d["verdict"] == "INCONCLUSIVE"
    assert d["if_fewshot_met"] == d["if_fewshot_not_met"] == "INCONCLUSIVE"
    assert d["rule"] == 4
    d = v.decide("FAIL", "PASS", "PENDING", "MET")
    assert d["provisional"] is False and d["verdict"] == "GO" and d["rule"] == 2


def test_decide_without_pending_names_no_branch():
    d = v.decide("PASS", "PASS", "FAIL", "NOT MET")
    assert d["provisional"] is False and d["verdict"] == "NO-GO" and d["rule"] == 6
    assert d["if_fewshot_met"] is None and d["if_fewshot_not_met"] is None


def test_inconclusive_always_counts_as_no_go():
    d = v.decide("INCONCLUSIVE", "PASS", "PASS", "NOT MET")
    assert d["verdict"] == "INCONCLUSIVE" and d["verdict"] != "GO"
    assert d["means"] == "counts as NO-GO; collect bank data" == v.INCONCLUSIVE_MEANS
    assert v.decide("PASS", "PASS", "PASS", "NOT MET")["means"] is None


def test_overall_returns_none_only_for_the_pending_branch():
    assert v.overall("PASS", "PASS", "PENDING", "NOT MET") is None
    assert v.overall("INCONCLUSIVE", "PASS", "PENDING", "NOT MET") == "INCONCLUSIVE"


def test_verdict_rules_are_the_seven_ordered_lines():
    assert isinstance(v.VERDICT_RULES, tuple) and len(v.VERDICT_RULES) == 7
    assert all(isinstance(r, str) and r for r in v.VERDICT_RULES)
    assert "(b) FAIL" in v.VERDICT_RULES[0] and "NO-GO" in v.VERDICT_RULES[0]
    assert "alternative GO" in v.VERDICT_RULES[1] and v.VERDICT_RULES[1].endswith("-> GO")
    assert "(a) FAIL" in v.VERDICT_RULES[2]
    assert "(a) INCONCLUSIVE" in v.VERDICT_RULES[3] and v.INCONCLUSIVE_MEANS in v.VERDICT_RULES[3]
    assert "(d) PASS" in v.VERDICT_RULES[4] and "(d) FAIL" in v.VERDICT_RULES[5]
    assert "(d) PENDING" in v.VERDICT_RULES[6] and "PROVISIONAL" in v.VERDICT_RULES[6]


def _expected(a, b, d, alt):
    """The seven VERDICT_RULES lines, transcribed independently of pilot.verdict: (verdict, rule)."""
    if b == "FAIL":
        return "NO-GO", 1
    if alt == "MET":
        return "GO", 2
    if a == "FAIL":
        return "NO-GO", 3
    if a == "INCONCLUSIVE":
        return "INCONCLUSIVE", 4
    if d == "PASS":
        return "GO", 5
    if d == "FAIL":
        return "NO-GO", 6
    return "PROVISIONAL: GO if the few-shot path of (d) is met, else NO-GO", [5, 6]


def test_verdict_truth_table():
    rows = list(itertools.product(("PASS", "FAIL", "INCONCLUSIVE"), ("PASS", "FAIL"), ("PASS", "FAIL", "PENDING"),
                                  ("MET", "NOT MET")))
    assert len(rows) == 36
    seen = set()
    for a, b, d, alt in rows:
        want, rule = _expected(a, b, d, alt)
        got = v.decide(a, b, d, alt)
        assert (got["verdict"], got["rule"]) == (want, rule), (a, b, d, alt, got)
        assert got["provisional"] is want.startswith("PROVISIONAL"), (a, b, d, alt)
        assert got["statuses"] == {"a": a, "b": b, "d": d, "alternative_go": alt}
        if d == "PENDING":                                          # the branches are the rule with d PASS / d FAIL
            assert got["if_fewshot_met"] == _expected(a, b, "PASS", alt)[0]
            assert got["if_fewshot_not_met"] == _expected(a, b, "FAIL", alt)[0]
        seen.add(got["verdict"])
    assert seen == {"GO", "NO-GO", "INCONCLUSIVE", "PROVISIONAL: GO if the few-shot path of (d) is met, else NO-GO"}


@pytest.mark.parametrize("args", [("pass", "PASS", "PASS", "MET"), ("PASS", "INCONCLUSIVE", "PASS", "MET"),
                                  ("PASS", "PASS", "MET", "MET"), ("PASS", "PASS", "PASS", "PASS"),
                                  (None, "PASS", "PASS", "MET"), ("PENDING", "PASS", "PASS", "MET")])
def test_unknown_status_is_refused(args):
    with pytest.raises(SystemExit, match="status"):
        v.decide(*args)
    with pytest.raises(SystemExit, match="status"):
        v.overall(*args)


# ---- (a): three-way non-inferiority (A2) ----

def test_criterion_a_pass_at_exact_boundaries():
    r = v.criterion_a(stat(-0.01, -0.02, 0.0), stat(-0.03, -0.07, 0.01), v.DEFAULT_CRITERIA["a"])
    assert r["status"] == "PASS"
    assert r["macro_ci_label"] == "not resolvable at this test size (n = 413)"
    assert r["micro"] == {"delta": -0.01, "ci": [-0.02, 0.0]} and r["macro"]["ci"] == [-0.07, 0.01]


def test_criterion_a_one_ulp_below_the_lower_bound_is_not_pass():
    rule = v.DEFAULT_CRITERIA["a"]
    r = v.criterion_a(stat(-0.01, down(-0.02), 0.0), stat(-0.03, -0.07, 0.01), rule)
    assert r["status"] == "INCONCLUSIVE"
    r = v.criterion_a(stat(-0.01, -0.02, 0.0), stat(down(-0.03), -0.07, 0.01), rule)
    assert r["status"] == "FAIL"                                    # macro point below its minimum fails


def test_criterion_a_fail_needs_the_upper_bound_strictly_below():
    rule = v.DEFAULT_CRITERIA["a"]
    r = v.criterion_a(stat(-0.04, -0.06, -0.02), stat(0.0, -0.03, 0.03), rule)
    assert r["status"] == "INCONCLUSIVE"                            # upper exactly -0.02: not FAIL
    r = v.criterion_a(stat(-0.04, -0.06, down(-0.02)), stat(0.0, -0.03, 0.03), rule)
    assert r["status"] == "FAIL"


def test_criterion_a_reads_its_thresholds_from_the_rule():
    rule = copy.deepcopy(v.DEFAULT_CRITERIA["a"])
    rule["micro_ci_lower_min"] = rule["micro_fail_ci_upper_below"] = -0.05
    assert v.criterion_a(stat(-0.01, -0.04, 0.0), stat(0.0, -0.03, 0.03), rule)["status"] == "PASS"
    assert v.criterion_a(stat(-0.01, -0.04, 0.0), stat(0.0, -0.03, 0.03), v.DEFAULT_CRITERIA["a"])["status"] \
        == "INCONCLUSIVE"


@pytest.mark.parametrize("micro", [None, {"delta": -0.01}, {"ci": [-0.02, 0.0]}, stat(-0.01, None, 0.0),
                                   {"delta": -0.01, "ci": None}, stat(float("nan"), -0.02, 0.0),
                                   stat(float("inf"), -0.02, 0.0), stat(-0.01, 0.01, -0.01),
                                   stat(-0.01, -0.02, float("nan")), stat(True, -0.02, 0.0),
                                   {"delta": -0.01, "ci": [-0.02]}, stat("-0.01", -0.02, 0.0)])
def test_criterion_a_refuses_missing_or_non_finite(micro):
    with pytest.raises(SystemExit):
        v.criterion_a(micro, stat(0.0, -0.03, 0.03), v.DEFAULT_CRITERIA["a"])
    with pytest.raises(SystemExit):
        v.criterion_a(stat(0.0, -0.01, 0.01), micro, v.DEFAULT_CRITERIA["a"])


def test_criterion_a_refuses_an_overlapping_rule():
    rule = copy.deepcopy(v.DEFAULT_CRITERIA["a"])
    rule["micro_fail_ci_upper_below"] = 0.0                         # FAIL would overlap PASS
    with pytest.raises(SystemExit, match="overlap"):
        v.criterion_a(stat(0.0, -0.01, 0.01), stat(0.0, -0.01, 0.01), rule)


# ---- the alternative GO (A4) ----

def test_alternative_go_needs_a_delta_strictly_above():
    rule = v.DEFAULT_CRITERIA["alternative_go"]
    assert v.alternative_go(stat(0.03, 0.01, 0.05), rule)["status"] == "NOT MET"
    assert v.alternative_go(stat(up(0.03), 0.01, 0.05), rule)["status"] == "MET"
    assert v.alternative_go(stat(0.05, 0.0, 0.1), rule)["status"] == "NOT MET"      # CI lower must exclude 0
    assert v.alternative_go(stat(0.05, up(0.0), 0.1), rule)["status"] == "MET"
    with pytest.raises(SystemExit):
        v.alternative_go(stat(0.05, None, 0.1), rule)


# ---- (d): any one path (A1) ----

def _d(cal=None, abst=None, few=None, rule=None):
    cal = cal if cal is not None else stat(0.005, 0.002, 0.008)                    # BERTurk better: NOT MET
    abst = abst if abst is not None else stat(-0.1, -0.2, -0.05)                   # NOT MET
    return v.criterion_d(cal, abst, few, rule or v.DEFAULT_CRITERIA["d"])


def _path(r, name):
    (p,) = [p for p in r["paths"] if p["path"] == name]
    return p


def test_criterion_d_calibration_path():
    assert _path(_d(cal=stat(-0.001, -0.002, down(0.0))), "calibration")["status"] == "MET"
    assert _path(_d(cal=stat(-0.001, -0.002, 0.0)), "calibration")["status"] == "NOT MET"   # CI upper must be < 0
    assert _path(_d(cal=stat(0.0, -0.002, -0.001)), "calibration")["status"] == "NOT MET"   # delta must be < 0
    assert _d(cal=stat(-0.001, -0.002, -0.0005))["status"] == "PASS"


def test_criterion_d_abstention_path():
    assert _path(_d(abst=stat(0.03, up(0.0), 0.06)), "abstention")["status"] == "MET"
    assert _path(_d(abst=stat(down(0.03), 0.01, 0.06)), "abstention")["status"] == "NOT MET"
    assert _path(_d(abst=stat(0.05, 0.0, 0.1)), "abstention")["status"] == "NOT MET"        # CI lower must be > 0


def test_criterion_d_fewshot_path():
    r = _d()
    assert _path(r, "fewshot")["status"] == "PENDING" and r["status"] == "PENDING"
    r = _d(few={"mean_delta": 0.05, "draws": [0.05]})
    assert _path(r, "fewshot")["status"] == "MET" and r["status"] == "PASS"
    r = _d(few={"mean_delta": down(0.05), "draws": [down(0.05)]})
    assert _path(r, "fewshot")["status"] == "NOT MET" and r["status"] == "FAIL"


def test_criterion_d_status_is_pass_when_any_path_is_met():
    met_cal = stat(-0.001, -0.002, -0.0005)
    assert _d(cal=met_cal, few={"mean_delta": -0.1, "draws": [-0.1]})["status"] == "PASS"
    assert _d(cal=met_cal)["status"] == "PASS"                         # PASS even while few-shot is pending
    assert _d(few={"mean_delta": -0.1, "draws": [-0.1]})["status"] == "FAIL"


@pytest.mark.parametrize("bad", [None, {"delta": 0.1}, stat(float("nan"), 0.0, 0.1), stat(0.1, 0.2, 0.1)])
def test_criterion_d_refuses_missing_or_non_finite(bad):
    with pytest.raises(SystemExit):
        v.criterion_d(bad, stat(-0.1, -0.2, -0.05), None, v.DEFAULT_CRITERIA["d"])
    with pytest.raises(SystemExit):
        v.criterion_d(stat(0.005, 0.002, 0.008), bad, None, v.DEFAULT_CRITERIA["d"])


@pytest.mark.parametrize("few", [{"mean_delta": 0.05}, {"mean_delta": float("nan"), "draws": [0.1]},
                                 {"mean_delta": 0.05, "draws": []}, {"mean_delta": 0.05, "draws": [0.04]},
                                 {"draws": [0.05]}, {"mean_delta": 0.05, "draws": [float("inf")]}])
def test_criterion_d_refuses_a_malformed_fewshot_result(few):
    with pytest.raises(SystemExit):
        _d(few=few)


def test_criterion_d_reads_its_thresholds_from_the_rule():
    rule = copy.deepcopy(v.DEFAULT_CRITERIA["d"])
    for p in rule["paths"]:
        if p["path"] == "abstention":
            p["delta_at_least"] = -1.0
            p["ci_lower_above"] = -1.0
    assert _d(rule=rule)["status"] == "PASS" and _d()["status"] == "PENDING"


# ---- the criteria dict (REP-01 defaults, frozen by 08-05) ----

def test_default_criteria_hold_the_rep01_defaults():
    c = v.DEFAULT_CRITERIA
    assert set(c) == {"a", "b", "c", "d", "alternative_go"}
    assert c["a"]["micro_ci_lower_min"] == -0.02 and c["a"]["micro_fail_ci_upper_below"] == -0.02
    assert c["a"]["macro_point_min"] == -0.03
    assert c["a"]["macro_ci_label"] == "not resolvable at this test size (n = 413)"
    assert c["b"]["seeds_per_model"] == 3
    assert c["c"]["status"] == "REPORTED (not gated)" == v.REPORTED
    paths = {p["path"]: p for p in c["d"]["paths"]}
    assert list(paths) == ["calibration", "abstention", "fewshot"]
    assert paths["calibration"]["metric"] == "brier"
    assert paths["calibration"]["delta_below"] == 0.0 and paths["calibration"]["ci_upper_below"] == 0.0
    assert paths["abstention"]["metric"] == "precision_at_80" and paths["abstention"]["coverage"] == 80
    assert paths["abstention"]["delta_at_least"] == 0.03 and paths["abstention"]["ci_lower_above"] == 0.0
    assert paths["fewshot"]["metric"] == "new_tag_macro_ap" and paths["fewshot"]["k"] == 10
    assert paths["fewshot"]["mean_delta_at_least"] == 0.05 and paths["fewshot"]["until_phase_9"] == "PENDING"
    assert "classwise_ece" in c["d"]["reported_only"] and "bootstrap" in c["d"]["reported_only"]["classwise_ece"]
    g = c["alternative_go"]
    assert g["metric"] == "micro_f1" and g["delta_above"] == 0.03 and g["ci_lower_above"] == 0.0
    assert g["requires"] == "b PASS"
    for k in c:
        assert c[k]["name"] and c[k]["rule"]
    assert all("points = fraction x 100" in c[k]["rule"] for k in ("a", "d", "alternative_go"))


def test_default_criteria_survive_a_json_round_trip():
    """08-05 freezes them into protocol.json and compares the loaded dict to this one."""
    assert json.loads(json.dumps(v.DEFAULT_CRITERIA)) == v.DEFAULT_CRITERIA


def test_criterion_functions_hold_no_threshold_literal():
    """Every threshold is read from the rule dict: a criterion function holds no numeric literal but 0 and 1."""
    for name in ("criterion_a", "alternative_go", "criterion_d"):
        tree = ast.parse(inspect.getsource(getattr(v, name)).lstrip())
        nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
        assert nums <= {0, 1}, (name, nums)


def test_verdict_is_numpy_and_stdlib_only():
    code = ("import sys, pilot.verdict; bad = [m for m in %r if m in sys.modules]; assert not bad, bad" % (HEAVY,))
    env = {k: val for k, val in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=pilot.data.REPO_ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
