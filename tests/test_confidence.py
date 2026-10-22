"""EVAL-03 calibration definitions and the report.py --confidence binding, offline: no model, no Ray.

The definitions in pilot.metrics are pinned to independent oracles on seeded random arrays: Brier to sklearn's
brier_score_loss and NLL to sklearn's log_loss on the flattened (message, tag) pairs, and every tag's classwise ECE
to torchmetrics' binary_calibration_error(n_bins=15, norm="l1") (probabilities drawn strictly inside (0, 1), because
torchmetrics puts an exact 1.0 into an extra bin). The bin edges, the NLL clipping, the zero-support tags in the
48-tag mean, the reliability counts and the empty-input refusals are pinned on hand-built arrays.

The EVAL-04 abstention metrics (pilot.metrics.message_confidence and selective) are pinned on hand-built arrays: the
integer coverage cut (exact rational ceiling for every n up to 300, and 104/146/167/188 of 208), stable ties in
split-file order with n_tied_at_cut, precision, accuracy, tp and fp on a 4-message toy with known confidences, the
AURC closed form of a perfect ranking, the inclusive decision rule, finiteness at the grid edges, the refusal of θ
outside (0, 1), the shared path for seed-averaged probabilities, and an AST guard that pilot/metrics.py names no
system outside its module docstring (D-10).

The EVAL-05 false-positive metrics: pilot.metrics.wilson equals scipy's binomtest(k, n).proportion_ci(method="wilson")
within 1e-12 for every k of n in {1, 7, 15, 208} (scipy in this test only), with exact 0.0 / 1.0 bounds at k = 0 /
k = n, None at n = 0 and refused non-integer or out-of-range counts; pilot.metrics.fp_rate counts only messages
without gold among the scored columns, restricts predictions and a per-tag θ to those columns, uses the inclusive θ
(np.nextafter below θ is not a prediction) and refuses an empty, unknown or repeated column list.

The --confidence binding (where T and θ come from) is exercised on synthetic dumps of the conftest snapshot: selection
candidates written like score.py's (tests/test_postfit.py write_candidate) with a postfit/1 record built by postfit.py
over them, final seeds bound by training.postfit_sha256, and prior baselines scored by score.py. Every wrong
combination is refused before a metric is computed; `import report` stays numpy only. A final seed reports on test and
hotels_probe only (_write_split_dump): it is refused on val and, under a manifest copy whose test entry holds val or
final_train content, on that content too, because final_train holds the val folds (07-REVIEW WR-01), in every report
mode: run_report (so evaluate.py, a --n subset val dump included) and --compare refuse it as well (07-09 review CR-01).

The relabelled-manifest tests (07-REVIEW CR-01) copy the snapshot's splits with val replaced by its test, HOTELS or
selection_train content (_relabel_copy, all of it or all but one row; selection_train: 07-09 review WR-01) and
monkeypatch pilot.data.DEFAULT_MANIFEST to the snapshot's manifest inside the test body, after every dump is written,
so the snapshot plays the committed data for pilot.data.holds_committed: every report mode that selects or applies
θ (run_report, --confidence val_selected and postfit_candidate, --compare) refuses such a θ source, the snapshot's own
val still passes with the same θ, and one data-marked test repeats the verifier's repro on the real committed splits.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

import pilot.data
import postfit as postfit_cli
import report
import score
from conftest import prepared
from intents import INTENTS
from pilot import dumps, metrics
from pilot.data import REPO_ROOT
from test_postfit import GRID, write_candidate

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)
TOL = 1e-12


def _draw(n=120, seed=7, lo=0.001, hi=0.999, rate=0.15):
    rng = np.random.default_rng(seed)
    return rng.random((n, 48)) < rate, rng.uniform(lo, hi, size=(n, 48))


# --- definitions against oracles -------------------------------------------------------------------------------

def test_brier_equals_sklearn_on_flattened_pairs():
    Y, P = _draw()
    assert abs(metrics.brier(Y, P) - brier_score_loss(Y.ravel(), P.ravel())) <= TOL


def test_nll_equals_sklearn_log_loss():
    Y, P = _draw()
    assert abs(metrics.nll(Y, P) - log_loss(Y.ravel(), P.ravel(), labels=[0, 1])) <= TOL


def test_nll_is_finite_at_zero_and_one():
    Y, P = _draw(n=4)
    P[0, 0], P[1, 1] = 0.0, 1.0
    Y[0, 0], Y[1, 1] = True, False                          # the worst case at both edges: clipped, not inf
    v = metrics.nll(Y, P)
    assert np.isfinite(v) and v > 0
    P2 = P.copy()
    P2[0, 0], P2[1, 1] = metrics.NLL_EPS, 1.0 - metrics.NLL_EPS
    assert v == metrics.nll(Y, P2)


def test_classwise_ece_per_tag_equals_torchmetrics():
    torch = pytest.importorskip("torch")
    from torchmetrics.functional.classification import binary_calibration_error
    Y, P = _draw(n=200, seed=11, lo=1e-6, hi=1 - 1e-6, rate=0.2)
    mean, per_tag = metrics.classwise_ece(Y, P)
    assert per_tag.shape == (48,) and per_tag.dtype == np.float64
    for j in range(48):
        want = float(binary_calibration_error(torch.tensor(P[:, j], dtype=torch.float64),
                                              torch.tensor(Y[:, j]).long(), n_bins=15, norm="l1"))
        assert abs(per_tag[j] - want) <= TOL, (TAGS[j], per_tag[j], want)
    assert mean == float(per_tag.mean())


def test_ece_bin_edges_put_one_in_the_last_bin():
    below = np.nextafter(1.0 / 15, 0.0)
    got = metrics.ece_bins(np.array([0.0, 1.0, 1.0 / 15, below, 0.5, np.nextafter(1.0, 0.0)]))
    assert got.tolist() == [0, 14, 1, 0, 7, 14]
    assert metrics.ECE_BINS == 15
    edges = np.linspace(0.0, 1.0, 16)                       # every interior edge opens its bin; just below it does not
    inner = edges[1:-1]
    assert metrics.ece_bins(inner).tolist() == list(range(1, 15))
    assert metrics.ece_bins(np.nextafter(inner, 0.0)).tolist() == list(range(0, 14))


def test_ece_averages_all_48_tags_including_zero_support():
    n = 10
    Y = np.zeros((n, 48), dtype=bool)
    P = np.full((n, 48), 0.02)
    Y[:, 0] = True
    P[:, 0] = 1.0                                           # tag 0 perfectly calibrated (bin 14, all positive)
    mean, per_tag = metrics.classwise_ece(Y, P)
    assert per_tag[0] == 0.0
    assert np.allclose(per_tag[1:], 0.02, rtol=0, atol=TOL)  # a zero-support tag's ECE is its mean p
    assert abs(mean - 47 * 0.02 / 48) <= TOL                # every tag is in the mean, none skipped
    cal = metrics.calibration(Y, P)
    assert cal["per_tag"][TAGS[1]] == {"ece": float(per_tag[1]), "support": 0}
    assert cal["per_tag"][TAGS[0]]["support"] == n


def test_reliability_bins_count_every_pair():
    Y, P = _draw(n=30, seed=3)
    P[:, :] = np.clip(P, 0.0, 0.6)                          # bins above 0.6 stay empty
    pooled = metrics.reliability(Y, P)
    assert len(pooled) == metrics.ECE_BINS
    assert sum(b["count"] for b in pooled) == 30 * 48
    assert pooled[0]["lo"] == 0.0 and pooled[-1]["hi"] == 1.0
    empty = [b for b in pooled if b["count"] == 0]
    assert empty and all(b["mean_p"] is None and b["freq"] is None for b in empty)
    for b in pooled:
        if b["count"]:
            assert b["lo"] <= b["mean_p"] < b["hi"] and 0.0 <= b["freq"] <= 1.0
    one = metrics.reliability(Y, P, tags=[TAGS[5]])
    assert sum(b["count"] for b in one) == 30
    cal = metrics.calibration(Y, P, [TAGS[5], TAGS[9]])
    assert cal["reliability"]["pooled"] == pooled
    assert sorted(cal["reliability"]["per_tag"]) == sorted([TAGS[5], TAGS[9]])
    assert cal["reliability"]["per_tag"][TAGS[5]] == one
    assert cal["n_bins"] == 15 and list(cal["per_tag"]) == TAGS


@pytest.mark.parametrize("fn", [metrics.brier, metrics.nll, lambda Y, P: metrics.classwise_ece(Y, P),
                                metrics.reliability, metrics.calibration])
def test_calibration_refuses_empty_or_misshaped_input(fn):
    with pytest.raises(SystemExit, match="no messages"):
        fn(np.zeros((0, 48), dtype=bool), np.zeros((0, 48)))
    with pytest.raises(SystemExit, match="must both be"):
        fn(np.zeros((3, 47), dtype=bool), np.zeros((3, 47)))
    with pytest.raises(SystemExit, match="must both be"):
        fn(np.zeros((3, 48), dtype=bool), np.zeros((3, 47)))


# --- abstention (EVAL-04): the one message-confidence rule, the coverage cut and AURC ---------------------------

def _rows(n, p, gold_rows=()):
    """P [n, 48] of one value p per row (a scalar or one per row) and Y with tag 0 gold on gold_rows only."""
    P = np.broadcast_to(np.asarray(p, dtype=np.float64).reshape(-1, 1), (n, 48)).copy()
    Y = np.zeros((n, 48), dtype=bool)
    Y[list(gold_rows), 0] = True
    return Y, P


@pytest.mark.parametrize("n,kept", [(100, [50, 70, 80, 90]), (208, [104, 146, 167, 188]), (7, [4, 5, 6, 7])])
def test_coverage_count_is_integer_ceiling(n, kept):
    Y, P = _draw(n=n, seed=n)
    s = metrics.selective(Y, P, 0.5)
    assert metrics.COVERAGES == (50, 70, 80, 90) and list(s["coverages"]) == ["50", "70", "80", "90"]
    assert [s["coverages"][c]["n_kept"] for c in ("50", "70", "80", "90")] == kept
    assert s["n"] == n


def test_coverage_count_is_the_exact_rational_ceiling_for_every_n():
    # a float cut drifts: ceil(70 * 0.01 * 10) is 8, not 7; the integer cut equals ceil(c * n / 100) exactly
    import math
    from fractions import Fraction
    for n in range(1, 301):
        s = metrics.selective(np.zeros((n, 48), dtype=bool), np.full((n, 48), 0.2), 0.5)
        got = [s["coverages"][str(c)]["n_kept"] for c in metrics.COVERAGES]
        assert got == [math.ceil(Fraction(c * n, 100)) for c in metrics.COVERAGES], n


def test_coverage_ties_keep_split_order_and_are_disclosed():
    # every row is the same (a baseline): one confidence, so the cut keeps the first k rows in split-file order
    n = 10
    Y, P = _rows(n, 0.2, gold_rows=range(5))               # rows 0-4 have a gold tag nobody predicts: all wrong
    s = metrics.selective(Y, P, 0.5)
    exact = np.array([0.0] * 5 + [1.0] * 5)                 # split-file order of the exact-match hits
    for c, row in s["coverages"].items():
        k = row["n_kept"]
        assert row["n_tied_at_cut"] == n
        assert row["accuracy"] == exact[:k].mean(), c
    assert s["coverages"]["50"]["accuracy"] == 0.0           # the first five, never the five right ones
    assert s["aurc"] == np.mean(1.0 - np.cumsum(exact) / np.arange(1, n + 1))


def _toy():
    """4 messages with known confidences at θ 0.5, most confident first (rows 0, 1, 2, 3)."""
    P = np.full((4, 48), 0.01)
    P[0, 0] = 0.99                                          # predicts {0}, gold {0}: right
    P[1, 1] = 0.9                                           # predicts {1}, gold {2}: wrong, one fp
    P[2] = 0.02                                             # predicts nothing, gold nothing: right
    P[3, 0], P[3, 1] = 0.6, 0.7                             # predicts {0, 1}, gold {0}: wrong, one tp and one fp
    Y = np.zeros((4, 48), dtype=bool)
    Y[0, 0] = Y[1, 2] = Y[3, 0] = True
    c = np.array([48 * np.log(0.99), 47 * np.log(0.99) + np.log(0.9), 48 * np.log(0.98),
                  46 * np.log(0.99) + np.log(0.6) + np.log(0.7)])
    return Y, P, c


def test_selective_precision_and_accuracy_on_toy():
    Y, P, c = _toy()
    got, pred = metrics.message_confidence(P, 0.5)
    assert np.allclose(got, c, rtol=0, atol=1e-12) and (np.diff(c) < 0).all()
    assert (pred == (P >= 0.5)).all()
    s = metrics.selective(Y, P, 0.5, coverages=(25, 50, 75, 100))
    want = {"25": (1, 1, 0, 1.0, 1.0), "50": (2, 1, 1, 0.5, 0.5), "75": (3, 1, 1, 0.5, 2 / 3),
            "100": (4, 2, 2, 0.5, 0.5)}
    for cov, (k, tp, fp, prec, acc) in want.items():
        row = s["coverages"][cov]
        assert (row["n_kept"], row["tp"], row["fp"], row["n_tied_at_cut"]) == (k, tp, fp, 1), cov
        assert row["precision"] == prec and abs(row["accuracy"] - acc) <= TOL, cov
    assert abs(s["aurc"] - (0 + 0.5 + 1 / 3 + 0.5) / 4) <= TOL
    # the kept message predicts no tag: 0/0 precision is 0.0, and it is right (exact match on a no-intent row)
    one = metrics.selective(Y[[2]], P[[2]], 0.5, coverages=(100,))["coverages"]["100"]
    assert (one["tp"], one["fp"], one["precision"], one["accuracy"]) == (0, 0, 0.0, 1.0)


@pytest.mark.parametrize("n,m", [(10, 3), (9, 0), (12, 12), (208, 61)])
def test_aurc_of_a_perfect_ranking_has_closed_form(n, m):
    # confidence falls with the row (p rises toward θ, nothing is predicted); the m right messages come first
    p = np.linspace(0.01, 0.4, n)
    Y, P = _rows(n, p, gold_rows=range(m, n))
    k = np.arange(1, n + 1)
    closed = float(np.mean(np.maximum(0, k - m) / k))
    assert abs(metrics.selective(Y, P, 0.5)["aurc"] - closed) <= TOL
    Yr, Pr = _rows(n, p, gold_rows=range(0, n - m))         # the same messages, the right ones ranked last
    worse = metrics.selective(Yr, Pr, 0.5)["aurc"]
    assert worse > closed if 0 < m < n else worse == closed


def test_aurc_is_zero_when_every_message_is_right():
    _, P = _draw(n=50, seed=3)
    for theta in (0.5, 0.05, np.linspace(0.1, 0.9, 48)):
        s = metrics.selective(P >= np.asarray(theta), P, theta)
        assert s["aurc"] == 0.0
        assert all(r["accuracy"] == 1.0 for r in s["coverages"].values())


def test_confidence_is_finite_at_grid_edges_and_inclusive():
    vals = np.array([1e-9, 0.5, 1 - 1e-9])
    P = np.resize(vals, (6, 48))                            # every value in every row, in shifting positions
    P[3:] = vals[:, None].repeat(48, axis=1)                # and rows of one value each
    for theta in (0.05, 0.5, 0.95, np.linspace(0.05, 0.95, 48)):
        c, _ = metrics.message_confidence(P, theta)
        assert c.shape == (6,) and np.isfinite(c).all(), theta
    c, pred = metrics.message_confidence(np.full((2, 48), 0.3), 0.3)
    assert pred.all() and np.allclose(c, 48 * np.log(0.3), rtol=0, atol=1e-12)   # P == θ counts as predicted
    c, pred = metrics.message_confidence(np.full((1, 48), 0.95), metrics.GRID[-1])
    assert pred.all() and np.isfinite(c).all()


@pytest.mark.parametrize("theta", [0.0, 1.0, float("nan"), -0.1, 1.5, float("inf"),
                                   np.r_[np.full(47, 0.5), 0.0], np.r_[np.full(47, 0.5), np.nan]])
def test_confidence_refuses_theta_outside_open_unit_interval(theta):
    Y, P = _draw(n=5)
    with pytest.raises(SystemExit, match="strictly between 0 and 1"):
        metrics.message_confidence(P, theta)
    with pytest.raises(SystemExit, match="strictly between 0 and 1"):
        metrics.selective(Y, P, theta)


def test_confidence_rule_serves_seed_averaged_probabilities():
    Y, P1 = _draw(n=40, seed=11)
    _, P2 = _draw(n=40, seed=12)
    avg = metrics.seed_average([P1, P2])
    for theta in (0.5, np.linspace(0.2, 0.8, 48)):
        c, pred = metrics.message_confidence(avg, theta)
        c_mean, pred_mean = metrics.message_confidence(np.mean([P1, P2], axis=0), theta)
        assert (c == c_mean).all() and (pred == pred_mean).all()
        assert metrics.selective(Y, avg, theta) == metrics.selective(Y, np.mean([P1, P2], axis=0), theta)
    assert (metrics.message_confidence(metrics.seed_average([P1]), 0.5)[0]
            == metrics.message_confidence(P1, 0.5)[0]).all()


def test_selective_refuses_empty():
    with pytest.raises(SystemExit, match="no messages"):
        metrics.selective(np.zeros((0, 48), dtype=bool), np.zeros((0, 48)), 0.5)
    with pytest.raises(SystemExit, match="must both be"):
        metrics.selective(np.zeros((3, 47), dtype=bool), np.zeros((3, 47)), 0.5)
    with pytest.raises(SystemExit, match="non-empty"):
        metrics.message_confidence(np.zeros((0, 48)), 0.5)
    with pytest.raises(SystemExit, match="theta must be one float"):
        metrics.message_confidence(np.full((2, 48), 0.5), np.full(47, 0.5))


def test_metrics_module_names_no_system():
    """D-10: no code in pilot/metrics.py names a system; only the module docstring may (it lists them)."""
    import ast
    path = os.path.join(REPO_ROOT, "pilot", "metrics.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    doc = tree.body[0]
    assert isinstance(doc, ast.Expr) and "laya" in doc.value.value.lower()   # the guard can see a name
    words = []
    for node in tree.body[1:]:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                words.append(sub.id)
            elif isinstance(sub, ast.Attribute):
                words.append(sub.attr)
            elif isinstance(sub, ast.arg):
                words.append(sub.arg)
            elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                words.append(sub.name)
            elif isinstance(sub, ast.alias):
                words += [sub.name, sub.asname or ""]
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                words.append(getattr(sub, "module", None) or "")
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                words.append(sub.value)
    assert len(words) > 100
    named = [w for w in words if "laya" in w.lower() or "berturk" in w.lower()]
    assert named == []


# --- EVAL-05: Wilson interval and the false-positive rule ------------------------------------------------------

@pytest.mark.parametrize("n", [1, 7, 15, 208])
def test_wilson_equals_scipy_on_a_grid(n):
    stats = pytest.importorskip("scipy.stats")
    assert metrics.WILSON_Z == float(stats.norm.ppf(0.975))
    for k in range(n + 1):
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        lo, hi = metrics.wilson(k, n)
        assert type(lo) is float and type(hi) is float     # unrounded float64, never a numpy scalar in JSON
        assert abs(lo - ci.low) <= TOL and abs(hi - ci.high) <= TOL, (k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
    lo, hi = metrics.wilson(2, 7)
    assert abs(lo - 0.0822) < 5e-5 and abs(hi - 0.6411) < 5e-5   # 2/7 reads as 29 %, its interval spans 8 % to 64 %


@pytest.mark.parametrize("n", [1, 2, 7, 15, 208, 1009])
def test_wilson_extremes_are_exact(n):
    assert metrics.wilson(0, n)[0] == 0.0 and metrics.wilson(0, n)[1] > 0.0
    assert metrics.wilson(n, n)[1] == 1.0 and metrics.wilson(n, n)[0] < 1.0
    assert metrics.wilson(np.int64(0), n)[0] == 0.0            # a numpy count from a .sum() is an integer too


def test_wilson_of_zero_trials_is_none_and_bad_counts_refused():
    assert metrics.wilson(0, 0) is None
    for k, n in [(-1, 5), (6, 5), (1.5, 5), (2.0, 5), (True, 5), ("1", 5)]:
        with pytest.raises(SystemExit, match="not an integer in 0..5"):
            metrics.wilson(k, n)


def _fp_arrays(n=6, p=0.1):
    return np.zeros((n, 48), dtype=bool), np.full((n, 48), p)


def test_fp_rate_counts_only_messages_without_gold():
    Y, P = _fp_arrays()
    Y[0, 0] = True                                          # gold message predicting another tag: not counted
    P[0, 5] = 0.9
    Y[1, 2] = True                                          # gold message predicting nothing: not counted
    P[2, 7] = P[2, 9] = 0.8                                 # no-gold message predicting two tags: one false positive
    P[3, 9] = 0.6                                           # no-gold message predicting one tag
    out = metrics.fp_rate(Y, P, 0.5)                        # messages 2..5 are counted, 2 and 3 are false positives
    assert (out["k"], out["n"], out["cols"]) == (2, 4, "all")
    assert out["rate"] == 0.5 and out["wilson95"] == metrics.wilson(2, 4)
    assert out["per_tag_fp"] == {TAGS[7]: 1, TAGS[9]: 2}     # nonzero counts only, the gold message's tag 5 absent
    assert list(out["per_tag_fp"]) == [TAGS[7], TAGS[9]]
    Y[2:, 1] = True                                         # every message has a gold tag: nothing is counted
    none = metrics.fp_rate(Y, P, 0.5)
    assert (none["k"], none["n"], none["rate"], none["wilson95"], none["per_tag_fp"]) == (0, 0, None, None, {})
    clean = metrics.fp_rate(*_fp_arrays(), 0.5)
    assert (clean["k"], clean["n"], clean["rate"]) == (0, 6, 0.0) and clean["wilson95"][0] == 0.0


def test_fp_rate_restricts_predictions_and_theta_to_cols():
    Y, P = _fp_arrays(n=4)
    cols = [TAGS[10], TAGS[3]]                              # given out of INTENTS order
    Y[0, 20] = True                                         # gold outside cols: still counted
    P[0, 3] = 0.7                                           # ... and its cols prediction is a false positive
    P[1, 30] = 0.99                                         # a prediction outside cols is not a false positive
    P[2, 10] = 0.4                                          # predicted only under its own per-tag θ 0.3
    Y[3, 10] = True                                         # gold inside cols: not counted
    out = metrics.fp_rate(Y, P, 0.5, cols)
    assert (out["k"], out["n"], out["cols"]) == (1, 3, cols)
    assert out["per_tag_fp"] == {TAGS[3]: 1}
    th = np.full(48, 0.5)
    th[10], th[30] = 0.3, 0.01                              # a per-tag θ outside cols plays no part
    out = metrics.fp_rate(Y, P, th, cols)
    assert (out["k"], out["n"]) == (2, 3) and list(out["per_tag_fp"]) == [TAGS[3], TAGS[10]]
    full = metrics.fp_rate(Y, P, th)                        # all 48 columns: messages 0 and 3 hold gold, so
    assert (full["k"], full["n"]) == (2, 2) and full["per_tag_fp"][TAGS[30]] == 2   # tag 30 fires on both counted


def test_fp_rate_threshold_is_inclusive():
    for theta in (0.5, 0.19, 0.95, 0.05):
        Y, P = _fp_arrays(n=2, p=0.0)
        P[0, 4] = theta                                     # exactly θ: a prediction
        P[1, 4] = np.nextafter(theta, 0.0)                  # the next float below θ: not a prediction
        out = metrics.fp_rate(Y, P, theta)
        assert (out["k"], out["n"], out["per_tag_fp"]) == (1, 2, {TAGS[4]: 1}), theta
        out = metrics.fp_rate(Y, P, np.full(48, theta), [TAGS[4]])
        assert (out["k"], out["n"]) == (1, 2), theta


def test_fp_rate_refuses_bad_cols():
    Y, P = _fp_arrays()
    for cols in ([], ["no_such_tag"], [TAGS[0], TAGS[0]], [TAGS[1], "x"]):
        with pytest.raises(SystemExit, match="distinct INTENTS tags"):
            metrics.fp_rate(Y, P, 0.5, cols)
    with pytest.raises(SystemExit, match="theta must be"):
        metrics.fp_rate(Y, P, np.full(47, 0.5))
    with pytest.raises(SystemExit, match="no messages"):
        metrics.fp_rate(Y[:0], P[:0], 0.5)


# --- report.py --confidence: where T and θ come from -----------------------------------------------------------

@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path, val entry, val uids) of the synthetic snapshot prepared under tmp_path."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    path = os.path.join(out_dir, "splits", "val.jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=("val",), data=raw)
    return out_dir, manifest, entry, [r["uid"] for r in pilot.data.parse_jsonl(raw)]


@pytest.fixture
def fitted(tmp_path, snap, capsys):
    """(candidate run dirs, record path, record dict, record sha256): a postfit/1 record over a synthetic grid."""
    run_dirs = [write_candidate(tmp_path / "dumps", snap, lr, e) for lr, e in GRID]
    out = tmp_path / "postfit" / "laya.json"
    postfit_cli.main(run_dirs + ["--manifest", snap[1], "--out", str(out)])
    capsys.readouterr()
    return run_dirs, str(out), json.loads(out.read_text(encoding="utf-8")), pilot.data.sha256_file(str(out))


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _cand(record, run_dir):
    rid = os.path.basename(run_dir)
    return [c for c in record["candidates"] if c["dump_run_id"] == rid][0]


def test_confidence_binding_selection_candidate_uses_record_t_and_theta(snap, fitted):
    run_dirs, path, record, sha = fitted
    for d in run_dirs:
        rep = report.confidence_report(d, "val", snap[1], path)
        c = _cand(record, d)
        head, raw = rep["variants"]["headline"], rep["variants"]["raw"]
        assert rep["binding"]["kind"] == "postfit_candidate" and rep["schema"] == report.CONFIDENCE_SCHEMA
        assert rep["binding"]["postfit"]["sha256"] == sha and rep["binding"]["postfit"]["system_kind"] == "laya"
        assert head["temperature"] == c["temperature"] and head["theta"] == c["theta"]["shipped"]
        assert raw["temperature"] == 1.0 and raw["theta"] == c["theta"]["raw"]
        assert rep["in_sample"] is True and rep["caveat"] == report.IN_SAMPLE_CAVEAT
        assert abs(head["calibration"]["nll"] - c["nll_fitted"]) <= 1e-8
        assert abs(raw["calibration"]["nll"] - c["nll_raw"]) <= 1e-8
        assert rep["definitions"] == {"calibration": metrics.CALIBRATION_RULES, "confidence": metrics.CONFIDENCE_RULE,
                                      "coverage": metrics.COVERAGE_RULE, "aurc": metrics.AURC_RULE,
                                      "false_positive": metrics.FP_RULE, "wilson": report.WILSON_TEXT}
        assert sorted(head["calibration"]["reliability"]["per_tag"]) == sorted(c["theta"]["shipped"]["eligible_tags"])


def test_confidence_binding_refuses_without_postfit(snap, fitted):
    with pytest.raises(SystemExit, match="ships T = 1; pass --postfit"):
        report.confidence_report(fitted[0][0], "val", snap[1])


def test_confidence_binding_refuses_other_kind_record(tmp_path, snap, fitted):
    other = dict(fitted[2], system_kind="berturk")
    path = _write_json(tmp_path / "other" / "berturk.json", other)
    with pytest.raises(SystemExit, match="is a berturk record, the dump is laya"):
        report.confidence_report(fitted[0][0], "val", snap[1], path)


def test_confidence_binding_refuses_dump_not_in_record(tmp_path, snap, fitted):
    stray = write_candidate(tmp_path / "stray", snap, 2e-05, 3, rng_seed=999)   # same identity, other logits
    with pytest.raises(SystemExit, match="lists 0 candidate"):
        report.confidence_report(stray, "val", snap[1], fitted[1])


def test_confidence_binding_refuses_non_postfit_schema(tmp_path, snap, fitted):
    wrong = _write_json(tmp_path / "x" / "conf.json", {"schema": "confidence/1"})
    for path in (wrong, _write_json(tmp_path / "x" / "list.json", [1, 2])):
        with pytest.raises(SystemExit, match="not a postfit/1 record"):
            report.confidence_report(fitted[0][0], "val", snap[1], path)
    broken = tmp_path / "x" / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a postfit/1 record"):
        report.confidence_report(fitted[0][0], "val", snap[1], str(broken))
    with pytest.raises(SystemExit, match="not found"):
        report.confidence_report(fitted[0][0], "val", snap[1], str(tmp_path / "missing.json"))


def test_confidence_binding_refuses_candidate_off_val(snap, fitted):
    d = fitted[0][0]
    _, _, side, _ = dumps.load_dump(d, "val", snap[1])
    with pytest.raises(SystemExit, match="reported on val only"):
        report.confidence_binding(d, "test", side, snap[1], fitted[1])


def _write_split_dump(run_dir, manifest, split, rng_seed=5):
    """A dump of `split` in a run dir that holds a val dump, of the same source (the tests/test_ood.py write_hotels_run
    recipe): score.sidecar_base of the verified split with the val sidecar's run_id, system, temperature, scorer, seed
    and training, and seeded float32 logits. Returns the run dir."""
    _, _, vside, _ = dumps.load_dump(run_dir, "val", manifest)
    path = os.path.join(os.path.dirname(manifest), "splits", "%s.jsonl" % split)
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=dumps.READ_ROLES, data=raw)
    uids = [r["uid"] for r in pilot.data.parse_jsonl(raw)]
    side = score.sidecar_base(entry, split, manifest, None)
    side.update({k: vside[k] for k in ("run_id", "system", "temperature", "scorer", "seed", "training")})
    logits = np.random.default_rng(rng_seed).normal(0.0, 3.0, size=(len(uids), 48)).astype(np.float32)
    dumps.write_dump(run_dir, split, logits, uids, side)
    return run_dir


def _final_seed(root, snap, record, sha, shipped=None, epoch=None):
    return write_candidate(root, snap, 2e-05, record["chosen"]["epoch"] if epoch is None else epoch,
                           split="final_train", calibrated=True,
                           shipped=record["chosen"]["temperature"] if shipped is None else shipped, postfit_sha256=sha)


def test_confidence_binding_final_seed_by_postfit_sha256(tmp_path, snap, fitted):
    """A final seed reports on test and hotels_probe (Phase 8's paths) with its record's T and θ, held-out."""
    _, path, record, sha = fitted
    T = record["chosen"]["temperature"]
    final = _write_split_dump(_final_seed(tmp_path / "final", snap, record, sha), snap[1], "test")
    rep = report.confidence_report(final, "test", snap[1], path)
    assert rep["binding"]["kind"] == "final_seed"
    assert rep["binding"]["theta_source"]["postfit_sha256"] == sha
    assert rep["variants"]["headline"]["temperature"] == T
    assert rep["variants"]["headline"]["theta"] == record["chosen"]["theta"]["shipped"]
    assert rep["variants"]["raw"]["temperature"] == 1.0
    assert rep["variants"]["raw"]["theta"] == record["chosen"]["theta"]["raw"]
    assert rep["in_sample"] is False and rep["caveat"] is None
    with pytest.raises(SystemExit, match="final seed .* is reported only with --postfit"):
        report.confidence_report(final, "test", snap[1])
    _write_split_dump(final, snap[1], "hotels_probe")
    ood = report.confidence_report(final, "hotels_probe", snap[1], path)
    assert ood["binding"]["kind"] == "final_seed" and ood["in_sample"] is False
    assert ood["variants"]["headline"]["theta"] == record["chosen"]["theta"]["shipped"]
    for v in ood["variants"].values():
        assert set(v["ood_fp"]) == {"0.5", "theta"} and "calibration" not in v


def test_confidence_binding_final_seed_refuses_other_record_or_t(tmp_path, snap, fitted):
    _, path, record, sha = fitted
    T = record["chosen"]["temperature"]
    final = _write_split_dump(_final_seed(tmp_path / "final", snap, record, sha, epoch=3), snap[1], "test")
    copy = tmp_path / "copy.json"
    copy.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")   # same content, other bytes
    copy = str(copy)
    assert pilot.data.sha256_file(copy) != sha
    with pytest.raises(SystemExit, match="final seed was bound to record"):
        report.confidence_report(final, "test", snap[1], copy)
    off = _write_split_dump(_final_seed(tmp_path / "off", snap, record, sha, shipped=T + 0.5, epoch=3), snap[1],
                            "test")
    with pytest.raises(SystemExit, match="record's chosen T"):
        report.confidence_report(off, "test", snap[1], path)


def test_final_seed_is_never_reported_on_val(tmp_path, snap, fitted):
    """07-REVIEW WR-01: a final seed trained on final_train, which holds the val folds; its val numbers are
    training-set numbers, refused with or without --postfit (before the record is read)."""
    _, path, record, sha = fitted
    final = _final_seed(tmp_path / "final", snap, record, sha)
    for args in ((path,), ()):
        with pytest.raises(SystemExit, match="trained on final_train, which holds the val folds"):
            report.confidence_report(final, "val", snap[1], *args)


@pytest.mark.parametrize("source, held", [("val", "val"), ("selection_train", "final_train")])
def test_final_seed_refuses_training_content_under_any_manifest(tmp_path, snap, fitted, monkeypatch, source, held):
    """07-REVIEW WR-01, keyed on the content: a manifest copy whose test entry holds the val content, or content of
    final_train (selection_train's rows are final_train's), cannot present a final seed's training-set numbers as
    held-out test numbers; refused before the postfit record is read."""
    _, path, record, sha = fitted
    rsnap = _relabel_copy(snap[0], snap[1], tmp_path / "r", "test", source)
    final = _write_split_dump(_final_seed(tmp_path / "final", rsnap, record, sha), rsnap[1], "test")
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", snap[1])     # after every dump: see the CR-01 tests
    with pytest.raises(SystemExit, match="holds committed %s content" % held):
        report.confidence_report(final, "test", rsnap[1], path)


def _drop_val(run_dir):
    for ext in ("npz", "json"):
        os.remove(os.path.join(run_dir, "val." + ext))
    return run_dir


@pytest.mark.parametrize("source, held", [("val", "val"), ("selection_train", "final_train")])
def test_final_seed_run_report_refuses_training_content(tmp_path, snap, fitted, monkeypatch, source, held):
    """07-09 review CR-01: the final-seed refusal holds in report.py RUN_DIR and in evaluate.py (which prints
    run_report), not only in --confidence. A final seed's run dir without a val dump (its val record is refused, so
    Phase 8's run dirs have none) and a test dump of a manifest copy whose test entry holds val or final_train content
    is refused, never reported at θ 0.5 under the test name; the snapshot's own test still reports, held-out."""
    import evaluate
    _, _, record, sha = fitted
    rsnap = _relabel_copy(snap[0], snap[1], tmp_path / "r", "test", source)
    final = _drop_val(_write_split_dump(_final_seed(tmp_path / "final", rsnap, record, sha), rsnap[1], "test"))
    ok = _drop_val(_write_split_dump(_final_seed(tmp_path / "ok", snap, record, sha), snap[1], "test"))
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", snap[1])     # after every dump: see the CR-01 tests
    want = "holds committed %s content" % held
    with pytest.raises(SystemExit, match=want):
        report.run_report(final, "test", rsnap[1])
    monkeypatch.setattr(evaluate.score, "main", lambda argv: (final, "test"))     # the dump score.py wrote
    with pytest.raises(SystemExit, match=want):
        evaluate.main([os.path.join(rsnap[0], "splits", "test.jsonl"), "--manifest", rsnap[1]])
    rep = report.run_report(ok, "test", snap[1])
    assert rep["split"]["name"] == "test" and rep["theta_source"] is None
    assert rep["variants"]["shipped"]["val_selected_reason"] == report.NO_VAL_REASON


def test_final_seed_subset_val_dump_is_never_reported(tmp_path, snap, fitted):
    """07-09 review CR-01, the second path: run_report reports a --n subset val dump on val without a θ source
    (SUBSET_VAL_REASON), so refuse_in_sample_theta never sees it; a final seed's subset val dump is refused by name."""
    _, _, record, sha = fitted
    sub = write_candidate(tmp_path / "sub", snap, 2e-05, record["chosen"]["epoch"], split="final_train",
                          calibrated=True, shipped=record["chosen"]["temperature"], postfit_sha256=sha, n=5)
    with pytest.raises(SystemExit, match="trained on final_train, which holds the val folds"):
        report.run_report(sub, "val", snap[1])


def test_compare_refuses_a_final_seed_dump_on_training_content(tmp_path, snap, fitted, monkeypatch):
    """07-09 review CR-01 in --compare: _load_side refuses the compared dump itself when it is a final seed's on
    training content, so the rule holds even next to a val dump that is not a final seed's (a planted sidecar: every
    real final seed is already refused there as an in-sample θ source); run_report refuses the same dir."""
    _, _, record, sha = fitted
    rsnap = _relabel_copy(snap[0], snap[1], tmp_path / "r", "test", "val")
    d = _write_split_dump(write_candidate(tmp_path / "x", rsnap, 2e-05, 1), rsnap[1], "test")
    logits, uids, side, _ = dumps.load_dump(d, "test", rsnap[1])
    planted = {k: v for k, v in side.items() if k not in ("dump_sha256", "schema_version")}
    planted["training"] = dict(side["training"], postfit_sha256=sha)
    dumps.write_dump(d, "test", logits, uids, planted)
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", snap[1])
    for call in (lambda: report.compare([d], [d], "test", rsnap[1], B=10),
                 lambda: report.run_report(d, "test", rsnap[1])):
        with pytest.raises(SystemExit, match="holds committed val content"):
            call()


def _prior(snap, root, splits=("val", "test"), model="prior"):
    out_dir, manifest = snap[0], snap[1]
    for name in splits:
        score.main([os.path.join(out_dir, "splits", "%s.jsonl" % name), "--model", model, "--manifest", manifest,
                    "--out-root", str(root)])
    return os.path.join(str(root), model)


def test_confidence_binding_refuses_postfit_for_baseline(tmp_path, snap, fitted, capsys):
    d = _prior(snap, tmp_path / "base")
    capsys.readouterr()
    with pytest.raises(SystemExit, match="no training block"):
        report.confidence_report(d, "val", snap[1], fitted[1])


def test_confidence_binding_baseline_selects_theta_on_val(tmp_path, snap, capsys):
    d = _prior(snap, tmp_path / "base")
    capsys.readouterr()
    rep = report.confidence_report(d, "val", snap[1])
    run = report.run_report(d, "val", snap[1])
    assert rep["binding"]["kind"] == "val_selected" and rep["binding"]["postfit"] is None
    assert rep["binding"]["theta_source"] == run["theta_source"]
    assert rep["variants"]["headline"]["theta"] == run["variants"]["shipped"]["rows"]["val_selected"]["theta"]
    assert rep["variants"]["raw"]["theta"] == run["variants"]["raw"]["rows"]["val_selected"]["theta"]
    assert rep["in_sample"] is True and rep["training"] is None
    test = report.confidence_report(d, "test", snap[1])                     # θ from val, applied to test
    assert test["in_sample"] is False and test["caveat"] is None
    assert test["variants"]["headline"]["theta"] == rep["variants"]["headline"]["theta"]


def test_confidence_refuses_theta_source_off_val(tmp_path, snap, capsys):
    only_test = _prior(snap, tmp_path / "t", splits=("test",))
    capsys.readouterr()
    with pytest.raises(SystemExit, match="no full val dump in .*θ is chosen on val only"):
        report.confidence_report(only_test, "test", snap[1])
    d = _prior(snap, tmp_path / "s")
    capsys.readouterr()
    _, uids, side, _ = dumps.load_dump(d, "val", snap[1])
    logits = dumps.load_dump(d, "val", snap[1])[0]
    planted = {k: v for k, v in side.items() if k not in ("dump_sha256", "schema_version")}
    planted["temperature"] = dict(side["temperature"], shipped=2.0)       # another source's val dump
    dumps.write_dump(d, "val", logits, uids, planted)
    with pytest.raises(SystemExit, match="differ in temperature.shipped"):
        report.confidence_report(d, "test", snap[1])


def test_confidence_refuses_subset_dump(tmp_path, snap, capsys):
    out_dir, manifest = snap[0], snap[1]
    run_dir, _ = score.main([os.path.join(out_dir, "splits", "val.jsonl"), "--model", "prior", "--manifest", manifest,
                             "--out-root", str(tmp_path / "sub"), "--n", "5"])
    capsys.readouterr()
    with pytest.raises(SystemExit, match="--n subset dump is never reported"):
        report.confidence_report(run_dir, "val", manifest)


# --- 07-REVIEW CR-01: θ is selected on banking val content only, under any manifest ------------------------------

def _relabel_copy(src_dir, src_manifest, dest, name, source, cut=0):
    """A snap-shaped copy of src_dir/splits under dest whose `name` split holds src_dir/splits/<source>.jsonl minus its
    first `cut` lines, with that entry's sha256, n_messages and uids_sha256 updated in dest/splits.json (the
    tests/test_consumers.py _relabel recipe). Returns (dest dir, dest manifest, verified dest val entry, val uids)."""
    import shutil
    dest = str(dest)
    shutil.copytree(os.path.join(str(src_dir), "splits"), os.path.join(dest, "splits"))
    with open(os.path.join(str(src_dir), "splits", "%s.jsonl" % source), "rb") as f:
        raw = b"".join(f.read().splitlines(keepends=True)[cut:])
    with open(os.path.join(dest, "splits", "%s.jsonl" % name), "wb") as f:
        f.write(raw)
    with open(src_manifest, encoding="utf-8") as f:
        m = json.load(f)
    rows = pilot.data.parse_jsonl(raw)
    m["splits"][name].update(sha256=pilot.data.sha256_bytes(raw), n_messages=len(rows),
                             uids_sha256=pilot.data.sha256_bytes("\n".join(r["uid"] for r in rows).encode("utf-8")))
    manifest = os.path.join(dest, "splits.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    path = os.path.join(dest, "splits", "val.jsonl")
    with open(path, "rb") as f:
        val = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=("val",), data=val)
    return dest, manifest, entry, [r["uid"] for r in pilot.data.parse_jsonl(val)]


@pytest.mark.parametrize("held", ["test", "hotels_probe", "selection_train"])
@pytest.mark.parametrize("cut", [0, 1])
def test_heldout_theta_source_refused_in_every_report_mode(tmp_path, snap, monkeypatch, capsys, held, cut):
    """07-REVIEW CR-01: val relabelled to the snapshot's test or HOTELS content (cut 0: the entry's hashes; cut 1: the
    (uid, message) pairs) is refused as the θ source by run_report, --confidence on val, test and hotels_probe
    (val_selected) and --compare, keyed on the content whatever the manifest's role says; so is val relabelled to
    selection_train content, the training messages of the baselines and the selection candidates (07-09 review
    WR-01)."""
    from test_ood import write_hotels_run
    rsnap = _relabel_copy(snap[0], snap[1], tmp_path / "r", "val", held, cut)
    prior = _prior(rsnap, tmp_path / "d")
    majority = _prior(rsnap, tmp_path / "d", model="majority")
    hotels, _ = write_hotels_run(tmp_path / "h", rsnap[:2])
    capsys.readouterr()
    # in the body, after every dump is written: with the patch active the snapshot plays the committed data
    # (pilot.data.committed_content reads DEFAULT_MANIFEST at call time), and a fixture or score.py run would see it too
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", snap[1])
    want = "holds committed %s content" % held
    for call in (lambda: report.run_report(prior, "val", rsnap[1]),
                 lambda: report.run_report(prior, "test", rsnap[1]),
                 lambda: report.confidence_report(prior, "val", rsnap[1]),
                 lambda: report.confidence_report(prior, "test", rsnap[1]),
                 lambda: report.confidence_report(hotels, "hotels_probe", rsnap[1]),
                 lambda: report.compare([prior], [majority], "test", rsnap[1], B=10)):
        with pytest.raises(SystemExit, match=want):
            call()


@pytest.mark.parametrize("held", ["test", "hotels_probe", "selection_train"])
def test_heldout_theta_source_refused_for_a_postfit_candidate(tmp_path, snap, monkeypatch, capsys, held):
    """07-REVIEW CR-01, the postfit_candidate binding: a candidate's val split is the one its record's T and θ were
    fitted on, so a val split holding test or HOTELS content is refused there as well, and so is one holding
    selection_train content, the candidates' own training messages, never in-sample val (07-09 review WR-01)."""
    rsnap = _relabel_copy(snap[0], snap[1], tmp_path / "r", "val", held)
    run_dirs = [write_candidate(tmp_path / "dumps", rsnap, lr, e) for lr, e in GRID]
    out = tmp_path / "pf" / "laya.json"
    postfit_cli.main(run_dirs + ["--manifest", rsnap[1], "--out", str(out)])
    capsys.readouterr()
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", snap[1])     # after postfit.py: see the test above
    with pytest.raises(SystemExit, match="holds committed %s content" % held):
        report.confidence_report(run_dirs[0], "val", rsnap[1], str(out))


def test_heldout_theta_check_passes_the_committed_val(tmp_path, snap, monkeypatch, capsys):
    """No false positive: with the snapshot counted as committed, its own val is a θ source, and run_report,
    --confidence on test and --compare give the same θ as without the patch."""
    prior = _prior(snap, tmp_path / "d")
    majority = _prior(snap, tmp_path / "d", model="majority")
    capsys.readouterr()

    def thetas():
        run = report.run_report(prior, "test", snap[1])
        conf = report.confidence_report(prior, "test", snap[1])
        return ([run["theta_source"]] + [v["rows"]["val_selected"]["theta"] for v in run["variants"].values()],
                [conf["binding"]["theta_source"]] + [v["theta"] for v in conf["variants"].values()])

    before = thetas()
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", snap[1])
    assert pilot.data.holds_committed(snap[2], [], "val")    # the patch is live: the snapshot's val is committed
    assert thetas() == before
    c = report.compare([prior], [majority], "test", snap[1], B=10)
    assert c["a"]["theta"] == before[0][1]


@pytest.mark.data
@pytest.mark.parametrize("held", ["test", "hotels_probe"])
def test_heldout_theta_source_refused_on_the_committed_data(tmp_path, capsys, held):
    """The verifier's CR-01 repro on the real committed splits, no monkeypatch: val relabelled to the committed test or
    HOTELS bytes is refused as the θ source by --confidence and by run_report on test."""
    data = os.path.join(REPO_ROOT, "data")
    if not all(os.path.isfile(os.path.join(data, "splits", "%s.jsonl" % s["name"])) for s in pilot.data.SPLITS):
        pytest.fail("data/splits/*.jsonl missing; run `make data`")
    rsnap = _relabel_copy(data, pilot.data.DEFAULT_MANIFEST, tmp_path / "r", "val", held)
    prior = _prior(rsnap, tmp_path / "d")
    capsys.readouterr()
    want = "holds committed %s content" % held
    with pytest.raises(SystemExit, match=want):
        report.confidence_report(prior, "test", rsnap[1])
    with pytest.raises(SystemExit, match=want):
        report.run_report(prior, "test", rsnap[1])


def test_confidence_out_writes_the_svg_next_to_the_json(tmp_path, snap, capsys):
    d = _prior(snap, tmp_path / "base")
    out = tmp_path / "conf" / "prior-val.json"
    report.main(["--confidence", d, "--split", "val", "--manifest", snap[1], "--out", str(out)])
    capsys.readouterr()
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert (tmp_path / "conf" / "prior-val.svg").read_bytes() == report.render_svg(rec).encode("utf-8")
    md = tmp_path / "conf.md"
    text = report.main(["--confidence-table", str(tmp_path / "conf"), "--out", str(md)])
    assert capsys.readouterr().out == text == report.render_confidence([rec])
    assert md.read_bytes() == text.encode("utf-8")
    assert "![reliability diagram prior (val)](results/confidence/prior-val.svg)" in text
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence", d, "--split", "val", "--manifest", snap[1], "--out", str(tmp_path / "x.svg")])
    assert e.value.code == 2 and "--out must name a .json file" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence-table", str(tmp_path / "conf"), "--table", str(tmp_path / "conf")])
    assert e.value.code == 2 and "exactly one of" in capsys.readouterr().err


def test_confidence_argument_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        report.main(["--postfit", "x.json", "results/dumps/prior"])
    assert e.value.code == 2 and "--postfit applies only to --confidence" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence", "results/dumps/prior"])
    assert e.value.code == 2 and "--confidence needs an explicit --split" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence", "results/dumps/prior", "results/dumps/prior", "--split", "val"])
    assert e.value.code == 2 and "exactly one of" in capsys.readouterr().err


def test_report_stays_numpy_only():
    banned = ("torch", "scipy", "laya", "ray", "transformers", "sklearn")
    code = "import sys, report; print(sorted(m for m in %r if m in sys.modules))" % (banned,)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
