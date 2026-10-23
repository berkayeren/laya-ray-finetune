"""pilot.metrics on hand-built [n, 48] arrays, cross-checked against scikit-learn: no model, no Ray.

Ports the semantics of the legacy tests/test_evaluate_metrics.py (empty input, all-no-intent rows, top-1 ignores
no-intent rows) to the dump-based metric functions, and pins the zero-support macro rule, the inclusive threshold,
unrounded values, INTENTS ordering and tie rules, and the overflow-free temperature sigmoid.
"""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from intents import INTENTS
from pilot import metrics

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)


def _gold(*tags):
    return np.array([t in tags for t in TAGS], dtype=bool)


def _probs(**hot):
    return np.array([hot.get(t, 0.0) for t in TAGS], dtype=np.float64)


def test_metrics_refuses_empty_input():
    with pytest.raises(SystemExit, match="no messages"):
        metrics.metrics(np.zeros((0, 48), dtype=bool), np.zeros((0, 48)), 0.5)


def test_single_row_input():
    out = metrics.metrics(_gold("card")[None], _probs(card=0.9)[None], 0.5)
    assert out["n"] == 1 and out["micro_f1"] == 1.0 and out["exact_match"] == 1.0 and out["top1_hit"] == 1.0


def test_all_no_intent_rows():
    Y = np.zeros((2, 48), dtype=bool)
    out = metrics.metrics(Y, np.stack([_probs(), _probs(card=0.9)]), 0.5)
    assert out["n"] == 2 and out["n_top1"] == 0 and out["top1_hit"] is None and out["n_no_intent"] == 2
    assert out["exact_match"] == 0.5
    assert out["n_tags_in_macro"] == 1 and out["macro_f1"] == 0.0     # card: zero support, predicted -> F1 0
    assert out["best_tags"] == {} and out["worst_tags"] == {}
    out = metrics.metrics(Y, np.zeros((2, 48)), 0.5)
    assert out["macro_f1"] is None and out["n_tags_in_macro"] == 0


def test_top1_ignores_no_intent_rows():
    Y = np.stack([_gold("card"), _gold("lost_stolen"), _gold()])
    P = np.stack([_probs(card=0.9), _probs(card=0.8, lost_stolen=0.1), _probs(card=0.7)])
    out = metrics.metrics(Y, P, 0.5)
    assert out["n_top1"] == 2 and out["n_no_intent"] == 1
    assert out["top1_hit"] == 0.5                           # the no-intent row is not a forced miss


def test_exact_match_counts_empty_prediction_on_no_intent_row():
    out = metrics.metrics(_gold()[None], np.full((1, 48), 0.4), 0.5)
    assert out["exact_match"] == 1.0 and out["mean_tags_predicted"] == 0.0


def test_threshold_is_inclusive():
    out = metrics.metrics(_gold("card")[None], _probs(card=0.5)[None], 0.5)
    assert out["per_tag"]["card"]["n_pred"] == 1 and out["per_tag"]["card"]["tp"] == 1 and out["micro_f1"] == 1.0


def _random_case(seed, n=60, density=0.1, pred_density=0.15):
    """Random gold at `density`; probabilities above 0.5 in about `pred_density` of the cells."""
    rng = np.random.default_rng(seed)
    Y = rng.random((n, 48)) < density
    u = rng.random((n, 48))
    return Y, np.where(rng.random((n, 48)) < pred_density, 0.5 + u / 2, u / 2)


@pytest.mark.parametrize("theta", [0.3, 0.5, 0.7])
def test_micro_subset_and_per_tag_match_sklearn(theta):
    Y, P = _random_case(7)
    out = metrics.metrics(Y, P, theta)
    pred = P >= theta
    assert abs(out["micro_f1"] - f1_score(Y, pred, average="micro", zero_division=0)) < 1e-12
    assert abs(out["exact_match"] - accuracy_score(Y, pred)) < 1e-12
    per = f1_score(Y, pred, average=None, zero_division=0)
    assert max(abs(out["per_tag"][t]["f1"] - per[j]) for j, t in enumerate(TAGS)) < 1e-12


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_macro_matches_sklearn_on_included_labels(seed):
    Y, P = _random_case(seed, n=25, density=0.05, pred_density=0.03)   # some tags: no support, no pred
    pred = P >= 0.5
    labels = [j for j in range(48) if Y[:, j].any() or pred[:, j].any()]
    assert len(labels) < 48                                 # the zero-support rule is exercised
    out = metrics.metrics(Y, P, 0.5)
    assert out["n_tags_in_macro"] == len(labels)
    want = f1_score(Y, pred, average="macro", labels=labels, zero_division=0)
    assert abs(out["macro_f1"] - want) < 1e-12


def test_values_are_unrounded():
    out = metrics.metrics(_gold("card", "lost_stolen")[None], _probs(card=0.9)[None], 0.5)
    assert out["micro_f1"] == 2 / 3 and out["micro_recall"] == 0.5


def test_per_tag_order_and_best_worst_ties():
    hit, miss = [TAGS[i] for i in (30, 2, 15)], [TAGS[i] for i in (40, 7, 20)]
    Y = np.stack([_gold(t) for t in hit + miss])
    P = np.stack([_probs(**{t: 0.9}) for t in hit] + [_probs() for _ in miss])
    out = metrics.metrics(Y, P, 0.5)
    assert list(out["per_tag"]) == TAGS
    assert list(out["best_tags"]) == [TAGS[i] for i in (2, 15, 30, 7, 20)]
    assert list(out["worst_tags"]) == [TAGS[i] for i in (7, 20, 40, 2, 15)]
    # top-1 tie: card (index 3) and lost_stolen (index 33) score the same; argmax picks the earlier tag
    out = metrics.metrics(_gold("lost_stolen")[None], _probs(card=0.9, lost_stolen=0.9)[None], 0.5)
    assert out["top1_hit"] == 0.0


def test_probabilities_stable_and_temperature():
    with np.errstate(all="raise"):
        p = metrics.probabilities(np.array([[-1e4, 0, 1e4]], dtype=np.float32), 1.0)
    assert p.dtype == np.float64 and p.tolist() == [[0.0, 0.5, 1.0]]
    z = np.random.default_rng(0).normal(0, 4, size=(5, 48)).astype(np.float32)
    np.testing.assert_allclose(metrics.probabilities(z, 2.0), metrics.probabilities(z / 2, 1.0), rtol=1e-12)
    for bad in (0, -1.0, float("nan")):
        with pytest.raises(SystemExit, match="temperature"):
            metrics.probabilities(z, bad)


def test_per_tag_theta_vector():
    Y = np.stack([_gold("card"), _gold("lost_stolen")])
    P = np.stack([_probs(card=0.6), _probs(lost_stolen=0.6)])
    th = np.full(48, 0.5)
    th[TAGS.index("lost_stolen")] = 0.7
    out = metrics.metrics(Y, P, th)
    assert out["per_tag"]["card"]["tp"] == 1 and out["per_tag"]["lost_stolen"]["fn"] == 1


# --- the val-selected θ rule (EVAL-02, D-12; per-tag objective and tie rule: research A2, user-confirmed) ----------

def _scores(*best):
    """GRID-aligned scores: 1.0 at the θ values in `best`, 0 elsewhere."""
    s = np.zeros(len(metrics.GRID))
    for th in best:
        s[int(round(th * 100)) - 5] = 1.0
    return s


def _rows(spec, n):
    """(Y, P) [n, 48] from {tag: (gold rows, P on gold rows, P on other rows)}; every other cell is 0."""
    Y, P = np.zeros((n, 48), dtype=bool), np.zeros((n, 48))
    for t, (gold, p_pos, p_neg) in spec.items():
        j = TAGS.index(t)
        Y[gold, j] = True
        P[:, j] = np.where(Y[:, j], p_pos, p_neg)
    return Y, P


def test_grid_is_exact():
    g = metrics.GRID
    assert len(g) == 91 and g.dtype == np.float64
    assert g[0] == 0.05 and g[-1] == 0.95 and g[2] == 0.07 and g[35] == 0.40
    assert np.abs(np.diff(g) - 0.01).max() < 1e-12
    assert all(g[i] == float("0.%02d" % (i + 5)) for i in range(91))   # every point equals its decimal literal
    c = metrics.GRID_CENTI
    assert c.dtype.kind == "i" and c.tolist() == list(range(5, 96))   # exact integer hundredths, 91 values
    assert (g == c / 100.0).all()


def test_pick_prefers_closest_to_half():
    assert metrics.pick(_scores(0.30, 0.60)) == 0.6                     # 0.6 is 0.1 from 0.5, 0.3 is 0.2


def test_pick_equidistant_takes_lower():
    assert metrics.pick(_scores(0.45, 0.55)) == 0.45


@pytest.mark.parametrize("k", range(1, 46))
def test_pick_equidistant_takes_lower_every_pair(k):
    assert metrics.pick(_scores((50 - k) / 100, (50 + k) / 100)) == (50 - k) / 100


def test_select_theta_tie_takes_lower_global_and_per_tag():
    # card's F1 (and so micro-F1) is 2/3 at exactly 0.30 and 0.70 and lower at every other grid point; the global and
    # the per-tag θ both go through pick, so both must take the lower of the two equally close points
    Y, P = np.zeros((21, 48), dtype=bool), np.zeros((21, 48))
    j = TAGS.index("card")
    Y[:10, j] = True
    P[:5, j], P[5:10, j], P[10:20, j], P[20, j] = 0.30, 0.70, 0.69, 0.29
    sel = metrics.select_theta(Y, P)
    assert sel["eligible_tags"] == ["card"]
    assert sel["global"] == 0.30 and sel["per_tag"] == {"card": 0.30}


def test_per_tag_tie_ignores_global_theta():
    # card ties at 0.30 / 0.70 on its own F1; lost_stolen drags the global micro-F1 optimum to 0.70. A2: card's own tie
    # still goes to the lower of the two equally close points (0.30), not toward the global θ (the rejected rule would
    # give 0.70); 0.30 / 0.70 is also one of the CR-01 float-distance pairs, so a failure here can be either regression
    Y, P = np.zeros((221, 48), dtype=bool), np.zeros((221, 48))
    j, k = TAGS.index("card"), TAGS.index("lost_stolen")
    Y[:10, j] = True
    P[:5, j], P[5:10, j], P[10:20, j], P[20, j] = 0.30, 0.70, 0.69, 0.29
    Y[21:121, k] = True
    P[21:121, k], P[121:221, k] = 0.95, 0.69
    sel = metrics.select_theta(Y, P)
    assert sel["eligible_tags"] == ["card", "lost_stolen"] and sel["global"] == 0.70
    assert sel["per_tag"] == {"card": 0.30, "lost_stolen": 0.70}


def test_pick_all_zero_scores_returns_half():
    assert metrics.pick(np.zeros(91)) == 0.5


def test_boundary_is_inclusive():
    # the only positive sits exactly on 0.40: one step above loses it, one step below adds a false positive
    Y, P = _rows({"card": ([0], 0.40, 0.39)}, 3)
    sel = metrics.select_theta(Y, P)
    assert sel["global"] == 0.40 and sel["per_tag"] == {}
    assert metrics.metrics(Y, P, metrics.theta_vector(sel))["micro_f1"] == 1.0


def test_eligibility_at_ten_positives():
    assert metrics.MIN_VAL_POSITIVES == 10
    Y, P = _rows({"card": (list(range(10)), 0.9, 0.7), "lost_stolen": (list(range(9)), 0.3, 0.2)}, 20)
    sel = metrics.select_theta(Y, P)
    assert sel["eligible_tags"] == ["card"] and sel["min_val_positives"] == 10
    assert sel["per_tag"] == {"card": 0.71}                 # card's own best range (0.70, 0.90], closest to 0.5
    assert sel["global"] == 0.30                            # micro-F1 best on (0.20, 0.30]
    vec = metrics.theta_vector(sel)
    assert vec.dtype == np.float64 and vec.shape == (48,)
    assert vec[TAGS.index("card")] == 0.71 and vec[TAGS.index("lost_stolen")] == 0.30   # 9 positives: global θ
    assert (np.delete(vec, TAGS.index("card")) == 0.30).all()


def test_zero_positive_tag_uses_global():
    Y, P = _rows({"card": (list(range(12)), 0.8, 0.1), "new": ([], 0.0, 0.6)}, 15)
    sel = metrics.select_theta(Y, P)
    assert "new" not in sel["eligible_tags"] and sel["eligible_tags"] == ["card"]
    assert metrics.theta_vector(sel)[TAGS.index("new")] == sel["global"]


def test_theta_selected_after_temperature():
    # one tag positive at z = 2 and negative at z = 1; θ lives on the reported (after-temperature) scale
    Z = np.full((2, 48), -10.0, dtype=np.float32)
    Z[0, TAGS.index("card")], Z[1, TAGS.index("card")] = 2.0, 1.0
    Y = _rows({"card": ([0], 1.0, 0.0)}, 2)[0]
    want = {1.0: 0.74, 2.0: 0.63}                           # best ranges (0.731, 0.881] and (0.622, 0.731]
    for T, theta in want.items():
        P = metrics.probabilities(Z, T)
        sel = metrics.select_theta(Y, P)
        assert sel["global"] == theta and sel["per_tag"] == {}
        assert metrics.metrics(Y, P, metrics.theta_vector(sel))["micro_f1"] == 1.0


def test_select_theta_refuses_empty_val():
    with pytest.raises(SystemExit, match="no val messages"):
        metrics.select_theta(np.zeros((0, 48), dtype=bool), np.zeros((0, 48)))


# --- new-tag average precision (AP_RULE; Phase 9's primary metric, frozen in Phase 8), sklearn as the oracle ---------

def test_average_precision_tied_scores_form_one_threshold():
    y = np.array([1, 0, 1, 0], dtype=bool)
    s = np.array([0.5, 0.5, 0.5, 0.1])
    assert abs(metrics.average_precision(y, s) - 2 / 3) <= 1e-12   # one threshold at 0.5: P 2/3, R 1
    assert abs(metrics.average_precision(y, s) - average_precision_score(y, s)) <= 1e-12


def test_average_precision_matches_sklearn():
    rng = np.random.default_rng(0)
    for i in range(20):
        y = rng.random(413) < rng.uniform(0.01, 0.3)
        y[rng.integers(413)] = True                         # at least one positive
        s = rng.random(413) + 0.8 * y
        if i % 2:
            s = np.round(s, 1)                              # heavy ties on every other pair
        got = metrics.average_precision(y, s)
        assert type(got) is float and abs(got - average_precision_score(y, s)) < 1e-12, i
    y = np.zeros(10, dtype=bool)
    y[[2, 7]] = True
    assert metrics.average_precision(y, y.astype(float)) == 1.0     # a perfect ranking


@pytest.mark.parametrize("y, s", [(np.zeros(5, dtype=bool), np.arange(5.0)),          # no positive
                                  (np.zeros(0, dtype=bool), np.zeros(0)),             # empty
                                  (np.ones(4, dtype=bool), np.zeros(5)),              # length mismatch
                                  (np.ones(3, dtype=bool), np.array([0.1, np.nan, 0.2])),
                                  (np.ones(3, dtype=bool), np.array([0.1, np.inf, 0.2])),
                                  (np.ones((2, 2), dtype=bool), np.zeros((2, 2)))])   # not one column
def test_average_precision_refuses(y, s):
    with pytest.raises(SystemExit):
        metrics.average_precision(y, s)


def test_new_tag_ap_per_tag_and_macro():
    rng = np.random.default_rng(1)
    Y = rng.random((60, 48)) < 0.2
    P = rng.random((60, 48))
    out = metrics.new_tag_ap(Y, P, ["savings", "loan"])
    assert out["tags"] == ["savings", "loan"] and list(out["per_tag"]) == ["savings", "loan"]
    for t in ("savings", "loan"):
        j = TAGS.index(t)
        assert abs(out["per_tag"][t] - average_precision_score(Y[:, j], P[:, j])) < 1e-12
    assert out["macro"] == (out["per_tag"]["savings"] + out["per_tag"]["loan"]) / 2


def test_new_tag_ap_refuses():
    rng = np.random.default_rng(2)
    Y = rng.random((30, 48)) < 0.3
    P = rng.random((30, 48))
    for tags in (["savings", "savings"], ["savings", "not_a_tag"], []):
        with pytest.raises(SystemExit):
            metrics.new_tag_ap(Y, P, tags)
    Y[:, TAGS.index("loan")] = False
    with pytest.raises(SystemExit, match="no positive"):
        metrics.new_tag_ap(Y, P, ["savings", "loan"])


def test_ap_rule_states_the_definition():
    rule = metrics.AP_RULE
    for word in ("tied scores form one threshold", "not interpolated", "average_precision_score", "refused",
                 "T-invariant"):
        assert word in rule, word
