"""pilot.metrics on hand-built [n, 48] arrays, cross-checked against scikit-learn: no model, no Ray.

Ports the semantics of the legacy tests/test_evaluate_metrics.py (empty input, all-no-intent rows, top-1 ignores
no-intent rows) to the dump-based metric functions, and pins the zero-support macro rule, the inclusive threshold,
unrounded values, INTENTS ordering and tie rules, and the overflow-free temperature sigmoid.
"""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

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
