"""EVAL-03 calibration definitions and the report.py --confidence binding, offline: no model, no Ray.

The definitions in pilot.metrics are pinned to independent oracles on seeded random arrays: Brier to sklearn's
brier_score_loss and NLL to sklearn's log_loss on the flattened (message, tag) pairs, and every tag's classwise ECE
to torchmetrics' binary_calibration_error(n_bins=15, norm="l1") (probabilities drawn strictly inside (0, 1), because
torchmetrics puts an exact 1.0 into an extra bin). The bin edges, the NLL clipping, the zero-support tags in the
48-tag mean, the reliability counts and the empty-input refusals are pinned on hand-built arrays.
"""
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from intents import INTENTS
from pilot import metrics

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
