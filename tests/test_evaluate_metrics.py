"""evaluate.metrics on hand-built gold/prob sets: no model, no Ray, no data files."""
import pytest

import evaluate
from intents import INTENTS

pytestmark = pytest.mark.unit


def _probs(**hot):
    return {k: hot.get(k, 0.0) for k in INTENTS}


def test_metrics_refuses_empty_input():
    with pytest.raises(SystemExit, match="no messages"):
        evaluate.metrics([], [])


def test_metrics_all_no_intent_rows_do_not_crash():
    out = evaluate.metrics([set(), set()], [_probs(), _probs(card=0.9)])
    assert out["n"] == 2 and out["n_top1"] == 0 and out["n_no_intent"] == 2
    assert out["top1_hit"] is None and out["majority_top1_hit"] is None
    assert out["macro_f1@0.5"] is None
    assert out["best_tags@0.5"] == {} and out["worst_tags@0.5"] == {}


def test_top1_ignores_no_intent_rows():
    golds = [{"card"}, {"lost_stolen"}, set()]
    probs = [_probs(card=0.9), _probs(card=0.8, lost_stolen=0.1), _probs(card=0.7)]
    out = evaluate.metrics(golds, probs)
    assert out["n_top1"] == 2 and out["n_no_intent"] == 1
    assert out["top1_hit"] == 0.5                           # the no-intent row is not a forced miss
    assert out["majority_top1_hit"] == 0.5
