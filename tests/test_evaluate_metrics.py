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
    assert out["n"] == 2
    assert out["macro_f1@0.5"] is None
    assert out["best_tags@0.5"] == {} and out["worst_tags@0.5"] == {}
