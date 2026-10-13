"""evaluate.metrics on hand-built gold/prob sets and the train-derived majority tag: no model, no Ray."""
import json
import os

import pytest

import evaluate
from conftest import prepared
from intents import INTENTS

pytestmark = pytest.mark.unit


def _probs(**hot):
    return {k: hot.get(k, 0.0) for k in INTENTS}


def test_metrics_refuses_empty_input():
    with pytest.raises(SystemExit, match="no messages"):
        evaluate.metrics([], [], "card")


def test_metrics_all_no_intent_rows_do_not_crash():
    out = evaluate.metrics([set(), set()], [_probs(), _probs(card=0.9)], "card")
    assert out["n"] == 2 and out["n_top1"] == 0 and out["n_no_intent"] == 2
    assert out["top1_hit"] is None and out["majority_top1_hit"] is None
    assert out["macro_f1@0.5"] is None
    assert out["best_tags@0.5"] == {} and out["worst_tags@0.5"] == {}


def test_top1_ignores_no_intent_rows():
    golds = [{"card"}, {"lost_stolen"}, set()]
    probs = [_probs(card=0.9), _probs(card=0.8, lost_stolen=0.1), _probs(card=0.7)]
    out = evaluate.metrics(golds, probs, "card")
    assert out["n_top1"] == 2 and out["n_no_intent"] == 1
    assert out["top1_hit"] == 0.5                           # the no-intent row is not a forced miss
    assert out["majority_tag"] == "card" and out["majority_top1_hit"] == 0.5


def test_majority_baseline_tag_is_fixed_by_the_evaluated_labels_never():
    golds = [{"lost_stolen"}, {"lost_stolen"}, {"card"}]   # lost_stolen is the evaluated split's own majority
    out = evaluate.metrics(golds, [_probs()] * 3, "card")
    assert out["majority_tag"] == "card" and out["majority_top1_hit"] == round(1 / 3, 3)


@pytest.mark.parametrize("role,source", [("test", "final_train"), ("val", "selection_train")])
def test_majority_tag_comes_from_the_matching_train_split(tmp_path, synthetic_raw, role, source):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    with open(os.path.join(out_dir, "splits", "%s.jsonl" % source), encoding="utf-8") as f:
        counts = {k: 0 for k in INTENTS}
        for l in f:
            for k, v in json.loads(l)["expected"].items():
                counts[k] += v
    assert evaluate.majority_tag(manifest_path, role) == (source, max(INTENTS, key=counts.get))


def test_majority_tag_refuses_a_tampered_train_split(tmp_path, synthetic_raw):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    with open(os.path.join(out_dir, "splits", "final_train.jsonl"), "ab") as f:
        f.write(b"\n")
    with pytest.raises(SystemExit, match="sha256"):
        evaluate.majority_tag(manifest_path, "test")


class _FakeAgent:
    def predict_batch(self, texts, questions, batch_size, sort_by_length):
        return [{"answers": {k: {"noul": 0.5} for k in questions}} for _ in texts]


def test_predictor_times_only_scoring_per_row():
    p = evaluate.Predictor.__new__(evaluate.Predictor)        # skip the checkpoint load
    p.agent = _FakeAgent()
    b = p({"uid": ["a", "b"], "text": ["x", "y"]})
    assert len(b["probs"]) == 2 and len(b["infer_secs"]) == 2
    assert all(t >= 0 for t in b["infer_secs"]) and b["infer_secs"][0] == b["infer_secs"][1]
