"""The EVAL-07 paired bootstrap (pilot.metrics.seed_average / bootstrap_weights / replicate_metrics /
paired_bootstrap) and report.py --compare: zero-width self and copy intervals, pairing, seed averaging of
probabilities, determinism for a seed, refusals, and a per-replicate oracle against pilot.metrics.metrics.
No model, torch or Ray: dump-level cases run score.py baselines on the synthetic snapshot (conftest).
"""
import os

import pytest

import report
import score
from conftest import prepared

pytestmark = pytest.mark.unit


@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path) of the synthetic snapshot prepared under tmp_path."""
    return prepared(tmp_path, synthetic_raw)


def _score_both(snap, root, model="prior"):
    """Score `model` on the synthetic val and test splits under root; return (run_dir, manifest)."""
    out_dir, manifest = snap
    for name in ("val", "test"):
        score.main([os.path.join(out_dir, "splits", "%s.jsonl" % name), "--model", model, "--manifest", manifest,
                    "--out-root", str(root)])
    return os.path.join(str(root), model), manifest


def _assert_zero_width(rep):
    assert set(rep["thresholds"]) == {"0.5", "val_selected"}
    for t, r in rep["thresholds"].items():
        assert r["B"] == rep["B"] and r["seed"] == rep["seed"]
        for name, m in r["metrics"].items():
            assert m["ci"] == [0.0, 0.0] and m["width"] == 0.0, (t, name, m)


def test_compare_self_is_zero_width(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    rep = report.compare([run_dir], [run_dir], "test", manifest, B=200, seed=0)
    assert rep["kind"] == "paired_bootstrap" and rep["n"] == 7 and rep["B"] == 200
    assert set(rep["thresholds"]["0.5"]["metrics"]) == {"micro_f1", "macro_f1", "exact_match", "top1_hit"}
    _assert_zero_width(rep)
