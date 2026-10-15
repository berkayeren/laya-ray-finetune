"""EVAL-08 CPU smoke: train a tiny Laya checkpoint with ray_train.py, then score and report it through the new path.

Offline end to end: a random-init tiny checkpoint (tests/tiny_laya.py) and the synthetic splits (tests/conftest.py),
so nothing downloads. ray_train.py runs 1 Ray worker for 1 epoch with the encoder frozen on 12 messages of the
synthetic selection_train (all 12 train: there is no calibration holdout, LAYA-05) through pilot/laya_adapter.py;
score.py dumps the uncalibrated checkpoint (T = 1) on the synthetic val and test splits and report.py turns the dumps
into metrics.
Runs in `make test` under the ray marker (CPU only, no weights or data).
"""
import json
import os
import subprocess
import sys

import pytest

import pilot.data
from conftest import make_raw
from pilot import dumps
from pilot.weights import identity
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


def _run(*args):
    """Run a repo script with the test interpreter (no PYTHONPATH, so it imports like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """(work dir, splits dir, manifest, uncalibrated checkpoint dir) of one ray_train.py run on the tiny checkpoint."""
    d = tmp_path_factory.mktemp("smoke")
    root, raw_sha256 = make_raw(str(d / "raw"))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    manifest = os.path.join(out_dir, "splits.json")
    base = tiny_checkpoint(d / "base")                      # absolute local --base: the pinned fetch is never taken
    ckpt = d / "ckpt"
    _run("ray_train.py", "--data", os.path.join(out_dir, "splits", "selection_train.jsonl"), "--manifest", manifest,
         "--base", base, "--out", ckpt, "--workers", "1", "--freeze-encoder", "--epochs", "1",
         "--max-messages", "12", "--storage", d / "ray")
    return d, os.path.join(out_dir, "splits"), manifest, str(ckpt)


def _score(splits, split, manifest, ckpt, out_root):
    _run("score.py", os.path.join(splits, "%s.jsonl" % split), "--model", "laya", "--checkpoint", ckpt,
         "--workers", "0", "--manifest", manifest, "--out-root", out_root)
    return os.path.join(str(out_root), dumps.run_id("laya", identity(ckpt)))


def _read_json(path):
    with open(str(path), encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.timeout(600)
def test_ray_train_then_score_and_report(trained):
    d, splits, manifest, ckpt = trained
    cfg = _read_json(os.path.join(ckpt, "rl_agent_config.json"))
    assert cfg["training"]["ray_train"] and cfg["training"]["calibrated"] is False, cfg.get("training")
    assert cfg["temperature"] == [1.0, 1.0, 1.0]            # uncalibrated: postfit.py fits T on val (LAYA-05)
    _score(splits, "val", manifest, ckpt, d / "dumps")
    run_dir = _score(splits, "test", manifest, ckpt, d / "dumps")
    _run("report.py", run_dir, "--split", "test", "--manifest", manifest, "--out", d / "r.json")
    rep = _read_json(d / "r.json")
    assert rep["split"]["name"] == "test" and rep["subset"] is False
    assert sorted(rep["variants"]) == ["raw", "shipped"]
    for v in rep["variants"].values():
        assert v["rows"]["0.5"]["n"] == 7 and v["rows"]["0.5"]["n_no_intent"] == 1
        sel = v["rows"]["val_selected"]
        assert sel is not None and isinstance(sel["theta"], dict) and sel["in_sample"] is False
    val_side = _read_json(os.path.join(run_dir, "val.json"))
    assert val_side["temperature"]["shipped"] == 1.0         # the uncalibrated checkpoint serves T 1
    assert rep["variants"]["shipped"]["temperature"] == 1.0
    assert rep["variants"]["raw"]["temperature"] == 1.0


@pytest.mark.timeout(600)
def test_evaluate_wrapper_matches_report(trained):
    """evaluate.py (the thin wrapper) prints and writes byte for byte what report.py computes from the same dump."""
    d, splits, manifest, ckpt = trained
    _score(splits, "val", manifest, ckpt, d / "dumps2")     # the val dump the val-selected θ comes from
    _run("evaluate.py", os.path.join(splits, "test.jsonl"), "--checkpoint", ckpt, "--workers", "0",
         "--manifest", manifest, "--out-root", d / "dumps2", "--out", d / "e.json")
    run_dir = os.path.join(str(d / "dumps2"), dumps.run_id("laya", identity(ckpt)))
    _run("report.py", run_dir, "--split", "test", "--manifest", manifest, "--out", d / "r2.json")
    with open(str(d / "e.json"), "rb") as f, open(str(d / "r2.json"), "rb") as g:
        e, r = f.read(), g.read()
    assert e == r
    assert json.loads(e)["variants"]["shipped"]["rows"]["val_selected"] is not None
