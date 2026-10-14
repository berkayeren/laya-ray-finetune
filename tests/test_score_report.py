"""The shared scoring path end to end on the synthetic snapshot: score.py (baseline -> dump) and report.py (dump ->
metrics), with no model, torch or Ray. Covers the dump contract (pilot.dumps), the role gate, the --n subset policy,
the train-only baseline source, and report.py's refusal of every tampered dump (D-04).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import prepared

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
TIMEOUT = 300


def _split_uids(path):
    with open(path, "rb") as f:
        return [r["uid"] for r in pilot.data.parse_jsonl(f.read())]


@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path) of the synthetic snapshot prepared under tmp_path."""
    return prepared(tmp_path, synthetic_raw)


def test_baseline_dump_round_trips_to_report(tmp_path, snap):
    out_dir, manifest = snap
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    run_dir, split = score.main([test_jsonl, "--model", "prior", "--manifest", manifest,
                                 "--out-root", str(tmp_path / "dumps")])
    assert (run_dir, split) == (str(tmp_path / "dumps" / "prior"), "test")
    z = np.load(os.path.join(run_dir, "test.npz"), allow_pickle=False)
    assert sorted(z.files) == ["logits", "uids"]
    assert z["logits"].dtype == np.float32 and z["logits"].shape == (7, 48)
    assert z["uids"].dtype.kind == "U" and list(z["uids"]) == _split_uids(test_jsonl)
    assert (z["logits"] == z["logits"][0]).all()             # the same baseline row for every message
    with open(os.path.join(run_dir, "test.json"), encoding="utf-8") as f:
        side_text = f.read()
    assert str(tmp_path) not in side_text and os.path.expanduser("~") not in side_text
    row = report.run_report(run_dir, "test", manifest)["variants"]["shipped"]["rows"]["0.5"]
    assert row["n"] == 7 and row["n_no_intent"] == 1


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable] + list(args), cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=TIMEOUT)


def test_cli_round_trip(tmp_path, snap):
    out_dir, manifest = snap
    r = _run("score.py", os.path.join(out_dir, "splits", "test.jsonl"), "--model", "majority",
             "--manifest", manifest, "--out-root", str(tmp_path / "dumps"))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "rep" / "majority.json"
    r = _run("report.py", str(tmp_path / "dumps" / "majority"), "--split", "test", "--out", str(out),
             "--manifest", manifest)
    assert r.returncode == 0, r.stderr
    with open(str(out), encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["run_id"] == "majority" and rep["subset"] is False
    assert rep["system"]["baseline_source"]["split"] == "final_train"
