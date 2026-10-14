"""score.py with Ray Data actors on the tiny offline checkpoint: the dump equals the in-process (--workers 0) dump.

Ray returns rows in any order; score.assemble puts them back in split-file order by uid, so both dumps hold the same
uids in the same order and logits within 1e-5 (thread counts differ between the two paths).

Runs in `make test` (ray marker, CPU only, no weights or data).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import score
from conftest import prepared
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.timeout(600)
def test_ray_and_inprocess_dumps_agree(tmp_path, synthetic_raw):
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    ckpt = tiny_checkpoint(tmp_path / "ckpt", temperature=1.3)
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    common = [test_jsonl, "--model", "laya", "--checkpoint", ckpt, "--manifest", manifest]
    run_a, _ = score.main(common + ["--workers", "0", "--out-root", str(tmp_path / "a")])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "score.py"] + common + ["--workers", "2", "--cpus-per-worker", "1",
                                                                 "--out-root", str(tmp_path / "b")],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    run_b = os.path.join(str(tmp_path / "b"), os.path.basename(run_a))
    a = np.load(os.path.join(run_a, "test.npz"), allow_pickle=False)
    b = np.load(os.path.join(run_b, "test.npz"), allow_pickle=False)
    assert list(a["uids"]) == list(b["uids"])
    np.testing.assert_allclose(a["logits"], b["logits"], rtol=0, atol=1e-5)
    with open(os.path.join(run_a, "test.json"), encoding="utf-8") as f:
        side_a = json.load(f)
    with open(os.path.join(run_b, "test.json"), encoding="utf-8") as f:
        side_b = json.load(f)
    assert side_a["scorer"]["workers"] == 0 and side_b["scorer"]["workers"] == 2
    assert side_b["scorer"]["torch_threads"] == 1 and side_b["scorer"]["device"] == "cpu"
    assert side_a["temperature"]["shipped"] == side_b["temperature"]["shipped"] == 1.3
