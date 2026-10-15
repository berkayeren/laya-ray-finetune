"""BERTurk on the Apple GPU (BASE-01 fp32, BASE-02 shared scorer): a tiny BERT base trained by berturk_train.py
--device mps keeps every parameter on mps in float32, and score.py --model berturk --device mps --workers 0 writes an
fp32 val dump (autocast off) whose logits match the CPU dump of the same export within 1e-4 (06-RESEARCH probe 5 saw
1.79e-05 on the pinned base).

Needs Apple MPS, so it is excluded from `make test`: run `pytest -m mps tests/test_berturk_mps.py` on the M4, with no
other MPS job running.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pilot.data
import score
from conftest import prepared
from pilot import berturk, dumps
from tiny_berturk import tiny_base

pytestmark = pytest.mark.mps

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600
TOLERANCE = 1e-4


def _run(*args):
    """Run a repo script with the test interpreter (no PYTHONPATH, so it imports like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


@pytest.mark.timeout(TIMEOUT)
def test_berturk_trains_and_scores_on_mps_in_fp32(tmp_path, synthetic_raw):
    import torch
    if not torch.backends.mps.is_available():
        pytest.fail("the mps marker needs Apple MPS; run it on the M4, or deselect it with -m 'not mps'")
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    splits = os.path.join(out_dir, "splits")
    base = tiny_base(tmp_path / "base")
    storage = tmp_path / "runs"
    argv = ["berturk_train.py", "--data", os.path.join(splits, "selection_train.jsonl"), "--manifest", manifest,
            "--base", base, "--device", "mps", "--epochs", "1", "--batch", "4", "--max-len", "32", "--seed", "0",
            "--storage", storage]
    rid = _run(*(argv + ["--print-run-id"])).stdout.strip()
    _run(*argv)
    export = str(storage / rid / "exports" / "epoch-001")
    with open(os.path.join(export, berturk.RECORD), encoding="utf-8") as f:
        rec = json.load(f)["record"]
    assert rec["device"] == "mps" and rec["param_device"] == "mps", rec
    assert rec["dtype"] == "torch.float32" and rec["amp"] is False and rec["bit_reproducible"] is False, rec

    val = os.path.join(splits, "val.jsonl")
    threads = torch.get_num_threads()
    try:
        loaded = {}
        for device in ("mps", "cpu"):
            d, name = score.main([val, "--model", "berturk", "--checkpoint", export, "--device", device,
                                  "--workers", "0", "--manifest", manifest, "--out-root", str(tmp_path / device)])
            assert name == "val"
            loaded[device] = dumps.load_dump(d, "val", manifest)
    finally:
        torch.set_num_threads(threads)
    (m, m_uids, m_side, _), (c, c_uids, c_side, _) = loaded["mps"], loaded["cpu"]
    assert m_side["system"]["kind"] == "berturk" and m_side["run_id"] == c_side["run_id"]
    assert (m_side["scorer"]["device"], m_side["scorer"]["dtype"], m_side["scorer"]["amp_enabled"]) \
        == ("mps", "torch.float32", False), m_side["scorer"]
    assert (c_side["scorer"]["device"], c_side["scorer"]["dtype"], c_side["scorer"]["amp_enabled"]) \
        == ("cpu", "torch.float32", False), c_side["scorer"]
    assert list(m_uids) == list(c_uids) and len(m_uids) > 0
    assert m.dtype == np.float32 and m.shape == c.shape
    diff = float(np.abs(m - c).max())
    print("BERTurk MPS vs CPU max abs logit difference: %.3e" % diff)
    assert diff < TOLERANCE, diff
