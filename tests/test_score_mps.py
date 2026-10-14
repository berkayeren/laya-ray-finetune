"""score.py --device mps on Apple MPS with the tiny offline checkpoint: the dump is fp32 (laya's fp16 autocast default
turned off by la.force_fp32 and recorded in the sidecar) and matches the CPU fp32 dump of the same split.

Needs Apple MPS: `pytest -m mps tests/test_score_mps.py` (excluded from `make test`).
"""
import json
import os

import numpy as np
import pytest

import score
from conftest import prepared
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.mps


@pytest.fixture
def tiny(tmp_path, synthetic_raw):
    """(val split path, manifest path, tiny checkpoint dir); torch's thread count is restored afterwards."""
    import torch
    threads = torch.get_num_threads()
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    yield os.path.join(out_dir, "splits", "val.jsonl"), manifest, tiny_checkpoint(tmp_path / "ckpt", temperature=1.3)
    torch.set_num_threads(threads)


def _score(split, manifest, ckpt, device, root):
    run_dir, name = score.main([split, "--model", "laya", "--checkpoint", ckpt, "--device", device, "--workers", "0",
                                "--manifest", manifest, "--out-root", str(root)])
    z = np.load(os.path.join(run_dir, name + ".npz"), allow_pickle=False)
    with open(os.path.join(run_dir, name + ".json"), encoding="utf-8") as f:
        return z["uids"], z["logits"], json.load(f)["scorer"]


@pytest.mark.timeout(600)
def test_mps_dump_is_fp32_and_matches_cpu(tmp_path, tiny):
    import torch
    if not torch.backends.mps.is_available():
        pytest.fail("the mps marker needs Apple MPS; run it on the M4, or deselect it with -m 'not mps'")
    split, manifest, ckpt = tiny
    m_uids, m, m_sc = _score(split, manifest, ckpt, "mps", tmp_path / "m")
    c_uids, c, c_sc = _score(split, manifest, ckpt, "cpu", tmp_path / "c")
    assert m_sc["device"] == "mps" and m_sc["dtype"] == "torch.float32" and m_sc["amp_enabled"] is False
    assert m_sc["laya_default_amp"] is True and m_sc["laya_default_dtype"] == "torch.float16"   # what was switched off
    assert c_sc["device"] == "cpu" and c_sc["laya_default_amp"] is False
    assert list(m_uids) == list(c_uids) and len(m_uids) > 0
    assert m.dtype == np.float32 and m.shape == c.shape
    assert float(np.abs(m - c).max()) <= 1e-3, float(np.abs(m - c).max())
