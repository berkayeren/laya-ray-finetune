"""ray_train.py --device mps trains the tiny checkpoint's full encoder in one Ray worker on the Apple GPU.

The run record must say the parameters landed on mps, list the spike's worker env (strict fallback 0, HIGH and LOW
watermarks 1.0) as KEY=VALUE strings, record checkpointing on, AMP off and AdamW's single-tensor loop, and hold live
MPS memory readings (LAYA-02, LAYA-03, LAYA-06). The output must load through the public laya.load.

Needs Apple MPS, so it is excluded from `make test`: run `pytest -m mps tests/test_train_mps.py`.
"""
import os
import subprocess
import sys

import pytest

import pilot.data
from conftest import make_raw
from pilot import runs
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.mps

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


@pytest.mark.timeout(600)
def test_mps_record(tmp_path):
    root, raw_sha256 = make_raw(str(tmp_path / "raw"))
    out_dir = str(tmp_path / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    base = tiny_checkpoint(tmp_path / "base")
    ckpt = tmp_path / "ckpt"
    _run("ray_train.py", "--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
         "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--out", ckpt, "--device", "mps",
         "--workers", "1", "--cpus-per-worker", "2", "--epochs", "1", "--max-messages", "12",
         "--storage", tmp_path / "ray")
    rec = runs.read_record(str(ckpt))
    assert rec["device"] == "mps" and rec["param_device"] == "mps"
    for kv in ("PYTORCH_ENABLE_MPS_FALLBACK=0", "PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0",
               "PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0", "OMP_NUM_THREADS=2"):
        assert kv in rec["env"], rec["env"]
    assert rec["env"] == sorted(rec["env"])
    assert rec["grad_checkpointing"] is True and rec["amp"] is False
    assert rec["adamw_foreach"] is False and rec["adamw_fused"] is False
    assert rec["torch_threads"] == 2 and rec["recommended_max_memory_bytes"] > 0
    (row,) = rec["epochs"]
    assert tuple(sorted(row)) == tuple(sorted(runs.EPOCH_FIELDS)), row
    assert row["peak_driver_bytes"] > 0 and row["peak_total_bytes"] >= row["peak_driver_bytes"]
    assert row["current_allocated_bytes"] is not None and row["train_s"] > 0 and row["optimizer_steps"] >= 1

    import laya
    agent = laya.load(str(ckpt), device="cpu")
    assert agent is not None


def _run_fail(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode != 0, "%s should have failed:\n%s" % (args[0], r.stdout[-4000:])
    return r


@pytest.mark.timeout(600)
def test_resume_on_mps(tmp_path):
    """Killed after epoch 1 and rerun: the MPS run resumes at epoch 2 (MPS is not bitwise deterministic, so only the
    resume mechanics are checked, RESEARCH Pitfall 3)."""
    import math
    root, raw_sha256 = make_raw(str(tmp_path / "raw"))
    out_dir = str(tmp_path / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    base = tiny_checkpoint(tmp_path / "base")
    args = ["ray_train.py", "--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
            "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--out", tmp_path / "ckpt",
            "--device", "mps", "--workers", "1", "--cpus-per-worker", "2", "--epochs", "2", "--max-messages", "12",
            "--storage", tmp_path / "ray"]
    r = _run_fail(*args, "--fail-after-epoch", "1")
    assert "rerun the same command" in r.stderr, r.stderr[-4000:]
    _run(*args)
    rec = runs.read_record(str(tmp_path / "ckpt"))
    assert rec["param_device"] == "mps"
    assert [row["epoch"] for row in rec["epochs"]] == [1, 2]
    assert [row["resumed"] for row in rec["epochs"]] == [False, True]
    assert all(math.isfinite(row["loss"]) for row in rec["epochs"])
    assert rec["segments"][1]["resume_load_s"] > 0 and rec["segments"][1]["device"] == "mps"
