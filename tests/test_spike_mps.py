"""MPS spike worker on Apple MPS with the tiny offline model: in-process (strict fallback 0) and inside a 1-worker
Ray Train run, the parameters land on MPS and the MPS memory readings are live. On this torch build the D-10 capture
path names a real unsupported op (aten::linalg_eig) under both PYTORCH_ENABLE_MPS_FALLBACK=0 and =1.

Needs Apple MPS: `pytest -m mps tests/test_spike_mps.py` (excluded from `make test`).
"""
import json
import os
import subprocess
import sys

import pytest

import mps_spike
from pilot import spike

pytestmark = pytest.mark.mps

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_child(tmp_path, mode, device="mps", cpus=2):
    """Write a tiny-source child config and run `mps_spike.py --child MODE CONFIG` the way the orchestrator does."""
    expect_env = spike.spike_env("0", cpus)
    cfg = {"source": "tiny", "device": device, "micro_batch": 2, "max_len": 32, "grad_accum": 2, "rung": "none",
           "fallback": "0", "purpose": "initial", "warmup_steps": 1, "measured_steps": 2, "seed": 0, "cpus": cpus,
           "storage": str(tmp_path / "ray"), "run_name": "spike-mps-test", "expect_env": expect_env,
           "hparams": mps_spike.HPARAMS, "out_json": str(tmp_path / "r.json")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, "mps_spike.py", "--child", mode, str(path)], cwd=REPO_ROOT,
                       env=dict(os.environ, **expect_env), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    with open(cfg["out_json"], encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.timeout(600)
def test_tiny_inprocess_strict_on_mps(tmp_path):
    payload = run_child(tmp_path, "inprocess")
    assert payload["status"] == "ok", payload
    assert payload["param_device"] == "mps" and payload["env_seen"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert payload["peak_driver_allocated_bytes"] > 0
    assert payload["recommended_max_memory_bytes"] > 0
    assert payload["fallback_ops"] == [], payload["fallback_ops"]           # the strict tiny run falls back nowhere


@pytest.mark.timeout(600)
def test_tiny_ray_worker_on_mps(tmp_path):
    payload = run_child(tmp_path, "ray")
    assert payload["status"] == "ok", payload
    assert payload["param_device"] == "mps" and payload["transport"] == "checkpoint"
    assert payload["ddp_wrapped"] is False


STRICT_PROBE = """
import torch
from pilot import spike
try:
    torch.linalg.eig(torch.eye(2, device="mps"))
    print("NO_ERROR")
except NotImplementedError as e:
    print(spike.op_name(str(e)))
"""

FALLBACK_PROBE = """
import json, warnings
import torch
from pilot import spike
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    torch.linalg.eig(torch.eye(2, device="mps"))
print(json.dumps(spike.tag_fallback_ops([str(w.message) for w in caught], [(0, "step")])))
"""


def probe(script, fallback):
    """Run a probe in a fresh interpreter: the fallback flag is read once, at `import torch` (RESEARCH Pitfall 3)."""
    r = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK=fallback))
    assert r.returncode == 0, r.stderr[-4000:]
    return r.stdout.strip().splitlines()[-1]


@pytest.mark.timeout(600)
def test_strict_and_fallback_capture_name_linalg_eig():
    assert probe(STRICT_PROBE, "0") == "aten::linalg_eig"
    assert json.loads(probe(FALLBACK_PROBE, "1")) == [{"op": "aten::linalg_eig", "stage": "step"}]
