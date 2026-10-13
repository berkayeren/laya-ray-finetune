"""MPS spike worker inside a real 1-worker Ray Train run on CPU (tiny offline model): the spike env reaches the
worker through ray.init(runtime_env) and the result comes back as run.json inside a Ray Checkpoint.

Runs in `make test` (ray marker, CPU only, no weights or data).
"""
import json
import os
import subprocess
import sys

import pytest

import mps_spike
from pilot import spike

pytestmark = pytest.mark.ray

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_child(tmp_path, mode, device="cpu", cpus=1):
    """Write a tiny-source child config and run `mps_spike.py --child MODE CONFIG` the way the orchestrator does."""
    expect_env = spike.spike_env("0", cpus)
    cfg = {"source": "tiny", "device": device, "micro_batch": 2, "max_len": 32, "grad_accum": 2, "rung": "none",
           "fallback": "0", "purpose": "initial", "warmup_steps": 1, "measured_steps": 2, "seed": 0, "cpus": cpus,
           "storage": str(tmp_path / "ray"), "run_name": "spike-ray-test", "expect_env": expect_env,
           "hparams": mps_spike.HPARAMS, "out_json": str(tmp_path / "r.json")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, "mps_spike.py", "--child", mode, str(path)], cwd=REPO_ROOT,
                       env=dict(os.environ, **expect_env), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    with open(cfg["out_json"], encoding="utf-8") as f:
        return json.load(f), expect_env


@pytest.mark.timeout(600)
def test_ray_worker_gets_env_and_returns_result_via_checkpoint(tmp_path):
    payload, expect_env = run_child(tmp_path, "ray")
    assert payload["status"] == "ok", payload
    assert payload["transport"] == "checkpoint"
    assert payload["env_seen"] == expect_env
    assert payload["param_device"] == "cpu" and payload["ddp_wrapped"] is False and payload["world_size"] == 1
    assert payload["regimes"]["padded"]["times"]["n"] == 2
    assert payload["regimes"]["padded"]["batch_shape"] == [2, 32]
    assert payload["regimes"]["real"]["times"]["n"] == 2
