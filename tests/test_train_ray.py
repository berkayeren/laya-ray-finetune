"""ray_train.py --device cpu trains the tiny checkpoint's full encoder in one Ray worker and writes its run record.

Offline (tests/tiny_laya.py + the synthetic splits of tests/conftest.py): 2 epochs on 12 messages with the encoder
trainable, so gradient checkpointing is on and AMP off. The checkpoint config's training.ray_train.record must name the
device the parameters landed on and hold one row per epoch with exactly pilot.runs.EPOCH_FIELDS (LAYA-02, LAYA-06).
Runs in `make test` under the ray marker (CPU only, no weights or data).
"""
import os
import subprocess
import sys

import pytest

import pilot.data
from conftest import make_raw
from pilot import runs
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
def cpu_run(tmp_path_factory):
    """Checkpoint dir of a 2-epoch full-encoder ray_train.py run on --device cpu."""
    d = tmp_path_factory.mktemp("train_cpu")
    root, raw_sha256 = make_raw(str(d / "raw"))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    base = tiny_checkpoint(d / "base")
    ckpt = d / "ckpt"
    _run("ray_train.py", "--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
         "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--out", ckpt, "--device", "cpu",
         "--workers", "1", "--epochs", "2", "--max-messages", "12", "--calib-frac", "0.2", "--storage", d / "ray")
    return str(ckpt)


@pytest.mark.timeout(600)
def test_cpu_record(cpu_run):
    rec = runs.read_record(cpu_run)
    assert rec["device"] == "cpu" and rec["param_device"] == "cpu"
    assert rec["grad_checkpointing"] is True and rec["amp"] is False
    assert rec["env"] == [] and rec["recommended_max_memory_bytes"] is None
    assert rec["adamw_foreach"] is None and rec["adamw_fused"] is None
    assert [row["epoch"] for row in rec["epochs"]] == [1, 2]
    for row in rec["epochs"]:
        assert tuple(sorted(row)) == tuple(sorted(runs.EPOCH_FIELDS)), row
        assert row["train_s"] > 0 and row["optimizer_steps"] >= 1 and row["n_items"] > 0
        assert row["peak_rss_bytes"] > 0 and row["peak_total_bytes"] == row["peak_rss_bytes"]
        assert row["peak_driver_bytes"] is None and row["current_allocated_bytes"] is None
        assert isinstance(row["swap_delta_bytes"], int)
    assert rec["epochs"][0]["save_s"] > 0 and rec["epochs"][1]["save_s"] == 0.0   # the last epoch saves after calib
    assert rec["load_s"] > 0 and rec["calib_s"] > 0 and rec["final_save_s"] > 0


@pytest.mark.timeout(600)
def test_cpu_record_keeps_local_paths_and_raw_env_out(cpu_run):
    import json
    with open(os.path.join(cpu_run, "rl_agent_config.json"), encoding="utf-8") as f:
        ray_cfg = json.load(f)["training"]["ray_train"]
    assert not {"data", "base_dir", "driver_node_id", "expect_env"} & set(ray_cfg), sorted(ray_cfg)
    assert ray_cfg["device"] == "cpu" and ray_cfg["cpus_per_worker"] == 1


@pytest.mark.timeout(600)
def test_neg_items_match_the_count(cpu_run):
    """Default --neg 8: train items are the sampled positives + 8 negatives; calibration keeps all 48 labels."""
    import json
    rec = runs.read_record(cpu_run)
    assert rec["neg"] == 8 and rec["neg_seed"] == 0
    assert rec["n_train_items"] == rec["n_train_items_counted"] > 0
    assert rec["n_train_items"] < 48 * rec["n_train_messages"]
    assert rec["items_skipped"] == {}
    assert all(row["n_items"] == rec["n_train_items"] for row in rec["epochs"])   # one worker: the whole shard
    with open(os.path.join(cpu_run, "rl_agent_config.json"), encoding="utf-8") as f:
        holdout = json.load(f)["training"]["ray_train"]["calibration_holdout"]
    assert holdout["n_messages"] > 0 and holdout["n_items"] == 48 * holdout["n_messages"]
