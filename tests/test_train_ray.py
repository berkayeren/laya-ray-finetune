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
def tiny_train(tmp_path_factory):
    """train(name, *flags) -> checkpoint dir of a full-encoder ray_train.py run of the tiny base on --device cpu."""
    d = tmp_path_factory.mktemp("train_cpu")
    root, raw_sha256 = make_raw(str(d / "raw"))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    base = tiny_checkpoint(d / "base")

    def train(name, *flags):
        ckpt = d / name
        _run("ray_train.py", "--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
             "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--out", ckpt, "--device", "cpu",
             "--workers", "1", "--max-messages", "12", "--calib-frac", "0.2", "--storage", d / "ray", *flags)
        return str(ckpt)
    return train


@pytest.fixture(scope="module")
def cpu_run(tiny_train):
    """Checkpoint dir of a 2-epoch full-encoder run at the default flags (neg 8, rung none)."""
    return tiny_train("ckpt", "--epochs", "2")


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
    # every epoch saves its train state (and export), the last one too; the final checkpoint saves after calibration
    assert all(row["save_s"] > 0 for row in rec["epochs"])
    assert [row["segment"] for row in rec["epochs"]] == [0, 0] and not any(row["resumed"] for row in rec["epochs"])
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


@pytest.mark.timeout(600)
def test_default_run_is_rung_none(cpu_run):
    rec = runs.read_record(cpu_run)
    assert rec["rung"] == "none" and rec["fallback"] is False and rec["weaker"] is False
    assert 0 < rec["trainable_params"] == rec["total_params"]
    # the tiny base keeps its own max_len, which the record names as a departure from the spike's 128
    assert rec["spike_departures"] == {"max_len": 64}


@pytest.mark.timeout(600)
def test_embeddings_frozen_run_is_labelled(tiny_train):
    rec = runs.read_record(tiny_train("ckpt_emb", "--epochs", "1", "--freeze-embeddings"))
    assert rec["rung"] == "embeddings_frozen" and rec["fallback"] is True and rec["weaker"] is False
    assert 0 < rec["trainable_params"] < rec["total_params"]


OOM_DRIVER = """
import ray, ray_train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def loop(c):
    with ray_train.mps_oom_exit(2, 17, "none"):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 9.10 GiB, other allocations: 1.02 GiB)")


ray.init(num_cpus=2, include_dashboard=False)
ray_train.fit_or_exit(TorchTrainer(loop, scaling_config=ScalingConfig(num_workers=1),
                                   run_config=RunConfig(name="oom", storage_path=%r)))
"""


@pytest.mark.timeout(600)
def test_worker_oom_ends_the_driver_with_one_line(tmp_path):
    """A worker MPS OOM crosses the Ray process boundary and ends the driver with the rung message, exit 1."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", OOM_DRIVER % str(tmp_path / "ray")], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 1, r.stderr[-4000:]
    # SystemExit prints the bare line; Ray's forwarded worker logs (prefixed, asynchronous) may follow it
    assert runs.oom_message(2, 17, "none") in r.stderr.splitlines(), r.stderr[-4000:]
