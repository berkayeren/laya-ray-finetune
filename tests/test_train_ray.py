"""ray_train.py --device cpu trains the tiny checkpoint's full encoder in one Ray worker and writes its run record.

Offline (tests/tiny_laya.py + the synthetic splits of tests/conftest.py): 2 epochs on 12 messages with the encoder
trainable, so gradient checkpointing is on and AMP off. The checkpoint config's training.ray_train.record must name the
device the parameters landed on and hold one row per epoch with exactly pilot.runs.EPOCH_FIELDS.
Every selected message trains (no calibration holdout), and every export and final checkpoint is uncalibrated at
T = 1, also from a base that ships another temperature.
The Laya val selection chain runs end to end on the tiny checkpoint: a selection run's epoch exports scored on val
build a postfit record, a --postfit final seed on final_train serves the record's val-fitted T, and
--stop-after-epoch K reproduces the selection run's epoch K.
Runs in `make test` under the ray marker (CPU only, no weights or data).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pilot.data
from conftest import make_raw
from pilot import defaults, dumps, postfit, runs
from pilot.data import sha256_file
from pilot.weights import identity
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


def _run(*args: Any) -> subprocess.CompletedProcess:
    """Run a repo script with the test interpreter (no PYTHONPATH, so it imports like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


@pytest.fixture(scope="module")
def tiny_train(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """train(name, *flags, base=None, split="selection_train") -> checkpoint dir of a full-encoder ray_train.py run on
    --device cpu, from the tiny base (temperature 1.0) unless `base` names another local checkpoint. train.storage is
    the Ray storage dir, train.work the module's work dir, train.data the prepared synthetic snapshot (splits/ and
    splits.json)."""
    d = tmp_path_factory.mktemp("train_cpu")
    root, raw_sha256 = make_raw(str(d / "raw"))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    tiny = tiny_checkpoint(d / "base")

    def train(name: str, *flags: Any, base: str | None = None, split: str = "selection_train") -> str:
        ckpt = d / name
        _run("ray_train.py", "--data", os.path.join(out_dir, "splits", split + ".jsonl"),
             "--manifest", os.path.join(out_dir, "splits.json"), "--base", base or tiny, "--out", ckpt,
             "--device", "cpu", "--workers", "1", "--max-messages", "12", "--storage", d / "ray", *flags)
        return str(ckpt)
    train.storage = str(d / "ray")
    train.work = d
    train.data = out_dir
    return train


@pytest.fixture(scope="module")
def cpu_run(tiny_train: Any) -> str:
    """Checkpoint dir of a 2-epoch full-encoder run at the default flags (the tuned default neg, rung none)."""
    return tiny_train("ckpt", "--epochs", "2")


@pytest.mark.timeout(600)
def test_cpu_record(cpu_run: str) -> None:
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
    # every epoch saves its train state (and export), the last one too; the final checkpoint saves after training
    assert all(row["save_s"] > 0 for row in rec["epochs"])
    assert [row["segment"] for row in rec["epochs"]] == [0, 0] and not any(row["resumed"] for row in rec["epochs"])
    # training no longer calibrates: the key stays, with None
    assert rec["calib_s"] is None
    assert rec["load_s"] > 0 and rec["final_save_s"] > 0


@pytest.mark.timeout(600)
def test_cpu_record_keeps_local_paths_and_raw_env_out(cpu_run: str) -> None:
    with open(os.path.join(cpu_run, "rl_agent_config.json"), encoding="utf-8") as f:
        ray_cfg = json.load(f)["training"]["ray_train"]
    assert not {"data", "base_dir", "driver_node_id", "expect_env"} & set(ray_cfg), sorted(ray_cfg)
    assert ray_cfg["device"] == "cpu" and ray_cfg["cpus_per_worker"] == 1


@pytest.mark.timeout(600)
def test_neg_items_match_the_count(cpu_run: str) -> None:
    """The default --neg (the tuned pilot.defaults.LAYA neg, 16): train items are the sampled positives + that many
    negatives, on every one of the 12 selected messages (no calibration holdout)."""
    rec = runs.read_record(cpu_run)
    assert rec["neg"] == defaults.LAYA["neg"] and rec["neg_seed"] == 0
    assert rec["n_train_messages"] == 12                    # the whole --max-messages subsample trains
    assert rec["n_train_items"] == rec["n_train_items_counted"] > 0
    assert rec["n_train_items"] < 48 * rec["n_train_messages"]
    assert rec["items_skipped"] == {}
    assert all(row["n_items"] == rec["n_train_items"] for row in rec["epochs"])   # one worker: the whole shard
    with open(os.path.join(cpu_run, "rl_agent_config.json"), encoding="utf-8") as f:
        ray_cfg = json.load(f)["training"]["ray_train"]
    assert "calibration_holdout" not in ray_cfg, sorted(ray_cfg)


def _config(path: str | Path) -> dict[str, Any]:
    """The rl_agent_config.json of a checkpoint dir."""
    with open(os.path.join(str(path), "rl_agent_config.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.timeout(600)
def test_uncalibrated_from_a_base_with_another_temperature(tiny_train: Any) -> None:
    """A base that ships T 1.3 never leaks it: the epoch export and the final checkpoint both reset the temperature to
    [1, 1, 1] with no temperature_by_options and say calibrated False."""
    base = tiny_checkpoint(tiny_train.work / "base_t13", temperature=1.3)
    assert _config(base)["temperature"] == [1.0, 1.0, 1.3]
    ckpt = tiny_train("ckpt_t13", "--epochs", "1", base=base)
    rid = runs.read_record(ckpt)["run_id"]
    for cfg in (_config(ckpt), _config(os.path.join(tiny_train.storage, rid, "exports", "epoch-001"))):
        assert cfg["temperature"] == [1.0, 1.0, 1.0], cfg["temperature"]
        assert cfg["temperature_by_options"] == {}
        assert cfg["training"]["calibrated"] is False and cfg["fine_tuned"] is True
        assert "laya_train_calibration" not in cfg["training"]


@pytest.mark.timeout(600)
def test_default_run_is_rung_none(cpu_run: str) -> None:
    rec = runs.read_record(cpu_run)
    assert rec["rung"] == "none" and rec["fallback"] is False and rec["weaker"] is False
    assert 0 < rec["trainable_params"] == rec["total_params"]
    # the tiny base keeps its own max_len, which the record names as a departure from the spike's 128
    assert rec["spike_departures"] == {"max_len": 64}


@pytest.mark.timeout(600)
def test_embeddings_frozen_run_is_labelled(tiny_train: Any) -> None:
    rec = runs.read_record(tiny_train("ckpt_emb", "--epochs", "1", "--freeze-embeddings"))
    assert rec["rung"] == "embeddings_frozen" and rec["fallback"] is True and rec["weaker"] is False
    assert 0 < rec["trainable_params"] < rec["total_params"]


OOM_DRIVER = """
import ray, ray_train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def loop(c: dict) -> None:
    with ray_train.mps_oom_exit(2, 17, "none"):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 9.10 GiB, other allocations: 1.02 GiB)")


ray.init(num_cpus=2, include_dashboard=False)
ray_train.fit_or_exit(TorchTrainer(loop, scaling_config=ScalingConfig(num_workers=1),
                                   run_config=RunConfig(name="oom", storage_path=%r)))
"""


@pytest.mark.timeout(600)
def test_worker_oom_ends_the_driver_with_one_line(tmp_path: Path) -> None:
    """A worker MPS OOM crosses the Ray process boundary and ends the driver with the rung message, exit 1."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", OOM_DRIVER % str(tmp_path / "ray")], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 1, r.stderr[-4000:]
    # SystemExit prints the bare line; Ray's forwarded worker logs (prefixed, asynchronous) may follow it
    assert runs.oom_message(2, 17, "none") in r.stderr.splitlines(), r.stderr[-4000:]


def _score_val(train: Any, ckpt: str) -> str:
    """score.py --workers 0 of a checkpoint on the synthetic val split into train.work/dumps; returns the dump dir."""
    root = str(train.work / "dumps")
    _run("score.py", os.path.join(train.data, "splits", "val.jsonl"), "--checkpoint", ckpt, "--workers", "0",
         "--cpus-per-worker", "1", "--manifest", os.path.join(train.data, "splits.json"), "--out-root", root)
    return os.path.join(root, dumps.run_id("laya", identity(ckpt)))


@pytest.mark.timeout(1200)
def test_postfit_final_seed_end_to_end(tiny_train: Any) -> None:
    """The Laya val selection on the tiny checkpoint: a 2-epoch selection run's two epoch exports, scored on val,
    give a postfit record (one-LR grid); a --postfit final seed on final_train trains the chosen recipe and epoch,
    serves the record's T bit for bit, records the binding, and its own (in-sample) val dump is refused by postfit."""
    manifest = os.path.join(tiny_train.data, "splits.json")
    sel = tiny_train("sel_2e5", "--epochs", "2", "--encoder-lr", "2e-05", "--head-lr", "0.0001")   # postfit.HEAD_LR
    rid = runs.read_record(sel)["run_id"]
    run_dirs = [_score_val(tiny_train, os.path.join(tiny_train.storage, rid, "exports", "epoch-%03d" % e))
                for e in (1, 2)]
    record = postfit.fit(run_dirs, manifest, lr_grid=(2e-05,), epochs=2)
    path = postfit.write_record(record, str(tiny_train.work / "postfit" / "laya.json"))
    sha, chosen = sha256_file(path), record["chosen"]
    assert chosen["train_run_id"] == rid and chosen["stop_after_epoch"] == (1 if chosen["epoch"] == 1 else None)
    T = chosen["temperature"]
    stop = ["--stop-after-epoch", "1"] if chosen["stop_after_epoch"] == 1 else []
    final = tiny_train("final_s1", "--epochs", "2", "--encoder-lr", "2e-05", "--head-lr", "0.0001", "--seed", "1",
                       "--postfit", path, "--no-export-epochs", *stop, split="final_train")

    cfg = _config(final)
    assert cfg["temperature"] == [1.0, 1.0, T] and cfg["temperature"][2] == T      # bit for bit, never rounded
    assert cfg["temperature_by_options"] == {}
    tr = cfg["training"]
    assert tr["calibrated"] is True and tr["epoch"] == chosen["epoch"]
    assert tr["postfit"]["sha256"] == sha and tr["postfit"]["temperature"] == T
    assert tr["postfit"]["chosen"]["train_run_id"] == rid and tr["postfit"]["theta"] == chosen["theta"]
    rec = runs.read_record(final)
    assert rec["postfit"] == {"sha256": sha, "chosen": tr["postfit"]["chosen"], "temperature": T}
    assert rec["canonical"]["split"] == "final_train" and rec["canonical"]["seed"] == 1
    assert rec["canonical"]["postfit_sha256"] == sha
    assert rec["canonical"].get("stop_after_epoch") == chosen["stop_after_epoch"]
    assert [row["epoch"] for row in rec["epochs"]] == list(range(1, chosen["epoch"] + 1))
    assert rec["exports"] == []                                                    # --no-export-epochs

    final_dump = _score_val(tiny_train, final)
    with open(os.path.join(final_dump, "val.json"), encoding="utf-8") as f:
        side = json.load(f)
    assert side["seed"] == 1 and side["training"]["postfit_sha256"] == sha
    assert side["training"]["calibrated"] is True and side["training"]["split"] == "final_train"
    assert abs(side["temperature"]["shipped"] - T) < 1e-6, (side["temperature"], T)
    with pytest.raises(SystemExit, match="selection_train"):                       # a final seed's val is in-sample
        postfit.candidate(final_dump, manifest)


@pytest.mark.timeout(600)
def test_stop_after_epoch_keeps_the_selection_trajectory(tiny_train: Any, cpu_run: str) -> None:
    """--stop-after-epoch 1 of the --epochs 2 schedule is the selection run's epoch 1 (the cosine T_max and
    the sigma anneal keep --epochs), so on cpu its final weights are byte for byte cpu_run's exports/epoch-001."""
    stopped = tiny_train("ckpt_stop1", "--epochs", "2", "--stop-after-epoch", "1")
    rid = runs.read_record(cpu_run)["run_id"]
    export = os.path.join(tiny_train.storage, rid, "exports", "epoch-001")
    with open(os.path.join(stopped, "model.safetensors"), "rb") as f, \
            open(os.path.join(export, "model.safetensors"), "rb") as g:
        assert f.read() == g.read()
    rec = runs.read_record(stopped)
    assert [row["epoch"] for row in rec["epochs"]] == [1] and rec["run_id"] != rid
    assert rec["canonical"]["stop_after_epoch"] == 1 and rec["canonical"]["epochs"] == 2 and rec["postfit"] is None
    cfg = _config(stopped)
    assert cfg["training"]["epoch"] == 1 and cfg["training"]["calibrated"] is False
    assert cfg["temperature"] == [1.0, 1.0, 1.0] and "postfit" not in cfg["training"]
