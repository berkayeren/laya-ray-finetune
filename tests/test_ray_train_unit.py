"""ray_train.py helpers that run inside Ray workers, tested without starting Ray or loading a model."""
import hashlib
import json

import pytest

ray_train = pytest.importorskip("ray_train")                # imports ray, torch and laya (the locked env has them)

pytestmark = pytest.mark.unit


def _write(path, rows):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_read_verified_rows_parses_the_verified_bytes(tmp_path):
    rows = [{"uid": "a", "state": "Kartım kayboldu"}, {"uid": "b", "state": "Evet."}]
    digest = _write(tmp_path / "s.jsonl", rows)
    assert ray_train.read_verified_rows(str(tmp_path / "s.jsonl"), digest) == rows


def test_read_verified_rows_refuses_a_file_changed_after_driver_verification(tmp_path):
    digest = _write(tmp_path / "s.jsonl", [{"uid": "a", "state": "x"}])
    _write(tmp_path / "s.jsonl", [{"uid": "a", "state": "y"}])
    with pytest.raises(SystemExit, match="verified on the driver"):
        ray_train.read_verified_rows(str(tmp_path / "s.jsonl"), digest)


def test_require_single_node_refuses_a_multi_node_cluster():
    ray_train.require_single_node([{"Alive": True}, {"Alive": False}])          # one alive node: fine
    with pytest.raises(SystemExit, match="single-node"):
        ray_train.require_single_node([{"Alive": True}, {"Alive": True}])


def test_holdout_messages_is_message_level_deterministic_and_disjoint():
    rows = [{"uid": "m%d" % i, "state": "t%d" % i} for i in range(100)]
    train, calib = ray_train.holdout_messages(rows, 40, 0.1, 0)
    assert len(calib) == 10                                  # min(40 messages, 10% of 100)
    assert {r["uid"] for r in train}.isdisjoint(r["uid"] for r in calib)
    assert len(train) + len(calib) == len(rows)
    again = ray_train.holdout_messages([dict(r) for r in rows], 40, 0.1, 0)   # what another rank computes
    assert again == (train, calib)
    assert len(ray_train.holdout_messages(rows, 3, 0.1, 0)[1]) == 3          # --calib-max-messages caps it


def test_read_verified_rows_keeps_unicode_line_separators_inside_messages(tmp_path, synthetic_raw):
    # json.dumps(ensure_ascii=False) leaves U+2028/U+0085/U+2029 raw; prepare() writes them, workers must read them
    from pilot import laya_adapter as la
    from conftest import LINE_SEPARATOR_TEXTS, prepared_with_line_separators
    import pilot.data
    out_dir, manifest_path = prepared_with_line_separators(tmp_path, synthetic_raw)
    path = str(tmp_path / "data" / "splits" / "final_train.jsonl")
    entry = pilot.data.verify_split_file(path, manifest_path=manifest_path, roles=("train",))
    rows = ray_train.read_verified_rows(path, entry["sha256"])
    assert rows == la.read_jsonl(path) and len(rows) == entry["n_messages"]
    assert {r["state"] for r in rows} >= set(LINE_SEPARATOR_TEXTS[:2])


def test_run_name_is_per_split_seed_and_start_time():
    entry = {"path": "splits/final_train.jsonl"}
    name = ray_train.run_name(entry, 3, now=0)
    assert name.startswith("laya-final_train-s3-") and name != ray_train.run_name(entry, 3, now=3600)
    assert ray_train.run_name({"path": "splits/selection_train.jsonl"}, 3, now=0) != name


def test_intermediate_checkpoint_config_is_marked_and_traceable():
    base = {"encoder": "x", "temperature": 1.3, "training": {"base_run": 1}}
    ray_cfg = {"provenance": {"data": "splits/selection_train.jsonl"}, "calibration_holdout": {"n_messages": 4}}
    out = ray_train.epoch_config(base, 128, 64, ray_cfg, 2)
    assert out["fine_tuned"] is True and out["max_len"] == 128 and out["head_max_len"] == 64
    assert out["training"] == {"base_run": 1, "ray_train": ray_cfg, "epoch": 2, "calibrated": False}
    assert base["training"] == {"base_run": 1}                                   # the base config is not mutated


def test_require_driver_node_refuses_a_worker_on_another_node():
    ray_train.require_driver_node("a" * 56, "a" * 56)
    with pytest.raises(SystemExit, match="single-node"):
        ray_train.require_driver_node("b" * 56, "a" * 56)


def _args(*argv):
    return ray_train.parser().parse_args(["--data", "d.jsonl", "--out", "o"] + list(argv))


def test_gpu_alias_resolves_to_cuda():
    assert ray_train.resolve(_args("--gpu")) == ray_train.resolve(_args("--device", "cuda"))
    for other in ("mps", "cpu"):
        with pytest.raises(ValueError, match="alias"):
            ray_train.resolve(_args("--gpu", "--device", other))


def test_resolve_defaults():
    r = ray_train.resolve(_args())
    assert r["device"] == "cpu" and r["amp"] is False and r["cpus_per_worker"] == 1
    assert r["grad_checkpointing"] is True and r["adamw_foreach"] is None and r["adamw_fused"] is None
    r = ray_train.resolve(_args("--device", "mps"))
    assert r["amp"] is False and r["cpus_per_worker"] == 4 and r["adamw_foreach"] is False and r["adamw_fused"] is False
    assert ray_train.resolve(_args("--device", "cuda"))["amp"] is True
    assert ray_train.resolve(_args("--device", "cuda", "--no-amp"))["amp"] is False
    assert ray_train.resolve(_args("--freeze-encoder"))["grad_checkpointing"] is False
    assert ray_train.resolve(_args("--no-grad-checkpointing"))["grad_checkpointing"] is False
    assert ray_train.resolve(_args("--device", "mps", "--cpus-per-worker", "2"))["cpus_per_worker"] == 2


def test_require_param_device_refuses_wrong_device():
    import torch
    model = torch.nn.Linear(2, 2)
    with pytest.raises(SystemExit, match="expected mps"):
        ray_train.require_param_device(model, "mps")
    ray_train.require_param_device(model, "cpu")


def test_check_env_refuses_a_missing_var(monkeypatch):
    monkeypatch.delenv("RAY_TRAIN_CANARY", raising=False)
    with pytest.raises(SystemExit, match="RAY_TRAIN_CANARY"):
        ray_train.check_env({"RAY_TRAIN_CANARY": "1"})
    monkeypatch.setenv("RAY_TRAIN_CANARY", "1")
    assert ray_train.check_env({"RAY_TRAIN_CANARY": "1"}) == {"RAY_TRAIN_CANARY": "1"}


@pytest.fixture
def tiny_run(tmp_path, synthetic_raw, monkeypatch):
    """argv for ray_train.main on the synthetic selection_train and a tiny local base; ray.init must never run."""
    import pilot.data
    from tiny_laya import tiny_checkpoint

    def no_ray(*a, **kw):
        raise AssertionError("ray.init reached; the device refusal must come first")
    monkeypatch.setattr(ray_train.ray, "init", no_ray)
    root, raw_sha256 = synthetic_raw()
    pilot.data.prepare(root, str(tmp_path / "data"), raw_sha256)
    return ["--data", str(tmp_path / "data" / "splits" / "selection_train.jsonl"),
            "--manifest", str(tmp_path / "data" / "splits.json"), "--base", tiny_checkpoint(tmp_path / "base"),
            "--out", str(tmp_path / "o"), "--storage", str(tmp_path / "ray"), "--workers", "1"]


def test_mps_unavailable_is_refused_before_ray(tiny_run, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="MPS is not available"):
        ray_train.main(tiny_run + ["--device", "mps"])
    assert not (tmp_path / "ray").exists() and not (tmp_path / "o").exists()


def test_cuda_unavailable_is_refused_before_ray(tiny_run, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for flag in (["--device", "cuda"], ["--gpu"]):
        with pytest.raises(SystemExit, match="CUDA is not available"):
            ray_train.main(tiny_run + flag)
    assert not (tmp_path / "ray").exists()
