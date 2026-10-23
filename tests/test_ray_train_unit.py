"""ray_train.py helpers that run inside Ray workers, tested without starting Ray or loading a model."""
import hashlib
import json
import os
import re

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


def _dests():
    return {act.dest for act in ray_train.parser()._actions if act.dest != "help"}


def test_every_flag_is_classified():
    """Every ray_train.py flag is in the run identity (directly or by its resolved value) or named as not changing
    training, so a new flag cannot silently drop out of the resume key."""
    from pilot import runs
    assert not set(runs.IDENTITY) & set(runs.NOT_IDENTITY)
    assert _dests() <= set(runs.IDENTITY) | set(runs.NOT_IDENTITY), sorted(
        _dests() - set(runs.IDENTITY) - set(runs.NOT_IDENTITY))
    assert set(runs.NOT_IDENTITY) <= _dests(), sorted(set(runs.NOT_IDENTITY) - _dests())


ENTRY = {"path": "splits/selection_train.jsonl", "sha256": "a" * 64, "role": "train"}
BASE_IDENT = {"repo": "r", "subfolder": "s", "revision": "c" * 40, "model_sha256": "b" * 64}


def _rid(*argv):
    from pilot import runs
    a = ray_train.parser().parse_args(list(argv))
    vals = ray_train.identity_values(a, ray_train.resolve(a), ENTRY, "multilingual", BASE_IDENT)
    return runs.run_id(runs.canonical(vals))


def test_run_id_ignores_paths_and_resources_and_follows_the_seed():
    base = ["--data", "d.jsonl", "--out", "o"]
    rid = _rid(*base)
    assert rid.startswith("laya-selection_train-neg8-s0-")
    for other in (["--out", "elsewhere"], ["--storage", "/x/ray"], ["--cpus-per-worker", "3"],
                  ["--manifest", "m.json"], ["--object-store-gb", "2"]):
        assert _rid(*(base + other)) == rid, other
    assert _rid("--out", "o", "--data", "d.jsonl") == rid                     # flag order never matters
    assert _rid("--seed", "3", "--neg", "2", *base) == _rid(*(base + ["--neg", "2", "--seed", "3"]))
    s1 = _rid(*(base + ["--seed", "1"]))
    assert s1 != rid and "-s1-" in s1
    assert _rid(*(base + ["--gpu"])) == _rid(*(base + ["--device", "cuda"])) != rid
    for other in (["--device", "mps"], ["--encoder-lr", "1e-5"], ["--neg", "0"], ["--epochs", "3"],
                  ["--freeze-embeddings"], ["--micro-batch", "4"], ["--neg-seed", "7"], ["--no-grad-checkpointing"]):
        assert _rid(*(base + other)) != rid, other
    assert "-full-" in _rid(*(base + ["--neg", "0"]))
    assert _rid(*(base + ["--neg-seed", "0"])) == rid                        # --neg-seed defaults to --seed


def test_the_committed_probe_run_id_is_historical():
    """The Phase 4 compute probe trained with the 40-message calibration holdout LAYA-05 removed (identity keys
    calib_max_messages and calib_frac), so its committed id laya-selection_train-neg8-s0-db11040d is a historical
    record: the same probe command now names another run, and results/compute/ stays byte-identical
    (tests/test_compute_report.py). base_files_sha256 is still None for --base multilingual (04-REVIEW WR-01: the pin
    fixes its tokenizer/ and encoder/) and canonical() leaves it out. Offline: the committed manifest, report and pin."""
    import os
    from pilot import runs
    from pilot.data import REPO_ROOT
    from pilot.weights import identity
    with open(os.path.join(REPO_ROOT, "results", "compute", "compute_checkpoint.json"), encoding="utf-8") as f:
        meta = json.load(f)["meta"]
    with open(os.path.join(REPO_ROOT, "data", "splits.json"), encoding="utf-8") as f:
        entry = json.load(f)["splits"]["selection_train"]
    assert entry["sha256"] == meta["split"]["sha256"]
    p, start = meta["protocol"], meta["protocol"]["spike_start"]
    a = ray_train.parser().parse_args([
        "--data", "d.jsonl", "--out", "o", "--base", "multilingual", "--device", p["device"], "--workers", "1",
        "--cpus-per-worker", str(p["cpus"]), "--epochs", str(p["epochs"]), "--micro-batch", str(start["micro_batch"]),
        "--grad-accum", str(start["grad_accum"]), "--max-len", str(start["max_len"]), "--neg", str(p["neg"]),
        "--seed", str(p["seed"])])
    vals = ray_train.identity_values(a, ray_train.resolve(a), entry, "multilingual", identity("multilingual"))
    canon = runs.canonical(vals)
    assert vals["base_files_sha256"] is None and "base_files_sha256" not in canon
    assert not [k for k in canon if k.startswith("calib")], sorted(canon)
    rid = runs.run_id(canon)
    assert re.fullmatch(r"laya-selection_train-neg8-s0-[0-9a-f]{8}", rid), rid
    assert p["run_id"] == "laya-selection_train-neg8-s0-db11040d" and rid != p["run_id"]


def test_a_local_base_names_the_run_by_its_tokenizer_and_encoder(tiny_run, tmp_path, capsys):
    """04-REVIEW WR-01: two local bases with the same weights and rl_agent_config.json but another tokenizer/ or
    encoder/ file get different --print-run-id output, so neither resumes nor returns the other's run."""
    import shutil
    from pilot import runs
    from pilot.weights import identity
    base = tiny_run[tiny_run.index("--base") + 1]

    def rid(b):
        argv = list(tiny_run)
        argv[argv.index("--base") + 1] = str(b)
        ray_train.main(argv + ["--print-run-id"])
        return capsys.readouterr().out.strip()

    first = rid(base)
    assert first == rid(base)                                                 # stable for the same base
    ids = {first}
    for rel in ("tokenizer/tokenizer.json", "encoder/config.json"):
        other = tmp_path / ("other-" + rel.split("/")[0])
        shutil.copytree(base, other)
        with open(other / rel, "a", encoding="utf-8") as f:
            f.write("\n")                                                     # same JSON, other bytes
        same, changed = identity(base), identity(str(other))
        assert (same["model_sha256"], same["config_sha256"]) == (changed["model_sha256"], changed["config_sha256"])
        ids.add(rid(other))
    assert len(ids) == 3 and all(re.fullmatch(r"laya-selection_train-neg8-s0-[0-9a-f]{8}", i) for i in ids), ids
    assert "base_files_sha256" in runs.IDENTITY and "base_files_sha256" in runs.DROPPED_WHEN_NONE


def test_the_worker_checks_the_tokenizer_config_rewrite_after_the_base_load():
    """04-REVIEW IN-01: laya.train.load_checkpoint only warns when its tokenizer_config.json rewrite fails, so the
    worker refuses an un-normalized file right after the load (tests/test_laya_adapter.py runs the refusal)."""
    import inspect
    src = inspect.getsource(ray_train.train_loop_per_worker)
    load = src.index('la.load_checkpoint(c["base_dir"])')
    assert load < src.index('check_tokenizer_config_rewritten(c["base_dir"], "base checkpoint")') < src.index(
        "load_s = time.perf_counter() - t_load")


def test_intermediate_checkpoint_config_is_marked_and_traceable():
    """An export resets the temperature to T = 1 with no buckets, never inheriting a base's fitted T (LAYA-05)."""
    base = {"encoder": "x", "temperature": [1.0, 1.0, 1.3], "temperature_by_options": {"noul:2": 1.3},
            "training": {"base_run": 1}}
    ray_cfg = {"provenance": {"data": "splits/selection_train.jsonl"}}
    out = ray_train.epoch_config(base, 128, 64, ray_cfg, 2)
    assert out["fine_tuned"] is True and out["max_len"] == 128 and out["head_max_len"] == 64
    assert out["temperature"] == [1.0, 1.0, 1.0] and out["temperature_by_options"] == {}
    assert out["temperature"] is not ray_train.UNCALIBRATED_TEMPERATURE       # a copy, never the shared list
    assert out["training"] == {"base_run": 1, "ray_train": ray_cfg, "epoch": 2, "calibrated": False}
    assert base == {"encoder": "x", "temperature": [1.0, 1.0, 1.3], "temperature_by_options": {"noul:2": 1.3},
                    "training": {"base_run": 1}}                                 # the base config is not mutated


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


def test_freeze_flags_conflict(capsys):
    base = ["--data", "d.jsonl", "--out", "o"]
    with pytest.raises(SystemExit) as e:
        ray_train.main(base + ["--freeze-embeddings", "--freeze-encoder"])
    assert e.value.code == 2 and "--freeze-encoder already freezes the embeddings" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        ray_train.main(base + ["--neg", "-1"])
    assert e.value.code == 2 and "--neg must be 0" in capsys.readouterr().err


@pytest.mark.parametrize("flags, msg", [
    (["--micro-batch", "0"], "--micro-batch must be at least 1"),
    (["--grad-accum", "-1"], "--grad-accum must be at least 1"),
    (["--workers", "0"], "--workers must be at least 1"),
    (["--cpus-per-worker", "0"], "--cpus-per-worker must be at least 1"),
    (["--max-messages", "0"], "--max-messages must be at least 1"),
    (["--max-len", "0"], "--max-len must be at least 1"),                    # 04-REVIEW IN-10
    (["--head-max-len", "-5"], "--head-max-len must be at least 1"),
])
def test_bad_sizes_are_refused_before_ray(flags, msg, capsys):
    """04-REVIEW WR-03: refused by argparse, before the split check, the fetch and ray.init (not in the worker)."""
    with pytest.raises(SystemExit) as e:
        ray_train.main(["--data", "d.jsonl", "--out", "o"] + flags)
    assert e.value.code == 2 and msg in capsys.readouterr().err


@pytest.mark.parametrize("flag, value", [("--calib-frac", "0.2"), ("--calib-max-messages", "40"),
                                         ("--calib-batch", "16")])
def test_removed_calibration_flags_are_refused(flag, value, capsys):
    """LAYA-05: training no longer holds out calibration messages, so the calibration flags are gone, not ignored."""
    with pytest.raises(SystemExit) as e:
        ray_train.parser().parse_args(["--data", "d.jsonl", "--out", "o", flag, value])
    assert e.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


def test_mps_oom_becomes_the_rung_message():
    from pilot import runs
    with pytest.raises(RuntimeError) as e:
        with ray_train.mps_oom_exit(2, 17, "none"):
            raise RuntimeError("MPS backend out of memory (MPS allocated: 9.10 GiB, other allocations: 1 GiB)")
    assert str(e.value) == runs.oom_message(2, 17, "none") and e.value.__cause__ is not None
    with pytest.raises(RuntimeError, match="^shape mismatch$"):               # any other error propagates unchanged
        with ray_train.mps_oom_exit(2, 17, "none"):
            raise RuntimeError("shape mismatch")


def test_driver_turns_the_worker_oom_into_one_line():
    from ray.train import WorkerGroupError
    from pilot import runs
    msg = runs.oom_message(1, 5, "embeddings_frozen")

    class Trainer:
        def __init__(self, err):
            self.err = err

        def fit(self):
            raise self.err
    with pytest.raises(SystemExit) as e:
        ray_train.fit_or_exit(Trainer(WorkerGroupError("[Rank 0]\nRuntimeError: %s\n" % msg, {})))
    assert e.value.code == msg
    other = WorkerGroupError("[Rank 0]\nValueError: bad batch\n", {})
    with pytest.raises(SystemExit) as e:
        ray_train.fit_or_exit(Trainer(other), "ray/laya-x")
    assert e.value.code == ("training failed (ValueError: bad batch); rerun the same command to resume from the last "
                            "epoch checkpoint in ray/laya-x")
    with pytest.raises(KeyError):                                            # not a Ray failure: unchanged
        ray_train.fit_or_exit(Trainer(KeyError("x")), "ray/laya-x")


def test_a_worker_refusal_is_shown_as_is_not_as_resume_advice():
    """04-REVIEW CR-01: a SystemExit in the worker (e.g. the tokenizer-config backstop after the base load) names its own
    fix; "rerun the same command to resume" would only refuse again. The text is Ray 2.59's WorkerGroupError form."""
    from ray.train import WorkerGroupError
    msg = ("base checkpoint: laya could not rewrite tokenizer/tokenizer_config.json on load (a read-only dir, or not a "
           "JSON object), so the tokenizer it loaded is not the one the identity binds; make the checkpoint dir "
           "writable (or copy it) and rerun")

    class Trainer:
        def fit(self):
            raise WorkerGroupError("Training failed due to worker errors:\n[Rank 0 Error Snippet]:\nTraceback (most "
                                   "recent call last):\n  File \"ray_train.py\", line 540, in train_loop_per_worker\n"
                                   "    check_tokenizer_config_rewritten(c[\"base_dir\"], \"base checkpoint\")\n"
                                   "SystemExit: %s\n" % msg, {})
    with pytest.raises(SystemExit) as e:
        ray_train.fit_or_exit(Trainer(), "ray/laya-x")
    assert e.value.code == msg and "rerun the same command" not in e.value.code
    assert ray_train.worker_refusal("ValueError: bad batch\nSystemExit: \n") is None


def test_a_raw_mps_oom_outside_the_step_gets_the_rung_line_not_resume_advice():
    """04-REVIEW WR-01: an MPS OOM no mps_oom_exit wrapped (model move, resume, save) must not say "rerun
    to resume" (the rerun resumes into the same allocation); it gets the rung line the compute ladder walks on."""
    from ray.train import WorkerGroupError
    from pilot import runs

    class Trainer:
        def fit(self):
            raise WorkerGroupError("[Rank 0]\nTraceback (most recent call last):\n  File \"x.py\"\nRuntimeError: MPS "
                                   "backend out of memory (MPS allocated: 9.10 GiB, other allocations: 1 GiB)\n", {})
    with pytest.raises(SystemExit) as e:
        ray_train.fit_or_exit(Trainer(), "ray/laya-x", "embeddings_frozen")
    line = runs.oom_message(None, None, "embeddings_frozen", "outside the training step")
    assert e.value.code == line and runs.failure_line(line) == line
    assert "rerun the same command" not in line and "--micro-batch 4" in line
    with pytest.raises(RuntimeError) as e:                                   # a wrapped stage names itself
        with ray_train.mps_oom_exit(3, None, "none", "while saving the checkpoint"):
            raise RuntimeError("MPS backend out of memory (MPS allocated: 9.10 GiB)")
    assert str(e.value) == runs.oom_message(3, None, "none", "while saving the checkpoint")
    assert str(e.value).startswith(runs.OOM_MARK + " while saving the checkpoint at epoch 3 (rung none)")


def test_a_resume_with_no_epoch_left_never_loads_the_optimizer_state():
    """04-REVIEW WR-01 (end-of-phase gate): a rerun after a final-save OOM resumes a train state that
    already holds the last epoch. Its AdamW moments must stay off the device, and an OOM while it moves the model must
    say rerun the same command, never a rung flag that would change the run_id and retrain every epoch."""
    import torch
    from pilot import runs
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    model(torch.ones(1, 3)).sum().backward()
    opt.step()
    sched.step()
    saved = {"epoch": 2, "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": None}

    def fresh():
        o = torch.optim.AdamW(model.parameters())
        return o, torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=4)
    o, s = fresh()
    assert ray_train.restore_optimizer(saved, 2, o, s, None, "none") is False
    assert not o.state and s.last_epoch == 0                # no AdamW moment materialised, the scheduler untouched
    o, s = fresh()
    assert ray_train.restore_optimizer(saved, 3, o, s, None, "none") is True
    assert len(o.state) == 2 and s.last_epoch == 1          # an epoch left: weight and bias moments, scheduler step
    assert ray_train.training_done(saved, 2) and not ray_train.training_done(saved, 3)
    assert not ray_train.training_done(None, 2)             # a fresh start
    for stage in (runs.STAGE_FINALIZE_MOVE, runs.STAGE_FINAL_SAVE):
        line = runs.oom_message(None, None, "none", stage)
        assert runs.resume_oom(line) and "--freeze-embeddings" not in line


def test_the_trainer_passes_the_resume_stages_by_name():
    """04-REVIEW IN-01: oom_message matches a stage text exactly, so a reworded copy in ray_train.py would silently turn
    a post-training OOM back into next-rung advice. The trainer names every resume stage through pilot.runs and holds
    no copy of any stage text."""
    import inspect
    from pilot import runs
    names = {"STAGE_FINAL_SAVE", "STAGE_FINALIZE_MOVE"}
    assert set(runs.RESUME_STAGES) == {getattr(runs, n) for n in names}
    loop = inspect.getsource(ray_train.train_loop_per_worker)
    for n in names:
        assert "runs.%s" % n in loop, n
    src = inspect.getsource(ray_train)
    for stage in runs.RESUME_STAGES:
        assert stage not in src, stage


# laya's train-split / item-level calibration path and the holdout that fed it (LAYA-05)
CALIBRATION_NAMES = {"holdout_messages", "split_calibration", "calibration_records", "calibration_report",
                     "fit_temperature_map", "calib_items"}


def test_no_train_split_calibration():
    """LAYA-05 guard: the temperature comes only from message-level val data (postfit.py), so ray_train.py holds no
    calibration holdout, no laya calibration call and no calibration flag, and both the epoch exports and the final
    checkpoint write UNCALIBRATED_TEMPERATURE."""
    import ast
    import inspect
    with open(ray_train.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in CALIBRATION_NAMES:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in CALIBRATION_NAMES:
            found.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in CALIBRATION_NAMES:
            found.add(node.name)
    assert not found, sorted(found)
    assert not [d for d in _dests() if d.startswith("calib")], sorted(_dests())
    for fn in (ray_train.epoch_config, ray_train.train_loop_per_worker):
        assert "UNCALIBRATED_TEMPERATURE" in inspect.getsource(fn), fn.__name__
    assert ray_train.UNCALIBRATED_TEMPERATURE == [1.0, 1.0, 1.0]


def test_driver_names_the_dir_of_a_corrupted_snapshot():
    from ray.train import ControllerError
    from ray.train.v2._internal.exceptions import CheckpointManagerInitializationError

    class Trainer:
        def fit(self):
            raise ControllerError(CheckpointManagerInitializationError("snapshot references a missing checkpoint"))
    with pytest.raises(SystemExit) as e:
        ray_train.fit_or_exit(Trainer(), "ray/laya-x")
    assert e.value.code == ("ray/laya-x holds a Ray snapshot whose checkpoints are gone; delete that dir to start the "
                            "run fresh")


def test_resume_drops_exports_of_a_killed_segment_beyond_the_resumed_epoch(tmp_path):
    """04-REVIEW WR-02: an export written before its epoch registered (then a kill) is off the resumed trajectory."""
    d = tmp_path / "exports"
    for n in ("epoch-001", "epoch-002", "epoch-003", "epoch-003.tmp"):
        (d / n).mkdir(parents=True)
        (d / n / "model.safetensors").write_bytes(b"x")
    (d / "notes").mkdir()                                                     # not an export: left alone
    assert ray_train.prune_exports(str(d), 1) == ["epoch-002", "epoch-003", "epoch-003.tmp"]
    assert sorted(p.name for p in d.iterdir()) == ["epoch-001", "notes"]
    assert ray_train.prune_exports(str(d), 0) == ["epoch-001"]               # a fresh start keeps none
    assert ray_train.prune_exports(str(tmp_path / "missing"), 0) == []


def test_code_state_reads_git():
    st = ray_train.code_state()
    assert set(st) == {"git_sha", "git_dirty", "git_diff_sha256"}
    assert st["git_sha"] is None or len(st["git_sha"]) == 40
    assert st["git_diff_sha256"] is None or len(st["git_diff_sha256"]) == 64


def test_code_changed_on_resume_sees_uncommitted_edits():
    """04-REVIEW WR-04: the same HEAD with another uncommitted diff is a code change; unknown git state is flagged too.
    Flagged only (RESEARCH A3): nothing here refuses a resume."""
    def seg(sha="a" * 40, diff="d" * 64):
        return {"git_sha": sha, "git_dirty": diff != "e" * 64, "git_diff_sha256": diff}
    assert ray_train.code_changed([seg()]) is False                          # one segment: nothing to compare
    assert ray_train.code_changed([seg(), seg()]) is False
    assert ray_train.code_changed([seg(), seg(diff="f" * 64)]) is True       # edited, not committed, between segments
    assert ray_train.code_changed([seg(), seg(sha="b" * 40)]) is True        # another commit
    assert ray_train.code_changed([seg(sha=None, diff=None), seg(sha=None, diff=None)]) is True   # no git: unknown
    old = {"git_sha": "a" * 40, "git_dirty": False}                          # an older train state's segment
    assert ray_train.code_changed([old, seg()]) is True


def test_an_epoch_export_embeds_a_pre_save_snapshot():
    """04-REVIEW IN-06: the export lists itself, and the values only its save measures are None, not stale."""
    rows = [{"epoch": 1, "save_s": 4.0}, {"epoch": 2, "save_s": 0.0}]
    record = {"exports": ["exports/epoch-001"], "train_state_bytes": 123, "export_bytes": 45, "epochs": rows,
              "rung": "none"}
    snap = ray_train.export_record(record, "epoch-002")
    assert snap["exports"] == ["exports/epoch-001", "exports/epoch-002"]
    assert snap["train_state_bytes"] is None and snap["export_bytes"] is None and snap["rung"] == "none"
    assert snap["epochs"] == [{"epoch": 1, "save_s": 4.0}, {"epoch": 2, "save_s": None}]
    assert record["epochs"][1]["save_s"] == 0.0 and record["train_state_bytes"] == 123     # the live record is untouched


def test_code_state_dirty_flag_and_digest_read_the_same_files(tmp_path, monkeypatch):
    """04-REVIEW IN-08: a .planning/ edit leaves both clean; an untracked .py file or a code edit moves both."""
    import os
    import subprocess
    # 04-REVIEW IN-04: no global or system git config (commit.gpgsign, core.hooksPath, core.excludesFile) reaches the
    # temp repo, here or in code_state's own git calls, which inherit this env
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    (repo / ".planning").mkdir(parents=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / ".planning" / "STATE.md").write_text("one\n", encoding="utf-8")

    def git(*args):
        subprocess.run(("git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false",
                        "-c", "core.hooksPath=%s" % os.devnull) + args, cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(ray_train, "REPO_ROOT", str(repo))
    empty = hashlib.sha256(b"").hexdigest()
    st = ray_train.code_state()
    assert st["git_dirty"] is False and st["git_diff_sha256"] == empty and len(st["git_sha"]) == 40
    (repo / ".planning" / "STATE.md").write_text("two\n", encoding="utf-8")       # planning docs are not code
    assert ray_train.code_state() == st
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")                       # a new module an import can see
    new = ray_train.code_state()
    assert new["git_dirty"] is True and new["git_diff_sha256"] != empty
    (repo / "b.py").unlink()
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    edited = ray_train.code_state()
    assert edited["git_dirty"] is True and edited["git_diff_sha256"] not in (empty, new["git_diff_sha256"])


def test_a_stale_export_that_cannot_be_deleted_is_refused_not_kept(tmp_path, monkeypatch):
    """04-REVIEW IN-09: a failed deletion refuses the segment; write_export never keeps an existing export."""
    d = tmp_path / "exports"
    (d / "epoch-002").mkdir(parents=True)

    def fail(path, *a, **k):
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(ray_train.shutil, "rmtree", fail)
    with pytest.raises(SystemExit, match="cannot delete the stale export exports/epoch-002"):
        ray_train.prune_exports(str(d), 1)
    monkeypatch.undo()
    with pytest.raises(SystemExit, match="exports/epoch-002 already exists"):
        ray_train.write_export(None, None, {}, str(tmp_path), "epoch-002")


@pytest.mark.parametrize("k", ["0", "4", "5"])
def test_stop_after_epoch_bounds(k, capsys):
    """LAYA-08 boundary: K is 1..--epochs - 1; K = --epochs has one spelling (the omitted flag), refused by argparse
    before the split check, the fetch and ray.init."""
    with pytest.raises(SystemExit) as e:
        ray_train.main(["--data", "d.jsonl", "--out", "o", "--epochs", "4", "--stop-after-epoch", k])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--stop-after-epoch must be at least 1 and below --epochs (4)" in err, err


def test_stop_after_epoch_inside_the_schedule_resolves(tiny_run, capsys):
    ids = set()
    for k in ("1", "3"):
        ray_train.main(tiny_run + ["--epochs", "4", "--stop-after-epoch", k, "--print-run-id"])
        ids.add(capsys.readouterr().out.strip())
    ray_train.main(tiny_run + ["--epochs", "4", "--print-run-id"])
    ids.add(capsys.readouterr().out.strip())
    assert len(ids) == 3 and all(re.fullmatch(r"laya-selection_train-neg8-s0-[0-9a-f]{8}", i) for i in ids), ids


def test_run_id_follows_stop_after_epoch_and_postfit():
    """stop_after_epoch and postfit_sha256 name a final seed's run; without them a selection run keeps the id it had
    before they existed (laya-selection_train-neg8-s0-c3859768 for this command at 6ef1567)."""
    from pilot import runs
    base = ["--data", "d.jsonl", "--out", "o"]
    rid = _rid(*base)
    assert rid == "laya-selection_train-neg8-s0-c3859768"
    a = ray_train.parser().parse_args(base)
    vals = ray_train.identity_values(a, ray_train.resolve(a), ENTRY, "multilingual", BASE_IDENT)
    assert vals["stop_after_epoch"] is None and vals["postfit_sha256"] is None
    assert not {"stop_after_epoch", "postfit_sha256"} & set(runs.canonical(vals))
    stop3 = _rid(*(base + ["--stop-after-epoch", "3"]))
    assert stop3 != rid and _rid(*(base + ["--stop-after-epoch", "2"])) != stop3
    sha = ray_train.identity_values(a, ray_train.resolve(a), ENTRY, "multilingual", BASE_IDENT, "d" * 64)
    canon = runs.canonical(sha)
    assert canon["postfit_sha256"] == "d" * 64 and runs.run_id(canon) not in (rid, stop3)
    other = runs.canonical(dict(sha, postfit_sha256="e" * 64))
    assert runs.run_id(other) != runs.run_id(canon)
    assert _rid(*(base + ["--postfit", "results/postfit/laya.json"])) == rid      # the path itself never enters


@pytest.fixture
def final_cmd(tiny_run, tmp_path):
    """(argv of the final_train command bound to `record`, record path, record) for the tiny command: the chosen
    canonical is the selection canonical of tiny_run itself (epoch 4 of 4, so no --stop-after-epoch)."""
    import pilot.data
    from pilot import postfit, runs
    from pilot.weights import identity
    a = ray_train.parser().parse_args(tiny_run)
    base_dir = a.base
    entry = pilot.data.verify_split_file(a.data, manifest_path=a.manifest, roles=("train",))
    canon = runs.canonical(ray_train.identity_values(a, ray_train.resolve(a), entry, base_dir, identity(base_dir)))
    theta = {"global": 0.5, "per_tag": {}, "eligible_tags": []}
    record = {"schema": postfit.SCHEMA,
              "chosen": {"train_run_id": runs.run_id(canon), "dump_run_id": "laya-x", "encoder_lr": canon["encoder_lr"],
                         "epoch": canon["epochs"], "epochs_schedule": canon["epochs"], "stop_after_epoch": None,
                         "canonical": canon, "temperature": 1.5, "theta": {"shipped": theta, "raw": theta},
                         "val_micro_f1_at_global_theta": 0.5},
              "final": {"split": "final_train", "seeds": [0, 1, 2], "epochs": canon["epochs"], "stop_after_epoch": None},
              "protocol": {}, "rules": {}}
    path = str(tmp_path / "postfit" / "laya.json")
    postfit.write_record(record, path)
    argv = list(tiny_run)
    argv[argv.index("--data") + 1] = str(tmp_path / "data" / "splits" / "final_train.jsonl")
    return argv + ["--postfit", path], path, record


def test_postfit_refuses_a_mismatched_final_command_before_ray(final_cmd, tmp_path, capsys):
    """LAYA-08, T-05-11: a final seed is bound to the record by refusal. The matching command prints a final_train run
    id; another encoder LR, the selection split, a seed outside the final seeds, a stop epoch the record did not choose
    and a record of another schema are each refused naming the cause, and ray.init is never reached (tiny_run)."""
    from pilot import postfit
    argv, path, record = final_cmd
    ray_train.main(argv + ["--print-run-id"])
    assert re.fullmatch(r"laya-final_train-neg8-s0-[0-9a-f]{8}", capsys.readouterr().out.strip())
    ray_train.main(argv + ["--seed", "2", "--print-run-id"])                   # every final seed is accepted
    assert re.fullmatch(r"laya-final_train-neg8-s2-[0-9a-f]{8}", capsys.readouterr().out.strip())
    sel = list(argv)
    sel[sel.index("--data") + 1] = str(tmp_path / "data" / "splits" / "selection_train.jsonl")
    for extra, match in ((["--encoder-lr", "3e-05"], "encoder_lr is 3e-05 here but 2.5e-05 in the chosen selection run"),
                         (["--seed", "5"], "--seed 5 is not one of the record's final seeds \\[0, 1, 2\\]"),
                         (["--stop-after-epoch", "1"], "chosen candidate is epoch 4 of the --epochs 4 schedule, so omit "
                                                       "--stop-after-epoch"),
                         (["--neg", "4"], "neg is 4 here but 8"),
                         (["--max-messages", "7"], "max_messages is 7 here but None")):
        with pytest.raises(SystemExit, match=match):
            ray_train.main(argv + extra + ["--print-run-id"])
    with pytest.raises(SystemExit, match="trains the final seeds on final_train; --data is selection_train"):
        ray_train.main(sel + ["--print-run-id"])
    with pytest.raises(SystemExit, match="trains the final seeds on final_train"):
        ray_train.main(sel)                                                     # a training command refuses alike
    postfit.write_record(dict(record, schema="postfit/0"), path)
    with pytest.raises(SystemExit, match="schema 'postfit/0'"):
        ray_train.main(argv + ["--print-run-id"])
    bad_t = dict(record, chosen=dict(record["chosen"], temperature=7.0))
    postfit.write_record(bad_t, path)
    with pytest.raises(SystemExit, match="chosen temperature is 7.0"):
        ray_train.main(argv)
    assert not (tmp_path / "ray").exists() and not (tmp_path / "o").exists()


def test_check_final_names_the_first_differing_key():
    """The comparison is exact JSON equality over every key outside FINAL_FREE_KEYS, in sorted key order."""
    from pilot import postfit
    chosen = {"split": "selection_train", "seed": 0, "neg_seed": 0, "data_sha256": "a", "data_role": "train",
              "encoder_lr": 3e-05, "epochs": 4, "max_len": 128, "workers": 1}
    record = {"chosen": {"canonical": chosen, "epoch": 2, "epochs_schedule": 4, "stop_after_epoch": 2,
                         "train_run_id": "laya-sel"}, "final": {"split": "final_train", "seeds": [0, 1, 2]}}
    final = dict(chosen, split="final_train", seed=1, neg_seed=1, data_sha256="b", stop_after_epoch=2,
                 postfit_sha256="c" * 64)
    postfit.check_final(record, final)
    for change, match in (({"max_len": 128.0}, "max_len is 128.0 here but 128"),
                          ({"epochs": 2, "workers": 2}, "epochs is 2 here"),          # sorted: epochs before workers
                          ({"amp": False}, "amp is False here but <absent>"),        # a key the chosen run lacks
                          ({"stop_after_epoch": None}, "pass --stop-after-epoch 2")):
        with pytest.raises(SystemExit, match=match):
            postfit.check_final(record, dict(final, **change))
    with pytest.raises(SystemExit, match="workers is <absent> here but 1"):
        postfit.check_final(record, {k: v for k, v in final.items() if k != "workers"})


def test_check_final_binds_neg_seed_and_the_committed_final_data():
    """05-REVIEW WR-01: neg_seed must equal the seed when the chosen run's did, and a record built on the committed
    manifest binds data_sha256 to the committed final_train sha256 (read from data/splits.json at check time)."""
    import json
    from pilot import postfit
    from pilot.data import DEFAULT_MANIFEST
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        committed = json.load(f)["splits"]["final_train"]["sha256"]
    chosen = {"split": "selection_train", "seed": 0, "neg_seed": 0, "data_sha256": "a", "data_role": "train",
              "encoder_lr": 3e-05, "epochs": 4}
    record = {"chosen": {"canonical": chosen, "epoch": 4, "epochs_schedule": 4, "stop_after_epoch": None,
                         "train_run_id": "laya-sel"}, "final": {"split": "final_train", "seeds": [0, 1, 2]}}
    final = dict(chosen, split="final_train", seed=1, neg_seed=1, data_sha256="b", postfit_sha256="c" * 64)
    postfit.check_final(record, final)                       # a record of another manifest: data_sha256 stays free
    with pytest.raises(SystemExit, match="neg_seed is 0 but the seed is 1"):
        postfit.check_final(record, dict(final, neg_seed=0))
    on_committed = dict(record, manifest={"is_committed": True})
    with pytest.raises(SystemExit, match="not the committed final_train sha256 %s" % committed):
        postfit.check_final(on_committed, final)
    postfit.check_final(on_committed, dict(final, data_sha256=committed))


# ---- Phase 9 few-shot modes (FEW-01, FEW-02, D-01; 09-02): every refusal before ray.init ----

@pytest.mark.parametrize("flags, msg", [
    (["--fewshot-draw", "0"], "--fewshot-draw and --fewshot-k go together"),
    (["--fewshot-k", "2"], "--fewshot-draw and --fewshot-k go together"),
    (["--fewshot-base", "--fewshot-draw", "0", "--fewshot-k", "2"], "--fewshot-base trains a masked base"),
    (["--fewshot-base", "--postfit", "results/postfit/laya.json"], "--fewshot-base takes no --postfit"),
    (["--fewshot-draw", "0", "--fewshot-k", "2", "--postfit", "x.json"], "--fewshot-draw takes no --postfit"),
    (["--fewshot-base", "--max-messages", "5"], "--fewshot-base takes no --max-messages"),
    (["--fewshot-draw", "0", "--fewshot-k", "2", "--epochs", "4", "--stop-after-epoch", "1"],
     "--fewshot-draw takes no --stop-after-epoch or --mask-tags"),
    (["--fewshot-draw", "0", "--fewshot-k", "2", "--mask-tags", "card"],
     "--fewshot-draw takes no --stop-after-epoch or --mask-tags"),
    (["--mask-tags", "card"], "--mask-tags masks a Phase 9 base only"),
])
def test_fewshot_flag_combinations_are_argument_errors(flags, msg, capsys):
    """D-01: a few-shot command is one frozen-protocol mode; any other combination is an argparse error (exit 2),
    before the protocol, the split, the fetch or ray.init."""
    with pytest.raises(SystemExit) as e:
        ray_train.main(["--data", "d.jsonl", "--out", "o"] + flags)
    err = capsys.readouterr().err
    assert e.value.code == 2 and msg in err, err


@pytest.fixture
def fewshot_env(tmp_path, synthetic_raw, monkeypatch):
    """tests/test_fewshot_ray.py's synthetic frozen protocol installed in this process (tag card, k 1 and 2, draw 0,
    the tiny recipes, a postfit record whose chosen canonical is the tiny command's), with ray.init and the pinned
    fetch failing the test if reached. Returns a namespace: ns (synthetic_protocol's), base(*extra, data=None) and
    cont(base_dir, *extra) argv builders, and plant(canonical=..., ray_train=True) -> a tiny checkpoint whose config
    names a base run (training.ray_train run_id and canonical), as a continued run's --base."""
    import types
    from conftest import prepared
    from fewshot_synthetic import postfit_systems, synthetic_protocol
    from pilot import fewshot as fs
    from pilot import runs
    from pilot.weights import identity
    from test_fewshot_ray import COMMON, chosen_canonical, rules
    from tiny_laya import tiny_checkpoint

    def no_ray(*a, **kw):
        raise AssertionError("ray.init reached; a few-shot refusal must come first")

    def no_fetch(*a, **kw):
        raise AssertionError("the pinned fetch reached; a few-shot refusal must come first")
    monkeypatch.setattr(ray_train.ray, "init", no_ray)
    monkeypatch.setattr(ray_train, "fetch", no_fetch)
    tiny = tiny_checkpoint(tmp_path / "tiny")
    pre_dir, pre_manifest = prepared(tmp_path, synthetic_raw, name="pre")
    chosen = chosen_canonical(["--data", os.path.join(pre_dir, "splits", "selection_train.jsonl"),
                               "--manifest", pre_manifest, "--base", tiny, "--out", "o", "--epochs", "2"] + COMMON)
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, rules=rules(),
                            systems=postfit_systems(tmp_path, {"laya": chosen}))

    def split(name):
        return os.path.join(ns.out_dir, "splits", name + ".jsonl")

    def base(*extra, data=None):
        return ["--data", data or split("selection_train"), "--manifest", ns.manifest, "--base", tiny,
                "--out", str(tmp_path / "o"), "--storage", str(tmp_path / "ray"), "--epochs", "2",
                "--stop-after-epoch", "1", "--fewshot-base"] + COMMON + list(extra)

    def cont(base_dir, *extra):
        return ["--data", split("selection_train"), "--manifest", ns.manifest, "--base", str(base_dir),
                "--out", str(tmp_path / "o"), "--storage", str(tmp_path / "ray"), "--epochs", "1",
                "--fewshot-draw", "0", "--fewshot-k", "2"] + COMMON + list(extra)

    a = ray_train.parser().parse_args(base())
    entry = pilot_data().verify_split_file(a.data, manifest_path=a.manifest, roles=("train",))
    base_canon = runs.canonical(ray_train.identity_values(a, ray_train.resolve(a), entry, tiny, identity(tiny),
                                                          mask_tags=["card"], fewshot=fs.binding(ns.pin, "base")))

    def plant(canonical=None, ray_train_block=True, name="planted"):
        d = tiny_checkpoint(tmp_path / name)
        path = os.path.join(d, "rl_agent_config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        canon = base_canon if canonical is None else canonical
        if ray_train_block:
            cfg["training"] = {"ray_train": {"run_id": runs.run_id(canon), "canonical": canon}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return d
    return types.SimpleNamespace(ns=ns, base=base, cont=cont, plant=plant, base_canon=base_canon, tmp=tmp_path)


def pilot_data():
    import pilot.data
    return pilot.data


def test_fewshot_base_and_continued_commands_print_their_run_ids_without_ray(fewshot_env, capsys):
    """The positive path the refusals below depart from: the frozen base command and a continued command from a base
    of this protocol print a run id (ray.init and the fetch are never reached)."""
    ray_train.main(fewshot_env.base("--print-run-id"))
    rid = capsys.readouterr().out.strip()
    assert re.fullmatch(r"laya-selection_train-neg2-s0-[0-9a-f]{8}", rid), rid
    ray_train.main(fewshot_env.base("--mask-tags", "card", "--print-run-id"))      # the frozen tags, spelled out
    assert capsys.readouterr().out.strip() == rid
    ray_train.main(fewshot_env.cont(fewshot_env.plant(), "--print-run-id"))
    cont = capsys.readouterr().out.strip()
    assert re.fullmatch(r"laya-selection_train-neg2-s0-[0-9a-f]{8}", cont) and cont != rid
    assert not (fewshot_env.tmp / "ray").exists() and not (fewshot_env.tmp / "o").exists()


@pytest.mark.parametrize("case", ["missing", "unpinned", "changed"])
def test_fewshot_protocol_is_required_before_the_split_is_read(fewshot_env, monkeypatch, case):
    """D-01: require_frozen() runs first: a missing, unpinned or changed protocol is refused before verify_split_file
    reads --data (monkeypatched to fail the test if reached)."""
    from pilot import protocol as pr

    def no_split(*a, **kw):
        raise AssertionError("verify_split_file reached before the protocol check")
    monkeypatch.setattr(ray_train, "verify_split_file", no_split)
    if case == "missing":
        monkeypatch.setattr(pr, "PROTOCOL_PATH", str(fewshot_env.tmp / "absent.json"))
        match = "absent.json not found; the decision protocol must be frozen"
    elif case == "unpinned":
        monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)
        match = "the decision protocol is not frozen"
    else:
        with open(fewshot_env.ns.protocol_path, "ab") as f:
            f.write(b" ")
        match = "the protocol may not change after the freeze"
    for argv in (fewshot_env.base(), fewshot_env.cont(fewshot_env.plant())):
        with pytest.raises(SystemExit, match=match):
            ray_train.main(argv)


def test_fewshot_decision_record_is_required(fewshot_env):
    """D-08: the projection's decision record names the mode; without it no few-shot command trains."""
    os.remove(fewshot_env.ns.decision_path)
    with pytest.raises(SystemExit, match="run `python fewshot.py --projection"):
        ray_train.main(fewshot_env.base("--print-run-id"))


def test_mask_tags_other_than_the_protocols_are_refused(fewshot_env):
    """D-03: --mask-tags duplicates a frozen value, so any other list (another tag, an extra tag, a repeated tag) is
    refused naming the frozen tags."""
    for tags in ("loan", "card,loan", "card,card"):
        with pytest.raises(SystemExit, match="--mask-tags %s is not the frozen protocol's held-out tags card; pass "
                                             "them" % tags):
            ray_train.main(fewshot_env.base("--mask-tags", tags, "--print-run-id"))


def test_fewshot_base_on_another_split_is_refused(fewshot_env):
    """D-02: the few-shot source is the frozen selection_train; final_train (a train split) is refused by
    check_source before the base is read."""
    final = os.path.join(fewshot_env.ns.out_dir, "splits", "final_train.jsonl")
    with pytest.raises(SystemExit, match="the source entry has sha256 .* the frozen few-shot source"):
        ray_train.main(fewshot_env.base("--print-run-id", data=final))


def test_fewshot_base_with_another_epochs_is_refused(fewshot_env):
    """D-03: the base recipe is rules.base_runs.laya: another --epochs is refused naming the key."""
    with pytest.raises(SystemExit, match="epochs is 3 here but 2 in the frozen rules.base_runs.laya"):
        ray_train.main(fewshot_env.base("--epochs", "3", "--print-run-id"))
    with pytest.raises(SystemExit, match="seed 5 is not one of the base seeds \\[0\\]"):
        ray_train.main(fewshot_env.base("--seed", "5", "--print-run-id"))


def test_fewshot_continued_from_multilingual_is_refused(fewshot_env):
    """D-06: a continued run starts from a masked base checkpoint, never the pinned weights (refused before the
    fetch, which fails the test if reached)."""
    argv = fewshot_env.cont("multilingual")
    with pytest.raises(SystemExit, match="pass a ray_train.py --fewshot-base checkpoint as --base"):
        ray_train.main(argv)


def test_fewshot_continued_from_a_checkpoint_without_a_base_binding_is_refused(fewshot_env):
    """T-09-06: the base names itself by its own config bytes: a checkpoint without a ray_train.py run (the tiny base
    itself), one whose run id is not its canonical config's, and a run that is not a masked base of this protocol are
    each refused."""
    with pytest.raises(SystemExit, match="holds no ray_train.py run id and canonical config; a continued run starts "
                                         "from a --fewshot-base checkpoint"):
        ray_train.main(fewshot_env.cont(fewshot_env.plant(ray_train_block=False, name="bare")))
    unbound = {k: v for k, v in fewshot_env.base_canon.items() if k not in ("fewshot", "mask_tags")}
    with pytest.raises(SystemExit, match="carries fewshot binding <absent>, not a masked base of this protocol"):
        ray_train.main(fewshot_env.cont(fewshot_env.plant(unbound, name="unbound"), "--print-run-id"))
    d = fewshot_env.plant(name="edited")
    path = os.path.join(d, "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["training"]["ray_train"]["run_id"] = "laya-selection_train-neg2-s0-00000000"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with pytest.raises(SystemExit, match="names run laya-selection_train-neg2-s0-00000000, but its canonical config"):
        ray_train.main(fewshot_env.cont(d, "--print-run-id"))


def test_fewshot_continued_seed_must_be_the_draw_seed(fewshot_env):
    """D-07: the training seed of a continued run is the draw seed."""
    with pytest.raises(SystemExit, match="seed is 1 here but draw 0's seed is 0; the training seed is the draw seed"):
        ray_train.main(fewshot_env.cont(fewshot_env.plant(), "--seed", "1", "--print-run-id"))


def test_fewshot_continued_k_outside_the_frozen_values_is_refused(fewshot_env):
    """FEW-02: k is one of the frozen k_values (1, 2 here) and the draw one of the frozen draws."""
    base = fewshot_env.plant()
    with pytest.raises(SystemExit, match="few-shot k 3 is not one of the frozen k values \\[1, 2\\]"):
        ray_train.main(fewshot_env.cont(base, "--fewshot-k", "3", "--print-run-id"))
    with pytest.raises(SystemExit, match="few-shot draw 1 is not one of the frozen draws 0..0"):
        ray_train.main(fewshot_env.cont(base, "--fewshot-draw", "1", "--print-run-id"))
    with pytest.raises(SystemExit, match="epochs is 2 here but 1 in the frozen continued_training"):
        ray_train.main(fewshot_env.cont(base, "--epochs", "2", "--print-run-id"))


def _proof_cfg(stage):
    from pilot import fewshot as fs
    if stage == "base":
        return {"fewshot": fs.binding("p" * 64, "base"), "mask_tags": ["card"]}
    return {"fewshot": fs.binding("p" * 64, "continued", 0, 2, "d" * 64, "laya-base"), "mask_tags": None,
            "fewshot_rows": {"support": ["bank_1_0", "bank_2_0"], "replay": ["bank_3_0"], "tags": ["card"]},
            "fewshot_base": {"run_id": "laya-base", "canonical": {"seed": 0}}}


def test_fewshot_proof_refuses_a_held_out_item_in_a_masked_base():
    """FEW-01, T-09-05: the worker's masking proof refuses a masked base with any item of a held-out question and
    records zero otherwise; it carries counts and tags, never a uid."""
    per_q = {"card": 0, "loan": 5}
    proof = ray_train.fewshot_proof(_proof_cfg("base"), per_q, 5, 5)
    assert proof == {"protocol_sha256": "p" * 64, "stage": "base", "tags": ["card"], "held_out_items": {"card": 0},
                     "items_per_question": per_q}
    with pytest.raises(SystemExit, match="the masked base built items of held-out question\\(s\\) \\(card 1\\); "
                                         "refusing to train"):
        ray_train.fewshot_proof(_proof_cfg("base"), dict(per_q, card=1), 6, 6)
    assert ray_train.fewshot_proof({"fewshot": None}, per_q, 5, 5) is None


def test_fewshot_proof_of_a_continued_run_needs_one_new_tag_item_per_support_row():
    """FEW-02: exactly one held-out item per support row (a replay row contributes none) and the frozen item count;
    the proof names the base by run id and canonical config and holds no uid."""
    per_q = {"card": 2, "loan": 3}
    proof = ray_train.fewshot_proof(_proof_cfg("continued"), per_q, 5, 5)
    assert proof["new_tag_items"] == {"card": 2} and proof["support_rows"] == 2 and proof["replay_rows"] == 1
    assert proof["base"] == {"run_id": "laya-base", "canonical": {"seed": 0}} and "bank_" not in json.dumps(proof)
    with pytest.raises(SystemExit, match="held-out question\\(s\\) \\(card 3\\) for 2 support rows"):
        ray_train.fewshot_proof(_proof_cfg("continued"), dict(per_q, card=3), 6, 6)
    with pytest.raises(SystemExit, match="built 6 items, the frozen Laya item rule gives 5"):
        ray_train.fewshot_proof(_proof_cfg("continued"), per_q, 6, 5)
