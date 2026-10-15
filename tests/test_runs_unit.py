"""pilot/runs.py: the torch-free run-record schema of ray_train.py (LAYA-06)."""
import json
import os
import subprocess
import sys

import pytest

from pilot import runs

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row(epoch, **kw):
    row = {k: 0 for k in runs.EPOCH_FIELDS}
    row.update(epoch=epoch, **kw)
    return row


def _ckpt(tmp_path, training):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "rl_agent_config.json").write_text(json.dumps({"encoder": "x", "training": training}), encoding="utf-8")
    return str(d)


def test_epoch_fields_are_the_thirteen_record_columns():
    assert runs.EPOCH_FIELDS == ("epoch", "loss", "train_s", "save_s", "optimizer_steps", "n_items",
                                 "peak_total_bytes", "peak_driver_bytes", "peak_rss_bytes", "current_allocated_bytes",
                                 "swap_delta_bytes", "segment", "resumed")


def test_epoch_fields_cover_segments():
    assert "segment" in runs.EPOCH_FIELDS and "resumed" in runs.EPOCH_FIELDS
    assert runs.SEGMENT_FIELDS == ("segment", "start_epoch", "resumed", "git_sha", "git_dirty", "git_diff_sha256",
                                   "device", "resume_load_s", "resume_peak_total_bytes")


def test_env_strings_are_sorted_key_value_pairs():
    env = {"PYTORCH_MPS_LOW_WATERMARK_RATIO": "1.0", "OMP_NUM_THREADS": "4", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    assert runs.env_strings(env) == ["OMP_NUM_THREADS=4", "PYTORCH_ENABLE_MPS_FALLBACK=0",
                                     "PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0"]
    assert runs.env_strings({}) == []


def test_read_record_returns_the_ray_train_record(tmp_path):
    record = {"device": "cpu", "epochs": [_row(1), _row(2, loss=0.5)]}
    d = _ckpt(tmp_path, {"ray_train": {"seed": 0, "record": record}, "epoch": 2})
    assert runs.read_record(d) == record


def test_read_record_refuses_a_checkpoint_without_a_record(tmp_path):
    d = _ckpt(tmp_path, {"ray_train": {"seed": 0}})
    with pytest.raises(SystemExit, match="holds no ray_train run record"):
        runs.read_record(d)


def test_read_record_refuses_an_epoch_row_missing_a_field(tmp_path):
    row = _row(2)
    del row["swap_delta_bytes"]
    d = _ckpt(tmp_path, {"ray_train": {"record": {"device": "cpu", "epochs": [_row(1), row]}}})
    with pytest.raises(SystemExit, match="run record epoch 2 lacks swap_delta_bytes"):
        runs.read_record(d)


def test_import_is_torch_free():
    code = "import sys, pilot.runs; print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "NO_TORCH", r.stdout


def _labelled(n_pos, n_labels=48):
    return {"expected": {"t%02d" % i: i < n_pos for i in range(n_labels)}}


def test_neg_items_counts_positives_plus_k_negatives():
    assert runs.neg_items([_labelled(2)], 8) == 10
    assert runs.neg_items([_labelled(0)], 8) == 8
    assert runs.neg_items([_labelled(45)], 8) == 48            # only 3 negatives left to keep
    assert runs.neg_items([_labelled(2)], 0) == 48             # K 0 = every label (data_full)
    assert runs.neg_items([_labelled(2), _labelled(0), _labelled(45)], 8) == 66
    with pytest.raises(ValueError, match="positive count"):
        runs.neg_items([_labelled(2)], -1)


def test_neg_items_equals_what_neg8_rows_keeps():
    from pilot import spike
    rows = [_labelled(p) for p in (0, 1, 3, 12, 41, 47)]
    for k in (1, 8, 20):
        kept = spike.neg8_rows(rows, 0, k)
        assert runs.neg_items(rows, k) == sum(len(r["expected"]) for r in kept)


@pytest.mark.data
def test_neg8_items_on_the_committed_train_splits():
    """The compute projection's torch-free count equals the spike's n_neg8_items (14,854) on selection_train."""
    import pilot.data
    for name, want in (("selection_train", 14854), ("final_train", 17001)):
        path = os.path.join(REPO_ROOT, "data", "splits", name + ".jsonl")
        entry = pilot.data.verify_split_file(path, roles=("train",))
        with open(path, "rb") as f:
            rows = pilot.data.parse_jsonl(f.read())
        assert len(rows) == entry["n_messages"]
        assert runs.neg_items(rows, 8) == want, name


def _cfg(freeze_encoder=False, freeze_embeddings=False, **kw):
    return dict(runs.SPIKE_START, freeze_encoder=freeze_encoder, freeze_embeddings=freeze_embeddings, **kw)


def _apply(flags, cfg):
    """The config a ray_train.py command gets when `flags` are added to it (the four flags the ladder uses)."""
    cfg, it = dict(cfg), iter(flags)
    for f in it:
        if f in ("--freeze-embeddings", "--freeze-encoder"):
            cfg[f[2:].replace("-", "_")] = True
        else:
            cfg[f[2:].replace("-", "_")] = int(next(it))
    return cfg


def test_rung_label_table():
    for (enc, emb, mb), want in (((False, False, 8), "none"), ((False, True, 8), "embeddings_frozen"),
                                 ((False, False, 4), "smaller_batch"),
                                 ((False, True, 4), "embeddings_frozen+smaller_batch"),
                                 ((True, False, 8), "encoder_frozen")):
        assert runs.rung_label(_cfg(enc, emb, micro_batch=mb)) == want, (enc, emb, mb)
    assert [r for r in runs.RUNG_NEXT if runs.weaker(r)] == ["encoder_frozen"]


def test_ladder_order():
    chain, r = [], "none"
    while r is not None:
        chain.append(r)
        r = runs.next_rung(r)
    assert chain == ["none", "embeddings_frozen", "embeddings_frozen+smaller_batch", "encoder_frozen"]
    assert runs.next_rung("smaller_batch") == "embeddings_frozen+smaller_batch"
    assert runs.rung_flags("none") == []
    assert runs.rung_flags("embeddings_frozen") == ["--freeze-embeddings"]
    assert runs.rung_flags("embeddings_frozen+smaller_batch") == ["--freeze-embeddings", "--micro-batch", "4",
                                                                  "--grad-accum", "8"]
    assert runs.rung_flags("encoder_frozen") == ["--freeze-encoder"]
    for rung in runs.RUNG_NEXT:                                 # the label is derived from what the flags set
        assert runs.rung_label(_apply(runs.rung_flags(rung), _cfg())) == rung


def test_spike_start_matches_the_committed_spike():
    with open(os.path.join(REPO_ROOT, "results", "spike", "mps_spike.json"), encoding="utf-8") as f:
        chosen = json.load(f)["verdict"]["chosen"]
    assert runs.SPIKE_START == {k: chosen[k] for k in ("micro_batch", "max_len", "grad_accum")}
    assert chosen["rung"] == "none"


def test_oom_message_names_the_next_rung():
    msg = runs.oom_message(2, 17, "none")
    assert msg.startswith(runs.OOM_MARK) and "\n" not in msg
    for part in ("epoch 2", "micro-batch 17", "rung none", "--freeze-embeddings"):
        assert part in msg, (part, msg)
    assert "--freeze-encoder" in runs.oom_message(1, 1, "embeddings_frozen+smaller_batch")
    assert "exhausted" in runs.oom_message(1, 3, "encoder_frozen")
    text = "Training failed due to worker errors:\n[Rank 0]\nTraceback (most recent call last):\n  File \"x.py\"\n" \
           "RuntimeError: MPS backend out of memory (MPS allocated: 9 GiB)\n\nThe above exception was the direct " \
           "cause of the following exception:\n\nRuntimeError: %s\n" % msg
    assert runs.failure_line(text) == msg
    assert runs.failure_line("Training failed due to worker errors:\nValueError: bad batch\n") is None


@pytest.mark.parametrize("stage", sorted(runs.RESUME_STAGES))
def test_an_oom_after_the_last_epoch_says_resume_not_the_next_rung(stage):
    """04-REVIEW WR-01: every epoch is saved there; a rung flag would change the run_id and retrain from epoch 0."""
    for label in ("none", "embeddings_frozen", "encoder_frozen"):
        msg = runs.oom_message(None, None, label, stage)
        assert msg.startswith(runs.OOM_MARK + " " + stage) and runs.failure_line(msg) == msg
        assert "rerun the same command to resume" in msg and runs.resume_oom(msg)
        for flag in ("--freeze-embeddings", "--freeze-encoder", "--micro-batch", "exhausted"):
            assert flag not in msg, (flag, msg)
    for line in (runs.oom_message(2, 17, "none"), runs.oom_message(None, None, "none", "outside the training step"),
                 runs.oom_message(1, None, "none", "while saving the checkpoint"), None, ""):
        assert not runs.resume_oom(line), line


def test_spike_departures():
    assert runs.spike_departures(_cfg(), 128) == {}
    assert runs.spike_departures(_cfg(), 256) == {"max_len": 256}
    smaller = runs.spike_departures(_cfg(micro_batch=4, grad_accum=8), 128)
    assert "effective_batch" not in smaller and smaller["grad_accum"] == 8           # 4 x 8 keeps the 32
    assert runs.spike_departures(_cfg(grad_accum=2), 128)["effective_batch"] == 16


def _identity(**kw):
    vals = {k: 0 for k in runs.IDENTITY}
    vals.update(split="selection_train", neg=8, seed=0, device="cpu", data_sha256="a" * 64)
    vals.update(kw)
    return vals


def test_canonical_refuses_missing_and_extra_keys():
    vals = _identity()
    assert runs.canonical(vals) == vals
    with pytest.raises(ValueError, match="missing seed"):
        runs.canonical({k: v for k, v in vals.items() if k != "seed"})
    with pytest.raises(ValueError, match="unexpected out"):
        runs.canonical(dict(vals, out="ckpt"))


def test_run_id_is_stable_and_follows_seed_and_neg():
    a = runs.run_id(runs.canonical(_identity()))
    assert a == runs.run_id(runs.canonical(_identity()))
    assert a.startswith("laya-selection_train-neg8-s0-") and len(a.rsplit("-", 1)[1]) == 8
    b = runs.run_id(runs.canonical(_identity(seed=1)))
    assert a != b and "-s0-" in a and "-s1-" in b
    assert "-full-" in runs.run_id(runs.canonical(_identity(neg=0)))
    assert runs.run_id(runs.canonical(_identity(encoder_lr=1e-5))) != a


def test_run_id_does_not_depend_on_key_order():
    vals = _identity()
    reordered = dict(reversed(list(vals.items())))
    assert list(reordered) != list(vals)
    assert runs.run_id(reordered) == runs.run_id(vals) == runs.run_id(runs.canonical(reordered))


def test_check_state_refuses_foreign_state():
    canon = runs.canonical(_identity())
    rid = runs.run_id(canon)
    state = {"kind": "train_state", "run_id": rid, "canonical": dict(canon), "epoch": 1}
    runs.check_state(state, rid, canon)                                     # the matching state passes
    for key, bad in (("kind", "final"), ("run_id", "laya-other"), ("canonical", dict(canon, seed=1))):
        with pytest.raises(SystemExit, match="another run \\(%s differs\\)" % key):
            runs.check_state(dict(state, **{key: bad}), rid, canon)
    with pytest.raises(SystemExit, match="kind differs"):
        runs.check_state({}, rid, canon)


def test_disk_need_bytes():
    m = 614 * 2**20
    assert runs.disk_need_bytes(m, 2, True) == 18 * m + 2 * m + m + 2**30
    assert runs.disk_need_bytes(m, 2, False) == 18 * m + m + 2**30
    assert runs.disk_need_bytes(m, 4, True) - runs.disk_need_bytes(m, 4, False) == 4 * m


def test_fail_after_epoch_and_export_epochs_never_change_the_run_id():
    assert {"fail_after_epoch", "export_epochs"} <= set(runs.NOT_IDENTITY)


def test_identity_holds_no_calibration_keys():
    """LAYA-05: training no longer calibrates, so no calibration value is part of the run identity, and only the two
    post-training stages (the final save and a finished rerun's model move) say rerun the same command."""
    keys = set(runs.IDENTITY) | set(runs.NOT_IDENTITY)
    assert not [k for k in keys if k.startswith("calib")], sorted(keys)
    assert set(runs.RESUME_STAGES) == {runs.STAGE_FINAL_SAVE, runs.STAGE_FINALIZE_MOVE}
    assert not hasattr(runs, "STAGE_CALIBRATE") and not hasattr(runs, "STAGE_RECALIBRATE_MOVE")


def test_last_line_and_free_bytes(tmp_path):
    """04-REVIEW IN-02: the one shared copy of the failure-text and free-disk helpers."""
    assert runs.last_line("Traceback\n  File x\nValueError: bad\n\n") == "ValueError: bad"
    assert runs.last_line("") == "" and runs.last_line("  \n", "none") == "none"
    assert runs.free_bytes(str(tmp_path / "not" / "yet" / "made")) > 0         # the nearest existing parent's volume
    assert runs.free_bytes(os.path.abspath(os.sep)) > 0


def test_canonical_keeps_the_final_seed_keys_only_when_set():
    """stop_after_epoch and postfit_sha256 enter a final seed's run id and stay out of every selection run's."""
    sel = _identity(stop_after_epoch=None, postfit_sha256=None, base_files_sha256=None)
    canon = runs.canonical(sel)
    assert not {"stop_after_epoch", "postfit_sha256", "base_files_sha256"} & set(canon)
    final = runs.canonical(dict(sel, split="final_train", stop_after_epoch=2, postfit_sha256="f" * 64))
    assert final["stop_after_epoch"] == 2 and final["postfit_sha256"] == "f" * 64
    assert runs.run_id(final) != runs.run_id(runs.canonical(dict(sel, split="final_train")))
    with pytest.raises(ValueError, match="missing stop_after_epoch"):            # still required, as None
        runs.canonical({k: v for k, v in sel.items() if k != "stop_after_epoch"})


def test_identity_and_not_identity_stay_disjoint():
    assert not set(runs.IDENTITY) & set(runs.NOT_IDENTITY)
    assert {"stop_after_epoch", "postfit_sha256"} <= set(runs.IDENTITY) & set(runs.DROPPED_WHEN_NONE)
    assert "postfit" in runs.NOT_IDENTITY and set(runs.DROPPED_WHEN_NONE) <= set(runs.IDENTITY)
