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


def test_epoch_fields_are_the_eleven_record_columns():
    assert runs.EPOCH_FIELDS == ("epoch", "loss", "train_s", "save_s", "optimizer_steps", "n_items",
                                 "peak_total_bytes", "peak_driver_bytes", "peak_rss_bytes", "current_allocated_bytes",
                                 "swap_delta_bytes")


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


def test_spike_departures():
    assert runs.spike_departures(_cfg(), 128) == {}
    assert runs.spike_departures(_cfg(), 256) == {"max_len": 256}
    smaller = runs.spike_departures(_cfg(micro_batch=4, grad_accum=8), 128)
    assert "effective_batch" not in smaller and smaller["grad_accum"] == 8           # 4 x 8 keeps the 32
    assert runs.spike_departures(_cfg(grad_accum=2), 128)["effective_batch"] == 16
