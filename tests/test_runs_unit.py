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
