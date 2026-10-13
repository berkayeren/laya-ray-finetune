"""MPS spike (mps_spike.py, pilot/spike.py) worker contracts, tested offline on a tiny ModernBERT + Laya head on CPU.

The tiny source runs the same worker path the pinned model runs (configure, move, device assertion, optimizer,
timed steps, payload), so these tests prove the refusals and the checkpointing-on / AMP-off step without weights.
"""
import os
import subprocess
import sys

import pytest

import mps_spike
from pilot import spike

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worker_cfg(**kw):
    c = {"source": "tiny", "device": "cpu", "transport": "return", "micro_batch": 2, "max_len": 32, "grad_accum": 2,
         "rung": "none", "fallback": "0", "purpose": "initial", "warmup_steps": 1, "measured_steps": 2, "seed": 0,
         "cpus": 1, "expect_env": {}, "hparams": mps_spike.HPARAMS}
    c.update(kw)
    return c


def test_worker_refuses_missing_env(monkeypatch):
    monkeypatch.delenv("SPIKE_CANARY", raising=False)
    payload = mps_spike.spike_worker(worker_cfg(expect_env={"SPIKE_CANARY": "1"}))
    assert payload["status"] == "refused", payload
    assert "SPIKE_CANARY" in payload["reason"], payload


def test_require_param_device_refuses_cpu_params():
    with pytest.raises(SystemExit, match="expected mps"):
        mps_spike.require_param_device(mps_spike.tiny_model(), "mps")


def test_configure_model_enables_checkpointing_and_freezes_only_embeddings():
    model = mps_spike.tiny_model()
    full = mps_spike.configure_model(model, "none")
    assert all(layer.gradient_checkpointing for layer in model.encoder.layers)
    assert model.head_checkpointing is True
    frozen_model = mps_spike.tiny_model()
    frozen = mps_spike.configure_model(frozen_model, "embeddings_frozen")
    assert full - frozen == 1000 * 64
    assert [n for n, p in frozen_model.named_parameters() if not p.requires_grad] == \
        ["encoder.embeddings.tok_embeddings.weight"]
    with pytest.raises(SystemExit, match="never FITS"):
        mps_spike.configure_model(mps_spike.tiny_model(), "freeze_encoder")


def test_worker_step_keeps_checkpointing_on_and_amp_off(monkeypatch):
    from laya import train as lt
    calls, real = [], lt._forward

    def recording(model, batch, device, amp, detach_encoder):
        calls.append((amp, detach_encoder, model.head_checkpointing, model.training))
        return real(model, batch, device, amp, detach_encoder)

    monkeypatch.setattr(lt, "_forward", recording)
    payload = mps_spike.spike_worker(worker_cfg())
    assert payload["status"] == "ok", payload
    # (1 warm-up + 2 measured) x 2 regimes x grad_accum 2
    assert len(calls) == 12 and set(calls) == {(False, False, True, True)}, set(calls)
    assert payload["param_device"] == "cpu" and payload["transport"] == "return"
    assert payload["regimes"]["padded"]["times"]["n"] == 2
    assert payload["regimes"]["padded"]["batch_shape"] == [2, 32]
    assert payload["regimes"]["real"]["times"]["n"] == 2


def test_orchestrator_import_is_torch_free():
    code = "import sys, mps_spike, pilot.spike; print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "NO_TORCH", r.stdout


def test_spike_env_sets_both_watermarks():
    env = spike.spike_env("0", 4)
    assert set(env) == {"PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                        "PYTORCH_MPS_LOW_WATERMARK_RATIO", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
                        "HF_HUB_OFFLINE"}
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "1.0"
    assert env["OMP_NUM_THREADS"] == "4" and env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    with pytest.raises(ValueError):
        spike.spike_env("2", 4)


def test_parse_grid_rejects_bad_entries():
    assert spike.parse_grid("4x128,8x256") == [(4, 128), (8, 256)]
    for bad in ("4x", "0x128", "4x128,4x128", ""):
        with pytest.raises(ValueError):
            spike.parse_grid(bad)


def test_child_outcome_maps_refused_and_crashed():
    assert spike.child_outcome(0, {"status": "ok"}) == {"status": "ok"}
    with pytest.raises(SystemExit, match="spike run refused: x"):
        spike.child_outcome(0, {"status": "refused", "reason": "x"})
    assert spike.child_outcome(1, None) == {"status": "crashed", "returncode": 1}
