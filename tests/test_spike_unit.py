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


# ---- verdict logic (plan 02-02): projection, OOM detection, memory gate, time gate, rung rules, config choice ----

FULL = {"grid": [list(g) for g in spike.DEFAULT_GRID], "warmup_steps": 5, "measured_steps": 20, "effective_batch": 32}
OOM_TEXT = ("MPS backend out of memory (MPS allocated: 9.10 GB, other allocations: 1.20 GB, max allowed: 11.84 GB). "
            "Tried to allocate 256.00 MB on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper "
            "limit for memory allocations (may cause system failure).")


def _times(n, median):
    return {"n": n, "median_s": median, "p90_s": median, "min_s": median, "max_s": median} if n else {"n": 0}


def attempt(mb, max_len, rung="none", status="ok", swap=0, real_median=10.0, n_items=14854, device="mps", padded_n=20,
            fallback="0"):
    """Minimal attempt dict shaped like mps_spike's worker payload."""
    return {"status": status, "purpose": "initial", "micro_batch": mb, "max_len": max_len, "grad_accum": 32 // mb,
            "rung": rung, "fallback": fallback, "device": device, "param_device": device, "n_neg8_items": n_items,
            "regimes": {"padded": {"times": _times(padded_n, 1.0), "swap_delta_bytes": swap, "swapouts_delta": 0},
                        "real": {"times": _times(20, real_median), "swap_delta_bytes": 0, "swapouts_delta": 0}},
            "peak_memory_bytes": 8 * 2**30}


def config(a, *more, standing=0, grad_accum=None):
    return {"micro_batch": a["micro_batch"], "max_len": a["max_len"],
            "grad_accum": grad_accum or 32 // a["micro_batch"],
            "attempts": [a] + list(more), "standing": standing}


def at_epoch(mb, max_len, epoch_s, **kw):
    """Attempt whose real median projects to epoch_s over 465 optimizer steps (14,854 items, effective batch 32)."""
    return attempt(mb, max_len, real_median=epoch_s / 465, **kw)


def chosen(v):
    return None if v["chosen"] is None else (v["chosen"]["micro_batch"], v["chosen"]["max_len"], v["chosen"]["rung"])


def test_projection_465_steps_for_both_micro_batches():
    from laya import train as lt
    assert spike.optimizer_steps(14854, 8, 4) == spike.optimizer_steps(14854, 4, 8) == 465
    assert spike.optimizer_steps(10, 4, 2) == 2
    for n, mb, accum in ((14854, 8, 4), (14854, 4, 8), (14854, 4, 2), (10, 4, 2), (1, 8, 4), (33, 4, 8)):
        assert spike.optimizer_steps(n, mb, accum) == lt.optimizer_updates(n, mb, accum, 1), (n, mb, accum)
    with pytest.raises(ValueError):
        spike.optimizer_steps(0, 4, 2)


def test_projection_time_gate_boundary():
    steps = spike.optimizer_steps(64, 4, 2)
    assert steps == 8 and spike.project_epoch_s(900.0, steps) == 7200.0
    ok = spike.verdict([config(attempt(4, 128, real_median=900.0, n_items=64), grad_accum=2)],
                       FULL)["per_config"][0]
    assert ok["steps_per_epoch"] == 8 and ok["projected_epoch_s"] == 7200.0 and ok["time_ok"] is True and ok["fits"] is True
    late = spike.verdict([config(attempt(4, 128, real_median=900.01, n_items=64), grad_accum=2)],
                         FULL)["per_config"][0]
    assert late["time_ok"] is False and late["fits"] is False


def test_oom_detection():
    assert spike.is_mps_oom(OOM_TEXT) is True
    assert spike.is_mps_oom("  " + OOM_TEXT) is True
    assert spike.is_mps_oom("CUDA out of memory. Tried to allocate 2.00 GiB") is False
    assert spike.is_mps_oom("") is False


def test_memory_gate_table():
    oom = {"status": "oom", "oom": {"regime": "padded", "stage": "measured", "step": 3, "message": OOM_TEXT}}
    ok, reason = spike.memory_ok(oom, 20)
    assert ok is False and "oom" in reason and "padded" in reason
    ok, reason = spike.memory_ok({"status": "crashed", "returncode": -9}, 20)
    assert ok is False and "crashed" in reason
    ok, reason = spike.memory_ok({"status": "error", "error": "ValueError: boom"}, 20)
    assert ok is False and "error" in reason
    ok, reason = spike.memory_ok(attempt(4, 128, padded_n=19), 20)
    assert ok is False and "19 of 20" in reason
    ok, reason = spike.memory_ok(attempt(4, 128, swap=1), 20)
    assert ok is False and "swap grew" in reason
    for delta in (0, -4096):
        ok, reason = spike.memory_ok(attempt(4, 128, swap=delta), 20)
        assert ok is True, reason


def test_protocol_complete_accepts_lists_and_tuples():
    assert spike.protocol_complete(FULL) is True
    assert spike.protocol_complete(dict(FULL, grid=list(spike.DEFAULT_GRID))) is True
    assert spike.protocol_complete(dict(FULL, grid=[[4, 128]])) is False
    assert spike.protocol_complete(dict(FULL, measured_steps=2)) is False
    assert spike.protocol_complete(dict(FULL, effective_batch=8)) is False


def test_verdict_does_not_fit_when_every_config_fails():
    v = spike.verdict([config(attempt(mb, ml, swap=1)) for mb, ml in spike.DEFAULT_GRID], FULL)
    assert v["result"] == "DOES NOT FIT" and v["chosen"] is None
    assert v["protocol_complete"] is True and v["kaggle_plan_required"] is True
    assert all(not p["memory_ok"] and "swap grew" in p["memory_reason"] for p in v["per_config"])


def test_verdict_fits_names_the_phase4_config():
    v = spike.verdict([config(at_epoch(4, 128, 3000.0))], FULL)
    assert v["result"] == "FITS" and v["kaggle_plan_required"] is False
    c = v["chosen"]
    assert (c["micro_batch"], c["max_len"], c["grad_accum"], c["rung"], c["fallback"]) == (4, 128, 8, "none", "0")
    assert c["projected_epoch_s"] == 3000.0 and c["peak_memory_bytes"] == 8 * 2**30


def test_verdict_cpu_param_device_never_fits():
    v = spike.verdict([config(attempt(4, 128, device="cpu"))], FULL)
    p = v["per_config"][0]
    assert p["device_ok"] is False and p["fits"] is False and v["result"] == "DOES NOT FIT"


def test_verdict_freeze_encoder_rung_never_fits():
    v = spike.verdict([config(attempt(4, 128, rung="freeze_encoder"))], FULL)
    p = v["per_config"][0]
    assert p["rung_ok"] is False and p["fits"] is False and "D-04" in p["reason"]
    assert v["result"] == "DOES NOT FIT"


def test_verdict_prefers_rung_none_over_faster_embeddings_frozen():
    v = spike.verdict([config(at_epoch(4, 128, 50.0, rung="embeddings_frozen")), config(at_epoch(8, 128, 100.0))],
                      FULL)
    assert v["result"] == "FITS" and chosen(v) == (8, 128, "none")
    only_frozen = spike.verdict([config(at_epoch(4, 128, 50.0, rung="embeddings_frozen"))], FULL)
    assert chosen(only_frozen) == (4, 128, "embeddings_frozen")


def test_verdict_tie_prefers_max_len_128_then_larger_micro_batch():
    v = spike.verdict([config(at_epoch(4, 128, 100.0)), config(at_epoch(8, 128, 103.0)),
                       config(at_epoch(8, 256, 101.0))], FULL)
    assert chosen(v) == (8, 128, "none") and v["tie_tolerance"] == spike.TIE_TOLERANCE


def test_verdict_outside_tolerance_fastest_wins():
    v = spike.verdict([config(at_epoch(4, 128, 100.0)), config(at_epoch(8, 128, 110.0))], FULL)
    assert chosen(v) == (4, 128, "none")
    wide = spike.verdict([config(at_epoch(4, 128, 100.0)), config(at_epoch(8, 128, 110.0))], FULL, tie_tolerance=0.2)
    assert chosen(wide) == (8, 128, "none")


def test_verdict_truncation_prefers_256():
    cfgs = [config(at_epoch(4, 128, 100.0)), config(at_epoch(4, 256, 300.0))]
    v = spike.verdict(cfgs, FULL, token_stats={"pct_messages_over_128": 2.0})
    assert chosen(v) == (4, 256, "none") and v["truncation_rule"].startswith("applied: 2.0% of messages over 128")
    low = spike.verdict(cfgs, FULL, token_stats={"pct_messages_over_128": 0.0})
    assert chosen(low) == (4, 128, "none") and low["truncation_rule"].startswith("applied: 0.0%")
    none = spike.verdict(cfgs, FULL, token_stats=None)
    assert chosen(none) == (4, 128, "none") and none["truncation_rule"].startswith("not applied")


def test_verdict_time_gate_fails_memory_ok_config():
    v = spike.verdict([config(at_epoch(4, 128, 9300.0))], FULL)
    p = v["per_config"][0]
    assert p["memory_ok"] is True and p["time_ok"] is False and p["projected_epoch_s"] == 9300.0
    assert p["fits"] is False and "D-02" in p["reason"] and v["result"] == "DOES NOT FIT"


def test_verdict_blocking_op_prevents_fits():
    cfgs = [config(at_epoch(4, 128, 100.0))]
    v = spike.verdict(cfgs, FULL, blocking={(4, 128, "none"): ["aten::x"]})
    p = v["per_config"][0]
    assert p["blocking_ops"] == ["aten::x"] and p["fits"] is False and "aten::x" in p["reason"]
    assert v["result"] == "DOES NOT FIT"
    assert spike.verdict(cfgs, FULL, blocking={(8, 128, "none"): ["aten::x"]})["result"] == "FITS"


def test_verdict_smoke_protocol_never_requires_kaggle():
    smoke = dict(FULL, warmup_steps=1)
    v = spike.verdict([config(attempt(4, 128, swap=1))], smoke)
    assert v["protocol_complete"] is False and v["result"] == "DOES NOT FIT"
    assert v["kaggle_plan_required"] is False


def test_verdict_counts_untrusted_attempts():
    err = {"status": "error", "error": "ValueError: boom", "micro_batch": 4, "max_len": 128, "rung": "none"}
    cfgs = [config(err, at_epoch(4, 128, 100.0), standing=1),
            config(dict(attempt(8, 128), status="crashed", returncode=-9)), config(at_epoch(8, 256, 200.0))]
    v = spike.verdict(cfgs, FULL)
    assert v["untrusted_attempts"] == 2
    assert v["per_config"][0]["fits"] is True and v["per_config"][1]["fits"] is False
    assert set(v["rules"]) >= {"D-01", "D-02", "D-03", "D-04", "D-08"}


def test_neg8_rows_keeps_positives_and_samples_eight_negatives():
    tags = ["t%02d" % i for i in range(20)]
    rows = [{"uid": "a", "expected": {t: t in ("t01", "t07") for t in tags}},
            {"uid": "b", "expected": {"p": True, "n1": False, "n2": False, "n3": False}}]
    before = [dict(r, expected=dict(r["expected"])) for r in rows]
    out = spike.neg8_rows(rows, 0)
    assert rows == before
    a, b = out
    assert {t for t, v in a["expected"].items() if v} == {"t01", "t07"}
    assert sum(1 for v in a["expected"].values() if not v) == 8
    assert b["expected"] == {"p": True, "n1": False, "n2": False, "n3": False}
    assert spike.neg8_rows(rows, 0) == out and a["uid"] == "a"


def test_pad_item_to_exact_length():
    item = {"state_ids": [7, 8, 9], "qid": 1}
    padded = spike.pad_item_to(item, 10)
    assert len(padded["state_ids"]) == 10 and padded["qid"] == 1
    assert item == {"state_ids": [7, 8, 9], "qid": 1}
    assert spike.pad_item_to({"state_ids": []}, 6)["state_ids"] == [5] * 6


def test_summarize_median_p90():
    s = spike.summarize([float(i) for i in range(10, 0, -1)])
    assert s["n"] == 10 and s["median_s"] == 5.5 and s["p90_s"] == 9.0 and s["min_s"] == 1.0 and s["max_s"] == 10.0
    assert spike.summarize([]) == {"n": 0}
