"""MPS spike (mps_spike.py, pilot/spike.py) worker contracts, tested offline on a tiny ModernBERT + Laya head on CPU.

The tiny source runs the same worker path the pinned model runs (configure, move, device assertion, optimizer,
timed steps, payload), so these tests prove the refusals and the checkpointing-on / AMP-off step without weights.
"""
import os
import subprocess
import sys

import pytest

import mps_spike
from pilot import laya_adapter, spike

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


def test_mps_spike_checks_the_laya_pin_before_the_fetch(monkeypatch):
    """IN-05: like every other adapter consumer, the spike refuses a wrong laya before the pinned download. Behavioural
    (04-REVIEW IN-05): main runs with a failing version check and a fetch that must not be reached."""
    calls = []

    def bad_version():
        calls.append("check_version")
        raise SystemExit("laya 9.9.9 is installed; the adapter is pinned to 0.4.6")

    def fetch(pin):
        calls.append("fetch")
        pytest.fail("the pinned weights were fetched before the laya version check")
    monkeypatch.setattr(sys, "argv", ["mps_spike.py", "--min-available-gib", "0", "--out", "unused"])
    monkeypatch.setattr(mps_spike.time, "sleep", lambda s: None)              # the 5 s idle swap sample
    monkeypatch.setattr(mps_spike, "verify_split_file", lambda *a, **k: {"sha256": "0" * 64})
    monkeypatch.setattr(mps_spike.la, "check_version", bad_version)
    monkeypatch.setattr(mps_spike, "fetch", fetch)
    with pytest.raises(SystemExit, match="pinned to 0.4.6"):
        mps_spike.main()
    assert calls == ["check_version"]


def test_configure_model_enables_checkpointing_and_freezes_only_embeddings():
    model = mps_spike.tiny_model()
    full = mps_spike.configure_model(model, "none")
    assert all(layer.gradient_checkpointing for layer in model.encoder.layers)
    assert laya_adapter.head_checkpointing_on(model) is True
    frozen_model = mps_spike.tiny_model()
    frozen = mps_spike.configure_model(frozen_model, "embeddings_frozen")
    assert full - frozen == 1000 * 64
    assert [n for n, p in frozen_model.named_parameters() if not p.requires_grad] == \
        ["encoder.embeddings.tok_embeddings.weight"]
    with pytest.raises(SystemExit, match="never FITS"):
        mps_spike.configure_model(mps_spike.tiny_model(), "freeze_encoder")


def test_worker_step_keeps_checkpointing_on_and_amp_off(monkeypatch):
    calls, real = [], laya_adapter.forward

    def recording(model, batch, device, amp, detach_encoder):
        calls.append((amp, detach_encoder, laya_adapter.head_checkpointing_on(model), model.training))
        return real(model, batch, device, amp, detach_encoder)

    monkeypatch.setattr(laya_adapter, "forward", recording)
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
    assert spike.optimizer_steps(14854, 8, 4) == spike.optimizer_steps(14854, 4, 8) == 465
    assert spike.optimizer_steps(10, 4, 2) == 2
    for n, mb, accum in ((14854, 8, 4), (14854, 4, 8), (14854, 4, 2), (10, 4, 2), (1, 8, 4), (33, 4, 8)):
        assert spike.optimizer_steps(n, mb, accum) == laya_adapter.optimizer_updates(n, mb, accum, 1), (n, mb, accum)
    with pytest.raises(ValueError):
        spike.optimizer_steps(0, 4, 2)


def test_projection_time_gate_boundary():
    steps = spike.optimizer_steps(64, 4, 2)
    assert steps == 8 and spike.project_epoch_s(900.0, steps) == 7200.0
    ok = spike.verdict([config(attempt(4, 128, real_median=900.0, n_items=64), grad_accum=2)],
                       FULL)["per_config"][0]
    assert ok["steps_per_epoch"] == 8 and ok["projected_epoch_s"] == 7200.0
    assert ok["time_ok"] is True and ok["fits"] is True
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


# ---- verdict wiring (plan 02-02 task 2): OOM / error capture in the worker, markdown verdict line ----

def test_worker_records_mps_oom_as_an_attempt(monkeypatch):
    calls, real = [], laya_adapter.forward

    def oom_on_fourth(*args):
        calls.append(1)
        if len(calls) == 4:                     # warm-up = calls 1-2, measured padded step 0 = calls 3-4
            raise RuntimeError(OOM_TEXT)
        return real(*args)

    monkeypatch.setattr(laya_adapter, "forward", oom_on_fourth)
    payload = mps_spike.spike_worker(worker_cfg())
    assert payload["status"] == "oom", payload
    assert payload["oom"]["regime"] == "padded" and payload["oom"]["stage"] == "measured"
    assert payload["oom"]["step"] == 0 and payload["oom"]["message"].startswith("MPS backend out of memory")
    assert "real" not in payload["regimes"] and len(calls) == 4      # nothing runs after an OOM
    assert payload["regimes"]["padded"]["swap_delta_bytes"] is not None
    ident = (payload["micro_batch"], payload["max_len"], payload["rung"], payload["purpose"])
    assert ident == (2, 32, "none", "initial")
    assert spike.memory_ok(payload, 2) == (False, "oom in padded measured step 0")


def test_worker_records_unexpected_error_and_reraises_nothing(monkeypatch):

    def boom(*args):
        raise RuntimeError("something else broke")

    monkeypatch.setattr(laya_adapter, "forward", boom)
    payload = mps_spike.spike_worker(worker_cfg())
    assert payload["status"] == "error" and payload["error"] == "RuntimeError: something else broke", payload
    assert payload["micro_batch"] == 2 and payload["transport"] == "return"


def _report(protocol, cfgs):
    meta = {"split": {"path": "data/splits/selection_train.jsonl", "sha256": "0" * 64, "n_messages": 1450},
            "model": {"repo": "r", "subfolder": "s", "revision": "v"}, "versions": {"torch": "x"},
            "platform": {"macos": "26", "machine": "arm64", "total_memory_bytes": 16 * 2**30},
            "protocol": dict(protocol, grid=",".join("%dx%d" % tuple(g) for g in protocol["grid"]), seed=0, cpus=4)}
    report = spike.build_report(meta, cfgs)
    report["verdict"] = spike.verdict(cfgs, meta["protocol"])
    return report


def test_markdown_ends_with_verdict_line():
    md = spike.render_markdown(_report(FULL, [config(at_epoch(4, 128, 3000.0))]))
    last = md.strip().splitlines()[-1]
    assert last.startswith("**Verdict: FITS** — Phase 4 starts from micro-batch 4, max_len 128, grad-accum 8, "
                           "rung none, PYTORCH_ENABLE_MPS_FALLBACK=0 (projected neg8 epoch 50.0 min)"), last
    assert "## Verdict" in md and "Kaggle plan required" not in md
    err = {"status": "error", "error": "x", "micro_batch": 4, "max_len": 128, "rung": "none"}
    bad = spike.render_markdown(_report(FULL, [config(err)]))
    assert bad.strip().splitlines()[-1].startswith("**Verdict: DOES NOT FIT**")
    assert "Kaggle plan required" in bad and "1 attempts ended in an unexpected error or crash" in bad


def test_markdown_smoke_run_is_labelled():
    smoke = dict(FULL, grid=[[4, 128]], warmup_steps=1, measured_steps=2, effective_batch=8)
    md = spike.render_markdown(_report(smoke, [config(attempt(4, 128, swap=1))]))
    last = md.strip().splitlines()[-1]
    assert last.startswith("SMOKE RUN (not the Phase 2 verdict): **Verdict: DOES NOT FIT**"), last
    assert "Kaggle plan required" not in md


# ---- fallback-op inventory (plan 02-03): op names, stage tags, the D-11 blocks rule, the attempt chain ----

STRICT_TEXT = ("The operator 'aten::linalg_eig' is not currently implemented for the MPS device. If you want this op "
               "to be added in priority during the prototype phase of this feature, please comment on "
               "https://github.com/pytorch/pytorch/issues/77764.")
FALLBACK_TEXT = ("The operator 'aten::linalg_eig' is not currently supported on the MPS backend and will fall back to "
                 "run on the CPU. This may have performance implications.")


def fallback_warning(op):
    return FALLBACK_TEXT.replace("aten::linalg_eig", op)


def strict_raised(mb, max_len, op="aten::x", stage="step", rung="none"):
    """A strict (fallback 0) attempt that hit NotImplementedError, shaped like mps_spike's worker payload."""
    return {"status": "not_implemented", "purpose": "initial", "micro_batch": mb, "max_len": max_len,
            "grad_accum": 32 // mb, "rung": rung, "fallback": "0", "device": "mps", "param_device": "mps",
            "error_op": {"op": op, "stage": stage, "message": "The operator '%s' is not ..." % op},
            "fallback_ops": [], "regimes": {}}


def fallback_rerun(mb, max_len, ops, real_median=10.0, rung="none"):
    a = attempt(mb, max_len, rung=rung, real_median=real_median, fallback="1")
    return dict(a, purpose="op_inventory", fallback_ops=ops)


def test_op_name_parses_strict_and_fallback_texts():
    assert spike.op_name(STRICT_TEXT) == "aten::linalg_eig"
    assert spike.op_name(FALLBACK_TEXT) == "aten::linalg_eig"
    assert spike.op_name("something else") is None
    assert spike.OP_RE.search(STRICT_TEXT).group(1) == "aten::linalg_eig"


def test_is_fallback_warning():
    assert spike.is_fallback_warning(FALLBACK_TEXT) is True
    assert spike.is_fallback_warning(STRICT_TEXT) is False
    assert spike.is_fallback_warning("torch.utils.checkpoint: use_reentrant is deprecated") is False


def test_tag_fallback_ops_assigns_stage_and_dedups():
    msgs = [fallback_warning("aten::a"), "an unrelated warning", fallback_warning("aten::b"),
            fallback_warning("aten::a")]
    assert spike.tag_fallback_ops(msgs, [(0, "setup"), (2, "step")]) == [
        {"op": "aten::a", "stage": "setup"}, {"op": "aten::b", "stage": "step"}, {"op": "aten::a", "stage": "step"}]
    assert spike.tag_fallback_ops(["unrelated"], [(0, "setup")]) == []
    assert spike.tag_fallback_ops([fallback_warning("aten::a")] * 3, [(0, "step")]) == [
        {"op": "aten::a", "stage": "step"}]


def test_blocks_table():
    for stage in ("setup", "tokenize"):
        b, reason = spike.blocks(stage, 5.0, 1.0, False)
        assert b is False and "only in %s" % stage in reason and "D-11" in reason
    b, reason = spike.blocks("step", 1.11, 1.00, True)
    assert b is True and "all-MPS" in reason
    b, reason = spike.blocks("step", 1.10, 1.00, False)
    assert b is False, reason                         # "more than 10%" is strict
    b, reason = spike.blocks("step", 2.0, None, False)
    assert b is True and "no all-MPS baseline" in reason and "fails" in reason
    b, reason = spike.blocks("step", 2.0, None, True)
    assert b is False and "passes" in reason


def test_followup_reruns_only_not_implemented_strict_attempts():
    raised = strict_raised(4, 128)
    assert spike.followup([raised], {"measured_steps": 20}) == {"rung": "none", "fallback": "1",
                                                                "purpose": "op_inventory"}
    frozen = strict_raised(4, 128, rung="embeddings_frozen")
    assert spike.followup([frozen], {"measured_steps": 20})["rung"] == "embeddings_frozen"
    assert spike.followup([attempt(4, 128)], {"measured_steps": 20}) is None
    chain = [raised, fallback_rerun(4, 128, [{"op": "aten::x", "stage": "step"}])]
    assert spike.followup(chain, {"measured_steps": 20}) is None
    assert spike.standing(chain) == 1 and spike.standing([raised]) == 0


def test_op_inventory_and_blocking_map():
    # the fallback rerun projects 465 x 20 s = 9300 s > 7200 s, and there is no all-MPS baseline: the op blocks
    slow = config(strict_raised(4, 128), fallback_rerun(4, 128, [{"op": "aten::x", "stage": "step"}],
                                                        real_median=20.0), standing=1)
    clean = config(attempt(8, 128))
    inv = spike.op_inventory([slow, clean])
    assert len(inv["entries"]) == 1, inv
    e = inv["entries"][0]
    assert (e["op"], e["stage"], e["micro_batch"], e["max_len"], e["rung"]) == ("aten::x", "step", 4, 128, "none")
    assert e["blocks"] is True and "no all-MPS baseline" in e["reason"]
    assert "1 config(s) raised" in inv["note"]
    assert spike.blocking_ops(inv) == {(4, 128, "none"): ["aten::x"]}
    # a setup-only op never blocks, and a fast fallback rerun keeps a step op non-blocking
    ok = config(strict_raised(4, 128, op="aten::s", stage="setup"),
                fallback_rerun(4, 128, [{"op": "aten::s", "stage": "setup"}, {"op": "aten::y", "stage": "step"}],
                               real_median=1.0), standing=1)
    inv = spike.op_inventory([ok])
    assert [(e["op"], e["stage"], e["blocks"]) for e in inv["entries"]] == [("aten::s", "setup", False),
                                                                             ("aten::y", "step", False)]
    assert spike.blocking_ops(inv) == {}
    # a strict failure whose fallback rerun never ran blocks its config
    lone = spike.op_inventory([config(strict_raised(8, 256, op="aten::z"))])
    assert lone["entries"][0]["blocks"] is True and lone["entries"][0]["reason"] == "no fallback attempt ran"
    nowhere = spike.op_inventory([clean, config(attempt(4, 256))])
    assert nowhere["entries"] == [] and "raised nowhere" in nowhere["note"]


def test_op_inventory_blocks_against_all_mps_baseline():
    # an all-MPS (strict, ok) attempt of the same config and rung is the D-11 baseline: 11% slower blocks
    base = attempt(4, 128, real_median=1.0)
    chain = [strict_raised(4, 128), base, fallback_rerun(4, 128, [{"op": "aten::x", "stage": "step"}],
                                                         real_median=1.11)]
    e = spike.op_inventory([config(*chain, standing=2)])["entries"][0]
    assert e["blocks"] is True and "all-MPS" in e["reason"]
    chain[2] = fallback_rerun(4, 128, [{"op": "aten::x", "stage": "step"}], real_median=1.05)
    assert spike.op_inventory([config(*chain, standing=2)])["entries"][0]["blocks"] is False


def test_verdict_reads_last_attempt_as_standing():
    chain = [strict_raised(4, 128), fallback_rerun(4, 128, [{"op": "aten::s", "stage": "setup"}], real_median=1.0)]
    v = spike.verdict([config(*chain, standing=spike.standing(chain))], FULL)
    p = v["per_config"][0]
    assert p["fallback"] == "1" and p["status"] == "ok" and p["fits"] is True
    assert v["chosen"]["fallback"] == "1"


# ---- inventory wiring (plan 02-03 task 2): worker capture, per-config attempt chain, markdown section ----

def test_worker_tags_fallback_warnings_by_stage(monkeypatch):
    import warnings
    real_items, real_configure, real_forward = mps_spike.tiny_items, mps_spike.configure_model, laya_adapter.forward

    def items(*args):
        warnings.warn(fallback_warning("aten::tok"))
        return real_items(*args)

    def configure(*args):
        warnings.warn(fallback_warning("aten::cfg"))
        warnings.warn("an unrelated warning")
        return real_configure(*args)

    def forward(*args):
        warnings.warn(fallback_warning("aten::fwd"))
        return real_forward(*args)

    monkeypatch.setattr(mps_spike, "tiny_items", items)
    monkeypatch.setattr(mps_spike, "configure_model", configure)
    monkeypatch.setattr(laya_adapter, "forward", forward)
    payload = mps_spike.spike_worker(worker_cfg())
    assert payload["status"] == "ok", payload
    assert payload["fallback_ops"] == [{"op": "aten::tok", "stage": "tokenize"}, {"op": "aten::cfg", "stage": "setup"},
                                       {"op": "aten::fwd", "stage": "step"}]
    assert payload["other_warnings"] >= 1 and "env_seen" in payload


def test_worker_records_not_implemented_as_an_attempt(monkeypatch):
    import warnings
    calls, real = [], laya_adapter.forward

    def strict_on_third(*args):
        calls.append(1)
        warnings.warn(fallback_warning("aten::w"))
        if len(calls) == 3:                     # warm-up = calls 1-2, measured padded step 0 starts at call 3
            raise NotImplementedError(STRICT_TEXT)
        return real(*args)

    monkeypatch.setattr(laya_adapter, "forward", strict_on_third)
    payload = mps_spike.spike_worker(worker_cfg())
    assert payload["status"] == "not_implemented", payload
    assert payload["error_op"]["op"] == "aten::linalg_eig" and payload["error_op"]["stage"] == "step"
    assert payload["error_op"]["message"].startswith("The operator 'aten::linalg_eig'")
    assert payload["fallback_ops"] == [{"op": "aten::w", "stage": "step"}]
    assert payload["n_neg8_items"] == 64 and payload["param_device"] == "cpu" and payload["trainable_params"] > 0
    ident = (payload["micro_batch"], payload["max_len"], payload["rung"], payload["fallback"], payload["purpose"])
    assert ident == (2, 32, "none", "0", "initial") and payload["transport"] == "return"
    assert len(calls) == 3                                                  # nothing runs after the raise


def test_run_config_reruns_strict_failure_with_fallback_in_a_new_launch():
    seen = []

    def launcher(cfg, mode):
        seen.append((cfg["fallback"], cfg["expect_env"]["PYTORCH_ENABLE_MPS_FALLBACK"], cfg["purpose"], cfg["rung"],
                     cfg["run_name"]))
        if cfg["fallback"] == "0":
            return dict(strict_raised(4, 128), fallback_ops=[])
        return fallback_rerun(4, 128, [{"op": "aten::x", "stage": "step"}])

    base = {"micro_batch": 4, "max_len": 128, "grad_accum": 8, "device": "mps"}
    entry = mps_spike.run_config(base, 4, {"measured_steps": 20}, launcher=launcher)
    assert [s[:4] for s in seen] == [("0", "0", "initial", "none"), ("1", "1", "op_inventory", "none")]
    assert seen[0][4] != seen[1][4]
    assert entry["standing"] == 1 and [a["fallback"] for a in entry["attempts"]] == ["0", "1"]
    clean = mps_spike.run_config(base, 4, {"measured_steps": 20}, launcher=lambda cfg, mode: attempt(4, 128))
    assert len(clean["attempts"]) == 1 and clean["standing"] == 0
    crashed = mps_spike.run_config(base, 4, {"measured_steps": 20},
                                   launcher=lambda cfg, mode: {"status": "crashed", "returncode": -9})
    assert crashed["attempts"][0]["fallback"] == "0" and crashed["attempts"][0]["rung"] == "none"


def test_markdown_has_fallback_op_inventory_section():
    cfgs = [config(strict_raised(4, 128), fallback_rerun(4, 128, [{"op": "aten::x", "stage": "step"}],
                                                         real_median=20.0), standing=1)]
    report = _report(FULL, cfgs)
    report["op_inventory"] = spike.op_inventory(cfgs)
    md = spike.render_markdown(report)
    assert md.index("## Fallback-op inventory") < md.index("## Verdict")
    assert "| `aten::x` | step | 4x128 | none | yes | no all-MPS baseline" in md
    assert "| op_inventory | ok |" in md and "| initial | not_implemented |" in md
    empty = _report(FULL, [config(attempt(4, 128))])
    empty["op_inventory"] = spike.op_inventory(empty["configs"])
    assert "raised nowhere" in spike.render_markdown(empty)


# ---- token lengths (plan 02-04, D-09b): counts and percentiles that drive the D-03 truncation rule ----

def test_token_stats_counts_messages_over_thresholds():
    ts = spike.token_stats([[50, 60], [120, 130], [300]], [5, 80, 250])
    assert ts["n_messages"] == 3 and ts["n_items"] == 5
    assert ts["pct_messages_over_128"] == 66.67 and ts["pct_messages_over_256"] == 33.33
    assert ts["n_items_over_128"] == 2 and ts["n_items_over_256"] == 1
    assert ts["seq_len"]["max"] == 300 and ts["seq_len"]["min"] == 50
    assert ts["state_len"]["median"] == 80
    assert set(ts["seq_len"]) == set(ts["state_len"]) == {"min", "median", "p90", "p99", "max"}

    def numbers(v):
        return all(numbers(x) for x in v.values()) if isinstance(v, dict) else \
            isinstance(v, (int, float)) and not isinstance(v, bool)
    assert numbers(ts), ts


def test_token_stats_empty_split_refused():
    with pytest.raises(ValueError):
        spike.token_stats([], [])


def test_verdict_applies_truncation_rule_from_token_stats():
    ts = spike.token_stats([[80, 89], [40]], [30, 5])
    assert ts["pct_messages_over_128"] == 0.0
    v = spike.verdict([config(at_epoch(4, 128, 100.0)), config(at_epoch(4, 256, 300.0))], FULL, token_stats=ts)
    assert v["truncation_rule"].startswith("applied") and chosen(v) == (4, 128, "none")


def test_check_neg8_refuses_a_mismatched_attempt():
    cfgs = [config(attempt(4, 128)), config(dict(attempt(8, 128), status="crashed"), standing=0)]
    mps_spike.check_neg8(cfgs, 14854)                       # a crashed attempt without a count is not compared
    with pytest.raises(SystemExit, match="differs from 14854"):
        mps_spike.check_neg8([config(attempt(4, 128, n_items=14000))], 14854)


def test_markdown_has_token_lengths_section():
    report = _report(FULL, [config(at_epoch(4, 128, 3000.0))])
    report["token_stats"] = spike.token_stats([[80, 89], [40]], [30, 5])
    report["neg8"] = {"n_items": 14854, "n_positives": 3254, "n_messages": 1450, "seed": 0}
    md = spike.render_markdown(report)
    assert md.index("## Token lengths (D-09b)") < md.index("## Verdict")
    assert "0.00% of messages over 128 tokens" in md and "14854 neg8 items" in md


# ---- quiet-machine preflight (plan 02-04, RESEARCH Pitfall 2): refuse to measure when memory is short ----

def test_preflight_refuses_below_threshold():
    swap, outs = [3.0e9, 3.0e9], [10, 10]
    low = spike.preflight(4.4 * 2**30, swap, outs, 9.0)
    assert low["passed"] is False and low["available_gib"] == 4.4 and low["override"] is False
    assert spike.preflight(9.0 * 2**30, swap, outs, 9.0)["passed"] is True
    forced = spike.preflight(4.4 * 2**30, swap, outs, 0)
    assert forced["passed"] is True and forced["override"] is True and forced["min_available_gib"] == 0
    assert spike.MIN_AVAILABLE_GIB == 9.0


def test_preflight_flags_swap_growth_while_idle():
    grew = spike.preflight(10 * 2**30, [3.0e9, 3.2e9], [10, 12], 9.0)
    assert grew["swap_grew_while_idle"] is True and grew["swapouts_grew_while_idle"] is True
    assert grew["swap_used_bytes"] == [3.0e9, 3.2e9] and grew["swapouts"] == [10, 12]
    flat = spike.preflight(10 * 2**30, [3.0e9, 3.0e9], [10, 10], 9.0)
    assert flat["swap_grew_while_idle"] is False and flat["swapouts_grew_while_idle"] is False
    assert spike.preflight(10 * 2**30, [3.0e9, 3.0e9], [None, 10], 9.0)["swapouts_grew_while_idle"] is None


def test_orchestrator_refuses_busy_machine_before_any_child(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "mps_spike.py", "--min-available-gib", "100000", "--out", str(out),
                        "--storage", str(tmp_path / "ray")], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0, r.stdout
    assert "GiB available" in r.stderr and "--min-available-gib 0" in r.stderr, r.stderr
    assert not (out / "mps_spike.json").exists()
    neg = subprocess.run([sys.executable, "mps_spike.py", "--min-available-gib", "-1"], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=60)
    assert neg.returncode == 2 and "--min-available-gib" in neg.stderr, neg.stderr


def test_markdown_preflight_section_and_busy_machine_flag():
    quiet = _report(FULL, [config(at_epoch(4, 128, 3000.0))])
    quiet["preflight"] = spike.preflight(12 * 2**30, [3.0e9, 3.0e9], [10, 10], 9.0)
    md = spike.render_markdown(quiet)
    assert md.index("## Preflight") < md.index("## Verdict") and "Busy-machine run" not in md
    for pf in (spike.preflight(4.0 * 2**30, [3.0e9, 3.0e9], [10, 10], 0),
               spike.preflight(12 * 2**30, [3.0e9, 3.1e9], [10, 10], 9.0)):
        busy = dict(quiet, preflight=pf)
        md = spike.render_markdown(busy)
        assert md.index("Busy-machine run: background memory pressure was present") < md.index("## Verdict")


# ---- attempt chain (plan 02-05, D-08): one swap-only rerun, then the embeddings-frozen rung; D-09a overhead ----

POLICY = {"measured_steps": 20, "swap_rerun": True}


def chained(status, rung="none", fallback="0", purpose="initial", swap=0):
    """An attempt of the 4x128 config with the given outcome, shaped like mps_spike's worker payload."""
    if status == "not_implemented":
        return dict(strict_raised(4, 128, rung=rung), fallback=fallback, purpose=purpose)
    a = dict(attempt(4, 128, rung=rung, swap=swap, fallback=fallback), purpose=purpose)
    if status == "oom":
        a.update(status="oom", oom={"regime": "padded", "stage": "measured", "step": 3, "message": OOM_TEXT})
    elif status == "crashed":
        a = {"status": "crashed", "returncode": -9, "purpose": purpose, "micro_batch": 4, "max_len": 128,
             "grad_accum": 8, "rung": rung, "fallback": fallback, "device": "mps"}
    elif status == "error":
        a.update(status="error", error="ValueError: boom")
    return a


def test_followup_swap_rerun_once_then_rung():
    first = chained("ok", swap=1)
    assert spike.followup([first], POLICY) == {"rung": "none", "fallback": "0", "purpose": "swap_rerun"}
    again = chained("ok", purpose="swap_rerun", swap=1)
    assert spike.followup([first, again], POLICY) == {"rung": "embeddings_frozen", "fallback": "0",
                                                       "purpose": "rung_embeddings_frozen"}
    assert spike.followup([first], dict(POLICY, swap_rerun=False))["rung"] == "embeddings_frozen"
    # a swap rerun that ran clean stands: the chain ends on it
    assert spike.followup([first, chained("ok", purpose="swap_rerun")], POLICY) is None
    # an attempt that stopped short of the measured steps is a memory failure, never a swap rerun
    short = dict(first, regimes=dict(first["regimes"], padded=dict(first["regimes"]["padded"], times=_times(19, 1.0))))
    assert spike.followup([short], POLICY)["purpose"] == "rung_embeddings_frozen"


def test_followup_oom_and_crash_go_to_embeddings_frozen():
    for status in ("oom", "crashed"):
        assert spike.followup([chained(status)], POLICY) == {"rung": "embeddings_frozen", "fallback": "0",
                                                             "purpose": "rung_embeddings_frozen"}, status
    chain = [chained("oom"), chained("oom", rung="embeddings_frozen", purpose="rung_embeddings_frozen")]
    assert spike.followup(chain, POLICY) is None


def test_followup_keeps_fallback_setting_on_the_rung():
    chain = [chained("not_implemented"), chained("oom", fallback="1", purpose="op_inventory")]
    assert spike.followup(chain, POLICY) == {"rung": "embeddings_frozen", "fallback": "1",
                                             "purpose": "rung_embeddings_frozen"}


def test_followup_error_or_passing_attempt_ends_chain():
    assert spike.followup([chained("error")], POLICY) is None
    assert spike.followup([chained("ok")], POLICY) is None
    assert spike.followup([chained("ok", rung="embeddings_frozen")], POLICY) is None


def test_followup_never_leaves_allowed_rungs():
    outcomes = ("ok", "ok_swap", "oom", "crashed", "not_implemented")
    ends = []

    def drive(chain, policy):
        nxt = spike.followup(chain, policy)
        if nxt is None:
            ends.append(len(chain))
            return
        assert nxt["rung"] in spike.RUNGS and nxt["fallback"] in ("0", "1"), nxt
        assert len(chain) < 5, [(a["rung"], a["fallback"], a["purpose"], a["status"]) for a in chain]
        for o in outcomes:
            # a fallback attempt never raises NotImplementedError, and a swap rerun repeats an op path that
            # already completed every step, so neither can return not_implemented
            if o == "not_implemented" and (nxt["fallback"] == "1" or nxt["purpose"] == "swap_rerun"):
                continue
            a = chained("ok" if o == "ok_swap" else o, rung=nxt["rung"], fallback=nxt["fallback"],
                        purpose=nxt["purpose"], swap=1 if o == "ok_swap" else 0)
            drive(chain + [a], policy)

    for policy in (POLICY, dict(POLICY, swap_rerun=False)):
        for o in outcomes:
            drive([chained("ok" if o == "ok_swap" else o, swap=1 if o == "ok_swap" else 0)], policy)
    assert ends and max(ends) == 5


def test_ray_overhead_delta():
    ray = dict(attempt(4, 128, real_median=1.1), torch_threads=4)
    inproc = dict(attempt(4, 128, real_median=1.0), torch_threads=4)
    o = spike.ray_overhead(ray, inproc, "chosen")
    assert (o["basis"], o["micro_batch"], o["max_len"], o["rung"], o["fallback"]) == ("chosen", 4, 128, "none", "0")
    assert o["ray"]["real_median_s"] == 1.1 and o["inprocess"]["real_median_s"] == 1.0
    assert o["delta_real_pct"] == 10.0 and o["delta_padded_pct"] == 0.0 and o["inprocess"]["status"] == "ok"
    gone = spike.ray_overhead(ray, {"status": "crashed", "returncode": -9}, "reference")
    assert gone["basis"] == "reference" and gone["inprocess"]["status"] == "crashed"
    assert gone["delta_real_pct"] is None and gone["delta_padded_pct"] is None
    assert gone["inprocess"]["real_median_s"] is None and gone["ray"]["real_median_s"] == 1.1


# ---- chain and overhead wiring (plan 02-05 task 2): rung in the worker, policy, in-process rerun, markdown ----

def test_worker_embeddings_frozen_rung_reports_fewer_trainable_params():
    payload = mps_spike.spike_worker(worker_cfg(rung="embeddings_frozen", purpose="rung_embeddings_frozen"))
    assert payload["status"] == "ok", payload
    assert payload["trainable_params"] == payload["total_params"] - 1000 * 64
    assert (payload["rung"], payload["purpose"]) == ("embeddings_frozen", "rung_embeddings_frozen")


def test_run_config_swap_rerun_then_embeddings_frozen_in_new_launches():
    seen = []

    def launcher(cfg, mode):
        seen.append((cfg["rung"], cfg["fallback"], cfg["purpose"], cfg["expect_env"]["PYTORCH_ENABLE_MPS_FALLBACK"],
                     mode))
        return attempt(4, 128, rung=cfg["rung"], swap=1 if cfg["rung"] == "none" else 0)

    base = {"micro_batch": 4, "max_len": 128, "grad_accum": 8, "device": "mps"}
    entry = mps_spike.run_config(base, 4, POLICY, launcher=launcher)
    assert seen == [("none", "0", "initial", "0", "ray"), ("none", "0", "swap_rerun", "0", "ray"),
                    ("embeddings_frozen", "0", "rung_embeddings_frozen", "0", "ray")]
    assert entry["standing"] == 2 and [a["purpose"] for a in entry["attempts"]] == [s[2] for s in seen]
    seen.clear()
    mps_spike.run_config(base, 4, dict(POLICY, swap_rerun=False), launcher=launcher)
    assert [s[2] for s in seen] == ["initial", "rung_embeddings_frozen"]


def test_measure_overhead_reruns_chosen_or_reference_in_process():
    launched = []

    def launcher(cfg, mode):
        launched.append((mode, cfg["purpose"], cfg["micro_batch"], cfg["max_len"], cfg["rung"], cfg["fallback"],
                         cfg["expect_env"]["OMP_NUM_THREADS"], cfg["cpus"]))
        return dict(attempt(cfg["micro_batch"], cfg["max_len"], real_median=1.0), torch_threads=cfg["cpus"])

    ray = dict(attempt(8, 128, real_median=1.1), torch_threads=4)
    configs = [config(attempt(4, 128, swap=1)), config(ray)]
    bases = {(4, 128): {"micro_batch": 4, "max_len": 128, "grad_accum": 8, "device": "mps", "cpus": 4},
             (8, 128): {"micro_batch": 8, "max_len": 128, "grad_accum": 4, "device": "mps", "cpus": 4}}
    before = [len(c["attempts"]) for c in configs]
    v = spike.verdict(configs, FULL)
    assert chosen(v) == (8, 128, "none")
    o = mps_spike.measure_overhead(v, configs, bases, 4, launcher=launcher)
    assert launched == [("inprocess", "ray_overhead", 8, 128, "none", "0", "4", 4)]
    assert o["basis"] == "chosen" and o["delta_real_pct"] == 10.0 and o["inprocess"]["torch_threads"] == 4
    assert [len(c["attempts"]) for c in configs] == before            # never added to any config's attempts
    launched.clear()
    nothing = spike.verdict([configs[0]], FULL)
    o = mps_spike.measure_overhead(nothing, [configs[0]], bases, 4, launcher=launcher)
    assert o["basis"] == "reference" and launched[0][2:6] == (4, 128, "none", "0")

    def refused(cfg, mode):
        raise SystemExit("spike run refused: worker env")
    o = mps_spike.measure_overhead(v, configs, bases, 4, launcher=refused)
    assert o["inprocess"]["status"] == "refused" and o["delta_real_pct"] is None


def test_swap_rerun_flag_defaults_on():
    r = subprocess.run([sys.executable, "mps_spike.py", "--help"], cwd=REPO_ROOT, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "--swap-rerun, --no-swap-rerun" in r.stdout, r.stdout


def test_markdown_run_policy_and_overhead_section():
    report = _report(FULL, [config(at_epoch(4, 128, 3000.0))])
    report["meta"]["policy"] = {"min_available_gib": 0.0, "swap_rerun": False, "tie_tolerance": 0.05}
    report["preflight"] = spike.preflight(4.0 * 2**30, [3.0e9, 3.0e9], [10, 10], 0)
    report["ray_overhead"] = spike.ray_overhead(dict(attempt(4, 128, real_median=1.1), torch_threads=4),
                                                dict(attempt(4, 128, real_median=1.0), torch_threads=4), "chosen")
    md = spike.render_markdown(report)
    assert "- Run policy: min_available_gib 0.0, swap_rerun False, tie_tolerance 0.05" in md
    assert md.index("Run policy") < md.index("| micro-batch |")
    assert md.index("## Ray vs in-process (D-09a)") < md.index("Busy-machine run") < md.index("## Verdict")
    assert "real +10.0%" in md
