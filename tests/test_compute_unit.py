"""pilot.compute (LAYA-07): the compute checkpoint's projection math, spike start config, report and markdown, on toy
inputs (offline, no torch, no Ray). The orchestrator end to end is tests/test_compute_ray.py.
"""
import json
import os
import subprocess
import sys

import pytest

from pilot import compute
from pilot.data import REPO_ROOT

pytestmark = pytest.mark.unit

SPIKE_REPORT = os.path.join(REPO_ROOT, "results", "spike", "mps_spike.json")
# the must_haves toy input: 14,854 / 17,001 neg8 items are the committed selection_train / final_train counts
TOY_LAYA = {"s_per_step": 2.0, "s_per_step_epoch1": 2.0, "epochs_measured": 2, "save_s_per_epoch": 10.0,
            "load_s": 20.0, "calib_s": 5.0, "final_save_s": 3.0, "micro_batch": 8, "grad_accum": 4, "max_len": 128,
            "neg": 8, "peak_total_bytes": 9 * 2**30, "peak_driver_bytes": 8 * 2**30, "peak_rss_bytes": 2**30,
            "swap_delta_bytes": 0, "rung": "none", "fallback": False, "weaker": False, "device": "mps",
            "param_device": "mps", "grad_checkpointing": True, "amp": False,
            "env": ["OMP_NUM_THREADS=4", "PYTORCH_ENABLE_MPS_FALLBACK=0"], "train_state_bytes": 6 * 2**30}
TOY_COUNTS = {"selection_train": {"n_messages": 1450, "n_items_neg8": 14854},
              "final_train": {"n_messages": 1658, "n_items_neg8": 17001}}
TOY_MATRIX = {"lr_grid": 3, "max_epochs": 4, "final_seeds": 3}


def toy_rows():
    return [{"epoch": e, "loss": 0.5, "train_s": 930.0, "save_s": 10.0, "optimizer_steps": 465, "n_items": 14444,
             "peak_total_bytes": 9 * 2**30, "peak_driver_bytes": 8 * 2**30, "peak_rss_bytes": 2**30,
             "current_allocated_bytes": 7 * 2**30, "swap_delta_bytes": 0, "segment": 0, "resumed": False}
            for e in (1, 2)]


def toy_measured():
    return {"laya": dict(TOY_LAYA), "epochs": toy_rows()}


def toy_meta(protocol_complete=True):
    return {"generated_utc": "2026-10-14T00:00:00Z",
            "split": {"path": "splits/selection_train.jsonl", "sha256": "a" * 64, "n_messages": 1450},
            "manifest": {"path": "data/splits.json", "sha256": "b" * 64, "is_committed": True},
            "spike_report": {"path": "results/spike/mps_spike.json", "sha256": "c" * 64},
            "model": {"repo": "org/laya", "subfolder": "multilingual", "revision": "d" * 40, "model_sha256": "e" * 64},
            "versions": {"torch": "2.14.1", "python": "3.12.0"},
            "platform": {"macos": "26.0", "machine": "arm64", "total_memory_bytes": 16 * 2**30},
            "protocol": {"device": "mps", "epochs": 2, "seed": 0, "cpus": 4, "neg": 8,
                         "spike_start": {"micro_batch": 8, "max_len": 128, "grad_accum": 4, "rung": "none",
                                         "fallback": 0},
                         "run_id": "laya-selection_train-neg8-s0-12345678",
                         "test_hooks": {"max_messages": None, "calib_frac": None},
                         "protocol_complete": protocol_complete}}


def toy_preflight(override=False):
    return {"available_gib": 9.5, "min_available_gib": 0 if override else 7.0, "swap_used_bytes": [0, 0],
            "swap_grew_while_idle": False, "swapouts": [1, 1], "swapouts_grew_while_idle": False, "passed": True,
            "override": override}


def line(projection, id_):
    return next(l for l in projection["lines"] if l["id"] == id_)


def test_projection_toy_numbers():
    p = compute.project(toy_measured(), TOY_COUNTS, TOY_MATRIX)
    sel, fin = line(p, "laya_selection"), line(p, "laya_final")
    # 3 x [4 x (465 steps x 2.0 s + 10 s) + 20 + 5 + 3] and 3 x [4 x (532 x 2.0 + 10) + 28]
    assert sel["seconds"] == pytest.approx(11364) and fin["seconds"] == pytest.approx(12972)
    assert sel["units_per_epoch"] == 465 and fin["units_per_epoch"] == 532
    assert sel["runs"] == 3 and sel["epochs"] == 4 and fin["runs"] == 3 and fin["epochs"] == 4
    assert sel["basis"] == fin["basis"] == "measured" and sel["unit"] == "optimizer step"
    assert sel["hours"] == pytest.approx(11364 / 3600) and "465" in sel["formula"]
    assert p["total_s"] == pytest.approx(sum(l["seconds"] for l in p["lines"]))


def test_measured_from_checkpoint_takes_the_largest_rate(tmp_path):
    rows = toy_rows()
    rows[0]["train_s"], rows[1]["train_s"] = 1000.0, 930.0  # epoch 1 includes the MPS warm-up
    record = {"device": "mps", "param_device": "mps", "env": ["PYTORCH_ENABLE_MPS_FALLBACK=0"],
              "grad_checkpointing": True, "amp": False, "rung": "none", "fallback": False, "weaker": False,
              "load_s": 20.0, "calib_s": 5.0, "final_save_s": 3.0, "train_state_bytes": 123, "epochs": rows}
    cfg = {"max_len": 128, "training": {"ray_train": {"micro_batch": 8, "grad_accum": 4, "neg": 8, "record": record}}}
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    m = compute.measured_from_checkpoint(str(tmp_path))
    assert m["laya"]["s_per_step"] == pytest.approx(1000.0 / 465)
    assert m["laya"]["s_per_step_epoch1"] == pytest.approx(1000.0 / 465)
    assert m["laya"]["epochs_measured"] == 2 and m["laya"]["max_len"] == 128 and m["laya"]["neg"] == 8
    assert m["laya"]["peak_total_bytes"] == 9 * 2**30 and m["laya"]["peak_driver_bytes"] == 8 * 2**30
    assert m["epochs"] == rows                              # LAYA-06: the rows reach the report unchanged


def test_spike_start_reads_the_committed_spike_report():
    s = compute.spike_start(SPIKE_REPORT)
    assert s == {"micro_batch": 8, "max_len": 128, "grad_accum": 4, "rung": "none", "fallback": 0}


def test_spike_start_refuses_a_does_not_fit_report(tmp_path):
    with open(SPIKE_REPORT, encoding="utf-8") as f:
        report = json.load(f)
    report["verdict"] = dict(report["verdict"], result="DOES NOT FIT", chosen=None)
    path = tmp_path / "spike.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="not FITS"):
        compute.spike_start(str(path))


def test_spike_start_refuses_a_fallback_config(tmp_path):
    with open(SPIKE_REPORT, encoding="utf-8") as f:
        report = json.load(f)
    report["verdict"]["chosen"]["fallback"] = "1"
    path = tmp_path / "spike.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="FALLBACK=1"):
        compute.spike_start(str(path))


def test_torch_free_import():
    code = "import sys, compute_checkpoint, pilot.compute; assert 'torch' not in sys.modules, 'torch imported'"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_markdown_banners_and_round_trip(tmp_path):
    p = compute.project(toy_measured(), TOY_COUNTS, TOY_MATRIX)
    smoke = compute.build_report(toy_meta(False), toy_measured(), TOY_COUNTS, TOY_MATRIX, p, toy_preflight(True))
    md = compute.render_markdown(smoke)
    assert md.startswith("# Compute checkpoint (Phase 4, LAYA-07)") and "SMOKE RUN" in md and "Busy-machine" in md
    real = compute.build_report(toy_meta(True), toy_measured(), TOY_COUNTS, TOY_MATRIX, p, toy_preflight(False))
    md = compute.render_markdown(real)
    assert "SMOKE RUN" not in md and "Busy-machine" not in md and "laya_selection" in md and "Total" in md
    path = compute.write_report(str(tmp_path), real)
    with open(tmp_path / compute.JSON_NAME, encoding="utf-8") as f:
        parsed = json.load(f)
    with open(path, encoding="utf-8") as f:
        assert f.read() == compute.render_markdown(parsed)
