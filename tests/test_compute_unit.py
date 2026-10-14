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
TOY_COUNTS = {"selection_train": {"n_messages": 1450, "n_items_neg8": 14854, "n_items_full": 69600},
              "val": {"n_messages": 208, "n_items_neg8": 2147, "n_items_full": 9984},
              "final_train": {"n_messages": 1658, "n_items_neg8": 17001, "n_items_full": 79584},
              "test": {"n_messages": 413, "n_items_neg8": 4228, "n_items_full": 19824},
              "hotels_probe": {"n_messages": 1009, "n_items_neg8": 8072, "n_items_full": 48432}}
TOY_MATRIX = {"lr_grid": 3, "max_epochs": 4, "final_seeds": 3, "neg": 8, "effective_batch": 32, "berturk_epochs": 10,
              "berturk_batch": 32, "fewshot_tags": 6, "fewshot_k": [5, 10], "fewshot_draws": 3,
              "fewshot_neg_per_pos": 4, "fewshot_replay": 200, "fewshot_epochs": 10, "fewshot_base_runs": 3,
              "eval_seeds": 3}
LINE_IDS = ["laya_selection", "laya_val_scoring", "laya_final", "berturk_selection", "berturk_final", "fewshot_base",
            "fewshot_incremental", "fewshot_scoring", "final_scoring"]
ZEROSHOT_VAL = os.path.join(REPO_ROOT, "results", "dumps", "laya-zeroshot-7b928d82", "val.json")


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
            "platform": {"macos": "26.6.2", "machine": "arm64", "total_memory_bytes": 16 * 2**30},
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


# ---- Task 2: the whole experiment matrix, what-if, disk, assumptions, path-free report ----

def full_report(protocol_complete=True, override=False):
    measured = toy_measured()
    p = compute.project(measured, TOY_COUNTS, TOY_MATRIX)
    return compute.build_report(toy_meta(protocol_complete), measured, TOY_COUNTS, TOY_MATRIX, p,
                                toy_preflight(override), what_if=compute.what_if(measured, TOY_COUNTS, TOY_MATRIX),
                                disk=compute.disk_budget(2**30, TOY_MATRIX),
                                assumptions=compute.assumptions(TOY_MATRIX, measured, TOY_COUNTS))


def test_matrix_lines_and_total():
    p = compute.project(toy_measured(), TOY_COUNTS, TOY_MATRIX)
    assert [l["id"] for l in p["lines"]] == LINE_IDS
    for l in p["lines"]:
        assert l["basis"] in ("measured", "estimate", "provisional"), l["id"]
        assert l["formula"] and l["source"] and l["seconds"] > 0, l["id"]
        assert l["hours"] == pytest.approx(l["seconds"] / 3600)
    assert p["total_s"] == pytest.approx(sum(l["seconds"] for l in p["lines"]))
    assert p["total_h"] == pytest.approx(p["total_s"] / 3600)
    for id_ in ("berturk_selection", "berturk_final"):
        l = line(p, id_)
        assert l["basis"] == "estimate" and l["s_per_unit"] == compute.ESTIMATES["berturk_s_per_step"]
    # 3 LRs x 10 epochs x ceil(1450 / 32) = 46 steps x the estimated s/step
    est = compute.ESTIMATES["berturk_s_per_step"]
    assert line(p, "berturk_selection")["seconds"] == pytest.approx(3 * 10 * 46 * est)
    assert line(p, "berturk_final")["seconds"] == pytest.approx(3 * 10 * 52 * est)
    assert line(p, "laya_selection")["basis"] == "measured"
    # few-shot depends on a protocol Phase 8 freezes, and its BERTurk part on the estimate
    assert line(p, "fewshot_incremental")["basis"] in ("estimate", "provisional")


def test_fewshot_line_counts():
    matrix = dict(TOY_MATRIX, fewshot_tags=6, fewshot_k=[5, 10], fewshot_neg_per_pos=4, fewshot_replay=200,
                  fewshot_draws=3, fewshot_epochs=10)
    p = compute.project(toy_measured(), TOY_COUNTS, matrix)
    inc = line(p, "fewshot_incremental")
    msgs = {(part["model"], part["k"]): part["messages"] for part in inc["parts"]}
    assert msgs == {("laya", 5): 350, ("laya", 10): 500, ("berturk", 5): 350, ("berturk", 10): 500}
    for part in inc["parts"]:
        assert part["runs"] == 3 and part["epochs"] == 10
    laya5 = next(part for part in inc["parts"] if part["model"] == "laya" and part["k"] == 5)
    items = -(-350 * 14854 // 1450)                         # ceil(350 x selection_train neg8 items per message)
    assert laya5["seconds"] == pytest.approx(3 * compute.laya_run_s(items, 10, TOY_LAYA))
    assert inc["seconds"] == pytest.approx(sum(part["seconds"] for part in inc["parts"]))
    base = line(p, "fewshot_base")
    assert {part["model"]: part["runs"] for part in base["parts"]} == {"laya": 3, "berturk": 3}
    scoring = line(p, "fewshot_scoring")
    assert {part["model"]: part["runs"] for part in scoring["parts"]} == {"laya": 6, "berturk": 6}
    assert all(part["units_per_epoch"] == 413 for part in scoring["parts"])


def test_scoring_rate_falls_back_to_the_committed_cpu_rate():
    with open(ZEROSHOT_VAL, encoding="utf-8") as f:
        side = json.load(f)
    rate = side["timing"]["seconds"] / side["split"]["n_messages"]
    p = compute.project(toy_measured(), TOY_COUNTS, TOY_MATRIX)
    val = line(p, "laya_val_scoring")
    assert val["s_per_unit"] == pytest.approx(rate) and val["basis"] == "measured"
    assert "cpu" in val["source"] and "2 Ray actors" in val["source"]
    assert val["runs"] == 12 and val["units_per_epoch"] == 208 and val["unit"] == "message"
    assert val["seconds"] == pytest.approx(12 * 208 * rate)
    final = line(p, "final_scoring")
    laya = next(part for part in final["parts"] if part["model"] == "laya")
    assert laya["s_per_unit"] == pytest.approx(rate) and laya["units_per_epoch"] == 208 + 413 + 1009
    assert laya["runs"] == 3


def test_what_if_table():
    rows = compute.what_if(toy_measured(), TOY_COUNTS, TOY_MATRIX)
    assert len(rows) == 10
    grid = [(r["lr_grid"], r["max_epochs"]) for r in rows if r["fewshot_base_runs"] == TOY_MATRIX["fewshot_base_runs"]]
    assert sorted(grid) == [(g, e) for g in (1, 2, 3) for e in (2, 3, 4)]
    fallback = [r for r in rows if r["fewshot_base_runs"] == 1]
    assert len(fallback) == 1 and "fallback" in fallback[0]["label"]
    assert all(r["total_h"] == pytest.approx(r["total_s"] / 3600) for r in rows)
    same = next(r for r in rows if r["lr_grid"] == 3 and r["max_epochs"] == 4 and r["fewshot_base_runs"] == 3)
    assert same["total_s"] == pytest.approx(compute.project(toy_measured(), TOY_COUNTS, TOY_MATRIX)["total_s"])
    assert fallback[0]["total_s"] < same["total_s"]


def test_disk_budget():
    m = 600 * 2**20
    d = compute.disk_budget(model_bytes=m, matrix=TOY_MATRIX)
    assert d["model_bytes"] == m and d["train_state_bytes"] == 6 * m
    assert d["per_run_peak_bytes"] == 3 * 6 * m + 4 * m + m
    assert d["selection_exports_bytes"] == 3 * 4 * m
    assert compute.disk_budget(m, TOY_MATRIX, train_state_bytes=5 * m)["train_state_bytes"] == 5 * m


def test_report_is_path_and_text_free():
    import re
    from test_repo_hygiene import text_problems
    report = full_report()
    parsed = json.loads(json.dumps(report, ensure_ascii=False))
    home = os.path.expanduser("~").rstrip("/")
    assert text_problems(parsed, home, {"1", "3", "40", "nan"}) == []
    number_like = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$|^(nan|inf|-inf)$", re.I)

    def strings(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from strings(v)
        elif isinstance(obj, str):
            yield obj
    assert [s for s in strings(parsed) if number_like.match(s.strip())] == []
    assert all(re.match(r"^[A-Z_]+=", e) for e in parsed["measured"]["laya"]["env"])
    for key in ("what_if", "disk", "assumptions"):
        assert parsed[key], key


def test_markdown_lists_everything():
    report = full_report(protocol_complete=True)
    md = compute.render_markdown(report)
    for l in report["projection"]["lines"]:
        assert l["what"] in md, l["id"]
    for text in ("Total", "What-if", "Disk", "Assumptions to confirm"):
        assert text in md, text
    for a in report["assumptions"]:
        assert a in md
    assert "SMOKE RUN" not in md
    assert "SMOKE RUN" in compute.render_markdown(full_report(protocol_complete=False))


def _synthetic(tmp_path):
    from conftest import make_raw
    import pilot.data
    root, raw_sha256 = make_raw(str(tmp_path / "raw"))
    out = str(tmp_path / "data")
    pilot.data.prepare(root, out, raw_sha256)
    return out


def test_split_counts_on_synthetic_splits(tmp_path):
    from pilot import runs
    from pilot.data import parse_jsonl
    out = _synthetic(tmp_path)
    counts = compute.split_counts(os.path.join(out, "splits.json"))
    assert sorted(counts) == sorted(["selection_train", "val", "final_train", "test", "hotels_probe"])
    for name, c in counts.items():
        with open(os.path.join(out, "splits", name + ".jsonl"), "rb") as f:
            rows = parse_jsonl(f.read())
        assert c == {"n_messages": len(rows), "n_items_neg8": runs.neg_items(rows, 8),
                     "n_items_full": runs.neg_items(rows, 0)}, name


def test_split_counts_refuse_a_tampered_split(tmp_path):
    out = _synthetic(tmp_path)
    with open(os.path.join(out, "splits", "test.jsonl"), "ab") as f:
        f.write(b"\n")
    with pytest.raises(SystemExit, match="sha256"):
        compute.split_counts(os.path.join(out, "splits.json"))


@pytest.mark.data
def test_split_counts_on_the_committed_splits():
    counts = compute.split_counts(os.path.join(REPO_ROOT, "data", "splits.json"))
    assert {k: v["n_items_neg8"] for k, v in counts.items()} == {
        "selection_train": 14854, "final_train": 17001, "val": 2147, "test": 4228, "hotels_probe": 8072}
    assert {k: v["n_messages"] for k, v in counts.items()} == {
        "selection_train": 1450, "final_train": 1658, "val": 208, "test": 413, "hotels_probe": 1009}
