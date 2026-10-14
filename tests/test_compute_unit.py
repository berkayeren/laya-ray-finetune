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
        assert l["basis"] in ("measured", "measured (cpu, wall)", "estimate", "provisional"), l["id"]
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
    assert val["s_per_unit"] == pytest.approx(rate) and val["basis"] == "measured (cpu, wall)"     # IN-03
    assert "cpu" in val["source"] and "2 Ray actors" in val["source"]
    measured = dict(toy_measured(), berturk={"status": "ok", "s_per_step": 0.5, "s_per_msg_infer": 0.004})
    assert line(compute.project(measured, TOY_COUNTS, TOY_MATRIX), "final_scoring")["basis"] == "measured (cpu, wall)"
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


# ---- plan 04-06 Task 1: the probe protocol (registration, ladder chain, resume summary, kill and reap) ----

def test_epoch_registered(tmp_path):
    rid = "laya-selection_train-neg8-s0-12345678"
    run_dir = tmp_path / rid
    run_dir.mkdir()
    assert compute.epoch_registered(str(tmp_path), rid, 1) is False          # no snapshot yet
    snap = run_dir / compute.SNAPSHOT

    def write(names, **extra):                                               # Ray 2.59's snapshot layout
        snap.write_text(json.dumps(dict({"ray_version": compute.SNAPSHOT_RAY, "checkpoint_results": [
            {"version": 0, "checkpoint_dir_name": n, "metrics": {"epoch": 1}} for n in names]}, **extra)),
            encoding="utf-8")
    write([])
    (run_dir / "epoch-001").mkdir()
    (run_dir / "epoch-001" / "train_state.pt").write_bytes(b"x")
    assert compute.epoch_registered(str(tmp_path), rid, 1) is False          # the snapshot does not name epoch-001
    write([], note="epoch-001")                                              # named, but not as a checkpoint entry
    assert compute.epoch_registered(str(tmp_path), rid, 1) is False
    write(["epoch-001"])
    snap.write_text(snap.read_text(encoding="utf-8")[:40], encoding="utf-8")  # Ray rewrites it in place: half-written
    assert compute.epoch_registered(str(tmp_path), rid, 1) is False
    write(["epoch-001"])
    (run_dir / "epoch-001" / "train_state.pt").unlink()
    assert compute.epoch_registered(str(tmp_path), rid, 1) is False          # named, but no train_state.pt
    (run_dir / "epoch-001" / "train_state.pt").write_bytes(b"x")
    assert compute.epoch_registered(str(tmp_path), rid, 1) is True
    assert compute.epoch_registered(str(tmp_path), rid, 2) is False


def _fake_launcher(statuses):
    seen, todo = [], list(statuses)

    def launch(rung):
        seen.append(rung)
        return {"status": todo.pop(0), "run_id": "laya-%d" % len(seen)}
    return launch, seen


def test_attempt_chain_walks_the_ladder_on_oom():
    launch, seen = _fake_launcher(["oom", "ok"])
    attempts, status = compute.attempt_chain("none", launch)
    assert status == "ok" and seen == ["none", "embeddings_frozen"]
    assert [a["rung"] for a in attempts] == ["none", "embeddings_frozen"]
    assert [a["status"] for a in attempts] == ["oom", "ok"]


def test_attempt_chain_ladder_exhausted():
    launch, seen = _fake_launcher(["oom"] * 4)
    attempts, status = compute.attempt_chain("none", launch)
    assert status == "ladder_exhausted"
    assert [a["rung"] for a in attempts] == ["none", "embeddings_frozen", "embeddings_frozen+smaller_batch",
                                             "encoder_frozen"]


def test_attempt_chain_stops_on_failure():
    launch, seen = _fake_launcher(["failed", "ok"])
    attempts, status = compute.attempt_chain("none", launch)
    assert status == "failed" and seen == ["none"] and attempts[0]["rung"] == "none"


def test_an_oom_after_the_last_epoch_does_not_walk_the_ladder():
    """04-REVIEW WR-01: a calibration or final-save OOM ends the chain as failed (rerun to resume), not on a weaker
    rung that would retrain under another run_id and record a fallback; a training-step OOM still walks it."""
    import compute_checkpoint as cc
    from pilot import runs
    calib = runs.oom_message(None, None, "none", runs.STAGE_CALIBRATE)
    a = cc.oom_attempt({"run_id": "laya-a"}, runs.failure_line("[Rank 0]\nRuntimeError: " + calib), {"mode": "kill"})
    assert a["status"] == "failed" and a["line"] == calib
    # 04-REVIEW IN-06: the report never advises a resume whose storage the orchestrator deletes
    assert calib.split("; ")[0] in a["reason"] and "rerun the same command" not in a["reason"]
    assert "rerun compute_checkpoint.py" in a["reason"]
    md = "\n".join(compute._attempt_lines([dict(a, rung="none")], "failed"))
    assert a["reason"] in md and "rerun the same command" not in md
    kept = cc.oom_attempt({"run_id": "laya-a"}, calib, {"mode": "kill"}, keep_run=True)
    assert "--keep-run kept" in kept["reason"] and "rerun the same command" not in kept["reason"]
    assert cc.oom_attempt({"run_id": "laya-a"}, runs.oom_message(2, 5, "none"), {})["status"] == "oom"
    launch, seen = _fake_launcher(["failed"])
    attempts, status = compute.attempt_chain("none", launch)
    assert status == "failed" and seen == ["none"]


def _segment(i, start, load=None, peak=None):
    return {"segment": i, "start_epoch": start, "resumed": start > 0, "git_sha": "a" * 40, "git_dirty": False,
            "device": "mps", "resume_load_s": load, "resume_peak_total_bytes": peak}


def test_resume_summary_two_segments():
    rows = toy_rows()
    rows[1].update(segment=1, resumed=True)
    record = {"segments": [_segment(0, 0), _segment(1, 1, 42.5, 9 * 2**30)], "epochs": rows}
    s = compute.resume_summary(record)
    assert s == {"segments": 2, "resume_load_s": 42.5, "resume_peak_total_bytes": 9 * 2**30,
                 "resume_demonstrated": True}


def test_resume_summary_one_segment_is_not_a_resume():
    record = {"segments": [_segment(0, 0)], "epochs": toy_rows()}
    s = compute.resume_summary(record)
    assert s["segments"] == 1 and s["resume_demonstrated"] is False and s["resume_load_s"] is None
    gap = toy_rows()
    gap[1].update(epoch=3, segment=1, resumed=True)                          # rows 1, 3: a gap
    s = compute.resume_summary({"segments": [_segment(0, 0), _segment(1, 1, 1.0, 1)], "epochs": gap})
    assert s["resume_demonstrated"] is False


def test_markdown_shows_attempts_and_a_missed_kill():
    measured = toy_measured()
    p = compute.project(measured, TOY_COUNTS, TOY_MATRIX)
    attempts = [{"rung": "none", "status": "oom", "run_id": "laya-a", "line": "MPS out of memory at epoch 1"},
                {"rung": "embeddings_frozen", "status": "ok", "run_id": "laya-b"}]
    interrupt = {"mode": "kill", "after_epoch": 1, "delay_s": 60.0, "missed": True, "leftover_processes": None}
    resume = {"segments": 1, "resume_load_s": None, "resume_peak_total_bytes": None, "resume_demonstrated": False}
    report = compute.build_report(toy_meta(), measured, TOY_COUNTS, TOY_MATRIX, p, toy_preflight(),
                                  attempts=attempts, status="ok", interrupt=interrupt, resume=resume)
    md = compute.render_markdown(report)
    assert "Interruption and resume" in md and "embeddings_frozen" in md and "MPS out of memory" in md
    assert "kill came too late" in md and "resume was not demonstrated" in md
    failed = compute.build_report(toy_meta(), None, None, TOY_MATRIX, None, toy_preflight(),
                                  attempts=[dict(a, status="oom") for a in attempts], status="ladder_exhausted")
    md = compute.render_markdown(failed)
    assert "ladder is exhausted" in md and "## Projection" not in md


def test_kill_after_registration_reaps_the_tree(tmp_path):
    import time
    import psutil
    import compute_checkpoint
    pidfile = tmp_path / "grandchild.pid"
    code = ("import subprocess, sys, time\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            "open(sys.argv[1], 'w').write(str(g.pid))\n"
            "time.sleep(300)\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(pidfile)], start_new_session=True)
    t0 = time.monotonic()

    def registered():
        return pidfile.exists() and pidfile.read_text() != "" and time.monotonic() - t0 > 1.0
    out = compute_checkpoint.kill_after_registration(proc, registered, 0, poll_s=0.1)
    grandchild = int(pidfile.read_text())
    assert out["mode"] == "kill" and out["after_epoch"] == 1 and out["missed"] is False
    assert out["leftover_processes"] == 0
    deadline = time.monotonic() + 10
    while psutil.pid_exists(grandchild) and time.monotonic() < deadline:     # reparented zombies are reaped by init
        time.sleep(0.1)
    assert not psutil.pid_exists(proc.pid) and not psutil.pid_exists(grandchild)


def test_kill_after_registration_records_a_missed_kill():
    import compute_checkpoint
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    out = compute_checkpoint.kill_after_registration(proc, lambda: False, 0, poll_s=0.05)
    assert out["missed"] is True and out["leftover_processes"] is None and proc.returncode == 0


# ---- plan 04-06 Task 2: measured BERTurk step and inference times, measured scoring rate ----

def test_berturk_child_on_tiny_cpu(tmp_path):
    out_json = tmp_path / "berturk.json"
    cfg = {"source": "tiny", "device": "cpu", "cpus": 1, "seed": 0, "warmup_steps": 1, "measured_steps": 3,
           "infer_msgs": 8, "out_json": str(out_json)}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "compute_checkpoint.py", "--child", "berturk", str(path)], cwd=REPO_ROOT,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    p = json.loads(out_json.read_text(encoding="utf-8"))
    assert p["status"] == "ok", p
    assert p["param_device"] == "cpu" and p["device"] == "cpu" and p["n_steps"] == 3
    assert p["s_per_step"] > 0 and p["p90_s"] >= p["s_per_step"] and p["s_per_msg_infer"] > 0
    assert p["peak_rss_bytes"] > 0 and p["peak_total_bytes"] > 0 and p["peak_driver_bytes"] is None
    assert p["batch"] == 32 and p["max_len"] == 128 and p["torch_threads"] == 1 and p["n_params"] > 0
    assert "plain PyTorch loop" in p["basis"]


def test_berturk_child_refuses_tampered_weights(tmp_path, monkeypatch):
    import hashlib
    import compute_checkpoint
    bert = tmp_path / "bert"
    bert.mkdir()
    digests = {}
    for name in ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"):
        (bert / name).write_bytes(name.encode("utf-8"))
        digests[name] = hashlib.sha256(name.encode("utf-8")).hexdigest()
    digests["model.safetensors"] = "0" * 64                                   # one planted file is not the pin
    monkeypatch.setitem(compute_checkpoint.pins.BERTURK, "sha256", digests)
    out_json = tmp_path / "berturk.json"
    cfg = {"source": "pinned", "dir": str(bert), "data": str(tmp_path / "none.jsonl"), "data_sha256": "1" * 64,
           "device": "cpu", "cpus": 1, "seed": 0, "warmup_steps": 1, "measured_steps": 1, "infer_msgs": 1,
           "out_json": str(out_json)}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    compute_checkpoint.run_child("berturk", str(path))
    p = json.loads(out_json.read_text(encoding="utf-8"))
    assert p["status"] == "refused" and "model.safetensors" in p["reason"] and "sha256" in p["reason"]


def test_project_uses_measured_berturk_and_scoring():
    measured = dict(toy_measured(), berturk={"status": "ok", "device": "mps", "s_per_step": 0.5,
                                             "s_per_msg_infer": 0.004},
                    scoring={"device": "mps", "dtype": "torch.float32", "amp_enabled": False, "n": 32,
                             "s_per_msg": 0.2})
    p = compute.project(measured, TOY_COUNTS, TOY_MATRIX)
    sel = line(p, "berturk_selection")
    assert sel["seconds"] == pytest.approx(3 * 10 * 46 * 0.5) and sel["basis"] == "measured"
    val = line(p, "laya_val_scoring")
    assert val["seconds"] == pytest.approx(3 * 4 * 208 * 0.2) and val["basis"] == "measured"
    assert "scoring probe" in val["source"]
    final = line(p, "final_scoring")
    assert final["basis"] == "measured"
    assert {part["model"]: part["s_per_unit"] for part in final["parts"]} == {"laya": 0.2, "berturk": 0.004}


def test_markdown_and_assumptions_label_the_measured_probes():
    measured = dict(toy_measured(), berturk={"status": "ok", "device": "mps", "param_device": "mps",
                                             "s_per_step": 0.5, "p90_s": 0.6, "n_steps": 20,
                                             "s_per_msg_infer": 0.004, "peak_total_bytes": 3 * 2**30,
                                             "peak_driver_bytes": 2 * 2**30, "peak_rss_bytes": 2**30,
                                             "basis": "plain PyTorch loop, fp32"},
                    scoring={"device": "mps", "dtype": "torch.float32", "amp_enabled": False, "n": 32,
                             "s_per_msg": 0.2, "basis": "--n 32 val subset"})
    a = compute.assumptions(TOY_MATRIX, measured, TOY_COUNTS)
    assert not any("BERTurk rates are estimates" in x for x in a)
    assert not any("committed CPU zero-shot rate" in x for x in a)
    assert any("HF Trainer" in x for x in a) and any("zero-shot weights" in x for x in a)
    report = compute.build_report(toy_meta(), measured, TOY_COUNTS, TOY_MATRIX,
                                  compute.project(measured, TOY_COUNTS, TOY_MATRIX), toy_preflight(), assumptions=a)
    md = compute.render_markdown(report)
    assert "## BERTurk timing probe" in md and "## Scoring probe" in md and "plain PyTorch loop" in md
    failed = dict(toy_measured(), berturk={"status": "oom", "reason": "MPS backend out of memory"})
    md = compute.render_markdown(compute.build_report(
        toy_meta(), failed, TOY_COUNTS, TOY_MATRIX, compute.project(failed, TOY_COUNTS, TOY_MATRIX), toy_preflight()))
    assert "## BERTurk timing probe" in md and "keep the estimate" in md
    assert any("BERTurk rates are estimates" in x for x in compute.assumptions(TOY_MATRIX, failed, TOY_COUNTS))


def test_probe_storage_is_dropped_when_a_step_after_the_chain_raises(tmp_path, monkeypatch):
    """04-REVIEW WR-06: a failure after a successful probe (here the split count) still deletes the probe's Ray run dir
    and temp checkpoint unless --keep-run."""
    import compute_checkpoint as cc
    base, storage, seen = tmp_path / "base", tmp_path / "storage", {}
    base.mkdir()
    (base / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(cc, "preflight", lambda g: {"passed": True})
    monkeypatch.setattr(cc, "verify_split_file", lambda *a, **k: {"path": "p", "sha256": "0" * 64, "n_messages": 1})
    monkeypatch.setattr(cc.compute, "spike_start", lambda p: {"micro_batch": 8, "max_len": 128, "grad_accum": 4,
                                                                "rung": "none", "fallback": 0})
    monkeypatch.setattr(cc.la, "check_version", lambda: None)
    monkeypatch.setattr(cc, "identity", lambda b: {"model_sha256": "0" * 64})
    monkeypatch.setattr(cc, "disk_preflight", lambda *a: None)
    monkeypatch.setattr(cc, "build_meta", lambda *a: {})

    def launch(rung, ctx):
        os.makedirs(os.path.join(ctx["storage"], "laya-x", "epoch-002"))
        ckpt = os.path.join(ctx["tmp"], "attempt-0", "ckpt")
        os.makedirs(ckpt)
        ctx["rids"].append("laya-x")
        ctx["result"] = {"measured": {"laya": {}}, "ckpt": ckpt, "rid": "laya-x"}
        seen["tmp"] = ctx["tmp"]
        return {"run_id": "laya-x", "status": "ok", "interrupt": {}, "resume": {}}

    def boom(manifest):
        raise RuntimeError("split count failed")
    monkeypatch.setattr(cc, "launch_rung", launch)
    monkeypatch.setattr(cc.compute, "split_counts", boom)
    with pytest.raises(RuntimeError, match="split count failed"):
        cc.main(["--base", str(base), "--storage", str(storage), "--no-berturk", "--no-score-probe",
                 "--min-available-gib", "0", "--out", str(tmp_path / "out")])
    assert not (storage / "laya-x").exists() and not os.path.exists(seen["tmp"])


def test_protocol_complete_pins_the_timing_basis():
    """04-REVIEW WR-07: --cpus (the torch thread count s/step depends on), --epochs and --seed are part of the protocol;
    a run off compute.PROTOCOL is a SMOKE RUN."""
    import argparse
    import compute_checkpoint as cc
    from pilot.data import DEFAULT_MANIFEST
    start = dict(compute.spike_start(SPIKE_REPORT))
    real = dict(device="mps", base="multilingual", data=cc.DEFAULT_DATA, manifest=DEFAULT_MANIFEST, max_messages=None,
                calib_frac=None, interrupt="kill", berturk=True, berturk_source="pinned", **compute.PROTOCOL)
    assert cc.protocol_complete(argparse.Namespace(**real), start) is True
    for k, v in (("cpus", 1), ("cpus", 10), ("epochs", 3), ("seed", 1)):
        assert cc.protocol_complete(argparse.Namespace(**dict(real, **{k: v})), start) is False, (k, v)
    md = compute.render_markdown(full_report(protocol_complete=False))
    assert "--cpus, --epochs or --seed other than cpus 4, epochs 2, seed 0" in md


def test_swap_growth_in_a_measured_epoch_is_bannered():
    """04-REVIEW WR-08: swap growth during a measured epoch (the spike's D-01 memory gate) gets a banner and an
    assumption, not just a table cell; none without growth."""
    quiet = full_report()
    assert "Swap grew during measured epoch" not in compute.render_markdown(quiet)
    assert not any("Swap grew" in x for x in quiet["assumptions"])
    measured = toy_measured()
    measured["epochs"][1]["swap_delta_bytes"] = 423 * 2**20
    assert compute.swap_grew_epochs(measured) == [2]
    p = compute.project(measured, TOY_COUNTS, TOY_MATRIX)
    swapped = compute.build_report(toy_meta(), measured, TOY_COUNTS, TOY_MATRIX, p, toy_preflight(False),
                                   assumptions=compute.assumptions(TOY_MATRIX, measured, TOY_COUNTS))
    md = compute.render_markdown(swapped)
    assert "**Swap grew during measured epoch(s) 2** (largest 423 MiB)" in md
    assert md.index("Swap grew during measured epoch") < md.index("## Measured run")
    assert any(x.startswith("Swap grew during measured epoch(s) 2 (largest 423 MiB)") for x in swapped["assumptions"])


def test_run_overhead_enters_every_laya_run():
    """04-REVIEW IN-03: the fixed cost of one ray_train.py process (the child's wall clock minus the record's terms)
    is added once per Laya run; a report without it says so in its assumptions."""
    rows = toy_rows()
    rows[1].update(segment=1, resumed=True)
    record = {"segments": [_segment(0, 0), _segment(1, 1, 7.0, 1)], "epochs": rows, "load_s": 20.0, "calib_s": 5.0,
              "final_save_s": 3.0}
    # the resumed child trained epoch 2 only: 930 + 10 s, plus 20 load + 7 resume load + 5 calibration + 3 final save
    assert compute.run_overhead(1000.0, record) == pytest.approx(1000.0 - 975.0)
    assert compute.run_overhead(10.0, record) == 0.0
    without = toy_measured()
    assert any("fixed overhead" in x for x in compute.assumptions(TOY_MATRIX, without, TOY_COUNTS))
    with_ = toy_measured()
    with_["laya"]["run_overhead_s"] = 25.0
    assert not any("fixed overhead" in x for x in compute.assumptions(TOY_MATRIX, with_, TOY_COUNTS))
    assert compute.laya_run_s(14854, 4, with_["laya"]) == pytest.approx(compute.laya_run_s(14854, 4, TOY_LAYA) + 25.0)
    p0, p1 = (compute.project(m, TOY_COUNTS, TOY_MATRIX) for m in (without, with_))
    assert line(p1, "laya_selection")["seconds"] == pytest.approx(line(p0, "laya_selection")["seconds"] + 3 * 25.0)
    assert "+ 25.0 s run overhead" in line(p1, "laya_selection")["formula"]
    assert "run overhead" not in line(p0, "laya_selection")["formula"]
    assert "run overhead 25.0 s" in compute.render_markdown(compute.build_report(
        toy_meta(), with_, TOY_COUNTS, TOY_MATRIX, p1, toy_preflight()))


def test_per_epoch_persist_and_scoring_overhead_are_charged_where_they_occur():
    """04-REVIEW IN-02: Ray's persist of each epoch checkpoint is inside save_s (charged per epoch) when the record says
    so, else disclosed; a score.py run's own overhead (the scoring probe's wall clock minus its inference seconds) is
    added once per Laya scoring run, else disclosed. A report without either keeps its numbers."""
    assert compute.laya_training_epochs(TOY_MATRIX) == 96 and compute.laya_scoring_runs(TOY_MATRIX) == 27
    old = toy_measured()
    a = compute.assumptions(TOY_MATRIX, old, TOY_COUNTS)
    assert any("before Ray persists" in x and "96 Laya training epochs" in x for x in a)
    new = toy_measured()
    new["laya"]["save_s_includes_persist"] = True
    assert not any("before Ray persists" in x for x in compute.assumptions(TOY_MATRIX, new, TOY_COUNTS))
    assert "include Ray's persist" in line(compute.project(new, TOY_COUNTS, TOY_MATRIX), "laya_selection")["source"]
    assert "persist" not in line(compute.project(old, TOY_COUNTS, TOY_MATRIX), "laya_selection")["source"]
    sc = {"status": "ok", "device": "mps", "dtype": "torch.float32", "amp_enabled": False, "n": 32, "s_per_msg": 0.2,
          "basis": "--n 32 val subset"}
    m0, m1 = dict(toy_measured(), scoring=dict(sc)), dict(toy_measured(), scoring=dict(sc, run_overhead_s=15.0))
    p0, p1 = (compute.project(m, TOY_COUNTS, TOY_MATRIX) for m in (m0, m1))
    for id_, n_runs in (("laya_val_scoring", 12), ("fewshot_scoring", 6), ("final_scoring", 9)):
        assert line(p1, id_)["seconds"] == pytest.approx(line(p0, id_)["seconds"] + n_runs * 15.0), id_
        assert "%d score.py runs x 15.0 s run overhead" % n_runs in line(p1, id_)["formula"], id_
        assert "run overhead" not in line(p0, id_)["formula"], id_
    assert p1["total_s"] == pytest.approx(p0["total_s"] + 27 * 15.0)
    assert all("run_overhead_s" not in part for part in line(p0, "final_scoring")["parts"])   # old reports unchanged
    assert [part.get("score_runs") for part in line(p1, "final_scoring")["parts"]] == [9, None]
    a0, a1 = (compute.assumptions(TOY_MATRIX, m, TOY_COUNTS) for m in (m0, m1))
    assert any("score.py run's own overhead" in x and "27 Laya score.py runs" in x for x in a0)
    assert not any("score.py run's own overhead" in x for x in a1)
    assert any("BERTurk scoring lines carry no per-run load term" in x for x in a1)
    md = compute.render_markdown(compute.build_report(toy_meta(), m1, TOY_COUNTS, TOY_MATRIX, p1, toy_preflight()))
    assert "run overhead 15.0 s per score.py run" in md


def test_a_stalled_child_is_killed_at_the_probe_timeout():
    """04-REVIEW IN-04: no child wait is unbounded; a child past --probe-timeout-s is killed (its whole tree)."""
    import compute_checkpoint
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
    out = compute_checkpoint.kill_after_registration(proc, lambda: False, 0, poll_s=0.05, timeout_s=0.5)
    assert out["timed_out"] is True and out["missed"] is False and proc.returncode is not None
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
    assert compute_checkpoint.wait_or_kill(proc, 0.5) is None and proc.returncode is not None
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    assert compute_checkpoint.wait_or_kill(proc, 60) == 0
    with pytest.raises(SystemExit):
        compute_checkpoint.main(["--probe-timeout-s", "0"])


def test_the_snapshot_format_was_validated_for_the_locked_ray():
    """04-REVIEW IN-07: epoch_registered reads Ray Train's private snapshot; a lock bump of ray must re-validate it."""
    import importlib.metadata
    assert importlib.metadata.version("ray") == compute.SNAPSHOT_RAY, (
        "ray moved: check %s's layout (checkpoint_results[].checkpoint_dir_name) in the new ray, then bump "
        "compute.SNAPSHOT_RAY" % compute.SNAPSHOT)
