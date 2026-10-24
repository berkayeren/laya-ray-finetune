"""compute_checkpoint.py end to end on CPU: preflight -> spike start config -> a fresh ray_train.py
child on the tiny offline checkpoint, interrupted after epoch 1 (--interrupt fail, ray_train.py's --fail-after-epoch 1)
and resumed by rerunning the same command -> run record -> projection -> compute_checkpoint.json/.md, then the probe is
cleaned up.

Offline (tests/tiny_laya.py + the synthetic splits of tests/conftest.py); the committed spike report is the default
--spike-report. The run is off the protocol (cpu, a local base, a synthetic split, test hooks), so the report must say
SMOKE RUN. Runs in `make test` under the ray marker.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pilot.data
from conftest import make_raw
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


@pytest.mark.timeout(TIMEOUT)
def test_compute_checkpoint_on_cpu(tmp_path: Path) -> None:
    root, raw_sha256 = make_raw(str(tmp_path / "raw"))
    data = str(tmp_path / "data")
    pilot.data.prepare(root, data, raw_sha256)
    base = tiny_checkpoint(tmp_path / "base")
    out, storage = tmp_path / "report", tmp_path / "ray"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "compute_checkpoint.py", "--device", "cpu", "--base", base,
                        "--data", os.path.join(data, "splits", "selection_train.jsonl"),
                        "--manifest", os.path.join(data, "splits.json"), "--epochs", "2", "--cpus", "2",
                        "--interrupt", "fail", "--berturk-source", "tiny",
                        "--min-available-gib", "0", "--max-messages", "12",
                        "--out", str(out), "--storage", str(storage)],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, "compute_checkpoint.py failed (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-3000:],
                                                                                  r.stderr[-3000:])
    with open(out / "compute_checkpoint.json", encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert report["schema"] == "compute_checkpoint/1"
    assert report["meta"]["protocol"]["protocol_complete"] is False
    laya = report["measured"]["laya"]
    assert laya["s_per_step"] > 0 and laya["device"] == "cpu" and laya["rung"] == "none"
    assert laya["micro_batch"] == 8 and laya["max_len"] == 128 and laya["grad_accum"] == 4 and laya["neg"] == 8
    assert laya["calib_s"] is None                          # training no longer calibrates
    assert report["projection"]["total_h"] > 0
    assert [row["epoch"] for row in report["measured"]["epochs"]] == [1, 2]
    assert [row["resumed"] for row in report["measured"]["epochs"]] == [False, True]
    assert report["status"] == "ok"
    assert [(x["rung"], x["status"]) for x in report["attempts"]] == [("none", "ok")]
    assert report["interrupt"]["mode"] == "fail" and report["interrupt"]["after_epoch"] == 1
    assert report["interrupt"]["missed"] is False and report["interrupt"]["post_interrupt_available_gib"] > 0
    assert report["resume"]["resume_demonstrated"] is True and report["resume"]["segments"] == 2
    assert report["resume"]["resume_load_s"] > 0 and report["resume"]["resume_peak_total_bytes"] > 0
    lines = {l["id"]: l for l in report["projection"]["lines"]}
    assert lines["laya_selection"]["seconds"] > 0 and lines["laya_final"]["seconds"] > 0
    # the resumed child's fixed cost outside its record (process and Ray start, ...), once per run
    assert report["measured"]["laya"]["run_overhead_s"] > 0 and "run overhead" in lines["laya_selection"]["formula"]
    assert not any("fixed overhead" in x for x in report["assumptions"])
    assert sorted(lines) == sorted(["laya_selection", "laya_val_scoring", "laya_final", "berturk_selection",
                                    "berturk_final", "fewshot_base", "fewshot_incremental", "fewshot_scoring",
                                    "final_scoring"])
    # the BERTurk timing child (tiny model, fresh process) and the scoring probe (--n 32 of val, in-process)
    berturk, scoring = report["measured"]["berturk"], report["measured"]["scoring"]
    assert berturk["status"] == "ok" and berturk["param_device"] == "cpu" and berturk["s_per_step"] > 0
    assert scoring["device"] == "cpu" and scoring["dtype"] == "torch.float32" and scoring["amp_enabled"] is False
    assert scoring["n"] == min(32, report["counts"]["val"]["n_messages"]) and scoring["s_per_msg"] > 0
    for id_ in ("berturk_selection", "berturk_final", "laya_val_scoring", "final_scoring"):
        assert lines[id_]["basis"] == "measured", id_
    assert lines["berturk_selection"]["s_per_unit"] == berturk["s_per_step"]
    assert lines["laya_val_scoring"]["s_per_unit"] == scoring["s_per_msg"]
    assert "scoring probe" in lines["laya_val_scoring"]["source"]
    # per-epoch persist inside save_s, a per-score.py-run overhead from the scoring probe
    assert report["measured"]["laya"]["save_s_includes_persist"] is True
    assert scoring["wall_s"] > scoring["run_overhead_s"] > 0
    assert "score.py runs x" in lines["laya_val_scoring"]["formula"]
    assert not any("before Ray persists" in x or "score.py run's own overhead" in x for x in report["assumptions"])
    assert sorted(report["counts"]) == sorted(["selection_train", "val", "final_train", "test", "hotels_probe"])
    assert len(report["what_if"]) == 10 and report["disk"]["model_bytes"] > 0 and report["assumptions"]
    with open(out / "compute_checkpoint.md", encoding="utf-8") as f:
        md = f.read()
    assert md.startswith("# Compute checkpoint") and "SMOKE RUN" in md
    assert "What-if" in md and "Disk" in md and "Assumptions to confirm" in md
    assert "Interruption and resume" in md and "Resume demonstrated" in md
    assert "BERTurk timing probe" in md and "Scoring probe" in md
    home = os.path.expanduser("~").rstrip("/")
    assert home not in text and home not in md
    # the probe's Ray run dir and checkpoint are gone (no --keep-run)
    run_id = report["meta"]["protocol"]["run_id"]
    assert run_id.startswith("laya-selection_train-neg8-s0-")
    assert run_id == report["attempts"][0]["run_id"]
    assert not (storage / run_id).exists(), "the probe's Ray run dir was not deleted"
    assert not storage.exists() or os.listdir(storage) == [], "the probe left Ray storage behind"
