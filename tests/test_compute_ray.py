"""compute_checkpoint.py end to end on CPU (LAYA-07): preflight -> spike start config -> a fresh ray_train.py child on
the tiny offline checkpoint -> run record -> projection -> compute_checkpoint.json/.md, then the probe is cleaned up.

Offline (tests/tiny_laya.py + the synthetic splits of tests/conftest.py); the committed spike report is the default
--spike-report. The run is off the protocol (cpu, a local base, a synthetic split, test hooks), so the report must say
SMOKE RUN. Runs in `make test` under the ray marker.
"""
import json
import os
import subprocess
import sys

import pytest

import pilot.data
from conftest import make_raw
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


@pytest.mark.timeout(TIMEOUT)
def test_compute_checkpoint_on_cpu(tmp_path):
    root, raw_sha256 = make_raw(str(tmp_path / "raw"))
    data = str(tmp_path / "data")
    pilot.data.prepare(root, data, raw_sha256)
    base = tiny_checkpoint(tmp_path / "base")
    out, storage = tmp_path / "report", tmp_path / "ray"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "compute_checkpoint.py", "--device", "cpu", "--base", base,
                        "--data", os.path.join(data, "splits", "selection_train.jsonl"),
                        "--manifest", os.path.join(data, "splits.json"), "--epochs", "1", "--cpus", "2",
                        "--min-available-gib", "0", "--max-messages", "12", "--calib-frac", "0.2",
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
    assert [row["epoch"] for row in report["measured"]["epochs"]] == [1]
    lines = {l["id"]: l for l in report["projection"]["lines"]}
    assert lines["laya_selection"]["seconds"] > 0 and lines["laya_final"]["seconds"] > 0
    with open(out / "compute_checkpoint.md", encoding="utf-8") as f:
        md = f.read()
    assert md.startswith("# Compute checkpoint") and "SMOKE RUN" in md
    home = os.path.expanduser("~").rstrip("/")
    assert home not in text and home not in md
    # the probe's Ray run dir and checkpoint are gone (no --keep-run)
    run_id = report["meta"]["protocol"]["run_id"]
    assert run_id.startswith("laya-selection_train-neg8-s0-")
    assert not (storage / run_id).exists(), "the probe's Ray run dir was not deleted"
