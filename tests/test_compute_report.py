"""The committed compute checkpoint (plan 04-07, LAYA-02/03/06/07) is the real M4 run and stays consistent with the code.

- results/compute/compute_checkpoint.json is a complete-protocol run (schema compute_checkpoint/1, status ok): one
  asserted MPS worker on the spike's start config (or a labelled fallback rung), gradient checkpointing on, AMP off,
  the spike's MPS env, at least 2 epoch rows with every peak below the machine's memory, and a demonstrated resume
- its projection and what-if table equal pilot.compute recomputed from its own measured block, counts and matrix, so
  the committed totals cannot be edited by hand or drift from the projection code
- results/compute/compute_checkpoint.md equals compute.render_markdown(report) byte for byte
- the JSON passes the repository's text and path hygiene check (no message text, no local home path)
"""
import json
import os

import pytest

from pilot import compute, runs
from pilot.data import REPO_ROOT

pytestmark = pytest.mark.unit

COMPUTE = os.path.join(REPO_ROOT, "results", "compute")
REPORT_JSON = os.path.join(COMPUTE, compute.JSON_NAME)
REPORT_MD = os.path.join(COMPUTE, compute.MD_NAME)
SPIKE_REPORT = os.path.join(REPO_ROOT, "results", "spike", "mps_spike.json")
MPS_ENV = ("PYTORCH_ENABLE_MPS_FALLBACK=0", "PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0", "PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0")


@pytest.fixture(scope="module")
def report():
    with open(REPORT_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_is_the_complete_protocol_run(report):
    assert report["schema"] == compute.SCHEMA
    assert report["meta"]["protocol"]["protocol_complete"] is True
    assert report["status"] == "ok"
    assert report["meta"]["protocol"]["device"] == "mps"


def test_laya_run_config(report):
    L = report["measured"]["laya"]
    assert L["device"] == "mps" and L["param_device"] == "mps", (L["device"], L["param_device"])
    assert L["rung"] == report["attempts"][-1]["rung"], (L["rung"], report["attempts"][-1]["rung"])
    if L["rung"] != "none":
        assert L["fallback"] is True, "a fallback rung must be labelled as one"
    else:
        with open(SPIKE_REPORT, encoding="utf-8") as f:
            chosen = json.load(f)["verdict"]["chosen"]
        assert {k: L[k] for k in ("micro_batch", "max_len", "grad_accum")} == \
            {k: chosen[k] for k in ("micro_batch", "max_len", "grad_accum")}
    assert L["grad_checkpointing"] is True and L["amp"] is False
    assert set(MPS_ENV) <= set(L["env"]), sorted(set(MPS_ENV) - set(L["env"]))


def test_epoch_rows(report):
    rows = report["measured"]["epochs"]
    total = report["meta"]["platform"]["total_memory_bytes"]
    assert len(rows) >= 2
    assert [r["epoch"] for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert set(r) == set(runs.EPOCH_FIELDS), sorted(set(r) ^ set(runs.EPOCH_FIELDS))
        assert 0 < r["peak_total_bytes"] < total, (r["epoch"], r["peak_total_bytes"], total)


def test_resume_demonstrated(report):
    assert report["resume"]["resume_demonstrated"] is True, report["resume"]
    assert report["interrupt"]["mode"] == "kill" and report["interrupt"]["missed"] is False, report["interrupt"]


def test_projection_recomputes(report):
    m, c, x = report["measured"], report["counts"], report["matrix"]
    assert report["projection"] == json.loads(json.dumps(compute.project(m, c, x)))
    assert report["what_if"] == json.loads(json.dumps(compute.what_if(m, c, x)))
    assert report["projection"]["total_h"] > 0


def test_markdown_renders_from_json(report):
    with open(REPORT_MD, "rb") as f:
        assert f.read() == compute.render_markdown(report).encode("utf-8")


def test_json_hygiene():
    from test_repo_hygiene import home_dir, local_split_texts, sidecar_problems
    assert sidecar_problems(REPORT_JSON, home_dir(), local_split_texts()) == []
