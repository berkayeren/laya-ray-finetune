"""The committed Phase 9 few-shot records (FEW-02 and later): bindings, byte rebuilds and the rules they apply.

This file grows through Phase 9: 09-04 adds the D-08 compute projection and decision record, 09-11 to 09-13 add the
runs manifest, the few-shot result record and the README block.

09-04 (D-08, ROADMAP Phase 9 criterion 1):
- results/fewshot/projection.json (fewshot-projection/1) is tracked, bound to the frozen protocol's sha256 and to the
  sha256 of results/compute/compute_checkpoint.json, mode primary with projection_fired false and no ladder failure,
  a total between 8.0 and 8.6 h whose lines sum to it, and pilot.fewshot.read_mode reads primary from it; it holds no
  uid string and no local path.
- the projection and the decision recompute from the committed compute rates and the record's own counts
  (pilot.compute.fewshot_projection, fewshot_decision), offline; the Laya lines use calib_s None (LAYA-05) and the
  notes say so.
- data-marked: `fewshot.py --projection` rebuilds the record byte for byte from the committed data.
Offline except the `data` tests.
"""
import json
import os
import subprocess

import pytest

import fewshot as runner
from pilot import compute
from pilot import fewshot as fs
from pilot import protocol as pr
from pilot.data import REPO_ROOT, sha256_bytes

pytestmark = pytest.mark.unit

PROJECTION = os.path.join(REPO_ROOT, "results", "fewshot", "projection.json")
COMPUTE = os.path.join(REPO_ROOT, "results", "compute", "compute_checkpoint.json")
LINE_IDS = ["fewshot_laya_bases", "fewshot_berturk_bases", "fewshot_laya_continued", "fewshot_berturk_continued",
            "fewshot_scoring"]


def committed():
    with open(PROJECTION, "rb") as f:
        raw = f.read()
    return raw, json.loads(raw.decode("utf-8"))


def frozen():
    return pr.require_frozen()


def measured():
    with open(COMPUTE, encoding="utf-8") as f:
        return json.load(f)["measured"]


def n_scored(proto):
    return {s: proto["splits"][s]["n_messages"] for s in ("val", "test")}


def test_projection_record_is_committed_and_bound():
    out = subprocess.run(["git", "ls-files", "--error-unmatch", "results/fewshot/projection.json"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, "results/fewshot/projection.json is not tracked: %s" % out.stderr
    raw, rec = committed()
    assert rec["schema"] == fs.DECISION_SCHEMA == "fewshot-projection/1"
    assert rec["protocol_sha256"] == pr.PROTOCOL_SHA256
    with open(COMPUTE, "rb") as f:
        assert rec["compute"] == {"path": "results/compute/compute_checkpoint.json", "sha256": sha256_bytes(f.read())}
    assert rec["counts"]["protocol_sha256"] == pr.PROTOCOL_SHA256
    assert rec["decision"]["mode"] == "primary" and rec["decision"]["disclosure"] is None
    assert rec["trigger"]["projection_fired"] is False and rec["trigger"]["ladder_failure"] is False
    assert rec["trigger"]["ladder_failure_text"] is None
    p = rec["projection"]
    assert [line["id"] for line in p["lines"]] == LINE_IDS
    assert 8.0 <= p["total_h"] <= 8.6, p["total_h"]
    assert sum(line["seconds"] for line in p["lines"]) == pytest.approx(p["total_s"], rel=1e-12)
    assert p["fallback_total_s"] < p["total_s"]
    assert fs.read_mode(pr.PROTOCOL_SHA256) == "primary"
    text = raw.decode("utf-8")
    assert "bank_" not in text and "hotel_" not in text
    assert os.path.expanduser("~") not in text and REPO_ROOT not in text
    assert raw.endswith(b"\n") and raw == (json.dumps(rec, ensure_ascii=False, indent=2, sort_keys=True)
                                          + "\n").encode("utf-8")


def test_projection_recomputes_from_the_committed_rates_and_counts():
    proto, _ = frozen()
    _, rec = committed()
    block = proto["fewshot"]
    p = compute.fewshot_projection(measured(), rec["counts"], block, n_scored(proto))
    assert json.loads(json.dumps(p)) == rec["projection"]
    d = compute.fewshot_decision(p, block)
    assert {"trigger": rec["trigger"], "decision": rec["decision"]} == json.loads(json.dumps(d))


def test_projection_lines_use_the_measured_terms_without_calibration():
    proto, _ = frozen()
    _, rec = committed()
    m = measured()
    assert m["laya"]["calib_s"] is not None                   # the Phase 4 probe calibrated; Phase 9 runs do not
    p = compute.fewshot_projection(m, rec["counts"], proto["fewshot"], n_scored(proto))
    lines = {line["id"]: line for line in p["lines"]}
    for line in p["lines"]:
        assert line["phase"] == 9 and line["basis"] == "measured" and line["source"] and line["formula"]
    rules = proto["fewshot"]["rules"]
    laya_terms = dict(m["laya"], calib_s=None)
    base = lines["fewshot_laya_bases"]
    assert base["runs"] == 3 and base["epochs"] == rules["base_runs"]["laya"]["stop_after_epoch"] == 3
    assert base["units_per_epoch"] == rec["counts"]["base"]["laya_steps_per_epoch"] == 456
    assert base["seconds"] == pytest.approx(3 * compute.laya_run_s(14580, 3, laya_terms))
    assert "calibration" not in base["formula"] and "calibration" not in base["source"]
    bert = lines["fewshot_berturk_bases"]
    assert bert["runs"] == 3 and bert["epochs"] == 29 and bert["units_per_epoch"] == 46
    assert bert["seconds"] == pytest.approx(3 * 29 * 46 * m["berturk"]["s_per_step"])
    cont = lines["fewshot_laya_continued"]
    assert [(q["draw"], q["k"]) for q in cont["parts"]] == [(c["draw"], c["k"]) for c in rec["counts"]["continued"]]
    assert [q["units_per_epoch"] for q in cont["parts"]] == [c["laya_steps_per_epoch"]
                                                             for c in rec["counts"]["continued"]]
    assert cont["runs"] == 6 and cont["epochs"] == rules["continued_training"]["epochs"] == 10
    bcont = lines["fewshot_berturk_continued"]
    assert [q["units_per_epoch"] for q in bcont["parts"]] == [c["berturk_steps_per_epoch"]
                                                              for c in rec["counts"]["continued"]]
    sc = {(q["model"], q["split"]): q for q in lines["fewshot_scoring"]["parts"]}
    assert {key: (q["runs"], q["units_per_epoch"]) for key, q in sc.items()} == {
        ("laya", "val"): (3, 208), ("laya", "test"): (9, 413), ("berturk", "val"): (3, 208),
        ("berturk", "test"): (9, 413)}
    assert any("LAYA-05" in n and "calib" in n for n in rec["notes"])
    # the fallback: one base per model, the continued runs and their scoring unchanged
    one = 2 * compute.laya_run_s(14580, 3, laya_terms) + 2 * 29 * 46 * m["berturk"]["s_per_step"] + \
        2 * (208 + 413) * (m["scoring"]["s_per_msg"] + m["berturk"]["s_per_msg_infer"])
    assert p["fallback_total_s"] == pytest.approx(p["total_s"] - one)


@pytest.mark.data
def test_projection_rebuilds(tmp_path):
    raw, _ = committed()
    out = str(tmp_path / "projection.json")
    runner.main(["--projection", "--out", out])
    with open(out, "rb") as f:
        assert f.read() == raw
