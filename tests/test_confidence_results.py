"""The committed Phase 7 confidence records (EVAL-03) stay consistent with the code, the Makefile and the postfit records.

- every tracked results/confidence/*.json is a val record (schema confidence/1, split val) that carries in_sample
  true and report.IN_SAMPLE_CAVEAT, both temperature variants with a calibration block, and, for a dump of a trained
  model that is not a final seed, the postfit_candidate binding (its T and θ come from the postfit record, never from
  the sidecar's shipped T = 1)
- the Makefile's LAYA_CHOSEN is the chosen dump of results/postfit/laya.json and has a committed record
- data marker: every committed record equals report.confidence_report recomputed from the committed dump, the
  committed manifest and the record it is bound to (needs `make data`, because the gold labels come from the verified
  splits), and a record bound to a postfit candidate reproduces that candidate's nll_fitted (headline) and nll_raw
  (raw) within 1e-8
"""
import glob
import json
import os
import re
import subprocess

import pytest

import report
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT

pytestmark = pytest.mark.unit

DUMPS = os.path.join(REPO_ROOT, "results", "dumps")
LAYA_RECORD = os.path.join(REPO_ROOT, "results", "postfit", "laya.json")
NLL_TOL = 1e-8


def tracked(*pathspec):
    """Tracked paths via git ls-files (never a directory walk, which would see untracked scratch records)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *pathspec], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    return [f for f in out.stdout.split("\0") if f]


def committed():
    """[(repo-relative path, parsed JSON)] of the tracked confidence records, in sorted path order."""
    out = []
    for rel in sorted(f for f in tracked("results/confidence/") if f.endswith(".json")):
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            out.append((rel, json.load(f)))
    return out


def _postfit_path(rec):
    pf = rec["binding"]["postfit"]
    return None if pf is None else os.path.join(REPO_ROOT, pf["path"])


def test_committed_records_are_val_and_in_sample():
    records = committed()
    assert records, "no tracked confidence records under results/confidence/"
    for rel, r in records:
        assert r["schema"] == report.CONFIDENCE_SCHEMA, rel
        assert r["split"]["name"] == "val" and r["in_sample"] is True, rel
        assert r["caveat"] == report.IN_SAMPLE_CAVEAT, rel
        assert rel == "results/confidence/%s-val.json" % r["run_id"], rel
        assert set(r["variants"]) == {"headline", "raw"}, rel
        for name, v in r["variants"].items():
            cal = v["calibration"]
            assert {"brier", "nll", "classwise_ece", "per_tag", "reliability"} <= set(cal), (rel, name)
            assert sum(b["count"] for b in cal["reliability"]["pooled"]) == r["n_messages"] * 48, (rel, name)
        assert r["variants"]["raw"]["temperature"] == 1.0, rel
        tr = r["training"]
        if tr is not None and r["binding"]["kind"] != "final_seed":
            assert r["binding"]["kind"] == "postfit_candidate", rel
            assert r["binding"]["postfit"] is not None, rel


def test_makefile_confidence_ids_match_postfit_records():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^LAYA_CHOSEN := (\S+)$", text, re.M)
    assert m, "Makefile has no LAYA_CHOSEN := line"
    with open(LAYA_RECORD, encoding="utf-8") as f:
        record = json.load(f)
    assert m.group(1) == record["chosen"]["dump_run_id"], "LAYA_CHOSEN is not results/postfit/laya.json's chosen dump"
    assert "results/confidence/%s-val.json" % m.group(1) in [rel for rel, _ in committed()]
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "confidence-reports" in phony.group(1).split(), "confidence-reports is not in .PHONY"
    recipe = re.search(r"^confidence-reports:\n((?:\t.*\n)+)", text, re.M)
    assert recipe, "Makefile has no confidence-reports target"
    assert ("$(PYTHON) report.py --confidence results/dumps/$(LAYA_CHOSEN) --split val --postfit "
            "results/postfit/laya.json --out $(CONFIDENCE)/$(LAYA_CHOSEN)-val.json") in recipe.group(1)


def _need_data():
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")


@pytest.mark.data
def test_confidence_records_reproduce_from_dumps():
    _need_data()
    records = committed()
    assert records
    for rel, r in records:
        got = report.confidence_report(os.path.join(DUMPS, r["run_id"]), "val", DEFAULT_MANIFEST, _postfit_path(r))
        assert json.loads(json.dumps(got, ensure_ascii=False)) == r, "%s does not reproduce" % rel


@pytest.mark.data
def test_nll_matches_postfit_record():
    _need_data()
    bound = [(rel, r) for rel, r in committed() if r["binding"]["kind"] == "postfit_candidate"]
    assert bound, "no committed record is bound to a postfit candidate"
    for rel, r in bound:
        with open(_postfit_path(r), encoding="utf-8") as f:
            record = json.load(f)
        cands = [c for c in record["candidates"] if c["dump_sha256"] == r["dump_sha256"]]
        assert len(cands) == 1, rel
        c = cands[0]
        assert abs(r["variants"]["headline"]["calibration"]["nll"] - c["nll_fitted"]) <= NLL_TOL, rel
        assert abs(r["variants"]["raw"]["calibration"]["nll"] - c["nll_raw"]) <= NLL_TOL, rel
