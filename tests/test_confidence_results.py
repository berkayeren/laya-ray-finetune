"""The committed Phase 7 confidence records (EVAL-03) stay consistent with the code, the Makefile and the postfit records.

- every tracked results/confidence/*.json is a val record (schema confidence/1, split val) that carries in_sample
  true and report.IN_SAMPLE_CAVEAT, both temperature variants with a calibration block, and, for a dump of a trained
  model that is not a final seed, the postfit_candidate binding (its T and θ come from the postfit record, never from
  the sidecar's shipped T = 1)
- the Makefile's LAYA_CHOSEN and BERTURK_CHOSEN are the chosen dumps of results/postfit/laya.json and berturk.json,
  and the tracked records are exactly those two and zero-shot Laya (LAYA_ZS, a reference row bound by report.py's
  val rule, val_selected): the two chosen ones bound to their own system kind's postfit record (07-RESEARCH Open
  Question 1, A10)
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
BERTURK_RECORD = os.path.join(REPO_ROOT, "results", "postfit", "berturk.json")
POSTFIT_BY_KIND = {"laya": "results/postfit/laya.json", "berturk": "results/postfit/berturk.json"}
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


def makefile():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        return f.read()


def makefile_var(text, name):
    m = re.search(r"^%s := (\S+)$" % name, text, re.M)
    assert m, "Makefile has no %s := line" % name
    return m.group(1)


def test_makefile_confidence_ids_match_postfit_records():
    text = makefile()
    for name, path in (("LAYA_CHOSEN", LAYA_RECORD), ("BERTURK_CHOSEN", BERTURK_RECORD)):
        run_id = makefile_var(text, name)
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        assert run_id == record["chosen"]["dump_run_id"], "%s is not %s's chosen dump" % (name, path)
        assert "results/confidence/%s-val.json" % run_id in [rel for rel, _ in committed()], name
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "confidence-reports" in phony.group(1).split(), "confidence-reports is not in .PHONY"
    recipe = re.search(r"^confidence-reports:\n((?:\t.*\n)+)", text, re.M)
    assert recipe, "Makefile has no confidence-reports target"
    for line in ("$(PYTHON) report.py --confidence results/dumps/$(LAYA_CHOSEN) --split val --postfit "
                 "results/postfit/laya.json --out $(CONFIDENCE)/$(LAYA_CHOSEN)-val.json",
                 "$(PYTHON) report.py --confidence results/dumps/$(BERTURK_CHOSEN) --split val --postfit "
                 "results/postfit/berturk.json --out $(CONFIDENCE)/$(BERTURK_CHOSEN)-val.json",
                 "$(PYTHON) report.py --confidence results/dumps/$(LAYA_ZS) --split val "
                 "--out $(CONFIDENCE)/$(LAYA_ZS)-val.json"):
        assert line in recipe.group(1), line


def test_committed_records_are_the_three_systems():
    text = makefile()
    chosen = {makefile_var(text, "LAYA_CHOSEN"): "laya", makefile_var(text, "BERTURK_CHOSEN"): "berturk"}
    zs = makefile_var(text, "LAYA_ZS")
    records = {r["run_id"]: r for _, r in committed()}
    assert set(records) == set(chosen) | {zs}, "tracked confidence records: %s" % sorted(records)
    for run_id, kind in chosen.items():
        r = records[run_id]
        assert r["system"]["kind"] == kind, run_id
        assert r["binding"]["kind"] == "postfit_candidate", run_id
        assert r["binding"]["postfit"]["path"] == POSTFIT_BY_KIND[kind], run_id
        assert r["binding"]["postfit"]["system_kind"] == kind, run_id
        assert r["training"] is not None and r["training"]["split"] == "selection_train", run_id
    r = records[zs]
    assert r["system"]["kind"] == "laya" and r["training"] is None, zs
    assert r["binding"]["kind"] == "val_selected" and r["binding"]["postfit"] is None, zs


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
    kinds = sorted(r["system"]["kind"] for _, r in bound)
    assert kinds == ["berturk", "laya"], "postfit-bound records: %s" % [r["run_id"] for _, r in bound]
    for rel, r in bound:
        with open(_postfit_path(r), encoding="utf-8") as f:
            record = json.load(f)
        cands = [c for c in record["candidates"] if c["dump_sha256"] == r["dump_sha256"]]
        assert len(cands) == 1, rel
        c = cands[0]
        assert abs(r["variants"]["headline"]["calibration"]["nll"] - c["nll_fitted"]) <= NLL_TOL, r["run_id"]
        assert abs(r["variants"]["raw"]["calibration"]["nll"] - c["nll_raw"]) <= NLL_TOL, r["run_id"]
        assert r["variants"]["headline"]["temperature"] == c["temperature"], r["run_id"]
