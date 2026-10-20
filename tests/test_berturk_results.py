"""The committed Phase 6 BERTurk postfit record and its BASE-03 reference check stay consistent and reproducible
(BASE-02, BASE-03, T-06-30, T-06-31).

- results/postfit/berturk.json is a postfit/1 record of system kind berturk under BERTurk's protocol
  (pilot.postfit.protocol("berturk"): one LR 2e-05, head LR tied, epochs 1..10, selection seed 0, final seeds 0, 1, 2),
  with exactly 10 candidates covering epochs 1..10 once each; every candidate's dump dir is tracked with a val dump
  that verifies (pilot.dumps.load_dump) to the candidate's dump_sha256 and training run id, and with no test dump
- the chosen candidate is what pilot.postfit.select returns on the recorded candidates (the pre-declared rule, never a
  hand pick), its T lies in [T_MIN, T_MAX] and stop_after_epoch is None exactly at the last epoch of the schedule
- the Makefile's BERTURK_SELECTION lists the record's candidate dump run ids in epoch order (make berturk-postfit)
- results/postfit/berturk-reference.json equals pilot.berturk.reference_check of the committed record (its path, the
  sha256 of its bytes and the committed file's own resolution), so it is never edited by hand; status "investigate"
  exactly when the exact fraction is below 707/1000, and an investigate status carries a non-empty resolution before
  any final-seed manifest (results/postfit/berturk-final.json) is tracked
- data marker: postfit.fit over the committed candidate dump dirs, serialized as write_record does, equals the
  committed bytes (needs `make data`, because the gold labels come from the verified splits)
"""
import glob
import json
import os
import re
import subprocess
from fractions import Fraction

import pytest

from pilot import berturk, dumps, postfit
from pilot.data import DEFAULT_MANIFEST, sha256_file

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPS = os.path.join(REPO_ROOT, "results", "dumps")
RECORD_REL = "results/postfit/berturk.json"
REFERENCE_REL = "results/postfit/berturk-reference.json"
FINAL_REL = "results/postfit/berturk-final.json"
RECORD = os.path.join(REPO_ROOT, RECORD_REL)
REFERENCE = os.path.join(REPO_ROOT, REFERENCE_REL)


def tracked(*pathspec):
    """Tracked paths via git ls-files (never a directory walk, which would see untracked scratch files)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *pathspec], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    return [f for f in out.stdout.split("\0") if f]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def as_json(v):
    return json.loads(json.dumps(v, ensure_ascii=False))


def test_committed_berturk_record_is_complete_and_consistent():
    assert tracked(RECORD_REL) == [RECORD_REL], "%s is not tracked; run `make berturk-postfit` and commit it" % RECORD_REL
    r = load(RECORD)
    assert r["schema"] == postfit.SCHEMA == "postfit/1"
    assert r["system_kind"] == "berturk"
    proto = postfit.protocol("berturk")
    for key in ("lr_grid", "head_lr", "epochs", "selection_seed", "final_seeds"):
        assert as_json(r["protocol"][key]) == as_json(proto[key]), key
    assert r["protocol"]["selection_split"] == "selection_train" and r["protocol"]["final_split"] == "final_train"
    assert r["rules"]["epoch_transfer"] == postfit.EPOCH_TRANSFER_TEMPLATE % 10
    cands = r["candidates"]
    assert len(cands) == 10 and sorted(c["epoch"] for c in cands) == list(range(1, 11))
    assert all(c["encoder_lr"] == 2e-05 and c["system_kind"] == "berturk" for c in cands)
    files = set(tracked("results/dumps/"))
    for c in cands:
        rid = c["dump_run_id"]
        for name in ("val.npz", "val.json"):
            assert "results/dumps/%s/%s" % (rid, name) in files, "%s/%s is not tracked" % (rid, name)
        for name in ("test.npz", "test.json"):
            assert "results/dumps/%s/%s" % (rid, name) not in files, "%s holds a tracked test dump" % rid
        logits, uids, side, entry = dumps.load_dump(os.path.join(DUMPS, rid), "val", DEFAULT_MANIFEST)
        assert side["run_id"] == rid and side["dump_sha256"] == c["dump_sha256"], rid
        assert side["training"]["run_id"] == c["train_run_id"] and side["n_subset"] is None, rid
        assert entry["role"] == "val" and logits.shape == (len(uids), 48), rid
    chosen = postfit.select(cands, **{k: v for k, v in proto.items() if k != "final_seeds"})
    c = r["chosen"]
    assert c["dump_run_id"] == chosen["dump_run_id"] and c["dump_sha256"] == chosen["dump_sha256"]
    assert c["epoch"] == chosen["epoch"] and as_json(c["theta"]) == as_json(chosen["theta"])
    assert postfit.T_MIN <= c["temperature"] <= postfit.T_MAX
    assert c["temperature"] == chosen["temperature"]
    assert (c["stop_after_epoch"] is None) == (c["epoch"] == 10)
    assert c["stop_after_epoch"] in (None, c["epoch"])
    assert r["final"]["split"] == "final_train" and r["final"]["seeds"] == [0, 1, 2]
    assert r["final"]["stop_after_epoch"] == c["stop_after_epoch"] and r["final"]["epochs"] == 10


def test_makefile_lists_the_berturk_selection_dumps():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^BERTURK_SELECTION := (.+)$", text, re.M)
    assert m, "Makefile has no BERTURK_SELECTION := line"
    assert m.group(1).split() == [c["dump_run_id"] for c in load(RECORD)["candidates"]]
    assert re.search(r"^berturk-postfit:$", text, re.M) and "berturk-postfit" in re.search(r"^\.PHONY:.*$", text,
                                                                                           re.M).group(0)


def test_committed_reference_check_equals_its_recompute():
    assert tracked(REFERENCE_REL) == [REFERENCE_REL], "%s is not tracked" % REFERENCE_REL
    record, sha = postfit.read_record(RECORD)
    assert sha == sha256_file(RECORD)
    committed = load(REFERENCE)
    assert committed["schema"] == berturk.REFERENCE_SCHEMA == "berturk-reference/1"
    assert committed["postfit"] == {"path": RECORD_REL, "sha256": sha}, "the reference check is of another record"
    got = berturk.reference_check(record, RECORD, sha, committed.get("resolution"))
    assert as_json(got) == committed, "berturk-reference.json differs from its recompute; never edit it by hand"
    num, den = committed["value"]["fraction"]
    below = Fraction(num, den) < Fraction(707, 1000)
    assert committed["status"] == ("investigate" if below else "ok") and committed["below_threshold"] is below
    if committed["status"] == "investigate":
        res = committed.get("resolution")
        assert (isinstance(res, str) and res.strip()) or not tracked(FINAL_REL), \
            "BASE-03 reads investigate with no resolution, yet a BERTurk final-seed manifest is tracked"


@pytest.mark.data
def test_berturk_record_reproduces_from_the_committed_dumps():
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")
    record = load(RECORD)
    got = postfit.fit([os.path.join(DUMPS, c["dump_run_id"]) for c in record["candidates"]], DEFAULT_MANIFEST)
    with open(RECORD, "rb") as f:
        assert postfit.record_bytes(got) == f.read(), "%s does not rebuild from the committed dumps" % RECORD_REL
