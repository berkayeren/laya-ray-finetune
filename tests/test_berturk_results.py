"""Committed Phase 6 BERTurk postfit records and their BASE-03 reference checks stay consistent and reproducible
(BASE-02, BASE-03, T-06-30, T-06-31, T-06-38, T-06-40).

The helpers take explicit paths and a protocol, so the current run and the superseded runs use the same checks:

- _check_record: a postfit/1 record of system kind berturk under the given protocol (the protocol block equals it, the
  selection and final splits, epoch_transfer), with one candidate per (LR, epoch) of the grid; every candidate's dump
  dir is tracked under the given dumps dir with a val dump that verifies (pilot.dumps.load_dump) to the candidate's
  dump_sha256 and training run id, and with no test dump; the chosen candidate is what pilot.postfit.select returns on
  the recorded candidates under that protocol (the pre-declared rule, never a hand pick), its T lies in
  [T_MIN, T_MAX] and stop_after_epoch is None exactly at the last epoch of the schedule
- _check_reference: the reference check equals pilot.berturk.reference_check of the record (the path the check was
  computed at, the sha256 of the record's bytes and the committed file's own resolution), so it is never edited by
  hand; status "investigate" exactly when the exact fraction is below 707/1000
- _reproduce (data marker): pilot.postfit.fit over the candidate dump dirs under the protocol, serialized as
  write_record does, equals the committed bytes (needs `make data`, because the gold labels come from the verified
  splits)

Superseded runs (pilot.berturk.SUPERSEDED, the BASE-03 fix of the 06-10 gate): the tied-LR selection lives byte for
byte under results/superseded/<name>/; its record's sha256 is the pinned postfit_sha256, its protocol is the pinned
protocol, its reference check still reads investigate with no resolution (so check_reference_for_final refuses it as
the gate of any final seed), and the record rebuilds from the moved dumps. test_no_superseded_run_is_current proves
none of it is current: no tracked results/dumps dir holds one of its dumps or is named after one of its candidates, no
tracked results/postfit file carries its sha256, and results/dumps holds BERTurk selection dumps of at most one train
run. Plan 06-16 adds the current-run tests (the head-LR run's record, its reference and its Makefile list) to the same
helpers.
"""
import glob
import json
import os
import subprocess
from fractions import Fraction

import pytest

from pilot import berturk, dumps, postfit
from pilot.data import DEFAULT_MANIFEST, sha256_file

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERSEDED_IDS = [s["name"] for s in berturk.SUPERSEDED]
# the reference value each superseded entry was retired at (plan 06-10: 150/551 = 0.2722 < 0.707)
SUPERSEDED_VALUE = {"berturk-tied-head-lr": {"status": "investigate", "fraction": [150, 551]}}


def tracked(*pathspec):
    """Tracked paths via git ls-files (never a directory walk, which would see untracked scratch files)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *pathspec], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    return [f for f in out.stdout.split("\0") if f]


def path(rel):
    return os.path.join(REPO_ROOT, rel)


def load(rel):
    with open(path(rel), encoding="utf-8") as f:
        return json.load(f)


def as_json(v):
    return json.loads(json.dumps(v, ensure_ascii=False))


def _check_record(record_rel, dumps_rel, proto):
    """The consistency checks of one committed BERTurk postfit record whose candidate dumps sit under dumps_rel and
    that ran under proto (pilot.postfit.protocol keys); returns the parsed record."""
    assert tracked(record_rel) == [record_rel], "%s is not tracked" % record_rel
    r = load(record_rel)
    assert r["schema"] == postfit.SCHEMA == "postfit/1"
    assert r["system_kind"] == "berturk"
    for key in ("lr_grid", "head_lr", "epochs", "selection_seed", "final_seeds"):
        assert as_json(r["protocol"][key]) == as_json(proto[key]), key
    assert r["protocol"]["selection_split"] == "selection_train" and r["protocol"]["final_split"] == "final_train"
    epochs = proto["epochs"]
    assert r["rules"]["epoch_transfer"] == postfit.EPOCH_TRANSFER_TEMPLATE % epochs
    cands = r["candidates"]
    want = sorted((lr, e) for lr in proto["lr_grid"] for e in range(1, epochs + 1))
    assert sorted((c["encoder_lr"], c["epoch"]) for c in cands) == want
    assert all(c["system_kind"] == "berturk" for c in cands)
    files = set(tracked(dumps_rel + "/"))
    for c in cands:
        rid = c["dump_run_id"]
        for name in ("val.npz", "val.json"):
            assert "%s/%s/%s" % (dumps_rel, rid, name) in files, "%s/%s is not tracked under %s" % (rid, name,
                                                                                                   dumps_rel)
        for name in ("test.npz", "test.json"):
            assert "%s/%s/%s" % (dumps_rel, rid, name) not in files, "%s holds a tracked test dump" % rid
        logits, uids, side, entry = dumps.load_dump(path("%s/%s" % (dumps_rel, rid)), "val", DEFAULT_MANIFEST)
        assert side["run_id"] == rid and side["dump_sha256"] == c["dump_sha256"], rid
        assert side["training"]["run_id"] == c["train_run_id"] and side["n_subset"] is None, rid
        assert entry["role"] == "val" and logits.shape == (len(uids), 48), rid
    chosen = postfit.select(cands, **{k: v for k, v in proto.items() if k != "final_seeds"})
    c = r["chosen"]
    assert c["dump_run_id"] == chosen["dump_run_id"] and c["dump_sha256"] == chosen["dump_sha256"]
    assert c["epoch"] == chosen["epoch"] and as_json(c["theta"]) == as_json(chosen["theta"])
    assert postfit.T_MIN <= c["temperature"] <= postfit.T_MAX
    assert c["temperature"] == chosen["temperature"]
    assert (c["stop_after_epoch"] is None) == (c["epoch"] == epochs)
    assert c["stop_after_epoch"] in (None, c["epoch"])
    assert r["final"]["split"] == "final_train" and r["final"]["seeds"] == list(proto["final_seeds"])
    assert r["final"]["stop_after_epoch"] == c["stop_after_epoch"] and r["final"]["epochs"] == epochs
    return r


def _check_reference(record_rel, reference_rel, path_at_check):
    """The committed reference check equals its recompute from the record (computed when the record sat at
    path_at_check); returns the parsed reference check."""
    assert tracked(reference_rel) == [reference_rel], "%s is not tracked" % reference_rel
    record, sha = postfit.read_record(path(record_rel))
    assert sha == sha256_file(path(record_rel))
    committed = load(reference_rel)
    assert committed["schema"] == berturk.REFERENCE_SCHEMA == "berturk-reference/1"
    assert committed["postfit"] == {"path": path_at_check, "sha256": sha}, "the reference check is of another record"
    got = berturk.reference_check(record, path(path_at_check), sha, committed.get("resolution"))
    assert as_json(got) == committed, "%s differs from its recompute; never edit it by hand" % reference_rel
    num, den = committed["value"]["fraction"]
    below = Fraction(num, den) < Fraction(707, 1000)
    assert committed["status"] == ("investigate" if below else "ok") and committed["below_threshold"] is below
    return committed


def _reproduce(record_rel, dumps_rel, grid):
    """pilot.postfit.fit over the record's candidate dump dirs (record order) under grid rebuilds its bytes."""
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")
    record = load(record_rel)
    dirs = [path("%s/%s" % (dumps_rel, c["dump_run_id"])) for c in record["candidates"]]
    got = postfit.fit(dirs, DEFAULT_MANIFEST, **grid)
    with open(path(record_rel), "rb") as f:
        assert postfit.record_bytes(got) == f.read(), "%s does not rebuild from %s" % (record_rel, dumps_rel)


@pytest.mark.parametrize("s", berturk.SUPERSEDED, ids=SUPERSEDED_IDS)
def test_superseded_record_is_complete_and_consistent(s):
    assert sha256_file(path(s["record"])) == s["postfit_sha256"], \
        "%s is not the byte-identical superseded record pinned in pilot.berturk.SUPERSEDED" % s["record"]
    r = _check_record(s["record"], s["dumps"], s["protocol"])
    assert {c["train_run_id"] for c in r["candidates"]} == {s["train_run_id"]}
    assert r["chosen"]["train_run_id"] == s["train_run_id"]


@pytest.mark.parametrize("s", berturk.SUPERSEDED, ids=SUPERSEDED_IDS)
def test_superseded_reference_check_equals_its_recompute(s):
    ref = _check_reference(s["record"], s["reference"], s["record_path_at_check"])
    assert ref["postfit"] == {"path": s["record_path_at_check"], "sha256": s["postfit_sha256"]}
    want = SUPERSEDED_VALUE[s["name"]]
    assert ref["status"] == want["status"] and ref["value"]["fraction"] == want["fraction"]
    assert ref["resolution"] is None
    # a superseded run can never gate a final seed
    with pytest.raises(SystemExit, match="BASE-03"):
        berturk.check_reference_for_final(path(s["reference"]), s["postfit_sha256"])


@pytest.mark.data
@pytest.mark.parametrize("s", berturk.SUPERSEDED, ids=SUPERSEDED_IDS)
def test_superseded_record_reproduces_from_its_dumps(s):
    _reproduce(s["record"], s["dumps"], s["protocol"])


def test_no_superseded_run_is_current():
    train_ids = {s["train_run_id"] for s in berturk.SUPERSEDED}
    cand_ids = {c["dump_run_id"] for s in berturk.SUPERSEDED for c in load(s["record"])["candidates"]}
    shas = {s["postfit_sha256"] for s in berturk.SUPERSEDED}
    files = tracked("results/dumps/")
    named = sorted({f.split("/")[2] for f in files} & cand_ids)
    assert named == [], "results/dumps holds dirs named after superseded candidates: %s" % named
    held, selection_runs = [], set()
    for rel in files:
        if not rel.endswith("/val.json") or not os.path.isfile(path(rel)):
            continue
        side = load(rel)
        if (side.get("system") or {}).get("kind") != "berturk":
            continue
        tr = side.get("training") or {}
        if tr.get("run_id") in train_ids:
            held.append(rel)
        if tr.get("split") == "selection_train":
            selection_runs.add(tr.get("run_id"))
    assert held == [], "results/dumps holds dumps of a superseded BERTurk run: %s" % held
    assert len(selection_runs) <= 1, "results/dumps holds BERTurk selection dumps of %d train runs: %s" % (
        len(selection_runs), sorted(selection_runs))
    records = [rel for rel in tracked("results/postfit/") if rel.endswith(".json") and os.path.isfile(path(rel))]
    stale = [rel for rel in records if sha256_file(path(rel)) in shas]
    assert stale == [], "results/postfit holds a superseded record: %s" % stale
