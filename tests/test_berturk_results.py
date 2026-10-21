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

Superseded runs (pilot.berturk.SUPERSEDED, the BASE-03 fixes of the 06-10 and 06-16 gates): an ordered chain of two
discarded selections, the tied-LR run and the head-LR, 10-epoch-grid run, each living byte for byte under
results/superseded/<name>/. For each one, its record's sha256 is the pinned postfit_sha256, its protocol is the pinned
protocol, its reference check still reads investigate with no resolution (so check_reference_for_final refuses it as
the gate of any final seed), and the record rebuilds from the moved dumps.
test_superseded_runs_change_one_recorded_key_per_step proves consecutive entries differ in exactly the recorded delta
(canonical config, run id and protocol). test_no_superseded_run_is_current proves none of it is current: no tracked
results/dumps dir holds one of its dumps or is named after one of its candidates, no tracked results/postfit file
carries its sha256, and results/dumps holds BERTurk selection dumps of at most one train run.

The current run: results/postfit/berturk.json is checked by the same helpers under pilot.postfit.protocol("berturk"),
every candidate's canonical head_lr is pilot.berturk.HEAD_LR and its sha256 is no superseded one; the Makefile's
BERTURK_SELECTION lists its candidate dump run ids in record order and berturk-postfit is phony;
results/postfit/berturk-reference.json equals its recompute, and an "investigate" status carries a resolution unless no
final seed exists yet (results/postfit/berturk-final.json untracked); the record rebuilds from the committed dumps with
the protocol dispatched by kind (data marker). test_exactly_one_current_berturk_selection_run (RP-2) proves the current
selection is one train run: every tracked BERTurk selection val sidecar under results/dumps names the record's chosen
train run, the BERTurk dump dirs there are exactly the record's candidates, and that run's canonical config is each
superseded run's chosen one with head_lr pilot.berturk.HEAD_LR and nothing else changed (plan 06-20 generalizes this
to SUPERSEDED[-1]'s chosen config with its delta applied).

Interim state (plans 06-17..06-19): plan 06-17 moved the head-LR run to results/superseded/, so until plan 06-20 commits
the 30-epoch record no current BERTurk record exists. Each of the five current-run tests first calls _current_record:
with results/postfit/berturk.json tracked it returns the record and the test runs as above; without it the helper
asserts the interim state instead of skipping (no current record, reference check or final manifest tracked, no
BERTURK_SELECTION line or berturk-postfit target in the Makefile, and every tracked BERTurk dump under results/dumps a
selection_train val dump of at most one train run that is not superseded) and the test ends there. Plan 06-20 makes
the record mandatory again.
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
SUPERSEDED_IDS = [s["name"] for s in berturk.SUPERSEDED]
# the reference value each superseded entry was retired at (plan 06-10: 150/551 = 0.2722 < 0.707; plan 06-16:
# 113/238 = 0.4748 < 0.707)
SUPERSEDED_VALUE = {"berturk-tied-head-lr": {"status": "investigate", "fraction": [150, 551]},
                    "berturk-10-epoch-grid": {"status": "investigate", "fraction": [113, 238]}}
RECORD = "results/postfit/berturk.json"
REFERENCE = "results/postfit/berturk-reference.json"
FINAL = "results/postfit/berturk-final.json"
DUMPS = "results/dumps"


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


def _sorted_json(v):
    return json.dumps(as_json(v), sort_keys=True)


def test_superseded_runs_change_one_recorded_key_per_step():
    """R2-2: SUPERSEDED is an ordered chain, oldest gate first, and each step changes exactly its recorded delta: the
    next run's chosen canonical config is the previous one's with delta applied and nothing else changed, its run id
    follows, and the protocol blocks differ only where delta touches a protocol key."""
    gates = [tuple(int(p) for p in s["gate"]["plan"].split("-")) for s in berturk.SUPERSEDED]
    assert gates == sorted(gates) and len(set(gates)) == len(gates), "SUPERSEDED is not ordered by its gate plan"
    for s in berturk.SUPERSEDED:
        d = s["delta"]
        assert isinstance(d, dict) and d, "%s: delta must be a non-empty dict" % s["name"]
        assert set(d) <= set(berturk.IDENTITY), "%s: delta keys %s are not run identity keys" % (s["name"], sorted(d))
        fixed = set(d) & {"split", "seed", "data_sha256", "base_model_sha256"}
        assert not fixed, "%s: delta changes %s, which no recipe fix may change" % (s["name"], sorted(fixed))
    for a, b in zip(berturk.SUPERSEDED, berturk.SUPERSEDED[1:]):
        old, new = (dict(load(s["record"])["chosen"]["canonical"]) for s in (a, b))
        assert berturk.run_id(old) == a["train_run_id"] and berturk.run_id(new) == b["train_run_id"]
        changed = sorted(k for k in set(old) | set(new) if as_json(old.get(k)) != as_json(new.get(k)))
        assert changed == sorted(a["delta"]), "%s -> %s changed %s, but its recorded delta is %s" % (
            a["name"], b["name"], changed, sorted(a["delta"]))
        applied = dict(old, **a["delta"])
        assert _sorted_json(applied) == _sorted_json(new), "%s with its delta applied is not %s's canonical config" % (
            a["name"], b["name"])
        assert berturk.run_id(applied) == b["train_run_id"]
        proto = dict(a["protocol"], **{k: v for k, v in a["delta"].items() if k in a["protocol"]})
        assert _sorted_json(proto) == _sorted_json(b["protocol"]), \
            "%s's protocol with its delta applied is not %s's protocol" % (a["name"], b["name"])


def _canonical_of_chosen(record):
    return record["chosen"]["canonical"]


INTERIM = ("no current BERTurk selection record: the interim state of plans 06-17..06-19; plan 06-20 makes the record "
           "mandatory again")


def _current_record():
    """The parsed current BERTurk postfit record, or None after asserting the interim state (plans 06-17..06-19):
    no current record, reference check or final manifest is tracked, the Makefile has no BERTURK_SELECTION line and no
    berturk-postfit target, and every tracked BERTurk dump under results/dumps is a selection_train val dump of at most
    one train run that is not superseded. Never a skip: a stray current artifact fails the calling test."""
    if tracked(RECORD) == [RECORD]:
        return load(RECORD)
    for rel in (REFERENCE, FINAL):
        assert tracked(rel) == [], "%s is tracked without %s; %s" % (rel, RECORD, INTERIM)
    with open(path("Makefile"), encoding="utf-8") as f:
        text = f.read()
    assert re.search(r"^BERTURK_SELECTION :=", text, re.M) is None, \
        "Makefile has a BERTURK_SELECTION := line; %s" % INTERIM
    assert re.search(r"^berturk-postfit:", text, re.M) is None, "Makefile has a berturk-postfit target; %s" % INTERIM
    superseded = {s["train_run_id"] for s in berturk.SUPERSEDED}
    runs, stray = set(), []
    for rel in tracked(DUMPS + "/"):
        if not rel.endswith(".json") or not os.path.isfile(path(rel)):
            continue
        side = load(rel)
        if (side.get("system") or {}).get("kind") != "berturk":
            continue
        tr = side.get("training") or {}
        if not rel.endswith("/val.json") or tr.get("split") != "selection_train":
            stray.append(rel)
        runs.add(tr.get("run_id"))
    assert stray == [], "BERTurk dumps other than selection_train val dumps under %s: %s; %s" % (DUMPS, stray, INTERIM)
    assert len(runs) <= 1, "BERTurk dumps of %d train runs under %s: %s; %s" % (len(runs), DUMPS, sorted(runs), INTERIM)
    assert not runs & superseded, "BERTurk dumps of a superseded run under %s: %s; %s" % (
        DUMPS, sorted(runs & superseded), INTERIM)
    return None


def test_committed_berturk_record_is_complete_and_consistent():
    if _current_record() is None:
        return
    r = _check_record(RECORD, DUMPS, postfit.protocol("berturk"))
    assert r["protocol"]["head_lr"] == berturk.HEAD_LR == 0.0001
    for c in r["candidates"]:
        assert c["canonical"]["head_lr"] == berturk.HEAD_LR, "%s: head_lr %r" % (c["dump_run_id"],
                                                                                 c["canonical"]["head_lr"])
        assert c["canonical"]["encoder_lr"] == c["encoder_lr"] == berturk.LR, c["dump_run_id"]
    assert _canonical_of_chosen(r)["head_lr"] == berturk.HEAD_LR
    sha = sha256_file(path(RECORD))
    assert sha not in {s["postfit_sha256"] for s in berturk.SUPERSEDED}, "%s is a superseded record" % RECORD
    assert r["chosen"]["train_run_id"] not in {s["train_run_id"] for s in berturk.SUPERSEDED}


def test_makefile_lists_the_berturk_selection_dumps():
    record = _current_record()
    if record is None:
        return
    with open(path("Makefile"), encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^BERTURK_SELECTION := (.+)$", text, re.M)
    assert m, "Makefile has no BERTURK_SELECTION := line"
    assert m.group(1).split() == [c["dump_run_id"] for c in record["candidates"]], \
        "BERTURK_SELECTION is not the record's candidate dump run ids in record order"
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "berturk-postfit" in phony.group(1).split(), "berturk-postfit is not in .PHONY"
    recipe = re.search(r"^berturk-postfit:\n((?:\t.*\n)+)", text, re.M)
    assert recipe, "Makefile has no berturk-postfit target"
    assert ("$(PYTHON) postfit.py $(addprefix results/dumps/,$(BERTURK_SELECTION)) --out %s" % RECORD) \
        in recipe.group(1), "berturk-postfit does not rebuild %s from results/dumps/$(BERTURK_SELECTION)" % RECORD


def test_committed_reference_check_equals_its_recompute():
    record = _current_record()
    if record is None:
        return
    ref = _check_reference(RECORD, REFERENCE, RECORD)
    assert ref["value"]["train_run_id"] == record["chosen"]["train_run_id"]
    assert ref["value"]["dump_run_id"] == record["chosen"]["dump_run_id"]
    if ref["status"] == "investigate":
        res = ref["resolution"]
        assert (isinstance(res, str) and res.strip()) or tracked(FINAL) == [], \
            "the BASE-03 check reads investigate with no resolution, yet %s is tracked (a final seed trained)" % FINAL
    else:
        assert ref["resolution"] is None


@pytest.mark.data
def test_berturk_record_reproduces_from_the_committed_dumps():
    if _current_record() is None:
        return
    _reproduce(RECORD, DUMPS, {})                # no grid: postfit.fit dispatches the protocol by system kind


def test_exactly_one_current_berturk_selection_run():
    record = _current_record()
    if record is None:
        return
    chosen_run = record["chosen"]["train_run_id"]
    runs, dirs = set(), set()
    for rel in tracked(DUMPS + "/"):
        if not rel.endswith("/val.json") or not os.path.isfile(path(rel)):
            continue
        side = load(rel)
        if (side.get("system") or {}).get("kind") != "berturk":
            continue
        dirs.add(rel.split("/")[2])
        tr = side.get("training") or {}
        if tr.get("split") == "selection_train":
            runs.add(tr.get("run_id"))
    assert runs == {chosen_run}, "results/dumps holds BERTurk selection dumps of %s, not only %s" % (sorted(runs),
                                                                                                     chosen_run)
    assert dirs == {c["dump_run_id"] for c in record["candidates"]}, \
        "the BERTurk dump dirs under results/dumps are not exactly the record's candidates"
    assert {c["train_run_id"] for c in record["candidates"]} == {chosen_run}
    canon = _canonical_of_chosen(record)
    assert berturk.run_id(canon) == chosen_run
    for s in berturk.SUPERSEDED:
        old = dict(load(s["record"])["chosen"]["canonical"])
        assert berturk.run_id(old) == s["train_run_id"] != chosen_run
        old["head_lr"] = berturk.HEAD_LR
        assert json.dumps(as_json(canon), sort_keys=True) == json.dumps(as_json(old), sort_keys=True), \
            "the current run's canonical config differs from the superseded %s in more than head_lr" % s["name"]
