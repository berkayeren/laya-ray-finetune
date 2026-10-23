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

The current run is the 30-epoch selection run of plans 06-18/06-19 (berturk-selection_train-s0-4f36ae05; plan 06-20
committed its record). results/postfit/berturk.json is mandatory: _current_record asserts it is tracked, so a missing
current record fails every current-run test (the interim state of plans 06-17..06-19 has ended). The record is checked
by the same helpers under pilot.postfit.protocol("berturk"); its protocol epochs and every candidate's canonical epochs
are pilot.berturk.EPOCHS (30), every candidate's canonical head_lr is pilot.berturk.HEAD_LR and its sha256 is no
superseded one; the Makefile's BERTURK_SELECTION lists its candidate dump run ids in record order and berturk-postfit is
phony; results/postfit/berturk-reference.json equals its recompute, and an "investigate" status carries a resolution
unless no final seed exists yet (results/postfit/berturk-final.json untracked); the record rebuilds from the committed
dumps with the protocol dispatched by kind (data marker). test_exactly_one_current_berturk_selection_run (R2-2) proves
the current selection is one train run: every tracked BERTurk selection val sidecar under results/dumps names the
record's chosen train run, the BERTurk dump dirs there are exactly the record's candidates, and that run's canonical
config is pilot.berturk.SUPERSEDED[-1]'s chosen one with SUPERSEDED[-1]["delta"] applied and nothing else changed (the
end of the chain that test_superseded_runs_change_one_recorded_key_per_step checks step by step), and its run id is no
superseded one. Both selection checks read berturk_val_sidecars, which skips a val sidecar whose training canonical
carries a fewshot binding: a Phase 9 masked BERTurk base trains on selection_train too and is never a selection run
(FEW-01; proven on a planted sidecar by test_berturk_selection_checks_skip_a_fewshot_dump).

Final seeds (plan 06-11): test_committed_berturk_final_manifest checks results/postfit/berturk-final.json
(postfit-final/1, system kind berturk): bound to the current record's sha256 (never a superseded one) and its chosen
settings (30-epoch schedule, the chosen stop epoch and T), seeds exactly [0, 1, 2] on final_train with distinct train
run ids, one config_sha256_sans_seed, each seed serving the chosen T over the chosen number of epochs, bare checkpoint
names, and a final seed's tracked results/dumps dir (none before the 08-11 scoring run) holds only test and
hotels_probe dumps whose sidecars carry protocol_sha256 == pilot.protocol.PROTOCOL_SHA256 (set): Phase 8's protocol
gate is the only way a final seed is scored, never on val.

README block (plan 06-12): test_readme_berturk_block_matches_the_records proves the block between the berturk-postfit
markers is pilot.berturk.render_markdown of the committed record, reference check and final manifest plus
pilot.berturk.load_superseded(); test_render_discloses_the_protocol_change proves every SUPERSEDED entry's run id,
exact fraction, gate reply, change and history dir, the extra-run count, the reference check's differences and
caveats and the epoch-grid deviation reach it; test_load_superseded_refuses_a_changed_history refuses a history
record with one byte changed, naming the file.
"""
import glob
import json
import os
import re
import subprocess
import sys
from fractions import Fraction

import pytest

from pilot import berturk, dumps, postfit, protocol
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


def fewshot_bound(side):
    """True when a sidecar's training canonical config carries a fewshot binding: a Phase 9 masked BERTurk base
    (FEW-01) trains on selection_train too, and its val dump is never a selection run's."""
    canon = (side.get("training") or {}).get("canonical")
    return isinstance(canon, dict) and "fewshot" in canon


def berturk_val_sidecars():
    """{rel: parsed val.json} of every tracked BERTurk val dump under results/dumps without a few-shot binding."""
    out = {}
    for rel in tracked(DUMPS + "/"):
        if not rel.endswith("/val.json") or not os.path.isfile(path(rel)):
            continue
        side = load(rel)
        if (side.get("system") or {}).get("kind") == "berturk" and not fewshot_bound(side):
            out[rel] = side
    return out


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
def test_superseded_reference_check_equals_its_recompute(s, tmp_path):
    ref = _check_reference(s["record"], s["reference"], s["record_path_at_check"])
    assert ref["postfit"] == {"path": s["record_path_at_check"], "sha256": s["postfit_sha256"]}
    want = SUPERSEDED_VALUE[s["name"]]
    assert ref["status"] == want["status"] and ref["value"]["fraction"] == want["fraction"]
    assert ref["resolution"] is None
    # a superseded run can never gate a final seed, not even with its status and fraction edited together (WR-01)
    record, sha = postfit.read_record(path(s["record"]))
    with pytest.raises(SystemExit, match="BASE-03"):
        berturk.check_reference_for_final(path(s["reference"]), s["postfit_sha256"], record)
    edited = dict(ref, status="ok", below_threshold=False, value=dict(ref["value"], fraction=[800, 1000]))
    edited_path = os.path.join(str(tmp_path), "berturk-reference.json")
    with open(edited_path, "w", encoding="utf-8") as f:
        json.dump(edited, f, ensure_ascii=False, indent=2)
    with pytest.raises(SystemExit, match="is not pilot.berturk.reference_check"):
        berturk.check_reference_for_final(edited_path, sha, record)


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
    for rel, side in berturk_val_sidecars().items():
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


def _current_record():
    """The parsed current BERTurk postfit record; it must be tracked (plan 06-20 ended the interim state of plans
    06-17..06-19, so a missing record fails the calling test, never skips it)."""
    assert tracked(RECORD) == [RECORD], "%s is not tracked; the current BERTurk selection record is mandatory" % RECORD
    return load(RECORD)


def test_committed_berturk_record_is_complete_and_consistent():
    _current_record()
    r = _check_record(RECORD, DUMPS, postfit.protocol("berturk"))
    assert r["protocol"]["head_lr"] == berturk.HEAD_LR == 0.0001
    assert r["protocol"]["epochs"] == berturk.EPOCHS == 30
    for c in r["candidates"]:
        assert c["canonical"]["epochs"] == berturk.EPOCHS, "%s: epochs %r" % (c["dump_run_id"],
                                                                               c["canonical"]["epochs"])
        assert c["canonical"]["head_lr"] == berturk.HEAD_LR, "%s: head_lr %r" % (c["dump_run_id"],
                                                                                 c["canonical"]["head_lr"])
        assert c["canonical"]["encoder_lr"] == c["encoder_lr"] == berturk.LR, c["dump_run_id"]
    assert _canonical_of_chosen(r)["head_lr"] == berturk.HEAD_LR
    sha = sha256_file(path(RECORD))
    assert sha not in {s["postfit_sha256"] for s in berturk.SUPERSEDED}, "%s is a superseded record" % RECORD
    assert r["chosen"]["train_run_id"] not in {s["train_run_id"] for s in berturk.SUPERSEDED}


def test_makefile_lists_the_berturk_selection_dumps():
    record = _current_record()
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
    ref = _check_reference(RECORD, REFERENCE, RECORD)
    assert ref["value"]["train_run_id"] == record["chosen"]["train_run_id"]
    assert ref["value"]["dump_run_id"] == record["chosen"]["dump_run_id"]
    if ref["status"] == "investigate":
        res = ref["resolution"]
        assert (isinstance(res, str) and res.strip()) or tracked(FINAL) == [], \
            "the BASE-03 check reads investigate with no resolution, yet %s is tracked (a final seed trained)" % FINAL
    else:
        assert ref["resolution"] is None
    # the committed check passes the final-seed gate, which recomputes it from the record (06-REVIEW WR-01)
    rec, sha = postfit.read_record(path(RECORD))
    assert berturk.check_reference_for_final(path(REFERENCE), sha, rec) == ref


@pytest.mark.data
def test_berturk_record_reproduces_from_the_committed_dumps():
    _current_record()
    _reproduce(RECORD, DUMPS, {})                # no grid: postfit.fit dispatches the protocol by system kind


def test_exactly_one_current_berturk_selection_run():
    record = _current_record()
    chosen_run = record["chosen"]["train_run_id"]
    runs, dirs = set(), set()
    for rel, side in berturk_val_sidecars().items():
        dirs.add(rel.split("/")[2])
        tr = side.get("training") or {}
        if tr.get("split") == "selection_train":
            runs.add(tr.get("run_id"))
    assert runs == {chosen_run}, "results/dumps holds BERTurk selection dumps of %s, not only %s" % (sorted(runs),
                                                                                                     chosen_run)
    assert dirs == {c["dump_run_id"] for c in record["candidates"]}, \
        "the BERTurk dump dirs under results/dumps are not exactly the record's candidates"
    assert {c["train_run_id"] for c in record["candidates"]} == {chosen_run}
    canon = dict(_canonical_of_chosen(record))
    assert berturk.run_id(canon) == chosen_run
    assert chosen_run not in {s["train_run_id"] for s in berturk.SUPERSEDED}, "%s is a superseded run" % chosen_run
    # the chain end: the current run is SUPERSEDED[-1]'s chosen config with its recorded delta applied, nothing else
    last = berturk.SUPERSEDED[-1]
    old = dict(load(last["record"])["chosen"]["canonical"])
    assert berturk.run_id(old) == last["train_run_id"] != chosen_run
    changed = sorted(k for k in set(old) | set(canon) if as_json(old.get(k)) != as_json(canon.get(k)))
    assert changed == sorted(last["delta"]), \
        "the current run changed %s from the superseded %s, but its delta is %s" % (changed, last["name"],
                                                                                   sorted(last["delta"]))
    applied = dict(old, **last["delta"])
    assert _sorted_json(applied) == _sorted_json(canon), \
        "the current run's canonical config is not the superseded %s's with its delta applied" % last["name"]
    assert berturk.run_id(applied) == chosen_run


def test_berturk_selection_checks_skip_a_fewshot_dump(tmp_path, monkeypatch):
    # a Phase 9 masked BERTurk base trains on selection_train too (FEW-01): its val dump is never a selection run
    module = sys.modules[__name__]
    real_tracked, real_path = module.tracked, module.path
    src = sorted(c["dump_run_id"] for c in _current_record()["candidates"])[0]
    side = load("%s/%s/val.json" % (DUMPS, src))
    assert side["system"]["kind"] == "berturk" and isinstance(side["training"]["canonical"], dict)
    side["training"]["run_id"] = "berturk-selection_train-s0-0badf00d"   # a second train run on selection_train
    rel = DUMPS + "/berturk-fewshot-planted/val.json"
    monkeypatch.setattr(module, "tracked", lambda *spec: real_tracked(*spec) + ([rel] if any(
        p.startswith(DUMPS) for p in spec) else []))
    monkeypatch.setattr(module, "path", lambda r: str(tmp_path / "val.json") if r == rel else real_path(r))
    (tmp_path / "val.json").write_text(json.dumps(side), encoding="utf-8")
    with pytest.raises(AssertionError):                         # without the binding the copy is a second run
        test_exactly_one_current_berturk_selection_run()
    side["training"]["canonical"]["fewshot"] = {"protocol_sha256": "a" * 64, "stage": "base"}
    (tmp_path / "val.json").write_text(json.dumps(side), encoding="utf-8")
    test_exactly_one_current_berturk_selection_run()
    test_no_superseded_run_is_current()
    assert rel not in berturk_val_sidecars() and len(berturk_val_sidecars()) == len(_current_record()["candidates"])


FINAL_RUN_RE = re.compile(r"^berturk-final_train-s([012])-[0-9a-f]{8}$")


def test_committed_berturk_final_manifest():
    """BASE-01 (plan 06-11): the committed manifest of the 3 BERTurk final seeds is bound to the current record (its
    sha256 and chosen settings, the 30-epoch schedule stopped at the chosen epoch), never to a superseded one; it lists
    all three seeds (never only the best), each serving the chosen T, and a final seed's tracked dumps are only its
    protocol-gated test and hotels_probe dumps (Phase 8)."""
    assert tracked(FINAL) == [FINAL], "%s is not tracked" % FINAL
    record = _current_record()
    m = load(FINAL)
    assert m["schema"] == postfit.FINAL_SCHEMA == "postfit-final/1" and m["system_kind"] == "berturk"
    sha = sha256_file(path(RECORD))
    assert m["postfit"] == {"path": RECORD, "sha256": sha}, "%s is not bound to the current %s" % (FINAL, RECORD)
    assert sha not in {s["postfit_sha256"] for s in berturk.SUPERSEDED}, "%s is bound to a superseded record" % FINAL
    keys = ("train_run_id", "encoder_lr", "epoch", "epochs_schedule", "stop_after_epoch", "temperature")
    chosen = record["chosen"]
    assert as_json(m["chosen"]) == as_json({k: chosen[k] for k in keys}), "the manifest's chosen settings differ"
    assert chosen["canonical"]["head_lr"] == berturk.HEAD_LR == 0.0001
    assert chosen["canonical"]["epochs"] == m["chosen"]["epochs_schedule"] == berturk.EPOCHS == 30
    assert m["chosen"]["train_run_id"] not in {s["train_run_id"] for s in berturk.SUPERSEDED}
    assert m["final_split"] == record["final"]["split"] == "final_train"
    seeds = m["seeds"]
    assert [e["seed"] for e in seeds] == [0, 1, 2] == record["final"]["seeds"]
    ids = [e["train_run_id"] for e in seeds]
    assert len(set(ids)) == 3, "the final seeds share a train run id: %s" % ids
    for e in seeds:
        mm = FINAL_RUN_RE.match(e["train_run_id"])
        assert mm and int(mm.group(1)) == e["seed"], e["train_run_id"]
    assert re.fullmatch(r"[0-9a-f]{64}", m["config_sha256_sans_seed"]), "one config_sha256_sans_seed for all seeds"
    for e in seeds:
        assert e["temperature"] == chosen["temperature"], "seed %d serves T %r" % (e["seed"], e["temperature"])
        assert len(e["epochs"]) == len(e["loss_by_epoch"]) == chosen["epoch"], "seed %d epochs" % e["seed"]
        assert [r["epoch"] for r in e["epochs"]] == list(range(1, chosen["epoch"] + 1))
        assert "/" not in e["checkpoint_name"] and "\\" not in e["checkpoint_name"], e["checkpoint_name"]
        assert e["dump_run_id"] == berturk.dump_run_id(e["checkpoint_identity"])
    gated = {"test.npz", "test.json", "hotels_probe.npz", "hotels_probe.json"}
    for e in seeds:                                         # Phase 8: gated test and HOTELS dumps only (08-08)
        names = [f.split("/")[3] for f in tracked("%s/%s/" % (DUMPS, e["dump_run_id"]))]
        assert set(names) <= gated, "%s final seed %d's dump dir holds %s; only gated test and hotels_probe dumps" % (
            "BERTurk", e["seed"], sorted(set(names) - gated))
        for n in names:
            if n.endswith(".json"):
                pin = load("%s/%s/%s" % (DUMPS, e["dump_run_id"], n)).get("protocol_sha256")
                assert protocol.PROTOCOL_SHA256 is not None and pin == protocol.PROTOCOL_SHA256, (
                    "%s/%s carries protocol_sha256 %r, the pinned protocol is %r" % (e["dump_run_id"], n, pin,
                                                                                    protocol.PROTOCOL_SHA256))


# ---- the README Phase 6 block (plan 06-12, BASE-03 disclosure, T-06-36, T-06-55, T-06-74) ----
README = "README.md"
BERTURK_START, BERTURK_END = "<!-- berturk-postfit:start -->", "<!-- berturk-postfit:end -->"


def _rendered_berturk():
    return berturk.render_markdown(_current_record(), load(REFERENCE), load(FINAL), berturk.load_superseded())


def test_readme_berturk_block_matches_the_records():
    """The README block between the berturk-postfit markers is render_markdown of the committed record, reference
    check and final manifest plus the superseded history; a hand edit or a stale block fails."""
    with open(path(README), encoding="utf-8") as f:
        text = f.read()
    assert text.count(BERTURK_START) == 1 and text.count(BERTURK_END) == 1, \
        "README.md must hold exactly one pair of the berturk-postfit markers"
    i, j = text.index(BERTURK_START), text.index(BERTURK_END)
    assert i < j, "the berturk-postfit end marker comes before the start marker"
    assert text[i + len(BERTURK_START):j].strip() == _rendered_berturk().strip(), (
        "the README berturk-postfit block differs from pilot.berturk.render_markdown of the committed records; "
        "regenerate it with `python postfit.py --render-berturk %s %s %s`" % (RECORD, REFERENCE, FINAL))
    assert text.index("## BERTurk baseline (Phase 6)") < i, "the block is not in the BERTurk baseline section"


def test_render_discloses_per_tag_theta_on_a_grid_edge():
    """06-REVIEW IN-02: the block names the chosen candidate's per-tag θ on a grid edge and counts them over every
    candidate, from the record (the global-θ edge line alone ignores them)."""
    record = _current_record()
    def edge(t):
        return [k for k, v in t["per_tag"].items() if v in (t["grid"]["start"], t["grid"]["stop"])]
    counts = {n: (sum(len(edge(c["theta"][n])) for c in record["candidates"]),
                  sum(bool(edge(c["theta"][n])) for c in record["candidates"])) for n in ("shipped", "raw")}
    line = next(l for l in _rendered_berturk().splitlines() if l.startswith("- Per-tag θ grid edge:"))
    assert "%d shipped per-tag θ (%d candidates) and %d raw per-tag θ (%d candidates)" % (
        counts["shipped"] + counts["raw"]) in line, line
    chosen = record["chosen"]["theta"]
    assert ("shipped none" in line) == (not edge(chosen["shipped"])) and ("raw none" in line) == (not edge(chosen["raw"]))


def test_render_discloses_the_protocol_change():
    """Every SUPERSEDED entry's train run id, exact fraction, gate reply and history dir reach the block, with the
    count of extra selection runs; the reference check's value, threshold, status, differences and caveats too."""
    text = _rendered_berturk()
    assert "Protocol changes after a val outcome (BASE-03 fixes)" in text
    for s in berturk.SUPERSEDED:
        num, den = SUPERSEDED_VALUE[s["name"]]["fraction"]
        for want in (s["train_run_id"], "%d/%d" % (num, den), "`%s`" % s["gate"]["reply"], s["dir"], s["change"]):
            assert want in text, "%s: %r is missing from the rendered block" % (s["name"], want)
    assert "150/551" in text and "113/238" in text
    assert "results/superseded/berturk-tied-head-lr" in text and "results/superseded/berturk-10-epoch-grid" in text
    assert "head_lr: tied to the encoder LR (2e-05) -> 0.0001" in text and "epochs: 10 -> 30" in text
    n = len(berturk.SUPERSEDED)
    assert ("BERTurk had %d more selection runs than Laya: %d in all" % (n, n + 1)) in text
    ref = load(REFERENCE)
    num, den = ref["value"]["fraction"]
    assert "%d/%d" % (num, den) in text and "0.707" in text and ref["status"] in text
    for line in ref["differences"] + ref["caveats"]:
        assert line in text, "the reference check's %r is missing" % line
    assert "berturk_epochs 10" in text and "max_epochs 4" in text, "the epoch-grid deviation is not disclosed"
    assert "not tuned" in text and "2e-05" in text
    for c in _current_record()["candidates"]:
        o = c["objective"]
        assert "%d/%d" % (2 * o["tp"], 2 * o["tp"] + o["fp"] + o["fn"]) in text, "epoch %d is missing" % c["epoch"]
    for e in load(FINAL)["seeds"]:
        assert e["train_run_id"] in text
    assert "/Users/" not in text and "/home/" not in text, "the rendered block holds a local path"
    assert "test" not in text.replace("no BERTurk test number exists", "").replace("test portion", ""), \
        "the rendered block mentions a test number"


def test_load_superseded_refuses_a_changed_history(tmp_path):
    """A history record with one byte changed (a tmp copy of results/superseded/) is refused, naming the file."""
    import shutil
    shutil.copytree(path("results/superseded"), tmp_path / "results" / "superseded")
    got = berturk.load_superseded(str(tmp_path))
    assert [g["entry"]["name"] for g in got] == SUPERSEDED_IDS
    s = berturk.SUPERSEDED[0]
    target = tmp_path / s["record"]
    raw = bytearray(target.read_bytes())
    raw[raw.index(b"0.0001") if b"0.0001" in raw else -2] ^= 1
    target.write_bytes(bytes(raw))
    with pytest.raises(SystemExit) as e:
        berturk.load_superseded(str(tmp_path))
    assert s["record"] in str(e.value.code), e.value.code


def test_render_berturk_refuses_a_laya_record_and_another_records_reference():
    import postfit as postfit_cli
    with pytest.raises(SystemExit, match="BERTurk"):
        berturk.render_markdown(load("results/postfit/laya.json"), load(REFERENCE))
    sha = sha256_file(path(RECORD))
    with pytest.raises(SystemExit) as e:
        postfit_cli.main(["--render-berturk", path(RECORD), path(berturk.SUPERSEDED[-1]["reference"])])
    assert isinstance(e.value.code, str) and sha in e.value.code, e.value.code
    with pytest.raises(SystemExit):
        postfit_cli.main(["--render-berturk", path(RECORD), path(REFERENCE), "--out", "x.md"])
