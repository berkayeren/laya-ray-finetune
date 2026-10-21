"""Committed Phase 5 Laya records stay consistent with the protocol in pilot/postfit.py (LAYA-08, T-05-32, T-05-34).

Plan 05-06 adds the selection-grid guard; plans 05-07, 05-08 and 05-09 add the postfit-record, final-manifest and
README tests.

- test_single_lr_protocol_constants: the 05-06 user override (2026-10-15) is in force: one encoder LR (LR_GRID
  (2e-05,)), head LR 0.0001, epochs 1..4, selection seed 0 and final seeds 0, 1, 2, and protocol("laya") is exactly
  those constants; BERTurk shares the one LR (A1). The approved compute budget (pilot.compute.MATRIX lr_grid 3,
  results/compute/) is not rewritten: Laya runs one of its three approved LR runs, a deviation the README discloses
- test_laya_selection_dumps_are_the_single_lr_grid: the tracked Laya selection val dumps (system.kind laya, training
  split selection_train; the zero-shot dump has no training block and the kind filter keeps the BERTurk dumps out) are
  exactly the grid postfit reads: one train run, one full val dump per (LR, epoch) of LR_GRID x 1..EPOCHS, seed 0,
  uncalibrated (shipped T 1.0), one rung and one canonical recipe, every dump verified (pilot.dumps.load_dump), and no
  test dump. A dump at any other encoder LR (3e-05, 5e-05) fails it
- test_committed_record_is_complete_and_consistent (plan 05-07): RECORD (results/postfit/laya.json, postfit/1, system
  kind laya) is tracked; its protocol and rules are pilot.postfit's constants; its candidates cover LR_GRID x
  1..EPOCHS once each and are exactly the tracked Laya selection dumps, each verified with its dump_sha256 and train run
  id and with no test dump; its chosen candidate is what pilot.postfit.select returns on the recorded candidates; T lies
  in [T_MIN, T_MAX]; stop_after_epoch is None exactly at epoch EPOCHS; the final block is final_train with FINAL_SEEDS
- test_makefile_lists_the_selection_dumps: the Makefile's LAYA_SELECTION is the record's candidate dump run ids in
  record order, and `make laya-postfit` rebuilds RECORD from results/dumps/$(LAYA_SELECTION)
- test_record_reproduces_from_the_committed_dumps (data): pilot.postfit.fit over the candidate dirs, with no grid
  arguments (the protocol dispatches by system kind), rebuilds RECORD byte for byte
- test_committed_final_manifest (plan 05-08): FINAL (results/postfit/laya-final.json, postfit-final/1) is tracked and
  bound to RECORD (its path and sha256, its chosen settings, the single encoder LR 2e-05 == LR_GRID[0]); it lists
  final_train and exactly seeds 0, 1, 2 (never only the best), with three distinct train run ids carrying their seed and
  one config_sha256_sans_seed; every seed serves the chosen T, has one row per epoch up to the chosen epoch, ran at the
  chosen rung, flags code_changed_on_resume as a bool, names a bare checkpoint dir and carries the dump run id of its
  checkpoint identity; no tracked results/dumps dir is named after a final seed's dump run id (none was scored in
  Phase 5)
- test_readme_postfit_block_matches_the_record (plan 05-09, LAYA-08): README.md holds exactly one pair of the
  laya-postfit markers, and the block between them equals pilot.postfit.render_markdown of the committed record and
  final manifest, so the reported chosen settings cannot be edited by hand or go stale
- test_render_discloses_the_selection: the rendered block names the single encoder LR 2e-05 against the approved
  lr_grid of pilot.compute.MATRIX (fixed, not tuned), the grid-edge verdict of the chosen epoch (never "interior
  optimum" while a candidate's shipped θ is clipped at a θ-grid edge), the θ-edge count over every candidate
  (05-REVIEW CR-01, WR-04), every candidate's exact objective fraction and every final seed's train run id, and holds
  no local path
- test_render_names_an_interior_optimum_only_without_a_clipped_theta: a copy of the record with every global θ inside
  the grid is an interior optimum without the margin-cutoff disclosure; the committed record is the reverse
- test_render_refuses_a_berturk_record: render_markdown refuses the BERTurk record (BERTurk has its own renderer)
- test_render_refuses_a_manifest_of_another_record: postfit.py --render of the Laya record with the BERTurk final
  manifest is refused with a message naming the Laya record's sha256 (not an argparse usage error)
"""
import glob
import json
import os
import re
import subprocess

import pytest

from pilot import berturk, compute, dumps, metrics, postfit
from pilot.data import DEFAULT_MANIFEST, sha256_file

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPS = "results/dumps"
SELECTION_RUN_RE = re.compile(r"^laya-selection_train-neg8-s0-[0-9a-f]{8}$")
RECORD = "results/postfit/laya.json"


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


def laya_selection_sidecars():
    """{dump dir name: parsed val.json} of every tracked Laya selection val dump: system.kind laya with a training
    block on selection_train (the kind is filtered first: BERTurk selection dumps carry that split too)."""
    out = {}
    for rel in tracked(DUMPS + "/"):
        if not rel.endswith("/val.json") or not os.path.isfile(path(rel)):
            continue
        side = load(rel)
        if (side.get("system") or {}).get("kind") != "laya":
            continue
        tr = side.get("training")
        if isinstance(tr, dict) and tr.get("split") == postfit.SELECTION_SPLIT:
            out[rel.split("/")[2]] = side
    return out


def test_single_lr_protocol_constants():
    # the 05-06 user override (2026-10-15, restated 2026-10-21): one encoder LR, fixed, not tuned
    assert postfit.LR_GRID == (2e-05,), postfit.LR_GRID
    assert postfit.HEAD_LR == 0.0001, postfit.HEAD_LR
    assert postfit.EPOCHS == 4, postfit.EPOCHS
    assert postfit.SELECTION_SEED == 0, postfit.SELECTION_SEED
    assert postfit.FINAL_SEEDS == (0, 1, 2), postfit.FINAL_SEEDS
    assert postfit.protocol("laya") == {"lr_grid": postfit.LR_GRID, "epochs": postfit.EPOCHS,
                                        "head_lr": postfit.HEAD_LR, "selection_seed": postfit.SELECTION_SEED,
                                        "final_seeds": postfit.FINAL_SEEDS}
    assert berturk.LR_GRID == postfit.LR_GRID, "A1: BERTurk shares Laya's one LR (%s vs %s)" % (berturk.LR_GRID,
                                                                                             postfit.LR_GRID)
    assert compute.MATRIX["lr_grid"] == 3, (
        "pilot.compute.MATRIX lr_grid is %r; the approved budget (plan 04-08) stays recorded as approved (3), and "
        "Laya runs one of its three LR runs (05-06 override, disclosed in the README)" % (compute.MATRIX["lr_grid"],))


def test_laya_selection_dumps_are_the_single_lr_grid():
    sides = laya_selection_sidecars()
    runs = {side["training"]["run_id"] for side in sides.values()}
    assert len(runs) == 1, "results/dumps holds Laya selection dumps of %s, not of exactly one train run" % sorted(runs)
    (run,) = runs
    assert SELECTION_RUN_RE.match(run), "Laya selection train run id %r does not match %s" % (run,
                                                                                            SELECTION_RUN_RE.pattern)

    pairs = sorted((side["training"]["encoder_lr"], side["training"]["epoch"]) for side in sides.values())
    want = [(lr, e) for lr in postfit.LR_GRID for e in range(1, postfit.EPOCHS + 1)]
    assert pairs == want, ("the Laya selection dumps cover (encoder LR, epoch) %s, not exactly the single-LR grid %s "
                           "(dirs %s)" % (pairs, want, sorted(sides)))
    assert len(sides) == len(postfit.LR_GRID) * postfit.EPOCHS == 4, sorted(sides)

    rungs, canons = set(), set()
    for d, side in sorted(sides.items()):
        tr = side["training"]
        assert side["seed"] == postfit.SELECTION_SEED, "%s: seed %r" % (d, side["seed"])
        assert side["n_subset"] is None, "%s: n_subset %r (a --n subset dump)" % (d, side["n_subset"])
        assert tr["calibrated"] is False, "%s: calibrated %r" % (d, tr["calibrated"])
        assert side["temperature"]["shipped"] == 1.0, "%s: shipped T %r" % (d, side["temperature"]["shipped"])
        assert tr["epochs_schedule"] == postfit.EPOCHS, "%s: epochs_schedule %r" % (d, tr["epochs_schedule"])
        canon = tr["canonical"]
        assert canon["split"] == postfit.SELECTION_SPLIT, "%s: canonical split %r" % (d, canon["split"])
        assert canon["encoder_lr"] == tr["encoder_lr"], "%s: canonical encoder_lr %r != block encoder_lr %r" % (
            d, canon["encoder_lr"], tr["encoder_lr"])
        assert canon["head_lr"] == postfit.HEAD_LR, "%s: canonical head_lr %r" % (d, canon["head_lr"])
        assert canon["epochs"] == postfit.EPOCHS, "%s: canonical epochs %r" % (d, canon["epochs"])
        rungs.add(tr["rung"])
        canons.add(json.dumps(as_json(canon), sort_keys=True))

        logits, uids, verified, _ = dumps.load_dump(path(DUMPS + "/" + d), "val", DEFAULT_MANIFEST)
        assert verified["run_id"] == d, "%s: verified run_id %r" % (d, verified["run_id"])
        assert verified["dump_sha256"] == side["dump_sha256"], "%s: dump_sha256 %r != sidecar %r" % (
            d, verified["dump_sha256"], side["dump_sha256"])
        assert logits.shape == (len(uids), 48), "%s: logits %s" % (d, logits.shape)
        held = tracked("%s/%s/test.npz" % (DUMPS, d), "%s/%s/test.json" % (DUMPS, d))
        assert not held, "%s holds a tracked test dump %s; no fine-tuned Laya is scored on test before Phase 8" % (
            d, held)

    assert len(rungs) == 1, "the Laya selection dumps ran at rungs %s, not one" % sorted(rungs)
    assert len(canons) == 1, "the Laya selection dumps carry %d canonical configs, not one recipe" % len(canons)


def _laya_record():
    """The parsed committed Laya postfit record; it must be tracked (plan 05-07), so a missing record fails the caller."""
    assert tracked(RECORD) == [RECORD], "%s is not tracked; build it with `make laya-postfit` and commit it" % RECORD
    return load(RECORD)


def test_committed_record_is_complete_and_consistent():
    r = _laya_record()
    proto = postfit.protocol("laya")
    assert r["schema"] == postfit.SCHEMA == "postfit/1"
    assert r["system_kind"] == "laya"
    for key in ("lr_grid", "head_lr", "epochs", "selection_seed", "final_seeds"):
        assert as_json(r["protocol"][key]) == as_json(proto[key]), key
    assert r["protocol"]["selection_split"] == "selection_train" and r["protocol"]["final_split"] == "final_train"

    rules = r["rules"]
    assert rules["temperature"] == postfit.TEMPERATURE_RULE
    assert rules["temperature_bounds"] == [postfit.T_MIN, postfit.T_MAX]
    assert rules["objective"] == postfit.OBJECTIVE and rules["tie_order"] == postfit.TIE_ORDER
    assert rules["epoch_transfer"] == postfit.EPOCH_TRANSFER
    assert rules["theta"]["tie_break"] == metrics.TIE_RULE
    assert rules["theta"]["min_val_positives"] == metrics.MIN_VAL_POSITIVES

    cands = r["candidates"]
    pairs = [(c["encoder_lr"], c["epoch"]) for c in cands]
    want = [(lr, e) for lr in postfit.LR_GRID for e in range(1, postfit.EPOCHS + 1)]
    assert sorted(pairs) == want and len(set(pairs)) == len(pairs), pairs
    assert {c["dump_run_id"] for c in cands} == set(laya_selection_sidecars()), \
        "the record's candidates are not exactly the tracked Laya selection dumps"
    assert all(c["system_kind"] == "laya" for c in cands)
    files = set(tracked(DUMPS + "/"))
    for c in cands:
        rid = c["dump_run_id"]
        for name in ("val.npz", "val.json"):
            assert "%s/%s/%s" % (DUMPS, rid, name) in files, "%s/%s is not tracked" % (rid, name)
        for name in ("test.npz", "test.json"):
            assert "%s/%s/%s" % (DUMPS, rid, name) not in files, "%s holds a tracked test dump" % rid
        logits, uids, side, entry = dumps.load_dump(path(DUMPS + "/" + rid), "val", DEFAULT_MANIFEST)
        assert side["run_id"] == rid and side["dump_sha256"] == c["dump_sha256"], rid
        assert side["training"]["run_id"] == c["train_run_id"] and side["n_subset"] is None, rid
        assert entry["role"] == "val" and logits.shape == (len(uids), 48), rid

    chosen = postfit.select(cands, **{k: v for k, v in proto.items() if k != "final_seeds"})
    c = r["chosen"]
    assert c["dump_run_id"] == chosen["dump_run_id"] and c["dump_sha256"] == chosen["dump_sha256"]
    assert c["epoch"] == chosen["epoch"] and as_json(c["theta"]) == as_json(chosen["theta"])
    assert c["temperature"] == chosen["temperature"]
    assert postfit.T_MIN <= c["temperature"] <= postfit.T_MAX
    if c["epoch"] == postfit.EPOCHS:
        assert c["stop_after_epoch"] is None
    else:
        assert c["stop_after_epoch"] == c["epoch"]
    for key in ("rung", "fallback", "weaker"):
        assert all(as_json(x[key]) == as_json(c[key]) for x in cands), key
    f = r["final"]
    assert f["split"] == "final_train" and f["seeds"] == list(postfit.FINAL_SEEDS)
    assert f["epochs"] == postfit.EPOCHS and f["stop_after_epoch"] == c["stop_after_epoch"]


def test_makefile_lists_the_selection_dumps():
    record = _laya_record()
    with open(path("Makefile"), encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^LAYA_SELECTION := (.+)$", text, re.M)
    assert m, "Makefile has no LAYA_SELECTION := line"
    assert m.group(1).split() == [c["dump_run_id"] for c in record["candidates"]], \
        "LAYA_SELECTION is not the record's candidate dump run ids in record order"
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "laya-postfit" in phony.group(1).split(), "laya-postfit is not in .PHONY"
    recipe = re.search(r"^laya-postfit:\n((?:\t.*\n)+)", text, re.M)
    assert recipe, "Makefile has no laya-postfit target"
    assert ("$(PYTHON) postfit.py $(addprefix results/dumps/,$(LAYA_SELECTION)) --out %s" % RECORD) \
        in recipe.group(1), "laya-postfit does not rebuild %s from results/dumps/$(LAYA_SELECTION)" % RECORD


@pytest.mark.data
def test_record_reproduces_from_the_committed_dumps():
    record = _laya_record()
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")
    dirs = [path("%s/%s" % (DUMPS, c["dump_run_id"])) for c in record["candidates"]]
    got = postfit.fit(dirs, DEFAULT_MANIFEST)       # no grid: postfit.fit dispatches the protocol by system kind
    with open(path(RECORD), "rb") as f:
        assert postfit.record_bytes(got) == f.read(), "%s does not rebuild from the committed dumps" % RECORD


FINAL = "results/postfit/laya-final.json"
FINAL_RUN_RE = re.compile(r"^laya-final_train-neg8-s([012])-[0-9a-f]{8}$")


def test_committed_final_manifest():
    """LAYA-08 (plan 05-08): the committed manifest of the 3 Laya final seeds is bound to the committed record (its
    sha256 and chosen settings, the 4-epoch schedule stopped at the chosen epoch, the single encoder LR 2e-05); it lists
    all three seeds (never only the best), each serving the chosen T at the chosen rung, and no final seed has a tracked
    dump (none was scored in Phase 5)."""
    assert tracked(FINAL) == [FINAL], "%s is not tracked" % FINAL
    record = _laya_record()
    m = load(FINAL)
    assert m["schema"] == postfit.FINAL_SCHEMA == "postfit-final/1", m["schema"]
    assert m["postfit"] == {"path": RECORD, "sha256": sha256_file(path(RECORD))}, \
        "%s is not bound to the committed %s" % (FINAL, RECORD)
    keys = ("train_run_id", "encoder_lr", "epoch", "epochs_schedule", "stop_after_epoch", "temperature")
    chosen = record["chosen"]
    assert as_json(m["chosen"]) == as_json({k: chosen[k] for k in keys}), "the manifest's chosen settings differ"
    assert chosen["encoder_lr"] == 2e-05 == postfit.LR_GRID[0], chosen["encoder_lr"]
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
        s = e["seed"]
        assert e["temperature"] == chosen["temperature"], "seed %d serves T %r" % (s, e["temperature"])
        assert len(e["epochs"]) == len(e["loss_by_epoch"]) == chosen["epoch"], "seed %d epochs" % s
        assert [r["epoch"] for r in e["epochs"]] == list(range(1, chosen["epoch"] + 1)), "seed %d epoch rows" % s
        for key in ("rung", "fallback", "weaker"):
            assert as_json(e[key]) == as_json(chosen[key]), "seed %d %s %r, chosen %r" % (s, key, e[key], chosen[key])
        assert isinstance(e["code_changed_on_resume"], bool), "seed %d code_changed_on_resume %r" % (
            s, e["code_changed_on_resume"])
        assert "/" not in e["checkpoint_name"] and "\\" not in e["checkpoint_name"], e["checkpoint_name"]
        assert e["dump_run_id"] == dumps.run_id("laya", e["checkpoint_identity"]), "seed %d dump_run_id" % s
    scored = sorted({f.split("/")[2] for f in tracked(DUMPS + "/")} & {e["dump_run_id"] for e in seeds})
    assert scored == [], "a Laya final seed was scored in Phase 5: %s" % scored


README = "README.md"
POSTFIT_START, POSTFIT_END = "<!-- laya-postfit:start -->", "<!-- laya-postfit:end -->"


def _rendered():
    record = _laya_record()
    return postfit.render_markdown(record, load(FINAL))


def test_readme_postfit_block_matches_the_record():
    with open(path(README), encoding="utf-8") as f:
        text = f.read()
    assert text.count(POSTFIT_START) == 1 and text.count(POSTFIT_END) == 1, \
        "README.md must hold exactly one pair of the laya-postfit markers"
    i, j = text.index(POSTFIT_START), text.index(POSTFIT_END)
    assert i < j, "the laya-postfit end marker comes before the start marker"
    block = text[i + len(POSTFIT_START):j]
    assert block.strip() == _rendered().strip(), (
        "the README laya-postfit block differs from pilot.postfit.render_markdown of %s and %s; regenerate it with "
        "`python postfit.py --render %s %s`" % (RECORD, FINAL, RECORD, FINAL))


def test_render_discloses_the_selection():
    record = _laya_record()
    text = _rendered()
    assert "2e-05" in text and "not tuned" in text, "the single fixed encoder LR is not disclosed"
    assert "lr_grid %d" % compute.MATRIX["lr_grid"] in text, "the approved lr_grid is not named"
    epoch, top = record["chosen"]["epoch"], record["protocol"]["epochs"]
    up, lo = postfit.theta_edges(record["candidates"], "shipped")
    clipped = bool(up or lo)
    edge = ("upper edge" if epoch == top else "lower edge" if epoch == 1 else
            "interior in the epoch grid" if clipped else "interior optimum")
    assert edge in text, "the grid-edge verdict %r is missing" % edge
    assert ("interior optimum" in text) is (not clipped and 1 < epoch < top), \
        "an epoch is called an interior optimum while a candidate's θ is clipped at a θ-grid edge (05-REVIEW CR-01)"
    n = len(record["candidates"])
    assert "%d of %d candidates choose the shipped global θ on the upper edge" % (len(up), n) in text, \
        "the θ-edge count over every candidate is missing (05-REVIEW WR-04)"
    assert ("T-dependent margin cutoff" in text) is clipped
    assert "max_epochs %d" % compute.MATRIX["max_epochs"] in text, "the approved max_epochs is not named"
    for c in record["candidates"]:
        s = postfit.score(c)
        assert "%d/%d" % (s.numerator, s.denominator) in text, "candidate epoch %d's fraction is missing" % c["epoch"]
    for e in load(FINAL)["seeds"]:
        assert e["train_run_id"] in text, "final seed %d's train run id is missing" % e["seed"]
    assert "/Users/" not in text and "/home/" not in text, "the rendered block holds a local path"


def test_render_names_an_interior_optimum_only_without_a_clipped_theta():
    """The committed record (every shipped θ on the upper edge) and a copy with every global θ inside the grid: only
    the copy is called an interior optimum, and only the committed one gets the margin-cutoff disclosure."""
    import copy
    record = _laya_record()
    inside = copy.deepcopy(record)
    for c in inside["candidates"]:
        for name in ("shipped", "raw"):
            c["theta"][name]["global"] = 0.5
    inside["chosen"]["theta"] = inside["candidates"][[c["dump_run_id"] for c in inside["candidates"]].index(
        inside["chosen"]["dump_run_id"])]["theta"]
    text = postfit.render_markdown(inside)
    assert "interior optimum" in text and "0 of 4 candidates" in text and "margin cutoff" not in text
    clipped = postfit.render_markdown(record)
    assert "interior optimum" not in clipped and "4 of 4 candidates" in clipped and "margin cutoff" in clipped


def test_render_refuses_a_berturk_record():
    with pytest.raises(SystemExit, match="laya"):
        postfit.render_markdown(load("results/postfit/berturk.json"))


def test_render_refuses_a_manifest_of_another_record():
    import postfit as postfit_cli
    sha = sha256_file(path(RECORD))
    with pytest.raises(SystemExit) as e:
        postfit_cli.main(["--render", path(RECORD), path("results/postfit/berturk-final.json")])
    assert isinstance(e.value.code, str) and sha in e.value.code, e.value.code
