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
"""
import json
import os
import re
import subprocess

import pytest

from pilot import berturk, compute, dumps, postfit
from pilot.data import DEFAULT_MANIFEST

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPS = "results/dumps"
SELECTION_RUN_RE = re.compile(r"^laya-selection_train-neg8-s0-[0-9a-f]{8}$")


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
