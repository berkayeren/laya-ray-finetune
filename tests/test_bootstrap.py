"""The EVAL-07 paired bootstrap (pilot.metrics.seed_average / bootstrap_weights / replicate_metrics /
paired_bootstrap) and report.py --compare: zero-width self and copy intervals, pairing, seed averaging of
probabilities, determinism for a seed, refusals, and a per-replicate oracle against pilot.metrics.metrics.
No model, torch or Ray: dump-level cases run score.py baselines on the synthetic snapshot (conftest).
"""
import io
import json
import os
import shutil

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import prepared
from intents import INTENTS
from pilot import dumps, metrics

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)


@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path) of the synthetic snapshot prepared under tmp_path."""
    return prepared(tmp_path, synthetic_raw)


def _score_both(snap, root, model="prior"):
    """Score `model` on the synthetic val and test splits under root; return (run_dir, manifest)."""
    out_dir, manifest = snap
    for name in ("val", "test"):
        score.main([os.path.join(out_dir, "splits", "%s.jsonl" % name), "--model", model, "--manifest", manifest,
                    "--out-root", str(root)])
    return os.path.join(str(root), model), manifest


def _assert_zero_width(rep):
    assert set(rep["thresholds"]) == {"0.5", "val_selected"}
    for t, r in rep["thresholds"].items():
        assert r["B"] == rep["B"] and r["seed"] == rep["seed"]
        for name, m in r["metrics"].items():
            assert m["ci"] == [0.0, 0.0] and m["width"] == 0.0, (t, name, m)


def test_compare_self_is_zero_width(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    rep = report.compare([run_dir], [run_dir], "test", manifest, B=200, seed=0)
    assert rep["kind"] == "paired_bootstrap" and rep["n"] == 7 and rep["B"] == 200
    assert set(rep["thresholds"]["0.5"]["metrics"]) == {"micro_f1", "macro_f1", "exact_match", "top1_hit"}
    _assert_zero_width(rep)


# --- toy inputs: pairing, seed averaging, determinism, the per-replicate oracle ----------------------------------

def _toy(n=40, seed=7):
    """(Y bool [n, 48], P float [n, 48]) with every row tagged or not at random and noisy probabilities."""
    rng = np.random.default_rng(seed)
    Y = rng.random((n, 48)) < 0.08
    P = np.clip(0.6 * Y + rng.normal(0.2, 0.2, size=(n, 48)), 0.0, 1.0)
    return Y, P


def test_different_systems_have_width():
    Y, P = _toy(n=20)
    Q = P.copy()
    for i in (2, 9, 15):                                     # B differs from A on 3 of 20 rows
        Q[i] = np.where(Y[i], 0.0, 1.0)
    r = metrics.paired_bootstrap(Y, P, 0.5, Q, 0.5, B=500, seed=0)
    f = r["metrics"]["micro_f1"]
    assert f["delta"] != 0.0 and f["width"] > 0.0, f
    same = metrics.paired_bootstrap(Y, P, 0.5, P.copy(), 0.5, B=500, seed=0)
    assert all(m["ci"] == [0.0, 0.0] for m in same["metrics"].values())


def test_seed_average_is_identity_for_one_seed():
    _, P = _toy()
    avg = metrics.seed_average([P])
    assert avg.dtype == np.float64 and np.array_equal(avg, P)


def test_seed_average_averages_probabilities():
    Y = np.zeros((1, 48), dtype=bool)
    Y[0, 3] = True
    p0, p1 = np.zeros((1, 48)), np.zeros((1, 48))
    p0[0, 3], p1[0, 3] = 0.4, 0.7
    avg = metrics.seed_average([p0, p1])
    assert avg[0, 3] == pytest.approx(0.55, abs=1e-15)
    tag = TAGS[3]
    assert metrics.metrics(Y, avg, 0.5)["per_tag"][tag]["n_pred"] == 1     # the averaged probability crosses 0.5
    assert metrics.metrics(Y, p0, 0.5)["per_tag"][tag]["n_pred"] == 0     # seed 0 alone does not
    assert metrics.metrics(Y, avg, 0.5)["micro_f1"] == 1.0
    with pytest.raises(SystemExit, match="shapes"):
        metrics.seed_average([p0, np.zeros((2, 48))])


def test_same_seed_reproduces_and_other_seed_differs():
    Y, P = _toy(n=60)
    _, Q = _toy(n=60, seed=8)
    r0 = metrics.paired_bootstrap(Y, P, 0.5, Q, 0.5, B=300, seed=0)
    assert metrics.paired_bootstrap(Y, P, 0.5, Q, 0.5, B=300, seed=0) == r0
    r1 = metrics.paired_bootstrap(Y, P, 0.5, Q, 0.5, B=300, seed=1)
    assert r1["seed"] == 1 and r1["metrics"]["micro_f1"]["ci"] != r0["metrics"]["micro_f1"]["ci"]
    idx0, _ = metrics.bootstrap_weights(60, 300, 0)
    assert np.array_equal(idx0, np.random.default_rng(0).integers(0, 60, size=(300, 60)))


def test_bootstrap_weights_count_the_draw():
    idx, W = metrics.bootstrap_weights(7, 5, 2)
    assert idx.shape == W.shape == (5, 7) and W.dtype == np.float64
    assert (W.sum(axis=1) == 7).all()
    for b in range(5):
        assert np.array_equal(W[b], np.bincount(idx[b], minlength=7))


def test_replicates_match_direct_recomputation():
    n, B = 30, 50
    Y, P = _toy(n=n, seed=11)
    rare = 7                                                 # one positive and never predicted: lost in some replicates
    Y[:, rare] = False
    Y[4, rare] = True
    P[:, rare] = 0.0
    Y[[0, 1]] = False                                        # no-intent rows: top-1 and exact match see them
    theta = np.full(48, 0.5)
    theta[:10] = 0.35
    pred = P >= theta
    idx, W = metrics.bootstrap_weights(n, B, 3)
    rep = metrics.replicate_metrics(W, Y, pred, P)
    lost = 0
    for b in range(B):
        m = metrics.metrics(Y[idx[b]], P[idx[b]], theta)
        lost += m["per_tag"][TAGS[rare]]["support"] == 0
        for name in metrics.BOOT_METRICS:
            want = m[name]
            got = rep[name][b]
            if want is None:                                 # undefined: macro -> NaN, top-1 -> 0.0
                assert np.isnan(got) if name == "macro_f1" else got == 0.0, (b, name, got)
            else:
                assert abs(got - want) <= 1e-12, (b, name, got, want)
    assert 0 < lost < B                                      # the zero-support macro rule was exercised


def test_replicate_with_zero_denominators_scores_zero():
    Y = np.zeros((2, 48), dtype=bool)
    pred = np.zeros((2, 48), dtype=bool)
    rep = metrics.replicate_metrics(np.array([[2.0, 0.0]]), Y, pred, np.zeros((2, 48)))
    assert rep["micro_f1"][0] == 0.0 and rep["top1_hit"][0] == 0.0 and rep["exact_match"][0] == 1.0
    assert np.isnan(rep["macro_f1"][0])
    r = metrics.paired_bootstrap(Y, np.zeros((2, 48)), 0.5, np.zeros((2, 48)), 0.5, B=20, seed=0)
    assert r["metrics"]["macro_f1"]["n_undefined"] == 20 and r["metrics"]["macro_f1"]["ci"] is None


def test_bootstrap_refuses_empty_inputs():
    Y, P = _toy(n=5)
    with pytest.raises(SystemExit, match="no seed dumps"):
        metrics.seed_average([])
    with pytest.raises(SystemExit, match="B >= 1"):
        metrics.paired_bootstrap(Y, P, 0.5, P, 0.5, B=0, seed=0)
    with pytest.raises(SystemExit, match="no messages"):
        metrics.paired_bootstrap(Y[:0], P[:0], 0.5, P[:0], 0.5, B=10, seed=0)
    with pytest.raises(SystemExit, match="must both be"):
        metrics.paired_bootstrap(Y, P, 0.5, P[:3], 0.5, B=10, seed=0)


# --- dump level: copies, seeds, pairing and refusals through report.compare --------------------------------------

def _rename_run(run_dir, dst):
    """Copy run_dir to dst and set every sidecar's run_id to dst's name (load_dump ties run_id to the dir name)."""
    shutil.copytree(run_dir, dst)
    for split in ("val", "test"):
        path = os.path.join(dst, split + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                side = json.load(f)
            side["run_id"] = os.path.basename(dst)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(side, f)
    return dst


def test_identical_copy_is_zero_width(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    copy = _rename_run(run_dir, str(tmp_path / "copies" / "prior-copy"))
    rep = report.compare([run_dir], [copy], "test", manifest, B=300, seed=0)
    assert rep["a"]["run_ids"] == ["prior"] and rep["b"]["run_ids"] == ["prior-copy"]
    _assert_zero_width(rep)


def test_compare_averages_seed_dumps(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    seed1 = _rename_run(run_dir, str(tmp_path / "seeds" / "prior-s1"))
    rng = np.random.default_rng(5)
    for split in ("val", "test"):                           # a second "seed" with other logits on both splits
        logits, uids, side, _ = dumps.load_dump(seed1, split, manifest)
        other = rng.normal(0.0, 2.0, size=logits.shape).astype(np.float32)
        dumps.write_dump(seed1, split, other, uids, {k: v for k, v in side.items()
                                                     if k not in ("dump_sha256", "schema_version")})
    rep = report.compare([run_dir, seed1], [run_dir], "test", manifest, B=100, seed=0)
    assert rep["a"]["run_ids"] == ["prior", "prior-s1"] and len(rep["a"]["dump_sha256"]["split"]) == 2
    L0, uids, _, entry = dumps.load_dump(run_dir, "test", manifest)
    L1 = dumps.load_dump(seed1, "test", manifest)[0]
    Y = dumps.read_gold(entry, manifest, uids)
    avg = (metrics.probabilities(L0, 1.0) + metrics.probabilities(L1, 1.0)) / 2
    assert rep["thresholds"]["0.5"]["metrics"]["micro_f1"]["a"] == metrics.metrics(Y, avg, 0.5)["micro_f1"]
    one = report.compare([run_dir], [run_dir], "test", manifest, B=100, seed=0)
    assert one["a"]["run_ids"] == ["prior"]                 # a single seed is the identity
    with pytest.raises(SystemExit, match="no seed dumps"):
        report.compare([], [run_dir], "test", manifest, B=100, seed=0)


def _swap_two_uids(run_dir, split):
    """Swap the first two uids of run_dir/<split>.npz, mark the dump a full-size subset so load_dump accepts it."""
    z = np.load(os.path.join(run_dir, split + ".npz"), allow_pickle=False)
    uids = z["uids"].copy()
    uids[[0, 1]] = uids[[1, 0]]
    buf = io.BytesIO()
    np.savez(buf, logits=z["logits"], uids=uids)
    data = buf.getvalue()
    with open(os.path.join(run_dir, split + ".npz"), "wb") as f:
        f.write(data)
    path = os.path.join(run_dir, split + ".json")
    with open(path, encoding="utf-8") as f:
        side = json.load(f)
    side.update(dump_sha256=pilot.data.sha256_bytes(data), n_subset=len(uids))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(side, f)


@pytest.mark.parametrize("split", ["test", "val"])
def test_compare_refuses_unpaired(tmp_path, snap, split):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    other = _rename_run(run_dir, str(tmp_path / "other" / "prior-b"))
    _swap_two_uids(other, split)
    dumps.load_dump(other, split, manifest)                 # the dump itself verifies: only pairing can catch it
    with pytest.raises(SystemExit) as e:
        report.compare([run_dir], [other], "test", manifest, B=50, seed=0)
    assert str(e.value.code).startswith("unpaired: "), e.value.code     # the message, not a tmp_path that names it


def test_compare_refuses_repeated_seed(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    elsewhere = shutil.copytree(run_dir, str(tmp_path / "elsewhere" / "prior"))
    for side in ([run_dir, run_dir], [run_dir, elsewhere]):  # the same dir twice, or the same run_id twice
        with pytest.raises(SystemExit) as e:
            report.compare(side, [run_dir], "test", manifest, B=50, seed=0)
        assert "more than once" in str(e.value.code), e.value.code
        with pytest.raises(SystemExit):
            report.compare([run_dir], side, "test", manifest, B=50, seed=0)


def test_compare_refuses_subset_or_missing_val(tmp_path, snap):
    out_dir, manifest = snap
    root = tmp_path / "sub"
    score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior", "--manifest", manifest,
                "--n", "3", "--out-root", str(root)])
    with pytest.raises(SystemExit, match="full val dump"):   # no val dump yet
        report.compare([str(root / "prior")], [str(root / "prior")], "test", manifest, B=50, seed=0)
    score.main([os.path.join(out_dir, "splits", "val.jsonl"), "--model", "prior", "--manifest", manifest,
                "--out-root", str(root)])
    with pytest.raises(SystemExit, match="--n"):             # a subset test dump, even compared with itself
        report.compare([str(root / "prior")], [str(root / "prior")], "test", manifest, B=50, seed=0)


def test_compare_records_provenance_and_no_decision(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    maj, _ = _score_both(snap, tmp_path / "dumps", model="majority")
    rep = report.compare([maj], [run_dir], "test", manifest, B=100, seed=4)
    text = json.dumps(rep, ensure_ascii=False)
    for word in ("verdict", "GO", "PASS", "FAIL"):
        assert word not in text.replace("GOLD", ""), word
    assert rep["B"] == 100 and rep["seed"] == 4 and rep["split"] == "test" and rep["theta_in_sample"] is False
    assert rep["procedure"].startswith("EVAL-07") and "protocol.json" in rep["theta_rule"] and rep["note"]
    _, _, side, _ = dumps.load_dump(maj, "test", manifest)
    _, _, vside, _ = dumps.load_dump(maj, "val", manifest)
    assert rep["a"]["dump_sha256"] == {"split": [side["dump_sha256"]], "val": [vside["dump_sha256"]]}
    assert rep["a"]["temperature"] == {"split": [1.0], "val": [1.0]} and "global" in rep["a"]["theta"]
    for r in rep["thresholds"].values():
        assert r["method"] == "percentile (numpy linear interpolation)" and r["level"] == 0.95 and r["n"] == 7


def test_cli_compare(tmp_path, snap, capsys):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    seed1 = _rename_run(run_dir, str(tmp_path / "seeds" / "prior-s1"))
    out = tmp_path / "cmp.json"
    rep = report.main(["--compare", run_dir + "," + seed1, run_dir, "--manifest", manifest, "--B", "50",
                       "--seed", "2", "--out", str(out)])
    with open(str(out), encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(rep))
    assert rep["B"] == 50 and rep["seed"] == 2 and rep["a"]["run_ids"] == ["prior", "prior-s1"]
    for argv in ([], [run_dir, "--compare", run_dir, run_dir], ["--compare", run_dir + ",", run_dir]):
        with pytest.raises(SystemExit) as e:
            report.main(argv + ["--manifest", manifest])
        assert e.value.code == 2
    capsys.readouterr()
