"""EVAL-03 calibration definitions and the report.py --confidence binding, offline: no model, no Ray.

The definitions in pilot.metrics are pinned to independent oracles on seeded random arrays: Brier to sklearn's
brier_score_loss and NLL to sklearn's log_loss on the flattened (message, tag) pairs, and every tag's classwise ECE
to torchmetrics' binary_calibration_error(n_bins=15, norm="l1") (probabilities drawn strictly inside (0, 1), because
torchmetrics puts an exact 1.0 into an extra bin). The bin edges, the NLL clipping, the zero-support tags in the
48-tag mean, the reliability counts and the empty-input refusals are pinned on hand-built arrays.

The --confidence binding (where T and θ come from) is exercised on synthetic dumps of the conftest snapshot: selection
candidates written like score.py's (tests/test_postfit.py write_candidate) with a postfit/1 record built by postfit.py
over them, final seeds bound by training.postfit_sha256, and prior baselines scored by score.py. Every wrong
combination is refused before a metric is computed; `import report` stays numpy only.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

import pilot.data
import postfit as postfit_cli
import report
import score
from conftest import prepared
from intents import INTENTS
from pilot import dumps, metrics
from pilot.data import REPO_ROOT
from test_postfit import GRID, write_candidate

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)
TOL = 1e-12


def _draw(n=120, seed=7, lo=0.001, hi=0.999, rate=0.15):
    rng = np.random.default_rng(seed)
    return rng.random((n, 48)) < rate, rng.uniform(lo, hi, size=(n, 48))


# --- definitions against oracles -------------------------------------------------------------------------------

def test_brier_equals_sklearn_on_flattened_pairs():
    Y, P = _draw()
    assert abs(metrics.brier(Y, P) - brier_score_loss(Y.ravel(), P.ravel())) <= TOL


def test_nll_equals_sklearn_log_loss():
    Y, P = _draw()
    assert abs(metrics.nll(Y, P) - log_loss(Y.ravel(), P.ravel(), labels=[0, 1])) <= TOL


def test_nll_is_finite_at_zero_and_one():
    Y, P = _draw(n=4)
    P[0, 0], P[1, 1] = 0.0, 1.0
    Y[0, 0], Y[1, 1] = True, False                          # the worst case at both edges: clipped, not inf
    v = metrics.nll(Y, P)
    assert np.isfinite(v) and v > 0
    P2 = P.copy()
    P2[0, 0], P2[1, 1] = metrics.NLL_EPS, 1.0 - metrics.NLL_EPS
    assert v == metrics.nll(Y, P2)


def test_classwise_ece_per_tag_equals_torchmetrics():
    torch = pytest.importorskip("torch")
    from torchmetrics.functional.classification import binary_calibration_error
    Y, P = _draw(n=200, seed=11, lo=1e-6, hi=1 - 1e-6, rate=0.2)
    mean, per_tag = metrics.classwise_ece(Y, P)
    assert per_tag.shape == (48,) and per_tag.dtype == np.float64
    for j in range(48):
        want = float(binary_calibration_error(torch.tensor(P[:, j], dtype=torch.float64),
                                              torch.tensor(Y[:, j]).long(), n_bins=15, norm="l1"))
        assert abs(per_tag[j] - want) <= TOL, (TAGS[j], per_tag[j], want)
    assert mean == float(per_tag.mean())


def test_ece_bin_edges_put_one_in_the_last_bin():
    below = np.nextafter(1.0 / 15, 0.0)
    got = metrics.ece_bins(np.array([0.0, 1.0, 1.0 / 15, below, 0.5, np.nextafter(1.0, 0.0)]))
    assert got.tolist() == [0, 14, 1, 0, 7, 14]
    assert metrics.ECE_BINS == 15
    edges = np.linspace(0.0, 1.0, 16)                       # every interior edge opens its bin; just below it does not
    inner = edges[1:-1]
    assert metrics.ece_bins(inner).tolist() == list(range(1, 15))
    assert metrics.ece_bins(np.nextafter(inner, 0.0)).tolist() == list(range(0, 14))


def test_ece_averages_all_48_tags_including_zero_support():
    n = 10
    Y = np.zeros((n, 48), dtype=bool)
    P = np.full((n, 48), 0.02)
    Y[:, 0] = True
    P[:, 0] = 1.0                                           # tag 0 perfectly calibrated (bin 14, all positive)
    mean, per_tag = metrics.classwise_ece(Y, P)
    assert per_tag[0] == 0.0
    assert np.allclose(per_tag[1:], 0.02, rtol=0, atol=TOL)  # a zero-support tag's ECE is its mean p
    assert abs(mean - 47 * 0.02 / 48) <= TOL                # every tag is in the mean, none skipped
    cal = metrics.calibration(Y, P)
    assert cal["per_tag"][TAGS[1]] == {"ece": float(per_tag[1]), "support": 0}
    assert cal["per_tag"][TAGS[0]]["support"] == n


def test_reliability_bins_count_every_pair():
    Y, P = _draw(n=30, seed=3)
    P[:, :] = np.clip(P, 0.0, 0.6)                          # bins above 0.6 stay empty
    pooled = metrics.reliability(Y, P)
    assert len(pooled) == metrics.ECE_BINS
    assert sum(b["count"] for b in pooled) == 30 * 48
    assert pooled[0]["lo"] == 0.0 and pooled[-1]["hi"] == 1.0
    empty = [b for b in pooled if b["count"] == 0]
    assert empty and all(b["mean_p"] is None and b["freq"] is None for b in empty)
    for b in pooled:
        if b["count"]:
            assert b["lo"] <= b["mean_p"] < b["hi"] and 0.0 <= b["freq"] <= 1.0
    one = metrics.reliability(Y, P, tags=[TAGS[5]])
    assert sum(b["count"] for b in one) == 30
    cal = metrics.calibration(Y, P, [TAGS[5], TAGS[9]])
    assert cal["reliability"]["pooled"] == pooled
    assert sorted(cal["reliability"]["per_tag"]) == sorted([TAGS[5], TAGS[9]])
    assert cal["reliability"]["per_tag"][TAGS[5]] == one
    assert cal["n_bins"] == 15 and list(cal["per_tag"]) == TAGS


@pytest.mark.parametrize("fn", [metrics.brier, metrics.nll, lambda Y, P: metrics.classwise_ece(Y, P),
                                metrics.reliability, metrics.calibration])
def test_calibration_refuses_empty_or_misshaped_input(fn):
    with pytest.raises(SystemExit, match="no messages"):
        fn(np.zeros((0, 48), dtype=bool), np.zeros((0, 48)))
    with pytest.raises(SystemExit, match="must both be"):
        fn(np.zeros((3, 47), dtype=bool), np.zeros((3, 47)))
    with pytest.raises(SystemExit, match="must both be"):
        fn(np.zeros((3, 48), dtype=bool), np.zeros((3, 47)))


# --- report.py --confidence: where T and θ come from -----------------------------------------------------------

@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path, val entry, val uids) of the synthetic snapshot prepared under tmp_path."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    path = os.path.join(out_dir, "splits", "val.jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=("val",), data=raw)
    return out_dir, manifest, entry, [r["uid"] for r in pilot.data.parse_jsonl(raw)]


@pytest.fixture
def fitted(tmp_path, snap, capsys):
    """(candidate run dirs, record path, record dict, record sha256): a postfit/1 record over a synthetic grid."""
    run_dirs = [write_candidate(tmp_path / "dumps", snap, lr, e) for lr, e in GRID]
    out = tmp_path / "postfit" / "laya.json"
    postfit_cli.main(run_dirs + ["--manifest", snap[1], "--out", str(out)])
    capsys.readouterr()
    return run_dirs, str(out), json.loads(out.read_text(encoding="utf-8")), pilot.data.sha256_file(str(out))


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _cand(record, run_dir):
    rid = os.path.basename(run_dir)
    return [c for c in record["candidates"] if c["dump_run_id"] == rid][0]


def test_confidence_binding_selection_candidate_uses_record_t_and_theta(snap, fitted):
    run_dirs, path, record, sha = fitted
    for d in run_dirs:
        rep = report.confidence_report(d, "val", snap[1], path)
        c = _cand(record, d)
        head, raw = rep["variants"]["headline"], rep["variants"]["raw"]
        assert rep["binding"]["kind"] == "postfit_candidate" and rep["schema"] == report.CONFIDENCE_SCHEMA
        assert rep["binding"]["postfit"]["sha256"] == sha and rep["binding"]["postfit"]["system_kind"] == "laya"
        assert head["temperature"] == c["temperature"] and head["theta"] == c["theta"]["shipped"]
        assert raw["temperature"] == 1.0 and raw["theta"] == c["theta"]["raw"]
        assert rep["in_sample"] is True and rep["caveat"] == report.IN_SAMPLE_CAVEAT
        assert abs(head["calibration"]["nll"] - c["nll_fitted"]) <= 1e-8
        assert abs(raw["calibration"]["nll"] - c["nll_raw"]) <= 1e-8
        assert rep["definitions"] == {"calibration": metrics.CALIBRATION_RULES}
        assert sorted(head["calibration"]["reliability"]["per_tag"]) == sorted(c["theta"]["shipped"]["eligible_tags"])


def test_confidence_binding_refuses_without_postfit(snap, fitted):
    with pytest.raises(SystemExit, match="ships T = 1; pass --postfit"):
        report.confidence_report(fitted[0][0], "val", snap[1])


def test_confidence_binding_refuses_other_kind_record(tmp_path, snap, fitted):
    other = dict(fitted[2], system_kind="berturk")
    path = _write_json(tmp_path / "other" / "berturk.json", other)
    with pytest.raises(SystemExit, match="is a berturk record, the dump is laya"):
        report.confidence_report(fitted[0][0], "val", snap[1], path)


def test_confidence_binding_refuses_dump_not_in_record(tmp_path, snap, fitted):
    stray = write_candidate(tmp_path / "stray", snap, 2e-05, 3, rng_seed=999)   # same identity, other logits
    with pytest.raises(SystemExit, match="lists 0 candidate"):
        report.confidence_report(stray, "val", snap[1], fitted[1])


def test_confidence_binding_refuses_non_postfit_schema(tmp_path, snap, fitted):
    wrong = _write_json(tmp_path / "x" / "conf.json", {"schema": "confidence/1"})
    for path in (wrong, _write_json(tmp_path / "x" / "list.json", [1, 2])):
        with pytest.raises(SystemExit, match="not a postfit/1 record"):
            report.confidence_report(fitted[0][0], "val", snap[1], path)
    broken = tmp_path / "x" / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a postfit/1 record"):
        report.confidence_report(fitted[0][0], "val", snap[1], str(broken))
    with pytest.raises(SystemExit, match="not found"):
        report.confidence_report(fitted[0][0], "val", snap[1], str(tmp_path / "missing.json"))


def test_confidence_binding_refuses_candidate_off_val(snap, fitted):
    d = fitted[0][0]
    _, _, side, _ = dumps.load_dump(d, "val", snap[1])
    with pytest.raises(SystemExit, match="reported on val only"):
        report.confidence_binding(d, "test", side, snap[1], fitted[1])


def test_confidence_binding_final_seed_by_postfit_sha256(tmp_path, snap, fitted):
    _, path, record, sha = fitted
    T = record["chosen"]["temperature"]
    final = write_candidate(tmp_path / "final", snap, 2e-05, record["chosen"]["epoch"], split="final_train",
                            calibrated=True, shipped=T, postfit_sha256=sha)
    rep = report.confidence_report(final, "val", snap[1], path)
    assert rep["binding"]["kind"] == "final_seed"
    assert rep["binding"]["theta_source"]["postfit_sha256"] == sha
    assert rep["variants"]["headline"]["temperature"] == T
    assert rep["variants"]["headline"]["theta"] == record["chosen"]["theta"]["shipped"]
    assert rep["variants"]["raw"]["temperature"] == 1.0
    assert rep["variants"]["raw"]["theta"] == record["chosen"]["theta"]["raw"]
    with pytest.raises(SystemExit, match="final seed .* is reported only with --postfit"):
        report.confidence_report(final, "val", snap[1])


def test_confidence_binding_final_seed_refuses_other_record_or_t(tmp_path, snap, fitted):
    _, path, record, sha = fitted
    T = record["chosen"]["temperature"]
    final = write_candidate(tmp_path / "final", snap, 2e-05, 3, split="final_train", calibrated=True, shipped=T,
                            postfit_sha256=sha)
    copy = tmp_path / "copy.json"
    copy.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")   # same content, other bytes
    copy = str(copy)
    assert pilot.data.sha256_file(copy) != sha
    with pytest.raises(SystemExit, match="final seed was bound to record"):
        report.confidence_report(final, "val", snap[1], copy)
    off = write_candidate(tmp_path / "off", snap, 2e-05, 3, split="final_train", calibrated=True, shipped=T + 0.5,
                          postfit_sha256=sha)
    with pytest.raises(SystemExit, match="record's chosen T"):
        report.confidence_report(off, "val", snap[1], path)


def _prior(snap, root, splits=("val", "test")):
    out_dir, manifest = snap[0], snap[1]
    for name in splits:
        score.main([os.path.join(out_dir, "splits", "%s.jsonl" % name), "--model", "prior", "--manifest", manifest,
                    "--out-root", str(root)])
    return os.path.join(str(root), "prior")


def test_confidence_binding_refuses_postfit_for_baseline(tmp_path, snap, fitted, capsys):
    d = _prior(snap, tmp_path / "base")
    capsys.readouterr()
    with pytest.raises(SystemExit, match="no training block"):
        report.confidence_report(d, "val", snap[1], fitted[1])


def test_confidence_binding_baseline_selects_theta_on_val(tmp_path, snap, capsys):
    d = _prior(snap, tmp_path / "base")
    capsys.readouterr()
    rep = report.confidence_report(d, "val", snap[1])
    run = report.run_report(d, "val", snap[1])
    assert rep["binding"]["kind"] == "val_selected" and rep["binding"]["postfit"] is None
    assert rep["binding"]["theta_source"] == run["theta_source"]
    assert rep["variants"]["headline"]["theta"] == run["variants"]["shipped"]["rows"]["val_selected"]["theta"]
    assert rep["variants"]["raw"]["theta"] == run["variants"]["raw"]["rows"]["val_selected"]["theta"]
    assert rep["in_sample"] is True and rep["training"] is None
    test = report.confidence_report(d, "test", snap[1])                     # θ from val, applied to test
    assert test["in_sample"] is False and test["caveat"] is None
    assert test["variants"]["headline"]["theta"] == rep["variants"]["headline"]["theta"]


def test_confidence_refuses_theta_source_off_val(tmp_path, snap, capsys):
    only_test = _prior(snap, tmp_path / "t", splits=("test",))
    capsys.readouterr()
    with pytest.raises(SystemExit, match="no full val dump in .*θ is chosen on val only"):
        report.confidence_report(only_test, "test", snap[1])
    d = _prior(snap, tmp_path / "s")
    capsys.readouterr()
    _, uids, side, _ = dumps.load_dump(d, "val", snap[1])
    logits = dumps.load_dump(d, "val", snap[1])[0]
    planted = {k: v for k, v in side.items() if k not in ("dump_sha256", "schema_version")}
    planted["temperature"] = dict(side["temperature"], shipped=2.0)       # another source's val dump
    dumps.write_dump(d, "val", logits, uids, planted)
    with pytest.raises(SystemExit, match="differ in temperature.shipped"):
        report.confidence_report(d, "test", snap[1])


def test_confidence_refuses_subset_dump(tmp_path, snap, capsys):
    out_dir, manifest = snap[0], snap[1]
    run_dir, _ = score.main([os.path.join(out_dir, "splits", "val.jsonl"), "--model", "prior", "--manifest", manifest,
                             "--out-root", str(tmp_path / "sub"), "--n", "5"])
    capsys.readouterr()
    with pytest.raises(SystemExit, match="--n subset dump is never reported"):
        report.confidence_report(run_dir, "val", manifest)


def test_confidence_out_writes_the_svg_next_to_the_json(tmp_path, snap, capsys):
    d = _prior(snap, tmp_path / "base")
    out = tmp_path / "conf" / "prior-val.json"
    report.main(["--confidence", d, "--split", "val", "--manifest", snap[1], "--out", str(out)])
    capsys.readouterr()
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert (tmp_path / "conf" / "prior-val.svg").read_bytes() == report.render_svg(rec).encode("utf-8")
    md = tmp_path / "conf.md"
    text = report.main(["--confidence-table", str(tmp_path / "conf"), "--out", str(md)])
    assert capsys.readouterr().out == text == report.render_confidence([rec])
    assert md.read_bytes() == text.encode("utf-8")
    assert "![reliability diagram prior (val)](results/confidence/prior-val.svg)" in text
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence", d, "--split", "val", "--manifest", snap[1], "--out", str(tmp_path / "x.svg")])
    assert e.value.code == 2 and "--out must name a .json file" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence-table", str(tmp_path / "conf"), "--table", str(tmp_path / "conf")])
    assert e.value.code == 2 and "exactly one of" in capsys.readouterr().err


def test_confidence_argument_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        report.main(["--postfit", "x.json", "results/dumps/prior"])
    assert e.value.code == 2 and "--postfit applies only to --confidence" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence", "results/dumps/prior"])
    assert e.value.code == 2 and "--confidence needs an explicit --split" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        report.main(["--confidence", "results/dumps/prior", "results/dumps/prior", "--split", "val"])
    assert e.value.code == 2 and "exactly one of" in capsys.readouterr().err


def test_report_stays_numpy_only():
    banned = ("torch", "scipy", "laya", "ray", "transformers", "sklearn")
    code = "import sys, report; print(sorted(m for m in %r if m in sys.modules))" % (banned,)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
