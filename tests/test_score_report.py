"""The shared scoring path end to end on the synthetic snapshot: score.py (baseline -> dump) and report.py (dump ->
metrics), with no model, torch or Ray. Covers the dump contract (pilot.dumps), the role gate, the --n subset policy,
the train-only baseline source, and report.py's refusal of every tampered dump (D-04).
"""
import ast
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import prepared, prepared_with_line_separators
from intents import INTENTS
from pilot import dumps

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
TIMEOUT = 300


def _split_uids(path):
    with open(path, "rb") as f:
        return [r["uid"] for r in pilot.data.parse_jsonl(f.read())]


@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path) of the synthetic snapshot prepared under tmp_path."""
    return prepared(tmp_path, synthetic_raw)


def test_baseline_dump_round_trips_to_report(tmp_path, snap):
    out_dir, manifest = snap
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    run_dir, split = score.main([test_jsonl, "--model", "prior", "--manifest", manifest,
                                 "--out-root", str(tmp_path / "dumps")])
    assert (run_dir, split) == (str(tmp_path / "dumps" / "prior"), "test")
    z = np.load(os.path.join(run_dir, "test.npz"), allow_pickle=False)
    assert sorted(z.files) == ["logits", "uids"]
    assert z["logits"].dtype == np.float32 and z["logits"].shape == (7, 48)
    assert z["uids"].dtype.kind == "U" and list(z["uids"]) == _split_uids(test_jsonl)
    assert (z["logits"] == z["logits"][0]).all()             # the same baseline row for every message
    with open(os.path.join(run_dir, "test.json"), encoding="utf-8") as f:
        side_text = f.read()
    assert str(tmp_path) not in side_text and os.path.expanduser("~") not in side_text
    row = report.run_report(run_dir, "test", manifest)["variants"]["shipped"]["rows"]["0.5"]
    assert row["n"] == 7 and row["n_no_intent"] == 1


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable] + list(args), cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=TIMEOUT)


def test_cli_round_trip(tmp_path, snap):
    out_dir, manifest = snap
    r = _run("score.py", os.path.join(out_dir, "splits", "test.jsonl"), "--model", "majority",
             "--manifest", manifest, "--out-root", str(tmp_path / "dumps"))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "rep" / "majority.json"
    r = _run("report.py", str(tmp_path / "dumps" / "majority"), "--split", "test", "--out", str(out),
             "--manifest", manifest)
    assert r.returncode == 0, r.stderr
    with open(str(out), encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["run_id"] == "majority" and rep["subset"] is False
    assert rep["system"]["baseline_source"]["split"] == "final_train"


# --- score.py refusals -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["selection_train", "final_train", "hotels_probe"])
def test_score_refuses_non_eval_roles(tmp_path, snap, name):
    out_dir, manifest = snap
    with pytest.raises(SystemExit, match="role"):
        score.main([os.path.join(out_dir, "splits", "%s.jsonl" % name), "--model", "prior", "--manifest", manifest,
                    "--out-root", str(tmp_path / "dumps")])
    assert not (tmp_path / "dumps").exists()


def test_score_refuses_tampered_split(tmp_path, snap):
    out_dir, manifest = snap
    path = os.path.join(out_dir, "splits", "test.jsonl")
    with open(path, "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit, match="sha256"):
        score.main([path, "--model", "prior", "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])


@pytest.mark.parametrize("n", ["0", "-5"])
def test_score_refuses_non_positive_n(snap, capsys, n):
    out_dir, manifest = snap
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior", "--manifest", manifest,
                    "--n", n])
    assert e.value.code == 2
    assert "--n must be a positive" in capsys.readouterr().err


def test_empty_split_is_refused_before_writing(tmp_path, snap, monkeypatch):
    out_dir, manifest = snap
    monkeypatch.setattr(score, "parse_jsonl", lambda raw: [])
    with pytest.raises(SystemExit, match="no messages to score"):
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior", "--manifest", manifest,
                    "--out-root", str(tmp_path / "dumps")])
    assert not (tmp_path / "dumps").exists()


def _good_logits(n=2):
    return np.zeros((n, 48), dtype=np.float32)


@pytest.mark.parametrize("logits,uids", [
    (np.zeros((0, 48), dtype=np.float32), []),                              # zero rows
    (np.zeros((2, 48), dtype=np.float64), ["a", "b"]),                      # float64 logits
    (np.where(np.eye(2, 48, dtype=bool), np.nan, 0).astype(np.float32), ["a", "b"]),   # a nan
    (_good_logits(), ["a", "a"]),                                           # duplicate uids
], ids=["zero_rows", "float64", "nan", "duplicate_uids"])
def test_write_dump_refuses_malformed(tmp_path, logits, uids):
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit, match="malformed"):
        dumps.write_dump(str(run_dir), "test", logits, uids, {"run_id": "run"})
    assert not run_dir.exists() or os.listdir(str(run_dir)) == []


# --- the --n subset policy ---------------------------------------------------------------------------------------

def test_single_message_subset_round_trips(tmp_path, snap):
    out_dir, manifest = snap
    run_dir, split = score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "majority",
                                 "--manifest", manifest, "--n", "1", "--out-root", str(tmp_path / "sub")])
    logits, uids, side, _ = dumps.load_dump(run_dir, split, manifest)
    assert logits.shape == (1, 48) and len(uids) == 1 and side["n_subset"] == 1
    rep = report.run_report(run_dir, split, manifest)
    assert rep["subset"] is True and rep["variants"]["shipped"]["rows"]["0.5"]["n"] == 1


def test_subset_never_under_results_dumps(snap):
    out_dir, manifest = snap
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    with pytest.raises(SystemExit, match="never go under results/dumps"):
        score.main([test_jsonl, "--model", "prior", "--manifest", manifest, "--n", "2",
                    "--out-root", os.path.join(REPO, "results", "dumps")])
    with pytest.raises(SystemExit, match="never go under results/dumps"):
        score.main([test_jsonl, "--model", "prior", "--manifest", manifest, "--n", "2",
                    "--out-root", os.path.join(REPO, "results", "dumps", "sub")])
    run_dir, split = score.main([test_jsonl, "--model", "prior", "--manifest", manifest, "--n", "2"])
    try:
        root = os.path.realpath(os.path.dirname(run_dir))
        assert not (root + os.sep).startswith(os.path.realpath(dumps.DUMP_ROOT) + os.sep)
        with open(os.path.join(run_dir, "test.json"), encoding="utf-8") as f:
            assert json.load(f)["n_subset"] == 2
        assert report.run_report(run_dir, split, manifest)["subset"] is True
    finally:
        shutil.rmtree(os.path.dirname(run_dir))


# --- baselines come from the matching train split only -----------------------------------------------------------

@pytest.mark.parametrize("role,source", [("test", "final_train"), ("val", "selection_train")])
def test_baseline_counts_come_from_matching_train_split(snap, role, source):
    out_dir, manifest = snap
    path = os.path.join(out_dir, "splits", "%s.jsonl" % source)
    with open(path, "rb") as f:
        rows = pilot.data.parse_jsonl(f.read())
    want = {t: sum(bool(r["expected"][t]) for r in rows) for t in INTENTS}
    with open(manifest, encoding="utf-8") as f:
        n_messages = json.load(f)["splits"][source]["n_messages"]
    name, entry, counts, n = score.train_counts(manifest, role)
    assert name == source and entry["path"] == "splits/%s.jsonl" % source
    assert counts == want and list(counts) == list(INTENTS)
    assert n == len(rows) == n_messages                     # no-intent rows count in the prior denominator
    assert any(not any(r["expected"].values()) for r in rows)


def test_train_counts_refuses_tampered_train_split(snap):
    out_dir, manifest = snap
    with open(os.path.join(out_dir, "splits", "final_train.jsonl"), "ab") as f:
        f.write(b"\n")
    with pytest.raises(SystemExit, match="sha256"):
        score.train_counts(manifest, "test")


def test_majority_tie_goes_to_earlier_intent(tmp_path, snap):
    out_dir, manifest = snap
    _, _, counts, _ = score.train_counts(manifest, "test")
    assert counts["card"] == counts["lost_stolen"] == max(counts.values())   # the synthetic tie
    run_dir, split = score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "majority",
                                 "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    logits = np.load(os.path.join(run_dir, "test.npz"), allow_pickle=False)["logits"]
    tags = list(INTENTS)
    hi = np.float32(np.log(1 - dumps.CLIP_EPS) - np.log1p(-(1 - dumps.CLIP_EPS)))
    assert tags.index("card") == 3 and tags.index("lost_stolen") == 33
    assert (logits[:, 3] == hi).all() and (logits[:, 33] == -hi).all() and (logits.argmax(axis=1) == 3).all()


def test_prior_zero_count_tag_is_clipped():
    counts = dict.fromkeys(INTENTS, 0)
    counts["card"] = 3
    row = score.baseline_logits("prior", counts, 4)
    want = np.float32(np.log(1e-6) - np.log1p(-1e-6))
    assert row.dtype == np.float32 and row.shape == (48,) and np.isfinite(row).all()
    assert row[list(INTENTS).index("lost_stolen")] == want
    assert row[3] == np.float32(np.log(0.75) - np.log1p(-0.75))


def test_line_separator_messages_stay_one_row(tmp_path, synthetic_raw):
    out_dir, manifest = prepared_with_line_separators(tmp_path, synthetic_raw)
    run_dir, split = score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior",
                                 "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    logits, uids, _, entry = dumps.load_dump(run_dir, split, manifest)
    assert logits.shape[0] == len(uids) == entry["n_messages"] and "bank_sep_2" in uids


# --- report.py refuses every tampered dump (D-04) ----------------------------------------------------------------

def _rewrite_npz(run_dir, **arrays):
    """Rewrite run_dir/test.npz from `arrays` (pickling allowed, as a tamperer could) and fix dump_sha256."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    with open(os.path.join(run_dir, "test.npz"), "wb") as f:
        f.write(data)
    _edit_sidecar(run_dir, lambda s: s.update(dump_sha256=pilot.data.sha256_bytes(data)))


def _edit_sidecar(run_dir, fn):
    path = os.path.join(run_dir, "test.json")
    with open(path, encoding="utf-8") as f:
        side = json.load(f)
    fn(side)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(side, f)


def _flip_byte(run_dir):
    path = os.path.join(run_dir, "test.npz")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    data[len(data) // 2] ^= 0xFF
    with open(path, "wb") as f:
        f.write(bytes(data))


def _arrays(run_dir):
    z = np.load(os.path.join(run_dir, "test.npz"), allow_pickle=False)
    return z["logits"], z["uids"]


TAMPER = {
    "npz_byte_flipped": _flip_byte,
    "sidecar_split_sha256": lambda d: _edit_sidecar(d, lambda s: s["split"].update(sha256="0" * 64)),
    "sidecar_role": lambda d: _edit_sidecar(d, lambda s: s["split"].update(role="val")),
    "questions_sha256": lambda d: _edit_sidecar(d, lambda s: s.update(questions_sha256="0" * 64)),
    "intents_order_sha256": lambda d: _edit_sidecar(d, lambda s: s.update(intents_order_sha256="0" * 64)),
    "uids_reordered": lambda d: _rewrite_npz(d, logits=_arrays(d)[0], uids=_arrays(d)[1][::-1]),
    "extra_npz_key": lambda d: _rewrite_npz(d, logits=_arrays(d)[0], uids=_arrays(d)[1], extra=np.zeros(1)),
    "float64_logits": lambda d: _rewrite_npz(d, logits=_arrays(d)[0].astype(np.float64), uids=_arrays(d)[1]),
    "pickled_uids": lambda d: _rewrite_npz(d, logits=_arrays(d)[0], uids=np.array(list(_arrays(d)[1]), dtype=object)),
}


@pytest.fixture
def good_run(tmp_path, snap):
    out_dir, manifest = snap
    run_dir, _ = score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior",
                             "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    return run_dir, manifest


def _copy_run(run_dir, tmp_path, name=None):
    dst = str(tmp_path / "bad" / (name or os.path.basename(run_dir)))
    shutil.copytree(run_dir, dst)
    return dst


@pytest.mark.parametrize("case", sorted(TAMPER))
def test_report_refuses_tampered_dump(tmp_path, good_run, case):
    run_dir, manifest = good_run
    bad = _copy_run(run_dir, tmp_path)
    report.run_report(bad, "test", manifest)                # the untouched copy verifies
    TAMPER[case](bad)
    with pytest.raises(SystemExit) as e:
        report.run_report(bad, "test", manifest)
    assert "score.py" in str(e.value.code)                  # a "<what>; rescore with score.py" refusal
    if case == "pickled_uids":
        assert "pickled" in str(e.value.code)


def test_report_refuses_renamed_run_dir(tmp_path, good_run):
    run_dir, manifest = good_run
    bad = _copy_run(run_dir, tmp_path, name="majority")
    with pytest.raises(SystemExit, match="run_id"):
        report.run_report(bad, "test", manifest)


def test_report_refuses_missing_dump(tmp_path, snap):
    _, manifest = snap
    with pytest.raises(SystemExit, match="not found"):
        report.run_report(str(tmp_path / "nothing" / "prior"), "test", manifest)


# --- guards: torch-free, no system-specific branch ---------------------------------------------------------------

def test_import_is_torch_free():
    code = ("import sys, report, score, pilot.metrics, pilot.dumps; "
            "print(sorted(m for m in ('torch', 'ray', 'laya') if m in sys.modules))")
    r = _run("-c", code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout


def test_metrics_code_has_no_system_branch():
    hits = []
    for rel in ("pilot/metrics.py", "report.py"):
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=rel)
        hits += ["%s:%d %r" % (rel, n.lineno, n.value) for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and n.value in ("majority", "prior", "laya")]
    assert hits == [], "system-specific constants in metric code (D-10): %s" % hits
