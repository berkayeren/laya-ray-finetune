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


# --- the committed audit dumps are replaced only on purpose (03-REVIEW WR-03, IN-04) -------------------------------

def test_existing_dump_is_not_overwritten_without_force(tmp_path, snap):
    out_dir, manifest = snap
    args = [os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior", "--manifest", manifest,
            "--out-root", str(tmp_path / "dumps")]
    run_dir, split = score.main(args)
    with open(os.path.join(run_dir, "test.npz"), "rb") as f:
        before = f.read()
    with pytest.raises(SystemExit, match="pass --force to rescore it"):
        score.main(args)
    with open(os.path.join(run_dir, "test.npz"), "rb") as f:
        assert f.read() == before                          # the refusal wrote nothing
    assert score.main(args + ["--force"]) == (run_dir, split)
    logits, uids, side, _ = dumps.load_dump(run_dir, split, manifest)
    assert logits.shape == (7, 48) and side["n_subset"] is None


def test_force_refusal_comes_before_any_model_load(tmp_path, snap, monkeypatch):
    from pilot.weights import identity
    from tiny_laya import tiny_checkpoint
    out_dir, manifest = snap
    ckpt = tiny_checkpoint(tmp_path / "ckpt")
    run_dir = tmp_path / "dumps" / dumps.run_id("laya", identity(ckpt))
    run_dir.mkdir(parents=True)
    (run_dir / "test.json").write_text("{}", encoding="utf-8")     # an existing dump, even a partial one, blocks

    def no_load(*a, **k):
        raise AssertionError("loaded a model before the overwrite check")
    monkeypatch.setattr(score, "load_agent", no_load)
    monkeypatch.setattr(score, "resolve_checkpoint", no_load)
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--checkpoint", ckpt, "--workers", "0",
                    "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    assert "pass --force to rescore it" in str(e.value.code)
    assert "ray" not in sys.modules or not sys.modules["ray"].is_initialized()


def test_out_of_repo_dump_prints_its_real_path(snap, capsys):
    out_dir, manifest = snap
    run_dir, split = score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior",
                                 "--manifest", manifest, "--n", "1"])
    try:
        out = capsys.readouterr().out
        want = os.path.join(os.path.abspath(run_dir), split + ".npz")
        assert want in out and "<outside-repo>" not in out, out
        assert os.path.isfile(want)
    finally:
        shutil.rmtree(os.path.dirname(run_dir))


def test_evaluate_defaults_to_a_temp_out_root(tmp_path, monkeypatch):
    import tempfile
    import evaluate
    calls = []

    def fake_score(argv, roles=None):
        calls.append(list(argv))
        assert roles == ("test", "val")                     # evaluate.py never scores hotels_probe (08-07)
        return str(tmp_path / "run"), "test"
    monkeypatch.setattr(score, "main", fake_score)
    monkeypatch.setattr(report, "run_report", lambda *a, **k: {})
    evaluate.main(["data/splits/test.jsonl", "--checkpoint", "x", "--workers", "0"])
    argv = calls[-1]
    assert "--out-root" in argv and "--force" not in argv and "--device" not in argv, argv
    root = argv[argv.index("--out-root") + 1]
    assert os.path.realpath(root).startswith(os.path.realpath(tempfile.gettempdir()) + os.sep), root
    assert os.path.isdir(root) and not os.listdir(root)
    os.rmdir(root)
    evaluate.main(["data/splits/test.jsonl", "--checkpoint", "x", "--workers", "0", "--force", "--device", "mps",
                   "--out-root", str(tmp_path / "given")])
    argv = calls[-1]
    assert "--force" in argv and argv[argv.index("--device") + 1] == "mps", argv
    assert argv[argv.index("--out-root") + 1] == str(tmp_path / "given")


def test_makefile_baselines_pass_force():
    with open(os.path.join(REPO, "Makefile"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    start = lines.index("baselines:") + 1
    recipe = []
    for line in lines[start:]:
        if not line.startswith("\t"):
            break
        recipe.append(line)
    scored = [line for line in recipe if "score.py" in line]
    assert len(scored) == 6 and all("--force" in line.split() for line in scored), scored


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


# --- val-selected θ: chosen on the same run's full val dump only (EVAL-02, D-11, D-12) ----------------------------

def _score_both(snap, root, model="prior"):
    out_dir, manifest = snap
    for name in ("val", "test"):
        score.main([os.path.join(out_dir, "splits", "%s.jsonl" % name), "--model", model, "--manifest", manifest,
                    "--out-root", str(root)])
    return os.path.join(str(root), model), manifest


def test_report_selects_theta_on_val_and_discloses_it(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    rep = report.run_report(run_dir, "test", manifest)
    assert set(rep["variants"]) == {"shipped", "raw"} and rep["headline"] == "shipped"
    assert rep["variants"]["shipped"]["temperature"] == 1.0 and rep["variants"]["raw"]["temperature"] == 1.0
    _, val_uids, val_side, entry = dumps.load_dump(run_dir, "val", manifest)
    assert rep["theta_source"] == {"run_id": "prior", "split": "val", "dump_sha256": val_side["dump_sha256"],
                                   "uids_sha256": entry["uids_sha256"], "n_messages": len(val_uids)}
    for v in rep["variants"].values():
        assert set(v["rows"]) == {"0.5", "val_selected"} and "val_selected_reason" not in v
        row = v["rows"]["val_selected"]
        th = row["theta"]
        assert set(th) == {"global", "per_tag", "eligible_tags", "min_val_positives", "grid", "objective",
                           "tie_break"}
        assert th["grid"] == {"start": 0.05, "stop": 0.95, "step": 0.01, "n": 91} and th["min_val_positives"] == 10
        assert list(th["per_tag"]) == th["eligible_tags"] and row["in_sample"] is False
        assert row["n"] == v["rows"]["0.5"]["n"]


def test_report_on_val_marks_theta_in_sample(tmp_path, snap):
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    rep = report.run_report(run_dir, "val", manifest)
    assert all(v["rows"]["val_selected"]["in_sample"] is True for v in rep["variants"].values())


def test_report_without_val_dump(tmp_path, good_run):
    run_dir, manifest = good_run
    rep = report.run_report(run_dir, "test", manifest)
    assert rep["theta_source"] is None
    for v in rep["variants"].values():
        assert v["rows"]["val_selected"] is None and v["val_selected_reason"] == "no val dump in run dir"
        assert v["rows"]["0.5"]["n"] == 7


def test_report_refuses_subset_val_dump(tmp_path, snap):
    out_dir, manifest = snap
    root = tmp_path / "dumps"
    score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior", "--manifest", manifest,
                "--out-root", str(root)])
    score.main([os.path.join(out_dir, "splits", "val.jsonl"), "--model", "prior", "--manifest", manifest,
                "--n", "2", "--out-root", str(root)])
    with pytest.raises(SystemExit, match="--n"):
        report.run_report(str(root / "prior"), "test", manifest)


def test_theta_never_reads_test(tmp_path, snap):
    """Two runs with identical val dumps and different test probabilities get the same θ."""
    a, manifest = _score_both(snap, tmp_path / "a")
    b, _ = _score_both(snap, tmp_path / "b")
    logits, uids, side, _ = dumps.load_dump(b, "test", manifest)
    rng = np.random.default_rng(0)
    other = rng.normal(0.0, 3.0, size=logits.shape).astype(np.float32)
    dumps.write_dump(b, "test", other, uids, {k: v for k, v in side.items()
                                               if k not in ("dump_sha256", "schema_version")})
    ra, rb = report.run_report(a, "test", manifest), report.run_report(b, "test", manifest)
    assert ra["dump_sha256"] != rb["dump_sha256"] and ra["theta_source"] == rb["theta_source"]
    for v in ("shipped", "raw"):
        assert ra["variants"][v]["rows"]["val_selected"]["theta"] == rb["variants"][v]["rows"]["val_selected"]["theta"]
        assert ra["variants"][v]["rows"]["0.5"] != rb["variants"][v]["rows"]["0.5"]


# --- θ comes only from a val dump of the same checkpoint, temperature and scorer (03-REVIEW WR-01) ------------------

SOURCE_EDITS = {
    "kind": ("system.kind", "another-system"),
    "checkpoint": ("system.checkpoint", "runs/other-ckpt"),
    "identity": ("system.checkpoint_identity", {"local": True, "model_sha256": "a" * 64, "config_sha256": "b" * 64}),
    "temperature": ("temperature.shipped", 2.5),
    "device": ("scorer.device", "mps"),
    "dtype": ("scorer.dtype", "torch.float16"),
    "amp": ("scorer.amp_enabled", True),
}


def _set_key(dotted, value):
    def fn(side):
        head, key = dotted.split(".")
        side.setdefault(head, {})[key] = value
    return fn


@pytest.mark.parametrize("case", sorted(SOURCE_EDITS))
def test_report_refuses_val_and_split_from_different_sources(tmp_path, snap, case):
    dotted, value = SOURCE_EDITS[case]
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    _edit_sidecar(run_dir, _set_key(dotted, value))        # the test dump now claims another source than val
    with pytest.raises(SystemExit) as e:
        report.run_report(run_dir, "test", manifest)
    assert dotted in str(e.value.code) and "val dump" in str(e.value.code), e.value.code


def _rename_checkpoint(run_dir, ident):
    for name, ckpt in (("val", "runs/ckpt"), ("test", "runs/ckpt-renamed")):
        path = os.path.join(run_dir, "%s.json" % name)
        with open(path, encoding="utf-8") as f:
            side = json.load(f)
        side["system"].update(checkpoint=ckpt, checkpoint_identity=ident)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(side, f)


def test_a_moved_checkpoint_with_the_same_identity_is_the_same_source(tmp_path, snap):
    """04-REVIEW IN-12, IN-05: the path counts only when no checkpoint_identity binds every file laya.load reads
    (weights, config, tokenizer/ and encoder/); then a change in any of them is a change of the identity."""
    files = {"encoder/config.json": "e" * 64, "tokenizer/tokenizer.json": "f" * 64,
             "tokenizer/tokenizer_config.json#normalized": "0" * 64}
    ident = {"local": True, "model_sha256": "a" * 64, "config_sha256": "b" * 64, "files_sha256": files}
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    _rename_checkpoint(run_dir, ident)
    rep = report.run_report(run_dir, "test", manifest)
    assert rep["variants"]["shipped"]["rows"]["val_selected"] is not None
    for changed in (dict(ident, model_sha256="c" * 64),
                    dict(ident, files_sha256=dict(files, **{"tokenizer/tokenizer.json": "1" * 64}))):
        _rename_checkpoint(run_dir, ident)
        _edit_sidecar(run_dir, _set_key("system.checkpoint_identity", changed))
        with pytest.raises(SystemExit, match=r"system\.checkpoint_identity"):
            report.run_report(run_dir, "test", manifest)
    # an identity from before IN-05 binds weights and config only: a renamed dir is still refused on the path
    _rename_checkpoint(run_dir, {"local": True, "model_sha256": "a" * 64, "config_sha256": "b" * 64})
    with pytest.raises(SystemExit, match=r"system\.checkpoint "):
        report.run_report(run_dir, "test", manifest)
    assert report.binds_every_file({"revision": "7b928d82", "model_sha256": "a" * 64})
    assert not report.binds_every_file(None) and not report.binds_every_file({"local": True})
    # 04-REVIEW WR-02: a local identity without tokenizer/ or encoder/ does not bind what laya.load reads
    assert report.binds_every_file(ident)
    for rel in dumps.LOCAL_REQUIRED_FILES:
        assert not report.binds_every_file(dict(ident, files_sha256={k: v for k, v in files.items() if k != rel}))
    assert not report.binds_every_file(dict(ident, files_sha256={}))


def test_compare_refuses_mixed_sources(tmp_path, snap):
    a, manifest = _score_both(snap, tmp_path / "a")
    b, _ = _score_both(snap, tmp_path / "b", model="majority")
    _edit_sidecar(a, _set_key("scorer.device", "mps"))
    with pytest.raises(SystemExit, match=r"scorer\.device"):
        report.compare([a], [b], "test", manifest, B=50, seed=0)
    with pytest.raises(SystemExit, match=r"scorer\.device"):
        report.compare([b], [a], "test", manifest, B=50, seed=0)


def test_baseline_source_may_differ_between_val_and_test(tmp_path, snap):
    """A baseline's val and test dumps come from different training splits by design (TRAINED_ON)."""
    run_dir, manifest = _score_both(snap, tmp_path / "dumps")
    sources = {s: dumps.load_dump(run_dir, s, manifest)[2]["system"]["baseline_source"]["split"]
               for s in ("val", "test")}
    assert sources == {"val": "selection_train", "test": "final_train"}
    rep = report.run_report(run_dir, "test", manifest)
    assert rep["theta_source"] is not None and rep["variants"]["shipped"]["rows"]["val_selected"] is not None
    assert report.compare([run_dir], [run_dir], "test", manifest, B=50, seed=0)["n"] == 7


def test_local_run_id_includes_the_config_hash():
    from pilot import pins
    from pilot.weights import identity
    files = {"encoder/config.json": "c" * 64}
    rid = dumps.run_id("laya", {"local": True, "model_sha256": "a" * 64, "config_sha256": "b" * 64,
                                "files_sha256": files})
    assert rid == "laya-" + "a" * 12 + "-" + "b" * 8 + "-" + dumps.files_digest(files)[:8]
    # 04-REVIEW IN-05: tokenizer/ and encoder/ name the run too
    assert dumps.run_id("laya", {"local": True, "model_sha256": "a" * 64, "config_sha256": "b" * 64,
                                 "files_sha256": {"encoder/config.json": "d" * 64}}) != rid
    with pytest.raises(SystemExit, match="files_sha256"):
        dumps.run_id("laya", {"local": True, "model_sha256": "a" * 64, "config_sha256": "b" * 64})
    assert dumps.run_id("laya", identity("multilingual")) == "laya-zeroshot-" + pins.LAYA["revision"][:8] == \
        "laya-zeroshot-7b928d82"
    assert dumps.run_id("majority") == "majority" and dumps.run_id("prior") == "prior"
    for ident in ({"local": True, "model_sha256": "a" * 64}, {"local": True, "model_sha256": "a" * 64,
                                                               "config_sha256": None}):
        with pytest.raises(SystemExit, match="config_sha256"):
            dumps.run_id("laya", ident)


# --- a --n subset val dump reported on val: θ 0.5 rows, no θ source (03-REVIEW WR-04) ------------------------------

def test_val_subset_report_has_no_theta_source(tmp_path, snap):
    out_dir, manifest = snap
    run_dir, split = score.main([os.path.join(out_dir, "splits", "val.jsonl"), "--model", "prior",
                                 "--manifest", manifest, "--n", "2", "--out-root", str(tmp_path / "sub")])
    rep = report.run_report(run_dir, split, manifest)
    assert split == "val" and rep["theta_source"] is None and rep["subset"] is True
    for v in rep["variants"].values():
        assert v["rows"]["0.5"]["n"] == 2 and v["rows"]["val_selected"] is None
        assert v["val_selected_reason"] == report.SUBSET_VAL_REASON
    assert report.SUBSET_VAL_REASON == "val dump is a --n subset; θ is selected only on a full val dump"


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


def _truncate_zip(run_dir):
    """Cut the npz to half its bytes (the zip's central directory is lost) and rewrite dump_sha256 to match, so only
    the npz parser can catch it (03-REVIEW IN-03)."""
    path = os.path.join(run_dir, "test.npz")
    with open(path, "rb") as f:
        data = f.read()
    data = data[:len(data) // 2]
    with open(path, "wb") as f:
        f.write(data)
    _edit_sidecar(run_dir, lambda s: s.update(dump_sha256=pilot.data.sha256_bytes(data)))


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
    "truncated_zip": _truncate_zip,
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
        assert "pickled" in str(e.value.code) and "allow_pickle" in str(e.value.code)
    if case == "truncated_zip":                             # one line naming the exception, no traceback (IN-03)
        assert "cannot be read" in str(e.value.code) and "BadZipFile" in str(e.value.code)
        assert "pickled" not in str(e.value.code) and "\n" not in str(e.value.code)


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
