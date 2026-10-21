"""Consumers refuse tampered, unregistered or wrong-role splits before any model load or Ray startup (D-03).

Each test runs the real script as a subprocess inside a hermetic copy of the repo whose data/splits.json is
built from the synthetic snapshot, so the copied pilot package's DEFAULT_MANIFEST points at it. evaluate.py is a
thin wrapper over score.py + report.py, so its refusals come from score.py; test_evaluate_is_a_thin_wrapper keeps it
free of metric and model-loading code (the assumption-delta invariant of plan 03-01).
"""
import ast
import os
import shutil
import subprocess
import sys

import pytest

import pilot.data

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
TIMEOUT = 300                                               # ray/torch/laya import time on a cold cache


@pytest.fixture
def tmp_repo(tmp_path, synthetic_raw):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("evaluate.py", "score.py", "report.py", "ray_train.py", "intents.py"):
        shutil.copyfile(os.path.join(REPO, name), str(repo / name))
    shutil.copytree(os.path.join(REPO, "pilot"), str(repo / "pilot"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    root, raw_sha256 = synthetic_raw()
    pilot.data.prepare(root, str(repo / "data"), raw_sha256)
    return repo


def _run(repo, *args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable] + list(args), cwd=str(repo), env=env, capture_output=True, text=True,
                          timeout=TIMEOUT)


def _tamper(path):
    with open(path, "ab") as f:
        f.write(b"x")


def test_evaluate_refuses_tampered_split(tmp_repo):
    _tamper(str(tmp_repo / "data" / "splits" / "test.jsonl"))
    r = _run(tmp_repo, "evaluate.py", "data/splits/test.jsonl", "--workers", "0")
    assert r.returncode != 0
    assert "sha256" in r.stderr, r.stderr


def test_evaluate_refuses_unregistered_copy(tmp_repo):
    shutil.copyfile(str(tmp_repo / "data" / "splits" / "test.jsonl"), str(tmp_repo / "copy.jsonl"))
    r = _run(tmp_repo, "evaluate.py", "copy.jsonl", "--workers", "0")
    assert r.returncode != 0
    assert "not a split registered" in r.stderr, r.stderr


@pytest.mark.parametrize("name", ["hotels_probe", "selection_train", "final_train"])
def test_evaluate_refuses_non_eval_role(tmp_repo, name):
    r = _run(tmp_repo, "evaluate.py", "data/splits/%s.jsonl" % name, "--workers", "0")
    assert r.returncode != 0
    assert "role" in r.stderr, r.stderr


@pytest.fixture
def variant(tmp_repo, synthetic_raw):
    """The documented `prepare_data.py --no-include-no-intent --out OTHER_DIR` variant inside tmp_repo."""
    root, raw_sha256 = synthetic_raw()
    pilot.data.prepare(root, str(tmp_repo / "other"), raw_sha256, include_no_intent=False)
    return tmp_repo


def test_evaluate_reads_the_variant_only_with_its_manifest(variant):
    r = _run(variant, "evaluate.py", "other/splits/test.jsonl", "--workers", "0")
    assert r.returncode != 0 and "not a split registered" in r.stderr, r.stderr
    # verification passes with --manifest; the missing checkpoint dir is the next (pre-model-load) refusal
    r = _run(variant, "evaluate.py", "other/splits/test.jsonl", "--workers", "0",
             "--manifest", "other/splits.json", "--checkpoint", "no_such_ckpt")
    assert r.returncode != 0 and "--checkpoint must be" in r.stderr, r.stderr


def test_ray_train_reads_the_variant_only_with_its_manifest(variant):
    args = ["ray_train.py", "--data", "other/splits/selection_train.jsonl", "--out", "ckpt",
            "--base", str(variant / "no_such_base"), "--workers", "1"]
    r = _run(variant, *args)
    assert r.returncode != 0 and "not a split registered" in r.stderr, r.stderr
    r = _run(variant, *(args + ["--manifest", "other/splits.json"]))
    assert r.returncode != 0 and "--base must be" in r.stderr, r.stderr
    assert not (variant / "ray_results").exists()


def test_ray_train_refuses_tampered_split_before_ray(tmp_repo):
    _tamper(str(tmp_repo / "data" / "splits" / "selection_train.jsonl"))
    # an absolute, missing --base guarantees no hub download even if the guard ever regresses
    r = _run(tmp_repo, "ray_train.py", "--data", "data/splits/selection_train.jsonl", "--out", "ckpt",
             "--base", str(tmp_repo / "no_such_base"), "--workers", "1")
    assert r.returncode != 0
    assert "sha256" in r.stderr, r.stderr
    assert not (tmp_repo / "ray_results").exists()
    assert not (tmp_repo / "ckpt").exists()


def test_ray_train_refuses_non_train_role(tmp_repo):
    r = _run(tmp_repo, "ray_train.py", "--data", "data/splits/test.jsonl", "--out", "ckpt",
             "--base", str(tmp_repo / "no_such_base"), "--workers", "1")
    assert r.returncode != 0
    assert "role" in r.stderr, r.stderr
    assert not (tmp_repo / "ray_results").exists()


def test_ray_train_refuses_non_empty_out_before_ray(tmp_repo):
    (tmp_repo / "ckpt").mkdir()
    (tmp_repo / "ckpt" / "model.safetensors").write_bytes(b"stale")
    r = _run(tmp_repo, "ray_train.py", "--data", "data/splits/selection_train.jsonl", "--out", "ckpt",
             "--base", str(tmp_repo / "no_such_base"), "--workers", "1")
    assert r.returncode != 0
    assert "not empty" in r.stderr, r.stderr
    assert not (tmp_repo / "ray_results").exists()


@pytest.mark.timeout(900)
def test_score_refuses_a_finetuned_checkpoint_on_test_before_any_load(tmp_repo):
    """ROADMAP Phase 5 criterion 4 (RESEARCH A6): under the committed manifest (tmp_repo's own data/splits.json) a
    fine-tuned checkpoint is refused on test by score.py and evaluate.py before any load; val, and a checkpoint that is
    not fine-tuned on test, are unaffected."""
    import json
    from tiny_laya import tiny_checkpoint
    plain = tiny_checkpoint(tmp_repo / "plain")
    shutil.copytree(plain, str(tmp_repo / "ckpt"))
    cfg_path = tmp_repo / "ckpt" / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["fine_tuned"] = True
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    for script in ("score.py", "evaluate.py"):
        r = _run(tmp_repo, script, "data/splits/test.jsonl", "--checkpoint", "ckpt", "--workers", "0")
        assert r.returncode != 0, (script, r.stdout, r.stderr)
        assert "fine-tuned" in r.stderr and "Phase 8" in r.stderr, (script, r.stderr)
        assert "Traceback" not in r.stderr, (script, r.stderr)
    assert not (tmp_repo / "results" / "dumps").exists()
    r = _run(tmp_repo, "score.py", "data/splits/val.jsonl", "--checkpoint", "ckpt", "--workers", "0",
             "--out-root", "ok")
    assert r.returncode == 0, r.stderr
    r = _run(tmp_repo, "score.py", "data/splits/test.jsonl", "--checkpoint", "plain", "--workers", "0",
             "--out-root", "ok")
    assert r.returncode == 0, r.stderr


@pytest.mark.timeout(900)
def test_score_interlock_is_keyed_on_the_test_content(tmp_repo, synthetic_raw):
    """05-REVIEW CR-02: the interlock follows the scored content, not the manifest bytes. A copy of the committed
    manifest with one extra newline next to the committed split bytes, and the --no-include-no-intent variant (other
    bytes and uids, the same (uid, message) pairs minus the no-intent rows), both still refuse a fine-tuned checkpoint
    on test before any load; val under the edited manifest copy still scores."""
    import json
    from tiny_laya import tiny_checkpoint
    shutil.copytree(tiny_checkpoint(tmp_repo / "plain"), str(tmp_repo / "ckpt"))
    cfg_path = tmp_repo / "ckpt" / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["fine_tuned"] = True
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    shutil.copytree(str(tmp_repo / "data" / "splits"), str(tmp_repo / "copy" / "splits"))
    (tmp_repo / "copy" / "splits.json").write_bytes((tmp_repo / "data" / "splits.json").read_bytes() + b"\n")
    root, raw_sha256 = synthetic_raw()
    pilot.data.prepare(root, str(tmp_repo / "noint"), raw_sha256, include_no_intent=False)
    for d in ("copy", "noint"):
        r = _run(tmp_repo, "score.py", "%s/splits/test.jsonl" % d, "--checkpoint", "ckpt", "--workers", "0",
                 "--manifest", "%s/splits.json" % d, "--out-root", "out")
        assert r.returncode != 0, (d, r.stdout, r.stderr)
        assert "committed test content" in r.stderr and "Phase 8" in r.stderr, (d, r.stderr)
        assert "Traceback" not in r.stderr, (d, r.stderr)
    assert not (tmp_repo / "out").exists()
    r = _run(tmp_repo, "score.py", "copy/splits/val.jsonl", "--checkpoint", "ckpt", "--workers", "0",
             "--manifest", "copy/splits.json", "--out-root", "out")
    assert r.returncode == 0, r.stderr


def _relabel(repo, name, raw):
    """A manifest copy (copy/splits.json) whose `name` entry registers the bytes `raw`; returns the split path."""
    import json
    shutil.copytree(str(repo / "data" / "splits"), str(repo / "copy" / "splits"), dirs_exist_ok=True)
    (repo / "copy" / "splits" / (name + ".jsonl")).write_bytes(raw)
    path = repo / "copy" / "splits.json"
    m = json.loads((path if path.exists() else repo / "data" / "splits.json").read_text(encoding="utf-8"))
    rows = pilot.data.parse_jsonl(raw)
    m["splits"][name].update(sha256=pilot.data.sha256_bytes(raw), n_messages=len(rows),
                             uids_sha256=pilot.data.sha256_bytes("\n".join(r["uid"] for r in rows).encode("utf-8")))
    path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return "copy/splits/%s.jsonl" % name


@pytest.mark.parametrize("script", ["ray_train.py", "berturk_train.py"])
@pytest.mark.parametrize("name, held, cut", [
    ("selection_train", "val", 0),           # the whole committed val bytes: the entry's hashes
    ("selection_train", "val", 1),           # all but one row: the (uid, message) pairs
    ("selection_train", "test", 1),
    ("final_train", "test", 0),
])
def test_trainers_refuse_held_out_content_under_any_manifest(tmp_repo, script, name, held, cut):
    """06-REVIEW WR-02: a manifest copy that registers committed val or test content as a train split is refused by
    both trainers before any fetch, load or Ray start (pilot.data.refuse_held_out; final_train keeps the val folds)."""
    shutil.copyfile(os.path.join(REPO, "berturk_train.py"), str(tmp_repo / "berturk_train.py"))
    lines = (tmp_repo / "data" / "splits" / (held + ".jsonl")).read_bytes().splitlines(keepends=True)
    data = _relabel(tmp_repo, name, b"".join(lines[cut:]))
    r = _run(tmp_repo, script, "--data", data, "--manifest", "copy/splits.json", "--out", "ckpt",
             "--base", str(tmp_repo / "no_such_base"), "--storage", "runs")
    assert r.returncode != 0 and "holds committed %s content" % held in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr
    assert not (tmp_repo / "runs").exists() and not (tmp_repo / "ckpt").exists()


@pytest.mark.parametrize("n", ["0", "-5"])
def test_evaluate_refuses_non_positive_n(tmp_repo, n):
    r = _run(tmp_repo, "evaluate.py", "data/splits/test.jsonl", "--workers", "0", "--n", n)
    assert r.returncode == 2 and "--n must be a positive" in r.stderr, r.stderr


def test_ray_train_refuses_zero_epochs_before_ray(tmp_repo):
    r = _run(tmp_repo, "ray_train.py", "--data", "data/splits/selection_train.jsonl", "--out", "ckpt",
             "--base", str(tmp_repo / "no_such_base"), "--workers", "1", "--epochs", "0")
    assert r.returncode == 2 and "--epochs must be at least 1" in r.stderr, r.stderr
    assert not (tmp_repo / "ray_results").exists()


def test_evaluate_missing_split_gets_a_make_data_hint(tmp_repo):
    r = _run(tmp_repo, "evaluate.py", "data/splits/tset.jsonl", "--workers", "0")
    assert r.returncode != 0 and "not found" in r.stderr and "make data" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr


def test_evaluate_is_a_thin_wrapper():
    """evaluate.py imports score and report, defines only main and holds no metric, threshold or model-loading code."""
    with open(os.path.join(REPO, "evaluate.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {"score", "report"} <= imported, imported
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    forbidden = {"metrics", "f1", "select_theta", "majority_tag", "resolve_checkpoint", "load_agent"}
    assert not set(funcs) & forbidden, funcs
    assert funcs == ["main"], funcs
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.Lambda))]


@pytest.mark.parametrize("flags, fix", [
    (["--device", "mps", "--workers", "2"], "--workers 1"),
    (["--gpu", "--device", "mps", "--workers", "1"], "alias"),
    (["--amp", "--workers", "1"], "CUDA-only"),
    (["--device", "mps", "--amp", "--workers", "1"], "CUDA-only"),
], ids=["mps-workers-2", "gpu-with-mps", "amp-on-cpu", "amp-on-mps"])
def test_device_flag_conflicts_are_argument_errors(tmp_repo, flags, fix):
    r = _run(tmp_repo, "ray_train.py", "--data", "data/splits/selection_train.jsonl", "--out", "ckpt",
             "--base", str(tmp_repo / "no_such_base"), *flags)
    assert r.returncode == 2 and fix in r.stderr, r.stderr
    assert not (tmp_repo / "ray_results").exists()
