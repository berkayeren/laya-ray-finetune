"""Consumers refuse tampered, unregistered or wrong-role splits before any model load or Ray startup (D-03).

Each test runs the real script as a subprocess inside a hermetic copy of the repo whose data/splits.json is
built from the synthetic snapshot, so the copied pilot package's DEFAULT_MANIFEST points at it.
"""
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
    for name in ("evaluate.py", "ray_train.py", "intents.py"):
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
