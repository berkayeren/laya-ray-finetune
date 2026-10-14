"""ray_train.py run identity and resume (LAYA-04, LAYA-06) on the tiny checkpoint, offline, on --device cpu.

The run_id comes from the command's canonical config and seed and names the Ray experiment dir, so the same command
always lands in storage/<run_id>/ and a different seed in another dir; --print-run-id prints it without starting Ray.
Runs in `make test` under the ray marker (CPU only, no weights or data).
"""
import os
import re
import subprocess
import sys

import pytest

import pilot.data
from conftest import make_raw
from pilot import runs
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


def _run(*args, ok=True):
    """Run a repo script with the test interpreter (no PYTHONPATH, so it imports like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    if ok:
        assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """(work dir, ARGS): the synthetic splits, the tiny base and the shared flags of every run in this module."""
    d = tmp_path_factory.mktemp("resume")
    root, raw_sha256 = make_raw(str(d / "raw"))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    base = tiny_checkpoint(d / "base")
    args = ["--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
            "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--device", "cpu", "--workers", "1",
            "--cpus-per-worker", "1", "--epochs", "2", "--max-messages", "12", "--calib-frac", "0.2", "--neg", "2"]
    return d, args


@pytest.fixture(scope="module")
def full_run(prepared):
    """(rid, out dir) of one uninterrupted run of ARGS into storage a."""
    d, args = prepared
    rid = _run("ray_train.py", *args, "--out", d / "x", "--storage", d / "a", "--print-run-id").stdout.strip()
    _run("ray_train.py", *args, "--storage", d / "a", "--out", d / "full")
    return rid, d / "full"


@pytest.mark.timeout(600)
def test_print_run_id_names_the_storage_dir(prepared, full_run):
    d, args = prepared
    r = _run("ray_train.py", *args, "--out", d / "x", "--storage", d / "p", "--print-run-id")
    lines = r.stdout.splitlines()
    assert len(lines) == 1 and re.fullmatch(r"laya-selection_train-neg2-s0-[0-9a-f]{8}", lines[0]), r.stdout
    assert not (d / "p").exists() and not (d / "x").exists()        # no Ray, no storage, no output
    rid, full = full_run
    assert lines[0] == rid
    assert (d / "a" / rid).is_dir()
    rec = runs.read_record(str(full))
    assert rec["run_id"] == rid and runs.run_id(rec["canonical"]) == rid
    s1 = _run("ray_train.py", *args, "--seed", "1", "--out", d / "x", "--storage", d / "p",
              "--print-run-id").stdout.strip()
    assert s1 != rid and "-s1-" in s1
