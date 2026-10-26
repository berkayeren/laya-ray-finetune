"""Offline CPU searches through tune.py on the tiny models and the synthetic snapshot (in `make test`, ray marker).

The synthetic snapshot (tests/conftest.py) gains 10 rows in each of folds 0-11, so the tiny Laya model learns on
tuning-train while tuning-dev stays the two fold-12 and fold-13 rows. Three pinned search points (the hidden --points):
a strong neg-8 config, a weak one at the space's lowest LRs and a strong 6-epoch config. On CPU the runs are
bit-reproducible, so ASHA stops the weak trial at a rung every time, on a real score gap.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pilot.data
from conftest import add_raw_row, make_raw
from pilot import tune as pt
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 900
LAYA_POINTS = [{"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 4},
               {"encoder_lr": 5e-06, "head_lr": 3e-05, "neg": 4, "epochs": 4},
               {"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 6}]
GUARDED = ("results", "data", "protocol.json")


def _run(*args: Any) -> subprocess.CompletedProcess:
    """Run a repo script with the test interpreter (no PYTHONPATH, so it imports like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


def _guarded_status() -> str:
    """git status --porcelain of results/, data/ and protocol.json of the repo."""
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *GUARDED], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str, str]:
    """(work dir, selection_train path, manifest, tiny Laya base) of the enlarged synthetic snapshot."""
    d = tmp_path_factory.mktemp("tune")
    root, raw_sha256 = make_raw(str(d / "raw"))
    for fold in range(12):                                  # tuning-train rows the tiny model can learn from
        for j in range(10):
            row = ({"text": "Kartım kayboldu", "intents": ["card", "lost_stolen"]} if j % 2 == 0
                   else {"text": "Evet .", "intents": ["affirm"]})
            add_raw_row(root, raw_sha256, "banking", fold, dict(row, uid="bank_%d_%d" % (fold, 10 + j)))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    return (d, os.path.join(out_dir, "splits", "selection_train.jsonl"), os.path.join(out_dir, "splits.json"),
            tiny_checkpoint(d / "base"))


@pytest.mark.timeout(TIMEOUT)
def test_laya_search_on_cpu(snapshot: tuple[Path, str, str, str]) -> None:
    """tune.py --system laya runs every trial through ray_train.py's own loop, one at a time, scores tuning-dev after
    every epoch, lets ASHA stop the weak trial, keeps no weights and writes a tune/1 record read_record accepts."""
    d, data, manifest, base = snapshot
    storage, out = d / "s", d / "laya.json"
    before = _guarded_status()
    _run("tune.py", "--system", "laya", "--device", "cpu", "--cpus", "1", "--trials", "3", "--seed", "0",
         "--base", base, "--data", data, "--manifest", manifest, "--storage", storage, "--out", out,
         "--points", json.dumps(LAYA_POINTS))
    record, _sha = pt.read_record(str(out))
    assert record["system"] == "laya" and record["schema"] == pt.SCHEMA
    assert [t["config"] for t in record["trials"]] == LAYA_POINTS
    assert record["search"]["points_to_evaluate"] == LAYA_POINTS
    assert record["n_asha_stopped"] >= 1 and any(t["asha_stopped"] for t in record["trials"])
    assert all(t["attempts"] == 1 for t in record["trials"])
    with open(data, "rb") as f:
        train_rows, dev_rows, block = pt.partition(pilot.data.parse_jsonl(f.read()))
    assert record["data"]["n_train"] == len(train_rows) and record["data"]["n_dev"] == len(dev_rows) == 2
    assert record["data"]["train_uids_sha256"] == block["train_uids_sha256"]
    assert record["data"]["dev_uids_sha256"] == block["dev_uids_sha256"]
    assert record["notes"] == list(pt.NOTES)
    lines = 0
    for path in glob.glob(os.path.join(str(storage), "laya", "trial_*", "result.json")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    res = json.loads(line)
                    assert res["n_train_messages"] == record["data"]["n_train"], path
                    assert res["n_dev_messages"] == record["data"]["n_dev"], path
                    lines += 1
    assert lines == sum(len(t["curve"]) for t in record["trials"])
    # no weights: no safetensors, no export, no epoch checkpoint, and no trial's --out path
    for root, dirs, files in os.walk(str(storage)):
        assert not [n for n in files if n.endswith(".safetensors")], root
        assert "exports" not in dirs and not [n for n in dirs if n.startswith("epoch-")], root
        assert os.path.basename(root) != "out", root
    assert not glob.glob(os.path.join(str(storage), "laya-train", "*", "out"))
    assert _guarded_status() == before                      # nothing under results/, data/ or protocol.json changed
