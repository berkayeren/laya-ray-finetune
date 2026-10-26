"""Offline CPU searches through tune.py on the tiny models and the synthetic snapshot (in `make test`, ray marker).

The synthetic snapshot (tests/conftest.py) gains 10 rows in each of folds 0-11, so the tiny models learn on
tuning-train while tuning-dev stays the two fold-12 and fold-13 rows. Laya: three pinned search points (the hidden
--points): a strong neg-8 config, a weak one at the space's lowest LRs and a strong 6-epoch config. BERTurk (the tiny
BERT base of tests/tiny_berturk.py): three 10-epoch points, a strong one, a weak one at the space's lowest LRs and a
strong batch-32 one; each trial is one process running berturk_train.fit. On CPU the runs are bit-reproducible, so ASHA
stops the weak trial at a rung every time, on a real score gap.
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
from tiny_berturk import tiny_base
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 900
LAYA_POINTS = [{"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 4},
               {"encoder_lr": 5e-06, "head_lr": 3e-05, "neg": 4, "epochs": 4},
               {"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 6}]
BERTURK_POINTS = [{"encoder_lr": 0.0001, "head_lr": 0.001, "batch": 16, "warmup_ratio": 0.06, "weight_decay": 0.0,
                   "epochs": 10},
                  {"encoder_lr": 5e-06, "head_lr": 3e-05, "batch": 32, "warmup_ratio": 0.1, "weight_decay": 0.01,
                   "epochs": 10},
                  {"encoder_lr": 0.0001, "head_lr": 0.001, "batch": 32, "warmup_ratio": 0.1, "weight_decay": 0.0,
                   "epochs": 10}]
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


Snapshot = tuple[Path, str, str, str, str]       # (work dir, selection_train, manifest, tiny Laya base, tiny BERT base)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Snapshot:
    """(work dir, selection_train path, manifest, tiny Laya base, tiny BERT base) of the enlarged synthetic
    snapshot."""
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
            tiny_checkpoint(d / "base"), tiny_base(d / "bert_base"))


LayaSearch = tuple[Path, Path, str]                 # (storage, out, guarded status before the search)


@pytest.fixture(scope="module")
def laya_search(snapshot: Snapshot) -> LayaSearch:
    """The CPU Laya smoke search over LAYA_POINTS, run once for the tests that read its record or storage."""
    d, data, manifest, base, _ = snapshot
    storage, out = d / "s", d / "laya.json"
    before = _guarded_status()
    _run("tune.py", "--system", "laya", "--device", "cpu", "--cpus", "1", "--trials", "3", "--seed", "0",
         "--base", base, "--data", data, "--manifest", manifest, "--storage", storage, "--out", out,
         "--points", json.dumps(LAYA_POINTS))
    return storage, out, before


@pytest.mark.timeout(TIMEOUT)
def test_laya_search_on_cpu(snapshot: Snapshot, laya_search: LayaSearch) -> None:
    """tune.py --system laya runs every trial through ray_train.py's own loop, one at a time, scores tuning-dev after
    every epoch, lets ASHA stop the weak trial, keeps no weights and writes a tune/1 record read_record accepts."""
    _, data, _, _, _ = snapshot
    storage, out, before = laya_search
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
    # the search file: one driver start, finished, bound to the record's bytes
    with open(os.path.join(str(storage), "laya-search.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert list(doc) == ["spec", "segments", "finished", "record_sha256"]
    assert doc["finished"] is True and doc["record_sha256"] == _sha and doc["segments"] == record["segments"]
    assert len(doc["segments"]) == 1 and doc["segments"][0]["preflight"] is None
    assert _guarded_status() == before                      # nothing under results/, data/ or protocol.json changed


def test_trials_never_overlap(laya_search: LayaSearch) -> None:
    """One trial at a time (the generator's max_concurrent=1): every trial's first start is later than the last epoch
    report (Tune's integer-second timestamp) of the trial before it."""
    storage, _, _ = laya_search
    exp = os.path.join(str(storage), "laya")
    for k in range(1, len(LAYA_POINTS)):
        with open(os.path.join(exp, pt.trial_dirname_of(k), pt.ATTEMPTS_FILE), encoding="utf-8") as f:
            started = json.load(f)["started"][0]
        with open(os.path.join(exp, pt.trial_dirname_of(k - 1), "result.json"), encoding="utf-8") as f:
            last = max(json.loads(line)["timestamp"] for line in f if line.strip())
        assert started > last, (k, started, last)


def _run_logged(args: list[Any], log: Path, env_extra: dict[str, str]) -> tuple[int, str]:
    """(exit code, stdout + stderr) of a repo script run with the test interpreter, its output in a file (a killed
    driver's Ray children never hold a pipe open)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(env_extra)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, stdout=f,
                           stderr=subprocess.STDOUT, timeout=TIMEOUT)
    return r.returncode, log.read_text(encoding="utf-8")


def _search_file(storage: Path) -> dict[str, Any]:
    with open(storage / "laya-search.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.timeout(1200)
def test_laya_search_resumes_after_a_kill(snapshot: Snapshot) -> None:
    """Trial 1's first attempt SIGKILLs the driver (the hidden --kill-at-trial). Rerunning the identical command
    continues the search through Tuner.restore: trial 0 is not rerun, trial 1 restarts from scratch (attempts 2) and
    the record holds two segments. A third run of the same command starts no Ray and finds the record unchanged."""
    d, data, manifest, base, _ = snapshot
    storage, out = d / "kill", d / "kill.json"
    before = _guarded_status()
    argv = ["tune.py", "--system", "laya", "--device", "cpu", "--cpus", "1", "--trials", "3", "--seed", "0",
            "--base", base, "--data", data, "--manifest", manifest, "--storage", storage, "--out", out,
            "--points", json.dumps(LAYA_POINTS), "--kill-at-trial", "1"]
    # Ray reads TUNE_GLOBAL_CHECKPOINT_S (tune.py reads no environment variable): 0 snapshots the experiment every
    # controller step, so trial 0's finished state is on disk before trial 1 starts and the kill point is fixed
    env = {"TUNE_GLOBAL_CHECKPOINT_S": "0"}
    code, text = _run_logged(argv, d / "kill-1.log", env)
    assert code != 0, text[-4000:]
    assert not out.exists()
    doc = _search_file(storage)
    assert len(doc["segments"]) == 1 and doc["finished"] is False and doc["record_sha256"] is None
    code, text = _run_logged(argv, d / "kill-2.log", env)
    assert code == 0, text[-4000:]
    assert "continuing the search" in text
    record, sha = pt.read_record(str(out))
    assert [t["attempts"] for t in record["trials"]] == [1, 2, 1]
    assert [t["config"] for t in record["trials"]] == LAYA_POINTS
    assert len(record["segments"]) == 2
    doc = _search_file(storage)
    assert doc["finished"] is True and doc["record_sha256"] == sha and doc["segments"] == record["segments"]
    code, text = _run_logged(argv, d / "kill-3.log", env)
    assert code == 0, text[-4000:]
    assert "search finished; record unchanged -> %s (sha256 %s)" % (out, sha) in text
    assert "Started a local Ray instance" not in text
    assert pt.read_record(str(out))[1] == sha
    assert len(_search_file(storage)["segments"]) == 2
    assert _guarded_status() == before


def _check_record(record: dict[str, Any], storage: Path, system: str, data: str) -> None:
    """The partition counts and digests, the notes, and n_train_messages / n_dev_messages in every result.json line."""
    with open(data, "rb") as f:
        train_rows, dev_rows, block = pt.partition(pilot.data.parse_jsonl(f.read()))
    assert record["data"]["n_train"] == len(train_rows) and record["data"]["n_dev"] == len(dev_rows) == 2
    assert record["data"]["train_uids_sha256"] == block["train_uids_sha256"]
    assert record["data"]["dev_uids_sha256"] == block["dev_uids_sha256"]
    assert record["notes"] == list(pt.NOTES)
    lines = 0
    for path in glob.glob(os.path.join(str(storage), system, "trial_*", "result.json")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    res = json.loads(line)
                    assert res["n_train_messages"] == record["data"]["n_train"], path
                    assert res["n_dev_messages"] == record["data"]["n_dev"], path
                    lines += 1
    assert lines == sum(len(t["curve"]) for t in record["trials"])


@pytest.mark.timeout(TIMEOUT)
def test_berturk_search_on_cpu(snapshot: Snapshot) -> None:
    """tune.py --system berturk runs every trial as one process through berturk_train.fit (berturk_train.py's own HF
    Trainer recipe with the trial's LRs, batch, warm-up, weight decay and epochs), one at a time, scores tuning-dev
    through score.bert_logits after every epoch, lets ASHA stop the weak trial, keeps no weights and writes a tune/1
    record read_record accepts; every finished trial's done.json names the tuning partition as its tune key."""
    d, data, manifest, _, bert = snapshot
    storage, out = d / "sb", d / "berturk.json"
    before = _guarded_status()
    _run("tune.py", "--system", "berturk", "--device", "cpu", "--cpus", "1", "--trials", "3", "--seed", "0",
         "--base", bert, "--data", data, "--manifest", manifest, "--storage", storage, "--out", out,
         "--points", json.dumps(BERTURK_POINTS))
    record, _sha = pt.read_record(str(out))
    assert record["system"] == "berturk" and record["schema"] == pt.SCHEMA
    assert [t["config"] for t in record["trials"]] == BERTURK_POINTS
    assert record["search"]["points_to_evaluate"] == BERTURK_POINTS
    assert record["search"]["scheduler"]["max_t"] == 30
    assert record["fixed"] == pt.FIXED["berturk"]
    assert record["n_asha_stopped"] >= 1 and any(t["asha_stopped"] for t in record["trials"])
    assert all(t["attempts"] == 1 for t in record["trials"])
    assert set(record["versions"]) == {"ray", "torch", "transformers", "accelerate"}
    _check_record(record, storage, "berturk", data)
    with open(data, "rb") as f:
        _, _, block = pt.partition(pilot.data.parse_jsonl(f.read()))
    done = glob.glob(os.path.join(str(storage), "berturk-train", "trial_*-a1", "*", "done.json"))
    assert done                                             # at least one trial ran to its last epoch
    for path in done:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["record"]["n_messages"] == record["data"]["n_train"], path
        assert doc["canonical"]["tune"] == block, path
        assert doc["record"]["epochs"][-1]["epoch"] == doc["epochs_schedule"] == 10, path
    # no weights: no safetensors, no export, and no trial wrote a final checkpoint
    for root, dirs, files in os.walk(str(storage)):
        assert not [n for n in files if n.endswith(".safetensors")], root
        assert "exports" not in dirs and not [n for n in dirs if n.startswith("epoch-")], root
    assert _guarded_status() == before                      # nothing under results/, data/ or protocol.json changed
