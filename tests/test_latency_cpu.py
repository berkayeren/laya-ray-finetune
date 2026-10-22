"""latency.py end to end on CPU, offline (EVAL-06): tiny final seeds bound to a synthetic postfit-final/1 manifest are
timed per message through score.py's load path, with warm-up and sync, into a labelled latency/1 report that renders
to markdown.

The tiny Laya checkpoint comes from tests/tiny_laya.py, the tiny BERTurk export from berturk_train.py on CPU (1 epoch,
as tests/test_score_berturk.py builds it); the data is the synthetic snapshot of tests/conftest.py. The quiet-machine
preflight is replaced by a recorded dict (no 5 s sleep). Offline, no Ray, no weights: runs in `make test`.
"""
import json
import os
import subprocess
import sys

import pytest

import latency
import pilot.data
from conftest import make_raw, prepared
from pilot import berturk, dumps
from pilot import latency as plat
from pilot.weights import identity
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600
TINY_T = 1.3
PREFLIGHT = {"available_gib": 12.0, "min_available_gib": 7.0, "swap_used_bytes": [0, 0],
             "swap_grew_while_idle": False, "swapouts": [None, None], "swapouts_grew_while_idle": None,
             "passed": True, "override": False}


@pytest.fixture(autouse=True)
def _threads(monkeypatch):
    """latency.py sets torch's process-wide thread count; restored so later tests keep the session's setting. The
    preflight is a recorded dict, so no test sleeps or depends on this machine's free memory."""
    import torch
    n = torch.get_num_threads()
    monkeypatch.setattr(latency, "preflight", lambda min_gib: dict(PREFLIGHT, min_available_gib=min_gib))
    yield
    torch.set_num_threads(n)


def final_manifest(path, seeds, system="laya"):
    """Write a synthetic postfit-final/1 manifest of `seeds` [(seed, checkpoint_name, identity, dump_run_id, T)]."""
    m = {"schema": "postfit-final/1", "postfit": {"path": "results/postfit/synthetic.json", "sha256": "0" * 64},
         "final_split": "final_train", "chosen": {},
         "seeds": [{"seed": seed, "train_run_id": "%s-final_train-s%d-synthetic" % (system, seed),
                    "checkpoint_name": name, "checkpoint_identity": ident, "dump_run_id": rid, "temperature": t}
                   for seed, name, ident, rid, t in seeds]}
    if system == "berturk":
        m["system_kind"] = "berturk"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    return str(path)


def laya_setup(tmp_path, synthetic_raw):
    """(splits dir, splits manifest, final manifest, checkpoint root) of one tiny Laya seed with T = 1.3."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    root = tmp_path / "ckpts"
    ckpt = tiny_checkpoint(root / "ckpt-s0", temperature=TINY_T)
    ident = identity(ckpt)
    m = final_manifest(tmp_path / "laya-final.json", [(0, "ckpt-s0", ident, dumps.run_id("laya", ident), TINY_T)])
    return os.path.join(out_dir, "splits"), manifest, m, str(root)


def run_args(system, splits, manifest, final, root, out, *extra):
    return ["--system", system, "--device", "cpu", "--final-manifest", final, "--checkpoint-root", root,
            "--split", os.path.join(splits, "val.jsonl"), "--manifest", manifest, "--warmup", "1",
            "--out", str(out)] + list(extra)


def val_cases(splits):
    with open(os.path.join(splits, "val.jsonl"), "rb") as f:
        return pilot.data.parse_jsonl(f.read())


def check_report(report, out, splits, system):
    cases = val_cases(splits)
    assert report["schema"] == "latency/1" and report["label"] == "reported, not gated"
    assert report["system"] == system and report["device"] == "cpu" and report["warmup"] == 1
    assert len(report["seeds"]) == 1
    row = report["seeds"][0]
    assert row["state"] == {"device": "cpu", "dtype": "torch.float32", "amp_enabled": False}
    s = row["summary"]
    assert s["n"] == len(cases) == len(row["times_s"]) == report["split"]["n_messages"]
    assert 0 < s["min"] <= s["p50"] <= s["p95"] <= s["max"] and s["min"] <= s["mean"] <= s["max"]
    assert report["pooled"] == s                            # one seed: the pooled summary is that seed's
    assert report["final_manifest"]["is_committed"] is False
    assert report["split"]["uids_sha256"] == dumps.uids_sha256([c["uid"] for c in cases])
    with open(out, encoding="utf-8") as f:
        text = f.read()
    assert json.loads(text) == report
    for c in cases:                                         # times only: no message text or uid in the report
        assert c["state"] not in text and '"%s"' % c["uid"] not in text
    assert str(REPO) not in text and os.path.expanduser("~") not in text


def test_tiny_laya_latency_on_cpu(tmp_path, synthetic_raw, capsys):
    splits, manifest, m, root = laya_setup(tmp_path, synthetic_raw)
    out = tmp_path / "latency" / "laya-cpu.json"
    report = latency.main(run_args("laya", splits, manifest, m, root, out))
    check_report(report, out, splits, "laya")
    assert report["seeds"][0]["temperature"] == TINY_T and "laya" in report["versions"]
    assert report["timed_path"] == plat.TIMED_PATH["laya"]
    capsys.readouterr()
    md = latency.main(["--render", str(out.parent)])
    assert capsys.readouterr().out == md
    assert "reported, not gated" in md and "| pooled |" in md and "Busy-machine" not in md


# ---- BERTurk ----

@pytest.fixture(scope="module")
def berturk_built(tmp_path_factory):
    """(splits dir, splits manifest, a 1-epoch CPU export of the tiny BERTurk base), built once for the module."""
    from tiny_berturk import tiny_base
    tmp = tmp_path_factory.mktemp("latency_berturk")
    root, raw_sha256 = make_raw(str(tmp / "raw"))
    out_dir = str(tmp / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    manifest = os.path.join(out_dir, "splits.json")
    splits = os.path.join(out_dir, "splits")
    base = tiny_base(tmp / "base")
    storage = tmp / "runs"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [sys.executable, "berturk_train.py", "--data", os.path.join(splits, "selection_train.jsonl"),
            "--manifest", manifest, "--base", base, "--device", "cpu", "--epochs", "1", "--batch", "4",
            "--max-len", "32", "--seed", "0", "--storage", str(storage)]
    rid = subprocess.run(argv + ["--print-run-id"], cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=TIMEOUT, check=True).stdout.strip()
    r = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, r.stderr[-4000:]
    return splits, manifest, str(storage / rid / "exports")


def berturk_setup(berturk_built, tmp_path, change=None):
    """(final manifest, checkpoint root) of the module's export as seed 0 "epoch-001" (T 1.0); `change` edits the
    recorded identity."""
    _, _, root = berturk_built
    ckpt = os.path.join(root, "epoch-001")
    ident = berturk.identity(ckpt)
    rid = berturk.dump_run_id(ident)
    if change is not None:
        ident = json.loads(json.dumps(ident))
        change(ident)
    m = final_manifest(tmp_path / "berturk-final.json", [(0, "epoch-001", ident, rid, 1.0)], system="berturk")
    return m, root


def test_tiny_berturk_latency_on_cpu(berturk_built, tmp_path):
    splits, manifest, _ = berturk_built
    m, root = berturk_setup(berturk_built, tmp_path)
    out = tmp_path / "latency" / "berturk-cpu.json"
    report = latency.main(run_args("berturk", splits, manifest, m, root, out))
    check_report(report, out, splits, "berturk")
    assert report["seeds"][0]["temperature"] == 1.0 and "laya" not in report["versions"]
    assert report["timed_path"] == plat.TIMED_PATH["berturk"]


def test_berturk_identity_mismatch_refused_before_load(berturk_built, tmp_path, monkeypatch):
    """One digest of the recorded identity changed: refused before score.load_bert (or any check of the files)."""
    def boom(*a, **k):
        raise AssertionError("a BERTurk checkpoint was loaded before the refusal")
    monkeypatch.setattr(latency.score, "load_bert", boom)
    monkeypatch.setattr(latency.score, "check_berturk", boom)
    splits, manifest, _ = berturk_built
    m, root = berturk_setup(berturk_built, tmp_path, change=lambda ident: ident.update(model_sha256="0" * 64))
    out = tmp_path / "latency" / "berturk-cpu.json"
    with pytest.raises(SystemExit, match="not the checkpoint the final manifest names for seed 0"):
        latency.main(run_args("berturk", splits, manifest, m, root, out))
    assert not out.exists()
