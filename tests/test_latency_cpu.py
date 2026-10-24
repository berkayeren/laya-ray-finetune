"""latency.py end to end on CPU, offline: tiny final seeds bound to a synthetic postfit-final/1 manifest are timed per
message through score.py's load path, each seed after its own untimed pass over every val message (the full-pass
warm-up, shown on two seeds in one process) and with sync, into a labelled latency/1 report that
renders to markdown.

The tiny Laya checkpoint comes from tests/tiny_laya.py, the tiny BERTurk export from berturk_train.py on CPU (1 epoch,
as tests/test_score_berturk.py builds it); the data is the synthetic snapshot of tests/conftest.py. The quiet-machine
preflight is replaced by a recorded dict (no 5 s sleep). Offline, no Ray, no weights: runs in `make test`.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, NoReturn

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
def _threads(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """latency.py sets torch's process-wide thread count; restored so later tests keep the session's setting. The
    preflight is a recorded dict, so no test sleeps or depends on this machine's free memory."""
    import torch
    n = torch.get_num_threads()
    monkeypatch.setattr(latency, "preflight", lambda min_gib: dict(PREFLIGHT, min_available_gib=min_gib))
    yield
    torch.set_num_threads(n)


def final_manifest(path: Path, seeds: list[tuple[int, str, dict[str, Any], str, float]], system: str = "laya") -> str:
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


def laya_setup(tmp_path: Path, synthetic_raw: Callable[..., tuple[str, dict[str, str]]],
               seeds: tuple[int, ...] = (0,)) -> tuple[str, str, str, str]:
    """(splits dir, splits manifest, final manifest, checkpoint root) of one tiny Laya seed per entry of `seeds`
    (checkpoint ckpt-s<seed>, T = 1.3), listed in that order."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    root = tmp_path / "ckpts"
    rows = []
    for seed in seeds:
        name = "ckpt-s%d" % seed
        ident = identity(tiny_checkpoint(root / name, temperature=TINY_T, seed=seed))
        rows.append((seed, name, ident, dumps.run_id("laya", ident), TINY_T))
    m = final_manifest(tmp_path / "laya-final.json", rows)
    return os.path.join(out_dir, "splits"), manifest, m, str(root)


def run_args(system: str, splits: str, manifest: str, final: str, root: str, out: Path | str, *extra: str) -> list[str]:
    """The latency.py argv timing `system` on cpu over splits/val.jsonl into `out`, plus `extra`."""
    return ["--system", system, "--device", "cpu", "--final-manifest", final, "--checkpoint-root", root,
            "--split", os.path.join(splits, "val.jsonl"), "--manifest", manifest, "--out", str(out)] + list(extra)


def val_cases(splits: str) -> list[dict[str, Any]]:
    """The parsed rows of splits/val.jsonl, in file order."""
    with open(os.path.join(splits, "val.jsonl"), "rb") as f:
        return pilot.data.parse_jsonl(f.read())


def check_report(report: dict[str, Any], out: Path, splits: str, system: str) -> None:
    """Assert the shape of a one-seed cpu latency/1 report and that its file holds only times (no text, uid or path)."""
    cases = val_cases(splits)
    assert report["schema"] == "latency/1" and report["label"] == "reported, not gated"
    assert report["system"] == system and report["device"] == "cpu"
    assert report["warmup"] == len(cases) and report["warmup_policy"] == plat.WARMUP_POLICY
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


def test_tiny_laya_latency_on_cpu(tmp_path: Path, synthetic_raw: Callable[..., tuple[str, dict[str, str]]],
                                  capsys: pytest.CaptureFixture[str]) -> None:
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


def test_every_seed_warms_every_timed_message_first(tmp_path: Path,
                                                    synthetic_raw: Callable[..., tuple[str, dict[str, str]]],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Two tiny Laya seeds timed in one process each follow the identical sequence,
    whatever their position: load, one untimed forward of exactly the val texts in split-file order, one sync, then
    every message timed (sync, clock, forward, sync, clock). So no seed's timed loop absorbs a first-occurrence cost
    that another seed's warm-up paid."""
    splits, manifest, m, root = laya_setup(tmp_path, synthetic_raw, seeds=(0, 1))
    texts = [c["state"] for c in val_cases(splits)]
    n = len(texts)
    events, forwards = [], []
    real_time, real_predictor, real_noul = plat.time_messages, latency.score.Predictor, latency.la.noul_logits

    def recorder(fn: Callable[[str], Any], texts_: list[str], sync: Callable[[], Any],
                 clock: Callable[[], float] = time.perf_counter) -> list[float]:
        events.append(("seed",))

        def fwd(t: str) -> Any:
            events.append(("fwd", t))
            return fn(t)

        def rsync() -> Any:
            events.append(("sync",))
            return sync()

        def rclock() -> float:
            events.append(("clock",))
            return clock()
        return real_time(fwd, texts_, rsync, clock=rclock)

    def predictor(*args: Any, **kwargs: Any) -> Any:
        events.append(("load",))
        return real_predictor(*args, **kwargs)

    def noul(*args: Any, **kwargs: Any) -> Any:
        forwards.append(1)
        return real_noul(*args, **kwargs)
    monkeypatch.setattr(plat, "time_messages", recorder)
    monkeypatch.setattr(latency.score, "Predictor", predictor)
    monkeypatch.setattr(latency.la, "noul_logits", noul)
    report = latency.main(run_args("laya", splits, manifest, m, root, tmp_path / "out.json"))

    starts = [i for i, e in enumerate(events) if e == ("load",)]
    assert len(starts) == 2
    blocks = [events[a:b] for a, b in zip(starts, starts[1:] + [len(events)])]
    expected = [("load",), ("seed",)] + [("fwd", t) for t in texts] + [("sync",)]
    for t in texts:
        expected += [("sync",), ("clock",), ("fwd", t), ("sync",), ("clock",)]
    for block in blocks:
        assert block == expected                            # identical for seed 0 and seed 1
        warmed = {e[1] for e in block[2:2 + n]}
        assert all(t in warmed for t in texts)
    assert len(forwards) == 2 * 2 * n                       # 2 seeds x (warm-up pass + timed loop) x n messages
    assert [s["seed"] for s in report["seeds"]] == [0, 1]
    assert all(len(s["times_s"]) == n for s in report["seeds"])
    assert report["warmup"] == n and report["warmup_policy"] == plat.WARMUP_POLICY


# ---- BERTurk ----

@pytest.fixture(scope="module")
def berturk_built(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str, str]:
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


def berturk_setup(berturk_built: tuple[str, str, str], tmp_path: Path,
                  change: Callable[[dict[str, Any]], Any] | None = None) -> tuple[str, str]:
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


def test_tiny_berturk_latency_on_cpu(berturk_built: tuple[str, str, str], tmp_path: Path) -> None:
    splits, manifest, _ = berturk_built
    m, root = berturk_setup(berturk_built, tmp_path)
    out = tmp_path / "latency" / "berturk-cpu.json"
    report = latency.main(run_args("berturk", splits, manifest, m, root, out))
    check_report(report, out, splits, "berturk")
    assert report["seeds"][0]["temperature"] == 1.0 and "laya" not in report["versions"]
    assert report["timed_path"] == plat.TIMED_PATH["berturk"]


def test_berturk_identity_mismatch_refused_before_load(berturk_built: tuple[str, str, str], tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """One digest of the recorded identity changed: refused before score.load_bert (or any check of the files)."""
    def boom(*a: Any, **k: Any) -> NoReturn:
        raise AssertionError("a BERTurk checkpoint was loaded before the refusal")
    monkeypatch.setattr(latency.score, "load_bert", boom)
    monkeypatch.setattr(latency.score, "check_berturk", boom)
    splits, manifest, _ = berturk_built
    m, root = berturk_setup(berturk_built, tmp_path, change=lambda ident: ident.update(model_sha256="0" * 64))
    out = tmp_path / "latency" / "berturk-cpu.json"
    with pytest.raises(SystemExit, match="not the checkpoint the final manifest names for seed 0"):
        latency.main(run_args("berturk", splits, manifest, m, root, out))
    assert not out.exists()
