"""tune.py's guards and modes, in-process with ray.init replaced by a refusal: the results/ guard (only the committed
search writes there), the search file's spec check, the unrestorable experiment dir, the MPS preflight in the segment,
--check, --status and the mode exclusivity.

The synthetic snapshot (tests/conftest.py) and a tiny Laya checkpoint (tests/tiny_laya.py); nothing is ever written
under the repo's results/, data/ or protocol.json.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import pilot.data
import tune
from conftest import make_raw
from pilot import tune as pt
from test_tune_unit import _canonical, _result
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
GUARDED = ("results", "data", "protocol.json")
COMMITTED_OUT = os.path.join(REPO, "results", "tune", "laya.json")


class RayReached(Exception):
    """ray.init was reached: every refusal under test must come first."""


def _guarded_status() -> str:
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *GUARDED], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def snap(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str, str]:
    """(selection_train path, manifest, tiny Laya base) of the synthetic snapshot."""
    d = tmp_path_factory.mktemp("tune_record")
    root, raw_sha256 = make_raw(str(d / "raw"))
    pilot.data.prepare(root, str(d / "data"), raw_sha256)
    return (str(d / "data" / "splits" / "selection_train.jsonl"), str(d / "data" / "splits.json"),
            tiny_checkpoint(d / "base"))


@pytest.fixture
def no_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    import ray

    def boom(*a: Any, **kw: Any) -> None:
        raise RayReached("ray.init reached")
    monkeypatch.setattr(ray, "init", boom)


def _smoke(snap: tuple[str, str, str], storage: Path, out: Path, *extra: str) -> list[str]:
    """The argv of a 3-trial CPU search on the tiny base and the synthetic snapshot."""
    data, manifest, base = snap
    return ["--system", "laya", "--device", "cpu", "--cpus", "1", "--trials", "3", "--seed", "0", "--base", base,
            "--data", data, "--manifest", manifest, "--storage", str(storage), "--out", str(out), *extra]


# ---- the results/ guard ----

DEPARTURES = [(["--trials", "3"], "--trials"), (["--seed", "1"], "--seed"), (["--device", "cpu"], "--device"),
              (["--points", json.dumps([{"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 4}])],
               "--points"),
              (["--kill-at-trial", "1"], "--kill-at-trial"), ("BASE", "--base"), ("MANIFEST", "--manifest")]


@pytest.mark.parametrize("extra, flag", DEPARTURES, ids=[f for _, f in DEPARTURES])
def test_results_out_is_the_committed_search_only(extra: Any, flag: str, snap: tuple[str, str, str], tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch, no_ray: None) -> None:
    """--out results/tune/laya.json is refused, naming the departure, for anything but the committed search (20
    trials, seed 0, MPS, the pinned base, the committed manifest, no --points, no test hook), before bind (no fetch,
    no split read) and before ray.init."""
    data, manifest, base = snap
    if extra == "BASE":
        extra = ["--base", base]
    elif extra == "MANIFEST":
        extra = ["--manifest", manifest]

    def no_bind(*a: Any, **kw: Any) -> None:
        raise AssertionError("bind reached; the results/ guard must refuse first")
    monkeypatch.setattr(tune, "bind", no_bind)
    before = _guarded_status()
    with pytest.raises(SystemExit) as e:
        tune.main(["--system", "laya", "--out", COMMITTED_OUT, "--storage", str(tmp_path / "s"), *extra])
    msg = str(e.value.code)
    assert flag in msg and "the committed search" in msg and "pass another --out" in msg, msg
    assert not (tmp_path / "s").exists()
    assert _guarded_status() == before == ""


@pytest.mark.parametrize("name", ["laya.json", os.path.join("tune", "x.json"), os.path.join("tune", "berturk.json")])
@pytest.mark.parametrize("settings", ["committed", "smoke"])
def test_results_out_elsewhere_is_refused(name: str, settings: str, snap: tuple[str, str, str], tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch, no_ray: None) -> None:
    """Under results/ a --system laya search may write results/tune/laya.json only, whatever its settings."""
    monkeypatch.setattr(tune, "bind", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("bind reached")))
    out = os.path.join(REPO, "results", name)
    argv = (["--system", "laya", "--out", out, "--storage", str(tmp_path / "s")] if settings == "committed"
            else _smoke(snap, tmp_path / "s", Path(out)))
    with pytest.raises(SystemExit, match="results/tune/laya.json"):
        tune.main(argv)
    assert not os.path.exists(out) and _guarded_status() == ""


def test_the_committed_search_passes_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed search's own settings reach bind (stubbed here: the guard is all this test runs)."""
    def stop(*a: Any, **kw: Any) -> None:
        raise RayReached("bind reached")
    monkeypatch.setattr(tune, "bind", stop)
    with pytest.raises(RayReached):
        tune.main(["--system", "laya"])
    with pytest.raises(RayReached):
        tune.main(["--system", "berturk", "--out", os.path.join(REPO, "results", "tune", "berturk.json")])


# ---- the search file ----

def test_a_changed_spec_is_refused(snap: tuple[str, str, str], tmp_path: Path, no_ray: None) -> None:
    """The first start writes STORAGE/laya-search.json (one segment, not finished) before ray.init; a start of another
    spec is refused naming the key, before ray.init."""
    storage, out = tmp_path / "s", tmp_path / "laya.json"
    with pytest.raises(RayReached):
        tune.main(_smoke(snap, storage, out))
    doc = json.loads((storage / "laya-search.json").read_text(encoding="utf-8"))
    assert list(doc) == ["spec", "segments", "finished", "record_sha256"]
    assert doc["finished"] is False and doc["record_sha256"] is None and len(doc["segments"]) == 1
    assert list(doc["segments"][0]) == list(pt.SEGMENT_KEYS) and doc["segments"][0]["preflight"] is None
    assert doc["spec"]["seed"] == 0 and doc["spec"]["n_trials"] == 3 and doc["spec"]["system"] == "laya"
    text = json.dumps(doc)
    assert str(tmp_path) not in text and snap[2] not in text  # no path in the spec
    for flag, value, key in (("--seed", "1", "seed"), ("--trials", "4", "n_trials")):
        argv = _smoke(snap, storage, out)
        argv[argv.index(flag) + 1] = value
        with pytest.raises(SystemExit, match=r"another spec \(key %s\)" % key):
            tune.main(argv)
    with pytest.raises(RayReached):                         # the identical command appends its segment
        tune.main(_smoke(snap, storage, out, "--kill-at-trial", "1"))
    assert len(json.loads((storage / "laya-search.json").read_text(encoding="utf-8"))["segments"]) == 2


def test_an_unrestorable_experiment_dir_is_refused(snap: tuple[str, str, str], tmp_path: Path, no_ray: None) -> None:
    """STORAGE/laya holding a stray file and no Tune state is refused before ray.init, naming the three paths to
    delete; so is an experiment dir without its search file."""
    storage, out = tmp_path / "s", tmp_path / "laya.json"
    with pytest.raises(RayReached):
        tune.main(_smoke(snap, storage, out))
    (storage / "laya").mkdir()
    (storage / "laya" / "stray.txt").write_text("x", encoding="utf-8")
    paths = [str(storage / "laya"), str(storage / "laya-train"), str(storage / "laya-search.json")]
    with pytest.raises(SystemExit, match="cannot restore") as e:
        tune.main(_smoke(snap, storage, out))
    assert all(p in str(e.value.code) for p in paths), e.value.code
    (storage / "laya-search.json").unlink()
    with pytest.raises(SystemExit, match="without its search file") as e:
        tune.main(_smoke(snap, storage, out))
    assert all(p in str(e.value.code) for p in paths), e.value.code


@pytest.mark.parametrize("gib", [None, "0"])
def test_mps_preflight_lands_in_the_segment(gib: str | None, snap: tuple[str, str, str], tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch, no_ray: None) -> None:
    """On --device mps compute_checkpoint.preflight(--min-available-gib, default 7.0) runs before ray.init and its
    dict is the segment's preflight; on --device cpu preflight is None and the preflight never runs."""
    import torch

    import compute_checkpoint
    seen = []
    pf = {"available_gib": 11.5, "min_available_gib": 7.0, "override": False, "passed": True,
          "swap_grew_while_idle": False}

    def fake(min_available_gib: float) -> dict[str, Any]:
        seen.append(min_available_gib)
        return dict(pf)
    monkeypatch.setattr(compute_checkpoint, "preflight", fake)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    storage = tmp_path / "s"
    argv = [x if x != "cpu" else "mps" for x in _smoke(snap, storage, tmp_path / "laya.json")]
    with pytest.raises(RayReached):
        tune.main(argv + (["--min-available-gib", gib] if gib else []))
    doc = json.loads((storage / "laya-search.json").read_text(encoding="utf-8"))
    assert doc["spec"]["device"] == "mps" and doc["segments"][0]["preflight"] == pf
    assert seen == [7.0 if gib is None else 0.0]
    monkeypatch.setattr(compute_checkpoint, "preflight",
                        lambda *a: (_ for _ in ()).throw(AssertionError("preflight on cpu")))
    with pytest.raises(RayReached):
        tune.main(_smoke(snap, tmp_path / "c", tmp_path / "c.json"))
    assert json.loads((tmp_path / "c" / "laya-search.json").read_text(encoding="utf-8"))["segments"][0][
        "preflight"] is None


# ---- --check and --status ----

def test_check_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--check RECORD prints ok with the record's sha256; one changed byte is the read_record refusal (exit 1)."""
    path = pt.write_record(pt.build_record(_canonical()), str(tmp_path / "laya.json"))
    _, sha = pt.read_record(path)
    tune.main(["--check", path])
    assert capsys.readouterr().out.strip() == "ok %s %s" % (path, sha)
    raw = Path(path).read_bytes()
    Path(path).write_bytes(raw[:-1] + b" ")
    with pytest.raises(SystemExit, match="byte layout") as e:
        tune.main(["--check", path])
    assert e.value.code != 0


def _tree(root: Path) -> list[tuple[str, int, float]]:
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime) for p in root.rglob("*"))


def test_status_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str], no_ray: None) -> None:
    """--status --system laya --storage STORAGE prints the driver starts and one line per started trial (index,
    attempts, last epoch of its schedule, best dev micro-F1) from STORAGE alone: no Ray, nothing written."""
    storage = tmp_path / "s"
    seg = {"git_sha": "f" * 40, "git_dirty": False, "git_diff_sha256": "0" * 64, "preflight": None}
    tune.write_search(str(storage / "laya-search.json"),
                      {"spec": {"system": "laya", "n_trials": 3}, "segments": [seg, seg], "finished": False,
                       "record_sha256": None})
    curves = {0: [_result(1, 1, 5, 3), _result(2, 2, 2, 2)], 1: [_result(1, 1, 1, 1)]}
    for i, curve in curves.items():
        d = storage / "laya" / pt.trial_dirname_of(i)
        d.mkdir(parents=True)
        (d / "params.json").write_text(json.dumps({"encoder_lr": 1e-4, "head_lr": 1e-3, "neg": 8, "epochs": 4}),
                                       encoding="utf-8")
        (d / "result.json").write_text("".join(json.dumps(dict(r, timestamp=1, hostname="h")) + "\n" for r in curve),
                                       encoding="utf-8")
        for _ in range(i + 1):
            pt.bump_attempts(str(d))
    before = _tree(storage)
    tune.main(["--status", "--system", "laya", "--storage", str(storage)])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "search laya: 2 of 3 trials started, 2 driver starts, finished no"
    assert out[1] == "trial 0: attempts 1, epoch 2 of 4, best dev micro-F1 0.5000 (epoch 2)"
    assert out[2] == "trial 1: attempts 2, epoch 1 of 4, best dev micro-F1 0.5000 (epoch 1)"
    assert len(out) == 3
    assert _tree(storage) == before
    with pytest.raises(SystemExit, match="no search"):
        tune.main(["--status", "--system", "berturk", "--storage", str(storage)])


@pytest.mark.parametrize("argv", [["--check", "x.json", "--system", "laya"], ["--render", "x.json", "--status"],
                                  ["--check", "x.json", "--status"], ["--status"], []],
                         ids=["check-system", "render-status", "check-status", "status-no-system", "no-system"])
def test_modes_are_exclusive(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """--check, --status and --render exclude each other; --check and --render take no --system; --status and a
    search need one: each is an argument error (exit 2)."""
    with pytest.raises(SystemExit) as e:
        tune.main(argv)
    assert e.value.code == 2
