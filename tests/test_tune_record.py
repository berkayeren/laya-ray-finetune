"""tune.py's guards and modes, in-process with ray.init replaced by a refusal: the results/ guard (only the committed
search writes there), the search file's spec check, the unrestorable experiment dir, the MPS preflight in the segment,
--check, --status, the mode exclusivity, and the rendered summary (pilot.tune.render_markdown, --render, the
results/tune/tune.md guard).

The synthetic snapshot (tests/conftest.py) and a tiny Laya checkpoint (tests/tiny_laya.py); nothing is ever written
under the repo's results/, data/ or protocol.json.
"""
import json
import os
import re
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


def _bytes_or_none(path: str) -> bytes | None:
    """The bytes of `path`, or None when it does not exist: a refused run must leave it as it was (absent, or a
    committed record byte for byte)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


# ---- the results/ guard ----

DEPARTURES = [(["--trials", "3"], "--trials"), (["--seed", "1"], "--seed"), (["--device", "cpu"], "--device"),
              (["--points", json.dumps([{"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 4}])],
               "--points"),
              (["--kill-at-trial", "1"], "--kill-at-trial"), (["--fail-at-trial", "1"], "--fail-at-trial"),
              ("BASE", "--base"), ("MANIFEST", "--manifest")]


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
    before = _bytes_or_none(out)                            # results/tune/berturk.json is a committed record
    with pytest.raises(SystemExit, match="results/tune/laya.json"):
        tune.main(argv)
    assert _bytes_or_none(out) == before and _guarded_status() == ""


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


FINISHED_POINTS = [{"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 4},
                   {"encoder_lr": 5e-06, "head_lr": 3e-05, "neg": 4, "epochs": 4},
                   {"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 6}]


def _finished_search(snap: tuple[str, str, str], storage: Path, out: Path) -> tuple[list[str], str]:
    """(argv, record sha256) of a 3-trial search made finished without Ray: the first start writes the search file,
    each trial dir gets params.json, a one-epoch result.json (an ASHA stop at rung 1) and attempts.json, and the
    search file is marked finished with the record's sha256, as the end of main does. --out is not written."""
    argv = _smoke(snap, storage, out)
    with pytest.raises(RayReached):
        tune.main(argv)
    for i, point in enumerate(FINISHED_POINTS):
        d = storage / "laya" / pt.trial_dirname_of(i)
        d.mkdir(parents=True)
        (d / "params.json").write_text(json.dumps(point), encoding="utf-8")
        (d / "result.json").write_text(json.dumps(dict(_result(1, 1 + i, 2, 2), timestamp=1, pid=2)) + "\n",
                                       encoding="utf-8")
        pt.bump_attempts(str(d))
    a = tune.parse_args(argv)
    sp = tune.spec_path(str(storage), "laya")
    doc = tune.read_search(sp)
    record = tune.finish(a, tune.bind(a), doc)
    sha = pilot.data.sha256_bytes(pt.record_bytes(record))
    tune.write_search(sp, dict(doc, finished=True, record_sha256=sha))
    return argv, sha


def test_a_finished_search_checks_its_rebuild_before_writing_out(snap: tuple[str, str, str], tmp_path: Path,
                                                                 capsys: pytest.CaptureFixture[str],
                                                                 no_ray: None) -> None:
    """A finished search's rerun starts no Ray and writes --out only once the rebuilt record has the sha256 the search
    wrote: with a changed trial dir it is refused and --out stays absent."""
    storage, out = tmp_path / "s", tmp_path / "laya.json"
    argv, sha = _finished_search(snap, storage, out)
    result = storage / "laya" / pt.trial_dirname_of(2) / "result.json"
    good = result.read_text(encoding="utf-8")
    result.write_text(good.replace('"dev_tp": 3', '"dev_tp": 4'), encoding="utf-8")
    with pytest.raises(SystemExit, match="rebuilds a record of sha256 .* not the %s it wrote" % sha):
        tune.main(argv)
    assert not out.exists()
    result.write_text(good, encoding="utf-8")
    capsys.readouterr()
    tune.main(argv)
    assert "search finished; record written -> %s (sha256 %s)" % (out, sha) in capsys.readouterr().out
    assert pt.read_record(str(out))[1] == sha


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
                         ids=["check-system", "markdown-status", "check-status", "status-no-system", "no-system"])
def test_modes_are_exclusive(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """--check, --status and --render exclude each other; --check and --render take no --system; --status and a
    search need one: each is an argument error (exit 2)."""
    with pytest.raises(SystemExit) as e:
        tune.main(argv)
    assert e.value.code == 2


# ---- the rendered summary (results/tune/tune.md) ----

BERTURK_POINT = {"encoder_lr": 5e-06, "head_lr": 3e-05, "batch": 32, "warmup_ratio": 0.06, "weight_decay": 0.0,
                 "epochs": 10}


def _berturk_canonical() -> dict[str, Any]:
    """A synthetic canonical berturk record of one driver start: two trials (a 10-epoch and a 20-epoch schedule),
    both attempted once."""
    c = _canonical()
    c["system"] = "berturk"
    c["search"]["space"] = json.loads(json.dumps(pt.SPACES["berturk"]))
    c["search"]["scheduler"] = dict(pt.ASHA, max_t=30)
    c["fixed"] = dict(pt.FIXED["berturk"])
    c["base"] = {"name": "berturk", "model_sha256": "1" * 64, "config_sha256": "2" * 64}
    c["versions"] = {"ray": "2.59.0", "torch": "2.14.1", "transformers": "5.19.0", "accelerate": "1.15.0"}
    c["segments"] = c["segments"][:1]                      # one driver start
    c["trials"] = [{"index": 0, "config": dict(BERTURK_POINT), "attempts": 1,
                    "curve": [_result(e, 1 + e, 2, 2) for e in range(1, 11)]},
                   {"index": 1, "config": dict(BERTURK_POINT, batch=16, epochs=20), "attempts": 1,
                    "curve": [_result(1, 1, 3, 3)]}]
    return c


def _records() -> list[dict[str, Any]]:
    return [pt.build_record(_canonical()), pt.build_record(_berturk_canonical())]


def _cell(system: str, key: str, v: Any) -> str:
    return "%.1e" % v if pt.SPACES[system][key][0] == "loguniform" else str(v)


def test_render_markdown_of_two_records() -> None:
    """The title and the exploratory paragraph, one section per record in order, one table row per trial in index
    order, the best line equal to the record's best, the restarted-trial count and every NOTES sentence; no path, no
    timestamp, no uid; deterministic."""
    records = _records()
    text = pt.render_markdown(records)
    assert text == pt.render_markdown(_records()) and text.endswith("\n")
    head, intro = text.split("\n\n")[:2]
    assert head == pt.RENDER_TITLE
    for phrase in ("exploratory", "not a decision input", "does not change the frozen verdict", "protocol.json",
                   "folds 0-11", "folds 12-13", "never val, test or HOTELS", pt.OBJECTIVE):
        assert phrase in intro, phrase
    sections = [text.index("\n## %s\n" % name) for name in ("Laya", "BERTurk")]
    assert sections == sorted(sections)
    for r, start, end in zip(records, sections, sections[1:] + [text.index("\n## Notes\n")]):
        part = text[start:end]
        rows = [line for line in part.splitlines() if line.startswith("| ") and line[2].isdigit()]
        assert [int(line.split("|")[1]) for line in rows] == [t["index"] for t in r["trials"]]
        for line, t in zip(rows, r["trials"]):
            cells = [c.strip() for c in line.strip("|").split("|")]
            assert cells[1:1 + len(t["config"])] == [_cell(r["system"], k, v) for k, v in t["config"].items()]
            assert cells[-2] == "%.2f" % t["curve"][t["best_epoch"] - 1]["dev_theta"]
            assert cells[-1] == ("yes" if t["asha_stopped"] else "no")
        b = r["best"]
        row = r["trials"][b["index"]]["curve"][b["epoch"] - 1]
        best = ("Highest tuning-dev score: trial %d, epoch %d, dev micro-F1 %.4f (TP %d, FP %d, FN %d); %s"
                % (b["index"], b["epoch"], b["dev_micro_f1"], row["dev_tp"], row["dev_fp"], row["dev_fn"],
                   ", ".join("%s %s" % (k, _cell(r["system"], k, v)) for k, v in b["config"].items())))
        assert best in part.splitlines(), best
        restarted = sum(1 for t in r["trials"] if t["attempts"] > 1)
        assert "- Restarted trials (attempts above 1): %d of %d" % (restarted, len(r["trials"])) in part
    assert "- Restarted trials (attempts above 1): 1 of 2" in text   # the laya record's trial 1
    for note in pt.NOTES:
        assert "- %s" % note in text
    assert "recommend" not in text.lower()
    assert REPO not in text and "/Users/" not in text and "/tmp" not in text and "bank_" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}", text)


def test_render_discloses_a_restore() -> None:
    """A record of more than one driver start renders RESTORE_LINE (ASHA's rung table does not survive a restore) in
    its section, right after the driver starts; a one-start record does not."""
    one = _berturk_canonical()
    assert len(one["segments"]) == 1
    text = pt.render_markdown([pt.build_record(one)])
    assert "Restored" not in text and "rung table" not in text
    two = _berturk_canonical()
    two["segments"] = two["segments"] * 3
    lines = pt.render_markdown([pt.build_record(two)]).splitlines()
    i = [n for n, line in enumerate(lines) if line.startswith("- Driver starts: 3 ")]
    assert len(i) == 1 and lines[i[0] + 1] == pt.RESTORE_LINE % 2


def test_render_refuses_an_unchecked_record(tmp_path: Path) -> None:
    """--render reads each record through read_record: an edited best is refused."""
    rec = pt.build_record(_canonical())
    rec["best"] = dict(rec["best"], dev_micro_f1=0.75)
    path = tmp_path / "laya.json"
    path.write_bytes(pt.record_bytes(rec))
    with pytest.raises(SystemExit, match="does not rebuild"):
        tune.main(["--render", str(path)])


def test_render_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--render A B prints the markdown; with --out it writes exactly those bytes atomically. Under results/ the only
    --out is results/tune/tune.md (render_out_ok), refused before anything is written."""
    a, b = (pt.write_record(r, str(tmp_path / ("%s.json" % r["system"]))) for r in _records())
    tune.main(["--render", a, b])
    printed = capsys.readouterr().out
    assert printed == pt.render_markdown(_records())
    out = tmp_path / "tune.md"
    tune.main(["--render", a, b, "--out", str(out)])
    assert out.read_text(encoding="utf-8") == printed
    assert str(out) in capsys.readouterr().out
    before = _guarded_status()
    for bad in (os.path.join(REPO, "results", "x.md"), os.path.join(REPO, "results", "tune", "other.md"),
                os.path.join(REPO, "results", "tune", "laya.json")):
        with pytest.raises(SystemExit, match="results/tune/tune.md"):
            tune.main(["--render", a, b, "--out", bad])
        assert _guarded_status() == before
    assert not os.path.exists(os.path.join(REPO, "results", "x.md"))
    assert not os.path.exists(os.path.join(REPO, "results", "tune", "other.md"))
    assert tune.render_out_ok(os.path.join(REPO, "results", "tune", "tune.md"))
    assert tune.render_out_ok(str(tmp_path / "x.md"))
    assert not tune.render_out_ok(os.path.join(REPO, "results", "tune.md"))
    assert _guarded_status() == before == ""
