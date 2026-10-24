"""ray_train.py run identity and resume on the tiny checkpoint, offline, on --device cpu.

The run_id comes from the command's canonical config and seed and names the Ray experiment dir, so the same command
always lands in storage/<run_id>/ and a different seed in another dir; --print-run-id prints it without starting Ray.
A killed run resumes byte for byte, a finished run reruns without training, and a read-only base tokenizer/ or low
disk is refused before Ray starts.
Runs in `make test` under the ray marker (CPU only, no weights or data).
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pilot.data
from conftest import make_raw
from pilot import runs
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600
Prepared = tuple[Path, list[Any]]                           # the prepared fixture: (work dir, ARGS)
FullRun = tuple[str, Path]                                  # the full_run fixture: (run id, out dir)


def _run(*args: Any, ok: bool = True) -> subprocess.CompletedProcess:
    """Run a repo script with the test interpreter (no PYTHONPATH, so it imports like a user's command); with ok,
    assert it succeeds."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    if ok:
        assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    """(work dir, ARGS): the synthetic splits, the tiny base and the shared flags of every run in this module."""
    d = tmp_path_factory.mktemp("resume")
    root, raw_sha256 = make_raw(str(d / "raw"))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    base = tiny_checkpoint(d / "base")
    args = ["--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
            "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--device", "cpu", "--workers", "1",
            "--cpus-per-worker", "1", "--epochs", "2", "--max-messages", "12", "--neg", "2"]
    return d, args


@pytest.fixture(scope="module")
def full_run(prepared: Prepared) -> FullRun:
    """(rid, out dir) of one uninterrupted run of ARGS into storage a."""
    d, args = prepared
    rid = _run("ray_train.py", *args, "--out", d / "x", "--storage", d / "a", "--print-run-id").stdout.strip()
    _run("ray_train.py", *args, "--storage", d / "a", "--out", d / "full")
    return rid, d / "full"


@pytest.mark.timeout(600)
def test_print_run_id_names_the_storage_dir(prepared: Prepared, full_run: FullRun) -> None:
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


def _sha256(path: str | Path) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.timeout(600)
def test_killed_run_resumes_bitwise(prepared: Prepared, full_run: FullRun) -> None:
    """Killed after epoch 1 and rerun with the same command: the result is byte-identical to an uninterrupted run."""
    d, args = prepared
    rid, full = full_run
    r = _run("ray_train.py", *args, "--storage", d / "b", "--out", d / "res", "--fail-after-epoch", "1", ok=False)
    assert r.returncode != 0 and "rerun the same command" in r.stderr, r.stderr[-4000:]
    run_dir = d / "b" / rid
    assert (run_dir / "epoch-001" / "train_state.pt").is_file() and (run_dir / "epoch-001" / "rng-rank0.pt").is_file()
    assert (run_dir / "exports" / "epoch-001" / "model.safetensors").is_file()
    assert not (d / "res").exists()
    _run("ray_train.py", *args, "--storage", d / "b", "--out", d / "res")       # the same command resumes
    assert _sha256(d / "res" / "model.safetensors") == _sha256(full / "model.safetensors")
    rec = runs.read_record(str(d / "res"))
    assert [row["epoch"] for row in rec["epochs"]] == [1, 2]
    assert [row["segment"] for row in rec["epochs"]] == [0, 1]
    assert [row["resumed"] for row in rec["epochs"]] == [False, True]
    assert [s["segment"] for s in rec["segments"]] == [0, 1] and [s["start_epoch"] for s in rec["segments"]] == [0, 1]
    for s in rec["segments"]:
        assert tuple(s) == runs.SEGMENT_FIELDS, s
    assert rec["segments"][0]["resumed"] is False and rec["segments"][0]["resume_load_s"] is None
    assert rec["segments"][1]["resumed"] is True and rec["segments"][1]["resume_load_s"] > 0
    assert rec["segments"][1]["resume_peak_total_bytes"] > 0
    # same code in both segments; without git (an exported tree) the flag says it cannot tell
    assert rec["code_changed_on_resume"] is (rec["segments"][0]["git_sha"] is None)
    assert rec["train_state_bytes"] > 0 and rec["export_bytes"] > 0
    assert rec["exports"] == ["exports/epoch-001", "exports/epoch-002"]
    assert all(row["save_s"] > 0 for row in rec["epochs"]) and rec["save_s_includes_persist"] is True
    assert rec["run_id"] == rid
    assert sorted(p.name for p in (run_dir / "exports").iterdir()) == ["epoch-001", "epoch-002"]
    ex = runs.read_record(str(run_dir / "exports" / "epoch-002"))                # a pre-save snapshot
    assert ex["exports"] == ["exports/epoch-001", "exports/epoch-002"] and ex["epochs"][-1]["save_s"] is None
    assert ex["epochs"][0]["save_s"] > 0 and ex["train_state_bytes"] is None
    assert not (run_dir / "epoch-001").exists()                                  # num_to_keep=1 rotated it out
    full_rec = runs.read_record(str(full))
    assert [row["loss"] for row in rec["epochs"]] == [row["loss"] for row in full_rec["epochs"]]


@pytest.mark.timeout(600)
def test_a_failure_after_the_last_epoch_resumes_into_the_final_save_only(prepared: Prepared, full_run: FullRun) -> None:
    """The rerun after a failure past the last epoch's checkpoint (as after a
    final-save OOM) trains nothing and never loads the optimizer state; it writes the final checkpoint of the same
    weights the uninterrupted run published (uncalibrated)."""
    d, args = prepared
    rid, full = full_run
    r = _run("ray_train.py", *args, "--storage", d / "c", "--out", d / "fin", "--fail-after-epoch", "2", ok=False)
    assert r.returncode != 0 and (d / "c" / rid / "epoch-002" / "train_state.pt").is_file(), r.stderr[-4000:]
    r = _run("ray_train.py", *args, "--storage", d / "c", "--out", d / "fin")
    out = r.stdout + r.stderr
    assert "resuming %s after epoch 2" % rid in out and "the optimizer state is not loaded" in out, out[-4000:]
    assert "epoch 1/2" not in out and "epoch 2/2" not in out
    assert _sha256(d / "fin" / "model.safetensors") == _sha256(full / "model.safetensors")
    rec = runs.read_record(str(d / "fin"))
    assert [row["segment"] for row in rec["epochs"]] == [0, 0]
    assert [s["start_epoch"] for s in rec["segments"]] == [0, 2] and rec["calib_s"] is None


@pytest.mark.timeout(600)
def test_finished_run_reruns_without_training(prepared: Prepared, full_run: FullRun) -> None:
    d, args = prepared
    rid, full = full_run
    r = _run("ray_train.py", *args, "--storage", d / "a", "--out", d / "full2")
    assert "epoch 1/2" not in r.stdout + r.stderr
    assert _sha256(d / "full2" / "model.safetensors") == _sha256(full / "model.safetensors")
    assert os.path.isfile(d / "full2" / "final.json")


@pytest.mark.timeout(600)
def test_snapshot_with_deleted_checkpoints_names_the_dir(prepared: Prepared, full_run: FullRun, tmp_path: Path) -> None:
    import shutil
    d, args = prepared
    rid, full = full_run
    shutil.copytree(d / "a" / rid, tmp_path / "s" / rid)
    shutil.rmtree(tmp_path / "s" / rid / "final")                               # the snapshot still points at it
    r = _run("ray_train.py", *args, "--storage", tmp_path / "s", "--out", tmp_path / "o", ok=False)
    assert r.returncode != 0
    assert "holds a Ray snapshot whose checkpoints are gone; delete that dir" in r.stderr, r.stderr[-4000:]
    assert os.path.join(str(tmp_path / "s"), rid) in r.stderr


@pytest.mark.timeout(120)
@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_a_read_only_base_tokenizer_dir_is_refused_before_ray(prepared: Prepared, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """A base whose tokenizer_config.json laya must rewrite (a fresh save_checkpoint output holds
    TokenizersBackend) in a read-only tokenizer/ is refused on the driver, before --print-run-id and ray.init, with
    its own fix, not after a worker's full base load as "training failed ...; rerun the same command to resume"."""
    import ray_train
    d, args = prepared

    def no_ray(*a: Any, **kw: Any) -> None:
        raise AssertionError("ray.init reached; the read-only tokenizer/ must be refused first")
    base = tiny_checkpoint(tmp_path / "base")                 # its own base: a loaded one is already rewritten
    args = list(args)
    args[args.index("--base") + 1] = base
    tok_dir = os.path.join(base, "tokenizer")
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "rb") as f:
        before = f.read()
    monkeypatch.setattr(ray_train.ray, "init", no_ray)
    os.chmod(tok_dir, 0o555)
    try:
        for extra in ([], ["--print-run-id"]):
            with pytest.raises(SystemExit) as e:
                ray_train.main(args + ["--out", str(tmp_path / "o"), "--storage", str(tmp_path / "s")] + extra)
            assert str(e.value.code).startswith("base checkpoint: tokenizer/tokenizer_config.json needs laya's "
                                                "load-time rewrite, but tokenizer/ is not writable"), e.value.code
            assert "rerun the same command" not in str(e.value.code)
    finally:
        os.chmod(tok_dir, 0o755)
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "rb") as f:
        assert f.read() == before                             # nothing loaded it
    assert not (tmp_path / "s").exists() and not (tmp_path / "o").exists()


@pytest.mark.timeout(300)
@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_a_worker_refusal_ends_the_run_without_resume_advice(prepared: Prepared, tmp_path: Path) -> None:
    """With the driver preflight switched off, the worker's post-load backstop refuses the read-only
    base after ray.init; fit_or_exit shows that refusal as is, never "training failed ...; rerun the same command"."""
    d, args = prepared
    base = tiny_checkpoint(tmp_path / "base")
    args = [str(x) for x in args]
    args[args.index("--base") + 1] = base
    code = ("import sys\nimport ray_train\n"
            "ray_train.check_tokenizer_config_rewritable = lambda d, what: None\nray_train.main(sys.argv[1:])\n")
    tok_dir = os.path.join(base, "tokenizer")
    os.chmod(tok_dir, 0o555)
    try:
        r = _run("-c", code, *args, "--out", tmp_path / "o", "--storage", tmp_path / "s", ok=False)
    finally:
        os.chmod(tok_dir, 0o755)
    # the driver's own line; Ray's asynchronous worker logs may print after it
    exits = [ln for ln in r.stderr.splitlines() if ln.startswith("base checkpoint: laya could not rewrite "
                                                                  "tokenizer/tokenizer_config.json on load")]
    assert r.returncode == 1 and len(exits) == 1, r.stderr[-4000:]
    assert "training failed" not in r.stderr and "rerun the same command" not in r.stderr, r.stderr[-4000:]
    assert not (tmp_path / "o").exists()


@pytest.mark.timeout(600)
def test_low_disk_is_refused_before_ray(prepared: Prepared, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import collections
    import ray_train
    d, args = prepared

    def no_ray(*a: Any, **kw: Any) -> None:
        raise AssertionError("ray.init reached; the disk refusal must come first")
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(ray_train.ray, "init", no_ray)
    monkeypatch.setattr(ray_train.shutil, "disk_usage", lambda path: usage(2**40, 2**40 - 2**20, 2**20))
    with pytest.raises(SystemExit, match="GiB free under --storage"):
        ray_train.main(args + ["--out", str(tmp_path / "o"), "--storage", str(tmp_path / "s")])
    assert not (tmp_path / "s").exists() and not (tmp_path / "o").exists()
