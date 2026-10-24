"""score.py with Ray Data actors on the tiny offline checkpoint: the dump equals the in-process (--workers 0) dump.

Ray returns rows in any order; score.assemble puts them back in split-file order by uid, so both dumps hold the same
uids in the same order and logits within 1e-5 (thread counts differ between the two paths). Ray Data's actor fault
tolerance stays on: an actor killed once mid-split is restarted and the dump completes, while a
refusal (SystemExit) in an actor's constructor or batch ends the run promptly with its own one line and no dump.

Runs in `make test` (ray marker, CPU only, no weights or data).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest

import score
from conftest import SyntheticRaw, prepared
from pilot.data import parse_jsonl
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.timeout(600)
def test_ray_and_inprocess_dumps_agree(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    ckpt = tiny_checkpoint(tmp_path / "ckpt", temperature=1.3)
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    common = [test_jsonl, "--model", "laya", "--checkpoint", ckpt, "--manifest", manifest]
    run_a, _ = score.main(common + ["--workers", "0", "--out-root", str(tmp_path / "a")])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "score.py"] + common + ["--workers", "2", "--cpus-per-worker", "1",
                                                                 "--out-root", str(tmp_path / "b")],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    run_b = os.path.join(str(tmp_path / "b"), os.path.basename(run_a))
    a = np.load(os.path.join(run_a, "test.npz"), allow_pickle=False)
    b = np.load(os.path.join(run_b, "test.npz"), allow_pickle=False)
    assert list(a["uids"]) == list(b["uids"])
    np.testing.assert_allclose(a["logits"], b["logits"], rtol=0, atol=1e-5)
    with open(os.path.join(run_a, "test.json"), encoding="utf-8") as f:
        side_a = json.load(f)
    with open(os.path.join(run_b, "test.json"), encoding="utf-8") as f:
        side_b = json.load(f)
    assert side_a["scorer"]["workers"] == 0 and side_b["scorer"]["workers"] == 2
    assert side_b["scorer"]["torch_threads"] == 1 and side_b["scorer"]["device"] == "cpu"
    assert side_a["temperature"]["shipped"] == side_b["temperature"]["shipped"] == 1.3


# The driver-side preflight (pilot.weights.check_tokenizer_config_rewritable) is switched off so the refusal is raised
# where it would hang without actor_error: in the Ray Data actor's constructor (score.load_agent's post-load backstop).
BYPASS = """import sys
import pilot.weights
pilot.weights.check_tokenizer_config_rewritable = lambda d, what: None
import score
score.main(sys.argv[1:])
"""


@pytest.mark.timeout(240)
@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_a_refusal_in_a_scoring_actor_ends_the_run(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    """Ray Data restarts an actor whose worker exits forever (its default max_restarts -1, kept for transient deaths),
    so a SystemExit in score.Predictor's constructor (here the post-load tokenizer-config backstop on a read-only
    tokenizer/) would hang `score.py --workers 1` (the default). The Predictor raises it in Ray as a plain exception,
    so the run ends promptly with the refusal's own one line and writes no dump."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    ckpt = tiny_checkpoint(tmp_path / "ckpt")                 # a fresh save: laya must rewrite its tokenizer config
    tok_dir = os.path.join(ckpt, "tokenizer")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    os.chmod(tok_dir, 0o555)
    try:
        r = subprocess.run([sys.executable, "-c", BYPASS, os.path.join(out_dir, "splits", "test.jsonl"), "--model",
                            "laya", "--checkpoint", ckpt, "--workers", "1", "--cpus-per-worker", "1",
                            "--manifest", manifest, "--out-root", str(tmp_path / "dumps")],
                           cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)
    finally:
        os.chmod(tok_dir, 0o755)
    assert r.returncode == 1, r.stderr[-4000:]
    last = r.stderr.strip().splitlines()[-1]
    assert last.startswith("a scoring actor refused: checkpoint: laya could not rewrite "
                           "tokenizer/tokenizer_config.json on load"), r.stderr[-4000:]
    assert last.endswith("and rerun; nothing was dumped"), last
    assert not (tmp_path / "dumps").exists()


# A subclass shipped by value (defined in __main__) leaves what laya's GPU-OOM retry leaves behind on the agent, so
# score.check_agent refuses after the actor's first batch, the other place a refusal is raised inside Ray.
FELL_BACK = """import sys
import score
class FellBack(score.Predictor):
    def __call__(self, batch):
        self.agent.cpu_fallback_count = 1
        return super().__call__(batch)
score.Predictor = FellBack
score.main(sys.argv[1:])
"""


@pytest.mark.timeout(240)
def test_a_refusal_after_a_scoring_actors_batch_ends_the_run(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    """A SystemExit in score.Predictor.__call__ inside Ray (check_agent) ends the run promptly with
    its own one line and no dump, as one in the constructor does."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    ckpt = tiny_checkpoint(tmp_path / "ckpt")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", FELL_BACK, os.path.join(out_dir, "splits", "test.jsonl"), "--model",
                        "laya", "--checkpoint", ckpt, "--workers", "1", "--cpus-per-worker", "1",
                        "--manifest", manifest, "--out-root", str(tmp_path / "dumps")],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 1, r.stderr[-4000:]
    last = r.stderr.strip().splitlines()[-1]
    assert last == ("a scoring actor refused: laya fell back from cpu to CPU 1 time(s) mid-run (None); a dump comes "
                    "from one device: rescore with more free memory or --device cpu; nothing was dumped"), last
    assert not (tmp_path / "dumps").exists()


# score.Predictor is replaced by a subclass defined in __main__ (so Ray ships it by value) whose actor SIGKILLs its own
# worker on its first batch, once: a transient death (the OOM killer, a crash), not a refusal.
KILL_ONCE = """import os, signal, sys
import score
FLAG = sys.argv[1]
class KillOnce(score.Predictor):
    def __call__(self, batch):
        if not os.path.exists(FLAG):
            open(FLAG, "w").close()
            os.kill(os.getpid(), signal.SIGKILL)
        return super().__call__(batch)
score.Predictor = KillOnce
score.main(sys.argv[2:])
"""


@pytest.mark.timeout(300)
def test_a_scoring_actor_killed_once_is_restarted_and_the_run_completes(tmp_path: Path,
                                                                        synthetic_raw: SyntheticRaw) -> None:
    """score_laya keeps Ray Data's actor fault tolerance, so an actor killed mid-split is restarted, its batch
    retried, and the dump is complete (with max_restarts 0 the run would die on the first kill)."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    ckpt = tiny_checkpoint(tmp_path / "ckpt")
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    flag = tmp_path / "killed"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", KILL_ONCE, str(flag), test_jsonl, "--model", "laya", "--checkpoint",
                        ckpt, "--workers", "1", "--cpus-per-worker", "1", "--manifest", manifest,
                        "--out-root", str(tmp_path / "dumps")],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-4000:]
    assert flag.exists()                                        # the actor was killed once
    (run_dir,) = (tmp_path / "dumps").iterdir()
    with open(test_jsonl, "rb") as f:
        want = [c["uid"] for c in parse_jsonl(f.read())]
    assert list(np.load(run_dir / "test.npz", allow_pickle=False)["uids"]) == want


def test_a_refusal_is_raised_in_ray_as_a_plain_exception_and_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """In Ray the Predictor turns a SystemExit (constructor or batch) into actor_error, a plain
    RuntimeError marked REFUSAL; in-process it stays a SystemExit. refusal finds the message in Ray's error text."""
    def refuse(*args: Any) -> NoReturn:
        raise SystemExit("checkpoint: refused; fix it")
    monkeypatch.setattr(score, "load_agent", refuse)
    with pytest.raises(SystemExit, match="^checkpoint: refused; fix it$"):
        score.Predictor("d", "cpu", {}, 1)
    with pytest.raises(RuntimeError) as e:
        score.Predictor("d", "cpu", {}, 1, in_ray=True)
    assert str(e.value) == score.REFUSAL + "checkpoint: refused; fix it" and e.value.__cause__ is None
    p = score.Predictor.__new__(score.Predictor)
    p.agent, p.device = object(), "cpu"
    monkeypatch.setattr(score.la, "noul_logits", lambda agent, texts, questions: np.zeros((1, 48), np.float32))
    monkeypatch.setattr(score, "check_agent", refuse)
    batch = {"uid": np.array(["u"]), "text": np.array(["t"])}
    for p.in_ray, want in ((False, SystemExit), (True, RuntimeError)):
        with pytest.raises(want, match="checkpoint: refused; fix it"):
            p(batch)
    text = ("The actor died because of an error raised in its creation task, ray::MapWorker.__init__()\n"
            "Traceback (most recent call last):\n  ...\nRuntimeError: %scheckpoint: refused; fix it\n" % score.REFUSAL)
    assert score.refusal(text) == "checkpoint: refused; fix it"
    assert score.refusal("RuntimeError: something else") is None
    outer = RuntimeError("wrapped")
    outer.__cause__ = RuntimeError(score.REFUSAL + "x; fix")
    assert score.refusal(score.error_text(outer)) == "x; fix"


def test_actor_exit_reads_rays_worker_exit_detail() -> None:
    text = ("The actor died unexpectedly before finishing this task.\n\tclass_name: MapWorker(MapBatches(Predictor))\n"
            "The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit "
            "detail: Worker unexpectedly exits with a connection error code 2. End of file.")
    assert score.actor_exit(text) == "Worker unexpectedly exits with a connection error code 2. End of file"
    assert score.actor_exit("Some other failure\n  last line \n") == "last line"
