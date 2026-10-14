"""score.py with Ray Data actors on the tiny offline checkpoint: the dump equals the in-process (--workers 0) dump.

Ray returns rows in any order; score.assemble puts them back in split-file order by uid, so both dumps hold the same
uids in the same order and logits within 1e-5 (thread counts differ between the two paths).

Runs in `make test` (ray marker, CPU only, no weights or data).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import score
from conftest import prepared
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.timeout(600)
def test_ray_and_inprocess_dumps_agree(tmp_path, synthetic_raw):
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
# where it used to hang: in the Ray Data actor's constructor (score.load_agent's post-load backstop).
BYPASS = """import sys
import pilot.weights
pilot.weights.check_tokenizer_config_rewritable = lambda d, what: None
import score
score.main(sys.argv[1:])
"""


@pytest.mark.timeout(240)
@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_a_refusal_in_a_scoring_actor_ends_the_run(tmp_path, synthetic_raw):
    """04-REVIEW CR-01: Ray Data restarts a dead actor forever by default, so a SystemExit in score.Predictor's
    constructor (here the IN-01 backstop on a read-only tokenizer/) hung `score.py --workers 1` (the default). With
    max_restarts 0 the run ends promptly with the actor's own message and writes no dump."""
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
    assert last.startswith("a scoring actor died: checkpoint: laya could not rewrite tokenizer/tokenizer_config.json "
                           "on load"), r.stderr[-4000:]
    assert last.endswith("and rerun; nothing was dumped (rescore with --workers 0 to see the error in-process)"), last
    assert not (tmp_path / "dumps").exists()


def test_scoring_actors_are_never_restarted():
    """04-REVIEW CR-01: score_laya passes max_restarts=ACTOR_MAX_RESTARTS (0) to map_batches; Ray Data's default (-1)
    would restart an actor that refuses forever."""
    import inspect
    assert score.ACTOR_MAX_RESTARTS == 0
    assert "max_restarts=ACTOR_MAX_RESTARTS" in inspect.getsource(score.score_laya)
    text = ("The actor died unexpectedly before finishing this task.\n\tclass_name: MapWorker(MapBatches(Predictor))\n"
            "The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit "
            "detail: Worker exits unexpectedly. Worker exits with an exit code checkpoint: refused; fix it.")
    assert score.actor_exit(text) == "checkpoint: refused; fix it"
    assert score.actor_exit("Some other failure\n  last line \n") == "last line"
