"""demo.py on the Apple GPU with the real six final seeds: bound through protocol.json to the committed postfit records
and final manifests, loaded once in this process through score.py's loaders on MPS (fp32, autocast off), one
demo-authored message answered over HTTP as the frozen verdict's seed-averaged definition, and nothing under results/,
data/ or protocol.json changed.

The seed dirs (ckpt-final-s0..2, ckpt-berturk-final-s0..2) are gitignored and live only in the main checkout; the test
looks for them there (the dirname of git's common dir, so a worktree finds them too) and is skipped, naming the missing
dirs, when any is absent. Needs Apple MPS and about 6 GiB of free memory, so it is excluded from `make test`: run
`pytest -m mps tests/test_demo_mps.py` on the M4, with no other MPS job running.
"""
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import numpy as np
import pytest

import demo
import pilot.data
import pilot.protocol
from pilot import demo as pd
from pilot import metrics

pytestmark = pytest.mark.mps

LOAD_TIMEOUT = 300
TEXT = "Kartımı kaybettim, hemen dondurabilir misiniz?"      # demo-authored (the UI-SPEC's first example)
MPS_STATE = {"device": "mps", "dtype": "torch.float32", "amp_enabled": False}


def main_root() -> str:
    """The main checkout: the dirname of git's common dir (the repo root itself outside a worktree)."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=pilot.data.REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return os.path.dirname(common)


@pytest.fixture
def _threads() -> Iterator[None]:
    """The loaders set torch's process-wide thread count; restored afterwards."""
    import torch
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_real_final_seeds_on_mps(_threads: None) -> None:
    import torch
    from starlette.testclient import TestClient
    if not torch.backends.mps.is_available():
        pytest.fail("the mps marker needs Apple MPS; run it on the M4, or deselect it with -m 'not mps'")
    root = main_root()
    proto, _ = pilot.protocol.require_frozen()
    names = [s["checkpoint_name"] for k in pd.KINDS for s in pilot.protocol.final_seeds(proto, k)]
    missing = [n for n in names if not os.path.isdir(os.path.join(root, n))]
    if missing:
        pytest.skip("the final seed dirs %s are not in the main checkout %s" % (", ".join(missing), root))
    bindings = pd.protocol_bindings(pilot.data.REPO_ROOT)
    state = demo.new_state("mps")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-model")
    try:
        t = time.perf_counter()
        demo.start_loading(state, executor, lambda: bindings, root, "mps", 4).result(timeout=LOAD_TIMEOUT)
        load_s = time.perf_counter() - t
        assert state["chat"] == pd.READY, state["message"]
        app = demo.create_app(state, executor, pd.build_views(pilot.data.REPO_ROOT))
        with TestClient(app, base_url="http://127.0.0.1") as c:
            assert c.get("/api/status").json() == {"chat": "ready", "message": None, "device": "mps"}
            r = c.post("/api/classify", json={"text": TEXT})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == set(pd.KINDS)
        print("demo mps: load + warm-up %.1f s, MPS driver memory %.2f GiB"
              % (load_s, torch.mps.driver_allocated_memory() / 2 ** 30))
        for kind in pd.KINDS:
            b, got, system = bindings[kind], body[kind], state["systems"][kind]
            P = np.array(got["probabilities"], dtype=np.float64)
            assert P.shape == (48,) and ((P >= 0) & (P <= 1)).all()
            assert len(system["seeds"]) == len(b["seeds"]) == 3
            for s, ms in zip(system["seeds"], b["seeds"]):
                assert abs(s["temperature"] - ms["temperature"]) <= pd.T_MATCH
                assert abs(s["temperature"] - b["temperature"]) <= pd.T_MATCH
                assert s["check"]() == MPS_STATE
            logits = [s["forward"](TEXT) for s in system["seeds"]]          # the loaded forwards, again
            want = metrics.seed_average([metrics.probabilities(lg, s["temperature"])
                                         for lg, s in zip(logits, system["seeds"])])
            c, pred = metrics.message_confidence(want, b["theta"])
            assert np.array_equal(P, want[0]), kind
            assert np.array_equal(np.array(got["theta"]), b["theta"])
            assert got["predicted"] == [t for t in got["ranked"] if pred[0, metrics.TAGS.index(t)]]
            assert got["log_confidence"] == float(c[0])
            assert got["record"] == {"path": b["record_path"], "sha256": b["record_sha256"]}
            print("demo mps %s: predicted %s, confidence %.4f" % (kind, got["predicted"], got["confidence"]))
        for kind in pd.KINDS:                               # the dirs still hold exactly the manifest's checkpoints
            for s, _, ident in demo.bind_seeds(bindings[kind], root):
                assert ident == s["checkpoint_identity"]
    finally:
        executor.shutdown(wait=True)
    st = subprocess.run(["git", "status", "--porcelain", "--", "results", "data", "protocol.json"],
                        cwd=pilot.data.REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert st == "", st
