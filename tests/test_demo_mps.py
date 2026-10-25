"""demo.py on the Apple GPU with the real six final seeds: bound through protocol.json to the committed postfit records
and final manifests, loaded once in this process through score.py's loaders on MPS (fp32, autocast off), one
demo-authored message answered over HTTP as the frozen verdict's seed-averaged definition, and nothing under results/,
data/ or protocol.json changed.

The seed dirs (ckpt-final-s0..2, ckpt-berturk-final-s0..2) are gitignored and live only in the main checkout; the test
looks for them there (the dirname of git's common dir, so a worktree finds them too) and is skipped, naming the missing
dirs, when any is absent. Needs Apple MPS and about 6 GiB of free memory, so it is excluded from `make test`: run
`pytest -m mps tests/test_demo_mps.py` on the M4, with no other MPS job running.

test_cli_serves_end_to_end runs the server command of `make demo` (`python demo.py --device mps`, here on a free port
with --checkpoint-root the main checkout) as a subprocess through uvicorn: ready on the six seeds, both answers, every
dashboard route, 127.0.0.1 only (the machine's other address refused, a foreign Host header 400) and the classified
text in neither of its output streams. It needs the built UI (`npm --prefix ui run build`) and is skipped without it.
"""
import gc
import http.client
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import numpy as np
import pytest

import demo
import pilot.berturk
import pilot.data
import pilot.protocol
import pilot.runs
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


def seed_root() -> str:
    """The main checkout, skipping the test (naming the missing dirs) unless it holds every final seed dir of
    protocol.json."""
    root = main_root()
    proto, _ = pilot.protocol.require_frozen()
    names = [s["checkpoint_name"] for k in pd.KINDS for s in pilot.protocol.final_seeds(proto, k)]
    missing = [n for n in names if not os.path.isdir(os.path.join(root, n))]
    if missing:
        pytest.skip("the final seed dirs %s are not in the main checkout %s" % (", ".join(missing), root))
    return root


def git_status_clean() -> None:
    st = subprocess.run(["git", "status", "--porcelain", "--", "results", "data", "protocol.json"],
                        cwd=pilot.data.REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert st == "", st


def test_real_final_seeds_on_mps(_threads: None) -> None:
    import torch
    from starlette.testclient import TestClient
    if not torch.backends.mps.is_available():
        pytest.fail("the mps marker needs Apple MPS; run it on the M4, or deselect it with -m 'not mps'")
    root = seed_root()
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
            assert c.get("/api/status").json() == {"chat": "ready", "message": None, "device": "mps",
                                                   "training": "ready"}
            tr = c.get("/api/training").json()
            assert tr["status"] == "ready" and [tr["systems"][k]["status"] for k in pd.KINDS] == ["ok", "ok"], tr
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
    git_status_clean()


def test_real_training_rows() -> None:
    """The binding stage only (demo.bind_seeds, then pilot.demo.training_system; no model load, no MPS use): the six
    real final seeds' per-epoch rows, Ray Train's for Laya (3 seeds x 3 epochs) and the HF Trainer's for BERTurk
    (3 seeds x 29 epochs), each equal to the committed final manifest's epochs on SHARED_EPOCH_KEYS."""
    root = seed_root()
    bindings = pd.protocol_bindings(pilot.data.REPO_ROOT)
    t = time.perf_counter()
    for kind, fields, n_rows in (("laya", pilot.runs.EPOCH_FIELDS, 3), ("berturk", pilot.berturk.EPOCH_FIELDS, 29)):
        b = bindings[kind]
        view = pd.training_system(b, demo.bind_seeds(b, root))
        assert view["status"] == "ok" and view["fields"] == list(fields)
        assert len(view["seeds"]) == len(b["seeds"]) == 3
        total = 0.0
        for got, ms in zip(view["seeds"], b["seeds"]):
            assert (got["seed"], got["checkpoint_name"], got["train_run_id"]) == \
                (ms["seed"], ms["checkpoint_name"], ms["train_run_id"])
            assert got["device"] == "mps" and isinstance(got["n_train_messages"], int)
            assert len(got["rows"]) == n_rows and all(list(r) == list(fields) for r in got["rows"])
            assert [{k: r[k] for k in pd.SHARED_EPOCH_KEYS} for r in got["rows"]] == ms["epochs"]
            total += sum(r["train_s"] for r in got["rows"])
            print("demo training %s seed %d: %d rows, %d train messages, train_s %.1f"
                  % (kind, got["seed"], len(got["rows"]), got["n_train_messages"],
                     sum(r["train_s"] for r in got["rows"])))
        assert view["total_train_s"] == total
        print("demo training %s: total train_s %.1f over %d seeds" % (kind, total, len(view["seeds"])))
    print("demo training: binding stage %.1f s" % (time.perf_counter() - t))
    git_status_clean()


E2E_TEXT = "SENTINEL-e2e Hesabımdaki güncel bakiyeyi öğrenmek istiyorum."     # demo-authored; the sentinel is searched
E2E_SENTINEL = "SENTINEL-e2e"
E2E_ROUTES = ("/", "/api/verdict", "/api/selection", "/api/training", "/api/confidence", "/api/latency", "/api/fewshot")


def free_port() -> int:
    """A port free on 127.0.0.1 right now: bind port 0, read the port the kernel chose, close the socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    """(status, body) of one request; an HTTP error status is returned, never raised."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def non_loopback_ipv4() -> str | None:
    """The machine's outward IPv4 address (a UDP connect sends no packet; getsockname names the chosen source), or
    None when there is none or it is a 127.x address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            addr = s.getsockname()[0]
    except OSError:
        return None
    return None if addr.startswith("127.") or addr == "0.0.0.0" else addr


def test_cli_serves_end_to_end(tmp_path: pathlib.Path) -> None:
    root = seed_root()
    if not os.path.isfile(os.path.join(demo.UI_DIST, "index.html")):
        pytest.skip("ui/dist/index.html not found; build the UI first with `npm --prefix ui run build`")
    if "torch" in sys.modules:                          # free what the in-process tests above left on MPS
        import torch
        gc.collect()
        torch.mps.empty_cache()
    port = free_port()
    base = "http://127.0.0.1:%d" % port
    out_path, err_path = os.path.join(tmp_path, "stdout.txt"), os.path.join(tmp_path, "stderr.txt")
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen([sys.executable, "demo.py", "--device", "mps", "--checkpoint-root", root,
                                 "--port", str(port)], cwd=pilot.data.REPO_ROOT, stdout=out, stderr=err)
    try:
        t, status = time.perf_counter(), None
        while time.perf_counter() - t < LOAD_TIMEOUT:
            assert proc.poll() is None, "demo.py exited with %s" % proc.returncode
            try:
                code, body = get(base + "/api/status")
                status = json.loads(body) if code == 200 else None
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                status = None
            if status and status["chat"] in (pd.READY, "refused"):
                break
            time.sleep(1)
        load_s = time.perf_counter() - t
        assert status is not None and status["chat"] == pd.READY, status
        assert status["training"] == "ready" and status["device"] == "mps", status
        print("demo cli: ready after %.1f s (process start, load, warm-up)" % load_s)

        code, body = get(base + "/api/classify", data=json.dumps({"text": E2E_TEXT}).encode("utf-8"),
                         headers={"Content-Type": "application/json"})
        assert code == 200, body
        ans = json.loads(body)
        assert set(ans) == {"laya", "berturk"}
        for kind in pd.KINDS:
            assert len(ans[kind]["probabilities"]) == 48
            print("demo cli %s: predicted %s, confidence %.4f" % (kind, ans[kind]["predicted"], ans[kind]["confidence"]))

        for route in E2E_ROUTES:
            code, body = get(base + route)
            assert code == 200, (route, body[:200])
            if route == "/api/training":
                tr = json.loads(body)
                assert [tr["systems"][k]["status"] for k in pd.KINDS] == ["ok", "ok"], tr
            if route == "/api/confidence":
                svg = json.loads(body)["val"][0]["svg"]
        code, body = get(base + "/api/svg/" + svg)
        assert code == 200 and body.lstrip().startswith(b"<"), (svg, code)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            conn.request("GET", "/api/status", headers={"Host": "evil.example"})
            assert conn.getresponse().status == 400
        finally:
            conn.close()

        addr = non_loopback_ipv4()
        if addr is None:
            print("demo cli: no non-loopback IPv4 address on this machine; the refused-connection check is skipped")
        else:
            with pytest.raises((ConnectionRefusedError, TimeoutError)):
                socket.create_connection((addr, port), timeout=5).close()
            print("demo cli: a connection to %s:%d is refused (bound to 127.0.0.1 only)" % (addr, port))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    for path in (out_path, err_path):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        assert E2E_SENTINEL not in text, "%s holds the classified text" % os.path.basename(path)
    git_status_clean()
