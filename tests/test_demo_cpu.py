"""demo.py's chat path on CPU, offline: tiny final seeds of both systems, bound to synthetic postfit/1 records and
postfit-final/1 manifests, loaded through score.py's loaders on the demo's worker thread, answer one message over HTTP
exactly as the frozen verdict's seed-averaged definition computes it.

Two tiny Laya seeds come from tests/tiny_laya.py (T 1.3); two tiny BERTurk seeds are the epoch-001 and epoch-002 exports
of one 2-epoch berturk_train.py run on CPU over the synthetic snapshot of tests/conftest.py (T 1.0). The records' chosen
θ is 0.5 with one per-tag θ (0.3 for the first INTENTS tag). The startup refusals and the request validation are tested
here too. Offline, no Ray, no weights: runs in `make test`.
"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

import demo
import pilot.data
import score
from conftest import make_raw
from intents import INTENTS, QUESTIONS
from pilot import berturk, dumps, metrics
from pilot import demo as pd
from pilot import laya_adapter as la
from pilot.weights import identity
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600
LAYA_T = 1.3
BERT_T = 1.0
TEXT = "Kartım kayboldu ."                                  # words of the tiny vocabulary
FIRST_TAG = next(iter(INTENTS))
HEAVY = ("torch", "transformers", "laya", "ray")
WEB = ("fastapi", "starlette", "uvicorn")


@pytest.fixture(scope="module", autouse=True)
def _threads() -> Iterator[None]:
    """The loaders set torch's process-wide thread count; restored after the module."""
    import torch
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def write_record(path: Path, kind: str, temperature: float, per_tag: dict[str, float] | None = None) -> str:
    """A synthetic postfit/1 record of `kind`: chosen T and theta.shipped (global 0.5, FIRST_TAG at 0.3)."""
    shipped = {"global": 0.5, "per_tag": {FIRST_TAG: 0.3} if per_tag is None else per_tag}
    rec = {"schema": "postfit/1", "system_kind": kind,
           "chosen": {"temperature": temperature, "theta": {"shipped": shipped}}, "final": {}, "protocol": {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return str(path)


def write_manifest(path: Path, kind: str, record: str, seeds: list[tuple[int, str, dict[str, Any], str, float]],
                   record_sha256: str | None = None) -> str:
    """A synthetic postfit-final/1 manifest of `seeds` [(seed, checkpoint_name, identity, dump_run_id, T)], bound to
    `record`'s sha256 (or `record_sha256`)."""
    m = {"schema": "postfit-final/1",
         "postfit": {"path": "results/postfit/synthetic.json",
                     "sha256": record_sha256 or pilot.data.sha256_file(record)},
         "final_split": "final_train", "chosen": {},
         "seeds": [{"seed": seed, "train_run_id": "%s-final_train-s%d-synthetic" % (kind, seed),
                    "checkpoint_name": name, "checkpoint_identity": ident, "dump_run_id": rid, "temperature": t}
                   for seed, name, ident, rid, t in seeds]}
    if kind == "berturk":
        m["system_kind"] = "berturk"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    return str(path)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """{root, seeds: {kind: [(seed, name, identity, dump_run_id, T)]}, laya: {record, manifest}, berturk: {...}}:
    ckpt-s0, ckpt-s1 (tiny Laya) and bert-s0, bert-s1 (the two exports of a 2-epoch tiny BERTurk run) under one
    checkpoint root, with their synthetic records and manifests. Built once for the module."""
    from tiny_berturk import tiny_base
    tmp = tmp_path_factory.mktemp("demo_tiny")
    root = tmp / "ckpts"
    seeds: dict[str, list[tuple[int, str, dict[str, Any], str, float]]] = {"laya": [], "berturk": []}
    for seed in (0, 1):
        name = "ckpt-s%d" % seed
        ident = identity(tiny_checkpoint(root / name, temperature=LAYA_T, seed=seed))
        seeds["laya"].append((seed, name, ident, dumps.run_id("laya", ident), LAYA_T))
    raw, raw_sha256 = make_raw(str(tmp / "raw"))
    out_dir = str(tmp / "data")
    pilot.data.prepare(raw, out_dir, raw_sha256)
    base = tiny_base(tmp / "base")
    storage = tmp / "runs"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [sys.executable, "berturk_train.py", "--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
            "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--device", "cpu", "--epochs", "2",
            "--batch", "4", "--max-len", "32", "--seed", "0", "--storage", str(storage)]
    rid = subprocess.run(argv + ["--print-run-id"], cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=TIMEOUT, check=True).stdout.strip()
    r = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, r.stderr[-4000:]
    for seed, export in ((0, "epoch-001"), (1, "epoch-002")):
        name = "bert-s%d" % seed
        shutil.copytree(storage / rid / "exports" / export, root / name)
        ident = berturk.identity(str(root / name))
        seeds["berturk"].append((seed, name, ident, berturk.dump_run_id(ident), BERT_T))
    out: dict[str, Any] = {"root": str(root), "seeds": seeds, "tmp": str(tmp)}
    for kind, T in (("laya", LAYA_T), ("berturk", BERT_T)):
        rec = write_record(tmp / ("%s.json" % kind), kind, T)
        out[kind] = {"record": rec, "manifest": write_manifest(tmp / ("%s-final.json" % kind), kind, rec, seeds[kind])}
    return out


def bindings_of(paths: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{kind: pd.bind_system} of the records and manifests in `paths` (no protocol pins: synthetic files)."""
    return {k: pd.bind_system(k, paths[k]["record"], paths[k]["manifest"]) for k in pd.KINDS}


def load(paths: dict[str, Any], root: str, device: str = "cpu") -> tuple[dict[str, Any], ThreadPoolExecutor, Any]:
    """(state, executor, app) after demo.start_loading of `paths` on the worker thread has finished."""
    state = demo.new_state(device)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-model")
    demo.start_loading(state, executor, lambda: bindings_of(paths), root, device, 1).result(timeout=TIMEOUT)
    return state, executor, demo.create_app(state, executor, pd.build_views(REPO))


def client(app: Any) -> Any:
    """A TestClient on the Host 127.0.0.1 (TrustedHostMiddleware refuses the default "testserver")."""
    from starlette.testclient import TestClient
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture(scope="module")
def loaded(tiny: dict[str, Any]) -> Iterator[tuple[dict[str, Any], ThreadPoolExecutor, Any]]:
    """(state, executor, app) of the tiny seeds, loaded once for the module."""
    state, executor, app = load(tiny, tiny["root"])
    assert state["chat"] == pd.READY, state["message"]
    yield state, executor, app
    executor.shutdown(wait=True)


def independent_answer(tiny: dict[str, Any], kind: str, text: str) -> dict[str, Any]:
    """The verdict's definition of one system's answer, recomputed without demo.py: each seed's logits through
    score.py's loaders, metrics.probabilities at the served T, seed_average, theta_vector, message_confidence."""
    import torch
    with open(tiny[kind]["record"], encoding="utf-8") as f:
        shipped = json.load(f)["chosen"]["theta"]["shipped"]
    Ps = []
    for seed, name, ident, _, _ in tiny["seeds"][kind]:
        ckpt = os.path.join(tiny["root"], name)
        if kind == "laya":
            model_dir, expected = score.resolve_checkpoint(ckpt, ident)
            p = score.Predictor(model_dir, "cpu", expected, 1)
            with torch.no_grad():
                logits, T = la.noul_logits(p.agent, [text], QUESTIONS), p.temperature
        else:
            rec = score.check_berturk(ckpt, ident)
            tok, model, _ = score.load_bert(ckpt, ident, "cpu", 1)
            enc = tok([text], truncation=True, max_length=rec["canonical"]["max_len"], return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits.float().cpu().numpy()
            T = float(rec["temperature"])
        Ps.append(metrics.probabilities(logits, T))
    P = metrics.seed_average(Ps)
    th = metrics.theta_vector(shipped)
    c, pred = metrics.message_confidence(P, th)
    return {"P": P[0], "theta": th, "pred": pred[0], "c": float(c[0])}


def test_chat_answer_equals_the_verdict_definition(tiny: dict[str, Any],
                                                   loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    state, _, app = loaded
    with client(app) as c:
        assert c.get("/api/status").json() == {"chat": "ready", "message": None, "device": "cpu"}
        r = c.post("/api/classify", json={"text": TEXT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"laya", "berturk"}
    for kind in pd.KINDS:
        got, want = body[kind], independent_answer(tiny, kind, TEXT)
        assert np.array_equal(np.array(got["probabilities"], dtype=np.float64), want["P"]), kind
        assert np.array_equal(np.array(got["theta"], dtype=np.float64), want["theta"]), kind
        ranked = [metrics.TAGS[i] for i in np.argsort(-want["P"], kind="stable")]
        assert got["ranked"] == ranked and sorted(got["ranked"]) == sorted(INTENTS)
        assert got["predicted"] == [t for t in ranked if want["pred"][metrics.TAGS.index(t)]]
        assert got["log_confidence"] == want["c"] and got["confidence"] == float(np.exp(want["c"]))
        assert got["kind"] == kind and got["seeds"] == [0, 1]
        assert got["own_theta"] == [t == FIRST_TAG for t in metrics.TAGS] and got["n_own_theta"] == 1
        assert got["theta_global"] == 0.5 and got["theta"][0] == 0.3
        assert got["temperature"] == (LAYA_T if kind == "laya" else BERT_T)
        assert got["record"] == {"path": pilot.data.repo_relpath(tiny[kind]["record"]),
                                 "sha256": pilot.data.sha256_file(tiny[kind]["record"])}
    assert [s["check"]() for k in pd.KINDS for s in state["systems"][k]["seeds"]] == \
        [{"device": "cpu", "dtype": "torch.float32", "amp_enabled": False}] * 4


def test_no_ray_at_serving_time(tiny: dict[str, Any]) -> None:
    """A fresh process loads both tiny systems through demo.load_job, classifies one message, and has imported no
    Ray module."""
    paths = {k: tiny[k] for k in pd.KINDS}
    code = ("import json, sys, demo\n"
            "from pilot import demo as pd\n"
            "paths, root = json.loads(sys.argv[1]), sys.argv[2]\n"
            "bind = lambda: {k: pd.bind_system(k, paths[k]['record'], paths[k]['manifest']) for k in pd.KINDS}\n"
            "st = demo.new_state('cpu')\n"
            "demo.load_job(st, bind, root, 'cpu', 1)\n"
            "assert st['chat'] == 'ready', st['message']\n"
            "out = demo.classify(st['systems'], %r)\n"
            "assert set(out) == {'laya', 'berturk'}\n"
            "bad = sorted(m for m in sys.modules if m == 'ray' or m.startswith('ray.'))\n"
            "assert not bad, bad\n" % TEXT)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code, json.dumps(paths), tiny["root"]], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, r.stderr[-4000:]


def test_import_budget() -> None:
    """pilot.demo imports no model or web module; demo imports no model module at import time (fastapi, starlette
    and uvicorn only inside create_app and main, torch only inside the loaders)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for module, banned in (("pilot.demo", HEAVY + WEB), ("demo", HEAVY)):
        code = ("import sys, %s; bad = [m for m in %r if m in sys.modules]; assert not bad, bad" % (module, banned))
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (module, r.stderr)


def test_t_match_is_report_t_match() -> None:
    import latency
    import report
    assert pd.T_MATCH == report.T_MATCH == latency.T_TOLERANCE == 1e-12


# ---- startup refusals: one line, the chat refused, every other route serving ----

SENTINEL = "SENTINEL-7f3a"


def copy_root(tiny: dict[str, Any], tmp_path: Path) -> str:
    """A private copy of the module's checkpoint root (a test may tamper with it)."""
    root = tmp_path / "ckpts"
    shutil.copytree(tiny["root"], root)
    return str(root)


def assert_refused(paths: dict[str, Any], root: str, start: str, device: str = "cpu") -> str:
    """Load `paths` from `root`; assert the chat is refused with one line starting with `start`, /api/classify
    answers 503 with that line and /api/intents still answers 200. Returns the line."""
    state, executor, app = load(paths, root, device)
    try:
        msg = state["message"]
        assert state["chat"] == pd.REFUSED and state["systems"] is None, state
        assert msg.startswith(start), msg
        assert "\n" not in msg
        with client(app) as c:
            assert c.get("/api/status").json() == {"chat": "refused", "message": msg, "device": device}
            r = c.post("/api/classify", json={"text": TEXT})
            assert (r.status_code, r.json()) == (503, {"error": msg})
            r = c.get("/api/intents")
            assert r.status_code == 200 and len(r.json()["tags"]) == 48
    finally:
        executor.shutdown(wait=True)
    return msg


def no_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """score.Predictor and score.load_bert raise AssertionError: a refusal must come before any load (an
    AssertionError in load_job would surface as 'loading the final seeds failed (AssertionError)')."""
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("a checkpoint was loaded before the refusal")
    monkeypatch.setattr(demo.score, "Predictor", boom)
    monkeypatch.setattr(demo.score, "load_bert", boom)


def variant(tiny: dict[str, Any], tmp_path: Path, kind: str, record_t: float | None = None,
            manifest_t: float | None = None, record_sha256: str | None = None) -> dict[str, Any]:
    """The module's paths with `kind`'s record (chosen T record_t) and manifest (every seed's T manifest_t, bound to
    record_sha256) rewritten under tmp_path."""
    T = LAYA_T if kind == "laya" else BERT_T
    rec = write_record(tmp_path / ("%s.json" % kind), kind, T if record_t is None else record_t)
    seeds = [s[:4] + ((s[4] if manifest_t is None else manifest_t),) for s in tiny["seeds"][kind]]
    man = write_manifest(tmp_path / ("%s-final.json" % kind), kind, rec, seeds, record_sha256)
    return dict({k: tiny[k] for k in pd.KINDS}, **{kind: {"record": rec, "manifest": man}})


def test_refuses_a_missing_seed_dir(tiny: dict[str, Any], tmp_path: Path) -> None:
    root = copy_root(tiny, tmp_path)
    shutil.rmtree(os.path.join(root, "ckpt-s1"))
    msg = assert_refused(tiny, root, "--checkpoint-root/ckpt-s1 (seed 1) not found")
    assert msg.endswith("; pass --checkpoint-root DIR holding the final seeds the manifest names")


@pytest.mark.parametrize("name, rel", [("ckpt-s0", os.path.join("tokenizer", "tokenizer.json")),
                                       ("bert-s1", "vocab.txt")])
def test_refuses_a_changed_seed_before_any_load(tiny: dict[str, Any], tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch, name: str, rel: str) -> None:
    """One byte appended to a tokenizer file of a seed copy: refused by latency.bind_seed before any load."""
    root = copy_root(tiny, tmp_path)
    path = os.path.join(root, name, rel)
    assert os.path.isfile(path), path
    with open(path, "ab") as f:
        f.write(b" ")
    no_load(monkeypatch)
    msg = assert_refused(tiny, root, "--checkpoint-root/%s is not the checkpoint the final manifest names" % name)
    assert "identity or dump run id differs" in msg


def test_refuses_a_record_temperature_mismatch(tiny: dict[str, Any], tmp_path: Path) -> None:
    paths = variant(tiny, tmp_path, "laya", record_t=LAYA_T + 1e-9)
    assert_refused(paths, tiny["root"], "ckpt-s0 serves T 1.3, the postfit record %s chose T %r"
                   % (pilot.data.repo_relpath(paths["laya"]["record"]), LAYA_T + 1e-9))


def test_refuses_a_manifest_temperature_mismatch(tiny: dict[str, Any], tmp_path: Path) -> None:
    paths = variant(tiny, tmp_path, "laya", manifest_t=LAYA_T + 1e-9)
    assert_refused(paths, tiny["root"], "ckpt-s0 serves T 1.3, the final manifest records %r for seed 0"
                   % (LAYA_T + 1e-9))


def test_refuses_a_berturk_record_temperature_mismatch_before_its_load(tiny: dict[str, Any], tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """BERTurk's T comes from berturk_train.json, so both T checks run before score.load_bert."""
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("BERTurk was loaded before its T was checked")
    monkeypatch.setattr(demo.score, "load_bert", boom)
    paths = variant(tiny, tmp_path, "berturk", record_t=BERT_T + 1e-9)
    assert_refused(paths, tiny["root"], "bert-s0 serves T 1.0, the postfit record")


@pytest.mark.parametrize("which", ["record", "manifest"])
def test_refuses_a_changed_pinned_file(tiny: dict[str, Any], which: str) -> None:
    """A record or manifest whose sha256 is not the one protocol.json binds is refused naming the repo-relative path
    and `git checkout --`."""
    paths = tiny["laya"]
    pins = {"record_sha256": "0" * 64} if which == "record" else {"manifest_sha256": "0" * 64}
    with pytest.raises(SystemExit) as e:
        pd.bind_system("laya", paths["record"], paths["manifest"], **pins)
    rel = pilot.data.repo_relpath(paths[which])
    assert str(e.value.code) == ("%s has sha256 %s, protocol.json binds %s; restore it with `git checkout -- %s`"
                                 % (rel, pilot.data.sha256_file(paths[which]), "0" * 64, rel))


def test_refuses_a_manifest_bound_to_another_record(tiny: dict[str, Any], tmp_path: Path) -> None:
    paths = variant(tiny, tmp_path, "laya", record_sha256="1" * 64)
    msg = assert_refused(paths, tiny["root"], "%s is not bound to the postfit record %s"
                         % (pilot.data.repo_relpath(paths["laya"]["manifest"]),
                            pilot.data.repo_relpath(paths["laya"]["record"])))
    assert "git checkout --" in msg


def test_refuses_final_seeds_other_than_the_protocol(tiny: dict[str, Any]) -> None:
    paths = tiny["laya"]
    with open(paths["manifest"], encoding="utf-8") as f:
        seeds = json.load(f)["seeds"]
    keep = [{k: s[k] for k in ("seed", "train_run_id", "checkpoint_name", "dump_run_id")} for s in seeds]
    assert pd.bind_system("laya", paths["record"], paths["manifest"], final_seeds=keep)["seeds"] == seeds
    for other in (keep[:1], keep[::-1], [dict(keep[0], dump_run_id="laya-other")] + keep[1:]):
        with pytest.raises(SystemExit, match="protocol.json's final seeds of laya are"):
            pd.bind_system("laya", paths["record"], paths["manifest"], final_seeds=other)


def test_refuses_a_record_of_another_system(tiny: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="is a postfit record of berturk, not laya"):
        pd.bind_system("laya", tiny["berturk"]["record"], tiny["laya"]["manifest"])


def test_refuses_mps_when_it_is_not_available(tiny: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    no_load(monkeypatch)
    msg = assert_refused(tiny, tiny["root"], "--device mps, but MPS is not available here; use --device cpu",
                         device="mps")
    assert msg == "--device mps, but MPS is not available here; use --device cpu"


def test_refuses_a_load_failure_naming_only_its_type(tiny: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a: Any, **k: Any) -> Any:
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(demo.score, "Predictor", fail)
    msg = assert_refused(tiny, tiny["root"], "loading the final seeds failed (RuntimeError)")
    assert msg == "loading the final seeds failed (RuntimeError); restart `make demo`" and SENTINEL not in msg


def test_refuses_a_changed_protocol_in_one_line(tiny: dict[str, Any], tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """protocol_bindings calls require_frozen first: a protocol.json that is not the pinned one refuses the chat."""
    bad = tmp_path / "protocol.json"
    bad.write_bytes(b"{}\n")
    monkeypatch.setattr(pilot.protocol, "PROTOCOL_PATH", str(bad))
    state = demo.new_state("cpu")
    with ThreadPoolExecutor(max_workers=1) as ex:
        demo.start_loading(state, ex, pd.protocol_bindings, tiny["root"], "cpu", 1).result(timeout=TIMEOUT)
    assert state["chat"] == pd.REFUSED and "the frozen protocol is" in state["message"]
    assert "git checkout -- protocol.json" in state["message"]


def test_loading_state_answers_503(tiny: dict[str, Any]) -> None:
    """While the loader is still binding, /api/status reads loading and /api/classify answers 503 LOADING_TEXT."""
    import threading
    started, release = threading.Event(), threading.Event()

    def slow() -> dict[str, dict[str, Any]]:
        started.set()
        assert release.wait(TIMEOUT)
        return bindings_of(tiny)
    state = demo.new_state("cpu")
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = demo.start_loading(state, ex, slow, tiny["root"], "cpu", 1)
        assert started.wait(TIMEOUT)
        app = demo.create_app(state, ex, pd.build_views(REPO))
        with client(app) as c:
            assert c.get("/api/status").json() == {"chat": "loading", "message": None, "device": "cpu"}
            r = c.post("/api/classify", json={"text": TEXT})
            assert (r.status_code, r.json()) == (503, {"error": pd.LOADING_TEXT})
            assert c.get("/api/intents").status_code == 200
        release.set()
        future.result(timeout=TIMEOUT)
    assert state["chat"] == pd.READY, state["message"]


# ---- /api/classify request validation: fixed one-line errors, never the submitted text ----

JSON = {"Content-Type": "application/json"}
BODIES = [
    ("not_json", b"%s is not json" % SENTINEL.encode(), JSON, 400, pd.BODY_NOT_JSON),
    ("not_utf8", b'{"text": "' + SENTINEL.encode() + b'\xff"}', JSON, 400, pd.BODY_NOT_JSON),
    ("nested", b"[" * 4000 + b"]" * 4000, JSON, 400, None),     # parsed as a list, or too deep: a fixed line
    ("list", json.dumps([SENTINEL]).encode(), JSON, 400, pd.BODY_NOT_OBJECT),
    ("missing_text", json.dumps({"message": SENTINEL}).encode(), JSON, 400, pd.BODY_NOT_OBJECT),
    ("extra_key", json.dumps({"text": SENTINEL, "theta": 0.1}).encode(), JSON, 400, pd.BODY_NOT_OBJECT),
    ("not_string", json.dumps({"text": [SENTINEL]}).encode(), JSON, 400, pd.TEXT_NOT_STRING),
    ("whitespace", json.dumps({"text": " \t\n "}).encode(), JSON, 400, pd.TEXT_EMPTY),
    ("501_chars", json.dumps({"text": SENTINEL + "a" * (501 - len(SENTINEL))}).encode(), JSON, 400,
     "the message is longer than 500 characters; shorten it"),
    ("over_8192_bytes", json.dumps({"text": SENTINEL + "a" * 9000}).encode(), JSON, 413, demo.BODY_TOO_LARGE),
    ("text_plain", json.dumps({"text": SENTINEL}).encode(), {"Content-Type": "text/plain"}, 415, demo.NOT_JSON_TYPE),
    ("no_content_type", json.dumps({"text": SENTINEL}).encode(), {}, 415, demo.NOT_JSON_TYPE),
]


@pytest.mark.parametrize("name, body, headers, status, error", BODIES, ids=[b[0] for b in BODIES])
def test_classify_body_refused(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any], name: str, body: bytes,
                               headers: dict[str, str], status: int, error: str | None) -> None:
    _, _, app = loaded
    with client(app) as c:
        r = c.post("/api/classify", content=body, headers=headers)
    want = (pd.BODY_NOT_JSON, pd.BODY_NOT_OBJECT) if error is None else (error,)
    assert r.status_code == status and set(r.json()) == {"error"} and r.json()["error"] in want, (name, r.text)
    assert SENTINEL not in r.text and "\n" not in r.json()["error"]


def test_classify_body_of_500_characters_is_answered(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    _, _, app = loaded
    text = (TEXT + " ") * 40
    text = text[:500]
    assert len(text) == pd.MAX_CHARS
    with client(app) as c:
        r = c.post("/api/classify", json={"text": text}, headers={"Content-Type": "application/json; charset=utf-8"})
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"laya", "berturk"}


def test_classify_body_is_passed_unchanged(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    """parse_classify_body returns the text as sent: no strip, lowercasing, folding or normalization."""
    for text in ("  İptal  ", "İptal", "Kartım kayboldu .\n"):
        assert pd.parse_classify_body(json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")) == text


def test_classification_failure_names_only_the_type(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    """A forward raising ValueError(SENTINEL) answers 500 naming only the exception type; the chat stays ready."""
    state, _, _ = loaded

    def fail(text: str) -> Any:
        raise ValueError(SENTINEL + text)
    systems = {k: dict(v, seeds=[dict(s, forward=fail) for s in v["seeds"]]) for k, v in state["systems"].items()}
    broken = dict(state, systems=systems)
    with ThreadPoolExecutor(max_workers=1) as ex:
        with client(demo.create_app(broken, ex, pd.build_views(REPO))) as c:
            r = c.post("/api/classify", json={"text": TEXT})
            assert (r.status_code, r.json()) == (500, {"error": demo.CLASSIFY_FAILED % "ValueError"})
            assert SENTINEL not in r.text and TEXT not in r.text
            assert c.get("/api/status").json()["chat"] == "ready"


def test_views_route(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    """GET /api/intents serves the 48 stored questions in INTENTS order and the confidence rule; an unknown view is
    404; a refused view is 503 with its one line and source."""
    state, executor, app = loaded
    with client(app) as c:
        r = c.get("/api/intents")
        assert r.status_code == 200
        assert r.json() == {"tags": [{"tag": t, "question": QUESTIONS[t]["instructions"]} for t in INTENTS],
                            "confidence_rule": metrics.CONFIDENCE_RULE}
        assert c.get("/api/nothing").status_code == 404
        assert c.get("/api/status").status_code == 200
    refused = {"intents": {"error": "x.json changed; restore it", "source": "intents.py"}}
    with client(demo.create_app(state, executor, refused)) as c:
        r = c.get("/api/intents")
        assert (r.status_code, r.json()) == (503, refused["intents"])


def test_build_views_keeps_a_refusal_to_its_view(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad(root: str) -> dict[str, Any]:
        raise SystemExit("broken.json is not valid; restore it\nwith git")
    monkeypatch.setitem(pd.VIEWS, "broken", bad)
    monkeypatch.setitem(pd.VIEW_SOURCES, "broken", "results/broken.json")
    views = pd.build_views(REPO)
    assert views["broken"] == {"error": "broken.json is not valid; restore it with git", "source": "results/broken.json"}
    assert len(views["intents"]["tags"]) == 48


def test_untrusted_host_is_refused(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    from starlette.testclient import TestClient
    _, _, app = loaded
    with TestClient(app, base_url="http://evil.example") as c:
        assert c.get("/api/status").status_code == 400


def test_argument_errors_and_missing_ui_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """--port outside 1..65535, --threads 0, an unknown --device and any --host are argument errors (exit 2); a
    missing ui/dist/index.html is refused before anything loads or binds."""
    for argv in (["--port", "0"], ["--port", "65536"], ["--threads", "0"], ["--device", "cuda"],
                 ["--host", "0.0.0.0"]):
        with pytest.raises(SystemExit) as e:
            demo.main(argv)
        assert e.value.code == 2, argv
    capsys.readouterr()
    monkeypatch.setattr(demo, "UI_DIST", str(tmp_path / "dist"))
    monkeypatch.setattr(demo, "start_loading", lambda *a, **k: pytest.fail("loading started without a UI build"))
    with pytest.raises(SystemExit, match="ui/dist/index.html not found; build the UI first"):
        demo.main(["--device", "cpu"])
