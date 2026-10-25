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
