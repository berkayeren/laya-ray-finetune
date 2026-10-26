"""demo.py's chat path on CPU, offline: tiny final seeds of both systems, bound to synthetic postfit/1 records and
postfit-final/1 manifests, loaded through score.py's loaders on the demo's worker thread, answer one message over HTTP
exactly as the frozen verdict's seed-averaged definition computes it.

Two tiny Laya seeds come from tests/tiny_laya.py (T 1.3); two tiny BERTurk seeds are the epoch-001 and epoch-002 exports
of one 2-epoch berturk_train.py run on CPU over the synthetic snapshot of tests/conftest.py (T 1.0). The records' chosen
θ is 0.5 with one per-tag θ (0.3 for the first INTENTS tag). The startup refusals and the request validation are tested
here too, and so are the D-13 guardrails: the loopback-only bind with no --host flag, the CLI defaults, the unbuilt-UI
refusal, the trusted-host gate, docs and telemetry off, and a sentinel message that reaches no output. The training
view: the tiny Laya seeds carry a synthetic Ray Train run record (the BERTurk exports their real one), served by GET
/api/training from the bound checkpoints only, per model unavailable when a seed dir is missing, refused when the rows
differ from the manifest's, never filled from the manifest. Offline, no Ray, no weights: runs in `make test`.
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
import pilot.latency
import pilot.runs
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


Seed = tuple[int, str, dict[str, Any], str, float, list[dict[str, Any]]]


def write_manifest(path: Path, kind: str, record: str, seeds: list[Seed], record_sha256: str | None = None) -> str:
    """A synthetic postfit-final/1 manifest of `seeds` [(seed, checkpoint_name, identity, dump_run_id, T, epochs)],
    bound to `record`'s sha256 (or `record_sha256`); each seed's "epochs" holds its rows on pd.SHARED_EPOCH_KEYS, as
    postfit.py --final-manifest writes them."""
    m = {"schema": "postfit-final/1",
         "postfit": {"path": "results/postfit/synthetic.json",
                     "sha256": record_sha256 or pilot.data.sha256_file(record)},
         "final_split": "final_train", "chosen": {},
         "seeds": [{"seed": seed, "train_run_id": "%s-final_train-s%d-synthetic" % (kind, seed),
                    "checkpoint_name": name, "checkpoint_identity": ident, "dump_run_id": rid, "temperature": t,
                    "epochs": [{k: r[k] for k in pd.SHARED_EPOCH_KEYS} for r in epochs]}
                   for seed, name, ident, rid, t, epochs in seeds]}
    if kind == "berturk":
        m["system_kind"] = "berturk"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    return str(path)


def ray_rows(seed: int) -> list[dict[str, Any]]:
    """Three synthetic Ray Train epoch rows of a tiny Laya seed, every pilot.runs.EPOCH_FIELDS key set (CPU: the
    driver and allocator fields None), with losses distinct per seed."""
    return [{"epoch": e, "loss": 0.7 / e + 0.0123456789 * (seed + 1), "train_s": 10.0 * e + seed + 0.25,
             "save_s": 0.5, "optimizer_steps": 4 * e, "n_items": 40, "peak_total_bytes": 1000 * e + seed,
             "peak_driver_bytes": None, "peak_rss_bytes": 1000 * e + seed, "current_allocated_bytes": None,
             "swap_delta_bytes": 0, "segment": 0, "resumed": False} for e in (1, 2, 3)]


def write_ray_record(ckpt: str, seed: int) -> list[dict[str, Any]]:
    """Give a tiny Laya checkpoint a training.ray_train.record (device cpu, 12 train messages, ray_rows(seed)) in its
    rl_agent_config.json, as ray_train.py writes one; call it before the checkpoint's identity is taken. Returns the
    rows."""
    path = os.path.join(ckpt, "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    rows = ray_rows(seed)
    cfg.setdefault("training", {})["ray_train"] = {"record": {"device": "cpu", "n_train_messages": 12, "epochs": rows}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return rows


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """{root, seeds: {kind: [(seed, name, identity, dump_run_id, T)]}, laya: {record, manifest}, berturk: {...}}:
    ckpt-s0, ckpt-s1 (tiny Laya) and bert-s0, bert-s1 (the two exports of a 2-epoch tiny BERTurk run) under one
    checkpoint root, with their synthetic records and manifests. Built once for the module."""
    from tiny_berturk import tiny_base
    tmp = tmp_path_factory.mktemp("demo_tiny")
    root = tmp / "ckpts"
    seeds: dict[str, list[Seed]] = {"laya": [], "berturk": []}
    for seed in (0, 1):
        name = "ckpt-s%d" % seed
        rows = write_ray_record(tiny_checkpoint(root / name, temperature=LAYA_T, seed=seed), seed)
        ident = identity(str(root / name))
        seeds["laya"].append((seed, name, ident, dumps.run_id("laya", ident), LAYA_T, rows))
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
        rows = berturk.read_record(str(root / name), ident)["record"]["epochs"]
        seeds["berturk"].append((seed, name, ident, berturk.dump_run_id(ident), BERT_T, rows))
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
    for seed, name, ident, _, _, _ in tiny["seeds"][kind]:
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
        assert c.get("/api/status").json() == {"chat": "ready", "message": None, "device": "cpu", "training": "ready"}
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


def checkpoint_rows(root: str, kind: str, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(run record, epoch rows on the kind's EPOCH_FIELDS) read straight from a seed dir's file, without pilot.demo."""
    if kind == "laya":
        with open(os.path.join(root, name, "rl_agent_config.json"), encoding="utf-8") as f:
            record = json.load(f)["training"]["ray_train"]["record"]
        fields = pilot.runs.EPOCH_FIELDS
    else:
        with open(os.path.join(root, name, berturk.RECORD), encoding="utf-8") as f:
            record = json.load(f)["record"]
        fields = berturk.EPOCH_FIELDS
    return record, [{k: r[k] for k in fields} for r in record["epochs"]]


def test_training_rows_from_bound_checkpoints(tiny: dict[str, Any],
                                              loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    """GET /api/training serves every seed's per-epoch rows as its checkpoint records them (the tiny Laya seeds' Ray
    Train rows, the tiny BERTurk exports' HF Trainer rows: 1 and 2 epochs), the kind's EPOCH_FIELDS and the sum of
    train_s; /api/status reads training ready."""
    _, _, app = loaded
    with client(app) as c:
        assert c.get("/api/status").json()["training"] == "ready"
        r = c.get("/api/training")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"status", "systems"} and body["status"] == "ready" and set(body["systems"]) == set(pd.KINDS)
    for kind, fields in (("laya", pilot.runs.EPOCH_FIELDS), ("berturk", berturk.EPOCH_FIELDS)):
        t = body["systems"][kind]
        assert set(t) == {"status", "fields", "total_train_s", "seeds"} and t["status"] == "ok", t
        assert t["fields"] == list(fields)
        total = 0.0
        assert [s["seed"] for s in t["seeds"]] == [0, 1]
        for got, (seed, name, _, _, _, _) in zip(t["seeds"], tiny["seeds"][kind]):
            record, rows = checkpoint_rows(tiny["root"], kind, name)
            assert got == {"seed": seed, "checkpoint_name": name,
                           "train_run_id": "%s-final_train-s%d-synthetic" % (kind, seed), "device": "cpu",
                           "n_train_messages": record["n_train_messages" if kind == "laya" else "n_messages"],
                           "rows": rows}
            assert [list(row) for row in got["rows"]] == [list(fields)] * len(rows)
            total += sum(row["train_s"] for row in rows)
        assert t["total_train_s"] == total
    assert [len(s["rows"]) for s in body["systems"]["laya"]["seeds"]] == [3, 3]
    assert [len(s["rows"]) for s in body["systems"]["berturk"]["seeds"]] == [1, 2]


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
            assert c.get("/api/status").json() == {"chat": "refused", "message": msg, "device": device,
                                                   "training": "ready"}
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
    seeds = [s[:4] + ((s[4] if manifest_t is None else manifest_t),) + s[5:] for s in tiny["seeds"][kind]]
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


# ---- the training view: per model, from the checkpoints only, never a fallback (D-09, the user decision of 2026-10-25)

def training_of(paths: dict[str, Any], root: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """(state, /api/training JSON, its text) after loading `paths` from `root`; /api/status must read training ready
    and /api/intents must answer 200 whatever the chat state."""
    state, executor, app = load(paths, root)
    try:
        with client(app) as c:
            st = c.get("/api/status").json()
            assert st["training"] == "ready" and st["chat"] == state["chat"], st
            r = c.get("/api/training")
            assert r.status_code == 200, r.text
            assert c.get("/api/intents").status_code == 200
    finally:
        executor.shutdown(wait=True)
    return state, r.json(), r.text


def test_training_unavailable_without_checkpoints(tiny: dict[str, Any], tmp_path: Path) -> None:
    """ckpt-s1 removed: Laya's training view is unavailable with latency.bind_seed's one line (no rows), BERTurk's
    still charts, and the chat is refused with the same line."""
    root = copy_root(tiny, tmp_path)
    shutil.rmtree(os.path.join(root, "ckpt-s1"))
    state, body, _ = training_of(tiny, root)
    laya, bert = body["systems"]["laya"], body["systems"]["berturk"]
    assert body["status"] == "ready" and set(laya) == {"status", "message"} and laya["status"] == "unavailable"
    assert laya["message"].startswith("--checkpoint-root/ckpt-s1 (seed 1) not found") and "\n" not in laya["message"]
    assert state["chat"] == pd.REFUSED and state["message"] == laya["message"] and state["systems"] is None
    assert bert["status"] == "ok" and [len(s["rows"]) for s in bert["seeds"]] == [1, 2]
    want = [checkpoint_rows(root, "berturk", n)[1] for n in ("bert-s0", "bert-s1")]
    assert [s["rows"] for s in bert["seeds"]] == want


def test_training_refuses_rows_that_differ_from_the_manifest(tiny: dict[str, Any], tmp_path: Path) -> None:
    """A manifest whose loss on seed 1's last epoch differs from the checkpoint's makes that model's training view
    unavailable with one line naming the seed dir, the epoch and the key; the other model still charts and the chat
    (bound seeds, so it loads) answers."""
    for kind in pd.KINDS:
        T = LAYA_T if kind == "laya" else BERT_T
        d = tmp_path / kind
        d.mkdir()
        seeds = list(tiny["seeds"][kind])
        s1 = seeds[1]
        rows = [dict(r) for r in s1[5]]
        rows[-1]["loss"] = rows[-1]["loss"] + 1e-6
        seeds[1] = s1[:5] + (rows,)
        rec = write_record(d / ("%s.json" % kind), kind, T)
        man = write_manifest(d / ("%s-final.json" % kind), kind, rec, seeds)
        paths = dict({k: tiny[k] for k in pd.KINDS}, **{kind: {"record": rec, "manifest": man}})
        state, body, _ = training_of(paths, tiny["root"])
        got, other = body["systems"][kind], body["systems"]["berturk" if kind == "laya" else "laya"]
        assert got == {"status": "unavailable",
                       "message": "--checkpoint-root/%s: the run record's epoch rows differ from the final manifest's "
                                  "on epoch %d: loss; restore the final seed" % (s1[1], rows[-1]["epoch"])}, got
        assert other["status"] == "ok" and len(other["seeds"]) == 2
        assert state["chat"] == pd.READY, state["message"]


def test_training_never_reads_the_manifest_rows(tiny: dict[str, Any], tmp_path: Path) -> None:
    """Every seed dir removed: both training views are unavailable, and the /api/training JSON holds none of the
    manifests' epoch values (the manifest rows are never a fallback)."""
    root = copy_root(tiny, tmp_path)
    for kind in pd.KINDS:
        for s in tiny["seeds"][kind]:
            shutil.rmtree(os.path.join(root, s[1]))
    state, body, text = training_of(tiny, root)
    assert state["chat"] == pd.REFUSED
    for kind in pd.KINDS:
        t = body["systems"][kind]
        assert set(t) == {"status", "message"} and t["status"] == "unavailable", t
        first = tiny["seeds"][kind][0][1]
        assert t["message"].startswith("--checkpoint-root/%s (seed 0) not found" % first), t
    assert "rows" not in text and "total_train_s" not in text
    for kind in pd.KINDS:
        with open(tiny[kind]["manifest"], encoding="utf-8") as f:
            manifest = json.load(f)
        values = [r[k] for s in manifest["seeds"] for r in s["epochs"] for k in ("loss", "train_s")]
        assert values
        for v in values:
            assert repr(v) not in text and json.dumps(v) not in text, (kind, v)


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
            assert c.get("/api/status").json() == {"chat": "loading", "message": None, "device": "cpu",
                                                   "training": "loading"}
            r = c.post("/api/classify", json={"text": TEXT})
            assert (r.status_code, r.json()) == (503, {"error": pd.LOADING_TEXT})
            assert c.get("/api/intents").status_code == 200
            assert (c.get("/api/training").status_code, c.get("/api/training").json()) == (200, {"status": "loading"})
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


def test_a_refusal_inside_a_forward_refuses_the_chat(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    """A forward whose device or dtype check raises SystemExit (the model moved or was re-cast mid-forward) answers
    500 with its line and `restart make demo`, then the chat reads refused with that line and every later message
    answers 503 with it: the refused model never runs again."""
    state, _, _ = loaded
    calls = []

    def moved(text: str) -> Any:
        calls.append(1)
        raise SystemExit("the model left mps for cpu\nmid-forward; check --device")
    systems = {k: dict(v, seeds=[dict(s, forward=moved) for s in v["seeds"]]) for k, v in state["systems"].items()}
    broken = dict(state, systems=systems)
    line = demo.CHAT_BROKEN % "the model left mps for cpu mid-forward; check --device"
    with ThreadPoolExecutor(max_workers=1) as ex:
        with client(demo.create_app(broken, ex, pd.build_views(REPO))) as c:
            r = c.post("/api/classify", json={"text": TEXT})
            assert (r.status_code, r.json()) == (500, {"error": line}) and TEXT not in r.text
            assert c.get("/api/status").json() == {"chat": "refused", "message": line, "device": state["device"],
                                                   "training": "ready"}
            r = c.post("/api/classify", json={"text": TEXT})
            assert (r.status_code, r.json()) == (503, {"error": line})
    assert len(calls) == 1 and state["chat"] == pd.READY      # the module's own state is untouched


def test_a_request_queued_behind_a_refusal_runs_no_forward(
        loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any]) -> None:
    """11-REVIEW IN-08: a message accepted while the chat read ready, then queued on the one worker behind the forward
    that refused the chat, answers 503 with the refusal's line and runs no forward of the refused model: the worker
    re-reads the chat state (classify_if_ready), which alone also returns None without a forward."""
    import threading
    import types
    state, _, _ = loaded
    calls = []

    def counted(text: str) -> Any:
        calls.append(1)
        raise AssertionError("a refused model ran a forward")
    systems = {k: dict(v, seeds=[dict(s, forward=counted) for s in v["seeds"]]) for k, v in state["systems"].items()}
    broken = dict(state, systems=systems)
    line = demo.CHAT_BROKEN % "the model left mps for cpu mid-forward; check --device"
    release, queued = threading.Event(), threading.Event()

    def refusing_forward() -> None:                     # the request ahead in the queue, whose forward refuses
        assert release.wait(TIMEOUT)
        broken["message"], broken["chat"] = line, pd.REFUSED
    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(refusing_forward)

        def submit(fn: Any, *args: Any) -> Any:
            f = ex.submit(fn, *args)
            queued.set()
            return f
        with client(demo.create_app(broken, types.SimpleNamespace(submit=submit), pd.build_views(REPO))) as c:
            out: dict[str, Any] = {}
            t = threading.Thread(target=lambda: out.update(r=c.post("/api/classify", json={"text": TEXT})))
            t.start()
            assert queued.wait(TIMEOUT), "the request never reached the worker queue"
            release.set()
            t.join(TIMEOUT)
            assert (out["r"].status_code, out["r"].json()) == (503, {"error": line})
    assert calls == [] and demo.classify_if_ready(broken, TEXT) is None and calls == []
    assert state["chat"] == pd.READY                     # the module's own state is untouched


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


@pytest.mark.parametrize("exc", [PermissionError, IndexError, KeyError])
def test_build_views_keeps_any_exception_to_its_view(monkeypatch: pytest.MonkeyPatch, exc: type) -> None:
    """A builder raising a non-SystemExit exception becomes that view's {error, source} naming only the exception
    type (never its text); every other view still builds."""
    def bad(root: str) -> dict[str, Any]:
        raise exc(SENTINEL)
    monkeypatch.setitem(pd.VIEWS, "broken", bad)
    monkeypatch.setitem(pd.VIEW_SOURCES, "broken", "results/broken.json")
    views = pd.build_views(REPO)
    assert views["broken"] == {"error": pd.VIEW_FAILED % ("broken", exc.__name__), "source": "results/broken.json"}
    assert pd.view_refused(views["broken"]) and SENTINEL not in views["broken"]["error"]
    assert len(views["intents"]["tags"]) == 48 and not pd.view_refused(views["verdict"])


def test_a_seed_binding_exception_hides_only_its_system(tiny: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """bind_seeds raising PermissionError for Laya: Laya's training view is unavailable with LOAD_FAILED naming only
    the type, BERTurk's still charts, and the chat is refused with the same line before any load."""
    real = demo.bind_seeds

    def bind(binding: dict[str, Any], root: str, stopped: Any = None) -> Any:
        if binding["kind"] == "laya":
            raise PermissionError(SENTINEL)
        return real(binding, root, stopped)
    monkeypatch.setattr(demo, "bind_seeds", bind)
    no_load(monkeypatch)
    state, body, text = training_of(tiny, tiny["root"])
    line = demo.LOAD_FAILED % "PermissionError"
    assert body["systems"]["laya"] == {"status": "unavailable", "message": line}
    assert body["systems"]["berturk"]["status"] == "ok"
    assert state["chat"] == pd.REFUSED and state["message"] == line and SENTINEL not in text


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


# ---- guardrails (D-13, D-14, the user decision of 2026-10-25): loopback only, no --host, local Host names only ----

def run_main(argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[Any]]:
    """demo.main(argv) with uvicorn.run and demo.start_loading as recorders and demo.UI_DIST a tmp dir holding
    index.html. Returns {"run": [(args, kwargs)], "loading": [(args, kwargs)]} in call order."""
    import uvicorn
    calls: dict[str, list[Any]] = {"run": [], "loading": []}
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
    monkeypatch.setattr(demo, "UI_DIST", str(dist))
    monkeypatch.setattr(demo, "start_loading", lambda *a, **k: calls["loading"].append((a, k)))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls["run"].append((a, k)))
    demo.main(argv)
    return calls


def test_loopback_bind_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """main() serves on 127.0.0.1:8765 with uvicorn's access log off and log level warning; loading starts on the
    repo root with --device mps."""
    calls = run_main([], monkeypatch, tmp_path)
    assert len(calls["run"]) == 1 and len(calls["loading"]) == 1
    (args, kwargs), = calls["run"]
    assert len(args) == 1 and kwargs == {"host": "127.0.0.1", "port": 8765, "access_log": False,
                                         "log_level": "warning"}
    assert pd.HOST == "127.0.0.1" and pd.PORT == 8765
    (largs, _), = calls["loading"]
    assert largs[3:] == (REPO, "mps", pilot.latency.THREADS)


def test_main_stops_the_load_when_the_server_returns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Once uvicorn.run returns (Ctrl-C), main sets state["stop"] and shuts the worker down without waiting, so the
    interpreter's exit join waits for the one seed in flight only (11-REVIEW WR-06)."""
    calls = run_main(["--device", "cpu"], monkeypatch, tmp_path)
    (largs, _), = calls["loading"]
    state, executor = largs[0], largs[1]
    assert state["stop"] is True
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)


def test_load_stops_at_the_next_seed(tiny: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """load_system loads no further seed once stopped() reads true (SystemExit STOPPED), and load_job with stop set
    binds and loads nothing and refuses the chat with STOPPED."""
    loads = []

    def fake_seed(binding: dict[str, Any], s: dict[str, Any], *a: Any) -> dict[str, Any]:
        loads.append(s["seed"])
        return {"temperature": 1.0, "forward": None, "check": None}
    monkeypatch.setattr(demo, "_laya_seed", fake_seed)
    bindings = bindings_of(tiny)
    bound = demo.bind_seeds(bindings["laya"], tiny["root"])
    with pytest.raises(SystemExit) as e:
        demo.load_system(bindings["laya"], bound, "cpu", 1, lambda: len(loads) >= 1)
    assert e.value.code == demo.STOPPED and loads == [bound[0][0]["seed"]]
    no_load(monkeypatch)
    monkeypatch.setattr(demo, "bind_seeds", lambda *a: pytest.fail("a seed was bound after stop"))
    state = demo.new_state("cpu")
    state["stop"] = True
    demo.load_job(state, lambda: bindings, tiny["root"], "cpu", 1)
    assert (state["chat"], state["message"], state["systems"]) == (pd.REFUSED, demo.STOPPED, None)
    assert state["training_status"] == pd.READY


def test_binding_stops_at_the_next_seed(tiny: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """11-REVIEW IN-09: bind_seeds hashes no further seed once stopped() reads true, so a Ctrl-C during the binding
    stage waits for the one seed being hashed, not for the whole system: load_job with stop set during the first
    seed's hashing hashes no other seed, reads no training rows, loads nothing and refuses the chat with STOPPED."""
    bindings = bindings_of(tiny)
    real = demo.latency.bind_seed
    hashed = []
    monkeypatch.setattr(demo.latency, "bind_seed", lambda *a: hashed.append(a[2]["seed"]) or real(*a))
    with pytest.raises(SystemExit) as e:
        demo.bind_seeds(bindings["laya"], tiny["root"], lambda: len(hashed) >= 1)
    assert e.value.code == demo.STOPPED and hashed == [bindings["laya"]["seeds"][0]["seed"]]
    no_load(monkeypatch)
    monkeypatch.setattr(pd, "training_system", lambda *a: pytest.fail("training rows were read after stop"))
    state = demo.new_state("cpu")

    def stop_while_hashing(*a: Any) -> Any:
        hashed.append(a[2]["seed"])
        state["stop"] = True
        return real(*a)
    hashed.clear()
    monkeypatch.setattr(demo.latency, "bind_seed", stop_while_hashing)
    demo.load_job(state, lambda: bindings, tiny["root"], "cpu", 1)
    assert len(hashed) == 1
    assert (state["chat"], state["message"], state["systems"]) == (pd.REFUSED, demo.STOPPED, None)
    assert state["training_status"] == pd.READY
    assert state["training"] == {k: pd.training_unavailable(demo.STOPPED) for k in pd.KINDS}


def test_loopback_bind_with_another_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = run_main(["--port", "9001", "--device", "cpu"], monkeypatch, tmp_path)
    (_, kwargs), = calls["run"]
    assert kwargs == {"host": "127.0.0.1", "port": 9001, "access_log": False, "log_level": "warning"}


@pytest.mark.parametrize("var", ["HOST", "UVICORN_HOST", "DEMO_HOST"])
def test_loopback_host_ignores_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, var: str) -> None:
    """The host is the constant pilot.demo.HOST: no environment variable moves it."""
    monkeypatch.setenv(var, "0.0.0.0")
    (_, kwargs), = run_main([], monkeypatch, tmp_path)["run"]
    assert kwargs["host"] == "127.0.0.1"


def test_no_host_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """The parser has exactly --device (mps|cpu, default mps), --checkpoint-root (default the repo root), --port
    (default 8765) and --threads; no option names a host, and --host or --device cuda is an argument error."""
    ap = demo.parser()
    opts = sorted(o for a in ap._actions for o in a.option_strings)
    assert opts == sorted(["--checkpoint-root", "--device", "--port", "--threads", "-h", "--help"])
    assert not [o for o in opts if "host" in o.lower()]
    a = ap.parse_args([])
    assert (a.device, a.port, a.checkpoint_root, a.threads) == ("mps", 8765, REPO, pilot.latency.THREADS)
    assert next(x for x in ap._actions if "--device" in x.option_strings).choices == ("mps", "cpu")
    for argv in (["--host", "0.0.0.0"], ["--host=127.0.0.1"], ["--device", "cuda"], ["--bind", "0.0.0.0"]):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(argv)
        assert e.value.code == 2, argv
    capsys.readouterr()


def test_main_refuses_unbuilt_ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A ui/dist without index.html is refused naming the build command, before loading starts or uvicorn runs."""
    import uvicorn
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "assets.js").write_text("", encoding="utf-8")
    monkeypatch.setattr(demo, "UI_DIST", str(tmp_path / "dist"))
    monkeypatch.setattr(demo, "start_loading", lambda *a, **k: pytest.fail("loading started without a UI build"))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("uvicorn ran without a UI build"))
    with pytest.raises(SystemExit) as e:
        demo.main(["--device", "cpu"])
    assert "npm --prefix ui run build" in str(e.value.code) and "\n" not in str(e.value.code)


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "localhost.evil.example", "0.0.0.0",
                                  "192.168.1.5", "[::1]"])
def test_trusted_host_refuses_other_hosts(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any], host: str) -> None:
    """A request addressed to any other Host (a DNS-rebinding page, a LAN address) answers 400 on every route."""
    from starlette.testclient import TestClient
    _, _, app = loaded
    with TestClient(app, base_url="http://%s" % host) as c:
        assert c.get("/api/status").status_code == 400
        assert c.get("/api/intents").status_code == 400
        assert c.post("/api/classify", json={"text": TEXT}).status_code == 400


@pytest.mark.parametrize("base_url", ["http://127.0.0.1", "http://localhost", "http://127.0.0.1:8765",
                                      "http://localhost:9001"])
def test_trusted_host_accepts_local_names(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any],
                                          base_url: str) -> None:
    from starlette.testclient import TestClient
    _, _, app = loaded
    with TestClient(app, base_url=base_url) as c:
        assert c.get("/api/status").status_code == 200


# ---- no docs, no telemetry, no message text in any output (D-13) ----

MESSAGE_SENTINEL = "SENTINEL-c41d Kartımı kaybettim"


@pytest.mark.parametrize("with_ui", [False, True])
def test_docs_and_openapi_off(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any], tmp_path: Path,
                              with_ui: bool) -> None:
    """/docs, /redoc and /openapi.json answer 404, with and without the built UI mounted at /."""
    state, executor, _ = loaded
    ui = None
    if with_ui:
        (tmp_path / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
        ui = str(tmp_path)
    app = demo.create_app(state, executor, pd.build_views(REPO), ui)
    assert (app.docs_url, app.redoc_url, app.openapi_url) == (None, None, None)
    with client(app) as c:
        for path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
            r = c.get(path)
            assert r.status_code == 404, (path, r.status_code)
            assert "swagger" not in r.text.lower() and "redoc" not in r.text.lower()
        if with_ui:
            assert c.get("/").status_code == 200


def test_telemetry_off(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """create_app passes telemetry=demo.TELEMETRY (every flag False, every key one fastapi's TelemetryConfig knows) and
    the three doc URLs None; the app's effective config keeps every flag False even with FASTAPI_OTEL_AUTO_CONFIGURE
    set, and its native telemetry is disabled."""
    import fastapi
    from fastapi.telemetry import TelemetryConfig
    state, executor, _ = loaded
    seen: list[dict[str, Any]] = []
    real = fastapi.FastAPI

    def recording(*a: Any, **k: Any) -> Any:
        seen.append(k)
        return real(*a, **k)
    monkeypatch.setattr(fastapi, "FastAPI", recording)
    monkeypatch.setenv("FASTAPI_OTEL_AUTO_CONFIGURE", "true")
    app = demo.create_app(state, executor, pd.build_views(REPO))
    assert len(seen) == 1
    k = seen[0]
    assert k["telemetry"] == demo.TELEMETRY and (k["docs_url"], k["redoc_url"], k["openapi_url"]) == (None,) * 3
    assert demo.TELEMETRY == {"tracing": False, "metrics": False, "logs": False, "operation_spans": False,
                              "auto_configure": False}
    assert set(demo.TELEMETRY) <= set(TelemetryConfig.__annotations__)
    assert all(app._telemetry[f] is False for f in demo.TELEMETRY)
    assert app._native_telemetry.enabled() is False


def test_no_log_of_message_text(loaded: tuple[dict[str, Any], ThreadPoolExecutor, Any],
                                capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    """A sentinel message classified through the tiny seeds (and sent in a refused body and through a failing
    forward) appears in no captured stdout or stderr (file-descriptor level), no log record of any logger at DEBUG,
    no response body and nowhere in the chat state."""
    import logging
    state, _, app = loaded
    caplog.set_level(logging.DEBUG)                         # the root logger: every library's records propagate here
    capfd.readouterr()
    with client(app) as c:
        r = c.post("/api/classify", json={"text": MESSAGE_SENTINEL})
        assert r.status_code == 200, r.status_code
        assert set(r.json()) == {"laya", "berturk"}
        bad = c.post("/api/classify", json={"text": MESSAGE_SENTINEL, "extra": 1})
        assert bad.status_code == 400
    responses = r.text + bad.text

    def fail(text: str) -> Any:
        raise ValueError(text)                              # an exception carrying the text must not reach output
    systems = {k: dict(v, seeds=[dict(s, forward=fail) for s in v["seeds"]]) for k, v in state["systems"].items()}
    with ThreadPoolExecutor(max_workers=1) as ex:
        with client(demo.create_app(dict(state, systems=systems), ex, pd.build_views(REPO))) as c:
            failed = c.post("/api/classify", json={"text": MESSAGE_SENTINEL})
            assert failed.status_code == 500
    responses += failed.text
    out, err = capfd.readouterr()
    logged = [rec.getMessage() for rec in caplog.records] + [caplog.text]
    for needle in (MESSAGE_SENTINEL, "SENTINEL-c41d", json.dumps(MESSAGE_SENTINEL)[1:-1]):
        assert needle not in out and needle not in err, "message text reached stdout or stderr"
        assert not [m for m in logged if needle in m], "message text reached a log record"
        assert needle not in responses, "message text was echoed"
        assert needle not in repr(state), "message text was stored in the chat state"
