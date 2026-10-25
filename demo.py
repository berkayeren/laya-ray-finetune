"""A local, single-user demo over the finished, frozen results: chat with both fine-tuned systems, browse the records.

The chat: one message goes in, and fine-tuned Laya and fine-tuned BERTurk each answer it as the seed-averaged system the
frozen verdict judged (pilot.demo.system_answer: per final seed the probabilities at its served T, averaged over the
three seeds, predicted at the postfit record's theta.shipped with p >= θ, and the message confidence of
pilot.metrics.CONFIDENCE_RULE). The six final seeds load once, in this process, through the loaders that scored the
dumps (latency.py's precedent): score.resolve_checkpoint + score.Predictor for Laya (fp32, autocast off, check_agent
after every forward), score.check_berturk + score.load_bert for BERTurk (fp32, check_bert after every forward). No Ray
module is imported. The views (GET /api/<name>, pilot.demo.VIEWS) are built once at startup from the committed
records.

What it never does: train, call score.main (so score.py's protocol gate is neither triggered nor weakened), write
under results/, data/ or protocol.json, store or log message text (uvicorn's access log is off, FastAPI telemetry is
off, no handler prints a body), or bind beyond 127.0.0.1 (there is no --host flag).

Design: one worker thread (a ThreadPoolExecutor with max_workers 1) binds, loads and runs every forward, which
serializes MPS work: one forward in flight. Every repo refusal is a SystemExit, which a thread's excepthook ignores
and FastAPI lets escape, so it is caught at every boundary: in load_job (the chat state becomes refused with the
one line) and in every handler (a JSON error with the one line). A reply holds both answers or neither.

Refusals: at startup, each a one-line `<what>; <fix>` that leaves /api/status at refused, /api/classify answering 503
and every other route serving: protocol.json changed (pilot.protocol.require_frozen); a postfit record or final
manifest that does not hash to what protocol.json binds, a manifest not bound to its record or not listing the
protocol's final seeds (pilot.demo.bind_system); a missing seed dir or one whose identity or dump run id is not the
manifest's (latency.bind_seed, before any load); --device mps without MPS; a served T other than the manifest seed's
or the record's chosen T beyond 1e-12 (latency.check_temperature, pilot.demo.check_record_temperature). Per request:
400 (not a JSON object {"text": str}, empty, longer than 500 characters), 413 (a body over 8192 bytes), 415 (not
application/json), 503 (the chat is loading or refused), 500 (the classification failed: the exception type only,
never the text). Before the server starts: a missing ui/dist/index.html (build the UI first).

usage: python demo.py [--device mps|cpu] [--checkpoint-root DIR] [--port 8765] [--threads 4]
       make demo                                            # builds the UI, then runs python demo.py --device mps
"""
import argparse
import asyncio
import functools
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

import latency
import pilot.latency
import score
from intents import QUESTIONS
from pilot import demo as pd
from pilot import laya_adapter as la
from pilot.data import REPO_ROOT

UI_DIST = os.path.join(REPO_ROOT, "ui", "dist")
TELEMETRY = {"tracing": False, "metrics": False, "logs": False, "operation_spans": False, "auto_configure": False}
CLASSIFY_FAILED = "classification failed (%s); try again; if it repeats, restart `make demo`"
LOAD_FAILED = "loading the final seeds failed (%s); restart `make demo`"
NOT_JSON_TYPE = "the request body must be sent as Content-Type application/json; send {\"text\": \"...\"}"
BODY_TOO_LARGE = "the request body is larger than %d bytes; send one message of at most %d characters" % (
    pd.MAX_BODY_BYTES, pd.MAX_CHARS)


def parser() -> argparse.ArgumentParser:
    """The argument parser of demo.py (no --host: the server binds 127.0.0.1 only)."""
    ap = argparse.ArgumentParser(description="local demo: chat with fine-tuned Laya and BERTurk, browse the records")
    ap.add_argument("--device", default="mps", choices=("mps", "cpu"))
    ap.add_argument("--checkpoint-root", default=REPO_ROOT,
                    help="directory holding each final seed's checkpoint_name (default: the repo root)")
    ap.add_argument("--port", type=int, default=pd.PORT, help="port on 127.0.0.1 (default %d)" % pd.PORT)
    ap.add_argument("--threads", type=int, default=pilot.latency.THREADS, help="torch threads (default 4)")
    return ap


def bind_seeds(binding: dict[str, Any], root: str) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    """[(manifest seed, checkpoint dir, identity)] of every bound seed under `root`, via latency.bind_seed (identity
    and dump run id); hashes only, run for every seed before any load."""
    return [(s,) + latency.bind_seed(binding["kind"], root, s) for s in binding["seeds"]]


def _laya_seed(binding: dict[str, Any], s: dict[str, Any], ckpt: str, ident: dict[str, Any], device: str,
               threads: int) -> dict[str, Any]:
    """{temperature, forward, check} of one Laya seed loaded through score.Predictor; its T checked against the manifest
    seed and the record before any forward."""
    import torch
    model_dir, expected = score.resolve_checkpoint(ckpt, ident)
    la.check_version()
    p = score.Predictor(model_dir, device, expected, threads)
    latency.check_temperature(p.temperature, s)
    pd.check_record_temperature(p.temperature, binding, s)
    agent = p.agent

    def forward(text: str) -> np.ndarray:
        with torch.no_grad():
            logits = la.noul_logits(agent, [text], QUESTIONS)
        score.check_agent(agent, device)                    # laya may have moved or re-cast it mid-forward
        return np.asarray(logits, dtype=np.float32)
    return {"temperature": float(p.temperature), "forward": forward,
            "check": functools.partial(score.check_agent, agent, device)}


def _berturk_seed(binding: dict[str, Any], s: dict[str, Any], ckpt: str, ident: dict[str, Any], device: str,
                  threads: int) -> dict[str, Any]:
    """{temperature, forward, check} of one BERTurk seed loaded through score.load_bert; its T (berturk_train.json)
    checked against the manifest seed and the record before the load."""
    import torch
    rec = score.check_berturk(ckpt, ident)
    temp = float(rec["temperature"])
    latency.check_temperature(temp, s)
    pd.check_record_temperature(temp, binding, s)
    tok, model, _ = score.load_bert(ckpt, ident, device, threads)
    max_len = rec["canonical"]["max_len"]

    def forward(text: str) -> np.ndarray:
        with torch.no_grad():
            enc = tok([text], truncation=True, max_length=max_len, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float().cpu().numpy()
        score.check_bert(model, device)
        return np.asarray(logits, dtype=np.float32)
    return {"temperature": temp, "forward": forward, "check": functools.partial(score.check_bert, model, device)}


def load_system(binding: dict[str, Any], bound: list[tuple[dict[str, Any], str, dict[str, Any]]], device: str,
                threads: int) -> dict[str, Any]:
    """{binding, device, seeds: [{temperature, forward, check}]} of one system: every bound seed loaded on `device`
    through score.py's loaders; SystemExit before any load when --device mps has no MPS. forward(text) is float32
    [1, 48]; check() is the seed's score.check_agent / check_bert state {device, dtype, amp_enabled}."""
    import torch
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps, but MPS is not available here; use --device cpu")
    load_seed = _laya_seed if binding["kind"] == "laya" else _berturk_seed
    return {"binding": binding, "device": device,
            "seeds": [load_seed(binding, s, ckpt, ident, device, threads) for s, ckpt, ident in bound]}


def classify(systems: dict[str, dict[str, Any]], text: str) -> dict[str, dict[str, Any]]:
    """{kind: pilot.demo.system_answer} of both systems for one message, in pd.KINDS order; any exception propagates,
    so a reply is both answers or none."""
    out = {}
    for kind in pd.KINDS:
        system = systems[kind]
        logits = [s["forward"](text) for s in system["seeds"]]
        out[kind] = pd.system_answer(system["binding"], logits, [s["temperature"] for s in system["seeds"]])
    return out


def new_state(device: str) -> dict[str, Any]:
    """The chat state before loading: {chat: loading, message: None, device, systems: None}."""
    return {"chat": pd.LOADING, "message": None, "device": device, "systems": None}


def load_job(state: dict[str, Any], bindings_fn: Callable[[], dict[str, dict[str, Any]]], root: str, device: str,
             threads: int) -> None:
    """On the worker thread: bind both systems (bindings_fn), bind every seed dir of both before any load, load them,
    run one warm-up classify of pd.WARMUP_TEXT (output discarded), then set the state ready. A SystemExit sets it
    refused with its one line; any other exception refused with LOAD_FAILED naming only the exception type."""
    try:
        bindings = bindings_fn()
        bound = {kind: bind_seeds(bindings[kind], root) for kind in pd.KINDS}
        systems = {kind: load_system(bindings[kind], bound[kind], device, threads) for kind in pd.KINDS}
        classify(systems, pd.WARMUP_TEXT)
        state["systems"] = systems
        state["chat"] = pd.READY
    except SystemExit as e:
        state["message"] = str(e.code).replace("\n", " ")
        state["chat"] = pd.REFUSED
    except Exception as e:
        state["message"] = LOAD_FAILED % type(e).__name__
        state["chat"] = pd.REFUSED


def start_loading(state: dict[str, Any], executor: ThreadPoolExecutor,
                  bindings_fn: Callable[[], dict[str, dict[str, Any]]], root: str, device: str,
                  threads: int) -> Future:
    """Submit load_job to the worker thread; returns its Future (load_job never raises)."""
    return executor.submit(load_job, state, bindings_fn, root, device, threads)


def create_app(state: dict[str, Any], executor: ThreadPoolExecutor, views: dict[str, dict[str, Any]],
               ui_dist: str | None = None) -> Any:
    """The FastAPI app: GET /api/status, POST /api/classify, GET /api/<view>, then the built UI at / (when ui_dist is
    given). Docs, OpenAPI and telemetry are off; only the Host 127.0.0.1 or localhost is served; no CORS. No handler
    prints or logs a body."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.staticfiles import StaticFiles

    app = FastAPI(telemetry=TELEMETRY, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    def error(status: int, message: str) -> Any:
        return JSONResponse({"error": message}, status_code=status)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"chat": state["chat"], "message": state["message"], "device": state["device"]}

    @app.post("/api/classify")
    async def classify_route(request: Request) -> Any:
        if state["chat"] != pd.READY:
            return error(503, state["message"] or pd.LOADING_TEXT)
        if not request.headers.get("content-type", "").lower().startswith("application/json"):
            return error(415, NOT_JSON_TYPE)
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > pd.MAX_BODY_BYTES:
                return error(413, BODY_TOO_LARGE)
        try:
            text = pd.parse_classify_body(bytes(raw))
        except SystemExit as e:
            return error(400, str(e.code))
        try:
            return await asyncio.wrap_future(executor.submit(classify, state["systems"], text))
        except SystemExit as e:
            return error(500, str(e.code).replace("\n", " "))
        except Exception as e:
            return error(500, CLASSIFY_FAILED % type(e).__name__)

    @app.get("/api/{name}")
    def view(name: str) -> Any:
        if name not in views:
            return error(404, "no such view; the views are %s" % ", ".join(sorted(views)))
        v = views[name]
        if pd.view_refused(v):
            return JSONResponse(v, status_code=503)
        return v

    if ui_dist is not None:
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
    return app


def main(argv: list[str] | None = None) -> None:
    """CLI entry: refuse a bad argument or a missing UI build, start loading the seeds on the worker thread, serve
    the app on 127.0.0.1 until interrupted."""
    ap = parser()
    a = ap.parse_args(argv)
    if not 1 <= a.port <= 65535:
        ap.error("--port must be between 1 and 65535")
    if a.threads < 1:
        ap.error("--threads must be >= 1")
    if not os.path.isfile(os.path.join(UI_DIST, "index.html")):
        raise SystemExit("ui/dist/index.html not found; build the UI first: `make demo` runs npm --prefix ui ci and "
                         "npm --prefix ui run build")
    import uvicorn
    state = new_state(a.device)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-model")
    views = pd.build_views(REPO_ROOT)
    app = create_app(state, executor, views, UI_DIST)
    start_loading(state, executor, functools.partial(pd.protocol_bindings, REPO_ROOT), a.checkpoint_root, a.device,
                  a.threads)
    print("demo on http://127.0.0.1:%d (local only; stores nothing); the chat opens when the six final checkpoints "
          "are loaded" % a.port, flush=True)
    uvicorn.run(app, host=pd.HOST, port=a.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
