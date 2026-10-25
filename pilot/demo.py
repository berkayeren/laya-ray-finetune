"""The record-binding and verdict-mirroring half of the local demo (demo.py): torch-free and fastapi-free.

demo.py is a local, single-user demo over the finished, frozen results: a chat in which fine-tuned Laya and fine-tuned
BERTurk each answer one message, and views of the committed records. This module holds what needs no model and no web
framework: the server constants, the binding of each system to its frozen records, the chat answer of one system for
one message, the request-body check and the record views.

Design notes:
- Every number the demo shows has exactly one producing function in the repo. The chat mirrors report.verdict_from's
  headline variant (the seed-averaged system the frozen verdict judged): per final seed pilot.metrics.probabilities of
  its logits at the seed's served T, then pilot.metrics.seed_average over the seeds, then the inclusive rule p >= θ at
  pilot.metrics.theta_vector of the postfit record's chosen.theta.shipped, then pilot.metrics.message_confidence
  (CONFIDENCE_RULE). system_answer calls those four functions and never reimplements the sigmoid, the mean or the
  >= rule.
- The binding chain, every link before any model loads: protocol.json (pilot.protocol.require_frozen) -> each system's
  postfit record and final manifest at the sha256 protocol.json binds -> the manifest bound to the record
  (postfit.sha256) and listing exactly the protocol's final seeds (protocol_bindings, bind_system). demo.py then binds
  every seed dir (latency.bind_seed: identity and dump run id) and, after each load, the served T to the manifest seed's
  T (latency.check_temperature) and to the record's chosen T within T_MATCH (check_record_temperature, the bound
  report.final_seed_binding uses).
- parse_classify_body's refusals are fixed one-line texts that never contain the submitted text, and the text it
  returns is unchanged (no strip, lowercasing, folding or normalization), as latency.py feeds texts.
- A view builder's refusal (SystemExit) becomes that view's {"error", "source"}; every other view keeps serving.

This module never imports torch, transformers, laya, Ray, fastapi, starlette or uvicorn (a subprocess test guards it).

usage: from pilot import demo
       bindings = demo.protocol_bindings()                   # {kind: binding}, refused in one line on any mismatch
       answer = demo.system_answer(bindings["laya"], logits_by_seed, served_temperatures)
"""
import json
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np

import pilot.latency
import pilot.postfit
import pilot.protocol
from intents import QUESTIONS
from pilot import metrics
from pilot.data import REPO_ROOT, repo_relpath

HOST = "127.0.0.1"                                          # loopback only: demo.py has no --host flag
PORT = 8765
MAX_CHARS = 500                                             # characters of one chat message
MAX_BODY_BYTES = 8192                                       # bytes of one /api/classify body, read from the stream
T_MATCH = 1e-12                                             # served T vs the record's chosen T (= report.T_MATCH)
KINDS = pilot.protocol.KINDS
LOADING, READY, REFUSED = "loading", "ready", "refused"
LOADING_TEXT = "the six final checkpoints are still loading; the chat opens when /api/status reads ready"
# A demo-authored warm-up string, run once after the load and discarded. "Merhaba" alone is a message of the dataset,
# so the warm-up text is a sentence written for the demo that no split holds.
WARMUP_TEXT = "Merhaba, bu bir ısınma mesajıdır."

# the fixed request-body refusals: none of them contains any part of the submitted body
BODY_NOT_JSON = 'the request body is not UTF-8 JSON; send {"text": "..."} as application/json'
BODY_NOT_OBJECT = 'the request body must be a JSON object with exactly the key "text"; send {"text": "..."}'
TEXT_NOT_STRING = '"text" must be a string; send {"text": "..."}'
TEXT_EMPTY = "the message is empty; type a message"
TEXT_TOO_LONG = "the message is longer than %d characters; shorten it" % MAX_CHARS


def _pin(path: str, have: str, want: str | None) -> None:
    """SystemExit unless `want` is None or the bytes of `path` hashed to it (`have`)."""
    if want is not None and have != want:
        rel = repo_relpath(path)
        raise SystemExit("%s has sha256 %s, protocol.json binds %s; restore it with `git checkout -- %s`"
                         % (rel, have, want, rel))


def _seed_key(s: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """(seed, checkpoint_name, dump_run_id, train_run_id) of a manifest or protocol final seed."""
    return s.get("seed"), s.get("checkpoint_name"), s.get("dump_run_id"), s.get("train_run_id")


def bind_system(kind: str, record_path: str, manifest_path: str, record_sha256: str | None = None,
                manifest_sha256: str | None = None, final_seeds: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The binding of one system to its postfit record and final manifest, from one read of each; SystemExit naming
    the fix unless the record is a postfit/1 record of `kind` with a float chosen T, each file hashes to the given pin
    (when given), the manifest is a postfit-final/1 manifest of `kind` bound to the record's sha256 and, when
    final_seeds is given, lists exactly those seeds in order (seed, checkpoint_name, dump_run_id, train_run_id).

    Returns {kind, record_path, record_sha256, manifest_path, manifest_sha256 (repo-relative paths), seeds (the
    manifest seeds), temperature (the record's chosen T), theta (float64 [48], metrics.theta_vector of
    chosen.theta.shipped), theta_global, own_theta ([48] bool: the tag has its own θ)}. Hashes and parses only."""
    record, rec_sha = pilot.postfit.read_record(record_path)
    if record.get("system_kind") != kind:
        raise SystemExit("%s is a postfit record of %s, not %s; pass the record of %s"
                         % (repo_relpath(record_path), record.get("system_kind"), kind, kind))
    chosen = record["chosen"]
    if isinstance(chosen.get("temperature"), bool) or not isinstance(chosen.get("temperature"), float):
        raise SystemExit("%s has no float chosen temperature; restore it with `git checkout -- %s`"
                         % (repo_relpath(record_path), repo_relpath(record_path)))
    shipped = (chosen.get("theta") or {}).get("shipped")
    if not isinstance(shipped, dict) or not isinstance(shipped.get("per_tag"), dict) or "global" not in shipped:
        raise SystemExit("%s has no chosen theta.shipped {global, per_tag}; restore it with `git checkout -- %s`"
                         % (repo_relpath(record_path), repo_relpath(record_path)))
    _pin(record_path, rec_sha, record_sha256)
    manifest, msha = pilot.latency.read_final_manifest(manifest_path, kind)
    _pin(manifest_path, msha, manifest_sha256)
    bound_to = (manifest.get("postfit") or {}).get("sha256")
    if bound_to != rec_sha:
        raise SystemExit("%s is not bound to the postfit record %s (its postfit.sha256 is %s, the record's %s); "
                         "restore both with `git checkout -- %s %s`"
                         % (repo_relpath(manifest_path), repo_relpath(record_path), bound_to, rec_sha,
                            repo_relpath(manifest_path), repo_relpath(record_path)))
    if final_seeds is not None:
        have = [_seed_key(s) for s in manifest["seeds"]]
        want = [_seed_key(s) for s in final_seeds]
        if have != want:
            raise SystemExit("%s lists the seeds %s, protocol.json's final seeds of %s are %s; restore it with "
                             "`git checkout -- %s`" % (repo_relpath(manifest_path), have, kind, want,
                                                       repo_relpath(manifest_path)))
    return {"kind": kind, "record_path": repo_relpath(record_path), "record_sha256": rec_sha,
            "manifest_path": repo_relpath(manifest_path), "manifest_sha256": msha, "seeds": manifest["seeds"],
            "temperature": float(chosen["temperature"]), "theta": metrics.theta_vector(shipped),
            "theta_global": float(shipped["global"]), "own_theta": [t in shipped["per_tag"] for t in metrics.TAGS]}


def protocol_bindings(root: str = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """{kind: bind_system(...)} for both systems of the frozen protocol: require_frozen first, then each system's
    postfit record and final manifest (paths under `root`) at the sha256 protocol.json binds, and its final seeds."""
    proto, _ = pilot.protocol.require_frozen()
    out = {}
    for kind in KINDS:
        system = proto["systems"][kind]
        rec, man = system["postfit"], system["final_manifest"]
        out[kind] = bind_system(kind, os.path.join(root, rec["path"]), os.path.join(root, man["path"]),
                                rec["sha256"], man["sha256"], pilot.protocol.final_seeds(proto, kind))
    return out


def check_record_temperature(served: float, binding: dict[str, Any], s: dict[str, Any]) -> None:
    """SystemExit when a seed's served T differs from the bound record's chosen T by more than T_MATCH."""
    if abs(float(served) - binding["temperature"]) > T_MATCH:
        raise SystemExit("%s serves T %r, the postfit record %s chose T %r; the demo serves only the record's T: pass "
                         "--checkpoint-root DIR holding the final seeds the manifest names"
                         % (s["checkpoint_name"], served, binding["record_path"], binding["temperature"]))


def system_answer(binding: dict[str, Any], logits_by_seed: list[Any], temps: list[float]) -> dict[str, Any]:
    """The seed-averaged answer of one system for one message: P = metrics.seed_average of metrics.probabilities of
    each seed's logits [1, 48] at its served T, θ = the binding's theta (theta_vector of theta.shipped), (c, pred) =
    metrics.message_confidence(P, θ), so predicted means p >= θ; SystemExit unless there is one logits array and one
    T per bound seed.

    Returns {kind, seeds, temperature (the record's T), theta_global, n_own_theta, record {path, sha256},
    probabilities [48], theta [48], own_theta [48], ranked (the 48 tags by probability, ties in INTENTS order),
    predicted (the ranked tags with p >= θ; an empty list is No intent), log_confidence (c), confidence (exp c)}, every
    value a plain Python float, bool, int, str or list."""
    n = len(binding["seeds"])
    if len(logits_by_seed) != n or len(temps) != n:
        raise SystemExit("%s: %d logits and %d temperatures for %d bound seeds; restart `make demo`"
                         % (binding["kind"], len(logits_by_seed), len(temps), n))
    P = metrics.seed_average([metrics.probabilities(lg, T) for lg, T in zip(logits_by_seed, temps)])
    th = binding["theta"]
    c, pred = metrics.message_confidence(P, th)
    order = np.argsort(-P[0], kind="stable")                # stable: equal probabilities keep INTENTS order
    ranked = [metrics.TAGS[i] for i in order]
    predicted = [metrics.TAGS[i] for i in order if pred[0, i]]
    return {"kind": binding["kind"], "seeds": [int(s["seed"]) for s in binding["seeds"]],
            "temperature": float(binding["temperature"]), "theta_global": float(binding["theta_global"]),
            "n_own_theta": int(sum(binding["own_theta"])),
            "record": {"path": binding["record_path"], "sha256": binding["record_sha256"]},
            "probabilities": [float(p) for p in P[0]], "theta": [float(t) for t in th],
            "own_theta": [bool(o) for o in binding["own_theta"]], "ranked": ranked, "predicted": predicted,
            "log_confidence": float(c[0]), "confidence": float(math.exp(c[0]))}


def parse_classify_body(raw: bytes) -> str:
    """The message text of an /api/classify body: UTF-8 JSON of an object with exactly the key "text", a string that
    is not empty or only whitespace and at most MAX_CHARS characters. SystemExit with one of the fixed one-line texts
    otherwise (never any part of the body). Returns the text unchanged: no strip, lowercasing, folding or
    normalization."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise SystemExit(BODY_NOT_JSON) from None
    if not isinstance(body, dict) or set(body) != {"text"}:
        raise SystemExit(BODY_NOT_OBJECT)
    text = body["text"]
    if not isinstance(text, str):
        raise SystemExit(TEXT_NOT_STRING)
    if not text.strip():
        raise SystemExit(TEXT_EMPTY)
    if len(text) > MAX_CHARS:
        raise SystemExit(TEXT_TOO_LONG)
    return text


def intents_view(root: str) -> dict[str, Any]:
    """{tags: [{tag, question}] in INTENTS order, the question as stored in intents.QUESTIONS (the exact English model
    input), confidence_rule: metrics.CONFIDENCE_RULE}. `root` is unused (every view builder takes the repo root)."""
    return {"tags": [{"tag": t, "question": QUESTIONS[t]["instructions"]} for t in metrics.TAGS],
            "confidence_rule": metrics.CONFIDENCE_RULE}


VIEWS: dict[str, Callable[[str], dict[str, Any]]] = {"intents": intents_view}
VIEW_SOURCES = {"intents": "intents.py"}


def build_views(root: str = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """{name: view} of every VIEWS builder, each run once on `root`; a builder's SystemExit becomes that view's
    {"error": its one line, "source": VIEW_SOURCES[name]}, so one refused record breaks only its own view."""
    out = {}
    for name, build in VIEWS.items():
        try:
            out[name] = build(root)
        except SystemExit as e:
            out[name] = {"error": str(e.code).replace("\n", " "), "source": VIEW_SOURCES[name]}
    return out


def view_refused(view: dict[str, Any]) -> bool:
    """True for the {"error", "source"} placeholder build_views stores for a refused view."""
    return set(view) == {"error", "source"}
