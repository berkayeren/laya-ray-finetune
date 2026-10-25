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
- The verdict view mirrors report.py's --go-block, --final-table and --recommendation: the verdict record's bytes
  read once, pilot.protocol.require_frozen, then pilot.verdict.check_record, then the three frozen renders. Every
  verdict string it carries is a whole line (or a table cell) of those renders or a field of the record, and every
  charted delta is the record's own number; the demo writes no sentence. Each status carries a tone computed here from
  pilot.verdict's status constants, so the UI never needs a verdict word.
- The selection view reads only protocol-bound inputs: each postfit record, final manifest and the BERTurk reference
  check is hashed against protocol.json's inputs entry before it is parsed (bound_input); the superseded BERTurk
  history comes from pilot.berturk.load_superseded (its own pinned sha256s). Its renders are postfit.py --render and
  --render-berturk's.

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

import pilot.berturk
import pilot.latency
import pilot.memo
import pilot.postfit
import pilot.protocol
import pilot.verdict
from intents import QUESTIONS
from pilot import metrics
from pilot.data import REPO_ROOT, repo_relpath, sha256_bytes

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


# the tone of a status, computed here so the UI source never holds a verdict word; every other status is neutral
STATUS_TONE = {pilot.verdict.PASS: "positive", pilot.verdict.FAIL: "negative", pilot.verdict.INCONCLUSIVE: "caution"}
VERDICT_RECORD = "results/final/verdict.json"
# the four charted deltas, in order: (criterion label, metric label, criterion key, metric key, direction); (a)'s
# entry is criteria.a[metric key], (d)'s the criteria.d.paths item whose metric is the metric key
DELTA_ROWS = (("(a)", "micro-F1", "a", "micro", "higher is better"),
              ("(a)", "macro-F1", "a", "macro", "higher is better"),
              ("(d)", "Brier", "d", "brier", "lower is better"),
              ("(d)", "precision at 80% coverage", "d", "precision_at_80", "higher is better"))


def tone(status: Any) -> str:
    """The display tone of a status: STATUS_TONE's (PASS positive, FAIL negative, INCONCLUSIVE caution) or neutral."""
    return STATUS_TONE.get(status, "neutral") if isinstance(status, str) else "neutral"


def render_line(text: str, prefix: str, what: str) -> str:
    """The one line of `text` starting with `prefix`; SystemExit naming `what` and the prefix when there are zero or
    several such lines (the demo shows whole render lines only)."""
    hits = [l for l in text.split("\n") if l.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("the %s render holds %d lines starting %r, not one; restore its records with `git checkout -- "
                         "results`" % (what, len(hits), prefix))
    return hits[0]


def md_table(text: str, header_prefix: str, what: str) -> dict[str, list[Any]]:
    """{header: [cells], rows: [[cells]]} of the one markdown table of `text` whose header line starts with
    header_prefix: the |---| separator skipped, then every following line starting "|" up to the first other line.
    Cells are the line stripped of its outer "|" and split on " | ", each stripped. SystemExit naming `what` when there
    is no such header, several, or no separator."""
    lines = text.split("\n")
    at = [i for i, l in enumerate(lines) if l.startswith(header_prefix)]
    if len(at) != 1:
        raise SystemExit("the %s render holds %d tables headed %r, not one; restore its records with `git checkout -- "
                         "results`" % (what, len(at), header_prefix))
    i = at[0]

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split(" | ")]

    if i + 1 >= len(lines) or not lines[i + 1].startswith("|---"):
        raise SystemExit("the %s render's table headed %r has no |---| separator; restore its records with "
                         "`git checkout -- results`" % (what, header_prefix))
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append(cells(l))
    return {"header": cells(lines[i]), "rows": rows}


def _delta_entry(v: dict[str, Any], crit: str, key: str) -> dict[str, Any]:
    """The verdict record's {delta, ci} (plus status for (d)) of one DELTA_ROWS entry; SystemExit when it is absent."""
    c = v["criteria"][crit]
    if crit == "a":
        e = c.get(key)
    else:
        hits = [p for p in c.get("paths") or [] if p.get("metric") == key]
        e = hits[0] if len(hits) == 1 else None
    if not isinstance(e, dict) or "delta" not in e or not isinstance(e.get("ci"), list) or len(e["ci"]) != 2:
        raise SystemExit("the verdict record has no criterion (%s) %s delta and CI; restore it with `git checkout -- "
                         "%s`" % (crit, key, VERDICT_RECORD))
    return e


def verdict_view(root: str = REPO_ROOT, path: str | None = None) -> dict[str, Any]:
    """The frozen verdict as data, mirroring report.py's --go-block, --final-table and --recommendation: the record's
    bytes read once and hashed, parsed, pilot.protocol.require_frozen, pilot.verdict.check_record, then
    pilot.verdict.render_go, render_final_table and pilot.memo.render_recommendation (equal to those three commands'
    stdout). SystemExit on a missing or invalid record or one check_record refuses.

    Returns {sources, verdict, means, tone, lines {verdict, rule, means, recommendation} (whole render lines without
    their markdown markup), criteria [{criterion, status, tone, numbers}] (the GO table's cells), deltas [{criterion,
    metric, delta_points, ci_points [lo, hi], delta_text, ci_text, direction, status}] (DELTA_ROWS: the record's
    numbers x 100, unrounded; the texts in the GO render's formats), results and settings ({header, rows} of the
    results-on-test tables), caveats (pilot.memo.CAVEATS), renders {go, final_table, recommendation}}."""
    path = os.path.join(root, VERDICT_RECORD) if path is None else path
    rel = repo_relpath(path)
    if not os.path.isfile(path):
        raise SystemExit("verdict record %s not found; restore it with `git checkout -- %s`" % (rel, VERDICT_RECORD))
    with open(path, "rb") as f:
        raw = f.read()                                      # hash and parse the same bytes (report.py's sequence)
    sha = sha256_bytes(raw)
    try:
        v = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SystemExit("verdict record %s is not valid JSON (%s); restore it with `git checkout -- %s`"
                         % (rel, e, VERDICT_RECORD)) from None
    proto, pin = pilot.protocol.require_frozen()
    pilot.verdict.check_record(v, proto, pin)
    go = pilot.verdict.render_go(v)
    table = pilot.verdict.render_final_table(v)
    rec = pilot.memo.render_recommendation(v, rel, sha, pilot.memo.facts(v, proto, root))
    word = v["verdict"]["verdict"]
    go_lines = go.split("\n")
    rule = render_line(go, "- Rule ", "GO block")
    at = go_lines.index(rule) + 1
    means = go_lines[at] if at < len(go_lines) else ""
    if not means.startswith("- %s " % word):
        raise SystemExit("the GO block's line after %r is %r, not the meaning of %s; restore %s with `git checkout -- "
                         "%s`" % (rule, means, word, rel, VERDICT_RECORD))
    lines = {"verdict": render_line(go, "**Verdict: ", "GO block").strip("*"), "rule": rule[2:], "means": means[2:],
             "recommendation": render_line(rec, "**Recommendation: ", "recommendation").strip("*")}
    criteria = [{"criterion": r[0], "status": r[1], "tone": tone(r[1]), "numbers": r[2]}
                for r in md_table(go, pilot.memo.GO_TABLE_HEADER, "GO block")["rows"]]
    deltas = []
    for label, metric, crit, key, direction in DELTA_ROWS:
        e = _delta_entry(v, crit, key)
        d, (lo, hi) = 100 * e["delta"], (100 * e["ci"][0], 100 * e["ci"][1])
        deltas.append({"criterion": label, "metric": metric, "delta_points": d, "ci_points": [lo, hi],
                       "delta_text": "%+.2f points" % d, "ci_text": "CI [%+.2f, %+.2f]" % (lo, hi),
                       "direction": direction, "status": e.get("status") if crit == "d" else None})
    return {"sources": [rel, "protocol.json"], "verdict": word, "means": v["verdict"]["means"], "tone": tone(word),
            "lines": lines, "criteria": criteria, "deltas": deltas,
            "results": md_table(table, pilot.memo.RESULTS_TABLE_HEADER, "results-on-test"),
            "settings": md_table(table, "| system | T |", "results-on-test settings"),
            "caveats": list(pilot.memo.CAVEATS),
            "renders": {"go": go, "final_table": table, "recommendation": rec}}


VIEWS: dict[str, Callable[[str], dict[str, Any]]] = {"intents": intents_view, "verdict": verdict_view}
VIEW_SOURCES = {"intents": "intents.py", "verdict": VERDICT_RECORD}


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
