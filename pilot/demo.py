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
- A view builder's refusal (SystemExit), or any other exception (named by its type only), becomes that view's
  {"error", "source"}; every other view keeps serving.
- The verdict view mirrors report.py's --go-block, --final-table and --recommendation: the verdict record's bytes
  read once, pilot.protocol.require_frozen, then pilot.verdict.check_record, then the three frozen renders. Every
  verdict string it carries is a whole line (or a table cell) of those renders or a field of the record, and every
  charted delta is the record's own number; the demo writes no sentence. Each status carries a tone computed here from
  pilot.verdict's status constants, so the UI never needs a verdict word.
- The selection view reads only protocol-bound inputs: each postfit record, final manifest and the BERTurk reference
  check is hashed against protocol.json's inputs entry before it is parsed (bound_input); the superseded BERTurk
  history comes from pilot.berturk.load_superseded (its own pinned sha256s). Its renders are postfit.py --render and
  --render-berturk's.
- The training view (training_system) holds each final seed's per-epoch rows as its checkpoint recorded them: Ray
  Train's rows (pilot.runs.EPOCH_FIELDS) from the rl_agent_config.json bytes whose sha256 is the bound identity's
  config_sha256 (laya_epochs), the HF Trainer's rows (pilot.berturk.EPOCH_FIELDS) through pilot.berturk.read_record
  (berturk_epochs). demo.py reads them only after latency.bind_seed bound the seed dir, before any model loads. The
  rows must equal the committed final manifest's `epochs` on SHARED_EPOCH_KEYS; the manifest's rows serve only that
  check. Without the checkpoints a system's training view is unavailable with the one-line refusal
  (training_unavailable), never filled from the manifests (the user decision of 2026-10-25, D-09).
- The confidence, latency and few-shot group (D-10): the reliability diagrams are the committed SVG files, served by
  exact name from the startup allow-list (svg_allow_list) and never redrawn, each with the run id of the confidence/1
  record beside it and the val records' caveat verbatim (confidence_view); the latency/1 reports are protocol.json
  inputs, hashed against their pins before load_reports (latency_view), and their per-message times stay on the
  server; the few-shot results record is the one the checked verdict binds by sha256 (fewshot_view), every mean
  shown a cell of pilot.fewshot_report.render_markdown's text. No view builder writes a file.

This module never imports torch, transformers, laya, Ray, fastapi, starlette or uvicorn (a subprocess test guards it).

usage: from pilot import demo
       bindings = demo.protocol_bindings()                   # {kind: binding}, refused in one line on any mismatch
       answer = demo.system_answer(bindings["laya"], logits_by_seed, served_temperatures)
       views = demo.build_views()                            # {name: view or {"error", "source"}}: intents, verdict,
                                                             # selection, confidence, latency, fewshot (GET /api/<name>)
       svgs = demo.svg_allow_list()                          # {file name: repo-relative path} for GET /api/svg/<name>
       training = demo.training_system(binding, bound)        # one system's per-epoch rows of its bound seeds
"""
import json
import math
import os
import re
from collections.abc import Callable
from typing import Any

import numpy as np

import pilot.berturk
import pilot.fewshot_report
import pilot.latency
import pilot.memo
import pilot.postfit
import pilot.protocol
import pilot.runs
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


def checked_verdict(root: str = REPO_ROOT, path: str | None = None) -> tuple[dict[str, Any], str, str, dict[str, Any],
                                                                             str]:
    """(verdict, repo-relative path, sha256 of its bytes, protocol, protocol pin) of the verdict record: its bytes read
    once and hashed, parsed, pilot.protocol.require_frozen, then pilot.verdict.check_record on the record parsed
    (report.py's sequence). SystemExit on a missing or invalid record or one check_record refuses."""
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
    return v, rel, sha, proto, pin


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
    v, rel, sha, proto, _ = checked_verdict(root, path)
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


# each system's postfit record and final manifest, and the BERTurk reference check: protocol.json inputs, every one
SELECTION_RECORDS = {"laya": ("results/postfit/laya.json", "results/postfit/laya-final.json"),
                     "berturk": ("results/postfit/berturk.json", "results/postfit/berturk-final.json")}
BERTURK_REFERENCE = "results/postfit/berturk-reference.json"
# the Laya val selection render's edge lines, in render order (BERTurk's block is served whole, without edge lines)
EDGE_PREFIXES = ("- Grid edge:", "- Bounds:", "- θ grid edge:", "- Per-tag θ grid edge:")


def bound_input(root: str, rel: str, inputs: dict[str, str]) -> bytes:
    """The bytes of root/rel, read once; SystemExit naming the fix unless protocol.json's inputs list rel and the bytes
    hash to its sha256 there (the file is then the one the frozen protocol was built on)."""
    if rel not in inputs:
        raise SystemExit("%s is not a protocol.json input; the demo reads protocol-bound records only "
                         "(restore protocol.json with `git checkout -- protocol.json`)" % rel)
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        raise SystemExit("protocol input %s not found; restore it with `git checkout -- %s`" % (rel, rel))
    with open(path, "rb") as f:
        raw = f.read()
    have = sha256_bytes(raw)
    if have != inputs[rel]:
        raise SystemExit("protocol input %s changed: sha256 %s, the frozen protocol has %s; restore it with "
                         "`git checkout -- %s`" % (rel, have, inputs[rel], rel))
    return raw


def _bound_json(raw: bytes, rel: str) -> dict[str, Any]:
    """The JSON object of a bound input's bytes; SystemExit naming the restore otherwise."""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SystemExit("%s is not valid JSON (%s); restore it with `git checkout -- %s`" % (rel, e, rel)) from None
    if not isinstance(obj, dict):
        raise SystemExit("%s is not a JSON object; restore it with `git checkout -- %s`" % (rel, rel))
    return obj


def _bound_record(root: str, rel: str, inputs: dict[str, str], kind: str) -> tuple[dict[str, Any], str]:
    """(record, sha256) of a protocol-bound postfit record of `kind`, parsed by pilot.postfit.read_record; SystemExit
    unless its bytes are protocol.json's and it is a record of `kind`."""
    want = sha256_bytes(bound_input(root, rel, inputs))
    record, sha = pilot.postfit.read_record(os.path.join(root, rel))
    if sha != want or record.get("system_kind") != kind:
        raise SystemExit("%s changed while it was read, or is a record of %s, not %s; restore it with `git checkout "
                         "-- %s`" % (rel, record.get("system_kind"), kind, rel))
    return record, sha


def _bound_to(obj: dict[str, Any], rel: str, record_rel: str, record_sha: str) -> dict[str, Any]:
    """obj, refused unless its postfit.sha256 is the record's (as postfit.py --render and --render-berturk check)."""
    got = (obj.get("postfit") or {}).get("sha256")
    if got != record_sha:
        raise SystemExit("%s is bound to postfit record %s, not %s (%s); restore both with `git checkout -- %s %s`"
                         % (rel, got, record_sha, record_rel, rel, record_rel))
    return obj


def _candidate(c: dict[str, Any]) -> dict[str, Any]:
    """{epoch, fraction [2TP, 2TP + FP + FN] (unreduced, as the render prints it), score (pilot.postfit.score),
    val_micro_f1_at_global_theta, temperature, theta_global} of one postfit candidate."""
    o = c["objective"]
    return {"epoch": c["epoch"], "fraction": [2 * o["tp"], 2 * o["tp"] + o["fp"] + o["fn"]],
            "score": float(pilot.postfit.score(c)), "val_micro_f1_at_global_theta": c["val_micro_f1_at_global_theta"],
            "temperature": c["temperature"], "theta_global": c["theta"]["shipped"]["global"]}


def _chosen(record: dict[str, Any]) -> dict[str, Any]:
    """The record's chosen settings: epoch, schedule, T, θ shipped and raw (global, own-θ count), encoder and head LR."""
    ch = record["chosen"]
    th = ch["theta"]
    return {"epoch": ch["epoch"], "epochs_schedule": ch["epochs_schedule"], "temperature": ch["temperature"],
            "theta_shipped_global": th["shipped"]["global"], "n_own_theta": len(th["shipped"]["per_tag"]),
            "theta_raw_global": th["raw"]["global"], "n_own_theta_raw": len(th["raw"]["per_tag"]),
            "encoder_lr": ch["encoder_lr"], "head_lr": (ch.get("canonical") or {}).get("head_lr")}


def selection_view(root: str = REPO_ROOT) -> dict[str, Any]:
    """Both models' val selection as data, from protocol-bound inputs only: require_frozen, then every postfit record,
    final manifest and the BERTurk reference check hashed against protocol.json's inputs entry before it is parsed
    (bound_input), each manifest and the reference bound to its record; the superseded BERTurk runs through
    pilot.berturk.load_superseded (its pinned sha256s). The renders are postfit.py --render's
    (pilot.postfit.render_markdown) and --render-berturk's (pilot.berturk.render_markdown). SystemExit on any
    mismatch.

    Returns {sources, systems {laya, berturk: {kind, record {path, sha256}, candidates [{epoch, fraction, score,
    val_micro_f1_at_global_theta, temperature, theta_global}], chosen {epoch, epochs_schedule, temperature,
    theta_shipped_global, n_own_theta, theta_raw_global, n_own_theta_raw, encoder_lr, head_lr}, edge_lines (Laya's
    EDGE_PREFIXES lines without "- ", BERTurk []), render}}, reference {record {path, sha256}, fraction, value,
    threshold, status, tone, reference, threshold_rule}, superseded [{name, label, fraction, value, change,
    user_choice, candidates [{epoch, score}]}]}."""
    proto, _ = pilot.protocol.require_frozen()
    inputs = proto.get("inputs") or {}
    records, finals = {}, {}
    for kind, (rel, final_rel) in SELECTION_RECORDS.items():
        records[kind] = _bound_record(root, rel, inputs, kind)
        finals[kind] = _bound_to(_bound_json(bound_input(root, final_rel, inputs), final_rel), final_rel, rel,
                                 records[kind][1])
    raw_ref = bound_input(root, BERTURK_REFERENCE, inputs)
    reference = _bound_to(_bound_json(raw_ref, BERTURK_REFERENCE), BERTURK_REFERENCE, SELECTION_RECORDS["berturk"][0],
                          records["berturk"][1])
    superseded = pilot.berturk.load_superseded(root)
    renders = {"laya": pilot.postfit.render_markdown(records["laya"][0], finals["laya"]),
               "berturk": pilot.berturk.render_markdown(records["berturk"][0], reference, finals["berturk"],
                                                        superseded)}
    systems = {}
    for kind, (record, sha) in records.items():
        edges = [render_line(renders[kind], p, "Laya val selection")[2:] for p in EDGE_PREFIXES] \
            if kind == "laya" else []
        systems[kind] = {"kind": kind, "record": {"path": SELECTION_RECORDS[kind][0], "sha256": sha},
                         "candidates": [_candidate(c) for c in record["candidates"]], "chosen": _chosen(record),
                         "edge_lines": edges, "render": renders[kind]}
    value = reference.get("value") or {}
    ref = {"record": {"path": BERTURK_REFERENCE, "sha256": sha256_bytes(raw_ref)}, "fraction": value.get("fraction"),
           "value": value.get("val_micro_f1_at_global_theta"), "threshold": reference.get("threshold"),
           "status": reference.get("status"), "tone": tone(reference.get("status")),
           "reference": reference.get("reference"), "threshold_rule": reference.get("threshold_rule")}
    sup = []
    for s in superseded:
        e, sv = s["entry"], (s["reference"].get("value") or {})
        sup.append({"name": e["name"], "label": "superseded: %s" % e["name"], "fraction": sv.get("fraction"),
                    "value": sv.get("val_micro_f1_at_global_theta"), "change": e["change"],
                    "user_choice": dict(e["user_choice"]),
                    "candidates": [{"epoch": c["epoch"], "score": float(pilot.postfit.score(c))}
                                   for c in s["record"]["candidates"]]})
    sources = [p for pair in SELECTION_RECORDS.values() for p in pair] + [BERTURK_REFERENCE, "protocol.json"] + \
        [s["entry"][k] for s in superseded for k in ("record", "reference")]
    return {"sources": sources, "systems": systems, "reference": ref, "superseded": sup}


# ---- the training runs: per-epoch rows of every final seed, read from its bound checkpoint (D-09, D-10) ----

# the epoch keys the committed final manifests hold (postfit.py --final-manifest), compared with every checkpoint's rows
SHARED_EPOCH_KEYS = ("epoch", "loss", "train_s", "peak_total_bytes", "swap_delta_bytes")
TRAINING_FIELDS = {"laya": pilot.runs.EPOCH_FIELDS, "berturk": pilot.berturk.EPOCH_FIELDS}


def _where(ckpt: str, name: str | None = None) -> str:
    """--checkpoint-root/<dir>[/<file>] of a seed dir: the form latency.bind_seed names it in, never a local path."""
    where = "--checkpoint-root/" + os.path.basename(os.path.normpath(ckpt))
    return where if name is None else where + "/" + name


def _epoch_rows(epochs: Any, fields: tuple[str, ...], where: str) -> list[dict[str, Any]]:
    """Each epoch row of a run record restricted to `fields`, in that order; SystemExit when the record has no list
    of epoch rows or a row lacks one of the fields."""
    if not isinstance(epochs, list):
        raise SystemExit("%s: the run record has no list of epoch rows; restore the final seed" % where)
    rows = []
    for row in epochs:
        missing = [k for k in fields if not isinstance(row, dict) or k not in row]
        if missing:
            raise SystemExit("%s: run record epoch %s lacks %s; restore the final seed"
                             % (where, row.get("epoch") if isinstance(row, dict) else None, ", ".join(missing)))
        rows.append({k: row[k] for k in fields})
    return rows


def laya_epochs(ckpt: str, ident: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(record, rows) of a bound Laya final seed: rl_agent_config.json read once as bytes, parsed only when their
    sha256 is ident's config_sha256 (score.training_block's sequence), its training.ray_train.record, and every epoch
    row restricted to pilot.runs.EPOCH_FIELDS in that order. SystemExit on changed bytes, invalid JSON, no record or a
    row lacking a field."""
    where = _where(ckpt, "rl_agent_config.json")
    try:
        with open(os.path.join(ckpt, "rl_agent_config.json"), "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit("%s cannot be read (%s); restore the final seed" % (where, e.strerror or type(e).__name__))
    if sha256_bytes(raw) != ident.get("config_sha256"):
        raise SystemExit("%s changed after it was hashed; restore the final seed" % where)
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SystemExit("%s is not valid JSON; restore the final seed" % where) from None
    tr = cfg.get("training") if isinstance(cfg, dict) else None
    rt = tr.get("ray_train") if isinstance(tr, dict) else None
    record = rt.get("record") if isinstance(rt, dict) else None
    if not isinstance(record, dict) or not record:
        raise SystemExit("%s holds no training.ray_train.record; restore the final seed" % where)
    return record, _epoch_rows(record.get("epochs"), pilot.runs.EPOCH_FIELDS, where)


def berturk_epochs(ckpt: str, ident: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(record, rows) of a bound BERTurk final seed: berturk_train.json through pilot.berturk.read_record (bytes whose
    sha256 is ident's record_sha256), its run record ["record"], and every epoch row restricted to
    pilot.berturk.EPOCH_FIELDS in that order. SystemExit, naming the seed dir, on any refusal."""
    where = _where(ckpt, pilot.berturk.RECORD)
    try:
        rec = pilot.berturk.read_record(ckpt, ident)
    except SystemExit as e:
        raise SystemExit("%s: %s" % (_where(ckpt), str(e.code).replace("\n", " "))) from None
    except OSError as e:
        raise SystemExit("%s cannot be read (%s); restore the final seed" % (where, e.strerror or type(e).__name__))
    record = rec.get("record")
    if not isinstance(record, dict) or not record:
        raise SystemExit("%s holds no run record; restore the final seed" % where)
    return record, _epoch_rows(record.get("epochs"), pilot.berturk.EPOCH_FIELDS, where)


def _shared(rows: Any) -> list[dict[str, Any]] | None:
    """rows restricted to SHARED_EPOCH_KEYS (a missing key reads None); None when rows is not a list of objects."""
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return None
    return [{k: r.get(k) for k in SHARED_EPOCH_KEYS} for r in rows]


def _differs_on(have: list[dict[str, Any]], want: list[dict[str, Any]] | None) -> str:
    """Where a seed's rows differ from its manifest's on SHARED_EPOCH_KEYS: the row count, or the first differing
    row's epoch and keys; "" when they are equal."""
    if want is None:
        return "the epochs entry (the manifest has no list of epoch rows)"
    if len(have) != len(want):
        return "the number of rows (%d, the manifest %d)" % (len(have), len(want))
    for h, w in zip(have, want):
        keys = [k for k in SHARED_EPOCH_KEYS if h[k] != w[k] or type(h[k]) is not type(w[k])]
        if keys:
            return "epoch %s: %s" % (h["epoch"], ", ".join(keys))
    return ""


def training_system(binding: dict[str, Any],
                    bound: list[tuple[dict[str, Any], str, dict[str, Any]]]) -> dict[str, Any]:
    """The training view of one system from its bound seeds [(manifest seed, checkpoint dir, identity)]
    (demo.bind_seeds, after latency.bind_seed): each seed's rows read from its checkpoint (laya_epochs or
    berturk_epochs), refused unless they equal the manifest seed's `epochs` on SHARED_EPOCH_KEYS. The manifest's rows serve only that check and are
    never returned: rows come only from the checkpoints (the user decision of 2026-10-25, D-09).

    Returns {status: "ok", fields (the kind's EPOCH_FIELDS), total_train_s (the sum of train_s over every row of every
    seed), seeds [{seed, checkpoint_name, train_run_id, device, n_train_messages, rows [{field: value}]}]}; SystemExit
    with one line on any refusal."""
    kind = binding["kind"]
    if kind not in TRAINING_FIELDS:
        raise SystemExit("no training rows for system %r; the systems are %s" % (kind, ", ".join(TRAINING_FIELDS)))
    read = laya_epochs if kind == "laya" else berturk_epochs
    seeds, total = [], 0.0
    for s, ckpt, ident in bound:
        record, rows = read(ckpt, ident)
        on = _differs_on(_shared(rows) or [], _shared(s.get("epochs")))
        if on:
            raise SystemExit("%s: the run record's epoch rows differ from the final manifest's on %s; restore the "
                             "final seed" % (_where(ckpt), on))
        for r in rows:
            t = r["train_s"]
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
                raise SystemExit("%s: run record epoch %s has train_s %r, not a finite number; restore the final seed"
                                 % (_where(ckpt), r["epoch"], t))
            total += float(t)
        n = record.get("n_train_messages") if kind == "laya" else record.get("n_messages")
        seeds.append({"seed": s["seed"], "checkpoint_name": s["checkpoint_name"], "train_run_id": s["train_run_id"],
                      "device": record.get("device"), "n_train_messages": n, "rows": rows})
    return {"status": "ok", "fields": list(TRAINING_FIELDS[kind]), "total_train_s": total, "seeds": seeds}


def training_unavailable(line: str) -> dict[str, Any]:
    """The training view of a system whose seeds or rows were refused: {status: "unavailable", message: the one line}.
    It never carries rows from anywhere else (no fallback to the final manifests' epochs, D-09)."""
    return {"status": "unavailable", "message": str(line).replace("\n", " ")}


# ---- the confidence group: the committed reliability diagrams, served byte for byte (D-10) ----

# (directory, split) of the committed reliability SVGs: the val records' and the final seeds' test records' (a
# hotels_probe record has no diagram); the demo serves exactly these files and redraws none of them
SVG_DIRS = (("results/confidence", "val"), ("results/final/confidence", "test"))
SVG_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+-(val|test)\.svg$")
CONFIDENCE_SCHEMA = "confidence/1"                          # = report.CONFIDENCE_SCHEMA


def svg_allow_list(root: str = REPO_ROOT) -> dict[str, str]:
    """{file name: repo-relative path} of every committed reliability SVG: the *-val.svg names under
    results/confidence, then the *-test.svg names under results/final/confidence, each in sorted os.listdir order and
    matching SVG_NAME_RE with its directory's split; a symlink or a non-file is never listed. Built once at startup:
    GET /api/svg/{name} looks a name up here exactly and never joins a request into a path."""
    out = {}
    for rel_dir, split in SVG_DIRS:
        d = os.path.join(root, rel_dir)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            m = SVG_NAME_RE.match(name)
            p = os.path.join(d, name)
            if m and m.group(1) == split and os.path.isfile(p) and not os.path.islink(p):
                out[name] = rel_dir + "/" + name
    return out


def _confidence_record(root: str, rel: str, name: str, split: str) -> dict[str, Any]:
    """The confidence/1 record beside a reliability SVG (same stem, .json); SystemExit naming the restore unless it is
    valid JSON of that schema whose run_id and split name the SVG (<run_id>-<split>.svg) and, for a val record, which
    is in_sample with a caveat string."""
    if not os.path.isfile(os.path.join(root, rel)):
        raise SystemExit("%s has no confidence record %s; restore it with `git checkout -- %s`"
                         % (name, rel, rel))
    with open(os.path.join(root, rel), "rb") as f:
        r = _bound_json(f.read(), rel)
    sp = r.get("split") if isinstance(r.get("split"), dict) else {}
    if r.get("schema") != CONFIDENCE_SCHEMA or not isinstance(r.get("run_id"), str) or sp.get("name") != split \
            or "%s-%s.svg" % (r["run_id"], split) != name:
        raise SystemExit("%s is not the %s %s record of %s; restore it with `git checkout -- %s`"
                         % (rel, CONFIDENCE_SCHEMA, split, name, rel))
    if split == "val" and (r.get("in_sample") is not True or not isinstance(r.get("caveat"), str)):
        raise SystemExit("%s is a val record without in_sample true and its caveat; restore it with `git checkout -- "
                         "%s`" % (rel, rel))
    return r


def confidence_view(root: str = REPO_ROOT) -> dict[str, Any]:
    """The reliability diagrams as data: one entry per svg_allow_list SVG, its run id and split from the confidence/1
    record beside it; the val records' caveat fields (report.IN_SAMPLE_CAVEAT), unique, in order, passed verbatim.
    SystemExit on a missing or mismatched record or when either split has no diagram.

    Returns {sources, val: [{svg, path, run_id, split}], test: [{svg, path, run_id, split}], caveats: [str]}."""
    out: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    caveats: list[str] = []
    for name, rel in svg_allow_list(root).items():
        split = "val" if name.endswith("-val.svg") else "test"
        r = _confidence_record(root, rel[:-len(".svg")] + ".json", name, split)
        out[split].append({"svg": name, "path": rel, "run_id": r["run_id"], "split": split})
        if split == "val" and r["caveat"] not in caveats:
            caveats.append(r["caveat"])
    for rel_dir, split in SVG_DIRS:
        if not out[split]:
            raise SystemExit("%s holds no %s reliability diagram; restore it with `git checkout -- %s`"
                             % (rel_dir, split, rel_dir))
    return {"sources": [d for d, _ in SVG_DIRS], "val": out["val"], "test": out["test"], "caveats": caveats}


# ---- latency: the protocol-bound latency/1 reports, reported, not gated (D-10) ----

LATENCY_DIR = "results/latency"


def latency_view(root: str = REPO_ROOT) -> dict[str, Any]:
    """The latency reports as data: require_frozen, then every *.json under results/latency and every protocol.json
    input there hashed against its inputs entry (bound_input: an unpinned, missing or changed report is refused naming
    it), then pilot.latency.load_reports (schema and warm-up policy), whose reports must equal the bound bytes parsed.
    Per-message times and the checked state never leave the server: each seed keeps its seed, checkpoint_name and
    summary. render is pilot.latency.render_markdown's text (= latency.py --render results/latency).

    Returns {sources, label (pilot.latency.LABEL), reports [{path, system, device, seeds [{seed, checkpoint_name,
    summary}], pooled}] (sorted path order), render}."""
    proto, _ = pilot.protocol.require_frozen()
    inputs = proto.get("inputs") or {}
    d = os.path.join(root, LATENCY_DIR)
    names = sorted(n for n in os.listdir(d) if n.endswith(".json")) if os.path.isdir(d) else []
    rels = sorted({LATENCY_DIR + "/" + n for n in names} | {r for r in inputs if r.startswith(LATENCY_DIR + "/")})
    bound = {rel: _bound_json(bound_input(root, rel, inputs), rel) for rel in rels}
    reports = pilot.latency.load_reports(d)
    if reports != [bound[r] for r in rels]:
        raise SystemExit("%s changed while it was read; restore it with `git checkout -- %s`"
                         % (LATENCY_DIR, LATENCY_DIR))
    out = [{"path": rel, "system": r["system"], "device": r["device"],
            "seeds": [{"seed": s["seed"], "checkpoint_name": s["checkpoint_name"], "summary": s["summary"]}
                      for s in r["seeds"]], "pooled": r["pooled"]} for rel, r in zip(rels, reports)]
    return {"sources": rels + ["protocol.json"], "label": pilot.latency.LABEL, "reports": out,
            "render": pilot.latency.render_markdown(reports)}


# ---- few-shot: the results record the checked verdict binds (D-10) ----

FEWSHOT_RECORD = "results/fewshot/fewshot.json"
FEWSHOT_CAVEAT = "Few-shot:"                                # the pilot.memo.CAVEATS line shown above the forgetting
AP_BY_K_HEADER = "| k | Laya | BERTurk |"                   # the render's "New-tag macro-AP by k" table


def _fewshot_record(v: dict[str, Any], root: str, pin: str) -> tuple[dict[str, Any], str, str]:
    """(record, repo-relative path, sha256) of the few-shot results record a checked verdict binds: its fewshot.record
    path under root through pilot.fewshot_report.read_record, refused (report.memo_inputs' lines) unless the path is
    repo-relative, the file's sha256 is the verdict's and the record was built under the frozen protocol pin."""
    record = (v.get("fewshot") or {}).get("record")
    rel, want = (record.get("path"), record.get("sha256")) if isinstance(record, dict) else (None, None)
    if not isinstance(rel, str) or not isinstance(want, str) or os.path.isabs(rel) \
            or os.path.normpath(rel).split(os.sep)[0] == os.pardir:
        raise SystemExit("the verdict binds no repo-relative few-shot record (%r); %s"
                         % (record, pilot.verdict.FEWSHOT_FIX))
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        raise SystemExit("%s not found; build it with `make fewshot-reports`" % rel)
    try:
        rec, sha = pilot.fewshot_report.read_record(path)
    except SystemExit as e:                                 # its line names the path it was given: name rel instead
        raise SystemExit(str(e.code).replace(path, rel)) from None
    if sha != want:
        raise SystemExit("%s has sha256 %s, the verdict binds %s; rebuild it with `make fewshot-reports`, then "
                         "`make final-reports`" % (rel, sha, want))
    if rec.get("protocol_sha256") != pin:
        raise SystemExit("%s was built under protocol sha256 %s, not the frozen %s; rebuild it with `make "
                         "fewshot-reports`" % (rel, rec.get("protocol_sha256"), pin))
    return rec, rel, sha


def _ap_by_k(rec: dict[str, Any], render: str) -> dict[str, list[dict[str, Any]]]:
    """{kind: [{k, mean, mean_text, draws}]} of the record's by_k, k ascending: draws are the per_draw values in
    order, mean float(np.mean(draws)) as render_markdown computes it, mean_text "%.3f"; SystemExit unless every
    mean_text is the cell of its model in the render's "New-tag macro-AP by k" row of its k."""
    table = md_table(render, AP_BY_K_HEADER, "few-shot")
    cells = {r[0]: r for r in table["rows"]}
    out: dict[str, list[dict[str, Any]]] = {}
    for col, kind in enumerate(KINDS, 1):                   # the table's Laya and BERTurk columns, in KINDS order
        out[kind] = []
        for k in sorted(rec["by_k"], key=int):
            draws = [d[kind] for d in rec["by_k"][k]["per_draw"]]
            if not draws or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in draws):
                raise SystemExit("%s: by_k %s holds a non-numeric %s AP; rebuild it with `make fewshot-reports`"
                                 % (FEWSHOT_RECORD, k, kind))
            mean = float(np.mean(draws))
            label = "k = 0 (masked base)" if k == "0" else "k = %s" % k
            if (cells.get(label) or [None] * 3)[col] != "%.3f" % mean:
                raise SystemExit("the few-shot render's %r row does not show the %s mean %.3f; rebuild %s with `make "
                                 "fewshot-reports`" % (label, kind, mean, FEWSHOT_RECORD))
            out[kind].append({"k": int(k), "mean": mean, "mean_text": "%.3f" % mean,
                              "draws": [float(x) for x in draws]})
    return out


def _forgetting(f1: dict[str, Any]) -> list[dict[str, Any]]:
    """[{model, run, base_seed, new, old, forgetting}] of the record's f1 block in the render's forgetting-table order
    (per model its masked bases, then its continued runs by draw and k); run is the render's Run cell, forgetting
    None for a base."""
    rows = []
    for kind in KINDS:
        for key in sorted(f1[kind]["base"], key=lambda x: int(x[1:])):
            f = f1[kind]["base"][key]
            rows.append({"model": kind, "run": "base seed %s, k = 0" % key[1:], "base_seed": int(key[1:]),
                         "new": f["new"], "old": f["old"], "forgetting": None})
        for key in sorted(f1[kind]["continued"], key=lambda x: tuple(int(v[1:]) for v in x.split("-"))):
            f = f1[kind]["continued"][key]
            d, k = (int(v[1:]) for v in key.split("-"))
            rows.append({"model": kind, "run": "draw %d, k = %d" % (d, k), "base_seed": f["base_seed"],
                         "new": f["new"], "old": f["old"], "forgetting": f["forgetting"]})
    return rows


def fewshot_view(root: str = REPO_ROOT, path: str | None = None) -> dict[str, Any]:
    """The few-shot results as data: the verdict read and checked as verdict_view reads it (checked_verdict), then
    the few-shot record it binds (its sha256 the verdict's fewshot.record.sha256, built under the frozen pin). render
    is pilot.fewshot_report.render_markdown's text (= fewshot.py --render results/fewshot/fewshot.json); every mean
    shown is a cell of that render. SystemExit on any mismatch.

    Returns {sources, record {path, sha256}, primary_k, mean_delta_primary (record primary.mean_delta, unrounded),
    mean_delta_text ("%+.2f points", the render's), ap_by_k {laya, berturk: [{k, mean, mean_text, draws}]},
    forgetting [{model, run, base_seed, new, old, forgetting}], caveat (the pilot.memo.CAVEATS line starting
    "Few-shot:"), notes (the record's), render}."""
    v, vrel, _, _, pin = checked_verdict(root, path)
    rec, rel, sha = _fewshot_record(v, root, pin)
    render = pilot.fewshot_report.render_markdown(rec)
    primary = rec["primary"]
    delta_text = "%+.2f points" % (100 * primary["mean_delta"])
    render_line(render, "| mean over draws |  |  | %s |" % delta_text, "few-shot")
    caveat = [c for c in pilot.memo.CAVEATS if c.startswith(FEWSHOT_CAVEAT)]
    if len(caveat) != 1:
        raise SystemExit("pilot.memo.CAVEATS holds %d lines starting %r, not one" % (len(caveat), FEWSHOT_CAVEAT))
    return {"sources": [VERDICT_RECORD if path is None else vrel, rel, "protocol.json"],
            "record": {"path": rel, "sha256": sha},
            "primary_k": primary["k"], "mean_delta_primary": primary["mean_delta"], "mean_delta_text": delta_text,
            "ap_by_k": _ap_by_k(rec, render), "forgetting": _forgetting(rec["f1"]), "caveat": caveat[0],
            "notes": list(rec["notes"]), "render": render}


VIEWS: dict[str, Callable[[str], dict[str, Any]]] = {"intents": intents_view, "verdict": verdict_view,
                                                     "selection": selection_view, "confidence": confidence_view,
                                                     "latency": latency_view, "fewshot": fewshot_view}
VIEW_FAILED = "building the %s view failed (%s); check that its records are readable, then restart `make demo`"
VIEW_SOURCES = {"intents": "intents.py", "verdict": VERDICT_RECORD, "selection": SELECTION_RECORDS["laya"][0],
                "confidence": SVG_DIRS[0][0], "latency": LATENCY_DIR, "fewshot": FEWSHOT_RECORD}


def build_views(root: str = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """{name: view} of every VIEWS builder, each run once on `root`; a builder's SystemExit becomes that view's
    {"error": its one line, "source": VIEW_SOURCES[name]} and any other exception {"error": VIEW_FAILED naming only
    its type, "source": VIEW_SOURCES[name]}, so one refused or unreadable record breaks only its own view. Never
    raises an Exception."""
    out = {}
    for name, build in VIEWS.items():
        try:
            out[name] = build(root)
        except SystemExit as e:
            out[name] = {"error": str(e.code).replace("\n", " "), "source": VIEW_SOURCES[name]}
        except Exception as e:
            out[name] = {"error": VIEW_FAILED % (name, type(e).__name__), "source": VIEW_SOURCES[name]}
    return out


def view_refused(view: dict[str, Any]) -> bool:
    """True for the {"error", "source"} placeholder build_views stores for a refused view."""
    return set(view) == {"error", "source"}
