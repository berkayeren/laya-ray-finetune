"""The go/no-go recommendation and the one-page decision memo: renders of the frozen verdict, no judgement.

The recommendation is a mapping of the verdict/1 record's verdict field (RECOMMENDATION): GO -> GO, NO-GO -> NO-GO,
INCONCLUSIVE -> NO-GO; collect bank data, which is pilot.verdict.INCONCLUSIVE_MEANS without its "counts as ". Nothing
here decides, re-derives or overrides a status: every status and the verdict are computed by pilot.verdict from the
record's own numbers at the frozen thresholds, and the caller (report.py --recommendation) runs
pilot.protocol.require_frozen and pilot.verdict.check_record before any render, so a hand-edited record is refused,
never rendered. A provisional verdict and a verdict whose few-shot path of (d) is still PENDING are refused: they are
not final, and a recommendation is only ever the final verdict's mapping. Humans can audit and override the
recommendation outside the repo, from the records the block names; no override is written into the repo.

The block's facts are read from the records (facts): the test split's size and its no-intent count from the
verdict's split block, the final_train folds from pilot.data.SPLITS, the machine and its memory from the spike
record's meta.platform, and the scorer device of every test and HOTELS dump from its sidecar (scorer.device): the
final seeds' must be protocol.json's scoring device, zero-shot Laya's is the CPU it was scored on, and the majority
and prior baselines run no model. No hardware name is typed here: the chip is in no record. The module holds no
threshold and no float literal (a test walks its AST), and imports the stdlib, pilot.verdict and pilot.data only, so
report.py stays numpy only.

The decision memo is render_memo: results/final/memo.md, at most MEMO_MAX_LINES lines, written by `make memo`
(report.py --memo, after require_frozen and check_record). It writes no number of its own: every line that holds a
number is a whole line selected from a generated README render (MEMO_SOURCES: pilot.verdict.render_go,
pilot.verdict.render_final_table, the few-shot block, the final confidence block, the latency block), and the
connective text, the headings and the CAVEATS are digit-free constants. The few-shot per-draw table is not copied:
the memo carries the few-shot result through the GO block's few-shot path line and (d) row, and the README few-shot
block holds the per-draw values.

usage: from pilot import memo
       memo.recommendation(v)                                   # "NO-GO; collect bank data" for the committed verdict
       text = memo.render_recommendation(v, "results/final/verdict.json", sha256, memo.facts(v, protocol))
       python report.py --recommendation results/final/verdict.json [--out rec.md]
       text = memo.render_memo(v, *report.memo_inputs(v))       # the decision memo
       python report.py --memo results/final/verdict.json [--out results/final/memo.md]      # or: make memo
"""
import json
import os
import re
from typing import Any, Callable

import pilot.data
import pilot.verdict
from pilot.data import REPO_ROOT

RECOMMENDATION = {pilot.verdict.GO: "GO", pilot.verdict.NO_GO: "NO-GO",
                  pilot.verdict.INCONCLUSIVE: "NO-GO; collect bank data"}
RECOMMENDATION_HEADING = "### Recommendation (mechanical, from the frozen verdict)"
SPIKE_RECORD = "results/spike/mps_spike.json"
SPIKE_FIX = "restore it with `git checkout -- %s`" % SPIKE_RECORD
# the few-shot block of a verdict built without the few-shot results record ("phase" is a key of the committed record)
PENDING_FEWSHOT = {"status": pilot.verdict.PENDING, "phase": 9}
GIB = 2 ** 30
# a number token: digits with an optional sign and decimal part, not glued to a letter, digit or underscore ("fp16",
# "neg8" and "v2" are not numbers; "k = 10", "+65.87" and "rule 4" are)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_])")
# the caveats as digit-free keys: each is stated by a CAVEATS bullet and by the README Recommendation prose
CAVEAT_KEYS = ("translated", "small dataset", "single split", "laptop hardware", "no-intent test set", "carried over",
               "rule order", "never re-selected", "new-tag micro-F1", "post hoc", "epoch parity", "sampled negatives",
               "fp16", "no fallback", "descriptive", "reported, not gated", "bank-name review")
# the caveats in the order the memo prints them; no number: the sizes and settings live in the records
CAVEATS = (
    "Data: Multi3NLU++ Turkish is translated data, a small dataset and a single split; no bank data was used.",
    "Hardware: laptop hardware, one Apple laptop with unified memory; the final seeds were trained and scored on MPS "
    "in fp32 (protocol.json scoring), zero-shot Laya was scored on CPU and the majority and prior baselines run no "
    "model, as each dump's sidecar records.",
    "No-intent: the no-intent test set is small, so every no-intent false-positive rate is uncertain; its size is on "
    "the test split line.",
    "Settings carried over: each model's epoch, T and θ were chosen on val by one selection run and carried over to "
    "the final_train retrains; the seed-averaged systems use that θ, transferred and never re-selected.",
    "Rule order: the frozen rule order reads criterion (a) before (d), so a few-shot edge cannot change an "
    "INCONCLUSIVE verdict, and the alternative GO fires before an (a) FAIL; changing either needs a new protocol.",
    "Few-shot: BERTurk's new-tag micro-F1 is zero at its masked base's val θ chosen over all columns, because its "
    "untrained held-out rows push that θ up; results/fewshot/fewshot.json adds a post hoc variant at the θ of the "
    "trained columns only, descriptive and not a decision input; average precision needs no θ and is the primary "
    "metric.",
    "Training parity: epoch parity, not step parity; Laya trains on sampled negatives against BERTurk's full label "
    "rows; a continued Laya run starts from its fp16-rounded masked base; no fallback was used.",
    "Scope: the bootstrap intervals are descriptive; latency is reported, not gated; the human bank-name review is "
    "deferred to the milestone end.",
)


def recommendation(v: Any) -> str:
    """The recommendation of a final verdict/1 record: RECOMMENDATION[v["verdict"]["verdict"]]; SystemExit for another
    schema, a provisional verdict, a few-shot path of (d) still PENDING or an unknown verdict word."""
    fix = pilot.verdict.FEWSHOT_FIX
    schema = v.get("schema") if isinstance(v, dict) else type(v).__name__
    if schema != pilot.verdict.VERDICT_SCHEMA:
        raise SystemExit("not a %s record (schema %r); %s" % (pilot.verdict.VERDICT_SCHEMA, schema, fix))
    dec = v.get("verdict") if isinstance(v.get("verdict"), dict) else {}
    if dec.get("provisional") is not False:
        raise SystemExit("the verdict is provisional (%r); a recommendation is only the final verdict's mapping, %s"
                         % (dec.get("verdict"), fix))
    if v.get("fewshot") == PENDING_FEWSHOT:
        raise SystemExit("the verdict's few-shot path of (d) is PENDING; a recommendation needs the few-shot results "
                         "record, %s" % fix)
    word = dec.get("verdict")
    if not isinstance(word, str) or word not in RECOMMENDATION:
        raise SystemExit("the verdict %r is not one of %s; %s" % (word, ", ".join(RECOMMENDATION), fix))
    return RECOMMENDATION[word]


def scorer_device(root: str, run_id: str, split: str) -> str | None:
    """The scorer.device of results/dumps/<run_id>/<split>.json under root (None: a baseline that runs no model);
    SystemExit for a missing or malformed sidecar."""
    rel = "results/dumps/%s/%s.json" % (run_id, split)
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            device = json.load(f)["scorer"]["device"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit("%s holds no scorer.device (%s: %s); restore it with `git checkout -- %s`"
                         % (rel, type(e).__name__, e, rel))
    if device is not None and (not isinstance(device, str) or not device):
        raise SystemExit("%s records scorer device %r, not a device name or null; restore it with `git checkout -- %s`"
                         % (rel, device, rel))
    return device


def scorers(v: dict[str, Any], protocol: dict[str, Any], root: str = REPO_ROOT) -> list[tuple[str | None, list[str]]]:
    """[(device or None, [label])] of every test and HOTELS dump the verdict reports, grouped by the scorer.device of
    its sidecar in first-seen order: each fine-tuned model's final seeds (the verdict's per_seed dump run ids, label
    "the KIND final seeds"; every one must be protocol.json's scoring device), then the references in the results
    table's order (label: the run id; a HOTELS sidecar read when the reference has one); SystemExit otherwise."""
    want = (protocol.get("scoring") or {}).get("device") if isinstance(protocol, dict) else None
    out, rows = [], []
    for kind in pilot.verdict.KINDS:
        seeds = ((v.get("systems") or {}).get(kind) or {}).get("per_seed") or []
        if not seeds:
            raise SystemExit("the verdict lists no %s final seed; %s" % (kind, pilot.verdict.FEWSHOT_FIX))
        devices = {scorer_device(root, s["dump_run_id"], split) for s in seeds for split in ("test", "hotels_probe")}
        if devices != {want}:
            raise SystemExit("the %s final seeds' dump sidecars record scorer devices %s, protocol.json scoring is %r; "
                             "restore the gated dumps with `git checkout -- results/dumps/`" % (kind, sorted(
                                 map(str, devices)), want))
        rows.append((want, "the %s final seeds" % kind))
    for r in pilot.verdict._references(v):              # the results table's order; a malformed row is refused
        splits = ["test"] + (["hotels_probe"] if os.path.isfile(os.path.join(
            root, "results", "dumps", r["run_id"], "hotels_probe.json")) else [])
        for device in sorted({scorer_device(root, r["run_id"], split) for split in splits}, key=str):
            rows.append((device, r["run_id"]))
    for device, label in rows:
        for d, labels in out:
            if d == device:
                labels.append(label)
                break
        else:
            out.append((device, [label]))
    return out


def _and(labels: list[str]) -> str:
    """The labels joined as English: "A", "A and B", "A, B and C"."""
    return labels[0] if len(labels) == 1 else "%s and %s" % (", ".join(labels[:-1]), labels[-1])


def render_scorers(groups: list[tuple[str | None, list[str]]], device: str) -> str:
    """The scorer-device clause of the Hardware line: "mps for A and B (protocol.json scoring), cpu for C, no model for
    D and E"."""
    return ", ".join("%s for %s%s" % (d if d is not None else "no model", _and(labels),
                                     " (protocol.json scoring)" if d == device and d is not None else "")
                     for d, labels in groups)


def facts(v: dict[str, Any], protocol: dict[str, Any], root: str = REPO_ROOT) -> dict[str, Any]:
    """The block's record facts: {n, n_no_intent} of the verdict's split block, the final_train folds as "first-last"
    (pilot.data.SPLITS, one contiguous range), the machine and its memory in whole GiB (SPIKE_RECORD meta.platform),
    the scorer device (protocol.json scoring) and scorers(v, protocol, root), the scorer device of every test and
    HOTELS dump from its sidecar; SystemExit for anything missing or malformed."""
    sp = v.get("split") or {}
    n, n_no = sp.get("n"), sp.get("n_no_intent")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (n, n_no)):
        raise SystemExit("the verdict's split block has no integer n and n_no_intent; %s" % pilot.verdict.FEWSHOT_FIX)
    final = [s for s in pilot.data.SPLITS if s.get("name") == "final_train"]
    folds = final[0].get("folds") if len(final) == 1 else None
    if not folds or list(folds) != list(range(folds[0], folds[-1] + 1)):
        raise SystemExit("pilot.data.SPLITS final_train folds %r are not one contiguous range" % (folds,))
    path = os.path.join(root, SPIKE_RECORD)
    try:
        with open(path, encoding="utf-8") as f:
            plat = json.load(f)["meta"]["platform"]
        machine, mem = plat["machine"], plat["total_memory_bytes"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit("%s holds no meta.platform machine and total_memory_bytes (%s: %s); %s"
                         % (SPIKE_RECORD, type(e).__name__, e, SPIKE_FIX))
    if not isinstance(machine, str) or not machine or not isinstance(mem, int) or isinstance(mem, bool) \
            or mem <= 0 or mem % GIB:
        raise SystemExit("%s meta.platform reads machine %r, total_memory_bytes %r, not a whole number of GiB; %s"
                         % (SPIKE_RECORD, machine, mem, SPIKE_FIX))
    device = (protocol.get("scoring") or {}).get("device") if isinstance(protocol, dict) else None
    if not isinstance(device, str) or not device:
        raise SystemExit("protocol.json has no scoring device; restore it with `git checkout -- protocol.json`")
    return {"n": n, "n_no_intent": n_no, "folds": "%d-%d" % (folds[0], folds[-1]), "machine": machine,
            "memory_gib": mem // GIB, "device": device, "scorers": scorers(v, protocol, root)}


def _one(lines: list[str], test: Callable[[str], bool], what: str) -> str:
    """The one line of lines for which test holds; SystemExit when there is none or more than one."""
    hits = [l for l in lines if test(l)]
    if len(hits) != 1:
        raise SystemExit("the GO render holds %d %s lines, not exactly one; the verdict disagrees with its own GO "
                         "block, %s" % (len(hits), what, pilot.verdict.FEWSHOT_FIX))
    return hits[0]


def render_recommendation(v: dict[str, Any], verdict_path: str, verdict_sha256: str, facts: dict[str, Any]) -> str:
    """The README recommendation block (markdown, level-3 heading) of a final verdict/1 record: the recommendation
    (recommendation(v)), the verdict and the frozen rule line that fired (VERDICT_RULES verbatim), the GO render's
    Statuses line, the records with their sha256 prefixes (verdict_path is repo-relative), and the record facts
    (facts(v, protocol)). Refused unless pilot.verdict.render_go(v) holds exactly one "**Verdict: X**" line for v's
    verdict and one Statuses line, and v's verdict block carries its criteria's statuses."""
    rec = recommendation(v)
    dec, crit = v["verdict"], v.get("criteria") or {}
    want = {k: (crit.get(k) or {}).get("status") for k in ("a", "b", "d", "alternative_go")}
    if dec.get("statuses") != want:
        raise SystemExit("the verdict block's statuses %r are not its criteria's %r; the record was edited, %s"
                         % (dec.get("statuses"), want, pilot.verdict.FEWSHOT_FIX))
    rule = dec.get("rule")
    if not isinstance(rule, int) or isinstance(rule, bool) or not 1 <= rule <= len(pilot.verdict.VERDICT_RULES):
        raise SystemExit("the verdict's rule %r is not one rule of pilot.verdict.VERDICT_RULES; %s"
                         % (rule, pilot.verdict.FEWSHOT_FIX))
    lines = pilot.verdict.render_go(v).split("\n")
    _one(lines, lambda l: l.startswith("**Verdict: "), "**Verdict: ...**")
    _one(lines, lambda l: l == "**Verdict: %s**" % dec["verdict"], "**Verdict: %s**" % dec["verdict"])
    statuses = _one(lines, lambda l: l.startswith("Statuses: "), "Statuses")
    record = (v.get("fewshot") or {}).get("record") or {}
    if not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise SystemExit("the verdict's few-shot block names no record path and sha256; %s"
                         % pilot.verdict.FEWSHOT_FIX)
    out = [RECOMMENDATION_HEADING, "",
           "**Recommendation: %s**" % rec, "",
           "- Verdict: %s, by rule %d: %s." % (dec["verdict"], rule, pilot.verdict.VERDICT_RULES[rule - 1]),
           "- " + statuses,
           "- Records: %s sha256 `%s`, protocol.json sha256 `%s`, few-shot record %s sha256 `%s`. The verdict is "
           "computed mechanically by the frozen rule; humans can audit and override this recommendation from these "
           "records." % (verdict_path, verdict_sha256[:12], v["protocol"]["sha256"][:12], record["path"],
                         record["sha256"][:12]),
           "- Test split: n = %d (%d no-intent)." % (facts["n"], facts["n_no_intent"]),
           "- Final seeds: retrained on final_train, banking folds %s, with each model's epoch, T and θ carried over "
           "from its val selection." % facts["folds"],
           "- Hardware: %s, %d GiB unified memory (%s). Scorer device of the test and HOTELS dumps (their sidecars): "
           "%s." % (facts["machine"], facts["memory_gib"], SPIKE_RECORD, render_scorers(facts["scorers"],
                                                                                        facts["device"]))]
    return "\n".join(out) + "\n"


# ---- the one-page decision memo: whole render lines plus digit-free text ----

MEMO_TITLE = "# Decision memo: Laya vs BERTurk on Turkish banking intents"
MEMO_GENERATED = ("Generated by `make memo` from results/final/verdict.json, results/fewshot/fewshot.json, "
                  "results/final/confidence/ and results/latency/. Every line that holds a number is a line of a "
                  "generated README block; the memo writes no number of its own.")
MEMO_HEADINGS = ("## Recommendation", "## Evidence: the GO block", "## Accuracy on test: the results table",
                 "## Few-shot new intents", "## Caveats", "## Audit, override and reproduce")
MEMO_OVERRIDE = ("- This is the frozen protocol's mechanical verdict, rendered as computed, with no human judgement "
                 "added; humans can audit and override it from the records under Audit.")
MEMO_AUDIT = ("- Audit: protocol.json, results/final/verdict.json, results/fewshot/fewshot.json, "
              "results/final/confidence/, results/latency/ and results/dumps/ are the audit record; a reviewer may "
              "override the recommendation outside the repo.")
MEMO_REBUILD = ("- Rebuild: `make report` rebuilds every report, the README blocks and this memo from the committed "
                "dumps, without training or scoring, and fails on any file that differs; `make reproduce` runs the "
                "whole pipeline in order and skips every step its committed artifacts prove done.")
MEMO_MAX_LINES = 60                                         # one page: about sixty lines
MEMO_SOURCES = ("go", "table", "fewshot", "confidence", "latency")
GO_TABLE_HEADER = "| criterion | status | numbers |"
GO_TABLE_LAST = "| alternative GO"
RESULTS_TABLE_HEADER = "| system | run | micro-F1 | macro-F1 | exact match | top-1 |"


def _line(text: str, prefix: str, what: str) -> str:
    """The one line of text starting with prefix; SystemExit when there is none or more than one."""
    hits = [l for l in text.split("\n") if l.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("the %s render holds %d lines starting %r; the memo selects whole render lines"
                         % (what, len(hits), prefix))
    return hits[0]


def _table(text: str, header: str, what: str) -> list[str]:
    """The contiguous run of lines starting "|" from the one line equal to header; SystemExit otherwise."""
    lines = text.split("\n")
    at = [i for i, l in enumerate(lines) if l == header]
    if len(at) != 1:
        raise SystemExit("the %s render holds %d lines equal to %r; the memo selects whole render lines"
                         % (what, len(at), header))
    out = []
    for l in lines[at[0]:]:
        if not l.startswith("|"):
            break
        out.append(l)
    return out


def render_memo(v: dict[str, Any], fewshot_md: str, confidence_md: str, latency_md: str) -> str:
    """The one-page decision memo (markdown) of a final verdict/1 record: the recommendation (recommendation(v)) and
    the GO block's fired rule line, the evidence (the GO block's protocol and split lines, the final confidence
    block's no-intent line, the GO criteria table, the latency block's label line), the results table, the few-shot
    result (the GO block's few-shot path line, the few-shot block's criterion line), every CAVEATS bullet, the audit
    and override statement and the rebuild commands. fewshot_md, confidence_md and latency_md are the README renders
    of the verdict's few-shot record, results/final/confidence and results/latency (report.memo_inputs). Refused
    unless render_go(v) holds the verdict's "**Verdict: X**" line and its rule line, or when the memo is longer than
    MEMO_MAX_LINES lines."""
    rec = recommendation(v)
    dec = v["verdict"]
    go, table = pilot.verdict.render_go(v), pilot.verdict.render_final_table(v)
    src = dict(zip(MEMO_SOURCES, (go, table, fewshot_md, confidence_md, latency_md)))
    rule = dec.get("rule")
    if not isinstance(rule, int) or isinstance(rule, bool) or not 1 <= rule <= len(pilot.verdict.VERDICT_RULES):
        raise SystemExit("the verdict's rule %r is not one rule of pilot.verdict.VERDICT_RULES; %s"
                         % (rule, pilot.verdict.FEWSHOT_FIX))
    go_lines = go.split("\n")
    rule_line = "- Rule %d: %s" % (rule, pilot.verdict.VERDICT_RULES[rule - 1])
    for want in ("**Verdict: %s**" % dec["verdict"], rule_line):
        if go_lines.count(want) != 1:
            raise SystemExit("the GO render holds %d lines %r; the verdict disagrees with its own GO block, %s"
                             % (go_lines.count(want), want, pilot.verdict.FEWSHOT_FIX))
    criteria = _table(src["go"], GO_TABLE_HEADER, "go")
    if not criteria[-1].startswith(GO_TABLE_LAST):
        raise SystemExit("the GO render's criteria table ends %r, not the alternative GO row; the memo selects whole "
                         "render lines" % criteria[-1])
    h = MEMO_HEADINGS
    out = [MEMO_TITLE, MEMO_GENERATED,
           "", h[0], "**Recommendation: %s**" % rec, rule_line, MEMO_OVERRIDE,
           "", h[1], _line(src["go"], "- Protocol:", "go"), _line(src["go"], "- Split ", "go"),
           _line(src["confidence"], "The test no-intent set holds", "confidence"), ""] + criteria + [
           "", _line(src["latency"], "Latency is reported, not gated:", "latency"),
           "", h[2], ""] + _table(src["table"], RESULTS_TABLE_HEADER, "table") + [
           "", h[3], _line(src["go"], "- Few-shot path", "go"),
           _line(src["fewshot"], "Criterion (d)'s few-shot path reads", "fewshot"),
           "", h[4]] + ["- " + c for c in CAVEATS] + [
           "", h[5], MEMO_AUDIT, MEMO_REBUILD]
    if len(out) > MEMO_MAX_LINES:
        raise SystemExit("the memo is one page: %d lines, at most %d" % (len(out), MEMO_MAX_LINES))
    return "\n".join(out) + "\n"
