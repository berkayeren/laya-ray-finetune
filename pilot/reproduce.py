"""The step registry of the one-command reproduction: every pipeline step, its committed evidence, its done/todo
state and the printed plan, plus the eval-only rebuild, the command layer and full mode built on it.

Order: the steps run in one fixed pipeline order, STEP_NAMES: data, baselines, laya-selection, laya-postfit,
laya-final, berturk-selection, berturk-postfit, berturk-reference, berturk-final, confidence, latency, final-scoring,
final-reports, fewshot-projection, fewshot-bases, fewshot-continued, fewshot-scoring, fewshot-reports, report.

Done: a step is done only by committed, hash-verified artifacts read with the existing readers: the split and audit
verifiers of pilot.data, pilot.dumps.load_dump (with the gate keys of a gated dump), the protocol.json input pins
(the sha256 of a file's exact bytes == protocol["inputs"][path]), pilot.postfit.read_record,
pilot.latency.read_final_manifest and load_reports, pilot.verdict.check_record, pilot.fewshot.read_mode and read_runs,
and pilot.fewshot_report.read_record. Every evidence path lies under data/ or results/; a checkpoint (ckpt*), Ray
storage (ray_results/), BERTurk run storage (berturk_runs/) or a run dir is never consulted. So on a finished tree
every step through scoring is done and only the report step runs. Each protocol.json input is the evidence of
exactly one step.

Fail closed: a partial, tampered or out-of-order tree is refused, in one line, before anything runs, and is never
half-run or recomputed: a step with none of its artifacts is todo, with all of them verified done, with some missing
refused naming the missing paths and the fix; a present artifact that does not verify is refused with its reader's
own refusal (or the protocol pin mismatch naming both sha256 values) and the `git checkout -- PATH` restore fix; a
dump whose npz or sidecar alone exists (an orphan) is refused naming both paths; and once a step is todo, a later
step with any committed artifact is refused naming both steps. Full mode also refuses, before any command, a todo
step any of whose evidence files git HEAD holds (head_holds): a deleted record is never treated as work to redo,
whether or not a later step is done, and a gated scoring step (final-scoring, fewshot-scoring) is never rescored
around its committed dumps; the fix is `git checkout -- PATH`. A from-scratch run therefore needs a repository whose
HEAD does not hold the outputs it makes (see DISCLOSURE). The data step is local (data/splits/ and data/audit/ are
gitignored, written by `make data`) and exempt from the order rule; the report step always runs.

Torch-free: the stdlib, pilot.protocol and pilot.data at import time; numpy, scipy and the record readers are imported
inside the evidence functions, so computing the plan of a tree never imports torch, transformers, laya or Ray.

Eval-only (`make report`, reproduce.py --mode eval-only; it never trains or scores, and the final-seed re-hash is
reported, not gated): eval_only verifies first (plan(): the frozen protocol, every split and manifest hash, every
committed record and dump through the step registry), refuses before any command a missing record or dump, naming
its path and the step that makes it (eval-only never runs a todo step but the local data step, `make data`, run only
when the local split files are absent), then report_step rebuilds every report from the committed dumps (REBUILD:
fixed argv tuples, no score.py, training script, `latency.py --system`, --force or scoring target), rewrites the
README's generated blocks from their renders (README_BLOCKS, rewrite_blocks) and proves the tree unchanged:
`git status --porcelain` over CHECKED (results/, README.md, protocol.json, and the Makefile and runner sources that
decide what runs) must be empty before (else the run is refused, so a difference is always the rebuild's) and after
(else the run fails naming every differing path); a root that is not the top of a git work tree is refused before
any command (NO_GIT), because without git there is no committed copy to compare the rebuilt files with or to restore
them from. In full mode after steps ran, the report step compares a file git HEAD tracks with HEAD's bytes (an
already dirty file fails) and a file HEAD does not track (an output of those steps) with its bytes before the
rebuild; "git status clean" is printed only when it is. Every command is an argv list run without a shell by the
interpreter running this module (make targets get PYTHON=sys.executable).
verify_checkpoints (--verify-checkpoints) re-hashes the on-disk final seeds with score.py's identity functions against
their final manifests: reported, not gated (it loads no model, writes nothing and never refuses).

The command layer (reproduce.py --dry-run [--all]): commands(name, root) gives every step's argv tuples, built from
the committed records, protocol.json and the Makefile, never retyped: the selection runs and the few-shot runs from
protocol.json's fewshot block (rules.base_runs, the frozen epochs, the draws, pilot.fewshot.slots and base_seed_of
under the recorded mode), the final seeds from pilot.postfit.final_command and pilot.berturk.final_command of the
hash-verified postfit records, the baselines as the Makefile's score.py lines without --force, final-reports as
final_report_commands (the Makefile recipe; the from-scratch form without the verdict's --fewshot, since fewshot.json
comes later in the pipeline order), and fewshot-reports as `make fewshot-reports`, then the verdict with --fewshot and
`make memo` (fewshot_report_commands), so the committed verdict and the memo are that step's outputs. No command
passes --force or runs `make baselines`, and the gated scoring targets appear only in their own steps. dry_run prints
the plan, then each todo step's commands (every step's with --all), with DISCLOSURE, the retraining disclosure, once
before the first model step listed; it runs nothing. A from-scratch run is disclosed, not run here: MPS retraining is
not bit-identical, so it stops at the first output that differs from the committed record or a protocol pin, and the
gate is never weakened.

Full mode (`make reproduce`, reproduce.py --mode full, the default): run_full takes plan() (so a partial, tampered or
out-of-order tree is refused before any command), skips every done step ("skip NAME (done)"), runs each todo step's
commands in the pipeline order, evaluated just before the step runs (so a record an earlier step wrote feeds the next
step's commands), and ends with report_step, the eval-only rebuild with its pre and post tree checks. A committed path
under CHECKED that already differs from git HEAD (other than a todo step's evidence) is refused before any command.
DISCLOSURE prints once before the first model step; a selection run id is resolved with the step's --print-run-id
command (capture_command) and substituted for RUN_ID; the first non-zero exit stops the run in one line naming the
step and the command (a protocol gate refusal from score.py is final). After every step it ran, after_check re-reads
the outputs that step itself writes (after_items): its evidence, except an item whose committed file a later step
writes (LATER_OUTPUT: the bases' slots are recorded in the runs.json that fewshot-continued writes, so they are
verified with fewshot-continued), plus the deferred items of earlier steps whose file it writes. An artifact that is
missing, fails its reader, differs from its protocol.json pin or differs from the bytes git HEAD holds for it
(committed_bytes) stops the run with the retraining reason (RETRAIN_STOP), naming the step that wrote it. A gated
scoring step (GATE_PRODUCERS) runs only when the steps that train its checkpoints ran in the same invocation, and
never when git HEAD holds its dumps (the refusal above); otherwise it is refused before any command, because the
gated dumps are the exactly-once audit record. No command passes --force.
On a finished tree every step through fewshot-reports is done, so only the report regenerates.

usage:
    python -m pilot.reproduce --plan          # the 19-step plan of the working tree, each step done, todo or run
    python reproduce.py                       # = make reproduce: every todo step in order, then the report
    python reproduce.py --mode eval-only      # = make report: verify, rebuild every report, the memo and the README
    python reproduce.py --mode eval-only --dry-run [--all]   # print the plan and the commands; run nothing
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence

from pilot import protocol
from pilot.data import (AUDIT_FILES, REPO_ROOT, SPLITS, sha256_bytes, sha256_file, verify_audit_file,
                        verify_split_file, write_atomic)

STEP_NAMES = ("data", "baselines", "laya-selection", "laya-postfit", "laya-final", "berturk-selection",
              "berturk-postfit", "berturk-reference", "berturk-final", "confidence", "latency", "final-scoring",
              "final-reports", "fewshot-projection", "fewshot-bases", "fewshot-continued", "fewshot-scoring",
              "fewshot-reports", "report")
LOCAL = ("data",)                    # local, gitignored, written by `make data`: exempt from the order rule
ALWAYS = ("report",)                 # rebuilds every report, the README blocks and the memo: always runs
GATED = ("final-scoring", "fewshot-scoring")
MODEL_STEPS = ("baselines", "laya-selection", "laya-final", "berturk-selection", "berturk-final", "latency",
               "final-scoring", "fewshot-bases", "fewshot-continued", "fewshot-scoring")
RUNS = "results/fewshot/runs.json"
VERDICT = "results/final/verdict.json"
MEMO = "results/final/memo.md"
FEWSHOT_RECORD = "results/fewshot/fewshot.json"
CONFIDENCE_SCHEMA = "confidence/1"
MAKEFILE_VAR_RE = re.compile(r"^([A-Z_][A-Z0-9_]*) [:?]= (.*)$")     # `NAME := value` and the `NAME ?= default`

# annotation aliases: an evidence item or registry step is a plain dict, a command a tuple of argv strings
Item = dict[str, Any]                # one evidence item (_item)
Step = dict[str, Any]                # one registry step (_step)
Row = tuple[Step, str, list[Item]]   # one plan row: (step, state, items)
Evidence = Callable[[str, dict[str, Any], str, str], list[Item]]     # (root, proto, pin, manifest) -> [item]
Cmd = tuple[str, ...]                # a make target or a Python CLI, never a shell string (command_argv)
Line = tuple[Cmd | None, str | None]                                  # (command or None, note or None)
Builder = Callable[[str, dict[str, Any], str], list[Line]]           # (root, proto, pin) -> [line]
Slot = tuple[str, str, int, int | None, int | None]                  # a few-shot (kind, stage, seed, draw, k)
Runner = Callable[[Sequence[str], str], int]                          # (command, root) -> exit code
Blocks = Callable[[str], list[tuple[str, str]]]                       # root -> [(README block name, render)]
Out = Callable[[str], object]                                         # a print-like sink


def _one_line(msg: object) -> str:
    """msg as one line: every run of whitespace, newlines included, collapsed to one space."""
    return " ".join(str(msg).split())


def makefile_vars(root: str) -> dict[str, str]:
    """{NAME: value} of the `NAME := value` and `NAME ?= value` lines of root/Makefile (a ?= line gives the Makefile's
    own default); SystemExit when it is missing."""
    path = os.path.join(root, "Makefile")
    if not os.path.isfile(path):
        raise SystemExit("%s not found; restore the committed file with `git checkout -- Makefile`" % path)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = MAKEFILE_VAR_RE.match(line.rstrip("\n"))
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def _item(rel: str, present: bool, ok: bool, reason: str, files: list[str] | None = None,
          label: str | None = None) -> Item:
    """One evidence item: its display path, the committed files a restore names, present, verified, why not."""
    return {"path": rel, "files": list(files or [rel]), "present": present, "ok": ok, "reason": reason,
            "label": label or rel}


def _check(fn: Callable[..., object], *args: Any) -> str | None:
    """None when fn(*args) returns, else its SystemExit message on one line."""
    try:
        fn(*args)
    except SystemExit as e:
        return _one_line(e.code if e.code is not None else "refused")
    return None


def _file(root: str, rel: str, proto: dict[str, Any], check: Callable[[str], object] | None = None) -> Item:
    """The item of one committed file: present by os.path.lexists; a protocol.json input must hash to its pin, then
    check(path) must return."""
    p = os.path.join(root, rel)
    if not os.path.lexists(p):
        return _item(rel, False, False, "")
    if os.path.islink(p) or not os.path.isfile(p):
        return _item(rel, True, False, "%s is a symlink or not a file; a committed record is a file" % rel)
    want = proto["inputs"].get(rel)
    if want is not None:
        with open(p, "rb") as f:
            have = sha256_bytes(f.read())
        if have != want:
            return _item(rel, True, False, "%s sha256 %s is not the frozen protocol's %s" % (rel, have, want))
    reason = _check(check, p) if check else None
    return _item(rel, True, reason is None, reason or "")


def _dump(root: str, run_id: str, split: str, manifest: str, proto: dict[str, Any],
          keys: dict[str, Any] | None = None) -> Item:
    """The item of one dump results/dumps/<run_id>/<split>.{npz,json}: present when either file exists; verified when
    a pinned sidecar hashes to its protocol.json pin, load_dump passes and every `keys` item equals the sidecar's."""
    from pilot import dumps
    rel = "results/dumps/%s/%s" % (run_id, split)
    npz, js = rel + ".npz", rel + ".json"
    have = [os.path.lexists(os.path.join(root, r)) for r in (npz, js)]

    def item(present: bool, ok: bool, reason: str) -> Item:
        return _item(rel, present, ok, reason, files=[npz, js])
    if not any(have):
        return item(False, False, "")
    if not all(have):                                 # an orphan: never "run score.py", a gated split is scored once
        there, gone = (npz, js) if have[0] else (js, npz)
        return item(True, False, "%s is present without %s (an orphan: a dump is its npz and its sidecar, committed "
                                 "together)" % (there, gone))
    if js in proto["inputs"] and have[1]:
        pinned = _file(root, js, proto)
        if not pinned["ok"]:
            return item(True, False, pinned["reason"])
    try:
        _, _, side, _ = dumps.load_dump(os.path.join(root, "results", "dumps", run_id), split, manifest)
    except SystemExit as e:
        return item(True, False, _one_line(e.code))
    for k, v in sorted((keys or {}).items()):
        if side.get(k) != v:
            return item(True, False, "%s: sidecar %s %r, expected %r" % (js, k, side.get(k), v))
    return item(True, True, "")


def _json_schema(schema: str | None) -> Callable[[str], None]:
    """A check: the file is a JSON object, of `schema` when given."""
    def check(path: str) -> None:
        with open(path, "rb") as f:
            raw = f.read()
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise SystemExit("%s is not valid JSON (%s)" % (path, e))
        if not isinstance(obj, dict) or (schema is not None and obj.get("schema") != schema):
            raise SystemExit("%s: schema %r, expected %s" % (path, obj.get("schema") if isinstance(obj, dict) else
                                                             None, schema))
    return check


# ---- the evidence of each step (root, proto, pin, manifest) -> [item] ----

def ev_data(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The committed manifest (pinned; an anchor: verified, never counted as the step's local output) and the 5 split
    files and 3 audit files `make data` writes, each against the manifest."""
    anchor = _file(root, "data/splits.json", proto)
    anchor["anchor"] = True
    items = [anchor]
    for s in SPLITS:
        items.append(_file(root, "data/splits/%s.jsonl" % s["name"], proto,
                           lambda p: verify_split_file(p, manifest)))
    for rel in sorted(AUDIT_FILES.values()):
        items.append(_file(root, "data/" + rel, proto, lambda p: verify_audit_file(p, manifest)))
    return items


def ev_baselines(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The six baseline dumps (majority, prior and zero-shot Laya on val and test) and their three test reports."""
    items = [_dump(root, rid, split, manifest, proto) for rid in ("majority", "prior", protocol.ZEROSHOT)
             for split in ("val", "test")]
    return items + [_file(root, "results/reports/%s.json" % rid, proto, _json_schema(None))
                    for rid in ("majority", "prior", protocol.ZEROSHOT)]


def _selection(var: str) -> Evidence:
    """The evidence of a selection step: the val dump of every run id the Makefile variable var lists."""
    def ev(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
        return [_dump(root, rid, "val", manifest, proto) for rid in makefile_vars(root)[var].split()]
    return ev


def _postfit(kind: str) -> Evidence:
    """The evidence of a postfit step: results/postfit/<kind>.json, read with pilot.postfit.read_record."""
    def check(path: str) -> None:
        from pilot import postfit
        postfit.read_record(path)

    def ev(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
        return [_file(root, "results/postfit/%s.json" % kind, proto, check)]
    return ev


def _final(kind: str) -> Evidence:
    """The evidence of a final-seed step: results/postfit/<kind>-final.json, whose dump run ids must be the frozen
    protocol's final seeds of kind."""
    def check(path: str, proto: dict[str, Any]) -> None:
        from pilot import latency
        m, _ = latency.read_final_manifest(path, kind)
        ids = [s.get("dump_run_id") for s in m["seeds"]]
        want = proto["systems"][kind]["dump_run_ids"]
        if ids != want:
            raise SystemExit("%s lists the final dump run ids %s, the frozen protocol's are %s" % (path, ids, want))

    def ev(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
        return [_file(root, "results/postfit/%s-final.json" % kind, proto, lambda p: check(p, proto))]
    return ev


def ev_berturk_reference(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The BERTurk reference check record (schema berturk-reference/1)."""
    return [_file(root, "results/postfit/berturk-reference.json", proto, _json_schema("berturk-reference/1"))]


def ev_confidence(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The three val confidence records and their reliability SVGs (the chosen Laya and BERTurk selection candidates
    and zero-shot Laya, run ids from the Makefile)."""
    v = makefile_vars(root)
    items = []
    for rid in (v["LAYA_CHOSEN"], v["BERTURK_CHOSEN"], v["LAYA_ZS"]):
        items.append(_file(root, "results/confidence/%s-val.json" % rid, proto, _json_schema(CONFIDENCE_SCHEMA)))
        items.append(_file(root, "results/confidence/%s-val.svg" % rid, proto))
    return items


def ev_latency(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The four latency reports (each system on mps and cpu), plus the directory as one item read with
    pilot.latency.load_reports."""
    from pilot import latency
    items = [_file(root, "results/latency/%s-%s.json" % (kind, dev), proto)
             for kind in protocol.KINDS for dev in ("mps", "cpu")]
    d = os.path.join(root, "results", "latency")
    present = os.path.isdir(d) and any(n.endswith(".json") for n in os.listdir(d))
    reason = _check(latency.load_reports, d) if present else None
    items.append(_item("results/latency", present, present and reason is None, reason or ""))
    return items


def _final_seeds(proto: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """[(kind, final seed entry)] of the frozen protocol, Laya's seeds first."""
    return [(kind, s) for kind in protocol.KINDS for s in protocol.final_seeds(proto, kind)]


def ev_final_scoring(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The gated test and HOTELS dumps of every final seed (each carrying protocol_sha256 == pin) and the ungated
    zero-shot HOTELS reference."""
    items = [_dump(root, s["dump_run_id"], split, manifest, proto, {"protocol_sha256": pin})
             for _, s in _final_seeds(proto) for split in ("test", "hotels_probe")]
    return items + [_dump(root, protocol.ZEROSHOT, "hotels_probe", manifest, proto)]


def confidence_names(proto: dict[str, Any]) -> list[tuple[str, bool]]:
    """[(record name, with an SVG)] of the 15 committed results/final/confidence records: every final seed on test
    (with its reliability SVG) and hotels_probe, zero-shot Laya on both, the prior on test."""
    out = [("%s-%s" % (s["dump_run_id"], split), split == "test") for _, s in _final_seeds(proto)
           for split in ("test", "hotels_probe")]
    return out + [("%s-test" % protocol.ZEROSHOT, True), ("%s-hotels_probe" % protocol.ZEROSHOT, False),
                  ("prior-test", True)]


def ev_final_reports(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """results/final/: the verdict (pilot.verdict.check_record against the frozen protocol), the per-seed test
    reports and the confidence records with their SVGs."""
    def verdict(path: str) -> None:
        from pilot import verdict as pv
        with open(path, encoding="utf-8") as f:
            try:
                v = json.load(f)
            except ValueError as e:
                raise SystemExit("%s is not valid JSON (%s)" % (path, e))
        pv.check_record(v, proto, pin)
    items = [_file(root, VERDICT, proto, verdict)]
    items += [_file(root, "results/final/reports/%s-test.json" % s["dump_run_id"], proto, _json_schema(None))
              for _, s in _final_seeds(proto)]
    for name, svg in confidence_names(proto):
        items.append(_file(root, "results/final/confidence/%s.json" % name, proto, _json_schema(CONFIDENCE_SCHEMA)))
        if svg:
            items.append(_file(root, "results/final/confidence/%s.svg" % name, proto))
    return items


def ev_fewshot_projection(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The few-shot compute decision record results/fewshot/projection.json, read with pilot.fewshot.read_mode."""
    from pilot import fewshot
    return [_file(root, "results/fewshot/projection.json", proto, lambda p: fewshot.read_mode(pin, p))]


def read_runs(root: str, proto: dict[str, Any], pin: str) -> tuple[dict[str, Any] | None, str | None]:
    """(runs manifest, sha256) of root's results/fewshot/runs.json through pilot.fewshot.read_runs, bound to root's
    own few-shot compute decision record (pilot.fewshot.DECISION_PATH is read at call time, so it points at root's
    for the call), or (None, None) when the file does not exist."""
    from pilot import fewshot
    saved = fewshot.DECISION_PATH
    fewshot.DECISION_PATH = os.path.join(root, "results", "fewshot", "projection.json")
    try:
        return fewshot.read_runs(proto, pin, os.path.join(root, RUNS))
    finally:
        fewshot.DECISION_PATH = saved


def _runs_item(root: str, proto: dict[str, Any], pin: str) -> tuple[Item, dict[str, Any] | None, str | None]:
    """(the runs.json item, runs manifest or None, its sha256 or None)."""
    out = {}

    def check(path: str) -> None:
        out["runs"], out["sha"] = read_runs(root, proto, pin)
    item = _file(root, RUNS, proto, check)
    return item, out.get("runs") if item["ok"] else None, out.get("sha") if item["ok"] else None


def _fewshot_stage(stage: str) -> Evidence:
    """The evidence of a few-shot training step: runs.json plus one item per slot of stage (base or continued) under
    the recorded mode, present when the runs manifest lists it."""
    def ev(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
        from pilot import fewshot
        item, runs, _ = _runs_item(root, proto, pin)
        if runs is None:
            return [item]
        listed = {(e["kind"], e["stage"], e["seed"], e["draw"], e["k"]) for e in runs["entries"]}
        slots = [s for s in fewshot.slots(proto["fewshot"], runs["mode"]) if s[1] == stage]
        return [item] + [_item(RUNS, s in listed, s in listed, "", label="%s (%s)" % (RUNS, fewshot.slot_name(s)))
                         for s in slots]
    return ev


def ev_fewshot_scoring(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """Every dump runs.json lists: each masked base's ungated val dump and every gated test dump (protocol_sha256 ==
    pin, fewshot_runs_sha256 == the runs manifest's); none before runs.json verifies."""
    _, runs, sha = _runs_item(root, proto, pin)
    if runs is None:
        return []
    keys = {"protocol_sha256": pin, "fewshot_runs_sha256": sha}
    items = []
    for e in runs["entries"]:
        if e["stage"] == "base":
            items.append(_dump(root, e["dump_run_id"], "val", manifest, proto))
        items.append(_dump(root, e["dump_run_id"], "test", manifest, proto, keys))
    return items


def ev_fewshot_reports(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The few-shot results record results/fewshot/fewshot.json, whose sha256 must be the one the committed verdict
    binds when the verdict exists."""
    def check(path: str) -> None:
        from pilot import fewshot_report
        _, sha = fewshot_report.read_record(path)
        vp = os.path.join(root, VERDICT)
        if os.path.isfile(vp):
            with open(vp, encoding="utf-8") as f:
                want = ((json.load(f).get("fewshot") or {}).get("record") or {}).get("sha256")
            if want != sha:
                raise SystemExit("%s sha256 %s is not the %s that %s binds (fewshot.record.sha256)"
                                 % (path, sha, want, VERDICT))
    return [_file(root, "results/fewshot/fewshot.json", proto, check)]


def ev_report(root: str, proto: dict[str, Any], pin: str, manifest: str) -> list[Item]:
    """The report step has no evidence: it always runs."""
    return []


def _step(name: str, evidence: Evidence, fix: str, outputs: tuple[str, ...] = ()) -> Step:
    """One registry step; outputs are the committed files it writes that are no step's plan evidence (the memo):
    after_check requires them present and equal to the bytes git HEAD holds for them."""
    return {"name": name, "model": name in MODEL_STEPS, "gated": name in GATED, "evidence": evidence, "fix": fix,
            "outputs": tuple(outputs)}


_DUMPS_FIX = "restore the committed dumps with `git checkout -- results/dumps/`"
STEPS = [
    _step("data", ev_data, "run `make data` (the split and audit files are local and gitignored)"),
    _step("baselines", ev_baselines, "restore the committed baseline dumps and reports with `git checkout -- "
          "results/dumps/ results/reports/`; `make baselines` rescores with --force and never runs on a partly "
          "committed tree"),
    _step("laya-selection", _selection("LAYA_SELECTION"), _DUMPS_FIX + "; a selection candidate is never rescored"),
    _step("laya-postfit", _postfit("laya"), "restore it with `git checkout -- results/postfit/laya.json`"),
    _step("laya-final", _final("laya"), "restore it with `git checkout -- results/postfit/laya-final.json`"),
    _step("berturk-selection", _selection("BERTURK_SELECTION"), _DUMPS_FIX + "; a selection candidate is never "
          "rescored"),
    _step("berturk-postfit", _postfit("berturk"), "restore it with `git checkout -- results/postfit/berturk.json`"),
    _step("berturk-reference", ev_berturk_reference, "restore it with `git checkout -- "
          "results/postfit/berturk-reference.json`"),
    _step("berturk-final", _final("berturk"), "restore it with `git checkout -- results/postfit/berturk-final.json`"),
    _step("confidence", ev_confidence, "restore the committed records with `git checkout -- results/confidence/`"),
    _step("latency", ev_latency, "restore the committed reports with `git checkout -- results/latency/`; latency "
          "is timed on the final seeds and never retimed on a partly committed tree"),
    _step("final-scoring", ev_final_scoring, "restore the gated dumps with `git checkout -- results/dumps/`; a "
          "gated split is scored once and never rescored"),
    _step("final-reports", ev_final_reports, "restore the committed records with `git checkout -- results/final/`"),
    _step("fewshot-projection", ev_fewshot_projection, "restore it with `git checkout -- "
          "results/fewshot/projection.json`; the few-shot compute decision is recorded once"),
    _step("fewshot-bases", _fewshot_stage("base"), "restore it with `git checkout -- results/fewshot/runs.json`"),
    _step("fewshot-continued", _fewshot_stage("continued"), "restore it with `git checkout -- "
          "results/fewshot/runs.json`"),
    _step("fewshot-scoring", ev_fewshot_scoring, "restore the gated dumps with `git checkout -- results/dumps/`; a "
          "gated split is scored once and never rescored"),
    _step("fewshot-reports", ev_fewshot_reports, "restore it with `git checkout -- results/fewshot/fewshot.json`",
          outputs=(MEMO,)),
    _step("report", ev_report, "it always runs"),
]


def _restore(step: Step, item: Item) -> str:
    """The fix for one item that does not verify: `make data` for a local data file, else `git checkout -- FILES`."""
    if step["name"] in LOCAL and not item.get("anchor"):
        return "regenerate it with `make data`"
    return "restore the committed file with `git checkout -- %s`" % " ".join(item["files"])


def state(step: Step, items: list[Item]) -> str:
    """"run" (report), "todo" (no artifact present), "done" (every artifact present and verified); SystemExit for a
    present artifact that does not verify (its own refusal plus the restore fix) or a partly present step."""
    name = step["name"]
    if name in ALWAYS:
        return "run"
    for it in items:
        if it["present"] and not it["ok"]:
            fix = _restore(step, it)
            if "`make data`" in fix and "`make data`" in it["reason"]:
                raise SystemExit("%s: %s" % (name, it["reason"]))     # the verifier's own fix already ends it
            raise SystemExit("%s: %s; %s" % (name, it["reason"], fix))
    counted = [it for it in items if not it.get("anchor")]
    if not any(it["present"] for it in counted):
        return "todo"
    missing = [it["label"] for it in items if not it["present"]]
    if missing:
        raise SystemExit("%s is partly done: %d of %d artifacts present, missing %s; %s"
                         % (name, len(items) - len(missing), len(items), ", ".join(missing[:3]), step["fix"]))
    return "done"


def plan(root: str = REPO_ROOT) -> list[Row]:
    """[(step, state, items)] of the tree at `root`, in STEP_NAMES order, after require_frozen of root's protocol.json;
    SystemExit (one line) on the first partial, tampered or out-of-order step."""
    proto, pin = protocol.require_frozen(os.path.join(root, "protocol.json"))
    manifest = os.path.join(root, "data", "splits.json")
    rows, first_todo = [], None
    for step in STEPS:
        items = step["evidence"](root, proto, pin, manifest)
        st = state(step, items)
        if step["name"] not in LOCAL + ALWAYS:
            if st == "todo" and first_todo is None:
                first_todo = (step, items)
            elif st == "done" and first_todo is not None:
                earlier, its = first_todo
                gone = [it["label"] for it in its if not it["present"]]
                raise SystemExit("step %s is done but the earlier step %s is not (missing %s); the committed record is "
                                 "incomplete, never work to redo: %s"
                                 % (step["name"], earlier["name"], gone[0] if gone else "its artifacts",
                                    earlier["fix"]))
        rows.append((step, st, items))
    return rows


def render_plan(rows: list[Row], pin: str) -> str:
    """The printed plan: a header with the protocol sha256 prefix, then one line per step."""
    out = ["reproduce plan of the working tree (protocol.json sha256 %s...)" % pin[:12]]
    for i, (step, st, items) in enumerate(rows, 1):
        if st == "run":
            what = "always runs: rebuilds every report, the README blocks and the memo"
        elif st == "done":
            what = "%d artifact%s verified" % (len(items), "" if len(items) == 1 else "s")
        else:
            what = "no committed artifact: " + step["fix"] if step["name"] in LOCAL else "no committed artifact"
        out.append("%2d %-18s %-4s %s" % (i, step["name"], st, what))
    return "\n".join(out) + "\n"


# ---- eval-only: rebuild every report from the committed dumps, no training, no scoring ----

# the audit record a rebuild must leave byte for byte, plus what decides the commands eval-only runs: the Makefile
# (every REBUILD make target's recipe, and the run ids makefile_vars reads) and the runner itself,
# so an uncommitted edit to any of them is refused before any command
CHECKED = ("results/", "README.md", "protocol.json", "Makefile", "reproduce.py", "pilot/reproduce.py")
REBUILD = (
    ("make", "baseline-reports"),
    ("make", "laya-postfit"),
    ("make", "berturk-postfit"),
    ("postfit.py", "--reference-check", "results/postfit/berturk.json", "--out",
     "results/postfit/berturk-reference.json"),
    ("make", "confidence-reports"),
    ("latency.py", "--render", "results/latency", "--out", "results/latency/latency.md"),
    ("make", "fewshot-reports"),
    ("make", "final-reports"),
    ("make", "memo"),
)
README_BLOCKS = (
    ("laya-postfit", ("postfit.py", "--render", "results/postfit/laya.json", "results/postfit/laya-final.json")),
    ("berturk-postfit", ("postfit.py", "--render-berturk", "results/postfit/berturk.json",
                         "results/postfit/berturk-reference.json", "results/postfit/berturk-final.json")),
    ("confidence", ("report.py", "--confidence-table", "results/confidence")),
    ("latency", ("latency.py", "--render", "results/latency")),
    ("protocol", ("-m", "pilot.protocol", "--render")),
    ("final-results", ("report.py", "--final-table", VERDICT)),
    ("go-block", ("report.py", "--go-block", VERDICT)),
    ("final-confidence", ("report.py", "--confidence-table", "results/final/confidence")),
    ("fewshot", ("fewshot.py", "--render", "results/fewshot/fewshot.json")),
    ("results-table", ("report.py", "--table", "results/reports")),
    ("recommendation", ("report.py", "--recommendation", VERDICT)),
)
MODES = ("full", "eval-only")
README = "README.md"


def _show(cmd: Sequence[str]) -> str:
    """One command tuple as space-joined text, for messages."""
    return " ".join(cmd)


def command_argv(cmd: Sequence[str]) -> list[str]:
    """The argv list of one command tuple: a make target with PYTHON pinned to this interpreter, or a Python CLI (a
    script or -m module) run by this interpreter; SystemExit for anything else (never a shell string)."""
    cmd = tuple(cmd)
    if not cmd or not all(isinstance(c, str) and c for c in cmd):
        raise SystemExit("reproduce runs a tuple of non-empty strings, not %r" % (cmd,))
    if cmd[0] == "make" and len(cmd) > 1:
        return list(cmd) + ["PYTHON=" + sys.executable]
    if cmd[0] == "-m" or cmd[0].endswith(".py"):
        return [sys.executable] + list(cmd)
    raise SystemExit("reproduce runs a make target or a Python CLI (a .py script or -m module), never `%s`"
                     % _show(cmd))


def run_command(cmd: Sequence[str], root: str) -> int:
    """Run one command tuple in root, without a shell; its exit code."""
    sys.stdout.flush()
    return subprocess.run(command_argv(cmd), cwd=root).returncode


def _git(root: str, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    """git ARGS run in root with captured output, or None when git cannot be run."""
    try:
        return subprocess.run(("git",) + args, cwd=root, capture_output=True)
    except OSError:
        return None


UNTRACKED = "??"                     # the porcelain status of a file git HEAD does not track
NO_GIT = ("%s compares every rebuilt file with the bytes git HEAD holds, but %s is not the top of a git work tree, so "
          "there is no committed copy to compare with or restore from; it is refused before any command, so no file "
          "is overwritten: run it in a clone of the repository (a source copy: `git init`, `git add -A` and "
          "`git commit` first)")


def tree_changes(root: str) -> list[str] | None:
    """The sorted repo-relative paths `git status --porcelain --untracked-files=all` reports under CHECKED in root
    (modified, deleted, added or untracked), or None when git is missing or root is not the top of a work tree."""
    st = tree_status(root)
    return None if st is None else sorted(st)


def tree_status(root: str) -> dict[str, str] | None:
    """{repo-relative path: its two-letter porcelain status} of every path `git status --porcelain
    --untracked-files=all` reports under CHECKED in root (UNTRACKED for a file git HEAD does not track), or None when
    git is missing or root is not the top of a work tree."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or top.returncode != 0:
        return None
    if os.path.realpath(top.stdout.decode("utf-8", "surrogateescape").strip()) != os.path.realpath(root):
        return None
    st = _git(root, "status", "--porcelain", "-z", "--untracked-files=all", "--", *CHECKED)
    if st is None or st.returncode != 0:
        err = st.stderr.decode("utf-8", "replace").strip().splitlines() if st is not None else []
        raise SystemExit("git status in %s failed (%s); eval-only cannot compare the rebuilt files"
                         % (root, err[-1] if err else "git not runnable"))
    tokens, out, i = st.stdout.split(b"\0"), {}, 0
    while i < len(tokens):
        entry = tokens[i]
        i += 1
        if len(entry) < 4:
            continue
        out[entry[3:].decode("utf-8", "surrogateescape")] = entry[:2].decode("ascii", "replace")
        if entry[:1] in (b"R", b"C"):                          # -z: a rename's source path follows its target
            i += 1
    return out


def _hashes(root: str, rels: Iterable[str]) -> dict[str, str]:
    """{rel: sha256} of the regular files among the repo-relative rels in root."""
    return {rel: sha256_file(os.path.join(root, rel)) for rel in rels if os.path.isfile(os.path.join(root, rel))}


def snapshot(root: str) -> dict[str, str]:
    """{repo-relative path: sha256} of every file under CHECKED in root (a content fingerprint of a tree)."""
    out = {}
    for rel in CHECKED:
        p = os.path.join(root, rel.rstrip("/"))
        if os.path.isdir(p):
            for d, dirs, files in os.walk(p):
                dirs.sort()
                for name in sorted(files):
                    fp = os.path.join(d, name)
                    out[os.path.relpath(fp, root).replace(os.sep, "/")] = sha256_file(fp)
        elif os.path.isfile(p):
            out[rel] = sha256_file(p)
    return out


def _snapshot_diff(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """The sorted paths whose sha256 differs between two snapshots (added and removed paths included)."""
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def rewrite_blocks(text: str, renders: Iterable[tuple[str, str]]) -> str:
    """text with each <!-- name:start --> ... <!-- name:end --> body replaced by "\\n" + its render; SystemExit,
    before anything is written, unless each marker occurs exactly once and the start comes first."""
    for name, body in renders:
        s, e = "<!-- %s:start -->" % name, "<!-- %s:end -->" % name
        if text.count(s) != 1 or text.count(e) != 1 or text.index(s) > text.index(e):
            raise SystemExit("README.md must hold exactly one %s ... %s pair in order; restore it with "
                             "`git checkout -- README.md`" % (s, e))
        i, j = text.index(s) + len(s), text.index(e)
        text = text[:i] + "\n" + body + text[j:]
    return text


def render_blocks(root: str) -> list[tuple[str, str]]:
    """[(name, stdout)] of every README_BLOCKS render run in root (UTF-8, bytes kept as printed); SystemExit naming
    the render and its last stderr line when one exits non-zero."""
    out = []
    for name, cmd in README_BLOCKS:
        sys.stdout.flush()
        r = subprocess.run(command_argv(cmd), cwd=root, capture_output=True)
        if r.returncode != 0:
            lines = [l for l in r.stderr.decode("utf-8", "replace").splitlines() if l.strip()]
            raise SystemExit("README block %s: `%s` exited %d (%s)"
                             % (name, _show(cmd), r.returncode, lines[-1].strip() if lines else "no stderr"))
        out.append((name, r.stdout.decode("utf-8")))
    return out


def report_step(root: str = REPO_ROOT, run: Runner = run_command, blocks: Blocks = render_blocks,
                precheck: bool = True, out: Out = print) -> None:
    """The report step: every REBUILD command, then the README blocks; SystemExit when the CHECKED paths already
    differ from git HEAD (precheck, before any command), when a command fails, or afterwards, naming every path, when a
    file git HEAD tracks differs from HEAD's bytes (whatever it was before the rebuild, so an already dirty file
    fails) or a file HEAD does not track (full mode, after steps ran: their untracked outputs) was created, changed or
    removed by the rebuild. A root that is not the top of a git work tree is refused before any command (NO_GIT):
    without git there is no committed copy to compare with or restore from."""
    before = tree_status(root)
    if before is None:
        raise SystemExit(NO_GIT % ("eval-only", root))
    if before and precheck:
        raise SystemExit("eval-only compares every rebuilt file with the committed one, but these already differ: %s; "
                         "commit or restore them first" % ", ".join(sorted(before)))
    untracked = _hashes(root, [p for p, xy in before.items() if xy == UNTRACKED])
    for cmd in REBUILD:
        out("eval-only: " + _show(cmd))
        rc = run(cmd, root)
        if rc != 0:
            raise SystemExit("eval-only: `%s` exited %d" % (_show(cmd), rc))
    path = os.path.join(root, README)
    if not os.path.isfile(path):
        raise SystemExit("%s not found; restore the committed file with `git checkout -- README.md`" % path)
    with open(path, "rb") as f:
        text = f.read().decode("utf-8")
    renders = blocks(root)
    new = rewrite_blocks(text, renders)
    if new != text:
        write_atomic(path, new.encode("utf-8"))
    out("README blocks: %d rendered, README.md %s" % (len(renders), "rewritten" if new != text else "unchanged"))
    after = tree_status(root)
    if after is None:
        raise SystemExit("eval-only: %s is no longer the top of a git work tree; the rebuilt files cannot be compared "
                         "with git HEAD" % root)
    changed = {p for p, xy in after.items() if xy != UNTRACKED}                # differs from git HEAD
    changed |= set(_snapshot_diff(untracked, _hashes(root, [p for p, xy in after.items() if xy == UNTRACKED])))
    changed = sorted(changed)
    if changed:
        raise SystemExit("eval-only rebuilt %d file(s) that differ from the committed ones: %s; the committed results "
                         "are the audit record (git HEAD's bytes for a file it tracks, the bytes before the rebuild for "
                         "an untracked one): inspect with `git diff`, restore with `git checkout -- PATH`"
                         % (len(changed), ", ".join(changed)))
    if not after:
        how = "git status clean"
    else:
        how = ("git status lists only %d untracked file(s), the outputs of the steps that ran, each unchanged by the "
               "rebuild" % len(after))
    out("eval-only done: %d rebuild commands and %d README blocks left %s byte-identical to the committed ones (%s)"
        % (len(REBUILD), len(renders), ", ".join(CHECKED), how))


VERIFY_LINE = "verify-checkpoints (reported, not gated): "
ABSENT_DETAIL = "not needed: eval-only rebuilds from the committed dumps"


def _identity_diff(have: Any, want: Any) -> list[str]:
    """The identity keys (files_sha256 entries as files_sha256[NAME]) whose digests differ."""
    have, want = have if isinstance(have, dict) else {}, want if isinstance(want, dict) else {}
    out = []
    for k in sorted(set(have) | set(want)):
        a, b = have.get(k), want.get(k)
        if k == "files_sha256" and isinstance(a, dict) and isinstance(b, dict):
            out += ["files_sha256[%s]" % f for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
        elif a != b:
            out.append(k)
    return out


def verify_checkpoints(root: str = REPO_ROOT, checkpoint_root: str | None = None,
                       manifests: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Reported, not gated: [{kind, seed, checkpoint_name, status, detail}] of every final seed listed in
    root's results/postfit/<kind>-final.json (manifests={kind: path} overrides them), its dir under checkpoint_root
    (default root) re-hashed with the identity functions score.py uses (pilot.weights.identity + pilot.dumps.run_id,
    pilot.berturk.identity + dump_run_id) against the manifest's checkpoint_identity and dump_run_id: "match",
    "differs" (naming the differing keys, or the identity's own refusal for an incomplete dir) or "absent". Hashes only:
    no model load, no write, and never a refusal for differs or absent."""
    from pilot import berturk, dumps, latency, weights
    rows = []
    for kind in protocol.KINDS:
        path = (manifests or {}).get(kind) or os.path.join(root, "results", "postfit", "%s-final.json" % kind)
        m, _ = latency.read_final_manifest(path, kind)
        for s in m["seeds"]:
            d = os.path.join(checkpoint_root or root, s["checkpoint_name"])
            row = {"kind": kind, "seed": s["seed"], "checkpoint_name": s["checkpoint_name"]}
            if not os.path.isdir(d):
                rows.append(dict(row, status="absent", detail=ABSENT_DETAIL))
                continue
            try:
                if kind == "laya":
                    ident = weights.identity(d)
                    rid = dumps.run_id("laya", ident)
                else:
                    ident = berturk.identity(d)
                    rid = berturk.dump_run_id(ident)
            except SystemExit as e:
                rows.append(dict(row, status="differs", detail=_one_line(e.code)))
                continue
            keys = _identity_diff(ident, s["checkpoint_identity"]) + (["dump_run_id"] if rid != s["dump_run_id"]
                                                                       else [])
            rows.append(dict(row, status="differs" if keys else "match",
                             detail="differs in %s" % ", ".join(keys) if keys else ""))
    return rows


def _missing_refusal(step: Step, items: list[Item]) -> str:
    """Eval-only's refusal of a step that is not done: its first missing path and the step that makes it."""
    gone = [it for it in items if not it["present"]]
    if not gone:
        return "eval-only never trains or scores: step %s has no committed artifact; %s" % (step["name"], step["fix"])
    it = gone[0]
    return ("eval-only never trains or scores: %s is missing; it is made by step %s; restore the committed record "
            "with `git checkout -- %s`" % (it["path"], step["name"], " ".join(it["files"])))


def _print_verify(root: str, checkpoint_root: str | None, out: Out) -> None:
    """One VERIFY_LINE per verify_checkpoints row."""
    for r in verify_checkpoints(root, checkpoint_root):
        out("%s%s seed %s %s: %s%s" % (VERIFY_LINE, r["kind"], r["seed"], r["checkpoint_name"], r["status"],
                                      " (%s)" % r["detail"] if r["detail"] else ""))


def eval_only(root: str = REPO_ROOT, run: Runner = run_command, blocks: Blocks = render_blocks, out: Out = print,
              verify: bool = False, checkpoint_root: str | None = None) -> None:
    """Eval-only: verify the tree through plan(), refuse before any command a step other than data that is
    not done (its first missing path and the step that makes it), run `make data` only when the local split files are
    absent, then (verify: the verify_checkpoints rows first, reported, not gated) report_step."""
    rows = plan(root)
    for step, st, items in rows:
        if step["name"] not in LOCAL + ALWAYS and st != "done":
            raise SystemExit(_missing_refusal(step, items))
    if verify:
        _print_verify(root, checkpoint_root, out)
    if dict((step["name"], st) for step, st, _ in rows)["data"] == "todo":
        out("eval-only data step: make data (the local split files are absent)")
        rc = run(("make", "data"), root)
        if rc != 0:
            raise SystemExit("eval-only: `make data` exited %d" % rc)
        rows = plan(root)
        st = dict((step["name"], s) for step, s, _ in rows)["data"]
        if st != "done":
            raise SystemExit("eval-only: `make data` left the data step %s; regenerate with `make data`" % st)
    report_step(root, run, blocks, precheck=True, out=out)


# ---- the command layer: every step's commands, built from the committed records, protocol.json and the Makefile,
# never retyped; printed by dry_run, which runs nothing ----

RUN_ID = "<RUN_ID>"                  # a selection run id before its postfit record exists (--print-run-id prints it)
# the retraining disclosure, printed once before the first model step (dry_run, run_full)
DISCLOSURE = (
    "Full mode retrains every model from scratch. MPS retraining is not bit-identical, so the new selection exports, "
    "final seeds and few-shot checkpoints get other identities and dump run ids than the frozen record holds. The "
    "protocol gate (score.py require_admitted) admits on test and HOTELS only the protocol's pinned final seeds, and "
    "the latency reports are protocol.json inputs that a re-timing never reproduces. The runner therefore stops at the "
    "first step whose output differs from the committed record or a protocol pin, naming the path; it never weakens "
    "the gate and never reruns a gated scoring step. A record that git HEAD holds is never redone: a step whose "
    "committed artifacts are missing from the working tree is refused before any command, so a from-scratch run "
    "needs a repository whose HEAD does not hold the results. `git checkout -- results/` restores the audit record. "
    "ray_train.py resumes an existing ray_results/<run_id>/ and berturk_train.py refuses an existing "
    "berturk_runs/<run_id>/, so remove both before a from-scratch run.")
STORAGE = {"laya": "ray_results", "berturk": "berturk_runs"}
BASE = {"laya": "multilingual", "berturk": "berturk"}
FINAL_PREFIX = {"laya": "ckpt-final", "berturk": "ckpt-berturk-final"}
SELECTION_OUT = "ckpt-sel"
REFERENCE = "results/postfit/berturk-reference.json"
PROJECTION = "results/fewshot/projection.json"
LATENCY_DEVICES = ("mps", "cpu")


def _record(root: str, proto: dict[str, Any], kind: str) -> tuple[dict[str, Any] | None, str]:
    """(the committed postfit record of kind or None when absent, its repo-relative path), read with
    pilot.postfit.read_record after its protocol.json pin; SystemExit (the refusal and the restore fix) for a record
    that is present and does not verify, so an edited record never yields a command."""
    from pilot import postfit
    rel = "results/postfit/%s.json" % kind
    got = {}

    def check(p: str) -> None:
        got["record"] = postfit.read_record(p)[0]
    item = _file(root, rel, proto, check)
    if not item["present"]:
        return None, rel
    if not item["ok"]:
        raise SystemExit("%s; restore the committed file with `git checkout -- %s`" % (item["reason"], rel))
    if got["record"].get("system_kind") != kind:
        raise SystemExit("%s: system kind %r, expected %s; restore the committed file with `git checkout -- %s`"
                         % (rel, got["record"].get("system_kind"), kind, rel))
    return got["record"], rel


def _laya_argv(b: dict[str, Any], data: str, base: str, epochs: int, seed: int, lr: float, head_lr: float,
               stop: int | None = None, tail: Sequence[str] = ()) -> Cmd:
    """A ray_train.py argv in the recorded flag order from a protocol recipe b (fewshot.rules.base_runs.laya)."""
    argv = [b["trainer"], "--data", data, "--base", base, "--device", b["device"], "--workers", str(b["workers"]),
            "--micro-batch", str(b["micro_batch"]), "--grad-accum", str(b["grad_accum"]), "--max-len",
            str(b["max_len"]), "--epochs", str(epochs)]
    if stop is not None:
        argv += ["--stop-after-epoch", str(stop)]
    argv += ["--neg", str(b["neg"]), "--seed", str(seed), "--encoder-lr", repr(lr), "--head-lr", repr(head_lr)]
    return tuple(argv) + tuple(tail)


def _berturk_argv(b: dict[str, Any], data: str, base: str, epochs: int, seed: int, lr: float, head_lr: float,
                  stop: int | None = None, tail: Sequence[str] = ()) -> Cmd:
    """A berturk_train.py argv in the recorded flag order from a protocol recipe b (fewshot.rules.base_runs.berturk);
    --batch and --max-len only where b differs from berturk_train.py's defaults (pilot.berturk.BATCH, MAX_LEN), as
    the recorded commands pass neither."""
    from pilot import berturk
    argv = [b["trainer"], "--data", data, "--base", base, "--device", b["device"], "--epochs", str(epochs)]
    if stop is not None:
        argv += ["--stop-after-epoch", str(stop)]
    argv += ["--lr", repr(lr), "--head-lr", repr(head_lr)]
    for key, flag, default in (("batch", "--batch", berturk.BATCH), ("max_len", "--max-len", berturk.MAX_LEN)):
        if b.get(key, default) != default:
            argv += [flag, str(b[key])]
    return tuple(argv + ["--seed", str(seed)]) + tuple(tail)


def selection_command(proto: dict[str, Any], kind: str) -> Cmd:
    """The selection run of kind: protocol fewshot.rules.base_runs.<kind> (every field but stop_after_epoch, the full
    schedule) at pilot.postfit.SELECTION_SEED on the base runs' split; Laya with --storage ray_results --out ckpt-sel,
    BERTurk with --storage berturk_runs (its exports are its output)."""
    from pilot import postfit
    runs = proto["fewshot"]["rules"]["base_runs"]
    b, data = runs[kind], "data/splits/%s.jsonl" % runs["split"]
    if kind == "laya":
        return _laya_argv(b, data, BASE[kind], b["epochs"], postfit.SELECTION_SEED, b["encoder_lr"], b["head_lr"],
                          tail=("--storage", STORAGE[kind], "--out", SELECTION_OUT))
    return _berturk_argv(b, data, BASE[kind], b["epochs"], postfit.SELECTION_SEED, b["encoder_lr"], b["head_lr"],
                         tail=("--storage", STORAGE[kind]))


def _selection_lines(kind: str) -> Builder:
    """The lines of a selection step: the run id printer, the selection run, then per epoch the val scoring of its
    export and the export's prune; the run id is the postfit record's when it exists, else RUN_ID."""
    def build(root: str, proto: dict[str, Any], pin: str) -> list[Line]:
        rec, rel = _record(root, proto, kind)
        train, out = selection_command(proto, kind), []
        out.append((train + ("--print-run-id",), "prints the run id (%s/<run_id>/); it starts no Ray and trains "
                                                 "nothing" % STORAGE[kind]))
        out.append((train, None))
        if rec is None:
            run_id = RUN_ID
            out.append((None, "%s stands for the printed run id until %s exists" % (RUN_ID, rel)))
        else:
            ids = sorted({c.get("train_run_id") for c in rec["candidates"]})
            if len(ids) != 1:
                raise SystemExit("%s holds candidates of the train runs %s, not one selection run; restore the "
                                 "committed file with `git checkout -- %s`" % (rel, ids, rel))
            run_id = ids[0]
        scoring = proto["scoring"]
        for e in range(1, proto["fewshot"]["rules"]["base_runs"][kind]["epochs"] + 1):
            export = "%s/%s/exports/epoch-%03d" % (STORAGE[kind], run_id, e)
            out.append((("score.py", "data/splits/val.jsonl", "--model", kind, "--checkpoint", export, "--device",
                         scoring["device"], "--workers", str(scoring["workers"])), None))
            out.append((("postfit.py", "--prune", export), None))
        return out
    return build


def _final_lines(kind: str) -> Builder:
    """The lines of a final-seed step: one training command per final seed from the postfit record (deferred, with
    the command that prints them, while the record is absent), then the final manifest."""
    def build(root: str, proto: dict[str, Any], pin: str) -> list[Line]:
        from pilot import berturk, postfit
        rec, rel = _record(root, proto, kind)
        storage, prefix, out = STORAGE[kind], FINAL_PREFIX[kind], []
        if rec is None:
            seeds = [s["seed"] for s in protocol.final_seeds(proto, kind)]
            printer = (("postfit.py", "--print-final-commands", rel) + (("--reference", REFERENCE) if kind == "berturk"
                                                                        else ()) + ("--storage", storage,
                                                                                    "--out-prefix", prefix))
            out.append((printer, "deferred: the final-seed training commands are printed once %s exists" % rel))
            note = "deferred: printed once %s exists, it runs after the final-seed commands" % rel
        else:
            seeds, note = list(rec["final"]["seeds"]), None
            for s in seeds:
                cmd = (postfit.final_command(rec, s, rel, storage, prefix) if kind == "laya"
                       else berturk.final_command(rec, s, rel, REFERENCE, storage, prefix))
                out.append((tuple(cmd), None))
        out.append((("postfit.py", "--final-manifest", rel) + tuple("%s-s%d" % (prefix, s) for s in seeds)
                    + ("--out", "results/postfit/%s-final.json" % kind), note))
        return out
    return build


def latency_commands() -> list[Cmd]:
    """The four latency runs (each final seed of a system on a device, the pinned report paths) and the render."""
    runs = [("latency.py", "--system", kind, "--device", dev, "--out", "results/latency/%s-%s.json" % (kind, dev))
            for kind in protocol.KINDS for dev in LATENCY_DEVICES]
    return runs + [("latency.py", "--render", "results/latency", "--out", "results/latency/latency.md")]


def final_report_commands(proto: dict[str, Any], fewshot: str | None = None) -> list[Cmd]:
    """The Makefile final-reports recipe as argv tuples, from the protocol's final seeds, their postfit paths, ZEROSHOT
    and the prior (report.py creates the --out dirs): the 6 per-seed test reports, the verdict (with --fewshot fewshot
    when given) and the 15 confidence records."""
    seeds = _final_seeds(proto)

    def postfit_path(kind: str) -> str:
        return proto["systems"][kind]["postfit"]["path"]
    out = [("report.py", "results/dumps/%s" % s["dump_run_id"], "--split", "test", "--postfit", postfit_path(kind),
            "--out", "results/final/reports/%s-test.json" % s["dump_run_id"]) for kind, s in seeds]
    out.append(("report.py", "--verdict") + (("--fewshot", fewshot) if fewshot else ()) + ("--out", VERDICT))
    out += [("report.py", "--confidence", "results/dumps/%s" % s["dump_run_id"], "--split", split, "--postfit",
             postfit_path(kind), "--out", "results/final/confidence/%s-%s.json" % (s["dump_run_id"], split))
            for kind, s in seeds for split in ("test", "hotels_probe")]
    out += [("report.py", "--confidence", "results/dumps/%s" % protocol.ZEROSHOT, "--split", split, "--out",
             "results/final/confidence/%s-%s.json" % (protocol.ZEROSHOT, split)) for split in ("test", "hotels_probe")]
    return out + [("report.py", "--confidence", "results/dumps/prior", "--split", "test", "--out",
                   "results/final/confidence/prior-test.json")]


def fewshot_report_commands() -> list[Cmd]:
    """The fewshot-reports step: the few-shot record (`make fewshot-reports`), then the verdict that
    binds it (the Makefile final-reports line with --fewshot) and the memo (`make memo`), the committed forms of
    results/final/verdict.json and results/final/memo.md, which need the few-shot record."""
    return [("make", "fewshot-reports"), ("report.py", "--verdict", "--fewshot", FEWSHOT_RECORD, "--out", VERDICT),
            ("make", "memo")]


def fewshot_checkpoint(slot: Slot) -> str:
    """The checkpoint dir name of a few-shot slot (kind, stage, seed, draw, k), as runs.json lists it."""
    kind, stage, seed, draw, k = slot
    return ("ckpt-fewshot-%s-base-s%d" % (kind, seed) if stage == "base"
            else "ckpt-fewshot-%s-d%d-k%d" % (kind, draw, k))


def fewshot_command(proto: dict[str, Any], slot: Slot, mode: str) -> Cmd:
    """The training argv of one few-shot slot from protocol.json's fewshot block: a masked base at
    rules.base_runs.<kind> (--fewshot-base), or a continued run of the frozen epochs at rules.continued_training's LRs
    from the base pilot.fewshot.base_seed_of(block, draw, mode), seed = the draw's seed."""
    from pilot import fewshot
    kind, stage, seed, draw, k = slot
    block = proto["fewshot"]
    runs = block["rules"]["base_runs"]
    b, data = runs[kind], "data/splits/%s.jsonl" % runs["split"]
    argv = _laya_argv if kind == "laya" else _berturk_argv
    tail = ("--no-export-epochs", "--storage", STORAGE[kind], "--out", fewshot_checkpoint(slot))
    if stage == "base":
        return argv(b, data, BASE[kind], b["epochs"], seed, b["encoder_lr"], b["head_lr"], stop=b["stop_after_epoch"],
                    tail=("--fewshot-base",) + tail)
    ct = block["rules"]["continued_training"]
    base = fewshot_checkpoint((kind, "base", fewshot.base_seed_of(block, draw, mode), None, None))
    return argv(b, data, base, block["epochs"], seed, ct["encoder_lr"], ct["head_lr"],
                tail=("--fewshot-draw", str(draw), "--fewshot-k", str(k)) + tail)


def _fewshot_slots(root: str, proto: dict[str, Any], pin: str) -> tuple[list[Slot], str, str | None]:
    """(every slot of pilot.fewshot.slots in slot order, the mode, a note or None): the mode from root's few-shot
    compute decision record
    (pilot.fewshot.read_mode) when it exists, else primary with a note."""
    from pilot import fewshot
    path = os.path.join(root, PROJECTION)
    if os.path.lexists(path):
        mode, note = fewshot.read_mode(pin, path), None
    else:
        mode, note = "primary", "the mode is primary until %s records the few-shot compute decision" % PROJECTION
    return fewshot.slots(proto["fewshot"], mode), mode, note


def _fewshot_stage_lines(stage: str) -> Builder:
    """The lines of a few-shot training step: per slot of stage its training command, `fewshot.py --inspect` and,
    for Laya, `--drop-storage`; the continued stage ends with the runs manifest."""
    def build(root: str, proto: dict[str, Any], pin: str) -> list[Line]:
        slots, mode, note = _fewshot_slots(root, proto, pin)
        out = [(None, note)] if note else []
        for slot in slots:
            if slot[1] != stage:
                continue
            name = fewshot_checkpoint(slot)
            out.append((fewshot_command(proto, slot, mode), None))
            out.append((("fewshot.py", "--inspect", name), None))
            if slot[0] == "laya":
                out.append((("fewshot.py", "--drop-storage", name, "--storage", STORAGE["laya"]), None))
        if stage == "continued":
            out.append((("fewshot.py", "--runs-manifest") + tuple(fewshot_checkpoint(s) for s in slots)
                        + ("--out", RUNS), None))
        return out
    return build


def _fewshot_scoring_lines(root: str, proto: dict[str, Any], pin: str) -> list[Line]:
    """`make fewshot-scoring`, then `fewshot.py --prune` of every few-shot checkpoint."""
    slots, _, note = _fewshot_slots(root, proto, pin)
    return ([(None, note)] if note else []) + [(("make", "fewshot-scoring"), None),
                                                (("fewshot.py", "--prune") + tuple(fewshot_checkpoint(s)
                                                                                   for s in slots), None)]


def _baselines_lines(root: str, proto: dict[str, Any], pin: str) -> list[Line]:
    """The Makefile baselines recipe without --force (every baseline dump is absent when the step is todo, so none is
    overwritten): the six score.py lines, zero-shot Laya at the Makefile's SCORE_WORKERS and SCORE_CPUS, then
    baseline-reports. `make baselines` itself passes --force and is never run."""
    v = makefile_vars(root)
    ray = ("--workers", v["SCORE_WORKERS"], "--cpus-per-worker", v["SCORE_CPUS"])
    cmds = [("score.py", "data/splits/%s.jsonl" % split, "--model", model) + (ray if model == "laya" else ())
            for model in ("majority", "prior", "laya") for split in ("val", "test")]
    return [(c, None) for c in cmds] + [(("make", "baseline-reports"), None)]


def _make(target: str) -> Builder:
    """A step that is one make target."""
    def build(root: str, proto: dict[str, Any], pin: str) -> list[Line]:
        return [(("make", target), None)]
    return build


def _fixed(*cmds: Cmd) -> Builder:
    """A step of fixed command tuples."""
    def build(root: str, proto: dict[str, Any], pin: str) -> list[Line]:
        return [(c, None) for c in cmds]
    return build


BUILDERS = {
    "data": _make("data"),
    "baselines": _baselines_lines,
    "laya-selection": _selection_lines("laya"),
    "laya-postfit": _make("laya-postfit"),
    "laya-final": _final_lines("laya"),
    "berturk-selection": _selection_lines("berturk"),
    "berturk-postfit": _make("berturk-postfit"),
    "berturk-reference": _fixed(("postfit.py", "--reference-check", "results/postfit/berturk.json", "--out", REFERENCE)),
    "berturk-final": _final_lines("berturk"),
    "confidence": _make("confidence-reports"),
    "latency": lambda root, proto, pin: [(c, None) for c in latency_commands()],
    "final-scoring": _make("final-scoring"),
    "final-reports": lambda root, proto, pin: [(c, None) for c in final_report_commands(proto)],
    "fewshot-projection": _fixed(("fewshot.py", "--projection", "--out", PROJECTION)),
    "fewshot-bases": _fewshot_stage_lines("base"),
    "fewshot-continued": _fewshot_stage_lines("continued"),
    "fewshot-scoring": _fewshot_scoring_lines,
    "fewshot-reports": lambda root, proto, pin: [(c, None) for c in fewshot_report_commands()],
    "report": _fixed(*REBUILD),
}


def step_lines(name: str, root: str = REPO_ROOT) -> list[Line]:
    """[(command tuple or None, note or None)] of one step, in run order, from root's frozen protocol.json, Makefile
    and committed records; SystemExit for an unknown step or a record that does not verify."""
    if name not in BUILDERS:
        raise SystemExit("no step %r; the steps are %s" % (name, ", ".join(STEP_NAMES)))
    proto, pin = protocol.require_frozen(os.path.join(root, "protocol.json"))
    return BUILDERS[name](root, proto, pin)


def commands(name: str, root: str = REPO_ROOT) -> list[Cmd]:
    """The command tuples of one step (no interpreter: command_argv adds it), in run order."""
    return [c for c, _ in step_lines(name, root) if c is not None]


def display(cmd: Sequence[str]) -> str:
    """One command as a reader types it: "make TARGET" or "python SCRIPT ARGS" (shlex-quoted); SystemExit for
    anything command_argv would not run."""
    cmd = tuple(cmd)
    command_argv(cmd)
    if cmd[0] == "make":
        return "make " + shlex.join(cmd[1:])
    return "python " + shlex.join(cmd)


def dry_run(root: str = REPO_ROOT, all_steps: bool = False) -> str:
    """The text of a dry run: the plan of root, then each todo or always-run step (every step with
    all_steps) as a "== N name (state)" header and its displayed commands, with the DISCLOSURE once before the first
    model step listed. It runs nothing."""
    rows = plan(root)
    _, pin = protocol.require_frozen(os.path.join(root, "protocol.json"))
    out, disclosed, n = [render_plan(rows, pin).rstrip("\n")], False, 0
    for i, (step, st, _) in enumerate(rows, 1):
        if st == "done" and not all_steps:
            continue
        if step["model"] and not disclosed:
            out += ["", DISCLOSURE]
            disclosed = True
        out += ["", "== %2d %s (%s)" % (i, step["name"], st)]
        for cmd, note in step_lines(step["name"], root):
            if cmd is not None:
                out.append(display(cmd))
                n += 1
            if note:
                out.append("# " + note)
        if step["name"] in ALWAYS:
            out.append("README: rewrite the %d generated blocks of README_BLOCKS (%s)"
                       % (len(README_BLOCKS), ", ".join(name for name, _ in README_BLOCKS)))
    out += ["", "dry run: %d commands printed, nothing was run" % n]
    return "\n".join(out) + "\n"


# ---- full mode (`make reproduce`, reproduce.py --mode full): the todo steps in order, then the report ----

GATE_PRODUCERS = {"final-scoring": ("laya-final", "berturk-final"), "fewshot-scoring": ("fewshot-continued",)}
RETRAIN_STOP = ("a from-scratch run stops at the first output that differs from the committed record or a "
                "protocol.json pin (MPS retraining is not bit-identical): the protocol gate admits only the frozen "
                "record's checkpoints; `git checkout -- results/` restores the audit record")
DEFERRED_RE = re.compile(r"once (\S+) exists")
# a committed file that is one step's evidence but a later step's output: the after-check of the step that writes it
# verifies it (fewshot-bases' slots are recorded in the runs manifest that fewshot-continued writes; final-reports
# writes the verdict without --fewshot, and the committed verdict, with --fewshot, is fewshot-reports' output)
LATER_OUTPUT = {RUNS: "fewshot-continued", VERDICT: "fewshot-reports"}


def capture_command(cmd: Sequence[str], root: str) -> str:
    """The last non-empty stdout line of one command tuple run in root without a shell (a --print-run-id command);
    its stderr passes through. SystemExit naming the command when it exits non-zero."""
    sys.stdout.flush()
    r = subprocess.run(command_argv(cmd), cwd=root, stdout=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit("`%s` exited %d; it prints the run id the later commands need" % (display(cmd), r.returncode))
    lines = [l.strip() for l in r.stdout.decode("utf-8", "replace").splitlines() if l.strip()]
    return lines[-1] if lines else ""


def committed_bytes(root: str, rel: str) -> bytes | None:
    """The bytes git HEAD holds for the repo-relative file rel in root, or None when git is missing, root is not the
    top of a work tree, there is no HEAD or HEAD does not track rel as a file."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or top.returncode != 0:
        return None
    if os.path.realpath(top.stdout.decode("utf-8", "surrogateescape").strip()) != os.path.realpath(root):
        return None
    r = _git(root, "cat-file", "blob", "HEAD:" + rel)
    return r.stdout if r is not None and r.returncode == 0 else None


def head_holds(root: str, rel: str) -> bool:
    """True when git HEAD tracks the repo-relative file rel in root (root the top of a work tree), without reading
    the blob; False without git, outside a work tree's top, without a HEAD or for a path HEAD does not hold as a file."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or top.returncode != 0:
        return False
    if os.path.realpath(top.stdout.decode("utf-8", "surrogateescape").strip()) != os.path.realpath(root):
        return False
    r = _git(root, "cat-file", "-t", "HEAD:" + rel)
    return r is not None and r.returncode == 0 and r.stdout.strip() == b"blob"


def _held_refusal(step: Step, held: list[str]) -> str:
    """Full mode's refusal of a todo step whose committed artifacts git HEAD holds (deleted from the working tree)."""
    gated = (" the gated dumps are the exactly-once audit record and are never rescored;" if step.get("gated")
             else "")
    return ("step %s is todo, but git HEAD holds %d of its committed artifacts, missing from the working tree (first "
            "%s): a deleted committed record is never work to redo, and full mode reruns no step whose "
            "output HEAD holds;%s restore them with `git checkout -- %s`"
            % (step["name"], len(held), held[0], gated, " ".join(held)))


def _writer(name: str, item: Item, order: list[str]) -> str | None:
    """The later step (a name in order, after name) that writes a file of item, or None: the item is the evidence of
    step name, but its committed bytes are that later step's output (LATER_OUTPUT)."""
    for f in item["files"]:
        w = LATER_OUTPUT.get(f)
        if w and w != name and w in order and name in order and order.index(w) > order.index(name):
            return w
    return None


def after_items(step: Step, root: str, proto: dict[str, Any], pin: str) -> list[tuple[str, Item]]:
    """[(step name, item)] that after_check verifies once step ran: the step's own evidence items, except those whose
    committed file a later step writes (LATER_OUTPUT: fewshot-bases' slots live in the runs.json that
    fewshot-continued writes; the verdict, whose committed form fewshot-reports writes), plus the deferred items of
    earlier steps whose file this step writes and the step's outputs that are no step's evidence (the memo)."""
    manifest, order = os.path.join(root, "data", "splits.json"), [s["name"] for s in STEPS]
    name, out = step["name"], []
    for s in STEPS:
        if s["name"] == name:
            out += [(name, it) for it in s["evidence"](root, proto, pin, manifest) if _writer(name, it, order) is None]
        elif name in order and s["name"] in order and order.index(s["name"]) < order.index(name):
            out += [(s["name"], it) for it in s["evidence"](root, proto, pin, manifest)
                    if _writer(s["name"], it, order) == name]
    if all(s["name"] != name for s in STEPS):                # a step outside the registry: its own evidence only
        out = [(name, it) for it in step["evidence"](root, proto, pin, manifest)]
    return out + [(name, _file(root, rel, proto)) for rel in step.get("outputs", ())]


def after_check(step: Step, root: str, proto: dict[str, Any], pin: str) -> None:
    """After a step ran: the items of after_items (its own outputs, never a file a later step writes), every item
    present and verified (the readers and the protocol.json pins of the evidence functions), and every file of each
    item (a dump's npz and sidecar) equal to the bytes git HEAD holds for it when HEAD tracks it; SystemExit (one
    line, the RETRAIN_STOP reason) at the first failure. A step whose every evidence item is a later step's output (the
    masked bases, recorded by fewshot-continued's runs manifest) is verified with that step."""
    name = step["name"]
    manifest = os.path.join(root, "data", "splits.json")
    if not step["evidence"](root, proto, pin, manifest):
        raise SystemExit("step %s ran, but it left no evidence to verify; %s" % (name, RETRAIN_STOP))
    for _, it in after_items(step, root, proto, pin):
        if not it["present"]:
            raise SystemExit("step %s ran, but %s is missing; %s" % (name, it["label"], RETRAIN_STOP))
        if not it["ok"]:
            raise SystemExit("step %s ran, but %s; %s" % (name, it["reason"], RETRAIN_STOP))
        for rel in it["files"]:
            want = committed_bytes(root, rel)
            if want is None:
                continue
            with open(os.path.join(root, rel), "rb") as f:
                have = f.read()
            if have != want:
                raise SystemExit("step %s ran, but %s differs from the bytes git HEAD holds for it; %s"
                                 % (name, rel, RETRAIN_STOP))


def _gate_refusal(name: str) -> str:
    """Full mode's refusal of a todo gated step whose training steps are done (it would rescore around them)."""
    return ("%s has no committed gated dumps while the steps that train its checkpoints are done: the gated dumps are "
            "the exactly-once audit record and are never rescored; restore them with `git checkout -- results/dumps/`"
            % name)


def _substitute(cmd: Cmd, rid: str) -> Cmd:
    """cmd with every RUN_ID placeholder replaced by the resolved run id rid."""
    return tuple(a.replace(RUN_ID, rid) for a in cmd)


def run_full(root: str = REPO_ROOT, run: Runner = run_command,
             capture: Callable[[Sequence[str], str], str] = capture_command, blocks: Blocks = render_blocks,
             report: Callable[..., None] = report_step, out: Out = print, verify: bool = False,
             checkpoint_root: str | None = None) -> None:
    """Full mode: plan(root) (its fail-closed refusals), then every todo step in STEP_NAMES order, each
    step's commands evaluated just before it runs, then the report step. A done step is skipped; a todo step any of
    whose evidence files git HEAD holds is refused before any command (a deleted committed record, restored with
    `git checkout -- PATH`, never rerun); a todo gated step runs only after the steps that train its checkpoints
    (GATE_PRODUCERS) ran in this invocation, else it is refused before any command; DISCLOSURE prints once before the
    first model step; a selection run id (RUN_ID) is resolved
    with the step's --print-run-id command; the first non-zero exit stops the run naming the step and the command;
    after_check stops it at the first output that differs from the committed record or a protocol pin."""
    rows = plan(root)
    proto, pin = protocol.require_frozen(os.path.join(root, "protocol.json"))
    # fail closed: a todo step whose evidence git HEAD holds is a deleted committed record,
    # never work to redo; a gated step's dumps are scored once, so its rerun never depends on retraining differing
    for step, st, items in rows:
        if st != "todo" or step["name"] in LOCAL + ALWAYS:
            continue
        held = [f for it in items for f in it["files"] if head_holds(root, f)]
        if held:
            raise SystemExit(_held_refusal(step, held))
    todo = {step["name"] for step, st, _ in rows if st == "todo"}
    for step, st, _ in rows:                          # a gated step is never rerun around its own training
        if st == "todo" and not all(p in todo for p in GATE_PRODUCERS.get(step["name"], ())):
            raise SystemExit(_gate_refusal(step["name"]))
    # every output is compared with git HEAD, so a committed path that already differs from it (other than the
    # evidence of a todo step) is refused before any command, and so is a root without git
    status = tree_status(root)
    if status is None:
        raise SystemExit(NO_GIT % ("full mode", root))
    todo_files = {f for step, st, items in rows if st == "todo" for it in items for f in it["files"]}
    dirty = sorted(p for p, xy in status.items() if xy != UNTRACKED and p not in todo_files)
    if dirty:
        raise SystemExit("full mode compares every output with git HEAD, but these committed files already differ "
                         "from it: %s; commit or restore them with `git checkout -- PATH` first" % ", ".join(dirty))
    if verify:
        _print_verify(root, checkpoint_root, out)
    ran, disclosed = [], False
    for step, st, _ in rows:
        name = step["name"]
        if name in ALWAYS:
            continue
        if st == "done":
            out("skip %s (done)" % name)
            continue
        if not all(p in ran for p in GATE_PRODUCERS.get(name, ())):
            raise SystemExit(_gate_refusal(name))
        if step["model"] and not disclosed:
            out(DISCLOSURE)
            disclosed = True
        lines = step_lines(name, root)
        for cmd, note in lines:
            if note and note.startswith("deferred"):
                m = DEFERRED_RE.search(note)
                raise SystemExit("%s needs %s, which the earlier step did not produce; %s"
                                 % (name, m.group(1) if m else "a committed record", RETRAIN_STOP))
        cmds = [c for c, _ in lines if c is not None]
        if any(RUN_ID in a for c in cmds for a in c):
            if not cmds or cmds[0][-1:] != ("--print-run-id",):
                raise SystemExit("step %s holds %s but its first command is not a --print-run-id command"
                                 % (name, RUN_ID))
            out("run: " + display(cmds[0]))
            rid = capture(cmds[0], root)
            if not rid or RUN_ID in rid or not re.fullmatch(r"[A-Za-z0-9._-]+", rid):
                raise SystemExit("step %s: `%s` printed no run id (%r)" % (name, display(cmds[0]), rid))
            out("run id: " + rid)
            cmds = [_substitute(c, rid) for c in cmds[1:]]
        for cmd in cmds:
            out("run: " + display(cmd))
            rc = run(cmd, root)
            if rc != 0:
                raise SystemExit("step %s: `%s` exited %d; a protocol gate refusal printed above is final: a "
                                 "from-scratch run stops there; otherwise fix the cause and rerun `make reproduce`, "
                                 "which skips every finished step" % (name, display(cmd), rc))
        after_check(step, root, proto, pin)
        ran.append(name)
    report(root, run, blocks, precheck=not [n for n in ran if n not in LOCAL], out=out)


def main(argv: list[str] | None = None) -> str:
    """`python -m pilot.reproduce --plan`: print the step plan of the working tree and return its text."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="print the step plan of the working tree")
    a = ap.parse_args(argv)
    if not a.plan:
        ap.error("pass --plan")
    rows = plan()
    _, pin = protocol.require_frozen()
    text = render_plan(rows, pin)
    print(text, end="")
    return text


if __name__ == "__main__":
    main()
