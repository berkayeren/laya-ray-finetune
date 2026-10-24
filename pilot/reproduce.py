"""The step registry of the one-command reproduction (ENV-03, Phase 10): every pipeline step, its committed evidence,
its done/todo state and the printed plan.

D-01: the steps run in one fixed order, STEP_NAMES: data, baselines, laya-selection, laya-postfit, laya-final,
berturk-selection, berturk-postfit, berturk-reference, berturk-final, confidence, latency, final-scoring,
final-reports, fewshot-projection, fewshot-bases, fewshot-continued, fewshot-scoring, fewshot-reports, report.

D-02: a step is done only by committed, hash-verified artifacts read with the existing readers: the split and audit
verifiers of pilot.data, pilot.dumps.load_dump (with the gate keys of a gated dump), the protocol.json input pins
(the sha256 of a file's exact bytes == protocol["inputs"][path]), pilot.postfit.read_record,
pilot.latency.read_final_manifest and load_reports, pilot.verdict.check_record, pilot.fewshot.read_mode and read_runs,
and pilot.fewshot_report.read_record. Every evidence path lies under data/ or results/; a checkpoint (ckpt*), Ray
storage (ray_results/), BERTurk run storage (berturk_runs/) or a run dir is never consulted. So on a finished tree
every step through scoring is done and only the report step runs. Each protocol.json input is the evidence of
exactly one step.

D-03: a partial, tampered or out-of-order tree fails closed, in one line, before anything runs, and is never half-run
or recomputed: a step with none of its artifacts is todo, with all of them verified done, with some missing refused
naming the missing paths and the fix; a present artifact that does not verify is refused with its reader's own
refusal (or the protocol pin mismatch naming both sha256 values) and the `git checkout -- PATH` restore fix; a dump
whose npz or sidecar alone exists (an orphan) is refused naming both paths; and once a step is todo, a later step
with any committed artifact is refused naming both steps, so a deleted record is never treated as work to redo. The
data step is local (data/splits/ and data/audit/ are gitignored, written by `make data`) and exempt from the order
rule; the report step always runs.

Torch-free: the stdlib, pilot.protocol and pilot.data at import time; numpy, scipy and the record readers are imported
inside the evidence functions, so computing the plan of a tree never imports torch, transformers, laya or Ray.

Eval-only (D-05, D-06; `make report`, reproduce.py --mode eval-only): eval_only verifies first (plan(): the frozen
protocol, every split and manifest hash, every committed record and dump through the step registry), refuses before
any command a missing record or dump, naming its path and the step that makes it (eval-only never runs a todo step
but the local data step, `make data`, run only when the local split files are absent), then report_step rebuilds
every report from the committed dumps (REBUILD: fixed argv tuples, no score.py, training script, `latency.py
--system`, --force or scoring target), rewrites the README's generated blocks from their renders (README_BLOCKS,
rewrite_blocks) and proves the tree unchanged: `git status --porcelain` over CHECKED must be empty before (else the
run is refused, so a difference is always the rebuild's) and after (else the run fails naming every differing path),
with a sha256 snapshot of CHECKED in place of git when root is not a git work tree. Every command is an argv list run
without a shell by the interpreter running this module (make targets get PYTHON=sys.executable).
verify_checkpoints (--verify-checkpoints) re-hashes the on-disk final seeds with score.py's identity functions against
their final manifests: reported, not gated (it loads no model, writes nothing and never refuses).

usage:
    python -m pilot.reproduce --plan          # the 19-step plan of the working tree, each step done, todo or run
    python reproduce.py --mode eval-only      # = make report: verify, rebuild every report, the memo and the README
"""
import argparse
import json
import os
import re
import subprocess
import sys

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
CONFIDENCE_SCHEMA = "confidence/1"
MAKEFILE_VAR_RE = re.compile(r"^([A-Z_][A-Z0-9_]*) := (.*)$")


def _one_line(msg):
    return " ".join(str(msg).split())


def makefile_vars(root):
    """{NAME: value} of the `NAME := value` lines of root/Makefile; SystemExit when it is missing."""
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


def _item(rel, present, ok, reason, files=None, label=None):
    """One evidence item: its display path, the committed files a restore names, present, verified, why not."""
    return {"path": rel, "files": list(files or [rel]), "present": present, "ok": ok, "reason": reason,
            "label": label or rel}


def _check(fn, *args):
    """None when fn(*args) returns, else its SystemExit message on one line."""
    try:
        fn(*args)
    except SystemExit as e:
        return _one_line(e.code if e.code is not None else "refused")
    return None


def _file(root, rel, proto, check=None):
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


def _dump(root, run_id, split, manifest, proto, keys=None):
    """The item of one dump results/dumps/<run_id>/<split>.{npz,json}: present when either file exists; verified when
    a pinned sidecar hashes to its protocol.json pin, load_dump passes and every `keys` item equals the sidecar's."""
    from pilot import dumps
    rel = "results/dumps/%s/%s" % (run_id, split)
    npz, js = rel + ".npz", rel + ".json"
    have = [os.path.lexists(os.path.join(root, r)) for r in (npz, js)]

    def item(present, ok, reason):
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


def _json_schema(schema):
    """A check: the file is a JSON object, of `schema` when given."""
    def check(path):
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

def ev_data(root, proto, pin, manifest):
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


def ev_baselines(root, proto, pin, manifest):
    items = [_dump(root, rid, split, manifest, proto) for rid in ("majority", "prior", protocol.ZEROSHOT)
             for split in ("val", "test")]
    return items + [_file(root, "results/reports/%s.json" % rid, proto, _json_schema(None))
                    for rid in ("majority", "prior", protocol.ZEROSHOT)]


def _selection(var):
    def ev(root, proto, pin, manifest):
        return [_dump(root, rid, "val", manifest, proto) for rid in makefile_vars(root)[var].split()]
    return ev


def _postfit(kind):
    def check(path):
        from pilot import postfit
        postfit.read_record(path)

    def ev(root, proto, pin, manifest):
        return [_file(root, "results/postfit/%s.json" % kind, proto, check)]
    return ev


def _final(kind):
    def check(path, proto):
        from pilot import latency
        m, _ = latency.read_final_manifest(path, kind)
        ids = [s.get("dump_run_id") for s in m["seeds"]]
        want = proto["systems"][kind]["dump_run_ids"]
        if ids != want:
            raise SystemExit("%s lists the final dump run ids %s, the frozen protocol's are %s" % (path, ids, want))

    def ev(root, proto, pin, manifest):
        return [_file(root, "results/postfit/%s-final.json" % kind, proto, lambda p: check(p, proto))]
    return ev


def ev_berturk_reference(root, proto, pin, manifest):
    return [_file(root, "results/postfit/berturk-reference.json", proto, _json_schema("berturk-reference/1"))]


def ev_confidence(root, proto, pin, manifest):
    v = makefile_vars(root)
    items = []
    for rid in (v["LAYA_CHOSEN"], v["BERTURK_CHOSEN"], v["LAYA_ZS"]):
        items.append(_file(root, "results/confidence/%s-val.json" % rid, proto, _json_schema(CONFIDENCE_SCHEMA)))
        items.append(_file(root, "results/confidence/%s-val.svg" % rid, proto))
    return items


def ev_latency(root, proto, pin, manifest):
    from pilot import latency
    items = [_file(root, "results/latency/%s-%s.json" % (kind, dev), proto)
             for kind in protocol.KINDS for dev in ("mps", "cpu")]
    d = os.path.join(root, "results", "latency")
    present = os.path.isdir(d) and any(n.endswith(".json") for n in os.listdir(d))
    reason = _check(latency.load_reports, d) if present else None
    items.append(_item("results/latency", present, present and reason is None, reason or ""))
    return items


def _final_seeds(proto):
    return [(kind, s) for kind in protocol.KINDS for s in protocol.final_seeds(proto, kind)]


def ev_final_scoring(root, proto, pin, manifest):
    items = [_dump(root, s["dump_run_id"], split, manifest, proto, {"protocol_sha256": pin})
             for _, s in _final_seeds(proto) for split in ("test", "hotels_probe")]
    return items + [_dump(root, protocol.ZEROSHOT, "hotels_probe", manifest, proto)]


def confidence_names(proto):
    """[(record name, with an SVG)] of the 15 committed results/final/confidence records: every final seed on test
    (with its reliability SVG) and hotels_probe, zero-shot Laya on both, the prior on test."""
    out = [("%s-%s" % (s["dump_run_id"], split), split == "test") for _, s in _final_seeds(proto)
           for split in ("test", "hotels_probe")]
    return out + [("%s-test" % protocol.ZEROSHOT, True), ("%s-hotels_probe" % protocol.ZEROSHOT, False),
                  ("prior-test", True)]


def ev_final_reports(root, proto, pin, manifest):
    def verdict(path):
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


def ev_fewshot_projection(root, proto, pin, manifest):
    from pilot import fewshot
    return [_file(root, "results/fewshot/projection.json", proto, lambda p: fewshot.read_mode(pin, p))]


def read_runs(root, proto, pin):
    """(runs manifest, sha256) of root's results/fewshot/runs.json through pilot.fewshot.read_runs, bound to root's
    own D-08 decision record (pilot.fewshot.DECISION_PATH is read at call time, so it points at root's for the call),
    or (None, None) when the file does not exist."""
    from pilot import fewshot
    saved = fewshot.DECISION_PATH
    fewshot.DECISION_PATH = os.path.join(root, "results", "fewshot", "projection.json")
    try:
        return fewshot.read_runs(proto, pin, os.path.join(root, RUNS))
    finally:
        fewshot.DECISION_PATH = saved


def _runs_item(root, proto, pin):
    """(the runs.json item, runs manifest or None, its sha256 or None)."""
    out = {}

    def check(path):
        out["runs"], out["sha"] = read_runs(root, proto, pin)
    item = _file(root, RUNS, proto, check)
    return item, out.get("runs") if item["ok"] else None, out.get("sha") if item["ok"] else None


def _fewshot_stage(stage):
    def ev(root, proto, pin, manifest):
        from pilot import fewshot
        item, runs, _ = _runs_item(root, proto, pin)
        if runs is None:
            return [item]
        listed = {(e["kind"], e["stage"], e["seed"], e["draw"], e["k"]) for e in runs["entries"]}
        slots = [s for s in fewshot.slots(proto["fewshot"], runs["mode"]) if s[1] == stage]
        return [item] + [_item(RUNS, s in listed, s in listed, "", label="%s (%s)" % (RUNS, fewshot.slot_name(s)))
                         for s in slots]
    return ev


def ev_fewshot_scoring(root, proto, pin, manifest):
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


def ev_fewshot_reports(root, proto, pin, manifest):
    def check(path):
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


def ev_report(root, proto, pin, manifest):
    return []


def _step(name, evidence, fix):
    return {"name": name, "model": name in MODEL_STEPS, "gated": name in GATED, "evidence": evidence, "fix": fix}


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
          "results/fewshot/projection.json`; the D-08 decision is recorded once"),
    _step("fewshot-bases", _fewshot_stage("base"), "restore it with `git checkout -- results/fewshot/runs.json`"),
    _step("fewshot-continued", _fewshot_stage("continued"), "restore it with `git checkout -- "
          "results/fewshot/runs.json`"),
    _step("fewshot-scoring", ev_fewshot_scoring, "restore the gated dumps with `git checkout -- results/dumps/`; a "
          "gated split is scored once and never rescored"),
    _step("fewshot-reports", ev_fewshot_reports, "restore it with `git checkout -- results/fewshot/fewshot.json`"),
    _step("report", ev_report, "it always runs"),
]


def _restore(step, item):
    if step["name"] in LOCAL and not item.get("anchor"):
        return "regenerate it with `make data`"
    return "restore the committed file with `git checkout -- %s`" % " ".join(item["files"])


def state(step, items):
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


def plan(root=REPO_ROOT):
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


def render_plan(rows, pin):
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


# ---- eval-only (D-05, D-06): rebuild every report from the committed dumps, no training, no scoring ----

CHECKED = ("results/", "README.md", "protocol.json")       # the audit record a rebuild must leave byte for byte
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
MODES = ("eval-only",)
README = "README.md"


def _show(cmd):
    return " ".join(cmd)


def command_argv(cmd):
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


def run_command(cmd, root):
    """Run one command tuple in root, without a shell; its exit code."""
    sys.stdout.flush()
    return subprocess.run(command_argv(cmd), cwd=root).returncode


def _git(root, *args):
    try:
        return subprocess.run(("git",) + args, cwd=root, capture_output=True)
    except OSError:
        return None


def tree_changes(root):
    """The sorted repo-relative paths `git status --porcelain --untracked-files=all` reports under CHECKED in root
    (modified, deleted, added or untracked), or None when git is missing or root is not the top of a work tree."""
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
    tokens, out, i = st.stdout.split(b"\0"), set(), 0
    while i < len(tokens):
        entry = tokens[i]
        i += 1
        if len(entry) < 4:
            continue
        out.add(entry[3:].decode("utf-8", "surrogateescape"))
        if entry[:1] in (b"R", b"C"):                          # -z: a rename's source path follows its target
            i += 1
    return sorted(out)


def snapshot(root):
    """{repo-relative path: sha256} of every file under CHECKED in root: the comparison without git."""
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


def _snapshot_diff(before, after):
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def rewrite_blocks(text, renders):
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


def render_blocks(root):
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


def report_step(root=REPO_ROOT, run=run_command, blocks=render_blocks, precheck=True, out=print):
    """The report step: every REBUILD command, then the README blocks, on a tree whose CHECKED paths equal the
    committed ones; SystemExit when they already differ (precheck, before any command), when a command fails, or when
    a rebuilt file differs from the committed one afterwards (naming every such path)."""
    before_git = tree_changes(root)
    before_snap = None
    if before_git is None or not precheck:
        before_snap = snapshot(root)                 # without git (or with pre-existing changes): compare contents
    if before_git and precheck:
        raise SystemExit("eval-only compares every rebuilt file with the committed one, but these already differ: %s; "
                         "commit or restore them first" % ", ".join(before_git))
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
    after_git = tree_changes(root)
    changed = set()
    if after_git is not None:
        changed |= set(after_git) - set(before_git or [])
    if before_snap is not None:
        changed |= set(_snapshot_diff(before_snap, snapshot(root)))
    changed = sorted(changed)
    if changed:
        raise SystemExit("eval-only rebuilt %d file(s) that differ from the committed ones: %s; the committed results "
                         "are the audit record: inspect with `git diff`, restore with `git checkout -- PATH`"
                         % (len(changed), ", ".join(changed)))
    out("eval-only done: %d rebuild commands and %d README blocks left results/, README.md and protocol.json "
        "byte-identical to the committed ones (%s)"
        % (len(REBUILD), len(renders), "git status clean" if after_git is not None else "sha256 snapshot unchanged"))


VERIFY_LINE = "verify-checkpoints (reported, not gated): "
ABSENT_DETAIL = "not needed: eval-only rebuilds from the committed dumps"


def _identity_diff(have, want):
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


def verify_checkpoints(root=REPO_ROOT, checkpoint_root=None, manifests=None):
    """D-06, reported, not gated: [{kind, seed, checkpoint_name, status, detail}] of every final seed listed in
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


def _missing_refusal(step, items):
    gone = [it for it in items if not it["present"]]
    if not gone:
        return "eval-only never trains or scores: step %s has no committed artifact; %s" % (step["name"], step["fix"])
    it = gone[0]
    return ("eval-only never trains or scores: %s is missing; it is made by step %s; restore the committed record "
            "with `git checkout -- %s`" % (it["path"], step["name"], " ".join(it["files"])))


def eval_only(root=REPO_ROOT, run=run_command, blocks=render_blocks, out=print, verify=False, checkpoint_root=None):
    """Eval-only (D-05): verify the tree through plan(), refuse before any command a step other than data that is
    not done (its first missing path and the step that makes it), run `make data` only when the local split files are
    absent, then (verify: the verify_checkpoints rows first, reported, not gated) report_step."""
    rows = plan(root)
    for step, st, items in rows:
        if step["name"] not in LOCAL + ALWAYS and st != "done":
            raise SystemExit(_missing_refusal(step, items))
    if verify:
        for r in verify_checkpoints(root, checkpoint_root):
            out("%s%s seed %s %s: %s%s" % (VERIFY_LINE, r["kind"], r["seed"], r["checkpoint_name"], r["status"],
                                          " (%s)" % r["detail"] if r["detail"] else ""))
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


def main(argv=None):
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
