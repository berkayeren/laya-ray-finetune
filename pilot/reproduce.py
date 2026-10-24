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

usage:
    python -m pilot.reproduce --plan          # the 19-step plan of the working tree, each step done, todo or run
"""
import argparse
import json
import os
import re

from pilot import protocol
from pilot.data import AUDIT_FILES, REPO_ROOT, SPLITS, sha256_bytes, verify_audit_file, verify_split_file

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
            raise SystemExit("%s: %s; %s" % (name, it["reason"], _restore(step, it)))
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
                gone = [it for it in its if not it["present"]] or its
                raise SystemExit("step %s is done but the earlier step %s is not (missing %s); the committed record is "
                                 "incomplete: restore it with `git checkout -- %s`"
                                 % (step["name"], earlier["name"], gone[0]["label"] if gone else "its artifacts",
                                    " ".join(gone[0]["files"]) if gone else "results/"))
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
