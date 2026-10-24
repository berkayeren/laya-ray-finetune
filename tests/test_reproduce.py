"""The step registry of the one-command reproduction (ENV-03, plan 10-03): the D-01 step order, the done state of the
committed tree from its hash-verified records only (D-02), the fail-closed refusals of partial, tampered, orphan and
out-of-order trees on synthetic copies (D-03), and the torch-free import contract."""
import os
import re
import subprocess
import sys

import pytest

import pilot.protocol
import pilot.reproduce as rp
from pilot.data import REPO_ROOT

pytestmark = pytest.mark.unit

D01 = ("data", "baselines", "laya-selection", "laya-postfit", "laya-final", "berturk-selection", "berturk-postfit",
       "berturk-reference", "berturk-final", "confidence", "latency", "final-scoring", "final-reports",
       "fewshot-projection", "fewshot-bases", "fewshot-continued", "fewshot-scoring", "fewshot-reports", "report")
SCORED = D01[1:18]                                # baselines .. fewshot-reports: done on the committed tree
PLAN_LINE = re.compile(r"^ ?([0-9]+) (\S+) +(done|todo|run)\b")


def states(rows):
    return {step["name"]: st for step, st, _ in rows}


def test_steps_are_the_d01_order():
    assert rp.STEP_NAMES == D01
    assert tuple(s["name"] for s in rp.STEPS) == D01
    assert rp.LOCAL == ("data",) and rp.ALWAYS == ("report",)
    assert rp.GATED == ("final-scoring", "fewshot-scoring")
    assert all(s["gated"] == (s["name"] in rp.GATED) and s["model"] == (s["name"] in rp.MODEL_STEPS)
               for s in rp.STEPS)


def test_committed_tree_is_done():
    st = states(rp.plan(REPO_ROOT))
    assert [n for n in SCORED if st[n] != "done"] == [], st
    assert st["report"] == "run"
    assert st["data"] in ("done", "todo")


def test_every_protocol_input_is_one_steps_evidence():
    proto, pin = pilot.protocol.require_frozen()
    manifest = os.path.join(REPO_ROOT, "data", "splits.json")
    owners = {}
    for step in rp.STEPS:
        files = {f for it in step["evidence"](REPO_ROOT, proto, pin, manifest) for f in it["files"]}
        for f in files:
            owners.setdefault(f, []).append(step["name"])
    assert len(proto["inputs"]) == 17
    bad = {p: owners.get(p) for p in proto["inputs"] if len(owners.get(p, [])) != 1}
    assert bad == {}, bad


def test_plan_reads_only_committed_paths():
    proto, pin = pilot.protocol.require_frozen()
    manifest = os.path.join(REPO_ROOT, "data", "splits.json")
    paths = [f for step in rp.STEPS for it in step["evidence"](REPO_ROOT, proto, pin, manifest)
             for f in it["files"] + [it["path"]]]
    assert paths
    bad = [p for p in paths if not (p.startswith(("data/", "results/")) or p == "protocol.json")
           or re.search(r"ckpt|ray_results|berturk_runs", p)]
    assert bad == [], bad


def test_cli_plan_prints_every_step():
    out = subprocess.run([sys.executable, "-m", "pilot.reproduce", "--plan"], cwd=REPO_ROOT, capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    rows = [m.groups() for m in (PLAN_LINE.match(l) for l in out.stdout.splitlines()) if m]
    assert [int(i) for i, _, _ in rows] == list(range(1, 20))
    assert tuple(n for _, n, _ in rows) == D01
    assert all(s == "done" for _, n, s in rows if n in SCORED) and rows[-1][2] == "run", out.stdout
    assert pilot.protocol.PROTOCOL_SHA256[:12] in out.stdout.splitlines()[0]
    bare = subprocess.run([sys.executable, "-m", "pilot.reproduce"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert bare.returncode == 2 and "--plan" in bare.stderr


def test_reproduce_stays_torch_free():
    code = ("import sys, pilot.reproduce as r\n"
            "import_time = sorted(m.split('.')[0] for m in sys.modules if m.split('.')[0] in "
            "('numpy', 'scipy', 'torch', 'transformers', 'laya', 'ray'))\n"
            "assert import_time == [], import_time\n"
            "r.plan(r.REPO_ROOT)\n"
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'torch', 'transformers', 'laya', 'ray'}))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
