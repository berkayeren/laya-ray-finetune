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


# ---- synthetic trees (tmp copies; the committed tree is never touched) ----

@pytest.fixture
def tree(tmp_path):
    """A copy of the committed record (protocol.json, Makefile, data/splits.json, results/) without the local split
    and audit files."""
    import shutil
    for rel in ("protocol.json", "Makefile", "data/splits.json"):
        os.makedirs(os.path.dirname(os.path.join(tmp_path, rel)) or str(tmp_path), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel), os.path.join(tmp_path, rel))
    shutil.copytree(os.path.join(REPO_ROOT, "results"), os.path.join(tmp_path, "results"))
    return str(tmp_path)


def refusal(root):
    with pytest.raises(SystemExit) as e:
        rp.plan(root)
    msg = str(e.value.code)
    assert "\n" not in msg, msg
    return msg


def first_final_seed():
    proto, _ = pilot.protocol.require_frozen()
    return pilot.protocol.final_seeds(proto, "laya")[0]["dump_run_id"]


def test_state_todo_tail_is_accepted(tree):
    import json
    import shutil
    with open(os.path.join(tree, "results", "fewshot", "runs.json"), encoding="utf-8") as f:
        ids = {e["dump_run_id"] for e in json.load(f)["entries"]}
    assert len(ids) == 18
    shutil.rmtree(os.path.join(tree, "results", "fewshot"))
    for rid in ids:
        shutil.rmtree(os.path.join(tree, "results", "dumps", rid))
    st = states(rp.plan(tree))
    tail = D01[13:18]
    assert [n for n in tail if st[n] != "todo"] == [], st
    assert [n for n in D01[1:13] if st[n] != "done"] == [], st


def test_state_all_todo_without_results(tree):
    import shutil
    shutil.rmtree(os.path.join(tree, "results"))
    st = states(rp.plan(tree))
    assert [n for n in SCORED if st[n] != "todo"] == [], st
    assert st["report"] == "run" and st["data"] == "todo"


def test_state_refuses_a_partly_done_step(tree):
    rid = first_final_seed()
    for ext in ("npz", "json"):
        os.remove(os.path.join(tree, "results", "dumps", rid, "test." + ext))
    msg = refusal(tree)
    assert msg.startswith("final-scoring is partly done: 12 of 13 artifacts present"), msg
    assert "results/dumps/%s/test" % rid in msg and "git checkout" in msg, msg


def test_state_refuses_a_tampered_dump(tree):
    from pilot import dumps
    rid = rp.makefile_vars(REPO_ROOT)["LAYA_SELECTION"].split()[0]
    npz = os.path.join(tree, "results", "dumps", rid, "val.npz")
    with open(npz, "rb") as f:
        raw = bytearray(f.read())
    raw[len(raw) // 2] ^= 0xFF
    with open(npz, "wb") as f:
        f.write(bytes(raw))
    with pytest.raises(SystemExit) as e:
        dumps.load_dump(os.path.dirname(npz), "val", os.path.join(tree, "data", "splits.json"))
    msg = refusal(tree)
    assert msg.startswith("laya-selection: "), msg
    assert " ".join(str(e.value.code).split()) in msg, msg
    assert msg.endswith("`git checkout -- results/dumps/%s/val.npz results/dumps/%s/val.json`" % (rid, rid)), msg


def test_state_refuses_a_tampered_pinned_record(tree):
    from pilot.data import sha256_file
    path = os.path.join(tree, "results", "postfit", "laya.json")
    with open(path, "ab") as f:
        f.write(b"\n")
    proto, _ = pilot.protocol.require_frozen()
    msg = refusal(tree)
    assert msg.startswith("laya-postfit: results/postfit/laya.json sha256 %s" % sha256_file(path)), msg
    assert proto["inputs"]["results/postfit/laya.json"] in msg, msg
    assert msg.endswith("`git checkout -- results/postfit/laya.json`"), msg


def test_state_refuses_an_orphan_gated_npz(tree):
    rid = first_final_seed()
    os.remove(os.path.join(tree, "results", "dumps", rid, "test.json"))
    msg = refusal(tree)
    assert msg.startswith("final-scoring: "), msg
    assert "results/dumps/%s/test.npz" % rid in msg and "results/dumps/%s/test.json" % rid in msg, msg
    assert "orphan" in msg and "score.py" not in msg, msg
    assert msg.endswith("`git checkout -- results/dumps/%s/test.npz results/dumps/%s/test.json`" % (rid, rid)), msg


def test_state_refuses_a_step_done_after_a_todo_step(tree):
    import shutil
    shutil.rmtree(os.path.join(tree, "results", "confidence"))
    msg = refusal(tree)
    assert msg.startswith("step latency is done but the earlier step confidence is not"), msg
    assert "results/confidence/" in msg and "git checkout --" in msg, msg


def test_data_step_is_local(tree):
    st, want = states(rp.plan(tree)), states(rp.plan(REPO_ROOT))
    assert st["data"] == "todo"                    # no data/splits/: todo, exempt from the order rule
    assert {n: s for n, s in st.items() if n != "data"} == {n: s for n, s in want.items() if n != "data"}
    os.makedirs(os.path.join(tree, "data", "splits"))
    with open(os.path.join(tree, "data", "splits", "val.jsonl"), "wb") as f:
        f.write(b'{"uid": "x", "state": "y"}\n'[:-1] + b"!")
    msg = refusal(tree)
    assert msg.startswith("data: ") and "registered in" in msg, msg
    assert msg.endswith("regenerate with `make data`") and msg.count("make data") == 1, msg
