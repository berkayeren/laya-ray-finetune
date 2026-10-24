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


# ---- eval-only (plan 10-04, D-05, D-06): rebuild every report from the committed dumps, no training, no scoring ----

BANNED_SCRIPTS = ("score.py", "ray_train.py", "berturk_train.py", "compute_checkpoint.py", "mps_spike.py")
BANNED_TARGETS = ("baselines", "final-scoring", "fewshot-scoring")
START_RE = re.compile(r"<!-- (\S+):start -->")


def recipe(target):
    """The recipe lines of one Makefile target."""
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    i = lines.index(target + ":")
    out = []
    for line in lines[i + 1:]:
        if not line.startswith("\t"):
            break
        out.append(line[1:])
    return out


def test_eval_only_runs_no_model():
    cmds = list(rp.REBUILD) + [cmd for _, cmd in rp.README_BLOCKS]
    assert len(rp.REBUILD) == 9 and len(rp.README_BLOCKS) == 11
    for cmd in cmds:
        assert isinstance(cmd, tuple) and all(isinstance(c, str) for c in cmd), cmd
        assert not set(cmd) & set(BANNED_SCRIPTS), cmd
        assert "--force" not in cmd, cmd
        assert not (cmd[0] == "latency.py" and "--system" in cmd), cmd
        if cmd[0] == "make":
            assert cmd[1] not in BANNED_TARGETS, cmd
            body = " ".join(recipe(cmd[1]))             # the target's own recipe runs no model either
            assert not [s for s in BANNED_SCRIPTS if s in body] and "--force" not in body, (cmd, body)
            assert "latency.py --system" not in body and "$(MAKE)" not in body, (cmd, body)
        argv = rp.command_argv(cmd)
        assert argv[0] == sys.executable or argv[-1] == "PYTHON=" + sys.executable, argv
    for bad in (("bash", "-c", "make report"), ("make",), (), ("make report; rm -rf results",), ("make", "")):
        with pytest.raises(SystemExit):
            rp.command_argv(bad)
    assert rp.MODES == ("full", "eval-only")
    assert rp.CHECKED == ("results/", "README.md", "protocol.json")


def test_readme_blocks_cover_every_marker():
    with open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8") as f:
        names = START_RE.findall(f.read())
    assert len(names) == len(set(names)) == 11
    assert [n for n, _ in rp.README_BLOCKS] and {n for n, _ in rp.README_BLOCKS} == set(names)
    assert len(rp.README_BLOCKS) == len({n for n, _ in rp.README_BLOCKS})


def test_readme_rewriter_reproduces_the_readme():
    with open(os.path.join(REPO_ROOT, "README.md"), "rb") as f:
        raw = f.read()
    renders = rp.render_blocks(REPO_ROOT)
    assert [n for n, _ in renders] == [n for n, _ in rp.README_BLOCKS]
    assert all(body.endswith("\n") for _, body in renders)
    assert rp.rewrite_blocks(raw.decode("utf-8"), renders).encode("utf-8") == raw


def test_makefile_reproduce_and_report_targets():
    """make reproduce is the one target (MODE ?= full); make report is its eval-only form, recursing into it (D-05)."""
    assert recipe("reproduce") == ["$(PYTHON) reproduce.py --mode $(MODE) "
                                   "$(if $(VERIFY_CHECKPOINTS),--verify-checkpoints,)"]
    assert recipe("report") == ['$(MAKE) reproduce MODE=eval-only PYTHON="$(PYTHON)"']
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        text = f.read()
    assert re.findall(r"^MODE \?= (.*)$", text, re.M) == ["full"]
    assert not re.search(r"^MODE :?= ", text, re.M)
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    names = phony.group(1).split() if phony else []
    assert {"reproduce", "report", "final-reports", "final-scoring", "memo"} <= set(names), names
    assert re.search(r"^#   make reproduce  the whole pipeline in order \(data, val selection, final training, "
                     r"evaluation, few-shot, report\), skipping every step its committed, hash-verified artifacts "
                     r"prove done; MODE=eval-only is make report$", text, re.M)
    assert re.search(r"^#   make report  eval-only: .*fails on any rebuilt file that differs$", text, re.M)
    out = subprocess.run(["make", "-n", "report", "PYTHON=py", "VERIFY_CHECKPOINTS=1"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "py reproduce.py --mode eval-only --verify-checkpoints" in out.stdout.split("\n"), out.stdout
    out = subprocess.run(["make", "-n", "reproduce", "PYTHON=py"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.split("\n")[0].rstrip() == "py reproduce.py --mode full", out.stdout


def _data_done(monkeypatch):
    """The data step's evidence stubbed to one verified item, so a test of the committed record needs no local
    split files."""
    step = rp.STEPS[rp.STEP_NAMES.index("data")]
    monkeypatch.setitem(step, "evidence", lambda root, proto, pin, manifest: [
        rp._item("data/splits/val.jsonl", True, True, "")])


def test_full_mode_on_the_committed_tree_runs_only_the_report(monkeypatch):
    _data_done(monkeypatch)
    calls, reports, lines = [], [], []

    def run(cmd, root):
        calls.append(cmd)
        return 0

    def capture(cmd, root):
        calls.append(cmd)
        return "x"

    def report(root, run, blocks, precheck=True, out=print):
        reports.append((root, precheck))
    rp.run_full(REPO_ROOT, run=run, capture=capture, blocks=None, report=report, out=lines.append)
    assert calls == [] and reports == [(REPO_ROOT, True)], (calls, reports)
    assert lines == ["skip %s (done)" % n for n in D01[:18]], lines
    assert rp.DISCLOSURE not in lines and not [l for l in lines if l.startswith("run: ")]


# ---- eval-only fail paths (tmp git repos and synthetic trees; the committed tree is never touched) ----

def _git_env(monkeypatch):
    """No global or system git config (commit.gpgsign, core.hooksPath, excludes) reaches a temp repo."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git_in(root, *args):
    return subprocess.run(("git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c",
                           "commit.gpgsign=false", "-c", "core.hooksPath=%s" % os.devnull) + args, cwd=root,
                          check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tmp git repo with one committed results/x.json, README.md and protocol.json."""
    _git_env(monkeypatch)
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "results" / "x.json").write_bytes(b'{"a": 1}\n')
    (root / "README.md").write_bytes(b"# r\n")
    (root / "protocol.json").write_bytes(b"{}\n")
    git_in(root, "init", "-q")
    git_in(root, "add", "-A")
    git_in(root, "commit", "-q", "-m", "init")
    return str(root)


def fake_run(write=None):
    """A run() that records each command and, on the first one, writes `write` ({rel: bytes}) under root."""
    calls = []

    def run(cmd, root):
        if write and not calls:
            for rel, data in write.items():
                with open(os.path.join(root, rel), "wb") as f:
                    f.write(data)
        calls.append(cmd)
        return 0
    return run, calls


def no_blocks(root):
    return []


def test_readme_rewriter_refuses_bad_markers():
    good = "a\n<!-- x:start -->\nold\n<!-- x:end -->\nb\n"
    assert rp.rewrite_blocks(good, [("x", "new\n")]) == "a\n<!-- x:start -->\nnew\n<!-- x:end -->\nb\n"
    assert rp.rewrite_blocks(good, [("x", "old\n")]) == good           # every body already its render: unchanged
    bad = {"missing end": "<!-- x:start -->\nold\n",
           "duplicated start": "<!-- x:start -->\n<!-- x:start -->\nold\n<!-- x:end -->\n",
           "reversed pair": "<!-- x:end -->\nold\n<!-- x:start -->\n"}
    for case, text in bad.items():
        with pytest.raises(SystemExit) as e:
            rp.rewrite_blocks(text, [("x", "new\n")])
        msg = str(e.value.code)
        assert "<!-- x:start -->" in msg and "<!-- x:end -->" in msg, (case, msg)
        assert "git checkout -- README.md" in msg and "\n" not in msg, (case, msg)


def test_tree_changes_names_each_changed_path(repo, tmp_path):
    assert rp.tree_changes(repo) == []
    with open(os.path.join(repo, "results", "x.json"), "ab") as f:
        f.write(b" ")
    with open(os.path.join(repo, "results", "new.json"), "wb") as f:
        f.write(b"{}\n")
    os.makedirs(os.path.join(repo, "other"))
    with open(os.path.join(repo, "other", "y.txt"), "wb") as f:         # outside CHECKED: never named
        f.write(b"y\n")
    assert rp.tree_changes(repo) == ["results/new.json", "results/x.json"]
    os.remove(os.path.join(repo, "README.md"))
    assert rp.tree_changes(repo) == ["README.md", "results/new.json", "results/x.json"]
    assert rp.tree_changes(os.path.join(repo, "results")) is None      # not the top of the work tree
    plain = tmp_path / "plain"
    (plain / "results").mkdir(parents=True)
    (plain / "results" / "x.json").write_bytes(b'{"a": 1}\n')
    (plain / "README.md").write_bytes(b"# r\n")
    assert rp.tree_changes(str(plain)) is None
    snap = rp.snapshot(str(plain))
    assert sorted(snap) == ["README.md", "results/x.json"]
    (plain / "results" / "x.json").write_bytes(b'{"a": 2}\n')
    assert rp.snapshot(str(plain)) != snap and rp._snapshot_diff(snap, rp.snapshot(str(plain))) == ["results/x.json"]


def test_eval_only_refuses_a_dirty_tree_before_any_command(repo):
    with open(os.path.join(repo, "README.md"), "ab") as f:
        f.write(b"edited\n")
    run, calls = fake_run()
    with pytest.raises(SystemExit) as e:
        rp.report_step(repo, run=run, blocks=no_blocks, out=lambda s: None)
    msg = str(e.value.code)
    assert calls == [], calls
    assert msg.startswith("eval-only compares every rebuilt file with the committed one, but these already differ: "
                          "README.md;"), msg
    assert "\n" not in msg


def _rebuild_case(root, with_git):
    with open(os.path.join(root, "results", "x.json"), "rb") as f:
        committed = f.read()
    run, calls = fake_run({"results/x.json": committed})                    # identical bytes: green
    lines = []
    rp.report_step(root, run=run, blocks=no_blocks, out=lines.append)
    assert calls == list(rp.REBUILD)
    assert [l for l in lines if l.startswith("eval-only: ")] == ["eval-only: " + " ".join(c) for c in rp.REBUILD]
    assert lines[-1].startswith("eval-only done: ")
    assert ("git status clean" if with_git else "sha256 snapshot unchanged") in lines[-1]
    changed = bytearray(committed)
    changed[2] ^= 0x01
    run, calls = fake_run({"results/x.json": bytes(changed)})               # one byte changed: red, naming it
    with pytest.raises(SystemExit) as e:
        rp.report_step(root, run=run, blocks=no_blocks, out=lambda s: None)
    msg = str(e.value.code)
    assert msg.startswith("eval-only rebuilt 1 file(s) that differ from the committed ones: results/x.json;"), msg
    assert "git checkout -- PATH" in msg and "\n" not in msg
    assert calls == list(rp.REBUILD)


def test_eval_only_fails_naming_a_rebuilt_file_that_differs(repo, tmp_path):
    import shutil
    nogit = str(tmp_path / "nogit")                     # the snapshot fallback: the same tree without .git
    shutil.copytree(repo, nogit, ignore=shutil.ignore_patterns(".git"))
    assert rp.tree_changes(nogit) is None
    _rebuild_case(repo, True)
    run, calls = fake_run()                             # the differing file is left: the next run is refused up front
    with pytest.raises(SystemExit):
        rp.report_step(repo, run=run, blocks=no_blocks, out=lambda s: None)
    assert calls == []
    _rebuild_case(nogit, False)


def test_eval_only_names_the_step_of_a_missing_artifact(tree):
    import json
    import shutil
    os.rename(os.path.join(tree, "results", "postfit", "laya.json"), os.path.join(tree, "laya.json.saved"))
    run, calls = fake_run()
    with pytest.raises(SystemExit) as e:
        rp.eval_only(tree, run=run, blocks=no_blocks, out=lambda s: None)
    msg = str(e.value.code)
    assert "results/postfit/laya.json" in msg and "laya-postfit" in msg and "git checkout --" in msg, msg
    assert calls == [] and "\n" not in msg
    os.rename(os.path.join(tree, "laya.json.saved"), os.path.join(tree, "results", "postfit", "laya.json"))
    with open(os.path.join(tree, "results", "fewshot", "runs.json"), encoding="utf-8") as f:
        ids = {e["dump_run_id"] for e in json.load(f)["entries"]}
    shutil.rmtree(os.path.join(tree, "results", "fewshot"))           # every step from fewshot-projection onward
    for rid in ids:
        shutil.rmtree(os.path.join(tree, "results", "dumps", rid))
    run, calls = fake_run()
    with pytest.raises(SystemExit) as e:
        rp.eval_only(tree, run=run, blocks=no_blocks, out=lambda s: None)
    msg = str(e.value.code)
    assert msg == ("eval-only never trains or scores: results/fewshot/projection.json is missing; it is made by step "
                   "fewshot-projection; restore the committed record with `git checkout -- "
                   "results/fewshot/projection.json`"), msg
    assert calls == []                                  # the data step (todo in this tree) never ran either


def _need_data():
    import glob
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")


@pytest.mark.data
@pytest.mark.parametrize("how", ["eval-only", "make reproduce"])
def test_eval_only_end_to_end(tmp_path, monkeypatch, how):
    """A fresh copy of the working tree (every tracked file, plus the local data/splits/ and data/audit/), committed
    in a new repo: reproduce.py --mode eval-only, and `make reproduce` (full mode: every step from data to
    fewshot-reports skipped as done, no command of a todo step), exit 0, run all 9 rebuild commands and leave the copy
    clean."""
    import shutil
    _need_data()
    _git_env(monkeypatch)
    copy = tmp_path / "copy"
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
    files = [p.decode("utf-8") for p in tracked.split(b"\0") if p]
    for rel in files:
        dst = copy / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(REPO_ROOT, rel), dst, follow_symlinks=False)
    for d in ("splits", "audit"):
        shutil.copytree(os.path.join(REPO_ROOT, "data", d), copy / "data" / d)
    git_in(copy, "init", "-q")
    subprocess.run(("git", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"), cwd=copy, input=tracked,
                   check=True, capture_output=True)
    git_in(copy, "commit", "-q", "-m", "copy")
    assert rp.tree_changes(str(copy)) == []
    argv = ([sys.executable, "reproduce.py", "--mode", "eval-only"] if how == "eval-only"
            else ["make", "reproduce", "PYTHON=" + sys.executable])
    out = subprocess.run(argv, cwd=copy, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    lines = out.stdout.splitlines()
    assert ("reproduce in " + sys.executable) in lines[:2], lines[:2]
    skips = [l for l in lines if l.startswith("skip ")]
    if how == "eval-only":
        assert skips == []
    else:
        assert skips == ["skip %s (done)" % n for n in D01[:18]], skips
        assert not [l for l in lines if l.startswith("run: ")] and rp.DISCLOSURE not in lines
    assert [l for l in lines if l.startswith("eval-only: ")] == ["eval-only: " + " ".join(c) for c in rp.REBUILD]
    assert "README blocks: 11 rendered, README.md unchanged" in lines
    assert lines[-1].startswith("eval-only done: ") and "git status clean" in lines[-1]
    st = subprocess.run(["git", "status", "--porcelain", "--", "results/", "README.md", "protocol.json"], cwd=copy,
                        check=True, capture_output=True, text=True)
    assert st.stdout == ""


# ---- --verify-checkpoints (D-06): the on-disk final seeds re-hashed against their manifests, reported, not gated ----

ABSENT = "not needed: eval-only rebuilds from the committed dumps"


def laya_ckpt(d, salt):
    """A synthetic Laya-like checkpoint dir: weights, config, tokenizer/ and encoder/ files."""
    files = {"model.safetensors": b"w" + salt, "rl_agent_config.json": b'{"t": 1}\n',
             "tokenizer/tokenizer_config.json": b'{"tokenizer_class": "PreTrainedTokenizerFast"}\n',
             "tokenizer/tokenizer.json": b'{"v": 1}\n', "encoder/config.json": b'{"h": 2}\n'}
    for rel, data in files.items():
        os.makedirs(os.path.dirname(os.path.join(d, rel)) or d, exist_ok=True)
        with open(os.path.join(d, rel), "wb") as f:
            f.write(data)


def berturk_ckpt(d, salt):
    """A synthetic BERTurk checkpoint dir: every pilot.berturk.CHECKPOINT_FILES entry."""
    from pilot import berturk
    os.makedirs(d)
    for name in berturk.CHECKPOINT_FILES:
        with open(os.path.join(d, name), "wb") as f:
            f.write(name.encode("utf-8") + salt)


def final_manifest(path, kind, ckpt_root, make):
    """A postfit-final/1 manifest of three seeds whose identity and dump run id are computed from their dirs."""
    import json
    from pilot import berturk, dumps, weights
    seeds = []
    for s in range(3):
        name = "ckpt-%s-s%d" % (kind, s)
        d = os.path.join(ckpt_root, name)
        make(d, b"%d" % s)
        if kind == "laya":
            ident = weights.identity(d)
            rid = dumps.run_id("laya", ident)
        else:
            ident = berturk.identity(d)
            rid = berturk.dump_run_id(ident)
        seeds.append({"seed": s, "train_run_id": "r%d" % s, "checkpoint_name": name, "checkpoint_identity": ident,
                      "dump_run_id": rid, "temperature": 1.0})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": "postfit-final/1", "system_kind": kind, "seeds": seeds}, f)
    return seeds


def make_final_seeds(tmp_path):
    ckpts = str(tmp_path / "ckpts")
    os.makedirs(ckpts)
    manifests = {}
    for kind, make in (("laya", laya_ckpt), ("berturk", berturk_ckpt)):
        manifests[kind] = str(tmp_path / ("%s-final.json" % kind))
        final_manifest(manifests[kind], kind, ckpts, make)
    return ckpts, manifests


def test_verify_checkpoints_reports_match_differs_absent(tmp_path):
    import shutil
    ckpts, manifests = make_final_seeds(tmp_path)
    rows = rp.verify_checkpoints(REPO_ROOT, ckpts, manifests=manifests)
    assert [(r["kind"], r["seed"], r["checkpoint_name"], r["status"]) for r in rows] == [
        (k, s, "ckpt-%s-s%d" % (k, s), "match") for k in ("laya", "berturk") for s in range(3)]
    assert set(rows[0]) == {"kind", "seed", "checkpoint_name", "status", "detail"}
    for kind, weights_file in (("laya", "model.safetensors"), ("berturk", "model.safetensors")):
        p = os.path.join(ckpts, "ckpt-%s-s1" % kind, weights_file)
        with open(p, "rb") as f:
            raw = bytearray(f.read())
        raw[0] ^= 0x01
        with open(p, "wb") as f:
            f.write(bytes(raw))
        shutil.rmtree(os.path.join(ckpts, "ckpt-%s-s2" % kind))
    os.remove(os.path.join(ckpts, "ckpt-berturk-s0", "vocab.txt"))         # an incomplete BERTurk dir: differs
    rows = rp.verify_checkpoints(REPO_ROOT, ckpts, manifests=manifests)
    got = {(r["kind"], r["seed"]): r for r in rows}
    assert got[("laya", 0)]["status"] == "match"
    for kind in ("laya", "berturk"):
        assert got[(kind, 1)]["status"] == "differs" and "model_sha256" in got[(kind, 1)]["detail"], got[(kind, 1)]
        assert got[(kind, 2)]["status"] == "absent" and got[(kind, 2)]["detail"] == ABSENT, got[(kind, 2)]
    assert got[("berturk", 0)]["status"] == "differs" and got[("berturk", 0)]["detail"], got[("berturk", 0)]


def test_verify_checkpoints_writes_nothing(tmp_path, monkeypatch):
    import json
    ckpts, manifests = make_final_seeds(tmp_path)
    before = (rp.snapshot(str(tmp_path)), rp.snapshot(REPO_ROOT), sorted(os.listdir(str(tmp_path))))

    def tree_files(root):
        return sorted(os.path.relpath(os.path.join(d, n), root) for d, _, fs in os.walk(root) for n in fs)
    files = tree_files(str(tmp_path))
    rp.verify_checkpoints(REPO_ROOT, ckpts, manifests=manifests)
    assert (rp.snapshot(str(tmp_path)), rp.snapshot(REPO_ROOT), sorted(os.listdir(str(tmp_path)))) == before
    assert tree_files(str(tmp_path)) == files
    code = ("import json, sys, pilot.reproduce as r\n"
            "rows = r.verify_checkpoints(r.REPO_ROOT, sys.argv[1], manifests=json.loads(sys.argv[2]))\n"
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'torch', 'transformers', 'laya', 'ray'}))\n")
    out = subprocess.run([sys.executable, "-c", code, ckpts, json.dumps(manifests)], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    # eval_only(verify=True): the rows print before the rebuild, and differs or absent never changes the outcome
    root = tmp_path / "root"
    (root / "results" / "postfit").mkdir(parents=True)
    for kind, path in manifests.items():
        with open(path, "rb") as f, open(root / "results" / "postfit" / ("%s-final.json" % kind), "wb") as g:
            g.write(f.read())
    import shutil
    shutil.rmtree(os.path.join(ckpts, "ckpt-laya-s0"))
    calls = []
    monkeypatch.setattr(rp, "plan", lambda r: [(s, "run" if s["name"] in rp.ALWAYS else "done", []) for s in rp.STEPS])
    monkeypatch.setattr(rp, "report_step", lambda *a, **k: calls.append("report_step"))
    lines = []
    rp.eval_only(str(root), run=None, blocks=None, out=lambda s: (lines.append(s), calls.append("out")),
                 verify=True, checkpoint_root=ckpts)
    assert calls == ["out"] * 6 + ["report_step"], calls
    assert lines[0] == "verify-checkpoints (reported, not gated): laya seed 0 ckpt-laya-s0: absent (%s)" % ABSENT
    assert all(l.startswith("verify-checkpoints (reported, not gated): ") for l in lines)
    assert [re.search(r" ckpt-\S+: (\w+)", l).group(1) for l in lines] == ["absent"] + ["match"] * 5
    bad = subprocess.run([sys.executable, "reproduce.py", "--mode", "eval-only", "--checkpoint-root", ckpts],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert bad.returncode == 2 and "--checkpoint-root" in bad.stderr and "--verify-checkpoints" in bad.stderr


# ---- the command layer (plan 10-05, D-01, D-04): every step's commands, the dry run, the --force guard ----

TRAINERS = ("ray_train.py", "berturk_train.py")
SELECTION_STEPS = ("laya-selection", "berturk-selection")


@pytest.fixture
def bare(tmp_path):
    """A tree with only protocol.json, the Makefile and data/splits.json: every record absent (the from-scratch form,
    with the deferred final-seed commands and the primary few-shot mode)."""
    import shutil
    for rel in ("protocol.json", "Makefile", "data/splits.json"):
        os.makedirs(os.path.dirname(os.path.join(tmp_path, rel)) or str(tmp_path), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel), os.path.join(tmp_path, rel))
    return str(tmp_path)


def header(name):
    return "== %2d %s (" % (D01.index(name) + 1, name)


def test_every_step_has_commands(bare):
    for root in (REPO_ROOT, bare):
        for name in rp.STEP_NAMES:
            cmds = rp.commands(name, root)
            assert cmds and all(isinstance(c, tuple) and c for c in cmds), (root, name)
            for c in cmds:
                assert rp.display(c).startswith(("python ", "make ")), c
    text = rp.dry_run(REPO_ROOT, all_steps=True)
    heads = [l for l in text.splitlines() if l.startswith("== ")]
    assert heads == ["== %2d %s (%s)" % (i, n, "run" if n == "report" else "done") for i, n in enumerate(D01, 1)]
    assert rp.commands("report") == list(rp.REBUILD)
    assert "README: rewrite the 11 generated blocks of README_BLOCKS" in text
    sel = ("python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --device mps --workers 1 "
           "--micro-batch 8 --grad-accum 4 --max-len 128 --epochs 4 --neg 8 --seed 0 --encoder-lr 2e-05 --head-lr "
           "0.0001 --storage ray_results --out ckpt-sel")
    assert sel in text.splitlines() and sel + " --print-run-id" in text.splitlines()
    deferred = rp.step_lines("laya-final", bare)                       # no record: the printer and the manifest
    assert [c[:2] for c, _ in deferred] == [("postfit.py", "--print-final-commands"), ("postfit.py", "--final-manifest")]
    assert all(note and "deferred" in note and "results/postfit/laya.json exists" in note for _, note in deferred)
    assert rp.RUN_ID in " ".join(" ".join(c) for c in rp.commands("laya-selection", bare))


def every_command(roots):
    return [(root, name, c) for root in roots for name in rp.STEP_NAMES for c in rp.commands(name, root)]


def test_no_command_passes_force_or_rescores(bare):
    from pilot.data import SPLITS
    gated_targets = {("make", t): t for t in rp.GATED}
    seen = set()
    for root, name, c in every_command((REPO_ROOT, bare)):
        assert "--force" not in c and not any(a.startswith("--force") for a in c), (name, c)
        assert c != ("make", "baselines"), (name, c)
        assert not (c[0] == "make" and "baselines" in c[1:]), (name, c)
        if c[:2] in gated_targets:
            assert name == gated_targets[c[:2]], (name, c)              # a gated target only in its own step
            seen.add(c[:2])
        if c[0] == "score.py":
            assert name in ("baselines",) + SELECTION_STEPS, (name, c)
            if name in SELECTION_STEPS:                                # a selection export is scored on val only
                assert c[1] == "data/splits/val.jsonl", c
            else:                                                      # the ungated baselines, never a checkpoint
                assert c[1] in ("data/splits/val.jsonl", "data/splits/test.jsonl") and "--checkpoint" not in c, c
        assert not any("hotels_probe" in a for a in c if c[0] == "score.py"), c
        if c[0] in TRAINERS:
            assert name in SELECTION_STEPS + ("laya-final", "berturk-final", "fewshot-bases", "fewshot-continued"), c
        if c[0] == "latency.py" and "--system" in c:
            assert name == "latency", c
        if c[0] == "make":
            assert name in ("data", "baselines", "laya-postfit", "berturk-postfit", "confidence", "final-scoring",
                            "fewshot-scoring", "fewshot-reports", "report"), (name, c)
    assert seen == set(gated_targets)
    assert {s["name"] for s in SPLITS} >= {"val", "test", "hotels_probe"}


def test_dry_run_runs_nothing(monkeypatch, tree):
    def boom(*a, **k):
        raise AssertionError("the dry run ran %r" % (a,))
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(rp, "run_command", boom)
    monkeypatch.setattr(rp, "render_blocks", boom)
    before = rp.snapshot(REPO_ROOT)
    for root in (REPO_ROOT, tree):
        for all_steps in (False, True):
            assert rp.dry_run(root, all_steps=all_steps).endswith(", nothing was run\n")
    assert rp.snapshot(REPO_ROOT) == before
    monkeypatch.undo()
    out = subprocess.run([sys.executable, "reproduce.py", "--mode", "eval-only", "--dry-run"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert [l for l in out.stdout.splitlines() if l.startswith("== ")] == ["== 19 report (run)"] or \
        [l for l in out.stdout.splitlines() if l.startswith("== ")] == ["==  1 data (todo)", "== 19 report (run)"]
    assert "eval-only: " not in out.stdout and rp.DISCLOSURE not in out.stdout   # no model step listed, none run
    assert out.stdout.splitlines()[0] == "reproduce in " + sys.executable
    bad = subprocess.run([sys.executable, "reproduce.py", "--mode", "eval-only", "--all"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "--all" in bad.stderr and "--dry-run" in bad.stderr


def test_dry_run_prints_the_disclosure_before_the_first_model_step(tree):
    import json
    import shutil
    assert not re.search(r"[0-9]", rp.DISCLOSURE)
    for phrase in ("not bit-identical", "require_admitted", "test and HOTELS", "pinned final seeds", "latency",
                   "never reproduces", "first step whose output differs", "never weakens the gate",
                   "never reruns a gated scoring step", "git checkout -- results/", "ray_results/<run_id>/",
                   "berturk_runs/<run_id>/"):
        assert phrase in rp.DISCLOSURE, phrase
    text = rp.dry_run(REPO_ROOT, all_steps=True)
    plan_end = text.index("19 report ")
    assert text.count(rp.DISCLOSURE) == 1
    assert plan_end < text.index(rp.DISCLOSURE) < text.index(header("baselines"))
    assert text.index(header("data")) < text.index(rp.DISCLOSURE)
    with open(os.path.join(tree, "results", "fewshot", "runs.json"), encoding="utf-8") as f:
        ids = {e["dump_run_id"] for e in json.load(f)["entries"]}
    shutil.rmtree(os.path.join(tree, "results", "fewshot"))           # the 10-03 todo-tail tree
    for rid in ids:
        shutil.rmtree(os.path.join(tree, "results", "dumps", rid))
    text = rp.dry_run(tree)
    heads = [l for l in text.splitlines() if l.startswith("== ")]
    assert heads == ["==  1 data (todo)"] + [header(n) + "todo)" for n in D01[13:18]] + ["== 19 report (run)"]
    assert text.count(rp.DISCLOSURE) == 1
    projection = rp.display(("fewshot.py", "--projection", "--out", "results/fewshot/projection.json"))
    assert text.index(header("fewshot-projection")) < text.index(projection) < text.index(rp.DISCLOSURE) \
        < text.index(header("fewshot-bases"))
    assert "# the mode is primary until results/fewshot/projection.json records the D-08 decision" in text


# ---- the command bindings (plan 10-05): each command is the record, recipe or protocol field it reproduces ----

def committed(rel):
    import json
    with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def flags(cmd):
    """{--flag: value or True} of an argv tuple (a flag followed by another flag, or last, is a switch)."""
    out, c = {}, list(cmd)
    for i, a in enumerate(c):
        if a.startswith("--"):
            nxt = c[i + 1] if i + 1 < len(c) else None
            out[a] = True if nxt is None or nxt.startswith("--") else nxt
    return out


def expand(line, v):
    """A Makefile recipe line as an argv list: $(NAME) expanded from v, "$(PYTHON) " and " > /dev/null" dropped."""
    import shlex
    line = line.replace(" > /dev/null", "")
    assert line.startswith("$(PYTHON) "), line
    line = re.sub(r"\$\(([A-Z_]+)\)", lambda m: v[m.group(1)], line[len("$(PYTHON) "):])
    return shlex.split(line)


def test_baselines_commands_are_the_makefile_lines_without_force():
    import shlex
    lines = recipe("baselines")
    assert len(lines) == 7 and all(" --force" in l for l in lines[:6]) and lines[6].startswith("$(MAKE) baseline-reports")
    want = [shlex.split(l.replace(" --force", "").replace("$(SCORE_WORKERS)", "2").replace("$(SCORE_CPUS)", "4")
                        .replace("$(PYTHON) ", "")) for l in lines[:6]]
    v = rp.makefile_vars(REPO_ROOT)
    assert (v["SCORE_WORKERS"], v["SCORE_CPUS"]) == ("2", "4")
    cmds = rp.commands("baselines")
    assert [list(c) for c in cmds[:6]] == want
    assert cmds[6:] == [("make", "baseline-reports")]


def test_selection_commands_match_the_chosen_canonical():
    keys = {"laya": {"--device": "device", "--workers": "workers", "--micro-batch": "micro_batch",
                     "--grad-accum": "grad_accum", "--max-len": "max_len", "--epochs": "epochs", "--neg": "neg",
                     "--seed": "seed", "--encoder-lr": "encoder_lr", "--head-lr": "head_lr"},
            "berturk": {"--device": "device", "--epochs": "epochs", "--lr": "encoder_lr", "--head-lr": "head_lr",
                        "--seed": "seed"}}
    for kind, script, storage in (("laya", "ray_train.py", "ray_results"), ("berturk", "berturk_train.py",
                                                                             "berturk_runs")):
        rec = committed("results/postfit/%s.json" % kind)
        c, chosen = rec["chosen"]["canonical"], rec["chosen"]
        cmds = rp.commands("%s-selection" % kind)
        assert cmds[0] == cmds[1] + ("--print-run-id",) and cmds[1][0] == script
        f = flags(cmds[1])
        for flag, key in keys[kind].items():
            have = float(f[flag]) if isinstance(c[key], float) else (int(f[flag]) if isinstance(c[key], int) else f[flag])
            assert have == c[key], (kind, flag, f[flag], c[key])
        assert f["--data"] == "data/splits/%s.jsonl" % c["split"] and f["--storage"] == storage
        assert f["--base"] == ("multilingual" if kind == "laya" else "berturk")
        run_id = chosen["train_run_id"]
        assert {x["train_run_id"] for x in rec["candidates"]} == {run_id}
        exports = ["%s/%s/exports/epoch-%03d" % (storage, run_id, e) for e in range(1, chosen["epochs_schedule"] + 1)]
        scored = [flags(x)["--checkpoint"] for x in cmds if x[0] == "score.py"]
        pruned = [x[2] for x in cmds if x[:2] == ("postfit.py", "--prune")]
        assert scored == exports and pruned == exports, kind
        assert len(exports) == len(rec["candidates"]) and len(cmds) == 2 + 2 * len(exports)
        assert all(flags(x)["--model"] == kind and x[1] == "data/splits/val.jsonl" for x in cmds if x[0] == "score.py")


def test_final_seed_commands_are_the_records(bare):
    from pilot import berturk, postfit
    for kind, prefix, storage in (("laya", "ckpt-final", "ray_results"), ("berturk", "ckpt-berturk-final",
                                                                          "berturk_runs")):
        rel = "results/postfit/%s.json" % kind
        rec, _ = postfit.read_record(os.path.join(REPO_ROOT, rel))
        seeds = rec["final"]["seeds"]
        want = [tuple(postfit.final_command(rec, s, rel, storage, prefix)) if kind == "laya" else
                tuple(berturk.final_command(rec, s, rel, "results/postfit/berturk-reference.json", storage, prefix))
                for s in seeds]
        cmds = rp.commands("%s-final" % kind)
        assert cmds[:-1] == want
        manifest = committed("results/postfit/%s-final.json" % kind)
        names = [s["checkpoint_name"] for s in sorted(manifest["seeds"], key=lambda s: s["seed"])]
        assert [flags(c)["--out"] for c in cmds[:-1]] == names
        assert [int(flags(c)["--seed"]) for c in cmds[:-1]] == [s["seed"] for s in manifest["seeds"]] == seeds
        assert cmds[-1] == ("postfit.py", "--final-manifest", rel) + tuple(names) + ("--out",
                                                                                    "results/postfit/%s-final.json" % kind)
        printed = rp.commands("%s-final" % kind, bare)                 # no record: the printer, then the same manifest
        assert printed[1] == cmds[-1] and printed[0][:3] == ("postfit.py", "--print-final-commands", rel)


def test_final_reports_commands_match_the_makefile():
    proto, _ = pilot.protocol.require_frozen()
    v = rp.makefile_vars(REPO_ROOT)
    lines = recipe("final-reports")
    assert lines[:2] == ["mkdir -p $(FINAL)/reports", "mkdir -p $(FINAL)/confidence"]
    want = [expand(l, v) for l in lines[2:]]
    with_fewshot = rp.final_report_commands(proto, "results/fewshot/fewshot.json")
    assert [list(c) for c in with_fewshot] == want and len(want) == 22
    scratch = rp.final_report_commands(proto)
    assert len(scratch) == len(with_fewshot)
    differ = [i for i, (a, b) in enumerate(zip(scratch, with_fewshot)) if a != b]
    assert differ == [6] and scratch[6] == ("report.py", "--verdict", "--out", "results/final/verdict.json")
    assert with_fewshot[6] == ("report.py", "--verdict", "--fewshot", "results/fewshot/fewshot.json", "--out",
                               "results/final/verdict.json")
    assert rp.commands("final-reports") == scratch


def test_latency_commands_write_the_pinned_reports():
    proto, _ = pilot.protocol.require_frozen()
    cmds = rp.commands("latency")
    runs = [c for c in cmds if "--system" in c]
    assert len(runs) == 4 and cmds[-1] == ("latency.py", "--render", "results/latency", "--out",
                                           "results/latency/latency.md")
    outs = [flags(c)["--out"] for c in runs]
    assert sorted(outs) == sorted(p for p in proto["inputs"] if p.startswith("results/latency/"))
    assert outs == ["results/latency/%s-%s.json" % (flags(c)["--system"], flags(c)["--device"]) for c in runs]


def test_fewshot_commands_match_the_runs_manifest():
    runs = committed("results/fewshot/runs.json")
    entries = runs["entries"]
    assert len(entries) == 18 and runs["mode"] == "primary"
    bases, continued = rp.commands("fewshot-bases"), rp.commands("fewshot-continued")
    train = [c for c in bases + continued if c[0] in TRAINERS]
    assert len(train) == 18
    by_out = {flags(c)["--out"]: c for c in train}
    for e in entries:
        c = by_out[e["checkpoint_name"]]
        f = flags(c)
        assert c[0] == ("ray_train.py" if e["kind"] == "laya" else "berturk_train.py"), e["checkpoint_name"]
        assert int(f["--seed"]) == e["seed"] and f["--no-export-epochs"] is True
        assert f["--storage"] == ("ray_results" if e["kind"] == "laya" else "berturk_runs")
        if e["stage"] == "base":
            assert f["--fewshot-base"] is True and f["--base"] == ("multilingual" if e["kind"] == "laya" else "berturk")
            assert "--fewshot-draw" not in f and c in bases
        else:
            assert (int(f["--fewshot-draw"]), int(f["--fewshot-k"])) == (e["draw"], e["k"]) and c in continued
            assert f["--base"] == "ckpt-fewshot-%s-base-s%d" % (e["kind"], e["base_seed"])
            assert "--fewshot-base" not in f and f["--epochs"] == "10"
        i = (bases + continued).index(c)                                 # each run is inspected, Laya storage dropped
        nxt = (bases + continued)[i + 1:i + 3]
        assert nxt[0] == ("fewshot.py", "--inspect", e["checkpoint_name"])
        if e["kind"] == "laya":
            assert nxt[1] == ("fewshot.py", "--drop-storage", e["checkpoint_name"], "--storage", "ray_results")
    names = tuple(e["checkpoint_name"] for e in entries)
    assert continued[-1] == ("fewshot.py", "--runs-manifest") + names + ("--out", "results/fewshot/runs.json")
    assert rp.commands("fewshot-scoring") == [("make", "fewshot-scoring"), ("fewshot.py", "--prune") + names]


@pytest.mark.data
@pytest.mark.weights
@pytest.mark.mps
def test_print_run_id_reproduces_the_recorded_run_ids(tmp_path):
    """The 14 training commands that can print one (the 2 selections, the 6 final seeds, the 6 few-shot bases), with
    --storage and --out moved under tmp_path and --print-run-id appended, print the train run ids the committed
    records hold (the run id keeps paths out of the identity; nothing starts Ray or trains)."""
    _need_data()
    runs = committed("results/fewshot/runs.json")["entries"]
    cases = []
    for kind in ("laya", "berturk"):
        rec = committed("results/postfit/%s.json" % kind)
        cases.append(("%s selection" % kind, rp.commands("%s-selection" % kind)[1], rec["chosen"]["train_run_id"]))
        manifest = committed("results/postfit/%s-final.json" % kind)
        for c, s in zip(rp.commands("%s-final" % kind)[:-1], manifest["seeds"]):
            cases.append(("%s final seed %d" % (kind, s["seed"]), c, s["train_run_id"]))
    want_base = {(e["kind"], e["seed"]): e["train_run_id"] for e in runs if e["stage"] == "base"}
    for c in rp.commands("fewshot-bases"):
        if c[0] in TRAINERS:
            kind = "laya" if c[0] == "ray_train.py" else "berturk"
            seed = int(flags(c)["--seed"])
            cases.append(("%s few-shot base seed %d" % (kind, seed), c, want_base[(kind, seed)]))
    assert len(cases) == 14 and len(want_base) == 6
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    got = []
    for label, cmd, want in cases:
        argv = list(cmd)
        for flag in ("--storage", "--out"):
            if flag in argv:
                i = argv.index(flag)
                argv[i + 1] = str(tmp_path / flag.strip("-") / os.path.basename(argv[i + 1]))
        out = subprocess.run([sys.executable] + argv + ["--print-run-id"], cwd=REPO_ROOT, env=env,
                             capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, (label, out.stderr[-2000:])
        printed = [l for l in out.stdout.splitlines() if l.strip()][-1].strip()
        got.append((label, printed, want))
    assert [g for g in got if g[1] != g[2]] == [], got
    assert not os.listdir(tmp_path) or all(not os.listdir(tmp_path / d) for d in os.listdir(tmp_path))
