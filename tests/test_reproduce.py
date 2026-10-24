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
    assert rp.MODES == ("eval-only",)
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


def test_makefile_report_target():
    assert recipe("report") == ["$(PYTHON) reproduce.py --mode eval-only "
                                "$(if $(VERIFY_CHECKPOINTS),--verify-checkpoints,)"]
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        text = f.read()
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "report" in phony.group(1).split(), "report is not in .PHONY"
    assert re.search(r"^#   make report  eval-only: .*fails on any rebuilt file that differs$", text, re.M)
    assert "final-reports" in phony.group(1).split()


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
def test_eval_only_end_to_end(tmp_path, monkeypatch):
    """A fresh copy of the working tree (every tracked file, plus the local data/splits/ and data/audit/), committed
    in a new repo: reproduce.py --mode eval-only exits 0, runs all 9 rebuild commands and leaves the copy clean."""
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
    out = subprocess.run([sys.executable, "reproduce.py", "--mode", "eval-only"], cwd=copy, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    lines = out.stdout.splitlines()
    assert lines[0] == "reproduce in " + sys.executable
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
