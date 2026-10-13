"""Offline guards for the pinned weights path (ENV-02): fetch refusal paths, pickle-free download patterns, and no
unpinned load call anywhere in the repo. snapshot_download is monkeypatched with fake snapshot dirs; nothing here
touches the network.
"""
import ast
import fnmatch
import hashlib
import os
import subprocess
import sys
import types

import pytest

import pilot.data
import pilot.weights
from pilot import pins

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
PICKLE_NAMES = ["pytorch_model.bin", "tf_model.h5", "flax_model.msgpack", "multilingual/pytorch_model.bin"]


def fake_snapshot(tmp_path, pin, name=None, skip=None, corrupt=None):
    """Write tmp_path/snapshots/<name or SHA>[/subfolder]/<pinned files>; return (root, pin with matching digests)."""
    root = tmp_path / "snapshots" / (name or pin["revision"])
    d = root / pin["subfolder"] if pin["subfolder"] else root
    digests = {}
    for rel in pin["sha256"]:
        data = ("fake %s" % rel).encode("utf-8")
        digests[rel] = hashlib.sha256(data).hexdigest()
        if rel == skip:
            continue
        path = d / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tampered" if rel == corrupt else data)
    return str(root), dict(pin, sha256=digests)


def patch_download(monkeypatch, result):
    calls = []

    def fake(repo, revision=None, allow_patterns=None, dry_run=False):
        calls.append({"repo": repo, "revision": revision, "allow_patterns": allow_patterns, "dry_run": dry_run})
        return result
    monkeypatch.setattr(pilot.weights, "snapshot_download", fake)
    return calls


def test_fetch_refuses_snapshot_not_at_pinned_revision(tmp_path, monkeypatch):
    root, pin = fake_snapshot(tmp_path, pins.LAYA, name="0" * 40)
    patch_download(monkeypatch, root)
    with pytest.raises(SystemExit, match="resolved to revision"):
        pilot.weights.fetch(pin)


def test_fetch_refuses_digest_mismatch(tmp_path, monkeypatch):
    root, pin = fake_snapshot(tmp_path, pins.LAYA, corrupt="model.safetensors")
    patch_download(monkeypatch, root)
    with pytest.raises(SystemExit, match="sha256"):
        pilot.weights.fetch(pin)


def test_fetch_refuses_missing_pinned_file(tmp_path, monkeypatch):
    root, pin = fake_snapshot(tmp_path, pins.LAYA, skip="tokenizer/tokenizer.json")
    patch_download(monkeypatch, root)
    with pytest.raises(SystemExit, match="missing"):
        pilot.weights.fetch(pin)


def test_fetch_returns_subfolder_for_laya(tmp_path, monkeypatch):
    root, pin = fake_snapshot(tmp_path, pins.LAYA)
    calls = patch_download(monkeypatch, root)
    assert pilot.weights.fetch(pin) == os.path.join(root, "multilingual")
    assert calls == [{"repo": pins.LAYA["repo"], "revision": pins.LAYA["revision"],
                      "allow_patterns": pins.LAYA["allow_patterns"], "dry_run": False}]


def test_fetch_returns_root_for_berturk(tmp_path, monkeypatch):
    root, pin = fake_snapshot(tmp_path, pins.BERTURK)
    patch_download(monkeypatch, root)
    assert pilot.weights.fetch(pin) == root


def test_check_revision_refuses_other_commit(monkeypatch):
    sha = pins.LAYA["revision"]
    infos = [types.SimpleNamespace(commit_hash=sha, filename="multilingual/model.safetensors"),
             types.SimpleNamespace(commit_hash="f" * 40, filename="multilingual/rl_agent_config.json")]
    patch_download(monkeypatch, infos)
    with pytest.raises(SystemExit, match="rl_agent_config.json"):
        pilot.weights.check_revision(pins.LAYA)


def test_check_revision_refuses_empty_listing(monkeypatch):
    patch_download(monkeypatch, [])
    with pytest.raises(SystemExit):
        pilot.weights.check_revision(pins.LAYA)


def test_check_revision_counts_matching_files(monkeypatch):
    sha = pins.BERTURK["revision"]
    infos = [types.SimpleNamespace(commit_hash=sha, filename=f) for f in pins.BERTURK["allow_patterns"]]
    calls = patch_download(monkeypatch, infos)
    assert pilot.weights.check_revision(pins.BERTURK) == 4
    assert calls[0]["dry_run"] is True and calls[0]["revision"] == sha


@pytest.mark.parametrize("pin", [pins.LAYA, pins.BERTURK], ids=["laya", "berturk"])
def test_allow_patterns_never_match_pickle_tf_or_flax(pin):
    hits = [(p, n) for p in pin["allow_patterns"] for n in PICKLE_NAMES if fnmatch.fnmatch(n, p)]
    assert not hits, hits


def test_laya_digests_skip_tokenizer_config():
    # laya may rewrite tokenizer/tokenizer_config.json in the snapshot, so a digest on it would be fragile
    assert not [k for k in pins.LAYA["sha256"] if k.endswith("tokenizer_config.json")]


def _unpinned_calls(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
        if name == "resolve_checkpoint_dir":
            bad.append("%s:%d resolve_checkpoint_dir(...)" % (path, node.lineno))
        if (isinstance(fn, ast.Attribute) and fn.attr == "load" and isinstance(fn.value, ast.Name)
                and fn.value.id == "laya" and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            bad.append("%s:%d laya.load(%r)" % (path, node.lineno, node.args[0].value))
    return bad


def test_no_unpinned_load_call_in_repo():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = [f for f in out.split() if not f.startswith("tests/")]
    assert "evaluate.py" in files and "ray_train.py" in files
    bad = [b for f in files for b in _unpinned_calls(os.path.join(REPO, f))]
    assert not bad, "unpinned weight loads:\n" + "\n".join(bad)


def test_ast_guard_catches_unpinned_calls(tmp_path):
    p = tmp_path / "x.py"
    p.write_text('import laya\nfrom laya import train as lt\nlaya.load("convaiinnovations/laya")\n'
                 'lt.resolve_checkpoint_dir("multilingual")\n', encoding="utf-8")
    assert len(_unpinned_calls(str(p))) == 2


def test_load_agent_refuses_non_dir_without_importing_laya():
    code = ("import sys, evaluate\n"
            "try:\n    evaluate.load_agent('not-a-dir', 'cpu')\n"
            "except SystemExit as e:\n    print('REFUSED', e)\n"
            "print('LAYA_IMPORTED' if 'laya' in sys.modules else 'LAYA_NOT_IMPORTED')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "REFUSED" in r.stdout and "local Laya checkpoint dir" in r.stdout, r.stdout
    assert "LAYA_NOT_IMPORTED" in r.stdout, r.stdout


def test_identity_of_the_pinned_checkpoint_is_the_pin():
    got = pilot.weights.identity("multilingual")
    assert got["revision"] == pins.LAYA["revision"] and got["model_sha256"] == pins.LAYA["sha256"]["model.safetensors"]


def test_identity_of_a_local_checkpoint_hashes_its_files(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    got = pilot.weights.identity(str(tmp_path))
    assert got == {"local": True, "model_sha256": hashlib.sha256(b"weights").hexdigest(), "config_sha256": None}


def test_repo_relpath_hides_the_local_checkout_path(tmp_path):
    assert pilot.data.repo_relpath(pilot.data.DEFAULT_MANIFEST) == "data/splits.json"
    assert pilot.data.repo_relpath(str(tmp_path / "x.json")) == str(tmp_path / "x.json")
