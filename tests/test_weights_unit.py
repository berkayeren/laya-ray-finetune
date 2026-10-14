"""Offline guards for the pinned weights path (ENV-02): fetch refusal paths, pickle-free download patterns, and no
unpinned load call anywhere in the repo. snapshot_download is monkeypatched with fake snapshot dirs; nothing here
touches the network.
"""
import ast
import fnmatch
import hashlib
import json
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


LOADERS = ("load", "load_checkpoint", "resolve_checkpoint_dir")
HUB_DOWNLOADS = ("snapshot_download", "hf_hub_download")


def _unpinned_calls(path, in_pilot=False):
    """Weight loads that could bypass pilot.weights.fetch. Outside pilot/: any from-import of a loader or hub download,
    any bare-name load/load_checkpoint call (an aliased from-import), any hub download call. Everywhere: a loader
    called on a string literal (a hub id), resolve_checkpoint_dir, and a hub download without revision=."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not in_pilot \
                and (node.module or "").split(".")[0] in ("laya", "huggingface_hub"):
            names = [a.name for a in node.names if a.name in LOADERS + HUB_DOWNLOADS or a.name == "*"]
            if names:
                bad.append("%s:%d from %s import %s" % (path, node.lineno, node.module, ", ".join(names)))
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
        if name == "resolve_checkpoint_dir":
            bad.append("%s:%d resolve_checkpoint_dir(...)" % (path, node.lineno))
        elif name in HUB_DOWNLOADS:
            if not in_pilot:
                bad.append("%s:%d %s(...) outside pilot/ (use pilot.weights.fetch)" % (path, node.lineno, name))
            elif not any(k.arg == "revision" for k in node.keywords):
                bad.append("%s:%d %s(...) without revision=" % (path, node.lineno, name))
        elif isinstance(fn, ast.Name) and name in ("load", "load_checkpoint") and not in_pilot:
            bad.append("%s:%d bare %s(...)" % (path, node.lineno, name))
        elif (name in ("load", "load_checkpoint") and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            bad.append("%s:%d %s(%r)" % (path, node.lineno, name, node.args[0].value))
    return bad


def test_no_unpinned_load_call_in_repo():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = [f for f in out.split() if not f.startswith("tests/")]
    assert "score.py" in files and "evaluate.py" in files and "ray_train.py" in files and "pilot/weights.py" in files
    bad = [b for f in files for b in _unpinned_calls(os.path.join(REPO, f), in_pilot=f.startswith("pilot/"))]
    assert not bad, "unpinned weight loads:\n" + "\n".join(bad)


def test_ast_guard_catches_unpinned_calls(tmp_path):
    p = tmp_path / "x.py"
    p.write_text('import laya\nfrom laya import train as lt\nlaya.load("convaiinnovations/laya")\n'
                 'lt.resolve_checkpoint_dir("multilingual")\n'
                 'from laya import load\nload(path)\n'                                  # 2: from-import + bare call
                 'from laya.train import load_checkpoint as lc\n'                         # 1
                 'lt.load_checkpoint("convaiinnovations/laya-multilingual")\n'           # 1
                 'import huggingface_hub as hh\nhh.snapshot_download("x/y", revision="0" * 40)\n'   # 1: outside pilot/
                 'hh.hf_hub_download("x/y", "f")\n'                                       # 1
                 'json.load(f)\nlt.load_checkpoint(base_dir)\nlaya.load(fetch(pin))\n', encoding="utf-8")
    assert len(_unpinned_calls(str(p))) == 8, _unpinned_calls(str(p))
    q = tmp_path / "y.py"                                    # inside pilot/: hub downloads must pin a revision
    q.write_text('from huggingface_hub import snapshot_download\n'
                 'snapshot_download("x/y", revision=pin)\nsnapshot_download("x/y")\n', encoding="utf-8")
    assert len(_unpinned_calls(str(q), in_pilot=True)) == 1


def test_resolve_checkpoint_refuses_non_dir_without_importing_laya():
    code = ("import sys, score\n"
            "try:\n    score.resolve_checkpoint('not-a-dir')\n"
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
    assert got == {"local": True, "model_sha256": hashlib.sha256(b"weights").hexdigest(), "config_sha256": None,
                   "files_sha256": {}}


def test_identity_binds_the_tokenizer_and_encoder_files(tmp_path):
    """04-REVIEW IN-05: every file laya.load reads is in a local identity; tokenizer_config.json, which laya rewrites
    on the first load, by its normalized JSON, so the identity before and after that rewrite is the same."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "rl_agent_config.json").write_bytes(b"{}")
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "encoder").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_bytes(b"vocab")
    (tmp_path / "encoder" / "config.json").write_bytes(b"{\"hidden_size\": 64}")
    tcfg = tmp_path / "tokenizer" / "tokenizer_config.json"
    tcfg.write_text(json.dumps({"backend": "tokenizers", "tokenizer_class": "TokenizersBackend",
                                "extra_special_tokens": ["<a>"], "pad_token": "[PAD]"}), encoding="utf-8")
    (tmp_path / "tokenizer" / ".tokenizer_config.x.tmp").write_bytes(b"partial")      # a rewrite's temp file
    ident = pilot.weights.identity(str(tmp_path))
    assert sorted(ident["files_sha256"]) == sorted(["encoder/config.json", pilot.weights.NORMALIZED_KEY,
                                                    "tokenizer/tokenizer.json"])
    assert ident["files_sha256"]["tokenizer/tokenizer.json"] == hashlib.sha256(b"vocab").hexdigest()
    tcfg.write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "pad_token": "[PAD]",
                                "extra_special_tokens": {"extra_0": "<a>"}}, indent=2), encoding="utf-8")
    assert pilot.weights.identity(str(tmp_path)) == ident                   # laya's rewrite keeps the identity
    digests = pilot.weights.checkpoint_digests(str(tmp_path), ident)
    pilot.weights.verify_files(str(tmp_path), digests, "checkpoint")
    assert pilot.weights.NORMALIZED_KEY not in pilot.weights.laya_digests(digests)
    assert set(pilot.weights.laya_digests(digests)) == {"model.safetensors", "rl_agent_config.json",
                                                        "tokenizer/tokenizer.json", "encoder/config.json"}
    for rel, data in (("tokenizer/tokenizer.json", b"other vocab"), ("encoder/config.json", b"{}"),
                      ("tokenizer/tokenizer_config.json", b'{"pad_token": "[UNK]"}'), ("tokenizer/added.json", b"x")):
        before = (tmp_path / rel).read_bytes() if (tmp_path / rel).exists() else None
        (tmp_path / rel).write_bytes(data)
        assert pilot.weights.identity(str(tmp_path)) != ident, rel             # any scored file moves the identity
        if before is None:
            (tmp_path / rel).unlink()
        else:
            (tmp_path / rel).write_bytes(before)
    (tmp_path / "tokenizer" / "tokenizer_config.json").write_bytes(b'{"pad_token": "[UNK]"}')
    with pytest.raises(SystemExit, match="normalized sha256"):
        pilot.weights.verify_files(str(tmp_path), digests, "checkpoint")


def test_repo_relpath_never_returns_a_local_path(tmp_path, monkeypatch):
    assert pilot.data.repo_relpath(pilot.data.DEFAULT_MANIFEST) == "data/splits.json"
    assert pilot.data.repo_relpath(str(tmp_path / "x.json")) == "<outside-repo>/x.json"
    assert pilot.data.repo_relpath(str(tmp_path / "ckpt") + os.sep) == "<outside-repo>/ckpt"
    monkeypatch.chdir(REPO)
    rel = os.path.relpath(str(tmp_path / "v" / "splits.json"))              # a cwd-relative '../..' path
    assert pilot.data.repo_relpath(rel) == "<outside-repo>/splits.json"
    assert pilot.data.repo_relpath("./ckpt") == "ckpt"


def _assert_no_local_path(obj, tmp_path):
    text = json.dumps(obj, ensure_ascii=False)
    for p in {str(tmp_path), os.path.realpath(str(tmp_path)), os.path.expanduser("~"), REPO}:
        assert p not in text, (p, text)


def test_score_sidecar_carries_no_local_path(tmp_path):
    import score
    ckpt = tmp_path / "ckpts" / "run1"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"weights")
    (ckpt / "rl_agent_config.json").write_bytes(b"{}")
    manifest = tmp_path / "variant" / "splits.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"{}")
    entry = {"path": "splits/test.jsonl", "sha256": "0" * 64, "role": "test", "uids_sha256": "1" * 64,
             "n_messages": 20}
    side = dict(score.sidecar_base(entry, "test", str(manifest), 20),
                system=score.laya_system(str(ckpt), pilot.weights.identity(str(ckpt))))
    _assert_no_local_path(side, tmp_path)
    assert side["system"]["checkpoint"] == "<outside-repo>/run1"
    assert side["manifest"]["path"] == "<outside-repo>/splits.json"
    assert side["system"]["checkpoint_identity"]["model_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert side["manifest"]["is_committed"] is False
    assert score.laya_system("multilingual", pilot.weights.identity("multilingual"))["checkpoint"] == "multilingual"


def test_ray_train_checkpoint_provenance_carries_no_local_path(tmp_path):
    ray_train = pytest.importorskip("ray_train")
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "splits.json"
    manifest.write_bytes(b"{}")
    entry = {"path": "splits/final_train.jsonl", "sha256": "0" * 64, "role": "train"}
    prov = ray_train.provenance(entry, str(manifest), str(base), str(base))
    _assert_no_local_path(prov, tmp_path)
    assert prov["manifest"] == "<outside-repo>/splits.json" and prov["base"]["local"] is True


def test_resolve_checkpoint_runs_the_pinned_fetch_on_the_driver(tmp_path, monkeypatch):
    import score
    root, pin = fake_snapshot(tmp_path, pins.LAYA)
    patch_download(monkeypatch, root)
    monkeypatch.setattr(pins, "LAYA", pin)
    model_dir, expected = score.resolve_checkpoint("multilingual")
    assert model_dir == os.path.join(root, "multilingual") and expected == pin["sha256"]
    root, pin = fake_snapshot(tmp_path / "bad", pins.LAYA, corrupt="model.safetensors")
    patch_download(monkeypatch, root)
    monkeypatch.setattr(pins, "LAYA", pin)
    with pytest.raises(SystemExit, match="sha256"):                     # the digest message, not an actor death
        score.resolve_checkpoint("multilingual")


def test_checkpoint_digests_bind_a_local_load_to_its_recorded_identity(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "rl_agent_config.json").write_bytes(b"{}")
    ident = pilot.weights.identity(str(tmp_path))
    assert pilot.weights.checkpoint_digests(str(tmp_path), ident) == {
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        "rl_agent_config.json": hashlib.sha256(b"{}").hexdigest()}
    with pytest.raises(SystemExit, match="files_sha256"):               # an identity from before IN-05
        pilot.weights.checkpoint_digests(str(tmp_path), {k: v for k, v in ident.items() if k != "files_sha256"})
    assert pilot.weights.checkpoint_digests("multilingual", None) == pins.LAYA["sha256"]
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(SystemExit, match="lacks model.safetensors"):
        pilot.weights.checkpoint_digests(str(tmp_path), pilot.weights.identity(str(tmp_path)))


def test_score_load_is_refused_when_the_checkpoint_changed_after_its_identity(tmp_path):
    import score
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "rl_agent_config.json").write_bytes(b"{}")
    model_dir, expected = score.resolve_checkpoint(str(tmp_path))
    assert expected["model.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    (tmp_path / "model.safetensors").write_bytes(b"swapped")              # changed between identity and load
    with pytest.raises(ValueError, match="model.safetensors"):            # laya verifies before parsing any weight
        score.load_agent(model_dir, "cpu", expected)


def test_verify_files_refuses_a_changed_base(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    digests = {"model.safetensors": hashlib.sha256(b"weights").hexdigest()}
    pilot.weights.verify_files(str(tmp_path), digests, "base checkpoint")
    (tmp_path / "model.safetensors").write_bytes(b"swapped")
    with pytest.raises(SystemExit, match="base checkpoint: model.safetensors sha256"):
        pilot.weights.verify_files(str(tmp_path), digests, "base checkpoint")
    with pytest.raises(SystemExit, match="missing"):
        pilot.weights.verify_files(str(tmp_path), {"rl_agent_config.json": "0" * 64}, "base checkpoint")
