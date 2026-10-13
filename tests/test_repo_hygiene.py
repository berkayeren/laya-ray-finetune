"""Repository hygiene (D-02): nothing committed may carry message text, a local path or a non-manifest data file.

The pilot is meant to be audited by a bank's reviewers, so the repo must never hold customer or dataset message
text. From data/ only the manifest data/splits.json is tracked (it holds paths, hashes and counts, never text), no
.jsonl file is tracked anywhere, and no tracked text file contains the local home directory. The home directory is
computed at runtime, so this file holds no literal path. The bank-name check cannot be automated (the name is
deliberately absent from the repo) and stays a human review.
"""
import json
import os
import subprocess

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESSAGE_TEXT_KEYS = ("state", "text")
# Coarse backstop only: the longest legitimate manifest value is a 64-char sha256, but message text is often far
# shorter ("Evet.", "cari hesap"), so length cannot detect it. The key-name check and, when the split files exist
# locally, the exact-text check below are the real guards.
MAX_MANIFEST_STRING = 200
REQUIRED_TRACKED = ("requirements.txt", "pyproject.toml", "Makefile", "pilot/pins.py", "data/splits.json")


def tracked(*pathspec):
    """Tracked paths via git ls-files (never a directory walk, which would see gitignored data and .venv)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *pathspec], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    return [f for f in out.stdout.split("\0") if f]


def walk(obj, path="$"):
    """Yield (path, key, value) for every dict entry and (path, None, value) for every list item, recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, "%s.%s" % (path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield path, None, v
            yield from walk(v, "%s[%d]" % (path, i))


def test_only_manifest_tracked_under_data():
    assert tracked("data/") == ["data/splits.json"], \
        "only data/splits.json may be tracked under data/ (generated splits and audit files carry message text)"


def test_no_jsonl_tracked():
    hits = [f for f in tracked() if f.lower().endswith(".jsonl")]
    assert hits == [], "tracked .jsonl files (Laya JSONL holds message text): %s" % hits


def test_manifest_has_no_message_text():
    with open(os.path.join(REPO_ROOT, "data", "splits.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    bad_keys = ["%s.%s" % (p, k) for p, k, _ in walk(manifest) if k in MESSAGE_TEXT_KEYS]
    long_values = ["%s.%s (%d chars)" % (p, k, len(v)) for p, k, v in walk(manifest)
                   if isinstance(v, str) and len(v) > MAX_MANIFEST_STRING]
    assert bad_keys == [], "data/splits.json has message-text keys: %s" % bad_keys
    assert long_values == [], "data/splits.json has string values longer than %d chars: %s" % (
        MAX_MANIFEST_STRING, long_values)


def test_manifest_holds_no_split_message_text():
    split_dir = os.path.join(REPO_ROOT, "data", "splits")
    names = sorted(n for n in os.listdir(split_dir) if n.endswith(".jsonl")) if os.path.isdir(split_dir) else []
    if not names:
        pytest.skip("no local data/splits/*.jsonl to compare against (run `make data`)")
    texts = set()
    for name in names:
        with open(os.path.join(split_dir, name), encoding="utf-8") as f:
            texts |= {json.loads(l)["state"] for l in f.read().split("\n") if l.strip()}
    with open(os.path.join(REPO_ROOT, "data", "splits.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    hits = sorted("%s.%s" % (p, k) for p, k, v in walk(manifest) if isinstance(v, str) and v in texts)
    hits += sorted("%s (key)" % k for _, k, _ in walk(manifest) if k in texts)
    assert hits == [], "data/splits.json holds message text at: %s" % hits


def test_no_local_home_path_in_tracked_files():
    home = os.path.expanduser("~").rstrip("/\\")
    if len(home) < 2:
        pytest.skip("home directory %r is too short to search for" % home)
    hits = []
    for rel in tracked():
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            continue                    # deleted in the working tree, or a submodule entry
        with open(path, "rb") as f:
            raw = f.read()
        if b"\0" in raw:
            continue                    # binary file
        if home in raw.decode("utf-8", errors="ignore"):
            hits.append(rel)
    assert hits == [], "tracked files contain the local home directory path: %s" % hits


@pytest.mark.parametrize("rel", REQUIRED_TRACKED)
def test_required_files_tracked(rel):
    assert tracked(rel) == [rel], "%s must be tracked (reproducibility depends on it)" % rel


@pytest.mark.parametrize("rel", ["ray_results/laya-final_train-s0-20261013-120000/params.json",
                                 "ckpt/model.safetensors", "ckpt/rl_agent_config.json", "ckpt_s1/model.safetensors",
                                 "other/model.safetensors"])
def test_training_outputs_are_gitignored(rel):
    # documented commands write Ray storage and checkpoints into the repo root; `git add .` must not pick them up
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", "--", rel], cwd=REPO_ROOT)
    assert r.returncode == 0, "%s is not gitignored" % rel
