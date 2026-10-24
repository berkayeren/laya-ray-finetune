"""Fetch and verify the pinned model weights: Laya multilingual and the BERTurk baseline.

Every load of Laya or BERTurk goes through fetch(pin): it downloads only pin["allow_patterns"] (safetensors and
text configs, never pickle/TF/Flax files) at the full revision SHA, refuses a snapshot whose directory is not that
SHA, and sha256-checks every pinned file before returning the local dir. check_revision(pin) asks the hub, without
downloading, which commit each pinned file resolves to and refuses any other commit (e.g. a moving ref like "main").
After a first fetch, HF_HUB_OFFLINE=1 reuses the cache by SHA.

identity(checkpoint) names a local checkpoint by the sha256 of every file laya.load reads: the
weights, rl_agent_config.json, and every file under tokenizer/ and encoder/ (files_sha256). laya.load rewrites
tokenizer/tokenizer_config.json in place on the first load of a checkpoint transformers saved (laya 0.4.6
laya.agent._fix_tokenizer_config), so that one file is bound by the sha256 of its normalized JSON (NORMALIZED_KEY),
which the rewrite does not change; checkpoint_digests hands laya.load the raw-file digests and verify_files checks
the normalized one. checkpoint_digests refuses a local checkpoint without tokenizer/tokenizer_config.json or
encoder/config.json (pilot.dumps.LOCAL_REQUIRED_FILES): laya 0.4.6 would read them from the Hub by the config's encoder
name, unpinned. laya only warns when it cannot rewrite tokenizer_config.json, so check_tokenizer_config_rewritable
refuses, on the driver before any load or ray.init, a file laya would have to rewrite in a read-only tokenizer/ (or one
that is not a JSON object), and check_tokenizer_config_rewritten refuses, after the load, a file it left un-normalized
anyway (the backstop).

usage: python -m pilot.weights        # fetch and verify both pinned checkpoints, print sizes and free disk
"""
import hashlib
import json
import os
import shutil
from typing import Any

from huggingface_hub import snapshot_download

from pilot import pins
from pilot.data import sha256_file
from pilot.dumps import LOCAL_REQUIRED_FILES


COMPONENT_DIRS = ("tokenizer", "encoder")                  # read by laya.load next to the weights and the config
TOKENIZER_CONFIG = "tokenizer/tokenizer_config.json"       # rewritten in place by laya.load
NORMALIZED_KEY = TOKENIZER_CONFIG + "#normalized"           # its digest key: not a path, so laya.load never sees it


def fetch(pin: dict[str, Any]) -> str:
    """The local dir of `pin`'s files: downloaded (or read from the hub cache) at the full revision SHA, only
    pin["allow_patterns"], every pinned file sha256-checked; SystemExit when the snapshot is another revision."""
    root = snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=pin["allow_patterns"])
    got = os.path.basename(os.path.realpath(root))
    if got != pin["revision"]:
        raise SystemExit("%s resolved to revision %s, pinned %s" % (pin["repo"], got, pin["revision"]))
    d = os.path.join(root, pin["subfolder"]) if pin.get("subfolder") else root
    verify_files(d, pin["sha256"], "%s at pinned revision %s" % (pin["repo"], pin["revision"]))
    return d


def verify_files(d: str, sha256: dict[str, str], what: str) -> None:
    """Hard-fail unless every {path relative to d: sha256} file exists under d with that digest; NORMALIZED_KEY is
    checked against the normalized tokenizer config (tokenizer_config_sha256)."""
    for rel in sorted(sha256):
        path = os.path.join(d, TOKENIZER_CONFIG if rel == NORMALIZED_KEY else rel)
        if not os.path.isfile(path):
            raise SystemExit("%s: %s missing" % (what, rel))
        have = tokenizer_config_sha256(path) if rel == NORMALIZED_KEY else sha256_file(path)
        if have != sha256[rel]:
            raise SystemExit("%s: %s sha256 %s != expected %s" % (what, rel, have, sha256[rel]))


def checkpoint_digests(checkpoint: str, ident: dict[str, Any] | None) -> dict[str, str]:
    """{file: sha256} a load of `checkpoint` must match: the Laya pin for 'multilingual', else every file hashed into
    `ident` (identity(checkpoint): weights, config, tokenizer/ and encoder/), so the bytes loaded are the bytes the
    provenance records. A NORMALIZED_KEY entry is for verify_files only: pass laya.load laya_digests() of this.

    A local checkpoint without the weights, the config or a LOCAL_REQUIRED_FILES file is refused here, on the driver
    before any load or ray.init: without tokenizer/ or encoder/ laya.load and laya.train.load_checkpoint would read
    the tokenizer or the encoder config from the Hub by the config's encoder name, unpinned. `ident` may be None
    only for 'multilingual'."""
    if checkpoint == "multilingual":
        return dict(pins.LAYA["sha256"])
    name = os.path.basename(os.path.normpath(checkpoint))
    if ident.get("files_sha256") is None:
        raise SystemExit("the identity of checkpoint %s has no files_sha256 (tokenizer/ and encoder/); recompute it "
                         "with this pilot.weights.identity" % name)
    out = dict(ident["files_sha256"], **{"model.safetensors": ident["model_sha256"],
                                         "rl_agent_config.json": ident["config_sha256"]})
    missing = sorted(rel for rel, h in out.items() if h is None)
    missing += [TOKENIZER_CONFIG if rel == NORMALIZED_KEY else rel for rel in LOCAL_REQUIRED_FILES if rel not in out]
    if missing:
        raise SystemExit("checkpoint %s lacks %s (laya.load would fail, or read a missing tokenizer/ or encoder/ "
                         "unpinned from the Hub); use a complete checkpoint dir: model.safetensors, "
                         "rl_agent_config.json, tokenizer/ and encoder/, as laya.train.save_checkpoint writes it"
                         % (name, ", ".join(missing)))
    return out


def laya_digests(digests: dict[str, str]) -> dict[str, str]:
    """The expected_sha256 laya.load verifies itself: every raw-file digest of checkpoint_digests, without
    NORMALIZED_KEY (laya rewrites that file, so verify_files checks its normalized digest instead)."""
    return {rel: h for rel, h in digests.items() if rel != NORMALIZED_KEY}


def normalized_tokenizer_config(tcfg: dict[str, Any]) -> dict[str, Any]:
    """A tokenizer_config.json dict as laya.load leaves it, mirroring laya 0.4.6 laya.agent._fix_tokenizer_config so
    this module stays laya- and torch-free (tests/test_laya_adapter.py checks the mirror against laya): a missing or
    TokenizersBackend tokenizer_class becomes PreTrainedTokenizerFast without backend and is_local, and a list of
    extra_special_tokens becomes {"extra_<i>": token}. Idempotent, so the file before and after laya's rewrite has one
    normalized view."""
    t = dict(tcfg)
    if t.get("tokenizer_class") in (None, "TokenizersBackend"):
        t["tokenizer_class"] = "PreTrainedTokenizerFast"
        t.pop("backend", None)
        t.pop("is_local", None)
    extra = t.get("extra_special_tokens")
    if isinstance(extra, list):
        t["extra_special_tokens"] = {"extra_%d" % i: tok for i, tok in enumerate(extra)}
    return t


def tokenizer_config_sha256(path: str) -> str:
    """sha256 of the sorted, compact JSON of normalized_tokenizer_config(the file at `path`); of the raw bytes when it
    is not a JSON object (laya leaves such a file alone)."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        tcfg = json.loads(raw.decode("utf-8"))
    except ValueError:
        tcfg = None
    if not isinstance(tcfg, dict):
        return hashlib.sha256(raw).hexdigest()
    blob = json.dumps(normalized_tokenizer_config(tcfg), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_tokenizer_config_rewritable(d: str, what: str) -> None:
    """SystemExit, on the driver before any load or ray.init, when laya.load or laya.train.load_checkpoint would have
    to rewrite d's tokenizer/tokenizer_config.json but cannot: the file is not a JSON object, or it differs from its
    normalized_tokenizer_config view and tokenizer/ is not writable (laya 0.4.6 writes a mkstemp temp file there and
    os.replace's it over the file). Without this check the refusal would come only after the load, from
    check_tokenizer_config_rewritten, which stays the backstop: inside a Ray Data actor's constructor that would make
    Ray restart the actor forever, and inside a Ray Train worker it would come after ray.init and a full base load.
    A missing file is checkpoint_digests' refusal, not this one."""
    path = os.path.join(d, TOKENIZER_CONFIG)
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            tcfg = json.load(f)
    except ValueError:
        tcfg = None
    if not isinstance(tcfg, dict):
        raise SystemExit("%s: %s is not a JSON object, so laya cannot normalize it on load; use a checkpoint dir as "
                         "laya.train.save_checkpoint writes it" % (what, TOKENIZER_CONFIG))
    tok_dir = os.path.dirname(path)
    if normalized_tokenizer_config(tcfg) != tcfg and not os.access(tok_dir, os.W_OK | os.X_OK):
        raise SystemExit("%s: %s needs laya's load-time rewrite, but %s/ is not writable, so laya would load the "
                         "un-normalized file; make the checkpoint dir writable (or copy it) and rerun"
                         % (what, TOKENIZER_CONFIG, os.path.dirname(TOKENIZER_CONFIG)))


def check_tokenizer_config_rewritten(d: str, what: str) -> None:
    """SystemExit unless d's tokenizer/tokenizer_config.json, read after laya.load or laya.train.load_checkpoint, is a
    JSON object equal to its normalized_tokenizer_config view. laya 0.4.6's rewrite catches every
    error, only warns ("could not patch", e.g. on a read-only checkpoint dir) and loads the raw file; the identity binds
    the normalized view, so after a failed rewrite two raw configs one identity does not tell apart would load
    differently. A missing file is checkpoint_digests' refusal, not this one."""
    path = os.path.join(d, TOKENIZER_CONFIG)
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            tcfg = json.load(f)
    except ValueError:
        tcfg = None
    if not isinstance(tcfg, dict) or normalized_tokenizer_config(tcfg) != tcfg:
        raise SystemExit("%s: laya could not rewrite %s on load (a read-only dir, or not a JSON object), so the "
                         "tokenizer it loaded is not the one the identity binds; make the checkpoint dir writable "
                         "(or copy it) and rerun" % (what, TOKENIZER_CONFIG))


def component_digests(checkpoint: str) -> dict[str, str]:
    """{path relative to `checkpoint`: sha256} of every file under its tokenizer/ and encoder/ dirs (sorted, "/"
    separators; dotfiles such as a laya rewrite's temp file skipped), with TOKENIZER_CONFIG under NORMALIZED_KEY."""
    out = {}
    for sub in COMPONENT_DIRS:
        for root, dirs, files in os.walk(os.path.join(checkpoint, sub)):
            dirs.sort()
            for name in sorted(files):
                if name.startswith("."):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, checkpoint).replace(os.sep, "/")
                if rel == TOKENIZER_CONFIG:
                    out[NORMALIZED_KEY] = tokenizer_config_sha256(path)
                elif os.path.isfile(path):
                    out[rel] = sha256_file(path)
    return dict(sorted(out.items()))


def identity(checkpoint: str) -> dict[str, Any]:
    """Provenance of a model for reports: the Laya pin for 'multilingual', else the sha256 of a local checkpoint dir's
    weights and config (None for a file the dir lacks) and files_sha256, the component_digests of its tokenizer/ and
    encoder/, so every file laya.load reads is bound. Hashes only; never downloads or loads
    anything."""
    if checkpoint == "multilingual":
        return {"repo": pins.LAYA["repo"], "subfolder": pins.LAYA["subfolder"], "revision": pins.LAYA["revision"],
                "model_sha256": pins.LAYA["sha256"]["model.safetensors"]}
    out = {"local": True}
    for key, rel in (("model_sha256", "model.safetensors"), ("config_sha256", "rl_agent_config.json")):
        path = os.path.join(checkpoint, rel)
        out[key] = sha256_file(path) if os.path.isfile(path) else None
    out["files_sha256"] = component_digests(checkpoint)
    return out


def check_revision(pin: dict[str, Any]) -> int:
    """The number of pinned files; SystemExit unless the hub resolves every one of them (a dry run, nothing is
    downloaded) to pin["revision"]."""
    infos = snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=pin["allow_patterns"],
                              dry_run=True)
    if not infos:
        raise SystemExit("%s@%s: no files match %s" % (pin["repo"], pin["revision"], pin["allow_patterns"]))
    for info in infos:
        if info.commit_hash != pin["revision"]:
            raise SystemExit("%s/%s resolves to commit %s, pinned %s"
                             % (pin["repo"], info.filename, info.commit_hash, pin["revision"]))
    return len(infos)


def _size(d: str) -> tuple[int, int]:
    """(file count, total bytes) under `d`, following the snapshot symlinks."""
    n = total = 0
    for dirpath, _, files in os.walk(d):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):                        # follows the snapshot symlinks into blobs/
                n, total = n + 1, total + os.path.getsize(path)
    return n, total


def main() -> None:
    """Check, fetch and verify both pinned checkpoints; print their sizes and the free disk."""
    d = None
    for pin in (pins.LAYA, pins.BERTURK):
        check_revision(pin)
        d = fetch(pin)
        n, total = _size(d)
        print("%s@%s -> %s (%d files, %.0f MB)" % (pin["repo"], pin["revision"], d, n, total / 1e6))
    print("free disk: %.1f GB" % (shutil.disk_usage(d).free / 1e9))


if __name__ == "__main__":
    main()
