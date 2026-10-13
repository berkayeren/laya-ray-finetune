"""Fetch and verify the pinned model weights (ENV-02): Laya multilingual and the BERTurk baseline.

Every load of Laya or BERTurk goes through fetch(pin): it downloads only pin["allow_patterns"] (safetensors and
text configs, never pickle/TF/Flax files) at the full revision SHA, refuses a snapshot whose directory is not that
SHA, and sha256-checks every pinned file before returning the local dir. check_revision(pin) asks the hub, without
downloading, which commit each pinned file resolves to and refuses any other commit (e.g. a moving ref like "main").
After a first fetch, HF_HUB_OFFLINE=1 reuses the cache by SHA.

usage: python -m pilot.weights        # fetch and verify both pinned checkpoints, print sizes and free disk
"""
import os
import shutil

from huggingface_hub import snapshot_download

from pilot import pins
from pilot.data import sha256_file


def fetch(pin):
    root = snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=pin["allow_patterns"])
    got = os.path.basename(os.path.realpath(root))
    if got != pin["revision"]:
        raise SystemExit("%s resolved to revision %s, pinned %s" % (pin["repo"], got, pin["revision"]))
    d = os.path.join(root, pin["subfolder"]) if pin.get("subfolder") else root
    verify_files(d, pin["sha256"], "%s at pinned revision %s" % (pin["repo"], pin["revision"]))
    return d


def verify_files(d, sha256, what):
    """Hard-fail unless every {path relative to d: sha256} file exists under d with that digest."""
    for rel in sorted(sha256):
        path = os.path.join(d, rel)
        if not os.path.isfile(path):
            raise SystemExit("%s: %s missing" % (what, rel))
        have = sha256_file(path)
        if have != sha256[rel]:
            raise SystemExit("%s: %s sha256 %s != expected %s" % (what, rel, have, sha256[rel]))


def checkpoint_digests(checkpoint, ident):
    """{file: sha256} a load of `checkpoint` must match: the Laya pin for 'multilingual', else the weights and config
    hashed into `ident` (identity(checkpoint)), so the bytes loaded are the bytes the provenance records."""
    if checkpoint == "multilingual":
        return dict(pins.LAYA["sha256"])
    out = {"model.safetensors": ident["model_sha256"], "rl_agent_config.json": ident["config_sha256"]}
    missing = sorted(rel for rel, h in out.items() if h is None)
    if missing:
        raise SystemExit("checkpoint %s lacks %s"
                         % (os.path.basename(os.path.normpath(checkpoint)), ", ".join(missing)))
    return out


def identity(checkpoint):
    """Provenance of a model for reports: the Laya pin for 'multilingual', else the sha256 of a local checkpoint dir's
    weights and config (None for a file the dir lacks). Hashes only; never downloads or loads anything."""
    if checkpoint == "multilingual":
        return {"repo": pins.LAYA["repo"], "subfolder": pins.LAYA["subfolder"], "revision": pins.LAYA["revision"],
                "model_sha256": pins.LAYA["sha256"]["model.safetensors"]}
    out = {"local": True}
    for key, rel in (("model_sha256", "model.safetensors"), ("config_sha256", "rl_agent_config.json")):
        path = os.path.join(checkpoint, rel)
        out[key] = sha256_file(path) if os.path.isfile(path) else None
    return out


def check_revision(pin):
    infos = snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=pin["allow_patterns"],
                              dry_run=True)
    if not infos:
        raise SystemExit("%s@%s: no files match %s" % (pin["repo"], pin["revision"], pin["allow_patterns"]))
    for info in infos:
        if info.commit_hash != pin["revision"]:
            raise SystemExit("%s/%s resolves to commit %s, pinned %s"
                             % (pin["repo"], info.filename, info.commit_hash, pin["revision"]))
    return len(infos)


def _size(d):
    n = total = 0
    for dirpath, _, files in os.walk(d):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):                        # follows the snapshot symlinks into blobs/
                n, total = n + 1, total + os.path.getsize(path)
    return n, total


def main():
    d = None
    for pin in (pins.LAYA, pins.BERTURK):
        check_revision(pin)
        d = fetch(pin)
        n, total = _size(d)
        print("%s@%s -> %s (%d files, %.0f MB)" % (pin["repo"], pin["revision"], d, n, total / 1e6))
    print("free disk: %.1f GB" % (shutil.disk_usage(d).free / 1e9))


if __name__ == "__main__":
    main()
