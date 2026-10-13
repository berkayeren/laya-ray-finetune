"""Frozen Multi3NLU++ splits: pinned raw folds -> deterministic Laya JSONL -> sha256 manifest -> shared verifier.

Producer: `prepare()` verifies the 40 raw fold hashes, validates every raw row, hard-fails on any duplicate uid
across banking + HOTELS (D-14) or on drift of the banking-only OOD tag set, builds every split in memory
(four banking splits plus the HOTELS out-of-domain probe) and the report-only duplicate audit (pilot.audit, DATA-04)
under audit/, refuses to overwrite a differing data/splits.json (unless rewrite_manifest), then writes each file
atomically with splits.json last.
Consumer: `verify_split_file()` is the only split verifier; evaluate.py and ray_train.py call it right after argument
parsing and hard-fail (SystemExit) on any mismatch, including a manifest built from other question text, tag order,
schema or dataset pin (check_manifest_current); there is no override flag or environment variable.
`verify_audit_file()` applies the same rules to the audit outputs (a later dedup sensitivity row reads dup_flags.json).

Stdlib only at import time (huggingface_hub is imported lazily in fetch_raw), so data tests never load torch.
"""
import hashlib
import json
import os

from intents import INTENTS, QUESTIONS
from pilot import pins
from pilot.audit import build_audit, render_markdown

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "data", "splits.json")
SCHEMA_VERSION = 1
MANIFEST_NAME = "splits.json"

# Val = folds 14-15 (model selection); final models retrain on 0-15; test = 16-19 is never trained on.
# hotels_probe = every Turkish HOTELS message (all 20 folds, D-09), all 48 banking tags false (D-08); its role is
# never "train", so ray_train.py refuses it. Only the banking-only tags (manifest ood_scoring_tags) are valid gold.
SPLITS = [
    {"name": "selection_train", "domain": "banking", "folds": list(range(14)), "role": "train"},
    {"name": "val", "domain": "banking", "folds": [14, 15], "role": "val"},
    {"name": "final_train", "domain": "banking", "folds": list(range(16)), "role": "train"},
    {"name": "test", "domain": "banking", "folds": [16, 17, 18, 19], "role": "test"},
    {"name": "hotels_probe", "domain": "hotels", "folds": list(range(20)), "role": "ood_probe"},
]


AUDIT_FILES = {"audit_json": "audit/audit.json", "audit_md": "audit/audit.md", "dup_flags_json": "audit/dup_flags.json"}


def _json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def questions_sha256():
    return sha256_bytes(json.dumps(QUESTIONS, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def fetch_raw():
    """Download the pinned Turkish banking + hotels folds; return the snapshot root."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=pins.DATASET_REPO, repo_type="dataset", revision=pins.DATASET_REVISION,
                             allow_patterns=pins.DATASET_PATTERNS)
    got = os.path.basename(os.path.realpath(root))
    if got != pins.DATASET_REVISION:
        raise SystemExit("%s resolved to revision %s, pinned %s" % (pins.DATASET_REPO, got, pins.DATASET_REVISION))
    return root


def verify_raw(root, raw_sha256):
    for rel in sorted(raw_sha256):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            raise SystemExit("raw fold %s missing under %s" % (rel, root))
        have = sha256_file(path)
        if have != raw_sha256[rel]:
            raise SystemExit("raw fold %s sha256 %s != pinned %s" % (rel, have, raw_sha256[rel]))


def load_fold(root, domain, fold):
    path = os.path.join(root, "turkish", domain, "fold%d.json" % fold)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit("%s fold %d: expected a JSON list, got %s" % (domain, fold, type(raw).__name__))
    rows = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise SystemExit("%s fold %d row %d: expected an object, got %s" % (domain, fold, i, type(r).__name__))
        uid, text, intents = r.get("uid"), r.get("text"), r.get("intents")
        if not isinstance(uid, str) or not uid:
            raise SystemExit("%s fold %d row %d: uid must be a non-empty string, got %r" % (domain, fold, i, uid))
        if not isinstance(text, str) or not text:
            raise SystemExit("%s fold %d uid %s: text must be a non-empty string" % (domain, fold, uid))
        # no-intent rows OMIT the key in the raw folds; they are kept as all-negative rows (D-10)
        if intents is not None and (not isinstance(intents, list) or not all(isinstance(t, str) for t in intents)):
            raise SystemExit("%s fold %d uid %s: intents must be a list of strings, got %r" % (domain, fold, uid, intents))
        gold = sorted(set(intents or []))                    # hotels: hotel tags, kept only for counts/derivation
        if domain == "banking":
            for t in gold:
                if t not in INTENTS:
                    raise SystemExit("%s fold %d uid %s: unknown tag %r" % (domain, fold, uid, t))
        rows.append({"uid": uid, "text": text, "gold": gold, "fold": fold, "domain": domain})
    return rows


def to_case(row, gold=None):
    """Laya training schema with all 48 tags labelled (no negative sampling, D-07).

    gold overrides row["gold"]; HOTELS probe rows pass an empty gold so every banking tag is false (D-08).
    """
    gold = set(row["gold"] if gold is None else gold)
    return {"state": row["text"], "uid": row["uid"],
            "questions": {k: QUESTIONS[k] for k in INTENTS},
            "expected": {k: k in gold for k in INTENTS}}


def assert_unique_uids(rows):
    """Hard-fail (D-14) if any uid repeats across the raw banking + HOTELS rows (each raw message once).

    uid-keyed consumers (evaluate.py builds dicts by uid) would silently collapse colliding rows. uids are not
    namespaced: banking uses bank_ and HOTELS uses hotel_ prefixes, so the real data has zero collisions.
    """
    seen = {}
    for r in rows:
        seen.setdefault(r["uid"], []).append("%s fold %d" % (r["domain"], r["fold"]))
    dups = sorted(u for u, where in seen.items() if len(where) > 1)
    if dups:
        raise SystemExit("%d duplicate uid(s) in the raw banking + hotels folds: %s%s"
                         % (len(dups), "; ".join("%s (%s)" % (u, ", ".join(seen[u])) for u in dups[:10]),
                            " ..." if len(dups) > 10 else ""))
    return {"n_rows": len(rows), "n_unique": len(seen)}


def build(root, raw_sha256, include_no_intent=True, ood_tags=None):
    """Verify raw folds, then render every split in memory. Returns ({relpath: bytes}, manifest).

    include_no_intent=False leaves banking rows with no gold tags out of every banking split (D-10); the default
    keeps them as all-negative rows. The choice is recorded as manifest["options"]["include_no_intent"]. HOTELS
    probe rows are never dropped. The banking-only tags (INTENTS minus every HOTELS tag) are re-derived from the
    data; if ood_tags is given (pins.OOD_SCORING_TAGS) and differs, prep hard-fails before anything is written.
    """
    verify_raw(root, raw_sha256)
    cache = {}
    for key in sorted({(s["domain"], fold) for s in SPLITS for fold in s["folds"]}):
        cache[key] = load_fold(root, *key)                   # validate every raw row before rendering anything
    uid_check = assert_unique_uids([r for key in sorted(cache) for r in cache[key]])
    hotel_tags = {t for (domain, _), rows in cache.items() if domain == "hotels" for r in rows for t in r["gold"]}
    derived = sorted(set(INTENTS) - hotel_tags)
    if ood_tags is not None and sorted(ood_tags) != derived:
        raise SystemExit("banking-only tag set drifted: derived %s, pinned %s" % (derived, sorted(ood_tags)))
    files, entries = {}, {}
    # D-12/D-13: the audit reads the raw rows and only reports; split membership below never depends on it
    audit, flags = build_audit([r for key in sorted(cache) if key[0] == "banking" for r in cache[key]],
                               [r for key in sorted(cache) if key[0] == "hotels" for r in cache[key]])
    files["audit/audit.json"] = _json_bytes(audit)
    files["audit/audit.md"] = render_markdown(audit, flags).encode("utf-8")
    files["audit/dup_flags.json"] = _json_bytes(flags)
    audit_entries = {name: {"path": rel, "sha256": sha256_bytes(files[rel])}
                     for name, rel in AUDIT_FILES.items()}
    for s in SPLITS:
        rows = []
        for fold in s["folds"]:                              # ascending folds, original in-fold order
            rows += [r for r in cache[(s["domain"], fold)]
                     if include_no_intent or r["gold"] or s["domain"] != "banking"]
        probe = s["domain"] == "hotels"
        data = "".join(json.dumps(to_case(r, gold=() if probe else None), ensure_ascii=False) + "\n"
                       for r in rows).encode("utf-8")
        rel = "splits/%s.jsonl" % s["name"]
        files[rel] = data
        entries[s["name"]] = {
            "path": rel, "domain": s["domain"], "folds": list(s["folds"]), "role": s["role"],
            "n_messages": len(rows), "n_no_intent": sum(not r["gold"] for r in rows),
            "sha256": sha256_bytes(data),
            "uids_sha256": sha256_bytes("\n".join(r["uid"] for r in rows).encode("utf-8")),
        }
        if probe:
            entries[s["name"]]["gold_valid_tags"] = "ood_scoring_tags"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"repo": pins.DATASET_REPO, "revision": pins.DATASET_REVISION, "license": pins.DATASET_LICENSE,
                    "raw_sha256": dict(raw_sha256)},
        "intents_order": list(INTENTS),
        "questions_sha256": questions_sha256(),
        "options": {"include_no_intent": bool(include_no_intent)},
        "seeds": {"data_prep": None},                       # prep draws no random numbers; neg sampling is per run (Phase 4)
        "ood_scoring_tags": derived,                        # the only valid gold tags on hotels_probe (Pitfall 3)
        "uid_check": uid_check,
        "audit": audit_entries,
        "splits": entries,
    }
    return files, manifest


def serialize_manifest(manifest):
    return _json_bytes(manifest)


def write_atomic(path, data):
    # per-process temp name: a concurrent run never truncates the temp file another run is about to rename
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def prepare(raw_root, out_dir, raw_sha256, rewrite_manifest=False, include_no_intent=True, ood_tags=None):
    files, manifest = build(raw_root, raw_sha256, include_no_intent=include_no_intent, ood_tags=ood_tags)
    manifest_bytes = serialize_manifest(manifest)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not rewrite_manifest:
        with open(manifest_path, "rb") as f:
            old = f.read()
        if old != manifest_bytes:
            raise SystemExit("%s differs from the freshly built manifest; refusing to overwrite frozen splits "
                             "(rerun with --rewrite-manifest only to intentionally refreeze)" % manifest_path)
    for rel in sorted(files):
        path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_atomic(path, files[rel])
    write_atomic(manifest_path, manifest_bytes)            # last: the manifest only ever describes complete files
    for s in SPLITS:
        e = manifest["splits"][s["name"]]
        print("%s: %d messages (%d no-intent) -> %s" % (s["name"], e["n_messages"], e["n_no_intent"],
                                                         os.path.join(out_dir, e["path"])))
    print("audit (report only) -> %s" % os.path.join(out_dir, "audit"))
    return manifest


def _unsafe(rel):
    return os.path.isabs(rel) or rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/")


def check_manifest_current(m, manifest_path):
    """Hard-fail unless manifest `m` was built from the current schema, question text, tag order and dataset pin.

    File hashes alone are not enough: a manifest built from other question text would let train (question text
    from the JSONL rows) and eval (intents.QUESTIONS) silently see different model inputs.
    """
    stale = [k for k, have, want in (("schema_version", m.get("schema_version"), SCHEMA_VERSION),
                                     ("questions_sha256", m.get("questions_sha256"), questions_sha256()),
                                     ("intents_order", m.get("intents_order"), list(INTENTS)),
                                     ("dataset.revision", (m.get("dataset") or {}).get("revision"),
                                      pins.DATASET_REVISION))
             if have != want]
    if stale:
        raise SystemExit("%s was built from different code/pins (%s); regenerate with `make data`"
                         % (manifest_path, ", ".join(stale)))


def _registered(path, manifest_path, section, what, data=None):
    """Shared lookup: `path` must be registered under manifest[section] (safe relative paths) with a matching sha256.

    data: the bytes the caller already read from `path` and will parse; they are hashed instead of re-reading the
    file, so what is verified is exactly what is consumed (no verify-then-reread window).
    """
    if not os.path.isfile(manifest_path):
        raise SystemExit("manifest %s not found; run `make data`" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    check_manifest_current(m, manifest_path)
    entries = m.get(section)
    if not isinstance(entries, dict):
        raise SystemExit("%s has no %s section; run `make data`" % (manifest_path, section))
    for name, e in sorted(entries.items()):
        if _unsafe(e["path"]):
            raise SystemExit("%s: %s entry %s has unsafe path %r (absolute or '..')"
                             % (manifest_path, section, name, e["path"]))
    base = os.path.dirname(os.path.realpath(manifest_path))
    rel = os.path.relpath(os.path.realpath(path), base).replace(os.sep, "/")
    entry = next((e for _, e in sorted(entries.items()) if e["path"] == rel), None)
    if entry is None:
        raise SystemExit("%s is not %s registered in %s; run `make data`" % (path, what, manifest_path))
    have = sha256_file(path) if data is None else sha256_bytes(data)
    if have != entry["sha256"]:
        raise SystemExit("%s sha256 %s != %s registered in %s; regenerate with `make data`"
                         % (path, have, entry["sha256"], manifest_path))
    return entry


def verify_audit_file(path, manifest_path=DEFAULT_MANIFEST):
    """Hard-fail unless `path` is an audit output (audit.json, audit.md, dup_flags.json) with its registered sha256."""
    return _registered(path, manifest_path, "audit", "an audit file")


def verify_split_file(path, manifest_path=DEFAULT_MANIFEST, roles=None, data=None):
    """Hard-fail unless `path` is a split registered in the manifest with a matching sha256 (and allowed role).

    Pass data=<bytes read from path> to verify the exact bytes you are about to parse (see _registered).
    """
    entry = _registered(path, manifest_path, "splits", "a split", data=data)
    if roles is not None and entry["role"] not in roles:
        raise SystemExit("%s has role %s; expected one of %s" % (path, entry["role"], ", ".join(roles)))
    return entry
