"""Frozen Multi3NLU++ splits: pinned raw folds -> deterministic Laya JSONL -> sha256 manifest -> shared verifier.

Producer: `prepare()` verifies the 40 raw fold hashes, validates every raw row, hard-fails on any duplicate uid
across banking + HOTELS or on drift of the banking-only OOD tag set, builds every split in memory (four banking splits
plus the HOTELS out-of-domain probe) and the report-only duplicate audit (pilot.audit) under audit/, refuses to
overwrite a differing data/splits.json (unless rewrite_manifest), then writes each file atomically with splits.json
last.
Consumer: `verify_split_file()` is the only split verifier; every script that reads a split (score.py, ray_train.py,
berturk_train.py, latency.py, fewshot.py, and report.py through pilot.dumps) calls it before parsing and hard-fails
(SystemExit) on any mismatch, including a manifest built from other question text, tag order,
schema or dataset pin, or one whose split names, roles, paths, folds or domains differ from SPLITS
(check_manifest_current); there is no override flag or environment variable. --manifest only selects another
prepare-written manifest (e.g. the --no-include-no-intent variant), and reports record whether it is the committed one.
`verify_audit_file()` applies the same rules to the audit outputs (dup_flags.json is for a dedup sensitivity check).
The content checks (holds_committed, require_committed_files, refuse_held_out) key on the committed split files, so a
manifest copy that relabels held-out bytes cannot train on them or open them to a fine-tuned model.

Stdlib only at import time (huggingface_hub is imported lazily in fetch_raw), so data tests never load torch.
"""
import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any

from intents import INTENTS, QUESTIONS
from pilot import pins
from pilot.audit import build_audit, render_markdown

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "data", "splits.json")
SCHEMA_VERSION = 1
MANIFEST_NAME = "splits.json"

# Val = folds 14-15 (model selection); final models retrain on 0-15; test = 16-19 is never trained on.
# hotels_probe = every Turkish HOTELS message (all 20 folds), all 48 banking tags false; its role is
# never "train", so ray_train.py refuses it. Only the banking-only tags (manifest ood_scoring_tags) are valid gold.
SPLITS = [
    {"name": "selection_train", "domain": "banking", "folds": list(range(14)), "role": "train"},
    {"name": "val", "domain": "banking", "folds": [14, 15], "role": "val"},
    {"name": "final_train", "domain": "banking", "folds": list(range(16)), "role": "train"},
    {"name": "test", "domain": "banking", "folds": [16, 17, 18, 19], "role": "test"},
    {"name": "hotels_probe", "domain": "hotels", "folds": list(range(20)), "role": "ood_probe"},
]
# the training split whose models each evaluation split scores (val -> model selection runs, test -> final models)
TRAINED_ON = {"val": "selection_train", "test": "final_train"}


def audit_layout() -> dict[str, tuple]:
    """Audit partitions and dup-flag pairs derived from SPLITS, the only place fold boundaries are written down.

    partitions: the banking splits in SPLITS order that do not overlap an earlier one (selection_train, val, test);
    flag_pairs: (evaluation split, its folds, the TRAINED_ON split, its folds).
    """
    by = {s["name"]: s for s in SPLITS}
    partitions, used = [], set()
    for s in SPLITS:
        if s["domain"] == "banking" and not used & set(s["folds"]):
            partitions.append((s["name"], tuple(s["folds"])))
            used |= set(s["folds"])
    pairs = tuple((ev, tuple(by[ev]["folds"]), tr, tuple(by[tr]["folds"])) for ev, tr in sorted(TRAINED_ON.items()))
    return {"partitions": tuple(partitions), "flag_pairs": pairs}


AUDIT_FILES = {"audit_json": "audit/audit.json", "audit_md": "audit/audit.md", "dup_flags_json": "audit/dup_flags.json"}


def _json_bytes(obj: Any) -> bytes:
    """JSON file bytes: indent 2, sorted keys, ensure_ascii False, one trailing newline."""
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: str) -> str:
    """Hex sha256 of the file at `path`, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def parse_jsonl(raw: bytes) -> list[Any]:
    """Rows of JSONL bytes, split on "\n" only (the producer's terminator), stripped, blank and '#' lines skipped.

    Same rows as laya.train.read_jsonl (text-mode lines). Never str.splitlines(): it also splits on U+0085, U+2028,
    U+2029 and other separators that json.dumps(ensure_ascii=False) leaves raw inside message text.
    """
    lines = (l.strip() for l in raw.decode("utf-8").split("\n"))
    return [json.loads(l) for l in lines if l and not l.startswith("#")]


def repo_relpath(path: str) -> str:
    """`path` relative to the repo root, for reports and checkpoints; never a local or cwd-relative path.

    Outside the repo only the base name is kept ("<outside-repo>/NAME"); the recorded sha256 is the identity.
    """
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(REPO_ROOT))
    if rel == ".." or rel.startswith(".." + os.sep) or os.path.isabs(rel):
        return "<outside-repo>/" + os.path.basename(os.path.normpath(path))
    return rel.replace(os.sep, "/")


def questions_sha256() -> str:
    """sha256 of the sorted JSON of intents.QUESTIONS (the model input), recorded in the manifest and every dump."""
    return sha256_bytes(json.dumps(QUESTIONS, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def fetch_raw() -> str:
    """Download the pinned Turkish banking + hotels folds; return the snapshot root."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=pins.DATASET_REPO, repo_type="dataset", revision=pins.DATASET_REVISION,
                             allow_patterns=pins.DATASET_PATTERNS)
    got = os.path.basename(os.path.realpath(root))
    if got != pins.DATASET_REVISION:
        raise SystemExit("%s resolved to revision %s, pinned %s" % (pins.DATASET_REPO, got, pins.DATASET_REVISION))
    return root


def verify_raw(root: str, raw_sha256: dict[str, str]) -> None:
    """SystemExit unless every pinned raw fold exists under `root` with its pinned sha256."""
    for rel in sorted(raw_sha256):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            raise SystemExit("raw fold %s missing under %s" % (rel, root))
        have = sha256_file(path)
        if have != raw_sha256[rel]:
            raise SystemExit("raw fold %s sha256 %s != pinned %s" % (rel, have, raw_sha256[rel]))


def load_fold(root: str, domain: str, fold: int) -> list[dict[str, Any]]:
    """Validated rows {uid, text, gold, fold, domain} of one raw fold; SystemExit on a malformed row or, for banking,
    an unknown tag."""
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
        # no-intent rows OMIT the key in the raw folds; they are kept as all-negative rows (build's include_no_intent)
        if intents is not None and (not isinstance(intents, list) or not all(isinstance(t, str) for t in intents)):
            raise SystemExit("%s fold %d uid %s: intents must be a list of strings, got %r" % (domain, fold, uid, intents))
        gold = sorted(set(intents or []))                    # hotels: hotel tags, kept only for counts/derivation
        if domain == "banking":
            for t in gold:
                if t not in INTENTS:
                    raise SystemExit("%s fold %d uid %s: unknown tag %r" % (domain, fold, uid, t))
        rows.append({"uid": uid, "text": text, "gold": gold, "fold": fold, "domain": domain})
    return rows


def to_case(row: dict[str, Any], gold: Iterable[str] | None = None) -> dict[str, Any]:
    """Laya training schema with all 48 tags labelled (no prepare-time negative sampling).

    gold overrides row["gold"]; HOTELS probe rows pass an empty gold so every banking tag is false.
    """
    gold = set(row["gold"] if gold is None else gold)
    return {"state": row["text"], "uid": row["uid"],
            "questions": {k: QUESTIONS[k] for k in INTENTS},
            "expected": {k: k in gold for k in INTENTS}}


def assert_unique_uids(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Hard-fail if any uid repeats across the raw banking + HOTELS rows (each raw message once); returns the manifest's
    uid_check {n_rows, n_unique}.

    uid-keyed consumers (score.assemble and pilot.dumps.read_gold build dicts by uid) would silently collapse
    colliding rows. uids are not
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


def build(root: str, raw_sha256: dict[str, str], include_no_intent: bool = True,
          ood_tags: Iterable[str] | None = None) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Verify raw folds, then render every split in memory. Returns ({relpath: bytes}, manifest).

    include_no_intent=False leaves banking rows with no gold tags out of every banking split; the default
    keeps them as all-negative rows. The choice is recorded as manifest["options"]["include_no_intent"]. HOTELS
    probe rows are never dropped. The banking-only tags (INTENTS minus every HOTELS tag) are re-derived from the
    data; if ood_tags is given (pins.OOD_SCORING_TAGS) and differs, prep hard-fails before anything is written.
    """
    keys = sorted({(s["domain"], fold) for s in SPLITS for fold in s["folds"]})
    want = {"turkish/%s/fold%d.json" % key for key in keys}
    if set(raw_sha256) != want:                             # every fold read below must be a pinned, verified one
        raise SystemExit("raw_sha256 does not pin exactly the %d folds the splits read: missing %s, extra %s"
                         % (len(want), sorted(want - set(raw_sha256)), sorted(set(raw_sha256) - want)))
    verify_raw(root, raw_sha256)
    cache = {}
    for key in keys:
        cache[key] = load_fold(root, *key)                   # validate every raw row before rendering anything
    uid_check = assert_unique_uids([r for key in sorted(cache) for r in cache[key]])
    hotel_tags = {t for (domain, _), rows in cache.items() if domain == "hotels" for r in rows for t in r["gold"]}
    derived = sorted(set(INTENTS) - hotel_tags)
    if ood_tags is not None and sorted(ood_tags) != derived:
        raise SystemExit("banking-only tag set drifted: derived %s, pinned %s" % (derived, sorted(ood_tags)))
    files, entries = {}, {}
    # the audit reads the raw rows and only reports; split membership below never depends on it.
    # The dup flags cover only the banking rows the splits keep (no-intent rows drop out with include_no_intent off).
    banking = [r for key in sorted(cache) if key[0] == "banking" for r in cache[key]]
    audit, flags = build_audit(banking, [r for key in sorted(cache) if key[0] == "hotels" for r in cache[key]],
                               flag_rows=[r for r in banking if include_no_intent or r["gold"]], **audit_layout())
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
            # hotels_probe: n_no_intent counts rows with no raw HOTELS intent (every probe row is banking-negative),
            # not banking no-intent rows; kept under this key so the frozen manifest bytes do not change
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
        "seeds": {"data_prep": None},                       # prep draws no random numbers; neg sampling is per run
        "ood_scoring_tags": derived,                        # the only valid gold tags on hotels_probe
        "uid_check": uid_check,
        "audit": audit_entries,
        "splits": entries,
    }
    return files, manifest


def serialize_manifest(manifest: dict[str, Any]) -> bytes:
    """The manifest's file bytes (the frozen data/splits.json format)."""
    return _json_bytes(manifest)


def write_atomic(path: str, data: bytes) -> None:
    """Write `data` to `path` through a per-process temp file, fsync it, rename it over `path` and fsync the dir."""
    # per-process temp name: a concurrent run never truncates the temp file another run is about to rename
    tmp = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):                             # never leave a stray temp file next to the outputs
            os.unlink(tmp)
        raise
    fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(fd)                                        # make the rename itself durable
    finally:
        os.close(fd)


def prepare(raw_root: str, out_dir: str, raw_sha256: dict[str, str], rewrite_manifest: bool = False,
            include_no_intent: bool = True, ood_tags: Iterable[str] | None = None) -> dict[str, Any]:
    """Build every split, refuse to overwrite a differing OUT/splits.json unless rewrite_manifest, write the files
    atomically (the manifest last), print the counts and return the manifest."""
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


def _unsafe(rel: str) -> bool:
    """True for an absolute manifest path or one with a '..' component."""
    return os.path.isabs(rel) or rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/")


def check_manifest_current(m: dict[str, Any], manifest_path: str) -> None:
    """Hard-fail unless manifest `m` was built from the current schema, question text, tag order and dataset pin,
    and registers exactly the SPLITS (role, path, folds, domain) and AUDIT_FILES this code writes.

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
    # a manifest may differ only in what prepare() varies (options, counts, hashes): never in which file has which
    # role, so no manifest can relabel test as train or the OOD probe as test
    want = {s["name"]: s for s in SPLITS}
    splits = m.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(want):
        raise SystemExit("%s registers splits %s, expected %s; regenerate with `make data`"
                         % (manifest_path, sorted(splits) if isinstance(splits, dict) else splits, sorted(want)))
    for name in sorted(splits):
        e, s = splits[name], want[name]
        if (e.get("role"), e.get("path"), e.get("folds"), e.get("domain")) != \
                (s["role"], "splits/%s.jsonl" % name, list(s["folds"]), s["domain"]):
            raise SystemExit("%s: split %s does not match pilot.data.SPLITS (role/path/folds/domain); "
                             "regenerate with `make data`" % (manifest_path, name))
    audit = m.get("audit")
    if not isinstance(audit, dict) or {k: (e or {}).get("path") for k, e in audit.items()} != AUDIT_FILES:
        raise SystemExit("%s: audit entries do not match pilot.data.AUDIT_FILES; regenerate with `make data`"
                         % manifest_path)


def manifest_is_committed(manifest_path: str) -> bool:
    """True when `manifest_path` has the bytes of the repo's data/splits.json (reports flag any other manifest)."""
    return os.path.isfile(DEFAULT_MANIFEST) and sha256_file(manifest_path) == sha256_file(DEFAULT_MANIFEST)


def committed_content(name: str) -> tuple[dict[str, Any], set[tuple[Any, Any]]]:
    """(the `name` entry of the repo's data/splits.json, the set of (uid, message) pairs of that committed split file
    under data/). The set is empty when the file is absent (no `make data`); the entry's hashes still key the check."""
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        committed = json.load(f)["splits"][name]
    path = os.path.join(os.path.dirname(DEFAULT_MANIFEST), committed["path"])
    if not os.path.isfile(path):
        return committed, set()
    with open(path, "rb") as f:
        return committed, {(c.get("uid"), c.get("state")) for c in parse_jsonl(f.read())}


def holds_committed(entry: dict[str, Any], cases: Iterable[dict[str, Any]], name: str) -> bool:
    """True when a split (its entry in any manifest, and its parsed rows `cases`) holds the committed `name` split's
    content: its sha256 or uids_sha256 is the committed entry's, or any (uid, message) pair of `cases` is in the
    committed file. Keyed on the content, never on the manifest bytes or the role. A uid alone is not the key: the
    tests' synthetic splits reuse the uid scheme with other messages."""
    committed, pairs = committed_content(name)
    return entry.get("sha256") == committed["sha256"] or entry.get("uids_sha256") == committed["uids_sha256"] or \
        any((c.get("uid"), c.get("state")) in pairs for c in cases)


def require_committed_files(where: str, manifest_path: str, names: Iterable[str]) -> None:
    """SystemExit when a content check could not key on the (uid, message) pairs (fail closed):
    holds_committed reads them from the committed data/splits/<name>.jsonl and, when that file is absent, keys only on
    the entry hashes, which a relabel of all rows but one changes. Refused for a non-committed manifest prepared from
    the pinned dataset (its dataset.raw_sha256 is the committed manifest's: a manifest copy, or prepare_data.py --out
    OTHER without `make data`) while any committed `names` file is missing. The committed manifest needs none (its
    splits are the committed bytes); a manifest of another snapshot (the offline suite's synthetic data) keys on the
    hashes. Like the pair key itself, this catches relabelled bytes, not a forged manifest. Shared by report.py's
    content checks and score.py's protocol-gate trigger; DEFAULT_MANIFEST is read at call time."""
    if manifest_is_committed(manifest_path):
        return
    with open(manifest_path, encoding="utf-8") as f:
        raw = (json.load(f).get("dataset") or {}).get("raw_sha256")
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        if raw != (json.load(f).get("dataset") or {}).get("raw_sha256"):
            return
    for name in names:
        committed, pairs = committed_content(name)
        if not pairs:
            raise SystemExit("%s: the committed %s is missing, so %s (prepared from the pinned dataset, not the "
                             "committed manifest) cannot be content-checked against the committed splits; run `make "
                             "data` (or drop --manifest)" % (where, committed["path"], manifest_path))


HELD_OUT = {"selection_train": ("val", "test"), "final_train": ("test",)}   # final_train holds the val folds by design


def refuse_held_out(path: str, entry: dict[str, Any], cases: Iterable[dict[str, Any]]) -> None:
    """SystemExit when a train split (its entry in any manifest, and its parsed rows `cases`) holds committed held-out
    content (holds_committed): committed test content in any train split, and committed val content in
    selection_train too. A manifest copy that registers the val or test bytes as a train split therefore cannot train
    a model on them. Shared by ray_train.py and berturk_train.py, before any fetch, load or Ray."""
    name = os.path.splitext(os.path.basename(entry["path"]))[0]
    for held in HELD_OUT.get(name, ("val", "test")):
        if holds_committed(entry, cases, held):
            raise SystemExit("%s (%s) holds committed %s content; no model trains on a held-out split, so a manifest "
                             "may not register it as a train split; train on the committed %s (drop --manifest, or "
                             "run `make data`)" % (path, name, held, name))


def _registered(path: str, manifest_path: str, section: str, what: str, data: bytes | None = None) -> dict[str, Any]:
    """Shared lookup: `path` must be registered under manifest[section] (safe relative paths) with a matching sha256.

    data: the bytes the caller already read from `path` and will parse; they are hashed instead of re-reading the
    file, so what is verified is exactly what is consumed (no verify-then-reread window). Returns the manifest entry.
    """
    if not os.path.isfile(manifest_path):
        raise SystemExit("manifest %s not found; run `make data`" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    entries = m.get(section)
    if not isinstance(entries, dict):
        raise SystemExit("%s has no %s section; run `make data`" % (manifest_path, section))
    for name, e in sorted(entries.items()):
        if _unsafe(e["path"]):
            raise SystemExit("%s: %s entry %s has unsafe path %r (absolute or '..')"
                             % (manifest_path, section, name, e["path"]))
    check_manifest_current(m, manifest_path)
    if data is None and not os.path.isfile(path):
        raise SystemExit("%s not found; check the path, or run `make data` to write the registered files" % path)
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


def verify_audit_file(path: str, manifest_path: str = DEFAULT_MANIFEST) -> dict[str, Any]:
    """The manifest entry of `path`; hard-fail unless it is an audit output (audit.json, audit.md, dup_flags.json) with
    its registered sha256."""
    return _registered(path, manifest_path, "audit", "an audit file")


def verify_split_file(path: str, manifest_path: str = DEFAULT_MANIFEST, roles: Iterable[str] | None = None,
                      data: bytes | None = None) -> dict[str, Any]:
    """The manifest entry of `path`; hard-fail unless it is a split registered in the manifest with a matching sha256
    (and, when `roles` is given, one of those roles).

    Pass data=<bytes read from path> to verify the exact bytes you are about to parse (see _registered).
    """
    entry = _registered(path, manifest_path, "splits", "a split", data=data)
    if roles is not None and entry["role"] not in roles:
        raise SystemExit("%s has role %s; expected one of %s" % (path, entry["role"], ", ".join(roles)))
    return entry
