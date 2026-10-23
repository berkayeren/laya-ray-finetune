"""The per-message logit dump: the one record every metric of every system is computed from (EVAL-01).

A dump is results/dumps/<run_id>/<split>.npz plus a JSON sidecar <split>.json next to it.
- The npz holds exactly two arrays: `logits`, float32 [n, 48] with columns in INTENTS order, and `uids`, a numpy
  unicode array. Values are never rounded (D-01); rounding happens only when a number is displayed.
- Logits, not probabilities, are stored (D-02): P(tag) = sigmoid(logit / T), so one inference pass serves the shipped
  temperature, T = 1 and any later val-fitted temperature without rescoring. Baselines store clipped logits, T = 1.
- Rows of a full dump stay in split-file order, so sha256 of the newline-joined uids equals the manifest's
  uids_sha256 and proves complete, in-order coverage without reading any message text.
- The sidecar records provenance (split path/role/sha256, questions_sha256, the INTENTS-order hash, system identity,
  seed, run_id) and dump_sha256, the sha256 of the npz bytes. It holds repo-relative paths and hashes only. A dump of
  a ray_train.py checkpoint also records its training seed and a training block whose config_sha256_sans_seed
  (training_digest) is shared by the seeds of one recipe (03-REVIEW WR-05).

load_dump is the only reader and hard-fails (SystemExit, no override; D-04) on any mismatch: npz bytes vs
dump_sha256, other keys, dtypes or shapes, a pickled array (numpy loads use allow_pickle=False, because a pickled
array in a committed file would run arbitrary code), a split, questions or tag order that no longer matches the
manifest and code, a run dir renamed away from its run_id, or a full dump whose uids are not the split's.
read_gold returns the verified split's gold labels in dump-row order; it reads only the `expected` field.

A dump of the out-of-domain probe (hotels_probe, role ood_probe; EVAL-05) is readable too (READ_ROLES), with every
check above. Its gold is valid on the manifest's ood_scoring_tags only (the banking-only tags, all of them negative on
every HOTELS message), so gold_valid_tags names those columns and every 48-tag metric refuses the dump
(report.run_report, report.compare); only report.py --confidence turns it into the out-of-domain false-positive rate.
score.py does not write one before Phase 8, which opens it for hotels_probe behind its protocol gate.

Numpy and stdlib only: no torch, Ray or laya.

usage: written by score.py (write_dump), read by report.py (load_dump, read_gold, gold_valid_tags)
"""
import hashlib
import io
import json
import os
import zipfile

import numpy as np

from intents import INTENTS
from pilot import pins
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, check_manifest_current, manifest_is_committed, parse_jsonl,
                        questions_sha256, sha256_bytes, verify_split_file, write_atomic)

SCHEMA_VERSION = 1
DUMP_KEYS = ("logits", "uids")
CLIP_EPS = 1e-6                                             # baseline probabilities are clipped to [eps, 1 - eps]
DUMP_ROOT = os.path.join(REPO_ROOT, "results", "dumps")
LOGIT_NOTE = "margin z_true - z_false per tag; P(tag) = sigmoid(logit / T)"
TAGS = list(INTENTS)
EVAL_ROLES = ("val", "test")
# the roles load_dump and read_gold read: a hotels_probe dump (role ood_probe) is readable, but only its
# ood_scoring_tags columns are valid gold (gold_valid_tags); score.py does not write one before Phase 8
READ_ROLES = EVAL_ROLES + ("ood_probe",)
OOD_GOLD = "ood_scoring_tags"                               # the manifest entry's gold_valid_tags value on hotels_probe


def intents_order_sha256():
    """sha256 of the INTENTS tag order (the dump's column order)."""
    return sha256_bytes(json.dumps(TAGS, ensure_ascii=False).encode("utf-8"))


def uids_sha256(uids):
    """sha256 of the newline-joined uids: the same bytes pilot.data hashes into the manifest's uids_sha256."""
    return sha256_bytes("\n".join(uids).encode("utf-8"))


# the files_sha256 keys every local checkpoint identity must hold (pilot.weights.checkpoint_digests refuses one without
# them, report.binds_every_file does not count it as bound): without tokenizer/tokenizer_config.json or
# encoder/config.json laya 0.4.6 reads the tokenizer or the encoder config from the Hub by the config's encoder name,
# unpinned (04-REVIEW WR-02). The tokenizer config is bound by its normalized JSON (pilot.weights.NORMALIZED_KEY).
LOCAL_REQUIRED_FILES = ("encoder/config.json", "tokenizer/tokenizer_config.json#normalized")


def files_digest(files_sha256):
    """sha256 of the sorted, compact JSON of a local identity's files_sha256 (tokenizer/ and encoder/ digests)."""
    blob = json.dumps(files_sha256, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# the canonical-config keys two seeds of one recipe may differ in (pilot.runs IDENTITY): training_digest leaves them out
SEED_KEYS = ("seed", "neg_seed")


def training_digest(canonical, epoch):
    """config_sha256_sans_seed of a ray_train.py checkpoint (03-REVIEW WR-05): the sha256 of the compact, sorted JSON
    of its canonical run config without SEED_KEYS, plus the checkpoint's epoch. Two seeds of one recipe share it; any
    other hyperparameter, the training split or the epoch changes it, so report.py --compare averages only seeds."""
    blob = json.dumps({"canonical": {k: v for k, v in canonical.items() if k not in SEED_KEYS}, "epoch": epoch},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_id(kind, ident=None):
    """Stable, config-derived run id: majority, prior, laya-zeroshot-<revision[:8]> (the pinned checkpoint) or
    laya-<model_sha256[:12]>-<config_sha256[:8]>-<files_digest(files_sha256)[:8]> (a local checkpoint: weights,
    rl_agent_config.json, which carries the served temperature, max_len and calibration, and tokenizer/ and encoder/,
    so two checkpoints that differ in any file laya.load reads never share a dump dir; 03-REVIEW WR-01, 04-REVIEW
    IN-05). SystemExit on a local identity without config_sha256 or files_sha256."""
    if kind in ("majority", "prior"):
        return kind
    if kind == "laya" and isinstance(ident, dict):
        if ident.get("revision"):
            return "laya-zeroshot-" + ident["revision"][:8]
        if ident.get("local") and ident.get("model_sha256"):
            if not ident.get("config_sha256"):
                raise SystemExit("local checkpoint identity has no config_sha256 (rl_agent_config.json missing?); "
                                 "a local run id names both the weights and the config")
            if ident.get("files_sha256") is None:
                raise SystemExit("local checkpoint identity has no files_sha256 (tokenizer/ and encoder/); recompute "
                                 "it with this pilot.weights.identity")
            return "laya-%s-%s-%s" % (ident["model_sha256"][:12], ident["config_sha256"][:8],
                                      files_digest(ident["files_sha256"])[:8])
    raise SystemExit("unknown system %r; expected majority, prior or laya" % (kind,))


def _sidecar_bytes(sidecar):
    return (json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_dump(run_dir, split_name, logits, uids, sidecar):
    """Write run_dir/<split>.npz then run_dir/<split>.json (last, so a sidecar only describes a complete npz)."""
    uids = list(uids)
    reason = None
    if not isinstance(logits, np.ndarray) or logits.dtype != np.float32:
        reason = "logits must be a float32 numpy array, got %s" % getattr(logits, "dtype", type(logits).__name__)
    elif logits.ndim != 2 or logits.shape[1] != len(TAGS):
        reason = "logits shape %s is not [n, %d]" % (logits.shape, len(TAGS))
    elif logits.shape[0] == 0 or logits.shape[0] != len(uids):
        reason = "%d logit rows for %d uids (need the same, non-zero count)" % (logits.shape[0], len(uids))
    elif not np.isfinite(logits).all():
        reason = "logits hold a non-finite value"
    elif not all(isinstance(u, str) and u for u in uids):
        reason = "every uid must be a non-empty string"
    elif len(set(uids)) != len(uids):
        reason = "duplicate uids"
    if reason is not None:
        raise SystemExit("refusing to write a malformed dump: %s" % reason)
    os.makedirs(run_dir, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, logits=logits, uids=np.asarray(uids, dtype=str))
    data = buf.getvalue()
    path = os.path.join(run_dir, split_name + ".npz")
    write_atomic(path, data)
    side = dict(sidecar, schema_version=SCHEMA_VERSION, dump_sha256=sha256_bytes(data))
    write_atomic(os.path.join(run_dir, split_name + ".json"), _sidecar_bytes(side))
    return path


def load_dump(run_dir, split_name, manifest_path=DEFAULT_MANIFEST):
    """(logits, uids, sidecar, manifest entry) of a verified dump; SystemExit on any mismatch (D-04)."""
    npz = os.path.join(run_dir, split_name + ".npz")
    side_path = os.path.join(run_dir, split_name + ".json")
    for p in (npz, side_path):
        if not os.path.isfile(p):
            raise SystemExit("%s not found; run score.py to write the dump" % p)

    def refuse(what):
        raise SystemExit("%s: %s; rescore with score.py" % (npz, what))

    try:
        with open(side_path, encoding="utf-8") as f:
            side = json.load(f)
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); rescore with score.py" % (side_path, e))
    if not isinstance(side, dict):
        refuse("the sidecar is not a JSON object")
    if side.get("schema_version") != SCHEMA_VERSION:
        refuse("sidecar schema_version %r != %d" % (side.get("schema_version"), SCHEMA_VERSION))
    want_run = os.path.basename(os.path.normpath(run_dir))
    if side.get("run_id") != want_run:
        refuse("sidecar run_id %r != run dir name %r" % (side.get("run_id"), want_run))
    split = side.get("split") if isinstance(side.get("split"), dict) else {}
    if split.get("name") != split_name:
        refuse("sidecar split name %r != %r" % (split.get("name"), split_name))

    with open(npz, "rb") as f:
        raw = f.read()                                      # hash and parse the same bytes (no reread window)
    if sha256_bytes(raw) != side.get("dump_sha256"):
        refuse("npz sha256 %s != sidecar dump_sha256 %s" % (sha256_bytes(raw), side.get("dump_sha256")))
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        keys = set(z.files)
        if keys != set(DUMP_KEYS):
            refuse("npz keys %s, expected exactly %s" % (sorted(keys), list(DUMP_KEYS)))
        logits, uids = z["logits"], z["uids"]
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as e:   # 03-REVIEW IN-03: one line, never a traceback
        text = " ".join(str(e).split())
        if isinstance(e, ValueError) and "allow_pickle" in text:
            refuse("the npz holds a pickled array (object dtype); refusing (%s)" % text)
        refuse("the npz cannot be read (%s: %s)" % (type(e).__name__, text))
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] != len(TAGS):
        refuse("logits are %s %s, expected float32 [n, %d]" % (logits.dtype, logits.shape, len(TAGS)))
    if not np.isfinite(logits).all():
        refuse("logits hold a non-finite value")
    if uids.dtype.kind != "U" or uids.ndim != 1 or len(uids) != logits.shape[0]:
        refuse("uids are %s %s for %d logit rows, expected a unicode vector" % (uids.dtype, uids.shape,
                                                                                logits.shape[0]))
    uids = [str(u) for u in uids]

    if not os.path.isfile(manifest_path):
        raise SystemExit("manifest %s not found; run `make data`" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    check_manifest_current(m, manifest_path)
    entry = m["splits"].get(split_name)
    if entry is None:
        refuse("split %r is not registered in %s" % (split_name, manifest_path))
    if entry["role"] not in READ_ROLES:
        refuse("split %s has role %s; dumps cover role val, test or ood_probe only" % (split_name, entry["role"]))
    for key in ("sha256", "role", "path"):
        if split.get(key) != entry[key]:
            refuse("sidecar split %s %r != manifest %r" % (key, split.get(key), entry[key]))
    if side.get("questions_sha256") != questions_sha256():
        refuse("sidecar questions_sha256 %s != current QUESTIONS %s" % (side.get("questions_sha256"),
                                                                       questions_sha256()))
    if side.get("intents_order_sha256") != intents_order_sha256():
        refuse("sidecar intents_order_sha256 %s != current INTENTS order %s"
               % (side.get("intents_order_sha256"), intents_order_sha256()))

    n_subset = side.get("n_subset")
    if n_subset is None:
        if len(uids) != entry["n_messages"] or uids_sha256(uids) != entry["uids_sha256"]:
            refuse("uids (%d) are not the %d uids of split %s in split-file order (uids_sha256 mismatch)"
                   % (len(uids), entry["n_messages"], split_name))
    elif len(uids) != n_subset or len(set(uids)) != len(uids):
        refuse("subset dump holds %d uids (%d unique), sidecar n_subset %r" % (len(uids), len(set(uids)), n_subset))
    return logits, uids, side, entry


def read_gold(entry, manifest_path, uids):
    """bool [len(uids), 48] gold labels of the verified split, in dump-row order (only `expected` is read)."""
    path = os.path.join(os.path.dirname(os.path.realpath(manifest_path)), entry["path"])
    if not os.path.isfile(path):
        raise SystemExit("%s not found; run `make data` to write the registered splits" % path)
    with open(path, "rb") as f:
        raw = f.read()
    verify_split_file(path, manifest_path=manifest_path, roles=READ_ROLES, data=raw)
    by_uid = {r["uid"]: r["expected"] for r in parse_jsonl(raw)}
    missing = [u for u in uids if u not in by_uid]
    if missing:
        raise SystemExit("%d dump uid(s) are not in %s (e.g. %s); rescore with score.py"
                         % (len(missing), entry["path"], missing[0]))
    return np.array([[bool(by_uid[u].get(t, False)) for t in TAGS] for u in uids], dtype=bool).reshape(len(uids),
                                                                                                       len(TAGS))


def gold_valid_tags(entry, manifest_path=DEFAULT_MANIFEST):
    """The tags whose gold is valid on the split of manifest `entry`, in INTENTS order: all 48 for a banking split (no
    gold_valid_tags key), the manifest's ood_scoring_tags for hotels_probe (gold_valid_tags "ood_scoring_tags": the
    banking-only tags, the only columns an out-of-domain false positive is counted on). SystemExit on any other
    gold_valid_tags value, and on an ood_scoring_tags list that is empty, repeats a tag, names a tag not in INTENTS or,
    under the committed manifest, is not exactly pilot.pins.OOD_SCORING_TAGS (the 22 banking-only tags)."""
    want = entry.get("gold_valid_tags")
    if want is None:
        return list(TAGS)
    if want != OOD_GOLD:
        raise SystemExit("%s: gold_valid_tags %r is unknown; expected %r (hotels_probe) or no key (a banking split)"
                         % (manifest_path, want, OOD_GOLD))
    if not os.path.isfile(manifest_path):
        raise SystemExit("manifest %s not found; run `make data`" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        tags = json.load(f).get(OOD_GOLD)
    if (not isinstance(tags, list) or not tags or len(set(map(str, tags))) != len(tags)
            or any(t not in TAGS for t in tags)):
        raise SystemExit("%s: ood_scoring_tags %r must be a non-empty list of distinct INTENTS tags; regenerate the "
                         "manifest with `make data`" % (manifest_path, tags))
    if manifest_is_committed(manifest_path) and sorted(tags) != sorted(pins.OOD_SCORING_TAGS):
        raise SystemExit("%s: ood_scoring_tags %s != pilot.pins.OOD_SCORING_TAGS %s; the committed manifest's "
                         "banking-only tags drifted from the pins" % (manifest_path, sorted(tags),
                                                                        sorted(pins.OOD_SCORING_TAGS)))
    return [t for t in TAGS if t in set(tags)]
