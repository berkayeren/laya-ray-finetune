"""Repository hygiene: nothing committed may carry message text, a local path or a non-manifest data file.

The pilot is meant to be audited by a bank's reviewers, so the repo must never hold customer or dataset message
text. From data/ only the manifest data/splits.json is tracked (it holds paths, hashes and counts, never text), no
.jsonl file is tracked anywhere, and no tracked text file contains the local home directory. The home directory is
computed at runtime, so this file holds no literal path. The bank-name check cannot be automated (the name is
deliberately absent from the repo) and stays a human review.

Committed logit dumps under results/dumps/<run_id>/{val,test}.{npz,json} and reports under results/reports/ are
allowed because they are the auditable record, and checked here: an .npz must load with allow_pickle=False and hold
exactly a float32 [n, 48] finite "logits" array and a unicode "uids" array whose every uid looks like
bank_<fold>_<index>; a sidecar or report must have no message-text key, no local home path, no message text from the
local splits and no subset run (a subset dump is never committed). The compute checkpoint reports under
results/compute/ get the same sidecar check, so they hold numbers as JSON numbers, env vars only as KEY=VALUE strings
and repo-relative paths only. The postfit records under results/postfit/ (ids, digests and numbers only) get it too,
and only .json files may be tracked there; the same checks cover the BERTurk postfit records and BERTurk dump
sidecars. Training outputs (Ray storage, BERTurk run storage under berturk_runs/, every ckpt*/ checkpoint including
ckpt-berturk-final-s*) are proven gitignored. The generic home-path test skips binary files, so the .npz check is the
guard there. The two legacy reports (100 test messages, no no-intent rows) sit under results/legacy/ next to a README
that labels them as not comparable. The superseded BERTurk selections (the two runs the BERTurk reference check
discarded, pilot.berturk.SUPERSEDED) sit under results/superseded/, each next to a README that labels it superseded
and not comparable, and get the same dump and sidecar checks. The val confidence records under results/confidence/
(ids, digests, repo-relative paths and numbers only) get the sidecar check too, and only <run_id>-val.json and
<run_id>-val.svg may be tracked there. A HOTELS dump results/dumps/<run_id>/hotels_probe.{npz,json} holds
hotel_<fold>_<index> uids only (bank uids only in val and test dumps). score.py opens hotels_probe only behind its
protocol gate, so a HOTELS dump may be tracked only where the protocol opens it
(test_hotels_probe_dumps_only_where_the_protocol_opens_them): in a final seed's dir (its run id is a dump_run_id of
results/postfit/{laya,berturk}-final.json; tests/test_results.py checks its protocol_sha256) or in the zero-shot dir
results/dumps/laya-zeroshot-7b928d82/ (the ungated reference row), never in a selection candidate's or a baseline's dir
and never under results/superseded/. The latency reports (times, digests and repo-relative paths only) get the sidecar
check too, and only results/latency/(laya|berturk)-(mps|cpu).json and results/latency/latency.md may be tracked there.
Their per-seed "state" is the checked {device, dtype, amp_enabled} dict, the one "state" value the message-text key
check allows (CHECKED_STATE_KEYS); a string or any other "state" is still a finding. The frozen protocol.json is
required to be tracked and gets the text check, with no key named state or text at all. The published verdict under
results/final/ gets the sidecar check too, and only verdict.json, reports/<id>-test.json,
confidence/<id>-(test.json|test.svg|hotels_probe.json) and the decision memo memo.md (at the top level of
results/final/) may be tracked there (FINAL_PATH_RE). The few-shot records under results/fewshot/ (the compute
decision record, the runs manifest and the few-shot results record) get the sidecar check too, only projection.json,
runs.json and fewshot.json may be tracked there (FEWSHOT_PATH_RE), and no string in them may be a bank_ or hotel_ uid
(uid_problems): the frozen support-set uid lists live in protocol.json only. A few-shot checkpoint's dumps sit under
results/dumps/<run_id>/ like every other dump (tests/test_results.py checks them against the runs manifest); none of
them is a HOTELS dump, so the HOTELS rule still allows HOTELS dumps only in the final seeds' dirs and the zero-shot dir.
"""
import io
import json
import os
import re
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESSAGE_TEXT_KEYS = ("state", "text")
# the checked fp32 state latency.py records per seed under "state": never message text, which is a string
CHECKED_STATE_KEYS = {"device", "dtype", "amp_enabled"}
UID_RE = re.compile(r"^bank_[0-9]+_[0-9]+$")       # split uids are bank_<fold>_<index>; never text-like
HOTEL_UID_RE = re.compile(r"^hotel_[0-9]+_[0-9]+$")  # hotels_probe uids are hotel_<fold>_<index>, only in hotels dumps
DUMP_KEYS = {"logits", "uids"}
DUMP_COLUMNS = 48                                   # one logit per INTENTS tag
DUMP_PATH_RE = re.compile(r"^results/dumps/[^/]+/(val|test|hotels_probe)\.(npz|json)$")
# a superseded dir (pilot.berturk.SUPERSEDED) holds its README, its records and its val dumps only
SUPERSEDED_PATH_RE = re.compile(r"^results/superseded/[^/]+/(README\.md|postfit/[^/]+\.json|dumps/[^/]+/val\.(npz|json))$")
# val confidence records only: the test and HOTELS confidence records of the final seeds live under results/final/
CONFIDENCE_PATH_RE = re.compile(r"^results/confidence/[^/]+-val\.(json|svg)$")
# latency: one report per (system, device) of the final seeds, and their markdown
LATENCY_PATH_RE = re.compile(r"^results/latency/((laya|berturk)-(mps|cpu)\.json|latency\.md)$")
# the verdict record, the per-seed test reports and the test/HOTELS confidence records (a test record with its
# reliability SVG; a HOTELS record has no diagram), and the decision memo memo.md
FINAL_PATH_RE = re.compile(r"^results/final/(verdict\.json|memo\.md|reports/[^/]+-test\.json|"
                           r"confidence/[^/]+-(test\.(json|svg)|hotels_probe\.json))$")
# the few-shot compute decision record, the few-shot runs manifest and the few-shot results record
FEWSHOT_PATH_RE = re.compile(r"^results/fewshot/(projection|runs|fewshot)\.json$")
# Coarse backstop only: the longest legitimate manifest value is a 64-char sha256, but message text is often far
# shorter ("Evet.", "cari hesap"), so length cannot detect it. The key-name check and, when the split files exist
# locally, the exact-text check below are the real guards.
MAX_MANIFEST_STRING = 200
REQUIRED_TRACKED = ("requirements.txt", "pyproject.toml", "Makefile", "pilot/pins.py", "data/splits.json",
                    "protocol.json")


def tracked(*pathspec: str) -> list[str]:
    """Tracked paths via git ls-files (never a directory walk, which would see gitignored data and .venv)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *pathspec], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    return [f for f in out.stdout.split("\0") if f]


def walk(obj: Any, path: str = "$") -> Iterator[tuple[str, Any, Any]]:
    """Yield (path, key, value) for every dict entry and (path, None, value) for every list item, recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, "%s.%s" % (path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield path, None, v
            yield from walk(v, "%s[%d]" % (path, i))


def home_dir() -> str | None:
    """The local home directory, or None when it is too short to search for."""
    home = os.path.expanduser("~").rstrip("/\\")
    return home if len(home) >= 2 else None


def local_split_texts() -> set[str]:
    """Every message text in the local data/splits/*.jsonl (empty set when `make data` has not run)."""
    split_dir = os.path.join(REPO_ROOT, "data", "splits")
    names = sorted(n for n in os.listdir(split_dir) if n.endswith(".jsonl")) if os.path.isdir(split_dir) else []
    texts = set()
    for name in names:
        with open(os.path.join(split_dir, name), encoding="utf-8") as f:
            texts |= {json.loads(l)["state"] for l in f.read().split("\n") if l.strip()}
    return texts


def dump_problems(path: str) -> list[str]:
    """Findings for one dump .npz: pickles, keys, dtypes, shapes and uids that could carry text. A file named
    hotels_probe.npz must hold hotel_<fold>_<index> uids (HOTEL_UID_RE), every other dump bank uids (UID_RE)."""
    uid_re = HOTEL_UID_RE if os.path.basename(path) == "hotels_probe.npz" else UID_RE
    with open(path, "rb") as f:
        raw = f.read()
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=False)
    except (ValueError, OSError) as e:
        return ["not a readable npz archive (%s)" % type(e).__name__]
    if not isinstance(z, np.lib.npyio.NpzFile):
        return ["not an npz archive"]
    problems, arrays = [], {}
    with z:
        keys = set(z.files)
        problems += ["missing array %r" % k for k in sorted(DUMP_KEYS - keys)]
        problems += ["extra array %r" % k for k in sorted(keys - DUMP_KEYS)]
        for k in sorted(keys & DUMP_KEYS):
            try:
                arrays[k] = z[k]
            except ValueError:          # numpy refuses an object (pickled) array when allow_pickle=False
                problems.append("pickled array %r" % k)
    logits, uids = arrays.get("logits"), arrays.get("uids")
    if logits is not None:
        if logits.dtype != np.float32:
            problems.append("logits dtype %s, expected float32" % logits.dtype)
        if logits.ndim != 2:
            problems.append("logits are %d-D, expected 2-D" % logits.ndim)
        elif logits.shape[1] != DUMP_COLUMNS:
            problems.append("logits have %d columns, expected %d" % (logits.shape[1], DUMP_COLUMNS))
        if logits.dtype.kind == "f" and not np.isfinite(logits).all():
            problems.append("logits hold a non-finite value")
    if uids is not None:
        if uids.dtype.kind != "U" or uids.ndim != 1:
            problems.append("uids are %s %d-D, expected a 1-D unicode array" % (uids.dtype, uids.ndim))
        else:
            if logits is not None and logits.ndim >= 1 and len(uids) != logits.shape[0]:
                problems.append("%d uids for %d logit rows" % (len(uids), logits.shape[0]))
            bad = [u for u in uids.tolist() if not uid_re.match(u)]
            if bad:                     # at most 3, each cut short: a bad uid may be message text
                problems.append("%d uids do not match %s, e.g. %s" % (
                    len(bad), uid_re.pattern, ", ".join(repr(u[:12] + "...") for u in bad[:3])))
    return problems


def text_problems(obj: Any, home: str | None, texts: set[str]) -> list[str]:
    """Findings for a parsed JSON value: message-text keys, the local home path, or exact split message text."""
    problems = []
    for p, k, v in walk(obj):
        where = p if k is None else "%s.%s" % (p, k)
        if k in MESSAGE_TEXT_KEYS and not (k == "state" and isinstance(v, dict) and set(v) == CHECKED_STATE_KEYS):
            problems.append("message-text key at %s" % where)
        if k is not None and k in texts:
            problems.append("split message text used as a key under %s" % p)
        if isinstance(v, str):
            if home and home in v:
                problems.append("local home path at %s" % where)
            if v in texts:
                problems.append("split message text at %s" % where)
    return problems


def uid_problems(obj: Any) -> list[str]:
    """Findings for a parsed JSON value: every key or string value that is a bank_ or hotel_ uid (UID_RE,
    HOTEL_UID_RE). A few-shot record names runs and counts, never messages (the uid lists live in protocol.json)."""
    problems = []
    for p, k, v in walk(obj):
        if k is not None and (UID_RE.match(k) or HOTEL_UID_RE.match(k)):
            problems.append("uid used as a key under %s" % p)
        if isinstance(v, str) and (UID_RE.match(v) or HOTEL_UID_RE.match(v)):
            problems.append("uid string at %s" % (p if k is None else "%s.%s" % (p, k)))
    return problems


def sidecar_problems(path: str, home: str | None, texts: set[str]) -> list[str]:
    """Findings for one dump sidecar or report JSON: text_problems plus subset runs (never committed)."""
    with open(path, encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except ValueError:
            return ["not valid JSON"]
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    problems = text_problems(obj, home, texts)
    if obj.get("n_subset") is not None:
        problems.append("n_subset is not null (a subset run is never committed)")
    if obj.get("subset") is True:
        problems.append("subset is true (a subset run is never committed)")
    return problems


def _existing(rels: Iterable[str]) -> list[str]:
    return [r for r in rels if os.path.isfile(os.path.join(REPO_ROOT, r))]


def test_only_manifest_tracked_under_data() -> None:
    assert tracked("data/") == ["data/splits.json"], \
        "only data/splits.json may be tracked under data/ (generated splits and audit files carry message text)"


def test_no_jsonl_tracked() -> None:
    hits = [f for f in tracked() if f.lower().endswith(".jsonl")]
    assert hits == [], "tracked .jsonl files (Laya JSONL holds message text): %s" % hits


def test_manifest_has_no_message_text() -> None:
    with open(os.path.join(REPO_ROOT, "data", "splits.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    bad_keys = ["%s.%s" % (p, k) for p, k, _ in walk(manifest) if k in MESSAGE_TEXT_KEYS]
    long_values = ["%s.%s (%d chars)" % (p, k, len(v)) for p, k, v in walk(manifest)
                   if isinstance(v, str) and len(v) > MAX_MANIFEST_STRING]
    assert bad_keys == [], "data/splits.json has message-text keys: %s" % bad_keys
    assert long_values == [], "data/splits.json has string values longer than %d chars: %s" % (
        MAX_MANIFEST_STRING, long_values)


def test_manifest_holds_no_split_message_text() -> None:
    texts = local_split_texts()
    if not texts:
        pytest.skip("no local data/splits/*.jsonl to compare against (run `make data`)")
    with open(os.path.join(REPO_ROOT, "data", "splits.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    hits = sorted("%s.%s" % (p, k) for p, k, v in walk(manifest) if isinstance(v, str) and v in texts)
    hits += sorted("%s (key)" % k for _, k, _ in walk(manifest) if k in texts)
    assert hits == [], "data/splits.json holds message text at: %s" % hits


def test_no_local_home_path_in_tracked_files() -> None:
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
def test_required_files_tracked(rel: str) -> None:
    assert tracked(rel) == [rel], "%s must be tracked (reproducibility depends on it)" % rel


def test_frozen_protocol_holds_no_text_or_paths() -> None:
    """protocol.json (frozen once) holds ids, digests, repo-relative paths, numbers and rule texts only: no
    message-text key (no key named state or text at all), no local home path, no split message text."""
    path = os.path.join(REPO_ROOT, "protocol.json")
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    problems = text_problems(obj, home_dir(), local_split_texts())
    problems += ["key %r at %s" % (k, p) for p, k, _ in walk(obj) if k in MESSAGE_TEXT_KEYS]
    assert problems == [], "protocol.json holds text or local paths: %s" % problems


BERTURK_RUN = "berturk_runs/berturk-selection_train-s0-0123abcd/"


@pytest.mark.parametrize("rel", ["ray_results/laya-final_train-s0-20261013-120000/params.json",
                                 "ckpt/model.safetensors", "ckpt/rl_agent_config.json", "ckpt_s1/model.safetensors",
                                 "other/model.safetensors",
                                 BERTURK_RUN + "exports/epoch-001/model.safetensors",
                                 BERTURK_RUN + "exports/epoch-001/berturk_train.json", BERTURK_RUN + "done.json",
                                 "ckpt-berturk-final-s0/model.safetensors", "ckpt-berturk-final-s0/berturk_train.json",
                                 "tensorboard/laya-selection_train-neg8-s0-0123abcd/"
                                 "events.out.tfevents.1.host.segment-000",
                                 "tensorboard/berturk-selection_train-s0-0123abcd/events.out.tfevents.1.host.pid1"])
def test_training_outputs_are_gitignored(rel: str) -> None:
    # documented commands write Ray storage and checkpoints into the repo root; `git add .` must not pick them up
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", "--", rel], cwd=REPO_ROOT)
    assert r.returncode == 0, "%s is not gitignored" % rel


def test_tracked_dumps_hold_only_uids_and_logits() -> None:
    found = {rel: dump_problems(os.path.join(REPO_ROOT, rel))
             for rel in _existing(f for f in tracked("results/dumps/", "results/superseded/") if f.endswith(".npz"))}
    bad = {rel: p for rel, p in found.items() if p}
    assert bad == {}, "tracked dumps hold more than uids and float32 logits: %s" % bad


def test_tracked_sidecars_and_reports_hold_no_text_or_paths() -> None:
    home, texts = home_dir(), local_split_texts()
    rels = [f for f in tracked("results/dumps/", "results/reports/", "results/compute/", "results/postfit/",
                               "results/superseded/", "results/confidence/", "results/latency/", "results/final/",
                               "results/fewshot/")
            if f.endswith(".json")]
    bad = {rel: p for rel in _existing(rels) for p in [sidecar_problems(os.path.join(REPO_ROOT, rel), home, texts)]
           if p}
    assert bad == {}, "tracked sidecars or reports hold text, local paths or subset runs: %s" % bad


def test_only_json_under_results_postfit() -> None:
    bad = [f for f in tracked("results/postfit/") if not f.endswith(".json")]
    assert bad == [], "only postfit records (.json) may be tracked under results/postfit/: %s" % bad


def test_only_confidence_files_under_results_confidence() -> None:
    bad = [f for f in tracked("results/confidence/") if not CONFIDENCE_PATH_RE.match(f)]
    assert bad == [], "only results/confidence/<run_id>-val.(json|svg) may be tracked: %s" % bad


@pytest.mark.parametrize("rel,ok", [
    ("results/confidence/laya-x-val.json", True),
    ("results/confidence/laya-x-val.svg", True),
    ("results/confidence/laya-x-test.json", False),
    ("results/confidence/laya-x-hotels_probe.json", False),
    ("results/confidence/notes.md", False),
    ("results/confidence/x/laya-x-val.json", False),
])
def test_confidence_path_pin_flags_planted_paths(rel: str, ok: bool) -> None:
    assert bool(CONFIDENCE_PATH_RE.match(rel)) is ok, rel


def test_only_latency_files_under_results_latency() -> None:
    bad = [f for f in tracked("results/latency/") if not LATENCY_PATH_RE.match(f)]
    assert bad == [], "only results/latency/(laya|berturk)-(mps|cpu).json and latency.md may be tracked: %s" % bad


@pytest.mark.parametrize("rel,ok", [
    ("results/latency/laya-mps.json", True),
    ("results/latency/berturk-cpu.json", True),
    ("results/latency/latency.md", True),
    ("results/latency/laya-cuda.json", False),
    ("results/latency/notes.txt", False),
    ("results/latency/sub/laya-mps.json", False),
])
def test_latency_path_pin_flags_planted_paths(rel: str, ok: bool) -> None:
    assert bool(LATENCY_PATH_RE.match(rel)) is ok, rel


def test_only_fewshot_files_under_results_fewshot() -> None:
    files = tracked("results/fewshot/")
    assert "results/fewshot/projection.json" in files, "results/fewshot/projection.json is not tracked"
    bad = [f for f in files if not FEWSHOT_PATH_RE.match(f)]
    assert bad == [], "only results/fewshot/(projection|runs|fewshot).json may be tracked: %s" % bad


@pytest.mark.parametrize("rel,ok", [
    ("results/fewshot/projection.json", True),
    ("results/fewshot/runs.json", True),
    ("results/fewshot/fewshot.json", True),
    ("results/fewshot/fewshot.md", False),
    ("results/fewshot/x.json", False),
    ("results/fewshot/sub/runs.json", False),
    ("results/fewshot/runs.json.bak", False),
])
def test_fewshot_path_pin_flags_planted_paths(rel: str, ok: bool) -> None:
    assert bool(FEWSHOT_PATH_RE.match(rel)) is ok, rel


def test_no_uid_strings_under_results_fewshot() -> None:
    # uid lists live in protocol.json only (the frozen support sets); a few-shot record names runs, never messages
    bad = {}
    for rel in _existing(f for f in tracked("results/fewshot/") if f.endswith(".json")):
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            found = uid_problems(json.load(f))
        if found:
            bad[rel] = found
    assert bad == {}, "tracked results/fewshot/ records hold uid strings: %s" % bad


@pytest.mark.parametrize("obj,n", [
    ({"schema": "fewshot-runs/1", "entries": [{"dump_run_id": "laya-0123456789ab-01234567-89abcdef"}]}, 0),
    ({"support": ["bank_3_17"]}, 1),
    ({"draws": [{"replay": ["bank_0_1", "bank_12_40"]}]}, 2),
    ({"x": {"hotel_4_2": 1}}, 1),
    ({"note": "hotel_0_9"}, 1),
    ({"note": "bank_card", "tag": "bank_transfer"}, 0),
])
def test_uid_scan_flags_planted_uids(obj: Any, n: int) -> None:
    assert len(uid_problems(obj)) == n, uid_problems(obj)


def test_only_final_files_under_results_final() -> None:
    files = tracked("results/final/")
    assert "results/final/verdict.json" in files, "results/final/verdict.json is not tracked"
    bad = [f for f in files if not FINAL_PATH_RE.match(f)]
    assert bad == [], ("only results/final/verdict.json, memo.md, reports/<id>-test.json and "
                       "confidence/<id>-(test.json|test.svg|hotels_probe.json) may be tracked: %s" % bad)


@pytest.mark.parametrize("rel,ok", [
    ("results/final/verdict.json", True),
    ("results/final/reports/x-test.json", True),
    ("results/final/confidence/x-test.svg", True),
    ("results/final/confidence/x-test.json", True),
    ("results/final/confidence/x-hotels_probe.json", True),
    ("results/final/confidence/x-hotels_probe.svg", False),
    ("results/final/reports/x-val.json", False),
    ("results/final/confidence/x-val.json", False),
    ("results/final/notes.md", False),
    ("results/final/memo.md", True),
    ("results/final/sub/memo.md", False),
    ("results/final/sub/verdict.json", False),
])
def test_final_path_pin_flags_planted_paths(rel: str, ok: bool) -> None:
    assert bool(FINAL_PATH_RE.match(rel)) is ok, rel


def test_only_dump_files_under_results_dumps() -> None:
    bad = [f for f in tracked("results/dumps/") if not DUMP_PATH_RE.match(f)]
    assert bad == [], "only results/dumps/<run_id>/(val|test|hotels_probe).(npz|json) may be tracked: %s" % bad


@pytest.mark.parametrize("rel,ok", [
    ("results/dumps/x/hotels_probe.npz", True),
    ("results/dumps/x/hotels_probe.json", True),
    ("results/dumps/x/val.npz", True),
    ("results/dumps/x/hotels.npz", False),
    ("results/dumps/x/train.npz", False),
    ("results/dumps/x/y/hotels_probe.npz", False),
])
def test_dump_path_pin_flags_planted_paths(rel: str, ok: bool) -> None:
    assert bool(DUMP_PATH_RE.match(rel)) is ok, rel


HOTELS_DUMP_FILES = ("hotels_probe.npz", "hotels_probe.json")


ZERO_SHOT = "laya-zeroshot-7b928d82"                 # the ungated HOTELS reference row


def final_dump_ids() -> set[str]:
    """The dump_run_id of every final seed in the committed final manifests (results/postfit/<kind>-final.json)."""
    out = set()
    for kind in ("laya", "berturk"):
        with open(os.path.join(REPO_ROOT, "results", "postfit", "%s-final.json" % kind), encoding="utf-8") as f:
            out |= {e["dump_run_id"] for e in json.load(f)["seeds"]}
    return out


def hotels_dump_allowed(rel: str, final_ids: set[str]) -> bool:
    """True when the tracked HOTELS dump file `rel` sits where the protocol opens it: results/dumps/<rid>/ with rid a
    final seed's dump run id or the zero-shot run id; never under results/superseded/ or another run dir."""
    m = re.match(r"^results/dumps/([^/]+)/hotels_probe\.(npz|json)$", rel)
    return bool(m) and (m.group(1) in final_ids or m.group(1) == ZERO_SHOT)


def test_hotels_probe_dumps_only_where_the_protocol_opens_them() -> None:
    # score.py opens hotels_probe only behind its protocol gate: a final seed is scored on HOTELS once, and zero-shot
    # Laya is the ungated reference row; a selection candidate, a baseline or a superseded run never is
    final_ids = final_dump_ids()
    assert len(final_ids) == 6, final_ids
    found = [f for f in tracked("results/dumps/", "results/superseded/") if os.path.basename(f) in HOTELS_DUMP_FILES]
    bad = [f for f in found if not hotels_dump_allowed(f, final_ids)]
    assert bad == [], "a HOTELS dump may be tracked only in a final seed's dir or %s: %s" % (ZERO_SHOT, bad)


@pytest.mark.parametrize("rel,ok", [
    ("results/dumps/{final}/hotels_probe.npz", True),
    ("results/dumps/{final}/hotels_probe.json", True),
    ("results/dumps/laya-zeroshot-7b928d82/hotels_probe.npz", True),
    ("results/dumps/{selection}/hotels_probe.npz", False),          # a selection candidate's dir
    ("results/dumps/prior/hotels_probe.json", False),               # a baseline
    ("results/superseded/berturk-10-epoch-grid/dumps/{final}/hotels_probe.npz", False),
    ("results/superseded/x/dumps/laya-zeroshot-7b928d82/hotels_probe.json", False),
    ("results/dumps/{final}/sub/hotels_probe.npz", False),
    ("results/dumps/laya-0123456789ab-01234567-89abcdef/hotels_probe.npz", False),  # a few-shot dir
    ("results/dumps/berturk-0123456789ab-01234567-89abcdef/hotels_probe.json", False),
])
def test_hotels_dump_rule_flags_planted_paths(rel: str, ok: bool) -> None:
    final_ids = final_dump_ids()
    with open(os.path.join(REPO_ROOT, "results", "postfit", "laya.json"), encoding="utf-8") as f:
        selection = json.load(f)["chosen"]["dump_run_id"]
    rel = rel.format(final=sorted(final_ids)[0], selection=selection)
    assert bool(hotels_dump_allowed(rel, final_ids)) is ok, rel


def _logits(n: int = 2, cols: int = DUMP_COLUMNS, dtype: type = np.float32) -> np.ndarray:
    return np.arange(n * cols, dtype=dtype).reshape(n, cols) / 10


def _uids(n: int = 2) -> np.ndarray:
    return np.asarray(["bank_16_%d" % i for i in range(n)], dtype=str)


def _bad_logits(value: float) -> np.ndarray:
    x = _logits()
    x[0, 0] = value
    return x


def _hotel_uids(n: int = 2) -> np.ndarray:
    return np.asarray(["hotel_16_%d" % i for i in range(n)], dtype=str)


PLANTED_DUMPS = {               # name -> (arrays to save, expected finding fragment or None for clean[, file name])
    "clean": (lambda: {"logits": _logits(), "uids": _uids()}, None),
    "hotel_uid_in_test": (lambda: {"logits": _logits(), "uids": _hotel_uids()}, "do not match ^bank_",
                          "test.npz"),
    "hotel_uid_in_val": (lambda: {"logits": _logits(), "uids": np.asarray(["bank_14_0", "hotel_14_0"])},
                         "1 uids do not match ^bank_", "val.npz"),
    "bank_uid_in_hotels_probe": (lambda: {"logits": _logits(), "uids": _uids()}, "do not match ^hotel_",
                                 "hotels_probe.npz"),
    "hotel_uids_in_hotels_probe": (lambda: {"logits": _logits(), "uids": _hotel_uids()}, None, "hotels_probe.npz"),
    "hotel_text_uid_in_hotels_probe": (lambda: {"logits": _logits(), "uids": np.asarray(["hotel_16_0", "Peki"])},
                                       "1 uids do not match ^hotel_", "hotels_probe.npz"),
    "text_uid": (lambda: {"logits": _logits(), "uids": np.asarray(["bank_16_0", "Kartım kayboldu"])},
                 "uids do not match"),
    "extra_text_array": (lambda: {"logits": _logits(), "uids": _uids(), "text": np.asarray(["a", "b"])},
                         "extra array 'text'"),
    "float64_logits": (lambda: {"logits": _logits(dtype=np.float64), "uids": _uids()}, "logits dtype float64"),
    "47_columns": (lambda: {"logits": _logits(cols=47), "uids": _uids()}, "47 columns"),
    "non_finite": (lambda: {"logits": _bad_logits(np.nan), "uids": _uids()}, "non-finite"),
    "uid_count": (lambda: {"logits": _logits(), "uids": _uids(3)}, "3 uids for 2 logit rows"),
    "pickled_uids": (lambda: {"logits": _logits(), "uids": np.asarray(["bank_16_0", "bank_16_1"], dtype=object)},
                     "pickled array 'uids'"),
}


@pytest.mark.parametrize("name", sorted(PLANTED_DUMPS))
def test_dump_checker_flags_planted_violations(tmp_path: Path, name: str) -> None:
    build, expected = PLANTED_DUMPS[name][:2]
    path = tmp_path / (PLANTED_DUMPS[name][2] if len(PLANTED_DUMPS[name]) > 2 else "test.npz")
    np.savez(path, **build())           # np.savez pickles object arrays, as a careless writer would
    problems = dump_problems(str(path))
    if expected is None:
        assert problems == []
    else:
        assert len(problems) == 1 and expected in problems[0], problems


PLANTED_TEXT = "Kartım kayboldu"
CLEAN_SIDECAR = {"run_id": "laya-0123456789ab", "dump_sha256": "a" * 64, "n_subset": None, "schema_version": 1,
                 "split": {"name": "test", "path": "data/splits/test.jsonl", "role": "test", "sha256": "b" * 64},
                 "system": {"kind": "laya", "weights": {"model_sha256": "c" * 64}}, "uids_sha256": "d" * 64}


def _planted(**change: Any) -> dict[str, Any]:
    side = json.loads(json.dumps(CLEAN_SIDECAR))
    for dotted, value in change.items():
        node, keys = side, dotted.split("__")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return side


PLANTED_SIDECARS = {            # name -> (sidecar builder, expected finding fragment or None for clean)
    "clean": (lambda home: _planted(), None),
    "state_key": (lambda home: _planted(state="x"), "message-text key at $.state"),
    "checked_state": (lambda home: _planted(system__state={"device": "mps", "dtype": "torch.float32",
                                                           "amp_enabled": False}), None),
    "state_dict_other_keys": (lambda home: _planted(system__state={"device": "mps", "note": "x"}),
                              "message-text key at $.system.state"),
    "nested_text_key": (lambda home: _planted(system__meta__text="x"), "message-text key at $.system.meta.text"),
    "home_path": (lambda home: _planted(system__checkpoint=home + "/ckpt"), "local home path at $.system.checkpoint"),
    "n_subset": (lambda home: _planted(n_subset=5), "n_subset is not null"),
    "split_text_value": (lambda home: _planted(system__note=PLANTED_TEXT), "split message text at $.system.note"),
    "split_text_key": (lambda home: _planted(system__extra={PLANTED_TEXT: 1}), "used as a key under $.system.extra"),
}


@pytest.mark.parametrize("name", sorted(PLANTED_SIDECARS))
def test_sidecar_checker_flags_planted_violations(tmp_path: Path, name: str) -> None:
    home = home_dir()
    if home is None:
        pytest.skip("home directory is too short to plant")
    build, expected = PLANTED_SIDECARS[name]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(build(home), ensure_ascii=False), encoding="utf-8")
    problems = sidecar_problems(str(path), home, {PLANTED_TEXT})
    if expected is None:
        assert problems == []
    else:
        assert len(problems) == 1 and expected in problems[0], problems


def test_report_checker_flags_a_subset_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"run_id": "laya-0123456789ab", "subset": True}), encoding="utf-8")
    assert sidecar_problems(str(path), home_dir(), set()) == ["subset is true (a subset run is never committed)"]
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert sidecar_problems(str(path), home_dir(), set()) == ["not a JSON object"]


def test_legacy_results_are_labelled() -> None:
    # the 100-message, no-no-intent reports must never be read as the full-test protocol
    assert tracked("results/legacy/") == ["results/legacy/README.md", "results/legacy/finetuned_smoke.json",
                                          "results/legacy/zeroshot_multilingual.json"]
    with open(os.path.join(REPO_ROOT, "results", "legacy", "README.md"), encoding="utf-8") as f:
        label = f.read()
    assert "legacy protocol" in label and "not comparable" in label
    assert tracked("results/zeroshot_multilingual.json", "results/finetuned_smoke.json") == [], \
        "legacy reports must live under results/legacy/, not at the top of results/"


def test_superseded_results_are_labelled() -> None:
    # a superseded BERTurk selection is kept for audit, never read as current or as a test result
    from pilot import berturk               # lazy: the rest of this file must not need pilot.berturk
    files = tracked("results/superseded/")
    bad = [f for f in files if not SUPERSEDED_PATH_RE.match(f)]
    assert bad == [], "only README.md, postfit/*.json and dumps/<run id>/val.(npz|json) may be tracked per " \
                      "superseded dir: %s" % bad
    tests = [f for f in files if os.path.basename(f) in ("test.npz", "test.json")]
    assert tests == [], "a superseded run never holds a test dump: %s" % tests
    dirs = {"/".join(f.split("/")[:3]) for f in files}
    assert dirs == {s["dir"] for s in berturk.SUPERSEDED}, \
        "tracked superseded dirs %s != pilot.berturk.SUPERSEDED dirs" % sorted(dirs)
    for d in sorted(dirs):
        readme = d + "/README.md"
        assert readme in files, "%s is not tracked" % readme
        with open(os.path.join(REPO_ROOT, readme), encoding="utf-8") as f:
            label = f.read()
        for word in ("superseded", "not comparable", "reference check"):
            assert word in label, "%s does not say %r" % (readme, word)


@pytest.mark.parametrize("rel,ok", [
    ("results/superseded/x/README.md", True),
    ("results/superseded/x/postfit/berturk.json", True),
    ("results/superseded/x/dumps/y/val.npz", True),
    ("results/superseded/x/dumps/y/val.json", True),
    ("results/superseded/x/dumps/y/test.npz", False),
    ("results/superseded/x/notes.txt", False),
    ("results/superseded/x/postfit/a.npz", False),
    ("results/superseded/README.md", False),
    ("results/superseded/x/dumps/y/z/val.npz", False),
])
def test_superseded_path_pin_flags_planted_paths(rel: str, ok: bool) -> None:
    assert bool(SUPERSEDED_PATH_RE.match(rel)) is ok, rel
