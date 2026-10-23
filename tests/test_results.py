"""The committed Phase 3 results stay consistent with each other and with the README (D-03, D-08, T-03-29, T-03-30).

- the README results block (between the results-table markers) equals report.render_table of the tracked
  results/reports/*.json in sorted path order, so README numbers cannot be edited by hand or go stale
- every tracked run dir under results/dumps follows the Phase 8 protocol (dir_problems; every present dump verified
  with pilot.dumps.load_dump against the committed manifest, offline: load_dump reads no split file), true before and
  after the 08-11 scoring run (08-RESEARCH Pitfall 1):
  - a final seed's dir (its run id is a dump_run_id in results/postfit/{laya,berturk}-final.json, FINAL_DIRS) holds
    no val dump and only test and hotels_probe dumps, each with its sidecar, full (no subset), of the manifest's system
    kind, and carrying protocol_sha256 == pilot.protocol.PROTOCOL_SHA256, which must be set once any such dump exists
    (final_dir_problems; nothing is tracked there before the scoring run);
  - every other fine-tuned dir (_finetuned: system.kind laya with no pinned revision, or any system.kind berturk:
    the selection candidates) holds its val dump only, never test or HOTELS (ROADMAP Phase 5 criterion 4, Phase 6
    criterion 4);
  - majority and prior hold val and test only;
  - the zero-shot dir (ZERO_SHOT) holds val and test and may add one ungated hotels_probe dump, the reference row
    (08-RESEARCH A15: no protocol_sha256);
  - no dump outside a final seed's dir carries protocol_sha256.
  Both predicates are proven on planted cases: final_dir_problems on planted sidecars, and dir_problems on a planted
  post-scoring tree of synthetic dumps (a final seed's dir with the four gated files, a selection candidate's dir with
  a HOTELS dump, the zero-shot dir with an ungated one)
- the zero-shot run id in the Makefile (LAYA_ZS) is the one score.py derives from the pinned Laya identity
- the three single-system reports (majority, prior, zero-shot Laya) are full-test, not subset, with both temperature
  variants and a disclosed θ source
- data marker: every committed report and comparison is recomputed from the committed dumps and equals the committed
  JSON (needs `make data`, because the gold labels come from the verified splits)
"""
import glob
import json
import os
import re
import subprocess

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import prepared
from pilot import dumps, protocol
from pilot.data import DEFAULT_MANIFEST
from pilot.weights import identity

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(REPO_ROOT, "results", "reports")
DUMPS = os.path.join(REPO_ROOT, "results", "dumps")
START, END = "<!-- results-table:start -->", "<!-- results-table:end -->"
SYSTEMS = ("majority", "prior", "laya-zeroshot-7b928d82")
ZERO_SHOT = "laya-zeroshot-7b928d82"
GATED_FILES = ("test.npz", "test.json", "hotels_probe.npz", "hotels_probe.json")


def _final_dirs():
    """{dump_run_id: (kind, seed)} of every final seed in the committed final manifests."""
    out = {}
    for kind in ("laya", "berturk"):
        with open(os.path.join(REPO_ROOT, "results", "postfit", "%s-final.json" % kind), encoding="utf-8") as f:
            for e in json.load(f)["seeds"]:
                out[e["dump_run_id"]] = (kind, e["seed"])
    return out


FINAL_DIRS = _final_dirs()


def tracked(*pathspec):
    """Tracked paths via git ls-files (never a directory walk, which would see untracked scratch reports)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *pathspec], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    return [f for f in out.stdout.split("\0") if f]


def committed_reports():
    """[(repo-relative path, parsed JSON)] of the tracked reports, in sorted path order (the --table order)."""
    rels = sorted(f for f in tracked("results/reports/") if f.endswith(".json"))
    out = []
    for rel in rels:
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            out.append((rel, json.load(f)))
    return out


def run_dirs():
    return sorted({f.split("/")[2] for f in tracked("results/dumps/")})


def test_readme_table_matches_committed_reports():
    with open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8") as f:
        text = f.read()
    assert text.count(START) == 1 and text.count(END) == 1, "README needs exactly one results-table block"
    block = text.split(START, 1)[1].split(END, 1)[0]
    reports = [r for _, r in committed_reports()]
    assert reports, "no tracked reports under results/reports/"
    assert block.strip() == report.render_table(reports).strip(), \
        "README results table differs from `report.py --table results/reports`; paste the generated table"


def _finetuned(side):
    """A dump of a local (fine-tuned) checkpoint: Laya without the pinned revision, or any BERTurk checkpoint."""
    system = side.get("system") or {}
    ident = system.get("checkpoint_identity") or {}
    return (system.get("kind") == "laya" and not ident.get("revision")) or system.get("kind") == "berturk"


def test_finetuned_rule_covers_berturk():
    # planted sidecars: the baselines and the pinned zero-shot Laya keep their test dumps; every fine-tuned one may not
    def side(kind, ident):
        return {"system": {"kind": kind, "checkpoint_identity": ident}}
    assert _finetuned(side("majority", None)) is False
    assert _finetuned(side("prior", None)) is False
    assert _finetuned(side("laya", {"revision": "0" * 40, "model_sha256": "a" * 64})) is False
    assert _finetuned(side("laya", {"local": True, "model_sha256": "a" * 64})) is True
    assert _finetuned(side("berturk", {"local": True, "model_sha256": "b" * 64})) is True
    assert _finetuned({}) is False


def final_dir_problems(rid, names, sidecars, pin):
    """Problems of a final seed's dump dir `rid` holding the tracked file names `names`, with `sidecars` {split:
    parsed sidecar} of its present test and hotels_probe dumps, under the protocol pin `pin` (None while unfrozen):
    only the four GATED_FILES, each dump with its sidecar, never a val dump (a final seed's val is in-sample), and every
    sidecar full and bound to the set pin. [] for an empty dir (before the scoring run)."""
    names = set(names)
    out = ["%s/%s: a final seed's dump dir holds only its gated test and hotels_probe dumps (never val: its val is "
           "in-sample, and it is scored only through the protocol gate)" % (rid, n)
           for n in sorted(names - set(GATED_FILES))]
    if names and pin is None:
        out.append("%s holds dumps while pilot.protocol.PROTOCOL_SHA256 is unset: a final seed is scored only behind "
                   "the frozen protocol" % rid)
    for split in ("test", "hotels_probe"):
        pair = {split + ".npz", split + ".json"} & names
        if not pair:
            continue
        if len(pair) == 1:
            out.append("%s/%s: an npz without its sidecar or a sidecar without its npz" % (rid, split))
            continue
        side = sidecars.get(split) or {}
        got = side.get("protocol_sha256")
        if got is None:
            out.append("%s/%s.json has no protocol_sha256: an ungated fine-tuned %s dump" % (rid, split, split))
        elif pin is not None and got != pin:
            out.append("%s/%s.json has protocol_sha256 %s, the pinned protocol is %s" % (rid, split, got, pin))
        if side.get("n_subset") is not None:
            out.append("%s/%s.json is a subset dump (n_subset %r)" % (rid, split, side.get("n_subset")))
    return out


def dir_problems(root, rid, names, manifest=DEFAULT_MANIFEST, pin=None, final_dirs=None):
    """Problems of the dump dir root/rid holding the file names `names` under the Phase 8 results rule (module
    docstring); every present dump is verified with pilot.dumps.load_dump against `manifest`."""
    final_dirs = FINAL_DIRS if final_dirs is None else final_dirs
    names, d, out = set(names), os.path.join(root, rid), []

    def load(split):
        try:
            logits, uids, side, _ = dumps.load_dump(d, split, manifest)
        except SystemExit as e:
            out.append("%s/%s: %s" % (rid, split, e))
            return None
        if side["run_id"] != rid or side["n_subset"] is not None or logits.shape != (len(uids), 48):
            out.append("%s/%s: run id %r, n_subset %r, logits %s" % (rid, split, side["run_id"], side["n_subset"],
                                                                      logits.shape))
        return side

    if rid in final_dirs:
        sidecars = {}
        for split in ("test", "hotels_probe"):
            if split + ".json" in names:
                with open(os.path.join(d, split + ".json"), encoding="utf-8") as f:
                    sidecars[split] = json.load(f)
        out += final_dir_problems(rid, names, sidecars, pin)
        for split in sidecars:
            side = load(split)
            kind = ((side or {}).get("system") or {}).get("kind")
            if side is not None and kind != final_dirs[rid][0]:
                out.append("%s/%s: system kind %r, the final manifest's is %r" % (rid, split, kind, final_dirs[rid][0]))
        return out
    side = load("val")
    if side is None:
        return out
    pairs = {s: {s + ".npz", s + ".json"} for s in ("val", "test", "hotels_probe")}
    if _finetuned(side):
        need, allowed, what = pairs["val"], pairs["val"], "a fine-tuned dir outside the final seeds holds val only"
    elif rid == ZERO_SHOT:
        need = pairs["val"] | pairs["test"]
        allowed, what = need | pairs["hotels_probe"], "the zero-shot dir holds val, test and an ungated hotels_probe"
    else:
        need = allowed = pairs["val"] | pairs["test"]
        what = "a baseline dir (majority, prior) holds val and test only"
    out += ["%s: %s is missing (%s)" % (rid, n, what) for n in sorted(need - names)]
    out += ["%s/%s: %s" % (rid, n, what) for n in sorted(names - allowed)]
    loaded = [side]
    for split in ("test", "hotels_probe"):
        if pairs[split] <= names & allowed:
            loaded.append(load(split))
    for s in loaded:
        if s is not None and "protocol_sha256" in s:
            out.append("%s/%s carries protocol_sha256: only a final seed's dump is gated" % (rid, s["split"]["name"]))
    return out


def test_committed_dumps_follow_the_protocol():
    ids = run_dirs()
    assert set(SYSTEMS) <= set(ids), "missing committed dumps for %s" % sorted(set(SYSTEMS) - set(ids))
    assert len(FINAL_DIRS) == 6 and {k for k, _ in FINAL_DIRS.values()} == {"laya", "berturk"}
    problems = []
    for rid in ids:
        names = [f.split("/")[3] for f in tracked("results/dumps/%s/" % rid)]
        problems += dir_problems(DUMPS, rid, names, DEFAULT_MANIFEST, protocol.PROTOCOL_SHA256)
    assert problems == [], "\n".join(problems)


def test_makefile_run_id_matches():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        m = re.search(r"^LAYA_ZS := (\S+)$", f.read(), re.M)
    assert m, "Makefile has no LAYA_ZS := line"
    assert m.group(1) == dumps.run_id("laya", identity("multilingual"))


def test_reports_cover_three_systems():
    by_id = {r["run_id"]: r for _, r in committed_reports() if "variants" in r}
    assert sorted(by_id) == sorted(SYSTEMS)
    for rid, r in by_id.items():
        assert r["split"]["name"] == "test" and r["subset"] is False, rid
        assert set(r["variants"]) == {"shipped", "raw"}, rid
        assert r["theta_source"] is not None and r["theta_source"]["split"] == "val", rid
        row = r["variants"]["shipped"]["rows"]["0.5"]
        assert (row["n"], row["n_no_intent"]) == (413, 15), rid
    comps = {os.path.basename(p) for p, r in committed_reports() if r.get("kind") == "paired_bootstrap"}
    assert comps == {"compare-laya-zeroshot-7b928d82-vs-prior.json", "compare-laya-zeroshot-7b928d82-vs-self.json"}


@pytest.mark.data
def test_reports_reproduce_from_dumps():
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")
    reports = committed_reports()
    assert reports
    for rel, committed in reports:
        if committed.get("kind") == "paired_bootstrap":
            got = report.compare([os.path.join(DUMPS, i) for i in committed["a"]["run_ids"]],
                                 [os.path.join(DUMPS, i) for i in committed["b"]["run_ids"]],
                                 committed["split"], DEFAULT_MANIFEST, committed["B"], committed["seed"])
        else:
            got = report.run_report(os.path.join(DUMPS, committed["run_id"]), committed["split"]["name"],
                                    DEFAULT_MANIFEST)
        assert json.loads(json.dumps(got, ensure_ascii=False)) == committed, "%s does not reproduce" % rel


def test_final_dir_rule_flags_planted_cases():
    pin, rid = "a" * 64, "laya-final-dir"
    gated = ("test.npz", "test.json", "hotels_probe.npz", "hotels_probe.json")

    def side(**kw):
        return dict({"run_id": rid, "n_subset": None, "protocol_sha256": pin}, **kw)
    both = {"test": side(), "hotels_probe": side()}
    assert final_dir_problems(rid, gated, both, pin) == []
    assert final_dir_problems(rid, (), {}, None) == []                  # before the scoring run: nothing tracked
    assert final_dir_problems(rid, gated[:2], {"test": side()}, pin) == []
    flagged = {
        "val dump": (gated + ("val.npz", "val.json"), both, pin, "val"),
        "other file": (gated + ("notes.txt",), both, pin, "notes.txt"),
        "missing pin key": (gated, {"test": side(protocol_sha256=None), "hotels_probe": side()}, pin,
                            "protocol_sha256"),
        "other pin": (gated, {"test": side(), "hotels_probe": side(protocol_sha256="b" * 64)}, pin, "b" * 64),
        "pin unset": (gated, both, None, "PROTOCOL_SHA256"),
        "subset": (gated, {"test": side(n_subset=20), "hotels_probe": side()}, pin, "subset"),
        "npz without sidecar": (("test.npz",), {}, pin, "sidecar"),
    }
    for name, (names, sides, p, word) in flagged.items():
        got = final_dir_problems(rid, names, sides, p)
        assert got and any(word in g for g in got), (name, got)


def _plant(snap, root, rid, split, kind, pin=None):
    """A synthetic full dump of `split` in root/rid (score.sidecar_base of the synthetic manifest), of a pinned Laya
    (kind "zeroshot"), a local Laya or BERTurk checkpoint, or a baseline; pin stamps protocol_sha256."""
    out_dir, manifest = snap
    with open(manifest, encoding="utf-8") as f:
        entry = json.load(f)["splits"][split]
    with open(os.path.join(out_dir, "splits", "%s.jsonl" % split), "rb") as f:
        uids = [r["uid"] for r in pilot.data.parse_jsonl(f.read())]
    ident = {"zeroshot": {"revision": "0" * 40}, "laya": {"local": True}, "berturk": {"local": True, "kind": "berturk"}}
    system = {"kind": "laya" if kind == "zeroshot" else kind, "checkpoint_identity": ident.get(kind)}
    side = dict(score.sidecar_base(entry, split, manifest, None), run_id=rid, system=system)
    if pin is not None:
        side["protocol_sha256"] = pin
    logits = np.random.default_rng(len(rid)).normal(size=(len(uids), 48)).astype(np.float32)
    dumps.write_dump(os.path.join(str(root), rid), split, logits, uids, side)


def test_dir_rule_on_a_planted_post_scoring_tree(tmp_path, synthetic_raw, capsys):
    snap = prepared(tmp_path, synthetic_raw)
    capsys.readouterr()
    pin, root = "c" * 64, tmp_path / "dumps"
    final = {"laya-final-s0": ("laya", 0), "berturk-final-s0": ("berturk", 0)}

    def problems(rid, p=pin):
        names = sorted(os.listdir(os.path.join(str(root), rid)))
        return dir_problems(str(root), rid, names, snap[1], p, final)

    for rid, kind in (("laya-final-s0", "laya"), ("berturk-final-s0", "berturk")):
        for split in ("test", "hotels_probe"):
            _plant(snap, root, rid, split, kind, pin)
        assert problems(rid) == [], rid                             # the four gated files of a scored final seed
        assert any("PROTOCOL_SHA256" in x for x in problems(rid, None)), rid    # a dump while the pin is unset
        assert any(pin in x for x in problems(rid, "d" * 64)), rid             # bound to another protocol
    _plant(snap, root, "laya-final-s0", "val", "laya")
    assert any("val" in x for x in problems("laya-final-s0"))       # a final seed's val is in-sample: never a dump
    _plant(snap, root, "laya-sel", "val", "laya")
    assert problems("laya-sel") == []                               # a selection candidate: val only
    _plant(snap, root, "laya-sel", "hotels_probe", "laya")
    assert any("hotels_probe" in x for x in problems("laya-sel"))   # never HOTELS outside the final seeds
    _plant(snap, root, "berturk-sel", "val", "berturk")
    _plant(snap, root, "berturk-sel", "test", "berturk", pin)
    assert any("test" in x for x in problems("berturk-sel"))        # never test either, gated or not
    for split in ("val", "test", "hotels_probe"):
        _plant(snap, root, ZERO_SHOT, split, "zeroshot")
    assert problems(ZERO_SHOT) == []                                # the ungated HOTELS reference row (A15)
    _plant(snap, root, ZERO_SHOT, "hotels_probe", "zeroshot", pin)
    assert any("protocol_sha256" in x for x in problems(ZERO_SHOT))
    for split in ("val", "test"):
        _plant(snap, root, "prior", split, "prior")
    assert problems("prior") == []
    _plant(snap, root, "prior", "hotels_probe", "prior")
    assert any("hotels_probe" in x for x in problems("prior"))      # the baselines never reach HOTELS
