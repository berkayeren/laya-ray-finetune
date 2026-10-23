"""The committed Phase 3 results stay consistent with each other and with the README (D-03, D-08, T-03-29, T-03-30).

- the README results block (between the results-table markers) equals report.render_table of the tracked
  results/reports/*.json in sorted path order, so README numbers cannot be edited by hand or go stale
- every tracked run dir under results/dumps verifies its val dump with pilot.dumps.load_dump against the committed
  manifest (offline: load_dump reads no split file); a dir of a fine-tuned checkpoint (_finetuned: system.kind laya
  with no pinned revision, or any system.kind berturk) must hold no tracked test dump until Phase 8's protocol gate
  (ROADMAP Phase 5 criterion 4, Phase 6 criterion 4), and every other dir (majority, prior, zero-shot Laya) verifies
  its test dump too; the predicate is proven on planted sidecars
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

import pytest

import report
from pilot import dumps
from pilot.data import DEFAULT_MANIFEST
from pilot.weights import identity

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(REPO_ROOT, "results", "reports")
DUMPS = os.path.join(REPO_ROOT, "results", "dumps")
START, END = "<!-- results-table:start -->", "<!-- results-table:end -->"
SYSTEMS = ("majority", "prior", "laya-zeroshot-7b928d82")


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


def test_committed_dumps_verify_and_no_finetuned_test_dump():
    ids = run_dirs()
    assert set(SYSTEMS) <= set(ids), "missing committed dumps for %s" % sorted(set(SYSTEMS) - set(ids))
    for rid in ids:
        logits, uids, side, _ = dumps.load_dump(os.path.join(DUMPS, rid), "val", DEFAULT_MANIFEST)
        assert side["run_id"] == rid and side["n_subset"] is None and logits.shape == (len(uids), 48)
        if _finetuned(side):                                # until Phase 8's protocol gate (ROADMAP 5 SC4, 6 SC4)
            held = tracked("results/dumps/%s/test.npz" % rid, "results/dumps/%s/test.json" % rid)
            assert not held, ("fine-tuned (Laya or BERTurk) dump dir %s holds a test dump; no fine-tuned model is "
                              "scored on test before Phase 8" % rid)
            continue
        logits, uids, side, _ = dumps.load_dump(os.path.join(DUMPS, rid), "test", DEFAULT_MANIFEST)
        assert side["run_id"] == rid and side["n_subset"] is None and logits.shape == (len(uids), 48)


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
