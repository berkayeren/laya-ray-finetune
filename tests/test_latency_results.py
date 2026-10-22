"""The committed latency reports (plan 07-07, EVAL-06) are the real M4 runs of the final seeds and stay consistent with
the code: reported, not gated.

- every tracked results/latency/*.json is a latency/1 report labelled "reported, not gated" of one system (laya or
  berturk) on mps or cpu, bound to the committed final manifest results/postfit/<system>-final.json (its sha256,
  is_committed true) and timing exactly that manifest's seeds in order (seed, train run id, checkpoint name and
  identity, dump run id, served T)
- the protocol is the one latency.py fixes: the pilot.latency.WARMUP_POLICY warm-up (one untimed pass over all 208
  val messages per seed before its timed loop, warmup 208), pilot.latency.THREADS torch threads, fp32 with autocast
  off on the report's device for every seed, all 208 messages of the committed val split (its sha256 and uids_sha256
  equal to data/splits.json's val entry)
- every per-seed and pooled summary recomputes from the stored times with pilot.latency.summarize, so no number can be
  edited by hand
- each report passes the repository's sidecar text and path check (no message text, no uid list, no local path)
- exactly four reports are tracked, laya-mps, laya-cpu, berturk-mps and berturk-cpu, each named after its system and
  device, and results/latency/latency.md equals pilot.latency.render_markdown of them byte for byte
- README.md holds exactly one latency block, in the Phase 7 section, equal to pilot.latency.render_markdown of the
  reports (`latency.py --render results/latency`), and the Phase 7 prose before the generated blocks says
  "reported, not gated" too
"""
import json
import math
import os
import subprocess

import pytest

from pilot import latency
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT, sha256_file
from test_repo_hygiene import home_dir, local_split_texts, sidecar_problems

pytestmark = pytest.mark.unit

LATENCY = os.path.join(REPO_ROOT, "results", "latency")
DEVICES = ("mps", "cpu")
N_VAL = 208
SEED_KEYS = ("seed", "train_run_id", "checkpoint_name", "checkpoint_identity", "dump_run_id", "temperature")


def committed():
    """[(repo-relative path, parsed report)] of the tracked results/latency/*.json, in sorted order (git ls-files)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "results/latency/"], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    rels = sorted(f for f in out.stdout.split("\0") if f.endswith(".json"))
    reports = []
    for rel in rels:
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            reports.append((rel, json.load(f)))
    return reports


@pytest.fixture(scope="module")
def reports():
    found = committed()
    assert found, "no latency report is tracked under results/latency/"
    return found


def test_committed_reports_bind_the_final_manifests(reports):
    for rel, r in reports:
        assert r["schema"] == latency.SCHEMA and r["label"] == latency.LABEL, rel
        assert r["system"] in latency.SYSTEMS and r["device"] in DEVICES, rel
        fm = r["final_manifest"]
        path = "results/postfit/%s-final.json" % r["system"]
        assert fm["path"] == path, rel
        assert fm["sha256"] == sha256_file(os.path.join(REPO_ROOT, path)), \
            "%s was timed against another %s than the committed one" % (rel, path)
        assert fm["is_committed"] is True, rel
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
            manifest = json.load(f)
        assert [{k: s[k] for k in SEED_KEYS} for s in r["seeds"]] == \
            [{k: s[k] for k in SEED_KEYS} for s in manifest["seeds"]], \
            "%s does not time exactly the final seeds of %s in order" % (rel, path)


def test_protocol_settings(reports):
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        val = json.load(f)["splits"]["val"]
    for rel, r in reports:
        assert r.get("warmup_policy") == latency.WARMUP_POLICY, \
            "%s was timed before the %s warm-up; regenerate it with latency.py" % (rel, latency.WARMUP_POLICY)
        assert r["warmup"] == N_VAL and r["threads"] == latency.THREADS, rel
        assert r["dtype"] == "torch.float32" and r["amp_enabled"] is False, rel
        for s in r["seeds"]:
            assert s["state"] == {"device": r["device"], "dtype": "torch.float32", "amp_enabled": False}, \
                "%s seed %d" % (rel, s["seed"])
        sp = r["split"]
        assert sp["name"] == "val" and sp["n_messages"] == N_VAL, rel
        assert sp["sha256"] == val["sha256"] and sp["uids_sha256"] == val["uids_sha256"], \
            "%s was not timed on the committed val split" % rel


def test_summaries_recompute_from_times(reports):
    for rel, r in reports:
        pooled = []
        for s in r["seeds"]:
            times = s["times_s"]
            where = "%s seed %d on %s" % (r["system"], s["seed"], r["device"])
            assert len(times) == N_VAL, "%s seed %d" % (rel, s["seed"])
            assert all(isinstance(t, float) and math.isfinite(t) and t > 0 for t in times), \
                "%s seed %d" % (rel, s["seed"])
            assert s["summary"] == latency.summarize(times, where), "%s seed %d" % (rel, s["seed"])
            pooled += times
        assert r["pooled"] == latency.summarize(pooled, rel), rel
        assert r["pooled"]["n"] == N_VAL * len(r["seeds"]), rel


def test_reports_hold_no_text_or_paths(reports):
    home, texts = home_dir(), local_split_texts()
    bad = {rel: p for rel, _ in reports for p in [sidecar_problems(os.path.join(REPO_ROOT, rel), home, texts)] if p}
    assert bad == {}, "latency reports hold text, local paths or subset runs: %s" % bad


def test_exactly_four_reports(reports):
    names = [os.path.basename(rel) for rel, _ in reports]
    assert names == ["berturk-cpu.json", "berturk-mps.json", "laya-cpu.json", "laya-mps.json"], names
    for rel, r in reports:
        assert os.path.basename(rel) == "%s-%s.json" % (r["system"], r["device"]), rel


def test_markdown_matches_render():
    with open(os.path.join(LATENCY, latency.MD_NAME), encoding="utf-8") as f:
        md = f.read()
    assert md == latency.render_markdown(latency.load_reports(LATENCY)), \
        "results/latency/latency.md != render_markdown of the reports; rerun latency.py --render results/latency --out"
    assert latency.LABEL in md


README = os.path.join(REPO_ROOT, "README.md")
LATENCY_START, LATENCY_END = "<!-- latency:start -->", "<!-- latency:end -->"
PHASE7_HEADING = "## Confidence, out-of-domain and latency metrics (Phase 7)"


def test_readme_latency_block_matches_render():
    """The README block between the latency markers is pilot.latency.render_markdown of the committed reports; a hand
    edit or a stale block fails, and the section's prose states the label outside the generated block."""
    with open(README, encoding="utf-8") as f:
        text = f.read()
    assert text.count(LATENCY_START) == 1 and text.count(LATENCY_END) == 1, \
        "README.md must hold exactly one pair of the latency markers"
    i, j = text.index(LATENCY_START), text.index(LATENCY_END)
    assert i < j, "the latency end marker comes before the start marker"
    assert text[i + len(LATENCY_START):j].strip() == latency.render_markdown(latency.load_reports(LATENCY)).strip(), (
        "the README latency block differs from pilot.latency.render_markdown of the committed reports; "
        "regenerate it with `python latency.py --render results/latency`")
    assert text.count(PHASE7_HEADING) == 1 and text.index(PHASE7_HEADING) < i, \
        "the latency block is not in the Phase 7 section"
    prose = text[text.index(PHASE7_HEADING):i].split("<!-- confidence:start -->")[0]
    assert latency.LABEL in prose, "the Phase 7 prose must say %r next to the generated latency block" % latency.LABEL
