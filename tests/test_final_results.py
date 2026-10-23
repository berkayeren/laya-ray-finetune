"""The once-only scoring run of the six final seeds (plan 08-11, REP-01, ROADMAP Phase 8 criterion 4).

- `make final-scoring` is the exact command list: per protocol final seed (pilot.protocol.require_frozen(), Laya then
  BERTurk, seeds in protocol order) one score.py line on the committed test split and one on hotels_probe, each
  naming the seed's checkpoint_name under $(FINAL_ROOT) with --device mps --workers 0 (the protocol's scoring block),
  then the ungated zero-shot Laya HOTELS reference (08-RESEARCH A15) on its val dump's scorer; no line holds --force,
  --n or --out-root (the gate refuses them anyway)
- each final seed was scored exactly once: its dump dir tracks exactly test and hotels_probe npz + json, every dump
  verifies (pilot.dumps.load_dump, offline), is full (n_subset None, the protocol's message count), carries
  protocol_sha256 == pilot.protocol.PROTOCOL_SHA256, was scored on mps, fp32, autocast off, in-process, with the
  seed, the postfit record sha256 and the chosen T (1e-12) of the protocol, from the seed's checkpoint (base name);
  the zero-shot HOTELS reference is tracked, ungated (no protocol_sha256), on its val dump's scorer (cpu, 2 workers)
- git order (REP-01): protocol.json has exactly one commit, and it is a strict ancestor of the commit that added each
  gated dump file
"""
import json
import os
import re
import subprocess

import pytest

from pilot import dumps, protocol

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATED_SPLITS = ("test", "hotels_probe")
GATED_FILES = ("test.npz", "test.json", "hotels_probe.npz", "hotels_probe.json")
ZERO_SHOT = "laya-zeroshot-7b928d82"
DUMPS = os.path.join(REPO_ROOT, "results", "dumps")
ZERO_SHOT_LINE = ("$(PYTHON) score.py data/splits/hotels_probe.jsonl --model laya --workers $(SCORE_WORKERS) "
                  "--cpus-per-worker $(SCORE_CPUS)")


def makefile():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        return f.read()


def final_scoring_recipe(text):
    """The recipe lines (tab stripped) of the final-scoring target."""
    m = re.search(r"^final-scoring:\n((?:\t.*\n)+)", text, re.M)
    assert m, "Makefile has no final-scoring target"
    return [line[1:] for line in m.group(1).splitlines()]


def expected_gated_lines(proto):
    out = []
    for kind in protocol.KINDS:
        for s in protocol.final_seeds(proto, kind):
            for split in GATED_SPLITS:
                out.append("$(PYTHON) score.py data/splits/%s.jsonl --model %s --checkpoint $(FINAL_ROOT)/%s "
                           "--device mps --workers 0" % (split, kind, s["checkpoint_name"]))
    return out


def test_makefile_final_scoring_matches_the_protocol():
    proto, _ = protocol.require_frozen()
    assert proto["scoring"]["device"] == "mps" and proto["scoring"]["workers"] == 0
    assert tuple(proto["scoring"]["splits"]) == GATED_SPLITS
    text = makefile()
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "final-scoring" in phony.group(1).split(), "final-scoring is not in .PHONY"
    assert re.search(r"^FINAL_ROOT \?= \.$", text, re.M), "Makefile has no `FINAL_ROOT ?= .` line"
    lines = final_scoring_recipe(text)
    assert len(lines) == 13, lines
    gated = expected_gated_lines(proto)
    assert len(gated) == 12 and lines[:12] == gated, lines[:12]
    names = {s["checkpoint_name"] for kind in protocol.KINDS for s in protocol.final_seeds(proto, kind)}
    assert len(names) == 6 and all(sum(n == l.split("$(FINAL_ROOT)/")[1].split()[0] for l in lines[:12]) == 2
                                   for n in names)
    assert lines[12] == ZERO_SHOT_LINE and "--checkpoint" not in lines[12]
    for line in lines:
        words = line.split()
        assert "--force" not in words and "--n" not in words and "--out-root" not in words, line


def git(*args):
    """stdout of a git command in the repo, captured before any parsing."""
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return out.stdout


def tracked(*pathspec):
    return [f for f in git("ls-files", "-z", "--", *pathspec).split("\0") if f]


def gated_paths(proto):
    return ["results/dumps/%s/%s" % (s["dump_run_id"], name) for kind in protocol.KINDS
            for s in protocol.final_seeds(proto, kind) for name in GATED_FILES]


def test_final_seeds_scored_exactly_once():
    proto, pin = protocol.require_frozen()
    assert pin == protocol.PROTOCOL_SHA256
    seeds = [(kind, s) for kind in protocol.KINDS for s in protocol.final_seeds(proto, kind)]
    assert len(seeds) == 6
    for kind, s in seeds:
        rid = s["dump_run_id"]
        names = sorted(f.split("/")[3] for f in tracked("results/dumps/%s/" % rid))
        assert names == sorted(GATED_FILES), (rid, names)
        system = proto["systems"][kind]
        for split in GATED_SPLITS:
            logits, uids, side, _ = dumps.load_dump(os.path.join(DUMPS, rid), split)
            n = proto["splits"][split]["n_messages"]
            assert logits.shape == (n, 48) and len(uids) == n, (rid, split, logits.shape)
            assert side["run_id"] == rid and side["n_subset"] is None, (rid, split)
            assert side["split"]["name"] == split and side["split"]["n_messages"] == n, (rid, split)
            assert side.get("protocol_sha256") == pin, (rid, split, side.get("protocol_sha256"))
            sc = side["scorer"]
            assert (sc["device"], sc["dtype"], sc["amp_enabled"], sc["workers"]) == \
                ("mps", "torch.float32", False, 0), (rid, split, sc)
            assert side["system"]["kind"] == kind, (rid, split)
            assert side["seed"] == s["seed"], (rid, split, side["seed"])
            assert side["training"]["postfit_sha256"] == system["postfit"]["sha256"], (rid, split)
            assert side["training"]["split"] == "final_train", (rid, split)
            assert abs(side["temperature"]["shipped"] - system["chosen"]["temperature"]) <= 1e-12, (rid, split)
            assert side["system"]["checkpoint"].rsplit("/", 1)[-1] == s["checkpoint_name"], (rid, split)
    zs = sorted(f.split("/")[3] for f in tracked("results/dumps/%s/" % ZERO_SHOT))
    assert {"hotels_probe.npz", "hotels_probe.json"} <= set(zs), zs
    logits, uids, side, _ = dumps.load_dump(os.path.join(DUMPS, ZERO_SHOT), "hotels_probe")
    n = proto["splits"]["hotels_probe"]["n_messages"]
    assert logits.shape == (n, 48) and side["n_subset"] is None and side["run_id"] == ZERO_SHOT
    assert "protocol_sha256" not in side, "the zero-shot HOTELS reference is ungated (08-RESEARCH A15)"
    with open(os.path.join(DUMPS, ZERO_SHOT, "val.json"), encoding="utf-8") as f:
        val = json.load(f)
    assert side["scorer"]["device"] == "cpu" and side["scorer"]["workers"] == 2
    for key in ("device", "dtype", "amp_enabled", "workers", "torch_threads"):
        assert side["scorer"][key] == val["scorer"][key], key
    assert side["system"]["checkpoint_identity"] == val["system"]["checkpoint_identity"]


def test_protocol_json_has_one_commit():
    out = git("log", "--format=%H", "--", "protocol.json")
    commits = out.split()
    assert len(commits) == 1, commits


def test_gated_dumps_descend_from_the_freeze():
    proto, _ = protocol.require_frozen()
    out = git("log", "--format=%H", "--", "protocol.json")
    freeze = out.split()
    assert len(freeze) == 1, freeze
    paths = gated_paths(proto) + ["results/dumps/%s/hotels_probe.%s" % (ZERO_SHOT, ext) for ext in ("npz", "json")]
    assert set(paths) <= set(tracked("results/dumps/")), sorted(set(paths) - set(tracked("results/dumps/")))
    for path in paths:
        out = git("log", "--diff-filter=A", "--format=%H", "--", path)
        added = out.split()
        assert len(added) == 1, (path, added)
        assert added[0] != freeze[0], "%s was added in the freeze commit" % path
        r = subprocess.run(["git", "merge-base", "--is-ancestor", freeze[0], added[0]], cwd=REPO_ROOT,
                           capture_output=True)
        assert r.returncode == 0, "%s was added in %s, which does not descend from the freeze %s" % (
            path, added[0], freeze[0])
