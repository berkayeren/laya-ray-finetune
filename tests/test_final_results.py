"""The once-only scoring run of the six final seeds (plan 08-11, REP-01, ROADMAP Phase 8 criterion 4).

- `make final-scoring` is the exact command list: per protocol final seed (pilot.protocol.require_frozen(), Laya then
  BERTurk, seeds in protocol order) one score.py line on the committed test split and one on hotels_probe, each
  naming the seed's checkpoint_name under $(FINAL_ROOT) with --device mps --workers 0 (the protocol's scoring block),
  then the ungated zero-shot Laya HOTELS reference (08-RESEARCH A15) on its val dump's scorer; no line holds --force,
  --n or --out-root (the gate refuses them anyway)
"""
import os
import re

import pytest

from pilot import protocol

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATED_SPLITS = ("test", "hotels_probe")
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
