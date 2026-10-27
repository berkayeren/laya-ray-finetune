"""The committed exploratory search records under results/tune/, their rendered summary and the docs that name them.

- results/tune/laya.json and results/tune/berturk.json rebuild byte for byte from their own canonical parts
  (pilot.tune.read_record) and are the committed search: 20 trials (pilot.tune.TRIALS), search seed 0
  (pilot.tune.SEED), on mps, no pinned points, each system's space and ASHA scheduler, the committed manifest's
  selection_train, the pinned base, NOTES verbatim, and every driver start at a full git sha
- their partition digests and counts are pilot.tune.partition of the verified committed selection_train (the one test
  that reads the split file, marked `data`: run `make data` first)
- results/tune/tune.md is exactly `tune.py --render results/tune/laya.json results/tune/berturk.json`
  (pilot.tune.render_markdown of the checked records, laya then berturk), so no number in it can be edited by hand
- the search is no reproduction step: no STEP_NAMES entry, REBUILD command, README_BLOCKS command or Makefile line
  names tune.py or results/tune/, and README.md keeps exactly its README_BLOCKS generated blocks
- README.md names tune.py and pilot/tune.py in its Files table, results/tune/ in the results/ row and the search
  commands in its Run block, and its exploratory search section states the rules: exploratory, not a decision input,
  the verdict unchanged, the tuning partition, val, test and HOTELS never read, no trial weights, the pipeline's
  encoder LR fixed, resume by rerunning, the rendered summary; it holds no generated block
- CLAUDE.md and .claude/CLAUDE.md describe the search and say it does not change the verdict

Every test is offline but test_committed_partition_digests_match_the_split, which needs the committed split files.
"""
import json
import os
import re
import subprocess
import sys
from typing import Any

import pytest

from pilot import pins
from pilot import reproduce as rp
from pilot import tune as pt
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT, parse_jsonl, verify_split_file

pytestmark = pytest.mark.unit

TUNE = os.path.join(REPO_ROOT, "results", "tune")
SYSTEMS = ("laya", "berturk")
BASES = {"laya": ("multilingual", pins.LAYA), "berturk": ("berturk", pins.BERTURK)}
TUNE_MD = os.path.join(TUNE, "tune.md")
README = os.path.join(REPO_ROOT, "README.md")
SEARCH_HEADING = "\n## Exploratory hyperparameter search (Ray Tune)\n"
README_RUN = ("python tune.py --system berturk --device mps", "python tune.py --system laya --device mps",
              "python tune.py --status --system laya", "python tune.py --check results/tune/laya.json",
              "python tune.py --render results/tune/laya.json results/tune/berturk.json --out results/tune/tune.md")
README_RULES = ("exploratory", "not a decision input", "does not change the verdict", "folds 0-11", "folds 12-13",
                "val, test", "HOTELS", "no trial weights", "fixed, not tuned", "rerun the identical command",
                "results/tune/tune.md")
CLAUDE_NAMES = ("tune.py", "pilot/tune.py", "results/tune/", "noul_margins", "tune_score", "train_config",
                "bert_logits", "does not change the verdict")
DOT_CLAUDE_NAMES = ("tune.py", "pilot/tune.py", "results/tune/", "does not change the verdict")


def record(system: str) -> dict[str, Any]:
    return pt.read_record(os.path.join(TUNE, "%s.json" % system))[0]


def text_of(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def committed_entry() -> dict[str, Any]:
    """The committed manifest's selection_train entry."""
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        return json.load(f)["splits"][pt.SOURCE]


def readme_part(text: str, heading: str, end: str) -> str:
    """The README text from `heading` to the next `end` after it (exactly one `heading`)."""
    assert text.count(heading) == 1, "README.md must hold one %r" % heading.strip()
    i = text.index(heading)
    return text[i:text.index(end, i + len(heading))]


# ---- the committed records ----

@pytest.mark.parametrize("system", SYSTEMS)
def test_committed_records_rebuild_byte_for_byte(system: str) -> None:
    r, sha = pt.read_record(os.path.join(TUNE, "%s.json" % system))
    assert r["system"] == system and re.fullmatch(r"[0-9a-f]{64}", sha)


@pytest.mark.parametrize("system", SYSTEMS)
def test_committed_records_are_the_committed_search(system: str) -> None:
    r, entry = record(system), committed_entry()
    s, d = r["search"], r["data"]
    assert r["system"] == system
    assert s["n_trials"] == pt.TRIALS == 20 and len(r["trials"]) == 20
    assert s["seed"] == pt.SEED == 0
    assert r["device"] == "mps"
    assert s["points_to_evaluate"] is None
    assert s["space"] == pt.SPACES[system]
    assert s["scheduler"] == dict(pt.ASHA, max_t=pt.max_t(system))
    assert d["split"] == "selection_train" and d["sha256"] == entry["sha256"]
    assert d["manifest_is_committed"] is True
    name, pin = BASES[system]
    assert r["base"]["name"] == name and r["base"]["model_sha256"] == pin["sha256"]["model.safetensors"]
    assert r["notes"] == list(pt.NOTES)
    assert r["segments"] and all(re.fullmatch(r"[0-9a-f]{40}", g["git_sha"] or "") for g in r["segments"])
    assert record("laya")["data"] == record("berturk")["data"]


@pytest.mark.data
def test_committed_partition_digests_match_the_split() -> None:
    entry = committed_entry()
    path = os.path.join(os.path.dirname(DEFAULT_MANIFEST), entry["path"])
    with open(path, "rb") as f:
        raw = f.read()
    verify_split_file(path, roles=("train",), data=raw)
    train, dev, block = pt.partition(parse_jsonl(raw))
    want = {"train_uids_sha256": block["train_uids_sha256"], "dev_uids_sha256": block["dev_uids_sha256"],
            "n_train": len(train), "n_dev": len(dev), "dev_rows_with_train_text": pt.shared_texts(train, dev)}
    for system in SYSTEMS:
        d = record(system)["data"]
        assert {k: d[k] for k in want} == want, system


# ---- the rendered summary ----

def test_tune_md_is_the_render() -> None:
    with open(TUNE_MD, "rb") as f:
        raw = f.read()
    assert raw == pt.render_markdown([record("laya"), record("berturk")]).encode("utf-8")
    out = subprocess.run([sys.executable, "tune.py", "--render", "results/tune/laya.json", "results/tune/berturk.json"],
                         cwd=REPO_ROOT, check=True, capture_output=True)
    assert out.stdout == raw
    assert "not a decision input" in raw.decode("utf-8")


def test_the_search_is_not_a_reproduce_step() -> None:
    def names_tune(words: Any) -> bool:
        return any("tune.py" in w or "results/tune" in w or w == "tune" for w in words)
    assert not names_tune(rp.STEP_NAMES)
    assert not any(names_tune(cmd) for cmd in rp.REBUILD)
    assert not any(names_tune(cmd) for _, cmd in rp.README_BLOCKS)
    assert len(rp.README_BLOCKS) == 11
    makefile = text_of(os.path.join(REPO_ROOT, "Makefile"))
    assert "tune.py" not in makefile and "results/tune" not in makefile
    readme = text_of(README)
    assert len(re.findall(r"<!-- [a-z-]+:start -->", readme)) == len(rp.README_BLOCKS)


# ---- the docs ----

def test_readme_names_the_search_files_and_commands() -> None:
    text = text_of(README)
    rows = {}
    for line in readme_part(text, "\n## Files\n", "\n## Run\n").splitlines():
        m = re.match(r"^\| `([^`]+)` \| (.*) \|$", line)
        if m:
            rows[m.group(1)] = m.group(2)
    for path in ("tune.py", "pilot/tune.py"):
        assert path in rows, "the README Files table has no row for %s" % path
    assert "`tune/`" in rows["results/"], "the README results/ row does not name tune/"
    run = readme_part(text, "\n## Run\n", "\n## Scoring path\n")
    missing = [c for c in README_RUN if c not in run]
    assert not missing, "the README Run block does not hold %r" % missing
    assert "--status" in run and "--check" in run and "--render" in run


def test_readme_states_the_search_rules() -> None:
    text = text_of(README)
    assert text.count(SEARCH_HEADING) == 1, "README.md must hold one exploratory search section"
    i = text.index(SEARCH_HEADING)
    j = text.index("\n## ", i + len(SEARCH_HEADING))
    section = text[i:j]
    missing = [w for w in README_RULES if w not in section]
    assert not missing, "the README search section does not state %r" % missing
    assert "<!-- " not in section, "the README search section holds no generated block"
    assert text.index(SEARCH_HEADING) < text.index("\n## Demo UI\n")


def test_claude_docs_describe_the_search() -> None:
    text = text_of(os.path.join(REPO_ROOT, "CLAUDE.md"))
    missing = [w for w in CLAUDE_NAMES if w not in text]
    assert not missing, "CLAUDE.md does not name %r" % missing
    text = text_of(os.path.join(REPO_ROOT, ".claude", "CLAUDE.md"))
    missing = [w for w in DOT_CLAUDE_NAMES if w not in text]
    assert not missing, ".claude/CLAUDE.md does not name %r" % missing
