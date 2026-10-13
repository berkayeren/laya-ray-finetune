"""Real-data checks against the pinned Multi3NLU++ download (data marker; network on the first run only).

The pinned snapshot must verify against all 40 raw fold hashes, give the research split counts with and without
no-intent rows, produce the 1,009-message HOTELS probe with the 22 banking-only scoring tags and 3,080 unique raw uids,
and a fresh prepare into a tmp dir must reproduce the committed data/splits.json byte for byte.
"""
import json
import os

import pytest

import pilot.data
from pilot import pins

pytestmark = pytest.mark.data

SPLIT_NAMES = ("selection_train", "val", "final_train", "test")


@pytest.fixture(scope="module")
def raw_root():
    return pilot.data.fetch_raw()


@pytest.fixture(scope="module")
def default_run(raw_root, tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("real_default"))
    return out_dir, pilot.data.prepare(raw_root, out_dir, pins.RAW_SHA256, ood_tags=pins.OOD_SCORING_TAGS)


def test_pinned_snapshot_verifies_all_raw_hashes(raw_root):
    assert os.path.basename(os.path.realpath(raw_root)) == pins.DATASET_REVISION
    assert len(pins.RAW_SHA256) == 40
    pilot.data.verify_raw(raw_root, pins.RAW_SHA256)


def test_real_counts_with_no_intent_rows(default_run):
    _, m = default_run
    s = m["splits"]
    assert [s[k]["n_messages"] for k in SPLIT_NAMES] == [1450, 208, 1658, 413]
    assert [s[k]["n_no_intent"] for k in SPLIT_NAMES] == [66, 7, 73, 15]
    assert m["options"]["include_no_intent"] is True


def test_real_counts_without_no_intent_rows(raw_root, tmp_path):
    m = pilot.data.prepare(raw_root, str(tmp_path / "off"), pins.RAW_SHA256, include_no_intent=False,
                           ood_tags=pins.OOD_SCORING_TAGS)
    s = m["splits"]
    assert [s[k]["n_messages"] for k in SPLIT_NAMES] == [1384, 201, 1585, 398]
    assert (s["hotels_probe"]["n_messages"], s["hotels_probe"]["n_no_intent"]) == (1009, 274)   # flag never drops
    assert [s[k]["n_no_intent"] for k in SPLIT_NAMES] == [0, 0, 0, 0]
    assert m["options"]["include_no_intent"] is False


def test_fresh_prepare_reproduces_committed_manifest(default_run):
    out_dir, _ = default_run
    with open(os.path.join(out_dir, "splits.json"), "rb") as f:
        fresh = f.read()
    with open(pilot.data.DEFAULT_MANIFEST, "rb") as f:
        committed = f.read()
    assert fresh == committed


def test_real_test_split_is_disjoint_from_training(default_run):
    out_dir, _ = default_run

    def uids(name):
        with open(os.path.join(out_dir, "splits", "%s.jsonl" % name), encoding="utf-8") as f:
            return [json.loads(l)["uid"] for l in f]

    assert not set(uids("test")) & set(uids("final_train"))
    assert not set(uids("val")) & set(uids("selection_train"))
    assert uids("final_train") == uids("selection_train") + uids("val")


def test_real_hotels_probe(default_run):
    out_dir, m = default_run
    e = m["splits"]["hotels_probe"]
    assert (e["n_messages"], e["n_no_intent"], e["role"]) == (1009, 274, "ood_probe")
    assert e["folds"] == list(range(20)) and e["gold_valid_tags"] == "ood_scoring_tags"
    assert m["ood_scoring_tags"] == sorted(pins.OOD_SCORING_TAGS) and len(m["ood_scoring_tags"]) == 22
    assert m["uid_check"] == {"n_rows": 3080, "n_unique": 3080}
    with open(os.path.join(out_dir, e["path"]), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    assert len(rows) == 1009 and all(r["uid"].startswith("hotel_") for r in rows)
    assert all(len(r["expected"]) == 48 and not any(r["expected"].values()) for r in rows)
    banking = set()
    for name in SPLIT_NAMES:
        with open(os.path.join(out_dir, "splits", "%s.jsonl" % name), encoding="utf-8") as f:
            banking |= {json.loads(l)["uid"] for l in f}
    assert len(banking) == 2071 and not banking & {r["uid"] for r in rows}
