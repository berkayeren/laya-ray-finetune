"""Real-data checks against the pinned Multi3NLU++ download (data marker; network on the first run only).

The pinned snapshot must verify against all 40 raw fold hashes, give the research split counts with and without
no-intent rows, produce the 1,009-message HOTELS probe with the 22 banking-only scoring tags and 3,080 unique raw uids,
report the research duplicate audit numbers and flag lists (DATA-04), and a fresh prepare into a tmp dir must
reproduce the committed data/splits.json byte for byte.
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


NORMALIZED_TEST_FLAGS = ["bank_16_85", "bank_16_88", "bank_17_80", "bank_17_85", "bank_18_85", "bank_19_25",
                         "bank_19_68", "bank_19_79", "bank_19_83", "bank_19_94", "bank_19_97"]
NORMALIZED_VAL_FLAGS = ["bank_14_16", "bank_14_75", "bank_14_81", "bank_14_85"]


def test_real_audit_numbers_and_flags(default_run):
    out_dir, m = default_run
    manifest_path = os.path.join(out_dir, "splits.json")
    for e in m["audit"].values():
        pilot.data.verify_audit_file(os.path.join(out_dir, e["path"]), manifest_path=manifest_path)

    def load(rel):
        with open(os.path.join(out_dir, rel), encoding="utf-8") as f:
            return json.load(f)

    a, flags = load("audit/audit.json"), load("audit/dup_flags.json")
    e, n = a["exact"], a["normalized"]
    assert (e["n_groups"], e["n_rows"], e["n_cross_split_groups"], e["n_conflicting_groups"]) == (29, 63, 7, 2)
    assert (n["n_groups"], n["n_rows"], n["n_cross_split_groups"], n["n_conflicting_groups"]) == (47, 107, 14, 3)
    assert a["uids"] == {"n_rows": 3080, "n_unique": 3080, "duplicates": []}
    assert a["hotels_in_banking"]["n"] == 9 and len(a["hotels_in_banking"]["uids"]) == 9
    assert flags["normalized"] == {"test_in_final_train": NORMALIZED_TEST_FLAGS,
                                   "val_in_selection_train": NORMALIZED_VAL_FLAGS}
    assert (len(flags["exact"]["test_in_final_train"]), len(flags["exact"]["val_in_selection_train"])) == (5, 2)
    conflicts = {g["text"]: g["label_sets"] for g in n["groups"] if len(g["label_sets"]) > 1}
    assert sorted(conflicts) == ["cari hesap", "evet dün ödedim", "evet mevduat hesabı"]
    assert conflicts["cari hesap"] == [{"labels": ["account", "current"], "uids": ["bank_0_78"]},
                                       {"labels": ["current"], "uids": ["bank_5_77", "bank_6_78", "bank_9_78"]}]
    assert {u for s in conflicts["evet mevduat hesabı"] for u in s["uids"]} == {"bank_2_76", "bank_14_75"}
    assert {u for s in conflicts["evet dün ödedim"] for u in s["uids"]} == {"bank_6_90", "bank_10_88"}
    with open(pilot.data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        assert m["audit"] == json.load(f)["audit"]
