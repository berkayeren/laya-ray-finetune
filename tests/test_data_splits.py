"""Split contract of pilot.data on the synthetic snapshot: the no-intent flag (D-10), fold mapping, leakage
disjointness, row schema, ordering, one row per raw message (D-12), determinism and raw-hash refusal (D-01).

Every test works in tmp_path; the repo's data/ is never touched.
"""
import json
import os

import pytest

import pilot.data
from conftest import prepared
from intents import INTENTS, QUESTIONS

pytestmark = pytest.mark.unit

SPLIT_NAMES = ("selection_train", "val", "final_train", "test")
NO_INTENT_UIDS = {"bank_3_1": "selection_train", "bank_15_1": "val", "bank_18_1": "test"}


def _split(out_dir, name):
    return os.path.join(out_dir, "splits", "%s.jsonl" % name)


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _rows(out_dir, name):
    with open(_split(out_dir, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _uids(out_dir, name):
    return [r["uid"] for r in _rows(out_dir, name)]


def _manifest(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _fold(uid):
    return int(uid.split("_")[1])


def _edit_fold(root, raw_sha256, rel, edit):
    """Rewrite one raw fold with edit(rows) and re-pin its sha256 so only the edit is under test."""
    path = os.path.join(root, rel)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    edit(rows)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    raw_sha256[rel] = pilot.data.sha256_file(path)


# --- no-intent flag (D-10, DATA-01) ---------------------------------------------------------------------------

def test_no_intent_rows_kept_by_default_as_all_negative(tmp_path, synthetic_raw):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    for uid, split in NO_INTENT_UIDS.items():
        by_uid = {r["uid"]: r for r in _rows(out_dir, split)}
        assert uid in by_uid
        exp = by_uid[uid]["expected"]
        assert list(exp) == list(INTENTS) and not any(exp.values())
    assert "bank_15_1" in _uids(out_dir, "final_train") and "bank_3_1" in _uids(out_dir, "final_train")
    assert _manifest(manifest_path)["options"]["include_no_intent"] is True


def test_no_intent_flag_off_drops_exactly_those_rows(tmp_path, synthetic_raw):
    on_dir, _ = prepared(tmp_path, synthetic_raw, name="on")
    root, raw_sha256 = synthetic_raw()
    off_dir = str(tmp_path / "off")
    m = pilot.data.prepare(root, off_dir, raw_sha256, include_no_intent=False)
    for name in SPLIT_NAMES:
        on, off = _uids(on_dir, name), _uids(off_dir, name)
        assert off == [u for u in on if u not in NO_INTENT_UIDS], name
        assert m["splits"][name]["n_no_intent"] == 0
        assert m["splits"][name]["n_messages"] == len(off)
    assert sum(len(_uids(on_dir, n)) - len(_uids(off_dir, n)) for n in SPLIT_NAMES) == 5   # final_train holds two
    assert m["options"]["include_no_intent"] is False
    assert _manifest(os.path.join(off_dir, "splits.json"))["options"]["include_no_intent"] is False


def test_no_intent_null_and_empty_list_behave_like_absent_key(tmp_path, synthetic_raw):
    root, raw_sha256 = synthetic_raw()

    def add(rows):
        rows.append({"uid": "bank_6_1", "text": "Tamamdır", "intents": None, "lang": "tr"})
        rows.append({"uid": "bank_6_2", "text": "Peki o zaman", "intents": [], "lang": "tr"})

    _edit_fold(root, raw_sha256, "turkish/banking/fold6.json", add)
    on_dir, off_dir = str(tmp_path / "on"), str(tmp_path / "off")
    m_on = pilot.data.prepare(root, on_dir, raw_sha256)
    by_uid = {r["uid"]: r for r in _rows(on_dir, "selection_train")}
    for uid in ("bank_6_1", "bank_6_2"):
        assert not any(by_uid[uid]["expected"].values())
    assert m_on["splits"]["selection_train"]["n_no_intent"] == 3
    pilot.data.prepare(root, off_dir, raw_sha256, include_no_intent=False)
    off = _uids(off_dir, "selection_train")
    assert "bank_6_1" not in off and "bank_6_2" not in off and "bank_3_1" not in off


@pytest.mark.parametrize("text", ["", 7, None], ids=["empty", "int", "null"])
def test_no_intent_edge_bad_text_hard_fails(tmp_path, synthetic_raw, text):
    root, raw_sha256 = synthetic_raw()
    _edit_fold(root, raw_sha256, "turkish/banking/fold4.json",
               lambda rows: rows.append({"uid": "bank_4_9", "text": text, "lang": "tr"}))
    with pytest.raises(SystemExit, match="bank_4_9: text must be a non-empty string"):
        pilot.data.prepare(root, str(tmp_path / "data"), raw_sha256)
    assert not os.path.exists(str(tmp_path / "data"))


def test_no_intent_edge_unknown_banking_tag_hard_fails(tmp_path, synthetic_raw):
    root, raw_sha256 = synthetic_raw()
    _edit_fold(root, raw_sha256, "turkish/banking/fold11.json",
               lambda rows: rows.append({"uid": "bank_11_9", "text": "Kripto almak istiyorum",
                                         "intents": ["crypto_wallet"], "lang": "tr"}))
    with pytest.raises(SystemExit, match="unknown tag 'crypto_wallet'"):
        pilot.data.prepare(root, str(tmp_path / "data"), raw_sha256, include_no_intent=False)
    assert not os.path.exists(str(tmp_path / "data"))
