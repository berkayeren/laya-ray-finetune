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


# --- fold mapping and leakage (DATA-02) -----------------------------------------------------------------------

def test_fold_mapping_is_exact(tmp_path, synthetic_raw):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    want = {"selection_train": set(range(14)), "val": {14, 15}, "final_train": set(range(16)),
            "test": {16, 17, 18, 19}}
    m = _manifest(manifest_path)
    for name, folds in want.items():
        uids = _uids(out_dir, name)
        assert all(u.startswith("bank_") for u in uids), name
        assert {_fold(u) for u in uids} == folds, name
        assert m["splits"][name]["folds"] == sorted(folds) and m["splits"][name]["domain"] == "banking"
    assert {n: m["splits"][n]["role"] for n in SPLIT_NAMES} == {
        "selection_train": "train", "val": "val", "final_train": "train", "test": "test"}


def test_final_train_is_selection_train_then_val(tmp_path, synthetic_raw):
    out_dir, _ = prepared(tmp_path, synthetic_raw)
    assert _uids(out_dir, "final_train") == _uids(out_dir, "selection_train") + _uids(out_dir, "val")
    assert _read(_split(out_dir, "final_train")) == _read(_split(out_dir, "selection_train")) + _read(
        _split(out_dir, "val"))


def test_test_and_val_uids_are_disjoint_from_their_training_splits(tmp_path, synthetic_raw):
    out_dir, _ = prepared(tmp_path, synthetic_raw)
    assert not set(_uids(out_dir, "test")) & set(_uids(out_dir, "final_train"))
    assert not set(_uids(out_dir, "val")) & set(_uids(out_dir, "selection_train"))
    for name in SPLIT_NAMES:
        uids = _uids(out_dir, name)
        assert len(uids) == len(set(uids)), name


def test_hotels_rows_never_enter_banking_splits(tmp_path, synthetic_raw):
    out_dir, _ = prepared(tmp_path, synthetic_raw)
    for name in SPLIT_NAMES:
        assert not [u for u in _uids(out_dir, name) if not u.startswith("bank_")], name
    # hotel_7_1 shares the text "Evet." with bank_0_1; only the banking row is present
    assert [r["uid"] for r in _rows(out_dir, "final_train") if r["state"] == "Evet."] == ["bank_0_1"]


# --- row schema, ordering, one row per message ----------------------------------------------------------------

def test_row_schema_and_intents_order(tmp_path, synthetic_raw):
    out_dir, _ = prepared(tmp_path, synthetic_raw)
    for name in SPLIT_NAMES:
        with open(_split(out_dir, name), encoding="utf-8") as f:
            lines = f.read().split("\n")
        assert lines[-1] == "" and all(lines[:-1]), name               # one JSON object per line, final newline
        for line in lines[:-1]:
            r = json.loads(line)
            assert list(r) == ["state", "uid", "questions", "expected"]
            assert isinstance(r["state"], str) and r["state"] and isinstance(r["uid"], str)
            assert r["questions"] == QUESTIONS and list(r["questions"]) == list(INTENTS)
            assert list(r["expected"]) == list(INTENTS)
            assert all(type(v) is bool for v in r["expected"].values())
    by_uid = {r["uid"]: r for r in _rows(out_dir, "selection_train")}
    assert [k for k, v in by_uid["bank_2_1"]["expected"].items() if v] == [
        k for k in INTENTS if k in ("account", "current")]


def test_rows_follow_fold_then_in_fold_order(tmp_path, synthetic_raw):
    root, raw_sha256 = synthetic_raw()
    # shuffle the in-fold order of one fold so "sorted by uid" and "original order" disagree
    _edit_fold(root, raw_sha256, "turkish/banking/fold2.json", lambda rows: rows.reverse())
    out_dir = str(tmp_path / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    folds = {}
    for i in range(20):
        with open(os.path.join(root, "turkish", "banking", "fold%d.json" % i), encoding="utf-8") as f:
            folds[i] = [r["uid"] for r in json.load(f)]
    assert folds[2] == ["bank_2_1", "bank_2_0"]
    for name, rng in (("selection_train", range(14)), ("val", range(14, 16)), ("final_train", range(16)),
                      ("test", range(16, 20))):
        assert _uids(out_dir, name) == [u for i in rng for u in folds[i]], name


def test_one_row_per_raw_message_nothing_merged(tmp_path, synthetic_raw):
    root, raw_sha256 = synthetic_raw()
    _edit_fold(root, raw_sha256, "turkish/banking/fold1.json",       # exact duplicate text inside one split
               lambda rows: rows.append({"uid": "bank_1_1", "text": "Kartım kayboldu 1",
                                         "intents": ["card", "lost_stolen"], "lang": "tr"}))
    out_dir = str(tmp_path / "data")
    m = pilot.data.prepare(root, out_dir, raw_sha256)
    raw_counts = {}
    for i in range(20):
        with open(os.path.join(root, "turkish", "banking", "fold%d.json" % i), encoding="utf-8") as f:
            raw_counts[i] = len(json.load(f))
    for name in SPLIT_NAMES:
        want = sum(raw_counts[i] for i in m["splits"][name]["folds"])
        assert m["splits"][name]["n_messages"] == len(_uids(out_dir, name)) == want, name
    states = {name: [r["state"] for r in _rows(out_dir, name)] for name in SPLIT_NAMES}
    assert states["selection_train"].count("Kartım kayboldu 1") == 2
    assert "Evet." in states["selection_train"] and "evet" in states["test"]          # D-11 near-dup, both kept
    assert "İptal etmek istiyorum" in states["selection_train"] and "iptal etmek istiyorum" in states["test"]
    assert "cari hesap" in states["selection_train"] and "Cari hesap." in states["selection_train"]


# --- determinism and raw-hash refusal (D-01) ------------------------------------------------------------------

@pytest.mark.parametrize("include_no_intent", [True, False], ids=["default", "no_intent_off"])
def test_two_runs_give_identical_bytes(tmp_path, synthetic_raw, include_no_intent):
    root, raw_sha256 = synthetic_raw()
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    pilot.data.prepare(root, a, raw_sha256, include_no_intent=include_no_intent)
    pilot.data.prepare(root, b, raw_sha256, include_no_intent=include_no_intent)
    assert _read(os.path.join(a, "splits.json")) == _read(os.path.join(b, "splits.json"))
    for name in SPLIT_NAMES:
        assert _read(_split(a, name)) == _read(_split(b, name)), name


def test_one_byte_raw_change_hard_fails_before_any_output(tmp_path, synthetic_raw):
    root, raw_sha256 = synthetic_raw()                      # raw_sha256 stays the OLD pin, as with --data-dir
    rel = "turkish/banking/fold12.json"
    path = os.path.join(root, rel)
    with open(path, "rb") as f:
        data = bytearray(f.read())
    i = data.index(b"Kart")
    data[i] = ord("k")
    with open(path, "wb") as f:
        f.write(bytes(data))
    out_dir = tmp_path / "data"
    out_dir.mkdir()
    with pytest.raises(SystemExit, match=rel):
        pilot.data.prepare(root, str(out_dir), raw_sha256)
    assert not os.path.exists(str(out_dir / "splits.json"))
    assert not [p for p in out_dir.rglob("*") if p.is_file()]


def test_flag_off_refuses_to_overwrite_a_default_manifest(tmp_path, synthetic_raw):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    before = {p: _read(p) for p in [manifest_path] + [_split(out_dir, n) for n in SPLIT_NAMES]}
    root, raw_sha256 = synthetic_raw()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pilot.data.prepare(root, out_dir, raw_sha256, include_no_intent=False)
    assert {p: _read(p) for p in before} == before
