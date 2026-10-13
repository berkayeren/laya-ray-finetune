"""Duplicate audit (DATA-04): D-11 Turkish normalization, exact/normalized groups, conflicting labels (D-13),
dup flags (D-12), report-only guarantee, determinism and hash verification of the audit outputs.
"""
import json
import os
import shutil

import pytest

import pilot.data
from pilot.audit import build_audit, normalize_tr, render_markdown
from conftest import prepared

pytestmark = pytest.mark.unit

AUDIT_RELS = ("audit/audit.json", "audit/audit.md", "audit/dup_flags.json")


def _raw_rows(root):
    banking = [r for i in range(20) for r in pilot.data.load_fold(root, "banking", i)]
    hotels = [r for i in range(20) for r in pilot.data.load_fold(root, "hotels", i)]
    return banking, hotels


def _load(out_dir, rel):
    with open(os.path.join(out_dir, rel), encoding="utf-8") as f:
        return json.load(f)


def _group(section, uids):
    return next((g for g in section["groups"] if g["uids"] == sorted(uids)), None)


def test_normalize_tr_turkish_casing_punctuation_whitespace():
    assert normalize_tr("İSTANBUL IŞIK") == "istanbul ışık"
    assert normalize_tr("İyi akşamlar") == "iyi akşamlar"
    assert normalize_tr("Evet.") == "evet"
    assert normalize_tr("  Kartım   kayıp!! ") == "kartım kayıp"
    assert normalize_tr("Hesabım… ne'de?") == "hesabım ne de"
    assert normalize_tr("$100 ve %5") == "$100 ve 5"                       # symbols (S*) stay, punctuation goes


def test_normalize_tr_applies_nfc_before_the_dotted_capital_mapping():
    decomposed = "İptal"                                             # I + combining dot above
    assert decomposed != "İptal" and normalize_tr(decomposed) == "iptal" == normalize_tr("İptal")


def test_normalize_tr_does_not_merge_acronym_case_variants():
    # documented D-11 behaviour: the Turkish mapping makes PIN dotless, so it differs from lowercase pin
    assert normalize_tr("PIN kodumu") == "pın kodumu"
    assert normalize_tr("PIN kodumu") != normalize_tr("pin kodumu")


def test_synthetic_groups_exact_vs_normalized(tmp_path, synthetic_raw):
    audit, _ = build_audit(*_raw_rows(synthetic_raw()[0]))
    norm = {g["text"]: g for g in audit["normalized"]["groups"]}
    assert norm["evet"]["uids"] == ["bank_0_1", "bank_16_1"]
    assert norm["evet"]["partitions"] == ["selection_train", "test"]
    assert norm["iptal etmek istiyorum"]["uids"] == ["bank_17_1", "bank_5_1"]
    for uids in (["bank_0_1", "bank_16_1"], ["bank_5_1", "bank_17_1"], ["bank_2_1", "bank_9_1"]):
        assert _group(audit["exact"], uids) is None
    assert audit["normalized"]["n_groups"] == 3 and audit["normalized"]["n_rows"] == 6
    assert audit["normalized"]["n_cross_split_groups"] == 2               # cari hesap is folds 2 and 9: one partition
    assert audit["uids"] == {"n_rows": 51, "n_unique": 51, "duplicates": []}


def test_conflicting_labels_are_listed_and_never_fail(tmp_path, synthetic_raw):
    out_dir, _ = prepared(tmp_path, synthetic_raw)                         # D-13: prepare does not raise
    audit = _load(out_dir, "audit/audit.json")
    g = next(g for g in audit["normalized"]["groups"] if g["text"] == "cari hesap")
    assert g["uids"] == ["bank_2_1", "bank_9_1"]
    assert g["label_sets"] == [{"labels": ["account", "current"], "uids": ["bank_2_1"]},
                               {"labels": ["current"], "uids": ["bank_9_1"]}]
    assert audit["normalized"]["n_conflicting_groups"] == 1
    with open(os.path.join(out_dir, "audit/audit.md"), encoding="utf-8") as f:
        md = f.read()
    assert "bank_2_1 [account, current] vs bank_9_1 [current]" in md
    assert "reported only" in md and "acronym" in md.lower()


def test_dup_flags_and_hotels_in_banking(tmp_path, synthetic_raw):
    out_dir, _ = prepared(tmp_path, synthetic_raw)
    flags = _load(out_dir, "audit/dup_flags.json")
    assert flags["normalized"]["test_in_final_train"] == ["bank_16_1", "bank_17_1"]
    assert flags["normalized"]["val_in_selection_train"] == []
    assert flags["exact"] == {"test_in_final_train": [], "val_in_selection_train": []}
    audit = _load(out_dir, "audit/audit.json")
    assert audit["hotels_in_banking"] == {"n": 1, "uids": ["hotel_7_1"]}


def test_val_flag_uses_selection_train_only():
    rows = [{"uid": "bank_3_0", "text": "Tamam", "gold": [], "fold": 3},
            {"uid": "bank_14_0", "text": "tamam.", "gold": [], "fold": 14},
            {"uid": "bank_15_0", "text": "Peki", "gold": [], "fold": 15},
            {"uid": "bank_16_0", "text": "peki", "gold": [], "fold": 16}]
    audit, flags = build_audit(rows, [])
    assert flags["normalized"] == {"test_in_final_train": ["bank_16_0"], "val_in_selection_train": ["bank_14_0"]}
    assert flags["exact"] == {"test_in_final_train": [], "val_in_selection_train": []}
    assert [g["partitions"] for g in audit["normalized"]["groups"]] == [["test", "val"], ["selection_train", "val"]]


def test_report_only_splits_keep_every_raw_row(tmp_path, synthetic_raw):
    # D-12: duplicates and conflicting rows stay in their published splits, in raw fold order
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    root = str(tmp_path / "raw")
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    for s in pilot.data.SPLITS:
        raw = [r["uid"] for fold in s["folds"] for r in pilot.data.load_fold(root, s["domain"], fold)]
        with open(os.path.join(out_dir, m["splits"][s["name"]]["path"]), encoding="utf-8") as f:
            got = [json.loads(l)["uid"] for l in f]
        assert got == raw and m["splits"][s["name"]]["n_messages"] == len(raw), s["name"]
    for uid, split in (("bank_16_1", "test"), ("bank_17_1", "test"), ("bank_0_1", "selection_train"),
                       ("bank_9_1", "final_train"), ("bank_2_1", "final_train")):
        with open(os.path.join(out_dir, "splits/%s.jsonl" % split), encoding="utf-8") as f:
            assert uid in {json.loads(l)["uid"] for l in f}


def test_audit_outputs_are_byte_identical_across_builds(tmp_path, synthetic_raw):
    a, _ = prepared(tmp_path, synthetic_raw, name="a")
    b, _ = prepared(tmp_path, synthetic_raw, name="b")
    for rel in AUDIT_RELS + ("splits.json",):
        with open(os.path.join(a, rel), "rb") as fa, open(os.path.join(b, rel), "rb") as fb:
            assert fa.read() == fb.read(), rel


def test_render_markdown_is_pure(tmp_path, synthetic_raw):
    audit, flags = build_audit(*_raw_rows(synthetic_raw()[0]))
    assert render_markdown(audit, flags) == render_markdown(json.loads(json.dumps(audit)), flags)


def test_manifest_registers_audit_files_and_verifier_accepts_them(tmp_path, synthetic_raw):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    assert {k: e["path"] for k, e in m["audit"].items()} == {
        "audit_json": "audit/audit.json", "audit_md": "audit/audit.md", "dup_flags_json": "audit/dup_flags.json"}
    for rel in AUDIT_RELS:
        assert pilot.data.verify_audit_file(os.path.join(out_dir, rel), manifest_path=manifest_path)["path"] == rel


def test_verify_audit_file_refuses_tampered_and_unregistered(tmp_path, synthetic_raw):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    path = os.path.join(out_dir, "audit", "dup_flags.json")
    copy = os.path.join(out_dir, "audit", "dup_flags_copy.json")
    shutil.copy(path, copy)
    with pytest.raises(SystemExit, match="not an audit file registered"):
        pilot.data.verify_audit_file(copy, manifest_path=manifest_path)
    with pytest.raises(SystemExit, match="not a split registered"):              # sections do not cross over
        pilot.data.verify_split_file(path, manifest_path=manifest_path)
    with open(path, "rb") as f:
        data = bytearray(f.read())
    data[-2] ^= 1                                                                 # one byte
    with open(path, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(SystemExit, match="sha256"):
        pilot.data.verify_audit_file(path, manifest_path=manifest_path)


def test_verify_audit_file_refuses_unsafe_manifest_path(tmp_path, synthetic_raw):
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    m["audit"]["audit_md"]["path"] = "../audit.md"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f)
    with pytest.raises(SystemExit, match="unsafe path"):
        pilot.data.verify_audit_file(os.path.join(out_dir, "audit", "audit.json"), manifest_path=manifest_path)
