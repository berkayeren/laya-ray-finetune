"""Shared fixtures: a tiny synthetic Multi3NLU++ snapshot (20 banking + 20 hotel folds) with the edge rows
the data pipeline must handle (no-intent rows with the key absent, cross-split duplicates, Turkish I/İ casing,
conflicting labels, a hotels text equal to a banking text, optional duplicate uids).
"""
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import pilot.data
# the synthetic_raw fixture: synthetic_raw(dup_uid=None) -> (snapshot root, {raw fold relpath: sha256})
SyntheticRaw = Callable[..., tuple[str, dict[str, str]]]


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_raw(root: str, dup_uid: str | None = None) -> tuple[str, dict[str, str]]:
    """Write turkish/{banking,hotels}/fold{0..19}.json under root; return (root, {relpath: sha256}) for all 40."""
    folds = {}
    for i in range(20):
        folds[("banking", i)] = [{"uid": "bank_%d_0" % i, "text": "Kartım kayboldu %d" % i,
                                  "intents": ["card", "lost_stolen"], "lang": "tr"}]
        folds[("hotels", i)] = [{"uid": "hotel_%d_0" % i, "text": "Oda ayırtmak istiyorum %d" % i,
                                 "intents": ["booking"], "lang": "tr"}]
    edges = [
        ("banking", 0, {"uid": "bank_0_1", "text": "Evet.", "intents": ["affirm"]}),
        ("banking", 16, {"uid": "bank_16_1", "text": "evet", "intents": ["affirm"]}),          # normalized cross-split dup
        ("banking", 5, {"uid": "bank_5_1", "text": "İptal etmek istiyorum", "intents": ["cancel_close_leave_freeze"]}),
        ("banking", 17, {"uid": "bank_17_1", "text": "iptal etmek istiyorum", "intents": ["cancel_close_leave_freeze"]}),
        ("banking", 2, {"uid": "bank_2_1", "text": "cari hesap", "intents": ["account", "current"]}),
        ("banking", 9, {"uid": "bank_9_1", "text": "Cari hesap.", "intents": ["current"]}),     # conflicting labels
        ("banking", 3, {"uid": "bank_3_1", "text": "Hmm tamam o zaman"}),                       # no-intent: key absent
        ("banking", 15, {"uid": "bank_15_1", "text": "Bir dakika lütfen"}),
        ("banking", 18, {"uid": "bank_18_1", "text": "Anladım sanırım"}),
        ("hotels", 4, {"uid": "hotel_4_1", "text": "Peki"}),
        ("hotels", 7, {"uid": "hotel_7_1", "text": "Evet.", "intents": ["affirm"]}),            # equals a banking text
    ]
    if dup_uid == "within":
        edges.append(("banking", 1, {"uid": "bank_1_0", "text": "Kartımı bloke edin", "intents": ["card"]}))
    elif dup_uid == "cross":
        edges.append(("hotels", 0, {"uid": "bank_0_0", "text": "Havlu istiyorum", "intents": ["request_info"]}))
    elif dup_uid is not None:
        raise ValueError("dup_uid must be None, 'within' or 'cross', got %r" % (dup_uid,))
    for domain, i, row in edges:
        folds[(domain, i)].append(dict(row, lang="tr"))
    raw_sha256 = {}
    for (domain, i), rows in sorted(folds.items()):
        rel = "turkish/%s/fold%d.json" % (domain, i)
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        raw_sha256[rel] = _sha256(path)
    return root, raw_sha256


def add_raw_row(root: str, raw_sha256: dict[str, str], domain: str, fold: int, row: dict[str, Any]) -> None:
    """Append `row` to a synthetic raw fold and update its pinned sha256 in place."""
    rel = "turkish/%s/fold%d.json" % (domain, fold)
    path = os.path.join(root, rel)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    rows.append(dict(row, lang="tr"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    raw_sha256[rel] = _sha256(path)


# Unicode line separators that str.splitlines() splits on but a text-mode file (and JSONL) does not
LINE_SEPARATOR_TEXTS = ("Kartım\u2028kayboldu", "Kartım\x85bloke", "Hesap\u2029kapat")


def prepared_with_line_separators(tmp_path: Path, synthetic_raw: SyntheticRaw) -> tuple[str, str]:
    """prepare() over the synthetic snapshot plus banking rows whose text holds U+2028 / U+0085 / U+2029."""
    root, raw_sha256 = synthetic_raw()
    for i, (fold, text) in enumerate(zip((4, 12, 17), LINE_SEPARATOR_TEXTS)):   # train, train, test folds
        add_raw_row(root, raw_sha256, "banking", fold, {"uid": "bank_sep_%d" % i, "text": text, "intents": ["card"]})
    out_dir = str(tmp_path / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    return out_dir, os.path.join(out_dir, "splits.json")


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> SyntheticRaw:
    """Callable bound to tmp_path / "raw": synthetic_raw(dup_uid=None) -> (root, raw_sha256)."""
    def make(dup_uid: str | None = None) -> tuple[str, dict[str, str]]:
        return make_raw(str(tmp_path / "raw"), dup_uid=dup_uid)
    return make


def prepared(tmp_path: Path, synthetic_raw: SyntheticRaw, name: str = "data") -> tuple[str, str]:
    """Run pilot.data.prepare on the synthetic snapshot into tmp_path / name; return (out_dir, manifest_path)."""
    root, raw_sha256 = synthetic_raw()
    out_dir = str(tmp_path / name)
    pilot.data.prepare(root, out_dir, raw_sha256)
    return out_dir, os.path.join(out_dir, "splits.json")
