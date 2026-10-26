"""The data boundary of the exploratory Ray Tune search (tune.py), proven by content: a trial trains on selection_train
folds 0-11 (tuning-train) and scores folds 12-13 (tuning-dev), both split in memory from the verified selection_train
bytes, so no tuning-dev row is a training row and no trial reads val, test or HOTELS.

The partition is keyed on content, never on a file name: both sides are compared by uid and by (uid, message) pair,
each side against the (uid, message) pairs of the committed val, test and hotels_probe files
(pilot.data.committed_content), and the fold read from a uid is checked against the hash-verified raw Multi3NLU++
banking folds themselves. The synthetic test runs offline in `make test`; the tests marked data read the gitignored
data/splits/*.jsonl and the pinned raw dataset, so they need `make data` first (offline from the Hub cache after the
first fetch). A committed file that is absent fails those tests, naming `make data`: an intersection with an empty set
would prove nothing.

usage: pytest tests/test_tune_data.py -m "data or not data"
"""
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from conftest import SyntheticRaw, prepared
from pilot import pins
from pilot import tune as pt
from pilot.data import (REPO_ROOT, committed_content, fetch_raw, load_fold, parse_jsonl, verify_raw,
                        verify_split_file)

pytestmark = pytest.mark.unit

HELD_OUT = ("val", "test", "hotels_probe")


def _digest(rows: list[dict[str, Any]]) -> str:
    """The manifest's uids_sha256 formula: sha256 of the newline-joined uids, UTF-8."""
    return hashlib.sha256("\n".join(r["uid"] for r in rows).encode("utf-8")).hexdigest()


def _pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(r["uid"], r["state"]) for r in rows}


def _committed_split(name: str) -> str:
    """The path of the committed data/splits/<name>.jsonl; fails (never skips) when `make data` has not written it."""
    path = os.path.join(REPO_ROOT, "data", "splits", name + ".jsonl")
    assert os.path.isfile(path), "%s is absent; run `make data` first" % os.path.relpath(path, REPO_ROOT)
    return path


def _committed_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(manifest entry, parsed rows) of the committed selection_train, its bytes verified before they are parsed."""
    path = _committed_split(pt.SOURCE)
    with open(path, "rb") as f:
        raw = f.read()
    entry = verify_split_file(path, roles=("train",), data=raw)
    return entry, parse_jsonl(raw)


def _check_disjoint(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> None:
    """No uid and no (uid, message) pair on both sides."""
    shared_uids = {r["uid"] for r in train} & {r["uid"] for r in dev}
    assert shared_uids == set(), "uids on both sides of the tuning partition: %s" % sorted(shared_uids)[:5]
    assert _pairs(train) & _pairs(dev) == set(), "a (uid, message) pair is on both sides of the tuning partition"


def test_synthetic_partition_is_disjoint(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    """On the conftest snapshot, tuning-dev is exactly the fold 12 and 13 rows, every tuning-train row is in folds
    0-11, the sides share no uid or pair, their digests are the uids_sha256 formula, and train + dev in file order is
    the whole split."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    path = os.path.join(out_dir, "splits", pt.SOURCE + ".jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    verify_split_file(path, manifest, roles=("train",), data=raw)
    rows = parse_jsonl(raw)
    train, dev, block = pt.partition(rows)
    assert [r["uid"] for r in dev] == ["bank_12_0", "bank_13_0"]
    assert all(pt.fold_of(r["uid"]) in pt.TRAIN_FOLDS for r in train)
    _check_disjoint(train, dev)
    assert block == {"source": pt.SOURCE, "train_folds": list(pt.TRAIN_FOLDS), "dev_folds": list(pt.DEV_FOLDS),
                     "train_uids_sha256": _digest(train), "dev_uids_sha256": _digest(dev)}
    order = {r["uid"]: i for i, r in enumerate(rows)}
    assert sorted(train + dev, key=lambda r: order[r["uid"]]) == rows
    assert [order[r["uid"]] for r in train] == sorted(order[r["uid"]] for r in train)
    assert [order[r["uid"]] for r in dev] == sorted(order[r["uid"]] for r in dev)


@pytest.mark.data
def test_committed_partition_is_disjoint_and_held_out_free() -> None:
    """On the committed selection_train, tuning-dev is folds 12-13 and tuning-train folds 0-11, the sides cover every
    message, share no uid or (uid, message) pair, and neither holds a (uid, message) pair of the committed val, test or
    hotels_probe file. The duplicate texts across the sides are kept (the duplicate audit is report-only) and counted."""
    entry, rows = _committed_rows()
    train, dev, _ = pt.partition(rows)
    assert len(train) + len(dev) == entry["n_messages"] == len(rows)
    assert {pt.fold_of(r["uid"]) for r in dev} <= set(pt.DEV_FOLDS)
    assert {pt.fold_of(r["uid"]) for r in train} <= set(pt.TRAIN_FOLDS)
    assert {pt.fold_of(r["uid"]) for r in dev} == set(pt.DEV_FOLDS), "a tuning-dev fold holds no row"
    _check_disjoint(train, dev)
    shared = pt.shared_texts(train, dev)
    for name in HELD_OUT:
        _committed_split(name)
        _, pairs = committed_content(name)
        assert pairs, "the committed %s file holds no (uid, message) pair" % name
        for side, got in (("tuning-train", train), ("tuning-dev", dev)):
            hit = _pairs(got) & pairs
            assert hit == set(), ("%d %s pair(s) of the committed %s split (%d tuning-dev rows share a text with "
                                  "tuning-train)" % (len(hit), side, name, shared))


@pytest.mark.data
def test_prefix_folds_match_the_raw_folds() -> None:
    """For every selection_train fold, the uids of the hash-verified raw banking fold, in order, are the committed
    selection_train uids whose uid names that fold, in file order: the fold read from a uid is the dataset's own."""
    _, rows = _committed_rows()
    root = fetch_raw()
    verify_raw(root, pins.RAW_SHA256)
    by_fold: dict[int, list[str]] = {}
    for r in rows:
        by_fold.setdefault(pt.fold_of(r["uid"]), []).append(r["uid"])
    folds = list(pt.TRAIN_FOLDS + pt.DEV_FOLDS)
    assert sorted(by_fold) == folds
    for f in folds:
        raw = [r["uid"] for r in load_fold(root, "banking", f)]
        assert raw == by_fold[f], "fold %d: the raw uids differ from the committed selection_train's" % f


@pytest.mark.data
def test_committed_partition_digests_are_stable() -> None:
    """A second partition of the same verified bytes gives the same two digests, and neither is the manifest's
    selection_train uids_sha256 (a side is never the whole split)."""
    entry, rows = _committed_rows()
    _, _, first = pt.partition(rows)
    _, _, second = pt.partition(_committed_rows()[1])
    assert first == second
    assert first["train_uids_sha256"] != first["dev_uids_sha256"]
    assert entry["uids_sha256"] not in (first["train_uids_sha256"], first["dev_uids_sha256"])
    assert entry["uids_sha256"] == _digest(rows)
