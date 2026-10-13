"""ray_train.py helpers that run inside Ray workers, tested without starting Ray or loading a model."""
import hashlib
import json

import pytest

ray_train = pytest.importorskip("ray_train")                # imports ray, torch and laya (the locked env has them)

pytestmark = pytest.mark.unit


def _write(path, rows):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_read_verified_rows_parses_the_verified_bytes(tmp_path):
    rows = [{"uid": "a", "state": "Kartım kayboldu"}, {"uid": "b", "state": "Evet."}]
    digest = _write(tmp_path / "s.jsonl", rows)
    assert ray_train.read_verified_rows(str(tmp_path / "s.jsonl"), digest) == rows


def test_read_verified_rows_refuses_a_file_changed_after_driver_verification(tmp_path):
    digest = _write(tmp_path / "s.jsonl", [{"uid": "a", "state": "x"}])
    _write(tmp_path / "s.jsonl", [{"uid": "a", "state": "y"}])
    with pytest.raises(SystemExit, match="verified on the driver"):
        ray_train.read_verified_rows(str(tmp_path / "s.jsonl"), digest)


def test_require_single_node_refuses_a_multi_node_cluster():
    ray_train.require_single_node([{"Alive": True}, {"Alive": False}])          # one alive node: fine
    with pytest.raises(SystemExit, match="single-node"):
        ray_train.require_single_node([{"Alive": True}, {"Alive": True}])


def test_holdout_messages_is_message_level_deterministic_and_disjoint():
    rows = [{"uid": "m%d" % i, "state": "t%d" % i} for i in range(100)]
    train, calib = ray_train.holdout_messages(rows, 40, 0.1, 0)
    assert len(calib) == 10                                  # min(40 messages, 10% of 100)
    assert {r["uid"] for r in train}.isdisjoint(r["uid"] for r in calib)
    assert len(train) + len(calib) == len(rows)
    again = ray_train.holdout_messages([dict(r) for r in rows], 40, 0.1, 0)   # what another rank computes
    assert again == (train, calib)
    assert len(ray_train.holdout_messages(rows, 3, 0.1, 0)[1]) == 3          # --calib-max-messages caps it
